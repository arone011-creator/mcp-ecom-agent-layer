"""Approval tokens for high-risk tool calls.

Minted by non-LLM code -- the /approvals HTTP route in server.py, which is
deliberately not an MCP tool, so an agent cannot mint its own. In Phase 2
the chat UI calls it after a human presses a confirm button.

Bound to (session, tool, argument hash, nonce, expiry) and single-use.
Every one of those five is load-bearing:

  - session, so an approval cannot cross conversations;
  - tool, so approval for one capability is not approval for another;
  - argument hash, so approval to cancel order #3 cannot be spent on #7 --
    this is the binding that presence-checking alone misses, and it is the
    difference between a security boundary and a convention;
  - nonce and single use, so a captured token cannot be replayed;
  - expiry, so a token left in a transcript goes stale.

What this does NOT defend: anyone holding a bearer token can still call
POST /api/v1/orders/{id}/cancel directly. That is unchanged, and it is not
what this is for. The API's own defence is ownership plus status rules,
which stay where they are. This exists so a prompt-injected *agent* cannot
talk itself past a confirmation step, and it lives where that agent's
calls arrive.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time

import config


class ApprovalError(Exception):
    """A high-risk call that may not proceed."""


# Spent nonces, in process. Single-replica only -- the same limitation the
# API's in-memory rate limiter documents. Unlike that one this fails
# *unsafe* when forgotten: losing the set would let a token be replayed
# inside its window, which is why the window is five minutes rather than a
# day. A shared store is required before this service runs on more than
# one instance.
_spent: dict[str, float] = {}
_SWEEP_THRESHOLD = 1000


def reset_spent_nonces() -> None:
    """Test seam."""
    _spent.clear()


def _sweep(now: float) -> None:
    for nonce, expires in list(_spent.items()):
        if expires <= now:
            del _spent[nonce]


def _secret() -> str:
    secret = config.APPROVAL_SECRET
    if not secret:
        # Minting against an empty key would produce tokens that any other
        # misconfigured instance would happily validate. Better loudly
        # broken than quietly worthless -- the same call the token endpoint
        # makes about a missing NEXTAUTH_SECRET.
        raise RuntimeError("MCP_APPROVAL_SECRET is required")
    return secret


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _encode(payload: dict) -> str:
    return _b64(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def args_hash(args: dict) -> str:
    """Canonical, so a reordering of the same arguments is the same call.

    Hashed rather than embedded: the token travels through agent context,
    and the order it authorises should not be readable from it.
    """
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sign(body: str) -> str:
    return _b64(hmac.new(_secret().encode(), body.encode(), hashlib.sha256).digest())


def mint(
    session_id: str,
    tool_name: str,
    args: dict,
    ttl_seconds: int | None = None,
) -> str:
    """Issue an approval for exactly one call. Never call this from a tool."""
    ttl = config.APPROVAL_TTL_SECONDS if ttl_seconds is None else ttl_seconds

    payload = {
        "sid": session_id,
        "tool": tool_name,
        "args": args_hash(args),
        # What makes each approval one-shot: two approvals for the same
        # call are distinct tokens, and spending one leaves the other good.
        "nonce": secrets.token_urlsafe(16),
        "exp": time.time() + ttl,
    }

    body = _encode(payload)
    return f"{body}.{_sign(body)}"


def validate(token: str, session_id: str, tool_name: str, args: dict) -> None:
    """Raise ApprovalError unless this exact call was approved.

    Checked against the arguments of the call actually arriving, not the
    ones the token was minted for. Those are only equal if nothing was
    swapped in between, which is the entire point.

    The nonce is burnt last, so a failed check does not consume a
    legitimate approval and force the user to approve again.
    """
    if not token or token.count(".") != 1:
        raise ApprovalError("Approval token is malformed")

    body, signature = token.split(".")

    if not body or not signature:
        raise ApprovalError("Approval token is malformed")

    # compare_digest rather than ==, so the check does not leak the
    # signature one byte at a time through its own timing.
    if not hmac.compare_digest(signature, _sign(body)):
        raise ApprovalError("Approval token signature is invalid")

    try:
        payload = json.loads(_unb64(body))
    except Exception:
        raise ApprovalError("Approval token is malformed")

    now = time.time()

    if float(payload.get("exp", 0)) < now:
        raise ApprovalError("Approval token has expired")

    if payload.get("sid") != session_id:
        raise ApprovalError("Approval token belongs to another session")

    if payload.get("tool") != tool_name:
        raise ApprovalError("Approval token was issued for a different tool")

    if payload.get("args") != args_hash(args):
        raise ApprovalError("Approval token was issued for different arguments")

    nonce = payload.get("nonce", "")
    if nonce in _spent:
        raise ApprovalError("Approval token has already been used")

    if len(_spent) > _SWEEP_THRESHOLD:
        _sweep(now)

    _spent[nonce] = float(payload["exp"])
