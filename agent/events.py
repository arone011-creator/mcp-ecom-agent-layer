"""The event contract between this agent and the storefront's chat UI.

CANONICAL. The storefront's lib/assistant/events.ts is a translation of
this file, and contracts/assistant-events.v1.json is the golden stream
both sides test against. Change the shape here and that fixture fails on
both sides, which is the point: two prose descriptions of one contract
drift, one shared artefact does not.

Five types, one envelope, versioned from the first commit because Phase
3's interrupt payload consumes it too.
"""

from typing import Any

SCHEMA_VERSION = 1

EVENT_TYPES = frozenset(
    {"message", "tool_started", "tool_completed", "approval_required", "error"}
)


def _envelope(seq: int, type_: str, data: dict[str, Any]) -> dict[str, Any]:
    # Payload nested rather than flattened: an envelope key and a future
    # payload key can then never collide.
    return {"v": SCHEMA_VERSION, "seq": seq, "type": type_, "data": data}


def message(seq: int, text: str) -> dict[str, Any]:
    """Assistant prose. The only event whose content the model authored."""
    return _envelope(seq, "message", {"text": text})


def tool_started(
    seq: int, call_id: str, tool: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """A tool call beginning. Arguments stay structured, never rendered."""
    return _envelope(
        seq, "tool_started", {"call_id": call_id, "tool": tool, "arguments": arguments}
    )


def tool_completed(
    seq: int,
    call_id: str,
    tool: str,
    *,
    result: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """A tool call ending, paired to its start by call_id."""
    if error is not None and result is not None:
        raise ValueError("A completion is either a result or an error, never both")

    data: dict[str, Any] = {"call_id": call_id, "tool": tool, "ok": error is None}
    if error is None:
        data["result"] = result
    else:
        # Verbatim, for the same reason the loop passes it through
        # verbatim: the storefront writes these sentences to be acted on.
        data["error"] = error

    return _envelope(seq, "tool_completed", data)


def approval_required(
    seq: int, call_id: str, tool: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """A high-risk call waiting on a human.

    Deliberately carries no token or token handle. The storefront mints
    the approval after a click, from these arguments; a handle minted
    here would be the agent taking part in its own approval.
    """
    return _envelope(
        seq,
        "approval_required",
        {"call_id": call_id, "tool": tool, "arguments": arguments},
    )


def error(seq: int, message: str, *, retryable: bool) -> dict[str, Any]:
    """The turn failed, as distinct from a tool failing."""
    return _envelope(seq, "error", {"message": message, "retryable": retryable})
