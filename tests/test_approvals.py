# tests/test_approvals.py
#
# The attack this defends against: an agent reads a malicious product
# review, is talked into cancelling a different order, and reuses an
# approval the human granted for something else.
#
# Presence-checking a token does not stop that -- the agent has a valid
# token either way. Binding it to the arguments does, which is why most of
# this file is about what a token is NOT good for.

import time

import pytest

import approvals
from approvals import ApprovalError

SECRET = "test-secret"
ARGS = {"order_id": "o1"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", SECRET)
    approvals.reset_spent_nonces()


def test_a_freshly_minted_token_validates():
    token = approvals.mint("sess1", "cancel_order", ARGS)

    approvals.validate(token, "sess1", "cancel_order", ARGS)


def test_a_token_is_single_use():
    token = approvals.mint("sess1", "cancel_order", ARGS)
    approvals.validate(token, "sess1", "cancel_order", ARGS)

    # A captured token must not be replayable, even by the session it was
    # issued to.
    with pytest.raises(ApprovalError, match="already been used"):
        approvals.validate(token, "sess1", "cancel_order", ARGS)


def test_a_token_for_one_order_cannot_be_spent_on_another():
    # The whole reason the arguments are hashed into the token. Without
    # this, approval to cancel #3 is approval to cancel anything.
    token = approvals.mint("sess1", "cancel_order", {"order_id": "o3"})

    with pytest.raises(ApprovalError, match="different arguments"):
        approvals.validate(token, "sess1", "cancel_order", {"order_id": "o7"})


def test_a_token_for_one_tool_cannot_be_spent_on_another():
    token = approvals.mint("sess1", "add_to_cart", ARGS)

    with pytest.raises(ApprovalError, match="different tool"):
        approvals.validate(token, "sess1", "cancel_order", ARGS)


def test_a_token_from_another_session_is_refused():
    token = approvals.mint("sess1", "cancel_order", ARGS)

    with pytest.raises(ApprovalError, match="another session"):
        approvals.validate(token, "sess2", "cancel_order", ARGS)


def test_an_expired_token_is_refused():
    token = approvals.mint("sess1", "cancel_order", ARGS, ttl_seconds=-1)

    with pytest.raises(ApprovalError, match="expired"):
        approvals.validate(token, "sess1", "cancel_order", ARGS)


def test_a_tampered_payload_is_refused():
    # Swapping the body while keeping the signature is the obvious forgery.
    token = approvals.mint("sess1", "cancel_order", ARGS)
    _, signature = token.split(".")
    forged_body = approvals._encode(
        {
            "sid": "sess1",
            "tool": "cancel_order",
            "args": approvals.args_hash({"order_id": "o7"}),
            "nonce": "n",
            "exp": time.time() + 60,
        }
    )

    with pytest.raises(ApprovalError, match="signature"):
        approvals.validate(f"{forged_body}.{signature}", "sess1", "cancel_order", ARGS)


def test_a_token_signed_with_another_key_is_refused(monkeypatch):
    token = approvals.mint("sess1", "cancel_order", ARGS)
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", "different-secret")

    with pytest.raises(ApprovalError, match="signature"):
        approvals.validate(token, "sess1", "cancel_order", ARGS)


@pytest.mark.parametrize("bad", ["", "no-dot", "a.b.c", "..", "!!!.???", "."])
def test_a_malformed_token_is_refused(bad):
    with pytest.raises(ApprovalError):
        approvals.validate(bad, "sess1", "cancel_order", ARGS)


def test_argument_order_does_not_change_the_binding():
    # Canonical hashing, or a reordering of the same call would look like a
    # different one and a legitimate approval would be refused.
    token = approvals.mint("sess1", "cancel_order", {"a": 1, "b": 2})

    approvals.validate(token, "sess1", "cancel_order", {"b": 2, "a": 1})


def test_a_failed_validation_does_not_burn_the_nonce():
    # Otherwise a wrong guess would consume a legitimate approval and the
    # user would have to approve again for no reason.
    token = approvals.mint("sess1", "cancel_order", {"order_id": "o1"})

    with pytest.raises(ApprovalError):
        approvals.validate(token, "sess1", "cancel_order", {"order_id": "o7"})

    approvals.validate(token, "sess1", "cancel_order", {"order_id": "o1"})


def test_two_tokens_for_the_same_call_are_distinct():
    # The nonce, not the payload, is what makes each approval one-shot.
    first = approvals.mint("sess1", "cancel_order", ARGS)
    second = approvals.mint("sess1", "cancel_order", ARGS)

    assert first != second

    approvals.validate(first, "sess1", "cancel_order", ARGS)
    approvals.validate(second, "sess1", "cancel_order", ARGS)


def test_spending_one_token_does_not_invalidate_another():
    first = approvals.mint("sess1", "cancel_order", ARGS)
    second = approvals.mint("sess1", "cancel_order", ARGS)

    approvals.validate(first, "sess1", "cancel_order", ARGS)
    approvals.validate(second, "sess1", "cancel_order", ARGS)


def test_the_token_does_not_leak_what_it_authorises():
    # It travels through agent context. The order id should not be
    # readable from it by anything that picks it up.
    token = approvals.mint("sess1", "cancel_order", {"order_id": "o1"})

    assert "o1" not in token


def test_an_empty_secret_cannot_be_used_to_mint(monkeypatch):
    # Failing loudly beats issuing tokens that a misconfigured deploy would
    # happily validate against the same empty key.
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", "")

    with pytest.raises(RuntimeError, match="MCP_APPROVAL_SECRET"):
        approvals.mint("sess1", "cancel_order", ARGS)


def test_an_empty_secret_cannot_be_used_to_validate(monkeypatch):
    token = approvals.mint("sess1", "cancel_order", ARGS)
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", "")

    with pytest.raises(RuntimeError, match="MCP_APPROVAL_SECRET"):
        approvals.validate(token, "sess1", "cancel_order", ARGS)
