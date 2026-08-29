# tests/test_server.py
#
# Transport matters here for a security reason, not a deployment one. A
# stdio server carries one ambient identity per process, which is simply
# wrong for a multi-user chat app: every user would share whichever token
# the process happened to start with. HTTP with per-request auth is what
# makes "the caller" a per-call fact rather than a process-wide one.

import pytest

import approvals
import server
from server import MissingCredential

TOOL_SURFACE = {
    "search_products",
    "get_product",
    "check_inventory",
    "get_orders",
    "get_order",
    "get_cart",
    "add_to_cart",
    "remove_from_cart",
    "cancel_order",
}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", "test-secret")
    approvals.reset_spent_nonces()


class TestPerRequestIdentity:
    def test_a_request_without_a_bearer_token_is_refused(self):
        with pytest.raises(MissingCredential):
            server.api_for_headers({})

    def test_a_non_bearer_authorization_header_is_refused(self):
        with pytest.raises(MissingCredential):
            server.api_for_headers({"authorization": "Basic abc"})

    def test_an_empty_bearer_value_is_refused(self):
        with pytest.raises(MissingCredential):
            server.api_for_headers({"authorization": "Bearer   "})

    def test_the_scheme_is_matched_case_insensitively(self):
        # Clients differ on capitalisation and the RFC says it is
        # case-insensitive. Refusing "bearer" would be our bug.
        assert server.api_for_headers({"authorization": "bearer tok"}).token == "tok"

    def test_the_bearer_token_becomes_the_clients_credential(self):
        assert server.api_for_headers({"authorization": "Bearer tok_abc"}).token == (
            "tok_abc"
        )

    def test_each_request_gets_its_own_client(self):
        first = server.api_for_headers({"authorization": "Bearer tok_a"})
        second = server.api_for_headers({"authorization": "Bearer tok_b"})

        # No ambient identity anywhere: two callers must never share a
        # client, and a client cached between requests is an ambient
        # identity by another name.
        assert first is not second
        assert (first.token, second.token) == ("tok_a", "tok_b")


class TestSessionScope:
    def test_the_session_id_comes_from_the_header(self):
        assert server.session_id_for_headers({"mcp-session-id": "s1"}) == "s1"

    def test_a_missing_session_id_is_refused_rather_than_defaulted(self):
        # A default would put every caller in one approval scope, which is
        # the failure the session binding exists to prevent.
        with pytest.raises(MissingCredential):
            server.session_id_for_headers({})

    def test_a_blank_session_id_is_refused(self):
        with pytest.raises(MissingCredential):
            server.session_id_for_headers({"mcp-session-id": "   "})


class TestToolSurface:
    def test_every_tool_in_the_surface_is_registered(self):
        assert TOOL_SURFACE <= set(server.registered_tool_names())

    def test_nothing_beyond_the_surface_is_exposed(self):
        # A tool nobody designed is a capability nobody risk-assessed.
        assert set(server.registered_tool_names()) == TOOL_SURFACE

    def test_the_approval_mint_is_not_a_tool(self):
        # If an agent can mint its own approval, the approval means
        # nothing. This is the single most important assertion in the file.
        names = set(server.registered_tool_names())

        assert "mint_approval" not in names
        assert "approve" not in names
        assert not any("approv" in name for name in names)

    def test_the_credential_exchange_is_not_a_tool(self):
        # POST /api/v1/auth/token is how the server obtains a token.
        # Exposing it would hand the agent the credential exchange itself.
        names = set(server.registered_tool_names())

        assert not any("token" in name for name in names)
        assert not any("whoami" in name for name in names)


class TestApprovalMinting:
    def test_minting_requires_a_credential(self):
        status, body = server.mint_approval_for({}, {"tool": "cancel_order"})

        assert status == 401
        assert "error" in body

    def test_minting_requires_a_session(self):
        status, body = server.mint_approval_for(
            {"authorization": "Bearer tok"}, {"tool": "cancel_order"}
        )

        assert status == 401

    def test_it_mints_a_token_the_matching_call_can_spend(self):
        headers = {"authorization": "Bearer tok", "mcp-session-id": "s1"}

        status, body = server.mint_approval_for(
            headers, {"tool": "cancel_order", "args": {"order_id": "o1"}}
        )

        assert status == 200
        approvals.validate(body["token"], "s1", "cancel_order", {"order_id": "o1"})

    def test_the_minted_token_is_bound_to_the_arguments_it_was_given(self):
        headers = {"authorization": "Bearer tok", "mcp-session-id": "s1"}

        _, body = server.mint_approval_for(
            headers, {"tool": "cancel_order", "args": {"order_id": "o1"}}
        )

        with pytest.raises(approvals.ApprovalError):
            approvals.validate(
                body["token"], "s1", "cancel_order", {"order_id": "o7"}
            )

    def test_it_refuses_to_mint_for_a_tool_that_is_not_high_risk(self):
        # Minting for a low-risk tool would be meaningless, and minting for
        # an unknown name would be a way to probe what exists.
        headers = {"authorization": "Bearer tok", "mcp-session-id": "s1"}

        status, _ = server.mint_approval_for(
            headers, {"tool": "get_orders", "args": {}}
        )

        assert status == 400

    def test_it_refuses_to_mint_for_an_unnamed_tool(self):
        headers = {"authorization": "Bearer tok", "mcp-session-id": "s1"}

        status, _ = server.mint_approval_for(headers, {})

        assert status == 400


class TestHeaderAccess:
    def test_the_session_header_is_requested_explicitly(self, monkeypatch):
        # FastMCP's get_http_headers() strips mcp-session-id by default, as
        # an "MCP-related header". That default silently breaks every
        # cancel_order with "a session id is required" while every unit
        # test still passes, because the pure functions were never wrong.
        # Found by running the server, not by the suite -- so it gets a
        # test of its own.
        seen = {}

        def fake(include_all: bool = False):
            seen["include_all"] = include_all
            return {}

        monkeypatch.setattr(server, "get_http_headers", fake)

        server.request_headers()

        assert seen["include_all"] is True
