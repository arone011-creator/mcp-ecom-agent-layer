# tests/test_agent_approval.py
#
# The approval pause. These are the milestone's most important tests:
# everything else in M4 is about the agent being useful, and this is the
# part about it being safe.

import asyncio

from agent.loop import session_scoped_executor


class FakeTransport:
    def __init__(self, session_id):
        self._session_id = session_id

    def get_session_id(self):
        return self._session_id


class FakeClient:
    """Stands in for fastmcp.Client, recording how often a session opened."""

    opened = 0

    def __init__(self, transport):
        self.transport = transport
        self.calls = []

    async def __aenter__(self):
        FakeClient.opened += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))

        class Result:
            content = []

        return Result()


async def test_one_session_serves_every_call_in_a_turn(monkeypatch):
    # An approval token is bound to the mcp-session-id. A session per call
    # means the token minted for the call that paused is rejected by the
    # call that resumes -- and every unit test still passes, because the
    # pure functions were never the problem.
    FakeClient.opened = 0
    client = FakeClient(FakeTransport("session-abc"))

    monkeypatch.setattr("agent.loop._client_for", lambda url, token: client)

    async with session_scoped_executor("tok", url="http://x/mcp") as session:
        await session.execute("get_cart", {})
        await session.execute("get_orders", {"limit": 3})

    assert FakeClient.opened == 1
    assert session.session_id == "session-abc"
    assert [name for name, _ in client.calls] == ["get_cart", "get_orders"]
