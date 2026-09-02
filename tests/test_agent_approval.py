# tests/test_agent_approval.py
#
# The approval pause. These are the milestone's most important tests:
# everything else in M4 is about the agent being useful, and this is the
# part about it being safe.

import asyncio
import json

from agent.events import replay
from agent.loop import run_turn, session_scoped_executor
from tests.test_agent_loop import (
    FakeMessage,
    FakeToolCall,
    recording_executor,
    scripted_model,
)


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


# --- the pause -----------------------------------------------------------


def cancel_turn(order_id="ord_9"):
    """A model that asks to cancel, then acknowledges. Used by several tests."""
    return scripted_model(
        FakeMessage(
            tool_calls=[
                FakeToolCall("call_1", "cancel_order", '{"order_id":"%s"}' % order_id)
            ]
        ),
        FakeMessage(content="Cancelled."),
    )


async def declining(request):
    return {"approved": False}


async def test_a_high_risk_call_stops_before_anything_is_sent():
    # The MUST PROVE, first half. Refusing sends nothing to the MCP server.
    executor = recording_executor({})
    asked = []

    async def decline(request):
        asked.append(request)
        return {"approved": False}

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=executor,
        approve=decline,
    )

    assert executor.calls == []
    assert asked[0]["tool"] == "cancel_order"
    assert asked[0]["arguments"] == {"order_id": "ord_9"}
    assert state["answer"] == "Cancelled."


async def test_the_pause_emits_an_approval_required_event_with_structured_arguments():
    # The approval card is rendered from these. Prose would let an injected
    # review write the words next to the confirm button.
    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=recording_executor({}),
        approve=declining,
    )

    required = [e for e in state["events"] if e["type"] == "approval_required"]
    assert len(required) == 1
    assert required[0]["data"]["tool"] == "cancel_order"
    assert required[0]["data"]["arguments"] == {"order_id": "ord_9"}
    # Frozen in Task 4: the agent never mints, so it never names a token.
    assert "token" not in json.dumps(required[0])


async def test_a_declined_call_resolves_rather_than_hanging():
    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=recording_executor({}),
        approve=declining,
    )

    conversation = replay(state["events"])
    tool = conversation["tools"][0]

    assert tool["ok"] is False
    assert "declin" in tool["error"].lower()
    assert conversation["gaps"] == []


async def test_low_risk_calls_in_the_same_batch_do_not_run_before_the_pause():
    # LangGraph re-runs a node on resume. Anything executed before the
    # interrupt would execute twice; nothing runs before it, so it cannot.
    executor = recording_executor({})
    seen = []

    async def decline(request):
        seen.append(list(executor.calls))
        return {"approved": False}

    await run_turn(
        "check then cancel",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "get_order", '{"order_id":"ord_9"}'),
                    FakeToolCall("call_2", "cancel_order", '{"order_id":"ord_9"}'),
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
        approve=decline,
    )

    # Nothing had run at the moment the human was asked.
    assert seen[0] == []
    # And the read still ran exactly once afterwards.
    assert [name for name, _ in executor.calls] == ["get_order"]


# --- resume --------------------------------------------------------------


async def test_an_approved_call_carries_the_token_the_human_caused():
    executor = recording_executor({"cancel_order": {"status": "CANCELLED"}})

    async def approve(request):
        return {"approved": True, "token": "tok-from-the-storefront"}

    await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=executor,
        approve=approve,
    )

    name, arguments = executor.calls[0]
    assert name == "cancel_order"
    assert arguments["approval_token"] == "tok-from-the-storefront"


async def test_the_resumed_call_uses_the_arguments_the_human_saw():
    # The MUST PROVE, second half, and the subtlest of them. The model
    # does not run between the pause and the resume, so the arguments
    # cannot have been re-authored in between -- this asserts that rather
    # than trusting it.
    executor = recording_executor({"cancel_order": {"status": "CANCELLED"}})
    shown = {}

    async def approve(request):
        shown.update(request["arguments"])
        return {"approved": True, "token": "tok"}

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn("ord_42"),
        execute_tool=executor,
        approve=approve,
    )

    _, sent = executor.calls[0]
    event = [e for e in state["events"] if e["type"] == "approval_required"][0]

    assert shown == {"order_id": "ord_42"}
    assert event["data"]["arguments"] == {"order_id": "ord_42"}
    # What was sent is what was shown, plus the token code injected.
    assert {k: v for k, v in sent.items() if k != "approval_token"} == shown


async def test_the_approval_token_is_never_something_the_model_supplied():
    # The model is not shown the field, so it cannot fill it. If it
    # invents one anyway, code discards it.
    executor = recording_executor({"cancel_order": {"status": "CANCELLED"}})

    async def approve(request):
        return {"approved": True, "token": "real-token"}

    state = await run_turn(
        "cancel it",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "cancel_order",
                        '{"order_id":"ord_9","approval_token":"MODEL-INVENTED"}',
                    )
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
        approve=approve,
    )

    assert executor.calls[0][1]["approval_token"] == "real-token"
    # And the invented value never reached the human either.
    event = [e for e in state["events"] if e["type"] == "approval_required"][0]
    assert "MODEL-INVENTED" not in json.dumps(event)


async def test_the_agent_never_mints_its_own_approval():
    # An exit criterion, asserted structurally rather than behaviourally:
    # the agent package must not reference the minting route or module at
    # all. A behavioural test only proves it did not mint this time.
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent / "agent"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))

    assert "/approvals" not in source
    assert "import approvals" not in source
    assert "from approvals" not in source
