# tests/test_agent_server.py
#
# The agent's HTTP surface. The library underneath is tested elsewhere;
# what is worth testing here is the plumbing that turns one turn into a
# stream and one click into a resumed turn -- and the two refusals that
# stop anyone else spending this project's OpenAI credits.

import asyncio
from contextlib import asynccontextmanager

import pytest

import config
from agent_server import TurnRegistry


async def test_a_turn_can_be_registered_and_resolved():
    registry = TurnRegistry()
    turn_id = registry.open("session-1")

    assert registry.session_id(turn_id) == "session-1"


async def test_a_decision_reaches_the_waiting_turn():
    registry = TurnRegistry()
    turn_id = registry.open("session-1")

    async def wait():
        return await registry.wait_for_decision(turn_id, timeout=5)

    task = asyncio.create_task(wait())
    await asyncio.sleep(0)
    registry.decide(turn_id, {"approved": True, "token": "tok"})

    assert await task == {"approved": True, "token": "tok"}


async def test_a_decision_for_an_unknown_turn_is_refused():
    registry = TurnRegistry()

    with pytest.raises(KeyError):
        registry.decide("nope", {"approved": True})


async def test_a_second_decision_for_one_turn_is_refused():
    # A double-click must not resume twice. The storefront guards this
    # too (its Task 5), and neither side may rely on the other.
    registry = TurnRegistry()
    turn_id = registry.open("session-1")
    registry.decide(turn_id, {"approved": True, "token": "tok"})

    with pytest.raises(ValueError):
        registry.decide(turn_id, {"approved": True, "token": "tok"})


async def test_a_turn_that_is_never_decided_times_out_as_a_refusal():
    registry = TurnRegistry()
    turn_id = registry.open("session-1")

    assert await registry.wait_for_decision(turn_id, timeout=0.05) == {
        "approved": False,
        "reason": "expired",
    }


async def test_closing_a_turn_forgets_it():
    # A registry that only grows is a leak with a slow fuse.
    registry = TurnRegistry()
    turn_id = registry.open("session-1")
    registry.close(turn_id)

    with pytest.raises(KeyError):
        registry.session_id(turn_id)


# --- events must arrive as they happen -----------------------------------


async def test_the_hook_receives_every_event_the_turn_produced():
    # NAMED FOR WHAT IT CHECKS. This used to be called "published as the
    # turn runs, not at the end" and checked no such thing -- it asserts
    # on contents, and contents look identical whether the events arrive
    # one at a time or in a single lump at the end. They were arriving in
    # a lump, and this test was green throughout.
    #
    # The timing claim is now made where it can fail:
    # test_agent_events.py::test_the_tool_chip_reaches_the_caller_before_
    # the_turn_is_over.
    from agent.loop import run_turn
    from tests.test_agent_loop import (
        FakeMessage,
        FakeToolCall,
        recording_executor,
        scripted_model,
    )

    published = []

    await run_turn(
        "what did I order",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]}),
        on_event=published.append,
    )

    assert [e["type"] for e in published] == [
        "tool_started",
        "tool_completed",
        "message",
    ]


async def test_a_turn_without_the_hook_still_works():
    # on_event is optional; the eval harness and the tests do not use it.
    from agent.loop import run_turn
    from tests.test_agent_loop import FakeMessage, recording_executor, scripted_model

    state = await run_turn(
        "hello",
        model_call=scripted_model(FakeMessage(content="Hi.")),
        execute_tool=recording_executor({}),
    )

    assert state["answer"] == "Hi."


async def test_the_stream_carries_prose_fragments_before_the_finished_answer(
    monkeypatch,
):
    # The wiring, end to end through this module: the model's fragments
    # become message_delta frames on the same SSE stream the tool chips
    # ride, and they arrive BEFORE the message that closes them. Order is
    # the assertion -- fragments that turned up after the finished answer
    # would be a slower way of showing nothing.
    import json as json_

    import agent_server
    from tests.test_agent_loop import FakeMessage

    async def fake_tools(token, only=None):
        return []

    class FakeSession:
        session_id = "sess-1"

        async def execute(self, name, arguments):
            return {}

    @asynccontextmanager
    async def fake_session(token, url=None):
        yield FakeSession()

    def fake_model_call(model=None, on_usage=None, on_delta=None):
        async def call(messages, tools):
            for fragment in ["Your ", "order ", "is ORD-1."]:
                on_delta(fragment)
                # Let the consumer drain, as a real network read would.
                await asyncio.sleep(0)
            return FakeMessage(content="Your order is ORD-1.")

        return call

    monkeypatch.setattr(agent_server, "list_openai_tools", fake_tools)
    monkeypatch.setattr(agent_server, "session_scoped_executor", fake_session)
    monkeypatch.setattr(agent_server, "openai_model_call", fake_model_call)

    frames = []
    async for frame in agent_server._stream_turn("what did I order", "tok"):
        name, _, payload = frame.decode().partition("\n")
        frames.append(
            (name.removeprefix("event: "), json_.loads(payload.removeprefix("data: ")))
        )

    kinds = [
        data.get("type", "control") if name == "assistant" else "control"
        for name, data in frames
    ]

    assert kinds == ["control", "message_delta", "message_delta", "message_delta", "message"]

    fragments = [d["data"]["text"] for n, d in frames if d.get("type") == "message_delta"]
    assert "".join(fragments) == "Your order is ORD-1."

    # And they are out of band, so they cannot consume a number the
    # record needs or be mistaken for a dropped event.
    assert all(d["seq"] == -1 for n, d in frames if d.get("type") == "message_delta")


# --- the routes ----------------------------------------------------------

from starlette.testclient import TestClient  # noqa: E402

from agent_server import app  # noqa: E402


def test_health_needs_no_credentials():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    # The SHA is what makes a readiness check specific to this container.
    assert response.json()["sha"]


def test_a_turn_without_the_service_key_is_refused_before_spending_anything(monkeypatch):
    monkeypatch.setattr(config, "AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn",
            json={"utterance": "hi"},
            headers={"authorization": "Bearer customer-token"},
        )

    assert response.status_code == 401


def test_a_turn_without_a_customer_token_is_refused(monkeypatch):
    monkeypatch.setattr(config, "AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn", json={"utterance": "hi"}, headers={"x-agent-key": "k"}
        )

    assert response.status_code == 401


def test_an_unconfigured_service_refuses_everything(monkeypatch):
    # Absent key means closed, never open. Better loudly broken than
    # quietly reachable by anyone who finds the URL.
    monkeypatch.setattr(config, "AGENT_SERVICE_KEY", "")

    with TestClient(app) as client:
        response = client.post(
            "/turn",
            json={"utterance": "hi"},
            headers={"x-agent-key": "", "authorization": "Bearer t"},
        )

    assert response.status_code == 401


def test_a_decision_for_an_unknown_turn_is_a_404(monkeypatch):
    monkeypatch.setattr(config, "AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn/nope/decision",
            json={"approved": False},
            headers={"x-agent-key": "k"},
        )

    assert response.status_code == 404


def test_a_decision_without_the_service_key_is_refused(monkeypatch):
    monkeypatch.setattr(config, "AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn/anything/decision", json={"approved": True}
        )

    assert response.status_code == 401
