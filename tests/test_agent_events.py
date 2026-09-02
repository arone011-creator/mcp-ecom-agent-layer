# tests/test_agent_events.py
#
# The event contract between this agent and the storefront's chat UI.
# These tests are half of a cross-repository agreement: the golden stream
# in contracts/assistant-events.v1.json is tested here in Python and again
# in the storefront in TypeScript, so a shape change fails on both sides
# rather than surfacing as a broken chat window.

import json
from pathlib import Path

import pytest

from agent.events import (
    EVENT_TYPES,
    SCHEMA_VERSION,
    approval_required,
    error,
    message,
    replay,
    tool_completed,
    tool_started,
)
from agent.loop import run_turn
from tests.test_agent_loop import (
    FakeMessage,
    FakeToolCall,
    failing_executor,
    recording_executor,
    scripted_model,
)


def test_every_event_carries_the_schema_version():
    # Rule 2 of the storefront plan: versioned from the first commit,
    # because Phase 3's interrupt payload consumes this too.
    events = [
        message(0, "hi"),
        tool_started(1, "call_1", "get_orders", {"limit": 3}),
        tool_completed(2, "call_1", "get_orders", result=[{"orderNumber": "ORD-1"}]),
        approval_required(3, "call_2", "cancel_order", {"order_id": "ord_9"}),
        error(4, "boom", retryable=False),
    ]

    assert all(event["v"] == SCHEMA_VERSION for event in events)
    assert {event["type"] for event in events} == EVENT_TYPES


def test_the_payload_is_nested_so_it_cannot_collide_with_the_envelope():
    event = tool_started(0, "call_1", "get_orders", {"limit": 3})

    assert set(event) == {"v", "seq", "type", "data"}
    assert event["data"]["arguments"] == {"limit": 3}


def test_tool_started_arguments_stay_structured():
    # The approval card and the tool chips are rendered from these. A
    # pre-rendered string here is how agent prose gets into a UI that
    # promised never to trust it.
    event = tool_started(0, "call_1", "add_to_cart", {"product_id": "p1", "quantity": 2})

    assert isinstance(event["data"]["arguments"], dict)
    assert event["data"]["arguments"]["quantity"] == 2


def test_a_failed_tool_completion_carries_the_error_and_no_result():
    event = tool_completed(
        0, "call_1", "add_to_cart", error="409: Only 17 available; cart would hold 57"
    )

    assert event["data"]["ok"] is False
    assert "result" not in event["data"]
    assert "Only 17 available" in event["data"]["error"]


def test_a_completion_cannot_be_both_a_result_and_an_error():
    with pytest.raises(ValueError):
        tool_completed(0, "call_1", "get_cart", result={"itemCount": 1}, error="nope")


def test_every_event_survives_a_round_trip_through_json():
    # These cross a wire. A payload that only serialises by accident is a
    # bug waiting for the first non-primitive tool result.
    event = tool_completed(0, "call_1", "get_orders", result=[{"total": 12.5}])

    assert json.loads(json.dumps(event)) == event


# --- replay --------------------------------------------------------------


def test_replay_reconstructs_the_conversation_in_order():
    # The MUST PROVE of this task, in its machine-checkable half: the
    # event stream is a complete record, not a decoration alongside one.
    events = [
        tool_started(0, "call_1", "get_orders", {"limit": 3}),
        tool_completed(1, "call_1", "get_orders", result=[{"orderNumber": "ORD-1"}]),
        message(2, "You ordered ORD-1."),
    ]

    conversation = replay(events)

    assert conversation["text"] == ["You ordered ORD-1."]
    assert conversation["tools"] == [
        {
            "call_id": "call_1",
            "tool": "get_orders",
            "arguments": {"limit": 3},
            "ok": True,
            "result": [{"orderNumber": "ORD-1"}],
        }
    ]


def test_replay_pairs_a_failure_with_its_start():
    events = [
        tool_started(0, "call_1", "add_to_cart", {"product_id": "p1", "quantity": 57}),
        tool_completed(1, "call_1", "add_to_cart", error="409: Only 17 available"),
    ]

    tool = replay(events)["tools"][0]

    assert tool["ok"] is False
    assert tool["arguments"]["quantity"] == 57
    assert "Only 17 available" in tool["error"]


def test_replay_ignores_an_event_type_it_does_not_know():
    # Forward compatibility in the direction that actually happens: a
    # newer agent deployed against an older UI must not crash it.
    events = [
        message(0, "hi"),
        {"v": 1, "seq": 1, "type": "thinking_started", "data": {}},
        message(2, "bye"),
    ]

    assert replay(events)["text"] == ["hi", "bye"]


def test_replay_reports_a_gap_rather_than_hiding_it():
    # A dropped event means the conversation on screen is not the
    # conversation that happened. Silence would be the worse failure.
    events = [message(0, "hi"), message(2, "bye")]

    assert replay(events)["gaps"] == [1]


def test_replay_rejects_a_stream_from_a_future_schema():
    with pytest.raises(ValueError):
        replay([{"v": 2, "seq": 0, "type": "message", "data": {"text": "hi"}}])


# --- emission from the turn loop -----------------------------------------


def one_tool_turn():
    """The stock 'what did I order' turn, scripted. Used by several tests."""
    return {
        "model_call": scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        "execute_tool": recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]}),
    }


async def test_a_turn_emits_the_events_its_conversation_is_made_of():
    state = await run_turn("what did I order recently?", **one_tool_turn())

    assert [event["type"] for event in state["events"]] == [
        "tool_started",
        "tool_completed",
        "message",
    ]


async def test_the_emitted_stream_replays_to_the_conversation_that_happened():
    # The MUST PROVE. What the UI reconstructs from events must be what
    # the turn actually did - not an approximation assembled beside it.
    state = await run_turn("what did I order recently?", **one_tool_turn())

    conversation = replay(state["events"])

    assert conversation["text"] == [state["answer"]]
    assert conversation["gaps"] == []
    assert conversation["tools"][0]["tool"] == "get_orders"
    assert conversation["tools"][0]["arguments"] == {"limit": 3}
    assert conversation["tools"][0]["ok"] is True


async def test_sequence_numbers_are_monotonic_across_several_model_turns():
    state = await run_turn(
        "compare two products",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "get_product", '{"product_id":"p1"}'),
                    FakeToolCall("call_2", "get_product", '{"product_id":"p2"}'),
                ]
            ),
            FakeMessage(content="compared"),
        ),
        execute_tool=recording_executor({}),
    )

    assert [event["seq"] for event in state["events"]] == list(
        range(len(state["events"]))
    )


async def test_a_tool_failure_becomes_a_completed_event_not_a_lost_one():
    state = await run_turn(
        "add 57 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "call_1", "add_to_cart", '{"product_id":"p1","quantity":57}'
                    )
                ]
            ),
            FakeMessage(content="Only 17 are available."),
        ),
        execute_tool=failing_executor("409: Only 17 available; cart would hold 57"),
    )

    completed = [e for e in state["events"] if e["type"] == "tool_completed"][0]
    assert completed["data"]["ok"] is False
    assert "Only 17 available" in completed["data"]["error"]


async def test_a_refused_repeat_is_reported_rather_than_left_hanging():
    # The repeat guard short-circuits before the executor. Without an
    # explicit completion the UI would show a chip that spins forever.
    same_call = '{"product_id":"p1","quantity":57}'
    state = await run_turn(
        "add 57 headphones",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "add_to_cart", same_call)]),
            FakeMessage(tool_calls=[FakeToolCall("call_2", "add_to_cart", same_call)]),
            FakeMessage(content="I'll stop asking for 57."),
        ),
        execute_tool=failing_executor("409: Only 17 available; cart would hold 57"),
    )

    conversation = replay(state["events"])

    assert len(conversation["tools"]) == 2
    # Both chips resolve. Neither is left started-but-never-finished.
    assert all("ok" in tool for tool in conversation["tools"])
    assert conversation["gaps"] == []


async def test_no_bearer_token_or_identity_ever_appears_in_the_stream():
    # These events leave this process for a browser. Anything in them is
    # published.
    state = await run_turn("what did I order recently?", **one_tool_turn())

    serialised = json.dumps(state["events"]).lower()
    assert "bearer" not in serialised
    assert "authorization" not in serialised


# --- the cross-repository anchor -----------------------------------------

CONTRACT = (
    Path(__file__).resolve().parent.parent / "contracts" / "assistant-events.v1.json"
)


def test_the_golden_stream_replays_to_the_conversation_it_documents():
    # The storefront vendors this same file and asserts its TypeScript
    # parser reaches the same conversation. If either side changes shape,
    # one of the two tests fails - which is the whole mechanism.
    fixture = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert fixture["version"] == SCHEMA_VERSION
    assert replay(fixture["events"]) == fixture["expected"]


def test_the_golden_stream_covers_every_event_type():
    # A fixture exercising four of five types would let the fifth drift
    # silently between the repositories.
    fixture = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert {event["type"] for event in fixture["events"]} == EVENT_TYPES
