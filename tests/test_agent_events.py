# tests/test_agent_events.py
#
# The event contract between this agent and the storefront's chat UI.
# These tests are half of a cross-repository agreement: the golden stream
# in contracts/assistant-events.v1.json is tested here in Python and again
# in the storefront in TypeScript, so a shape change fails on both sides
# rather than surfacing as a broken chat window.

import json

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
