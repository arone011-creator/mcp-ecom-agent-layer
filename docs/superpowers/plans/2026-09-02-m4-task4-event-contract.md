# M4 Task 4 - The Event Contract and Emission

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the five-event contract as one versioned, machine-checkable artefact, and
make the agent turn emit it.

**Architecture:** A new `agent/events.py` owns the contract: the version, the five types,
the constructors, and a pure `replay()` reducer. `agent/loop.py` accumulates events into
`TurnState["events"]` through the same `operator.add` reducer that already carries
`messages`. A golden event stream in `contracts/` is the single artefact both repositories
test against, so "the two must agree exactly" is enforced by a test rather than by two
prose descriptions that drift.

**Tech Stack:** Python 3.12, LangGraph 1.2.11, pytest. No new dependencies.

---

## Why a golden fixture rather than a shared library

The contract lives in two languages in two separate git repositories. "Agree it in one
place and import it" (PLAN_M4_AGENT.txt Task 4) cannot mean a shared module, so it means a
shared *artefact*: `contracts/assistant-events.v1.json`, a recorded stream covering all five
event types. This repository's test asserts the emitter produces that shape and that
`replay()` reconstructs the expected conversation from it. The storefront vendors the same
file in its own Task 1 and asserts its parser reconstructs the same conversation. One file,
two consumers, and a divergence fails a test instead of surfacing in a browser.

## The contract, v1

Every event is an envelope with a fixed set of keys and a nested payload. Nesting the
payload under `data` means a future payload field can never collide with an envelope field.

```json
{"v": 1, "seq": 0, "type": "tool_started", "data": {...}}
```

| Field  | Meaning                                                                  |
|--------|--------------------------------------------------------------------------|
| `v`    | Schema version. `1` from the first commit. Rule 2 of the storefront plan. |
| `seq`  | Monotonic from 0, no gaps. Makes replay deterministic and gaps detectable. |
| `type` | One of the five below.                                                    |
| `data` | Payload, shape determined by `type`.                                      |

**`message`** — assistant prose.
```json
{"text": "You ordered ORD-1042 last Tuesday."}
```

**`tool_started`** — `arguments` is a structured object, never a rendered string. The UI
that wants to show "checking stock for 3" composes that itself from these fields.
```json
{"call_id": "call_1", "tool": "check_inventory", "arguments": {"product_id": "p1"}}
```

**`tool_completed`** — paired to its `tool_started` by `call_id`. `result` carries the tool
result itself, because the storefront's Task 6 renders product and order cards from tool
results rather than from agent prose. On failure `ok` is false, `result` is absent, and
`error` carries the storefront's message verbatim.
```json
{"call_id": "call_1", "tool": "check_inventory", "ok": true, "result": {"available": 17}}
{"call_id": "call_2", "tool": "add_to_cart", "ok": false, "error": "409: Only 17 available"}
```

**`approval_required`** — names WHICH action, in structured arguments. Emission is Task 5's
work; the type, its shape, and its place in the golden fixture are frozen here so Task 5 has
nothing left to invent.
```json
{"call_id": "call_3", "tool": "cancel_order", "arguments": {"order_id": "ord_9"}}
```

**`error`** — the turn itself failed, as distinct from a tool failing.
```json
{"message": "The assistant could not reach the shop.", "retryable": true}
```

### Two deliberate departures from the plan text

1. **No token handle in `approval_required`.** Section 3 of PLAN_M4_STOREFRONT.txt lists
   "a token handle", but its own Task 5 has the approve route mint from the event's
   arguments after a human click. A handle minted by the agent would be the agent
   participating in its own approval, which the whole design exists to prevent. `call_id`
   is the correlation key, and it is the only one needed: the storefront looks the order up
   fresh by id anyway.

2. **No cache-invalidation hint in `tool_completed`.** Rule 3 requires a chat-driven cart
   change to invalidate the same query the cart page uses. Which cache key a tool touches is
   the storefront's knowledge, not the agent's; the event carries `tool` and `ok` and the
   storefront maps. An agent shipping cache keys would be this repository asserting facts
   about a UI it cannot see.

### What is NOT in scope

Streaming transport. These events accumulate in turn state; delivering them over the wire as
they happen is the bridge route's job (storefront Task 3) against an agent HTTP surface that
does not exist yet (agent Task 8). Task 4 is the shape and the emission, not the pipe.

---

## File Structure

- **Create** `agent/events.py` — the contract, the constructors, and `replay()`. Kept out of
  `loop.py` because the storefront reads this file as the canonical definition and a reader
  should not have to skip past graph wiring to find it.
- **Create** `contracts/assistant-events.v1.json` — the golden stream.
- **Create** `contracts/README.md` — the contract in prose, marked canonical.
- **Create** `tests/test_agent_events.py` — the contract's tests.
- **Modify** `agent/loop.py` — add `events` to `TurnState`, emit from both nodes.
- **Modify** `tests/test_agent_loop.py` — no changes expected; it must keep passing
  unchanged, which is the proof that emission did not alter the loop's behaviour.

---

### Task 4.1: The envelope and the five constructors

**Files:**
- Create: `agent/events.py`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_events.py
import json

import pytest

from agent.events import (
    SCHEMA_VERSION,
    EVENT_TYPES,
    approval_required,
    error,
    message,
    tool_completed,
    tool_started,
)


def test_every_event_carries_the_schema_version():
    # Rule 2 of the storefront plan: versioned from the first commit,
    # because Phase 3 consumes this too.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'agent.events'`

- [ ] **Step 3: Write the implementation**

```python
# agent/events.py
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
        # verbatim: the storefront wrote this sentence to be acted on.
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add agent/events.py tests/test_agent_events.py
git commit -m "feat: freeze the assistant event contract at v1"
```

---

### Task 4.2: Replay

**Files:**
- Modify: `agent/events.py`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_agent_events.py
from agent.events import replay


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: FAIL, `ImportError: cannot import name 'replay'`

- [ ] **Step 3: Write the implementation**

```python
# append to agent/events.py


def replay(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a conversation from its event stream.

    The reference reducer. The storefront's UI implements the same
    reduction in TypeScript, and contracts/assistant-events.v1.json is
    what proves the two agree.
    """
    text: list[str] = []
    tools: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    failures: list[dict[str, Any]] = []
    seen: list[int] = []

    for event in events:
        if event.get("v") != SCHEMA_VERSION:
            raise ValueError(
                f"Event schema v{event.get('v')} cannot be replayed by a "
                f"v{SCHEMA_VERSION} reader"
            )

        seen.append(event["seq"])
        type_ = event["type"]
        data = event.get("data", {})

        if type_ == "message":
            text.append(data["text"])

        elif type_ in ("tool_started", "approval_required"):
            call_id = data["call_id"]
            order.append(call_id)
            tools[call_id] = {
                "call_id": call_id,
                "tool": data["tool"],
                "arguments": data["arguments"],
                "awaiting_approval": type_ == "approval_required",
            }

        elif type_ == "tool_completed":
            # A completion without its start still records: half a pair is
            # a symptom worth seeing, not one worth swallowing.
            call_id = data["call_id"]
            if call_id not in tools:
                order.append(call_id)
                tools[call_id] = {"call_id": call_id, "tool": data["tool"]}
            tools[call_id].pop("awaiting_approval", None)
            tools[call_id]["ok"] = data["ok"]
            if data["ok"]:
                tools[call_id]["result"] = data.get("result")
            else:
                tools[call_id]["error"] = data["error"]

        elif type_ == "error":
            failures.append(data)

        # Any other type is ignored on purpose. A newer agent must not be
        # able to crash an older reader.

    expected = range(min(seen), max(seen) + 1) if seen else []

    return {
        "text": text,
        "tools": [tools[call_id] for call_id in order],
        "errors": failures,
        "gaps": [seq for seq in expected if seq not in set(seen)],
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add agent/events.py tests/test_agent_events.py
git commit -m "feat: add the reference replay reducer for the event contract"
```

---

### Task 4.3: The loop emits

**Files:**
- Modify: `agent/loop.py`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_agent_events.py
from agent.loop import run_turn
from tests.test_agent_loop import FakeMessage, FakeToolCall, recording_executor, scripted_model


async def test_a_turn_emits_the_events_its_conversation_is_made_of():
    state = await run_turn(
        "what did I order recently?",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]}),
    )

    assert [event["type"] for event in state["events"]] == [
        "tool_started",
        "tool_completed",
        "message",
    ]


async def test_the_emitted_stream_replays_to_the_conversation_that_happened():
    # The MUST PROVE. What the UI reconstructs from events must be what
    # the turn actually did - not an approximation assembled beside it.
    state = await run_turn(
        "what did I order recently?",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]}),
    )

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
    from tests.test_agent_loop import failing_executor

    state = await run_turn(
        "add 57 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":57}')
                ]
            ),
            FakeMessage(content="Only 17 are available."),
        ),
        execute_tool=failing_executor("409: Only 17 available; cart would hold 57"),
    )

    completed = [e for e in state["events"] if e["type"] == "tool_completed"][0]
    assert completed["data"]["ok"] is False
    assert "Only 17 available" in completed["data"]["error"]


async def test_no_bearer_token_or_identity_ever_appears_in_the_stream():
    # These events leave this process for a browser. Anything in them is
    # published.
    import json

    state = await run_turn(
        "what did I order recently?",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]}),
    )

    serialised = json.dumps(state["events"]).lower()
    assert "bearer" not in serialised
    assert "authorization" not in serialised
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: FAIL, `KeyError: 'events'`

- [ ] **Step 3: Write the implementation**

Add `events` to `TurnState` in `agent/loop.py`, beside the reducers already there:

```python
class TurnState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tools: list[dict]
    answer: str | None
    # Every (tool, arguments) pair that has already failed this turn.
    # Accumulated so the loop can refuse an identical retry.
    failed: Annotated[list[str], operator.add]
    # The stream the chat UI is built from. Accumulated by the same
    # reducer as messages, so a node returns only what it added.
    events: Annotated[list[dict], operator.add]
```

Import the constructors at the top of `agent/loop.py`:

```python
from agent.events import message as message_event
from agent.events import replay  # re-exported for callers; see docstring
from agent.events import tool_completed, tool_started
```

Sequence numbers must be unique across nodes that cannot see each other's return values, so
derive them from the length of the stream so far. Add this helper next to `_signature`:

```python
def _next_seq(state: TurnState) -> int:
    """The next sequence number, derived from what has already been emitted.

    LangGraph nodes return only their additions, so a counter held in a
    node would restart. The accumulated length is the one number both
    nodes can agree on without sharing state.
    """
    return len(state.get("events", []))
```

In `call_model`, emit a `message` event only when the model produced prose:

```python
    async def call_model(state: TurnState) -> dict:
        message = await model_call(state["messages"], state.get("tools", []))
        dumped = message.model_dump(exclude_none=True)
        # content is None on a tool turn; the API rejects a null content
        # field on the way back in.
        dumped.setdefault("role", "assistant")

        events = []
        if message.content:
            events.append(message_event(_next_seq(state), message.content))

        return {"messages": [dumped], "answer": message.content, "events": events}
```

In `execute_tools`, emit a start and a completion around each call. Note the local `seq`
counter: this node emits several events and each needs its own number.

```python
    async def execute_tools(state: TurnState) -> dict:
        results = []
        events = []
        newly_failed = []
        already_failed = set(state.get("failed", []))
        seq = _next_seq(state)

        for call in _tool_calls_of(state["messages"][-1]):
            name = call["function"]["name"]
            # Arguments arrive as a JSON string and the escaping varies.
            # Parse; never string-match.
            arguments = json.loads(call["function"]["arguments"] or "{}")

            # Not a recoverable failure: identity is never the model's to
            # assert, so this refuses the turn rather than inviting a retry.
            reject_forbidden_arguments(name, arguments)

            signature = _signature(name, arguments)
            call_id = call["id"]

            events.append(tool_started(seq, call_id, name, arguments))
            seq += 1

            if signature in already_failed:
                refusal = (
                    "This exact call was already tried this turn and failed. "
                    "Read the earlier error and change the arguments rather "
                    "than repeating it."
                )
                results.append(_tool_message(call_id, {"error": refusal}))
                events.append(tool_completed(seq, call_id, name, error=refusal))
                seq += 1
                continue

            try:
                result = await execute_tool(name, arguments)
            except ToolError as failure:
                # Passed through verbatim. The storefront writes these to
                # be acted on -- a 409 carries the number that IS
                # available -- and re-wording or parsing them here would
                # be a second implementation of someone else's rule.
                newly_failed.append(signature)
                results.append(_tool_message(call_id, {"error": str(failure)}))
                events.append(tool_completed(seq, call_id, name, error=str(failure)))
                seq += 1
                continue

            results.append(_tool_message(call_id, result))
            events.append(tool_completed(seq, call_id, name, result=result))
            seq += 1

        return {"messages": results, "failed": newly_failed, "events": events}
```

Seed the key in `run_turn` so the first `_next_seq` has a list to measure:

```python
    return await app.ainvoke(
        {
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools or [],
            "answer": None,
            "failed": [],
            "events": [],
        },
        config={"recursion_limit": max_steps},
    )
```

- [ ] **Step 4: Run to verify it passes, and that nothing else broke**

Run: `python -m pytest tests/ -v`
Expected: PASS. The existing `tests/test_agent_loop.py` must pass **unchanged** — that is
the evidence emission did not alter the loop's behaviour, only observed it.

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py tests/test_agent_events.py
git commit -m "feat: emit the assistant event stream from the turn loop"
```

---

### Task 4.4: The golden stream both repositories test against

**Files:**
- Create: `contracts/assistant-events.v1.json`
- Create: `contracts/README.md`
- Test: `tests/test_agent_events.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_agent_events.py
from pathlib import Path

CONTRACT = Path(__file__).resolve().parent.parent / "contracts" / "assistant-events.v1.json"


def test_the_golden_stream_replays_to_the_conversation_it_documents():
    # The cross-repository anchor. The storefront vendors this same file
    # and asserts its TypeScript parser reaches the same conversation. If
    # either side changes shape, one of the two tests fails.
    fixture = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert fixture["version"] == SCHEMA_VERSION

    conversation = replay(fixture["events"])

    assert conversation == fixture["expected"]


def test_the_golden_stream_covers_every_event_type():
    # A fixture that exercises four of five types would let the fifth
    # drift silently between the repositories.
    fixture = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert {event["type"] for event in fixture["events"]} == EVENT_TYPES
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_agent_events.py -v`
Expected: FAIL, `FileNotFoundError`

- [ ] **Step 3: Write the fixture**

```json
{
  "version": 1,
  "description": "The canonical assistant event stream. Both repositories test their reducer against this file; see contracts/README.md.",
  "events": [
    {"v": 1, "seq": 0, "type": "tool_started", "data": {"call_id": "call_1", "tool": "get_orders", "arguments": {"limit": 3}}},
    {"v": 1, "seq": 1, "type": "tool_completed", "data": {"call_id": "call_1", "tool": "get_orders", "ok": true, "result": [{"orderNumber": "ORD-1042", "status": "PENDING"}]}},
    {"v": 1, "seq": 2, "type": "message", "data": {"text": "Your most recent order is ORD-1042, still pending."}},
    {"v": 1, "seq": 3, "type": "tool_started", "data": {"call_id": "call_2", "tool": "add_to_cart", "arguments": {"product_id": "p1", "quantity": 57}}},
    {"v": 1, "seq": 4, "type": "tool_completed", "data": {"call_id": "call_2", "tool": "add_to_cart", "ok": false, "error": "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 57"}},
    {"v": 1, "seq": 5, "type": "approval_required", "data": {"call_id": "call_3", "tool": "cancel_order", "arguments": {"order_id": "ord_9"}}},
    {"v": 1, "seq": 6, "type": "error", "data": {"message": "The assistant could not reach the shop.", "retryable": true}}
  ],
  "expected": {
    "text": ["Your most recent order is ORD-1042, still pending."],
    "tools": [
      {"call_id": "call_1", "tool": "get_orders", "arguments": {"limit": 3}, "ok": true, "result": [{"orderNumber": "ORD-1042", "status": "PENDING"}]},
      {"call_id": "call_2", "tool": "add_to_cart", "arguments": {"product_id": "p1", "quantity": 57}, "ok": false, "error": "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 57"},
      {"call_id": "call_3", "tool": "cancel_order", "arguments": {"order_id": "ord_9"}, "awaiting_approval": true}
    ],
    "errors": [{"message": "The assistant could not reach the shop.", "retryable": true}],
    "gaps": []
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Write `contracts/README.md`**

Document, in prose: that this directory is canonical; that `agent/events.py` and the
storefront's `lib/assistant/events.ts` are both implementations of it; the envelope; the
five payloads; the two deliberate departures from the plan text and why; and the instruction
that changing the shape means changing the fixture, which fails tests in both repositories
on purpose.

- [ ] **Step 6: Commit**

```bash
git add contracts tests/test_agent_events.py
git commit -m "docs: add the golden event stream both repositories test against"
```

---

### Task 4.5: Record the outcome in the plan

**Files:**
- Modify: `docs/PLAN_M4_AGENT.txt`
- Modify: `../mcp-ecom-web-app/docs/PLAN_M4_STOREFRONT.txt`

- [ ] **Step 1: Mark agent Task 4 done**, recording what was proved, that
  `approval_required` is defined but not yet emitted (Task 5 owns that), and that streaming
  transport is out of scope.

- [ ] **Step 2: Amend storefront Task 1** to point at `contracts/assistant-events.v1.json`
  as the artefact to vendor and test against, rather than leaving "coordinate with the agent
  plan's Task 4" as an instruction with no object. Record the two departures there too, so
  the storefront half does not implement a token handle that will never arrive.

- [ ] **Step 3: Commit both repositories.**

---

## Self-Review

**Spec coverage.** Task 4's two requirements — emit the five events the storefront defines
at an agreed version, and prove a stream replays to the same conversation — are covered by
4.1/4.3 and 4.2/4.3 respectively. Storefront Rules 1 and 2 are enforced here (structured
arguments, versioned envelope); Rules 3 and 4 are storefront-side by construction and are
recorded as such.

**Placeholders.** None. Every step carries the code it needs; step 4.4.5 describes a prose
document whose full content is the contract already specified above it.

**Type consistency.** `tool_started` / `tool_completed` / `approval_required` / `message` /
`error` are the names in `agent/events.py`, in the fixture, and in the tests. `message` is
imported as `message_event` inside `loop.py` because that module already binds `message` as
a local in `call_model`; the alias is deliberate, not a drift.
