# M4 Task 5 - The Approval Pause

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A high-risk tool stops the turn, surfaces a structured `approval_required`, and
proceeds only with a token a human caused to be minted — or does not proceed at all.

**Architecture:** The turn holds ONE MCP session for its lifetime, because an approval token
is bound to the session id and today every tool call opens a new one. The graph gains a
checkpointer and interrupts inside `execute_tools` **before anything executes**, so the
node's re-run on resume is harmless. `answer()` drives the resume itself through an injected
`approve` callback with a deadline, which keeps the held session inside one `async with` and
makes the timeout an `asyncio.wait_for` rather than a new mechanism.

**Tech Stack:** Python 3.12, LangGraph 1.2.11 (`interrupt`, `Command`, `InMemorySaver`),
fastmcp 2.14.7, pytest. No new dependencies.

---

## Two findings that shape this task

**1. The session must span the pause. Verified, not assumed.**

`approvals.validate` binds a token to `(session, tool, args_hash, nonce, expiry)`, and the
session is the transport-assigned `mcp-session-id`. Probed against production:

```
call 0 session 9000d44202b94a6297070313f294c089
call 1 session 79487bb01fb040e996254297251c6416
same client: True 709b60927d074b7d83cef985a81d021c
```

Today's `mcp_tool_executor` opens a `Client` per call, so it lands in the left-hand column: a
token minted for the call that paused would be rejected by the call that resumes, with
"Approval token belongs to another session". Every unit test would still pass, because the
pure functions are not the problem — the same shape of bug as the `include_all` header issue
already documented in `server.py`, and the same shape as the `cancel_order` session bug found
in Phase 1 verification. This is the third time on this project that a session identity has
been the thing that broke; it earns a test of its own.

So: one session, opened before the pause, held across it, used by the resumed call.

**2. That makes the timeout a resource question, not a tidiness one.**

The MUST PROVE already asks for a clean timeout. Holding a live MCP session across the wait
raises the stakes: a pause nobody answers now leaks a server-side session, not just a Python
task. The deadline is not optional politeness.

## Why the interrupt goes at the TOP of the node

LangGraph re-executes a node from its beginning when a thread resumes; `interrupt()` returns
the resume value on the second pass instead of raising. Any side effect performed *before*
the interrupt therefore happens twice.

`execute_tools` iterates a batch of tool calls. If the model asks for `get_order` and then
`cancel_order` in one batch and the interrupt fired mid-loop, `get_order` would run again on
resume. Harmless for a read; not a property to rely on. So the node collects approvals for
every high-risk call in the batch **before executing anything**, and only then runs the loop.
Re-running the collection is free.

## Why `answer()` drives the resume rather than returning a paused handle

Returning a paused handle and resuming through a second entry point is the natural
LangGraph shape, but it cannot hold an MCP client open across two separate calls. Instead
`answer()` takes an `approve` callback: it invokes, sees `__interrupt__`, awaits the callback
under a deadline, and invokes again with `Command(resume=...)` on the same thread — all
inside the one `async with` that owns the session.

The bridge route (storefront Task 3) supplies a callback that resolves when the browser POSTs
to `/api/assistant/approve`, which is exactly the connection it is already holding open to
stream events. The interrupt and the checkpointer are still real, so Phase 3 inherits the
checkpointed shape rather than a bespoke one.

## What the storefront needs, and why the frozen contract does not change

The storefront's approve route mints by calling the MCP server's `POST /approvals`, and must
mint against **the agent's** session id. That id is a server-side fact passed between the
bridge route and the agent — it is not added to the `approval_required` event, which reaches
a browser. The contract frozen in Task 4 stands unchanged; `session_id` rides the agent's own
interface (`PausedTurn.session_id`), one layer below the event stream.

## File Structure

- **Modify** `agent/tools.py` — `HIGH_RISK_TOOLS`, and `cancel_order` joins `AGENT_TOOLS`.
- **Modify** `agent/loop.py` — the held session, the checkpointer, the interrupt, the resume,
  the deadline, and token injection at call time.
- **Create** `tests/test_agent_approval.py` — the pause's tests. Kept separate from
  `test_agent_loop.py` because these are the milestone's most important assertions and should
  be findable as a group.
- **Modify** `docs/PLAN_M4_AGENT.txt`, `contracts/README.md`.

---

### Task 5.1: One session for the turn

**Files:**
- Modify: `agent/loop.py`
- Test: `tests/test_agent_approval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_approval.py
#
# The approval pause. These are the milestone's most important tests:
# everything else in M4 is about the agent being useful, and this is the
# part about it being safe.

import asyncio

import pytest

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -v`
Expected: FAIL, `ImportError: cannot import name 'session_scoped_executor'`

- [ ] **Step 3: Write the implementation**

In `agent/loop.py`, replace `mcp_tool_executor` with a session-scoped equivalent. Keep the
old name as a thin wrapper so nothing else breaks, and mark it as the read-only path.

```python
class McpSession:
    """One MCP session, and the executor that rides it.

    The session id is exposed because an approval token is minted against
    it: the storefront's approve route needs the id of the session the
    resumed call will actually use. A session per call would make that id
    a different one every time -- verified against production, and the
    third session-identity bug this project has had.
    """

    def __init__(self, client):
        self._client = client

    @property
    def session_id(self) -> str | None:
        return self._client.transport.get_session_id()

    async def execute(self, name: str, arguments: dict) -> Any:
        result = await self._client.call_tool(name, arguments)
        return json.loads(result.content[0].text) if result.content else None


def _client_for(url: str, token: str):
    """Seam. Tests replace this rather than the transport machinery."""
    from fastmcp import Client

    return Client(build_transport(url, token))


@asynccontextmanager
async def session_scoped_executor(token: str, url: str | None = None):
    """Hold one MCP session for the life of a turn."""
    client = _client_for(url or config.MCP_SERVER_URL, token)

    async with client:
        yield McpSession(client)
```

Add the imports at the top of the file:

```python
from contextlib import asynccontextmanager
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py tests/test_agent_approval.py
git commit -m "feat: hold one MCP session for the life of a turn"
```

---

### Task 5.2: The pause

**Files:**
- Modify: `agent/tools.py`, `agent/loop.py`
- Test: `tests/test_agent_approval.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_agent_approval.py
from agent.events import replay
from agent.loop import run_turn
from tests.test_agent_loop import FakeMessage, FakeToolCall, recording_executor, scripted_model


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
    async def decline(request):
        return {"approved": False}

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=recording_executor({}),
        approve=decline,
    )

    required = [e for e in state["events"] if e["type"] == "approval_required"]
    assert len(required) == 1
    assert required[0]["data"]["tool"] == "cancel_order"
    assert required[0]["data"]["arguments"] == {"order_id": "ord_9"}
    # Frozen in Task 4: the agent never mints, so it never names a token.
    assert "token" not in json.dumps(required[0])


async def test_a_declined_call_resolves_rather_than_hanging():
    async def decline(request):
        return {"approved": False}

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=recording_executor({}),
        approve=decline,
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -v`
Expected: FAIL, `TypeError: run_turn() got an unexpected keyword argument 'approve'`

- [ ] **Step 3: Write the implementation**

In `agent/tools.py`, name the tier:

```python
# High risk: irreversible from the customer's point of view, and gated by
# an approval token the MCP server checks. The agent is offered this tool
# only now that the pause exists to guard it.
HIGH_RISK_TOOLS = frozenset({"cancel_order"})
```

In `agent/loop.py`, add the approval collection at the top of `execute_tools`, before the
execution loop. The whole node becomes:

```python
    async def execute_tools(state: TurnState) -> dict:
        calls = _tool_calls_of(state["messages"][-1])

        # Parsed up front: the approval scan and the execution loop must
        # read the same arguments, and parsing twice invites them to drift.
        parsed = []
        for call in calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"] or "{}")
            # Not a recoverable failure: identity is never the model's to
            # assert, so this refuses the turn rather than inviting a retry.
            reject_forbidden_arguments(name, arguments)
            parsed.append((call["id"], name, arguments))

        # EVERY interrupt happens before ANY execution. LangGraph re-runs
        # this node from the top on resume, so a side effect before the
        # interrupt would happen twice.
        seq = _next_seq(state)
        approval_events = []
        decisions = {}

        for call_id, name, arguments in parsed:
            if name not in HIGH_RISK_TOOLS:
                continue

            approval_events.append(
                approval_required_event(seq, call_id, name, arguments)
            )
            seq += 1

            # Returns the resume value on the second pass rather than
            # pausing again.
            decisions[call_id] = interrupt(
                {"call_id": call_id, "tool": name, "arguments": arguments}
            )

        results = []
        events = list(approval_events)
        newly_failed = []
        already_failed = set(state.get("failed", []))

        for call_id, name, arguments in parsed:
            signature = _signature(name, arguments)

            decision = decisions.get(call_id)
            if decision is not None and not decision.get("approved"):
                declined = "The customer declined this action. Nothing was sent."
                results.append(_tool_message(call_id, {"error": declined}))
                events.append(tool_completed(seq, call_id, name, error=declined))
                seq += 1
                continue

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

            # The token is injected here, by code, from the decision the
            # human caused. It is never in the schema the model saw, so it
            # is never a value the model can have produced.
            sent = dict(arguments)
            if decision is not None:
                sent["approval_token"] = decision.get("token")

            try:
                result = await execute_tool(name, sent)
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

Note the `tool_started` for a declined call is deliberately absent — it never started. Its
`tool_completed` still fires, so the UI resolves the card rather than leaving it open.

Imports at the top of `agent/loop.py`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from agent.events import approval_required as approval_required_event
from agent.tools import HIGH_RISK_TOOLS
```

`build_graph` must compile with a checkpointer, or `interrupt` has nowhere to persist:

```python
def build_graph(
    model_call: ModelCall | None = None,
    execute_tool: ToolExecutor | None = None,
    checkpointer=None,
):
    """The turn as a graph.

    A checkpointer is required for the approval pause: interrupt() writes
    the paused state through it. InMemorySaver is honest about the current
    deployment -- one replica, one process - and is the same limitation
    approvals.py documents for its spent-nonce set.
    """
```

...and at the end:

```python
    return graph.compile(checkpointer=checkpointer)
```

`run_turn` gains the approval drive:

```python
async def run_turn(
    utterance: str,
    *,
    model_call: ModelCall,
    execute_tool: ToolExecutor,
    tools: list[dict] | None = None,
    max_steps: int = 25,
    approve: ApprovalCallback | None = None,
    approval_timeout_seconds: float = 300.0,
) -> TurnState:
    """One turn, start to finish, pausing for approval where required."""
    app = build_graph(model_call, execute_tool, checkpointer=InMemorySaver())

    config_ = {
        # A confused agent stops rather than looping. The repeat guard
        # already blocks the obvious case; this bounds the rest, and is
        # the first half of the cost ceiling Decision D calls for.
        "recursion_limit": max_steps,
        # One thread per turn. Nothing outlives the turn today; Phase 3
        # is where a thread becomes a conversation.
        "configurable": {"thread_id": uuid.uuid4().hex},
    }

    state = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools or [],
            "answer": None,
            "failed": [],
            "events": [],
        },
        config=config_,
    )

    while state.get("__interrupt__"):
        requests = [item.value for item in state["__interrupt__"]]

        if approve is None:
            # Refusing by default. A turn with no way to ask a human must
            # not proceed as though one had answered.
            decisions = [{"approved": False} for _ in requests]
        else:
            decisions = await _decide(approve, requests, approval_timeout_seconds)

        state = await app.ainvoke(
            Command(resume=decisions[0] if len(decisions) == 1 else decisions),
            config=config_,
        )

    return state
```

And the type alias beside the others:

```python
ApprovalCallback = Callable[[dict], Awaitable[dict]]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -v`
Expected: PASS. `_decide` does not exist yet — write it as a bare `await approve(...)` for
now; Task 5.4 gives it the deadline and its own tests.

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py agent/loop.py tests/test_agent_approval.py
git commit -m "feat: pause the turn on a high-risk tool call"
```

---

### Task 5.3: Resume uses the arguments the human saw

**Files:**
- Test: `tests/test_agent_approval.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_agent_approval.py


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
    # cannot have been re-authored in between -- this asserts that
    # structurally rather than trusting it.
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
    # invents one anyway, code overwrites it.
    executor = recording_executor({"cancel_order": {"status": "CANCELLED"}})

    async def approve(request):
        return {"approved": True, "token": "real-token"}

    await run_turn(
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


async def test_the_agent_never_mints_its_own_approval():
    # An exit criterion, asserted structurally rather than behaviourally:
    # the agent package must not reference the minting route or module at
    # all. A behavioural test only proves it did not mint this time.
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent / "agent"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )

    assert "/approvals" not in source
    assert "import approvals" not in source
    assert "from approvals" not in source
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -v`
Expected: the token-injection tests fail if 5.2's injection was not written; the
never-mints test should pass immediately, which is correct — it is a regression guard, not a
red-then-green step.

- [ ] **Step 3: Implementation** — none expected beyond 5.2. If a test fails, fix in
`agent/loop.py` and note what was wrong.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: PASS, including `tests/test_agent_loop.py` and `tests/test_agent_events.py`
unchanged.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agent_approval.py
git commit -m "test: the resumed call uses the arguments the human approved"
```

---

### Task 5.4: A pause nobody answers

**Files:**
- Modify: `agent/loop.py`
- Test: `tests/test_agent_approval.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_agent_approval.py


async def test_an_unanswered_pause_times_out_rather_than_waiting_forever():
    # LangGraph has no built-in timeout; the design document flags this.
    # A held MCP session makes it a resource leak, not just an idle task.
    executor = recording_executor({})

    async def never_answers(request):
        await asyncio.sleep(3600)

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=executor,
        approve=never_answers,
        approval_timeout_seconds=0.05,
    )

    assert executor.calls == []
    tool = replay(state["events"])["tools"][0]
    assert tool["ok"] is False
    assert "expired" in tool["error"].lower()


async def test_a_turn_with_no_way_to_ask_a_human_refuses_rather_than_proceeding():
    # The safe default. An agent deployed without an approval channel must
    # not behave as though every request had been granted.
    executor = recording_executor({})

    state = await run_turn(
        "cancel my most recent order",
        model_call=cancel_turn(),
        execute_tool=executor,
    )

    assert executor.calls == []
    assert state["answer"] == "Cancelled."  # the model's words; nothing happened
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_approval.py -k timeout -v`
Expected: FAIL — hangs or reports the wrong error text.

- [ ] **Step 3: Write the implementation**

```python
async def _decide(
    approve: ApprovalCallback, requests: list[dict], timeout_seconds: float
) -> list[dict]:
    """Ask a human, under a deadline.

    LangGraph has no timeout of its own, and the design document flags
    that. It matters more here than it would elsewhere: the turn is
    holding an MCP session open while it waits, so a pause nobody answers
    leaks a server-side session and not merely a Python task.

    A deadline that passes is a refusal, never a grant.
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*(approve(request) for request in requests)),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return [
            {"approved": False, "reason": "expired"} for _ in requests
        ]
```

The declined branch in `execute_tools` distinguishes the two, because "you said no" and
"nobody was there" are different things to show a customer:

```python
            if decision is not None and not decision.get("approved"):
                declined = (
                    "The approval request expired before anyone answered it. "
                    "Nothing was sent."
                    if decision.get("reason") == "expired"
                    else "The customer declined this action. Nothing was sent."
                )
```

Add `import asyncio` and `import uuid` at the top of `agent/loop.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py tests/test_agent_approval.py
git commit -m "feat: a pause nobody answers expires as a refusal"
```

---

### Task 5.5: Offer the tool

**Files:**
- Modify: `agent/tools.py`, `agent/loop.py`
- Test: `tests/test_agent_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_agent_tools.py


def test_the_agent_is_now_offered_the_high_risk_tool():
    # It was withheld until the pause existed. The tool is reachable; the
    # approval is what makes it safe, and the MCP server enforces that
    # independently of anything decided here.
    from agent.tools import AGENT_TOOLS, HIGH_RISK_TOOLS, KNOWN_TOOLS

    assert AGENT_TOOLS == KNOWN_TOOLS
    assert HIGH_RISK_TOOLS <= AGENT_TOOLS


def test_the_model_is_still_never_shown_the_approval_field(sample_tools):
    # Unchanged from Task 1, and worth re-asserting now that the tool is
    # actually offered: a field the model cannot see is a field it cannot
    # invent a value for.
    from agent.tools import translate_tools

    cancel = [
        t for t in translate_tools(sample_tools) if t["function"]["name"] == "cancel_order"
    ][0]

    assert "approval_token" not in cancel["function"]["parameters"]["properties"]
```

If `test_agent_tools.py` has no `sample_tools` fixture, reuse whatever it already builds a
`Tool` list with; do not introduce a second way of making one.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_tools.py -v`
Expected: FAIL on the first test — `AGENT_TOOLS` still excludes `cancel_order`.

- [ ] **Step 3: Write the implementation**

In `agent/tools.py`:

```python
# What the agent is offered. cancel_order was withheld until M4 Task 5,
# because a tool the agent is never shown is one it cannot call and there
# was no pause to guard it. The pause exists now, and the MCP server
# enforces the approval independently -- the agent's behaviour is the UX,
# the server's rejection is the boundary.
AGENT_TOOLS = READ_ONLY_TOOLS | MEDIUM_RISK_TOOLS | HIGH_RISK_TOOLS
```

`answer()` moves onto the held session and gains the callback:

```python
async def answer(
    utterance: str,
    token: str,
    *,
    model: str | None = None,
    approve: ApprovalCallback | None = None,
    approval_timeout_seconds: float = 300.0,
) -> TurnState:
    """The whole thing wired to the real model and the real MCP server.

    One session for the turn, held across any approval pause, because the
    token is minted against its id.
    """
    tools = await list_openai_tools(token, only=AGENT_TOOLS)

    async with session_scoped_executor(token) as session:
        state = await run_turn(
            utterance,
            model_call=openai_model_call(model),
            execute_tool=session.execute,
            tools=tools,
            approve=approve,
            approval_timeout_seconds=approval_timeout_seconds,
        )

    # The id the storefront must mint against. Server-side only: it is
    # deliberately NOT in the event stream, which reaches a browser.
    state["session_id"] = session.session_id
    return state
```

`session_id` is read after the context closes, which is why `McpSession` caches it — add to
`McpSession.__init__`:

```python
        self._session_id = None
```

and in `session_scoped_executor`, capture it once the session is live:

```python
    async with client:
        session = McpSession(client)
        session._session_id = client.transport.get_session_id()
        yield session
```

with the property returning the cached value. A session id read after the transport closes is
otherwise `None`, and the storefront would mint against nothing.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py agent/loop.py tests/test_agent_tools.py
git commit -m "feat: offer cancel_order now that the pause guards it"
```

---

### Task 5.6: The live gate

**Files:**
- Create: scratchpad script only. Nothing committed from this step except the record.

- [ ] **Step 1: Cancel a real order, end to end.**

The script must, against production:

1. Mint a customer bearer from `/api/v1/auth/token`.
2. Call `answer("cancel my most recent order", token, approve=...)`.
3. In the `approve` callback — standing in for the storefront's approve route — POST to the
   MCP server's `/approvals` with the customer's bearer, `mcp-session-id` set to the agent's
   session id, `{"tool": "cancel_order", "args": {"order_id": <the id in the request>}}`, and
   return `{"approved": True, "token": <minted>}`.
4. Print the event stream, the replay, and the order's status before and after.

The callback needs the session id before the turn returns. Expose it by having
`session_scoped_executor` set it on a holder the script can read, or run `answer` with an
`approve` closure that reads `state` lazily — simplest is a small `Holder` object the script
passes in. Whatever shape, it must NOT become a way for the agent itself to reach `/approvals`.

- [ ] **Step 2: Verify the four MUST PROVEs against the run**
  - the agent never called `/approvals` — the mint happened in the callback, outside the
    agent package, and the source-level test already asserts the package cannot;
  - the resumed call used the arguments in the `approval_required` event;
  - a declined run sends nothing — repeat with `{"approved": False}` and confirm the order
    is still PENDING;
  - the timeout path returns cleanly — repeat with a callback that sleeps.

- [ ] **Step 3: Cross-check the cancellation in the browser**, as Phase 1 verification did.
  A 200 from the API is not the same fact as an order showing CANCELLED.

- [ ] **Step 4: Record the results** in `docs/PLAN_M4_AGENT.txt` Task 5, including the
  session-identity finding, and add a note to `contracts/README.md` that
  `approval_required` is now emitted and that `session_id` travels server-side rather than
  in the event.

- [ ] **Step 5: Commit and push both repositories.**

---

## Self-Review

**Spec coverage.** The four MUST PROVEs map to: never mints (5.3, structural); resumed
arguments are the approved ones (5.3); refusing sends nothing (5.2); a token that never
arrives times out cleanly (5.4). The exit criterion "a high-risk action always pauses for a
human" is 5.2 plus the no-callback default in 5.4.

**Placeholders.** None. Step 5.6 describes a scratchpad script rather than committed code,
and states exactly what it must do and what it must not become.

**Type consistency.** `ApprovalCallback` takes the interrupt payload (`call_id`, `tool`,
`arguments`) and returns `{"approved": bool, "token": str | None, "reason": str | None}`.
That is the shape produced by `_decide`, consumed by `execute_tools`, and asserted in every
test. `McpSession.execute` matches the existing `ToolExecutor` alias, so it drops into
`run_turn` unchanged.

**One risk carried deliberately.** `InMemorySaver` and the held MCP session both assume one
process. That is true of the current deployment and is the same limitation `approvals.py`
already documents for its spent-nonce set. Phase 3, or a second replica, needs a shared
checkpointer — recorded in the plan rather than solved here.
