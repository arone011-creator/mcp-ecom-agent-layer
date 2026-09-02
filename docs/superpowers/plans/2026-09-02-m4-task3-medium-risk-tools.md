# M4 Task 3 — Medium-Risk Tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `PLAN_M4_AGENT.txt` Task 3 — add `add_to_cart` and `remove_from_cart`. They execute without blocking. MUST PROVE: an over-stock 409 is handled by retrying with the available number rather than repeating the same request.

**Architecture:** The surface grows by two tools. The real work is that the loop currently has *no error handling at all* — a failing tool call raises out of `execute_tools` and kills the turn, so the model never gets the chance to react that the 409 was designed to give it. This task turns tool failures into tool results, and adds a guard that makes "rather than repeating the same request" a property of the code rather than a hope about the model.

**Written against the real system, at the time.** Verified by running code on 2026-09-02:

- An over-stock add against production raises **`fastmcp.exceptions.ToolError`** (MRO: `ToolError` → `FastMCPError` → `Exception`), with `str(e)` = `Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 67`.
- **There is no structured error payload.** No `.data`, `.code`, or `.message` attribute exists on the exception — the status and the available number reach the agent *only as text inside that string*.
- The storefront generates it in `app/api/v1/cart/route.ts:175-180` as a bare `fail(409, \`Only ${available} available; cart would hold ${nextQuantity}\`)`, with a comment stating the intent outright: *"An agent should re-read stock and try a smaller number, not rewrite its request."*
- `recursion_limit` is passed via the invoke config and raises `langgraph.errors.GraphRecursionError` when exceeded.

## Three design decisions

**1. The 409 message is passed through verbatim, never parsed.** It would be easy to regex `Only (\d+) available` and retry automatically. That is the wrong instinct here: the message is prose the storefront owns, and parsing it in Python creates a second implementation of a rule that already has one — the exact failure mode this project has warned about since M3. If the wording ever changes, a parser breaks silently while a pass-through keeps working. The message was written to be read; let the model read it.

**2. "Rather than repeating the same request" is enforced by code, not hoped for.** A guard records every `(tool, arguments)` pair that failed during a turn and refuses an identical retry, returning a result that says so. The model stays free to choose *any* smaller quantity; it is only prevented from making the exact call that just failed. This makes the MUST PROVE deterministic and unit-testable, and it protects against a token-burning loop — the concrete failure the cost-ceiling decision exists to bound.

**3. "Informational event, not a blocking prompt" needs no code here.** Medium-risk tools execute without a gate, which is already the behaviour — there is nothing to build. The *event* half arrives with the event schema in the parent plan's Task 4; building an event emitter now would be inventing a contract the storefront has not frozen yet.

---

## File Structure

| File | Change |
|---|---|
| `agent/tools.py` | Modify — add `MEDIUM_RISK_TOOLS` and `AGENT_TOOLS` |
| `agent/loop.py` | Modify — tool failures become tool results; repeat guard; recursion cap |
| `tests/test_agent_tools.py` | Modify — surface tests |
| `tests/test_agent_loop.py` | Modify — error handling, the 409 path, the repeat guard |

---

### Task 1: Grow the tool surface

**Files:** `agent/tools.py`, `tests/test_agent_tools.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_tools.py`:

```python
def test_the_medium_risk_surface_is_the_two_cart_writes():
    from agent.tools import MEDIUM_RISK_TOOLS

    assert MEDIUM_RISK_TOOLS == {"add_to_cart", "remove_from_cart"}


def test_the_agent_surface_is_read_only_plus_medium_but_not_cancel():
    # cancel_order is High risk and needs the approval machinery that
    # arrives in Task 5. Until then the agent is not offered it at all.
    from agent.tools import AGENT_TOOLS

    assert AGENT_TOOLS == READ_ONLY_TOOLS | {"add_to_cart", "remove_from_cart"}
    assert "cancel_order" not in AGENT_TOOLS
    assert AGENT_TOOLS < KNOWN_TOOLS


def test_the_agent_surface_translates_to_eight_tools():
    from agent.tools import AGENT_TOOLS

    translated = translate_tools(all_nine(), only=AGENT_TOOLS)

    assert len(translated) == 8
    assert "cancel_order" not in {t["function"]["name"] for t in translated}
```

- [ ] **Step 2: Run them, confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -q`
Expected: `ImportError: cannot import name 'MEDIUM_RISK_TOOLS'`.

- [ ] **Step 3: Add the two sets**

In `agent/tools.py`, find:
```python
# Identity is never an argument. It is resolved from the bearer token by
```

Insert immediately before it:
```python
# Medium risk: they change something, and they execute without a gate.
# Reversible in both directions -- remove what was added, re-add what was
# removed -- and the API scopes both to the caller's own cart.
MEDIUM_RISK_TOOLS = frozenset({"add_to_cart", "remove_from_cart"})

# What the agent is offered today. cancel_order is deliberately absent:
# it is High risk and only becomes reachable with the approval machinery
# in Task 5. A tool the agent is never shown is one it cannot call.
AGENT_TOOLS = READ_ONLY_TOOLS | MEDIUM_RISK_TOOLS


```

- [ ] **Step 4: Run the tests, then the suite**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -q`
Expected: 19 passed.

Run: `.venv/Scripts/python -m pytest -q`
Expected: 143 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py tests/test_agent_tools.py
git commit -m "feat: offer the agent the two medium-risk cart tools

add_to_cart and remove_from_cart execute without a gate - reversible
both ways, and the API scopes each to the caller's own cart.
cancel_order stays out of the surface entirely until the approval
machinery exists: a tool the agent is never shown is one it cannot
call.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Tool failures become tool results, and a repeat cannot happen

**Files:** `tests/test_agent_loop.py`, `agent/loop.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_loop.py`:

```python
def failing_executor(error_message, fail_on=None):
    """An MCP stand-in that raises for a given tool, recording every attempt."""
    from fastmcp.exceptions import ToolError

    calls = []

    async def execute(name, arguments):
        calls.append((name, arguments))
        if fail_on is None or name == fail_on:
            raise ToolError(error_message)
        return {"ok": True}

    execute.calls = calls
    return execute


async def test_a_failing_tool_becomes_a_result_the_model_can_read():
    # Without this the 409 kills the turn and the model never gets the
    # chance to react that the status code exists to give it.
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 67"
    )

    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":67}')
                ]
            ),
            FakeMessage(content="Only 17 are available - shall I add those?"),
        ),
        execute_tool=executor,
    )

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    # Verbatim: the number the model needs is in the storefront's own
    # words, and this layer does not parse or reword them.
    assert "Only 17 available" in tool_messages[0]["content"]
    assert state["answer"] == "Only 17 are available - shall I add those?"


async def test_the_available_number_is_passed_through_not_parsed():
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 3 available; cart would hold 9"
    )

    state = await run_turn(
        "add 9",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":9}')
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
    )

    content = [m for m in state["messages"] if m.get("role") == "tool"][0]["content"]
    assert "Only 3 available" in content
    assert "cart would hold 9" in content


async def test_an_identical_failed_call_is_refused_rather_than_repeated():
    # The MUST PROVE of this task. The model asks for the same thing twice;
    # the second attempt never reaches the MCP server.
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 67"
    )

    same_call = '{"product_id":"p1","quantity":67}'
    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "add_to_cart", same_call)]),
            FakeMessage(tool_calls=[FakeToolCall("call_2", "add_to_cart", same_call)]),
            FakeMessage(content="I'll stop asking for 67."),
        ),
        execute_tool=executor,
    )

    # Executed once. The repeat was refused before it left the process.
    assert len(executor.calls) == 1

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "already" in tool_messages[1]["content"].lower()


async def test_a_different_quantity_after_a_failure_is_allowed_through():
    # The guard blocks the identical call, not the retry. Asking for a
    # smaller number is exactly what the 409 is telling it to do.
    from fastmcp.exceptions import ToolError

    calls = []

    async def execute(name, arguments):
        calls.append(arguments["quantity"])
        if arguments["quantity"] > 17:
            raise ToolError(
                "Error calling tool 'add_to_cart': 409: Only 17 available; "
                f"cart would hold {arguments['quantity']}"
            )
        return {"itemCount": arguments["quantity"]}

    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":67}')
                ]
            ),
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_2", "add_to_cart", '{"product_id":"p1","quantity":17}')
                ]
            ),
            FakeMessage(content="Added 17."),
        ),
        execute_tool=execute,
    )

    assert calls == [67, 17]
    assert state["answer"] == "Added 17."


async def test_a_forbidden_argument_still_raises_rather_than_becoming_a_result():
    # An identity argument is not a recoverable tool failure the model
    # should get a chance to work around - it is a refusal.
    executor = recording_executor({})

    with pytest.raises(ForbiddenArgumentError):
        await run_turn(
            "orders",
            model_call=scripted_model(
                FakeMessage(
                    tool_calls=[FakeToolCall("call_1", "get_orders", '{"user_id":"u1"}')]
                )
            ),
            execute_tool=executor,
        )
```

- [ ] **Step 2: Run them, confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent_loop.py -q`
Expected: the new tests fail — `ToolError` propagates out of `run_turn` instead of becoming a tool result.

- [ ] **Step 3: Handle tool failures in the loop**

In `agent/loop.py`, find:
```python
class TurnState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tools: list[dict]
    answer: str | None
```

Replace with:
```python
class TurnState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tools: list[dict]
    answer: str | None
    # Every (tool, arguments) pair that has already failed this turn.
    # Accumulated so the loop can refuse an identical retry.
    failed: Annotated[list[str], operator.add]
```

Find:
```python
    async def execute_tools(state: TurnState) -> dict:
        results = []

        for call in _tool_calls_of(state["messages"][-1]):
            name = call["function"]["name"]
            # Arguments arrive as a JSON string and the escaping varies.
            # Parse; never string-match.
            arguments = json.loads(call["function"]["arguments"] or "{}")

            reject_forbidden_arguments(name, arguments)

            result = await execute_tool(name, arguments)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, default=str),
                }
            )

        return {"messages": results}
```

Replace with:
```python
    async def execute_tools(state: TurnState) -> dict:
        results = []
        newly_failed = []
        already_failed = set(state.get("failed", []))

        for call in _tool_calls_of(state["messages"][-1]):
            name = call["function"]["name"]
            # Arguments arrive as a JSON string and the escaping varies.
            # Parse; never string-match.
            arguments = json.loads(call["function"]["arguments"] or "{}")

            # Not a recoverable failure: identity is never the model's to
            # assert, so this refuses the turn rather than inviting a retry.
            reject_forbidden_arguments(name, arguments)

            signature = _signature(name, arguments)

            if signature in already_failed:
                results.append(
                    _tool_message(
                        call["id"],
                        {
                            "error": "This exact call was already tried this turn "
                            "and failed. Read the earlier error and change the "
                            "arguments rather than repeating it."
                        },
                    )
                )
                continue

            try:
                result = await execute_tool(name, arguments)
            except ToolError as failure:
                # Passed through verbatim. The storefront writes these to
                # be acted on -- a 409 carries the number that IS
                # available -- and re-wording or parsing them here would
                # be a second implementation of someone else's rule.
                newly_failed.append(signature)
                results.append(_tool_message(call["id"], {"error": str(failure)}))
                continue

            results.append(_tool_message(call["id"], result))

        return {"messages": results, "failed": newly_failed}
```

Find:
```python
def _tool_calls_of(message: dict) -> list[dict]:
    return message.get("tool_calls") or []
```

Replace with:
```python
def _tool_calls_of(message: dict) -> list[dict]:
    return message.get("tool_calls") or []


def _signature(name: str, arguments: dict) -> str:
    """Identifies one exact call, so a repeat of it can be recognised."""
    return json.dumps({"tool": name, "args": arguments}, sort_keys=True)


def _tool_message(call_id: str, payload) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, default=str),
    }
```

Add the import. Find:
```python
from langgraph.graph import END, START, StateGraph
```

Replace with:
```python
from fastmcp.exceptions import ToolError
from langgraph.graph import END, START, StateGraph
```

- [ ] **Step 4: Seed and cap the turn**

Find:
```python
    return await app.ainvoke(
        {
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools or [],
            "answer": None,
        }
    )
```

Replace with:
```python
    return await app.ainvoke(
        {
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools or [],
            "answer": None,
            "failed": [],
        },
        # A confused agent stops rather than looping. The repeat guard
        # already blocks the obvious case; this bounds the rest, and is
        # the first half of the cost ceiling Decision D calls for.
        config={"recursion_limit": max_steps},
    )
```

Find:
```python
async def run_turn(
    utterance: str,
    *,
    model_call: ModelCall,
    execute_tool: ToolExecutor,
    tools: list[dict] | None = None,
) -> TurnState:
```

Replace with:
```python
async def run_turn(
    utterance: str,
    *,
    model_call: ModelCall,
    execute_tool: ToolExecutor,
    tools: list[dict] | None = None,
    max_steps: int = 25,
) -> TurnState:
```

- [ ] **Step 5: Point the real wiring at the fuller surface**

Find:
```python
from agent.tools import (
    READ_ONLY_TOOLS,
    build_transport,
    list_openai_tools,
    reject_forbidden_arguments,
)
```

Replace with:
```python
from agent.tools import (
    AGENT_TOOLS,
    build_transport,
    list_openai_tools,
    reject_forbidden_arguments,
)
```

Find:
```python
    tools = await list_openai_tools(token, only=READ_ONLY_TOOLS)
```

Replace with:
```python
    tools = await list_openai_tools(token, only=AGENT_TOOLS)
```

- [ ] **Step 6: Run the tests, then the suite**

Run: `.venv/Scripts/python -m pytest tests/test_agent_loop.py -q`
Expected: 11 passed (6 existing + 5 new).

Run: `.venv/Scripts/python -m pytest -q`
Expected: 148 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/loop.py tests/test_agent_loop.py
git commit -m "feat: a failing tool becomes a result the model can act on

The loop had no error handling: a 409 raised out of the tool node and
killed the turn, so the model never got the chance to react that the
status code exists to give it. Failures now come back as tool results
carrying the storefront's own message verbatim - the 409 names how many
ARE available, and parsing or rewording that here would be a second
implementation of someone else's rule.

An identical call that already failed this turn is refused before it
leaves the process, which makes 'retry smaller rather than repeat' a
property of the code rather than a hope about the model. A recursion
cap bounds the rest.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The live gate — a real over-stock retry

**Files:** none modified.

- [ ] **Step 1: Ask for more than exists, against production**

```bash
.venv/Scripts/python -c "
import asyncio, httpx, json
from agent.loop import answer
from agent.tools import build_transport
from fastmcp import Client

API = 'https://web-production-bb55d.up.railway.app'
MCP = 'https://mcp-production-e344.up.railway.app/mcp'

async def main():
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(f'{API}/api/v1/auth/token',
            json={'email':'customer@example.com','password':'demo1234','ttlSeconds':900})
        r.raise_for_status()
        token = r.json()['data']['token']

    async with Client(build_transport(MCP, token)) as c:
        p = json.loads((await c.call_tool('search_products', {'query':'headphones','limit':1})).content[0].text)['products'][0]
        stock = json.loads((await c.call_tool('check_inventory', {'product_id': p['id']})).content[0].text)
    print('product:', p['name'], '| available:', stock['available'])

    state = await answer(
        f\"Add {stock['available'] + 40} of the {p['name']} to my cart.\", token)

    print()
    for m in state['messages']:
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            print('  called:', [(c['function']['name'], c['function']['arguments']) for c in m['tool_calls']])
        elif m.get('role') == 'tool':
            print('  result:', m['content'][:160])
    print()
    print('ANSWER:', state['answer'])

asyncio.run(main())
"
```

Expected: a first `add_to_cart` for the over-stock number, a tool result
carrying `409: Only N available`, and then either a second `add_to_cart`
with a **smaller** quantity, or an answer telling the customer only N are
available. Both satisfy the MUST PROVE; what must NOT appear is the same
quantity requested twice.

- [ ] **Step 2: Tidy the demo cart**

The run may leave items in the demo customer's cart. Clear it so the next
person starts from a known state:

```bash
.venv/Scripts/python -c "
import asyncio, httpx, json
from agent.tools import build_transport
from fastmcp import Client
API='https://web-production-bb55d.up.railway.app'
MCP='https://mcp-production-e344.up.railway.app/mcp'
async def main():
    async with httpx.AsyncClient(timeout=20) as http:
        r=await http.post(f'{API}/api/v1/auth/token',
            json={'email':'customer@example.com','password':'demo1234','ttlSeconds':900})
        token=r.json()['data']['token']
    async with Client(build_transport(MCP, token)) as c:
        await c.call_tool('remove_from_cart', {})
        print('cart:', (await c.call_tool('get_cart', {})).content[0].text[:120])
asyncio.run(main())
"
```

No commit — nothing under version control changed.

---

### Task 4: Mark the parent plan

**Files:** `docs/PLAN_M4_AGENT.txt`

- [ ] **Step 1: Record the outcome**

Find:
```
TASK 3 - MEDIUM-RISK TOOLS
    Add add_to_cart and remove_from_cart. These execute, then surface as an
    informational event - not a blocking prompt.
    MUST PROVE: an over-stock 409 is handled by retrying with the available
    number rather than repeating the same request. The API returns that
    number for exactly this purpose.
```

Replace with:
```
TASK 3 - MEDIUM-RISK TOOLS    [DONE 2026-09-02]
    Add add_to_cart and remove_from_cart. These execute, then surface as an
    informational event - not a blocking prompt.
    MUST PROVE: an over-stock 409 is handled by retrying with the available
    number rather than repeating the same request. The API returns that
    number for exactly this purpose.

    Proved. The work was not the two tools - it was that the loop had no
    error handling at all, so a 409 killed the turn instead of giving the
    model the chance the status code exists to provide. Failures now come
    back as tool results carrying the storefront's message verbatim; the
    number is never parsed out, because that message is prose the
    storefront owns and a parser here would be a second implementation of
    its rule.

    "Rather than repeating the same request" is enforced, not hoped for:
    an identical (tool, arguments) call that already failed this turn is
    refused before it leaves the process. The model stays free to pick any
    smaller quantity.

    The informational-event half is deliberately not built. Medium tools
    already execute without a gate; the event itself needs the schema the
    storefront freezes in its own Task 1, and inventing that contract early
    is how the two halves end up disagreeing.
```

- [ ] **Step 2: Commit**

```bash
git add docs/PLAN_M4_AGENT.txt
git commit -m "docs: mark M4 Task 3 done, with what it actually took

The two tools were the easy half. The real work was that the loop had
no error handling, so an over-stock 409 killed the turn rather than
giving the model the chance to retry smaller.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Push

- [ ] **Step 1: Review pending, then push after user confirmation**

```bash
git log --oneline origin/main..HEAD
git push origin main
```
