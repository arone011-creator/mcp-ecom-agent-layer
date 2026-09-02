# M4 Task 2 — The Agent Loop, Read-Only Tools, on OpenAI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `PLAN_M4_AGENT.txt` Task 2 — a single turn: utterance in, tool calls out, answer back, wired to the six low-risk tools only. MUST PROVE: workflow 1 ("what did I order recently") end to end against the live MCP server, and that no user id is ever sent as an argument.

**Provider change:** this milestone now runs on **OpenAI**, not Anthropic — decided 2026-09-02 because the project has OpenAI credits and no Anthropic key. LangGraph was chosen partly because it is not tied to one vendor, and that holds: the MCP server, the nine tools, the approval-token architecture, the risk tiers, and the untrusted-content marking are all provider-independent and unchanged. What changes is the tool-definition shape, the model call, and the docs that named Claude.

**Architecture:** LangGraph owns the graph, state, and (from Task 5 of the parent plan) `interrupt`. The OpenAI SDK is called directly from a plain function node — no `langchain-openai` wrapper, so model parameters stay under our control. Two nodes and a router: `call_model` → (tool calls? → `execute_tools` → back to `call_model`) : END.

**Tech Stack:** Python 3.11+, `openai` 3.7.0, `langgraph` 1.2.11, `fastmcp` (existing). Model `gpt-4.1`, config-driven via `OPENAI_MODEL`.

**Written against the real libraries, at the time.** Everything below was verified by running code on 2026-09-02, not recalled:

- **LangGraph 1.2.11:** plain functions work as nodes; a graph compiles and runs with **no checkpointer**; `add_conditional_edges` gives the tool-loop shape; `ainvoke` exists (needed — the MCP client is async). `interrupt` and `Command(resume=...)` live at `langgraph.types`, `InMemorySaver` at `langgraph.checkpoint.memory` — both confirmed present for the parent plan's Task 5, neither used here.
- **The exact state semantics this loop depends on**, run end to end: `Annotated[list, operator.add]` accumulates across nodes (a model→tools→model cycle produced `user, assistant, tool, assistant` in order); a field with no reducer is *replaced* rather than accumulated (which is what makes `answer` hold the last model turn); and a key no node returns passes through untouched (which is how `tools` survives the loop).
- **OpenAI 3.7.0, verified with a live `gpt-4.1` round-trip:**
  - Tool shape `{"type": "function", "function": {"name", "description", "parameters"}}` is accepted with the MCP `inputSchema` passed straight through as `parameters` — Pydantic's `anyOf`/`default: null` and all. The translation is a **re-nesting, not a schema rewrite**.
  - `resp.choices[0].finish_reason == "tool_calls"` is the loop signal; `message.content` is `None` on a tool turn.
  - `tool_calls[i].id` (e.g. `call_tD6lmi...`), `.type == "function"`, `.function.name`, and `.function.arguments` — **a JSON string, not a dict.** Parse it; never string-match it.
  - Results go back as the assistant turn (`message.model_dump(exclude_none=True)`) followed by `{"role": "tool", "tool_call_id": ..., "content": <string>}`. Round two returned `finish_reason: "stop"` and a correct answer.
  - `max_completion_tokens` is the current parameter name.
  - `from openai.types.chat import ChatCompletionToolParam` — confirmed importable.

**Executed 2026-09-02. Both MUST PROVEs met; two deviations.**

The live gate is the one that matters: the agent called `get_orders` with
`{"limit":5}` — schema arguments only, no identity — and answered with the
demo account's three real orders at exactly the totals and `CANCELLED`
states left behind by the Phase 1 sweep work earlier the same day
($1,089.98 / $3,681.96 / $2,385.97). Those numbers could not have been
invented, which is what makes this end-to-end rather than a mock passing.

Deviations:

1. *Task 1, Step 4* — the plan expected "the 11 tool tests fail, everything
   else passes." A collection error interrupts the whole pytest run, so
   nothing else reports at all. Re-ran with `--ignore=tests/test_agent_tools.py`
   to confirm the other 118 were green. The step's intent held; its
   instruction was unrunnable as written.
2. *Task 5, Step 4* — the grep found three stale Claude references the plan
   only expected in §3: Decision B ("Claude tool definitions"), Task 0
   ("the Anthropic SDK"), and Task 1 ("Claude tool definitions"). All three
   contradicted code already shipped, so all three were fixed, and Tasks 0,
   1 and 2 were marked done with what each actually proved.

---

## File Structure

| File | Change |
|---|---|
| `agent/tools.py` | Modify — emit OpenAI's tool shape instead of Claude's; add the read-only filter and the forbidden-argument guard |
| `agent/loop.py` | Create — the LangGraph graph and the turn runner |
| `tests/test_agent_tools.py` | Modify — retarget the 11 tests, add guard/filter tests |
| `tests/test_agent_loop.py` | Create — the loop, with a stubbed model |
| `config.py` | Modify — `OPENAI_MODEL`; drop nothing else |
| `requirements.txt` | Modify — `anthropic` out, `openai` + `langgraph` in |
| `docs/PLAN_M4_AGENT.txt` | Modify — §3 rewritten for OpenAI; §1 Decision A note |

---

### Task 1: Dependencies and config for the provider switch

**Files:** `requirements.txt`, `config.py`

- [x] **Step 1: Swap the SDK in `requirements.txt`**

Find:
```
anthropic>=1,<2
fastmcp>=2.3,<3
httpx>=0.27,<0.29
pydantic>=2.7,<3
```

Replace with:
```
fastmcp>=2.3,<3
httpx>=0.27,<0.29
langgraph>=1.2,<2
openai>=3,<4
pydantic>=2.7,<3
```

- [x] **Step 2: Uninstall the now-unused SDK and reinstall**

Note: this venv's `pip.exe` shim is broken (exits 1 silently) — use `python -m pip`.

Run:
```bash
.venv/Scripts/python -m pip uninstall -y anthropic
.venv/Scripts/python -m pip install -r requirements-dev.txt
```
Expected: `anthropic` removed; `openai` and `langgraph` present. Confirm:
`.venv/Scripts/python -c "import openai, langgraph; print(openai.__version__)"` → `3.7.0`.

- [x] **Step 3: Add the model setting to `config.py`**

Find:
```python
# Where the agent finds the MCP server. Read by agent/, not by the server
# itself -- the server does not call itself.
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL", "https://mcp-production-e344.up.railway.app/mcp"
)
```

Replace with:
```python
# Where the agent finds the MCP server. Read by agent/, not by the server
# itself -- the server does not call itself.
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL", "https://mcp-production-e344.up.railway.app/mcp"
)

# The model behind the agent. OPENAI_API_KEY is read by the SDK itself and
# deliberately not mirrored here -- one fewer place a key can be logged.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
```

- [x] **Step 4: Confirm the suite still passes**

Run: `.venv/Scripts/python -m pytest -q`
Expected: the 11 tool tests FAIL (they import `to_claude_tool`, which Task 2 renames) — that is expected and is what Task 2 fixes. Everything else passes. If anything *else* broke, stop.

Do not commit yet — the tree is intentionally red until Task 2.

---

### Task 2: Retarget the tool translation to OpenAI's shape

**Files:** `tests/test_agent_tools.py`, `agent/tools.py`

- [x] **Step 1: Retarget the tests first**

In `tests/test_agent_tools.py`, change the import block:

Find:
```python
from agent.tools import (
    KNOWN_TOOLS,
    UnknownToolError,
    build_transport,
    to_claude_tool,
    translate_tools,
)
```

Replace with:
```python
from agent.tools import (
    FORBIDDEN_ARGUMENTS,
    KNOWN_TOOLS,
    READ_ONLY_TOOLS,
    ForbiddenArgumentError,
    UnknownToolError,
    build_transport,
    reject_forbidden_arguments,
    to_openai_tool,
    translate_tools,
)
```

Then replace the three shape-specific tests. Find:
```python
def test_translation_renames_the_schema_key_to_claude_spelling():
    schema = {
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
        "type": "object",
    }

    translated = to_claude_tool(mcp_tool("get_product", schema))

    assert translated == {
        "name": "get_product",
        "description": "d",
        "input_schema": schema,
    }
    assert "inputSchema" not in translated


def test_a_missing_description_becomes_empty_rather_than_none():
    translated = to_claude_tool(mcp_tool("get_cart", description=None))

    assert translated["description"] == ""
```

Replace with:
```python
def test_translation_nests_the_schema_the_way_openai_wants_it():
    # Verified against a live gpt-4.1 call: the MCP schema is accepted as
    # `parameters` unchanged, anyOf/default and all. This is a re-nesting,
    # not a schema rewrite.
    schema = {
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
        "type": "object",
    }

    translated = to_openai_tool(mcp_tool("get_product", schema))

    assert translated == {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "d",
            "parameters": schema,
        },
    }
    assert "inputSchema" not in translated["function"]


def test_a_missing_description_becomes_empty_rather_than_none():
    translated = to_openai_tool(mcp_tool("get_cart", description=None))

    assert translated["function"]["description"] == ""
```

Then update the remaining `to_claude_tool` call sites and their assertions. Find:
```python
def test_all_nine_translate():
    translated = translate_tools(all_nine())

    assert len(translated) == 9
    assert {t["name"] for t in translated} == KNOWN_TOOLS
```

Replace with:
```python
def test_all_nine_translate():
    translated = translate_tools(all_nine())

    assert len(translated) == 9
    assert {t["function"]["name"] for t in translated} == KNOWN_TOOLS
```

Find:
```python
    translated = to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" not in translated["input_schema"]["properties"]
    assert "order_id" in translated["input_schema"]["properties"]
    assert translated["input_schema"]["required"] == ["order_id"]
```

Replace with:
```python
    translated = to_openai_tool(mcp_tool("cancel_order", schema))

    params = translated["function"]["parameters"]
    assert "approval_token" not in params["properties"]
    assert "order_id" in params["properties"]
    assert params["required"] == ["order_id"]
```

Find:
```python
    to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" in schema["properties"]
```

Replace with:
```python
    to_openai_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" in schema["properties"]
```

Find:
```python
    translated = to_claude_tool(mcp_tool("add_to_cart", schema))

    assert set(translated["input_schema"]["properties"]) == {"product_id", "quantity"}
```

Replace with:
```python
    translated = to_openai_tool(mcp_tool("add_to_cart", schema))

    assert set(translated["function"]["parameters"]["properties"]) == {
        "product_id",
        "quantity",
    }
```

- [x] **Step 2: Add tests for the read-only filter and the forbidden-argument guard**

Append to `tests/test_agent_tools.py`:

```python
def test_the_read_only_surface_is_the_six_low_risk_tools():
    # Task 2 wires only the tools that cannot change anything. add_to_cart
    # and remove_from_cart are Medium; cancel_order is High and needs an
    # approval this task does not build.
    assert READ_ONLY_TOOLS == {
        "search_products",
        "get_product",
        "check_inventory",
        "get_orders",
        "get_order",
        "get_cart",
    }
    assert READ_ONLY_TOOLS < KNOWN_TOOLS


def test_translate_can_narrow_to_the_read_only_surface():
    translated = translate_tools(all_nine(), only=READ_ONLY_TOOLS)

    assert {t["function"]["name"] for t in translated} == READ_ONLY_TOOLS


def test_a_user_id_argument_is_refused():
    # Identity comes from the bearer token, resolved by the API's own
    # whoami. A user id in the arguments is the model asserting who the
    # caller is, which is the one thing it must never do -- and a model
    # can hallucinate a key that was never in the schema.
    for key in FORBIDDEN_ARGUMENTS:
        with pytest.raises(ForbiddenArgumentError):
            reject_forbidden_arguments("get_orders", {key: "u1"})


def test_ordinary_arguments_pass_the_guard():
    reject_forbidden_arguments("get_order", {"order_id": "o1"})
    reject_forbidden_arguments("search_products", {"query": "shoes", "limit": 5})


def test_the_guard_names_the_offending_key():
    with pytest.raises(ForbiddenArgumentError) as caught:
        reject_forbidden_arguments("get_orders", {"user_id": "u1", "limit": 5})

    assert "user_id" in str(caught.value)
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -q`
Expected: collection error — `ImportError: cannot import name 'FORBIDDEN_ARGUMENTS'`.

- [x] **Step 4: Update `agent/tools.py`**

Replace the module docstring's second paragraph and the Claude-specific import/function. Find:
```python
from typing import Any

from anthropic.types import ToolParam
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool

import config
```

Replace with:
```python
from typing import Any, Iterable

from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool
from openai.types.chat import ChatCompletionToolParam

import config
```

Find:
```python
class UnknownToolError(Exception):
    """The MCP server's tool surface is not the one this agent expects."""


def to_claude_tool(tool: Tool) -> ToolParam:
    """One MCP tool as a Claude tool definition."""
    schema = _without_injected_arguments(tool.name, tool.inputSchema)

    return {
        "name": tool.name,
        # Claude tolerates a missing description; an explicit empty string
        # is clearer than None travelling through the request builder.
        "description": tool.description or "",
        "input_schema": schema,
    }


def translate_tools(tools: list[Tool]) -> list[ToolParam]:
    """Every advertised tool, or an error naming what did not match."""
    advertised = {tool.name for tool in tools}

    unknown = advertised - KNOWN_TOOLS
    if unknown:
        raise UnknownToolError(
            f"MCP server advertises tools this agent does not know: {sorted(unknown)}"
        )

    missing = KNOWN_TOOLS - advertised
    if missing:
        raise UnknownToolError(
            f"MCP server is not advertising expected tools: {sorted(missing)}"
        )

    return [to_claude_tool(tool) for tool in tools]
```

Replace with:
```python
# The tools that cannot change anything. Task 2 of the M4 plan wires only
# these; Medium-risk cart writes and the High-risk cancellation arrive with
# the approval machinery that guards them.
READ_ONLY_TOOLS = frozenset(
    {
        "search_products",
        "get_product",
        "check_inventory",
        "get_orders",
        "get_order",
        "get_cart",
    }
)

# Identity is never an argument. It is resolved from the bearer token by
# the API's own whoami, and a model supplying one of these is the model
# asserting who the caller is. None of the nine schemas contain these --
# the guard exists because a model can invent a key that was never offered.
FORBIDDEN_ARGUMENTS = frozenset(
    {"user_id", "userId", "customer_id", "customerId", "email", "user", "customer"}
)


class UnknownToolError(Exception):
    """The MCP server's tool surface is not the one this agent expects."""


class ForbiddenArgumentError(Exception):
    """A tool call carried an argument the model has no business supplying."""


def to_openai_tool(tool: Tool) -> ChatCompletionToolParam:
    """One MCP tool as an OpenAI tool definition.

    Verified against a live call: the MCP schema is accepted as
    `parameters` unchanged. This re-nests; it does not rewrite.
    """
    schema = _without_injected_arguments(tool.name, tool.inputSchema)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            # An explicit empty string is clearer than None travelling
            # through the request builder.
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def translate_tools(
    tools: list[Tool], only: Iterable[str] | None = None
) -> list[ChatCompletionToolParam]:
    """Every advertised tool, or an error naming what did not match.

    `only` narrows the surface handed to the model without weakening the
    check: the full nine must still be advertised, so a tool going missing
    is still caught even when this turn does not use it.
    """
    advertised = {tool.name for tool in tools}

    unknown = advertised - KNOWN_TOOLS
    if unknown:
        raise UnknownToolError(
            f"MCP server advertises tools this agent does not know: {sorted(unknown)}"
        )

    missing = KNOWN_TOOLS - advertised
    if missing:
        raise UnknownToolError(
            f"MCP server is not advertising expected tools: {sorted(missing)}"
        )

    wanted = set(only) if only is not None else KNOWN_TOOLS

    return [to_openai_tool(tool) for tool in tools if tool.name in wanted]


def reject_forbidden_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Raise if a tool call carries an identity argument. Called before every
    execution, not just the risky ones -- a read tool scoped to the wrong
    customer is the leak, not the write."""
    offending = sorted(set(arguments) & FORBIDDEN_ARGUMENTS)
    if offending:
        raise ForbiddenArgumentError(
            f"{name} was called with identity arguments the model may not supply: "
            f"{offending}"
        )
```

Then update the trailing helper. Find:
```python
async def list_claude_tools(token: str, url: str | None = None) -> list[ToolParam]:
    """Connect, list, translate. The agent's whole view of what it can do."""
    from fastmcp import Client

    transport = build_transport(url or config.MCP_SERVER_URL, token)

    async with Client(transport) as client:
        return translate_tools(await client.list_tools())
```

Replace with:
```python
async def list_openai_tools(
    token: str, url: str | None = None, only: Iterable[str] | None = None
) -> list[ChatCompletionToolParam]:
    """Connect, list, translate. The agent's whole view of what it can do."""
    from fastmcp import Client

    transport = build_transport(url or config.MCP_SERVER_URL, token)

    async with Client(transport) as client:
        return translate_tools(await client.list_tools(), only=only)
```

Finally update the module docstring. Find:
```
Two shapes for the same nine capabilities. MCP advertises `inputSchema`;
Claude's tool definition wants `input_schema`. The rename is most of the
work, and the two things that are NOT a rename are the point of this
module:
```

Replace with:
```
Two shapes for the same nine capabilities. MCP advertises `inputSchema`;
OpenAI's tool definition wants it nested as `function.parameters`. The
re-nesting is most of the work, and the things that are NOT a re-nesting
are the point of this module:
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -q`
Expected: 16 passed (11 retargeted + 5 new).

- [x] **Step 6: Full suite, then commit**

Run: `.venv/Scripts/python -m pytest -q`
Expected: 134 passed.

```bash
git add requirements.txt config.py agent/tools.py tests/test_agent_tools.py
git commit -m "feat: retarget the tool surface from Claude to OpenAI

The project has OpenAI credits and no Anthropic key, so M4 runs on
gpt-4.1. LangGraph was picked partly for not being tied to a vendor and
that holds - the MCP server, the nine tools, the approval architecture
and the risk tiers are untouched.

Adds two things the loop needs: a read-only surface (the six tools that
cannot change anything, which is all Task 2 wires) and a guard refusing
identity arguments, since identity comes from the bearer token and a
model can invent a key that was never in the schema.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The loop

**Files:** `tests/test_agent_loop.py`, `agent/loop.py`

- [x] **Step 1: Write the failing tests**

Create `tests/test_agent_loop.py`:

```python
# tests/test_agent_loop.py
#
# One turn: utterance in, tool calls out, answer back. The model is stubbed
# -- what is under test is the loop's own behaviour (does it execute what
# the model asked for, feed results back in the shape the API wants, stop
# when the model stops, and refuse an identity argument), not the model's
# judgement. That is what the eval harness is for.

import json

import pytest

from agent.loop import build_graph, run_turn
from agent.tools import ForbiddenArgumentError


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        dumped = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            dumped = {k: v for k, v in dumped.items() if v is not None}
        return dumped


def scripted_model(*turns):
    """A model that returns each scripted turn in order."""
    remaining = list(turns)

    async def call(messages, tools):
        return remaining.pop(0)

    return call


def recording_executor(results):
    """An MCP stand-in that records what it was asked to run."""
    calls = []

    async def execute(name, arguments):
        calls.append((name, arguments))
        return results.get(name, {"ok": True})

    execute.calls = calls
    return execute


async def test_a_turn_with_no_tool_call_answers_directly():
    state = await run_turn(
        "hello",
        model_call=scripted_model(FakeMessage(content="Hi there.")),
        execute_tool=recording_executor({}),
    )

    assert state["answer"] == "Hi there."


async def test_a_tool_call_is_executed_and_its_result_fed_back():
    executor = recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]})

    state = await run_turn(
        "what did I order recently?",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=executor,
    )

    assert executor.calls == [("get_orders", {"limit": 3})]
    assert state["answer"] == "You ordered ORD-1."

    # The result must go back as a tool message keyed to the call id.
    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "ORD-1" in tool_messages[0]["content"]


async def test_arguments_are_parsed_as_json_not_string_matched():
    # The API returns arguments as a JSON string; escaping varies.
    executor = recording_executor({})

    await run_turn(
        "find shoes",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "search_products", '{"query":"shoes","limit":5}')
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
    )

    name, arguments = executor.calls[0]
    assert arguments == {"query": "shoes", "limit": 5}
    assert isinstance(arguments, dict)


async def test_several_tool_calls_in_one_turn_all_execute():
    executor = recording_executor({})

    await run_turn(
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
        execute_tool=executor,
    )

    assert [c[1]["product_id"] for c in executor.calls] == ["p1", "p2"]


async def test_no_user_id_is_ever_sent_as_an_argument():
    # The MUST PROVE of this task. A hallucinated identity argument is
    # refused before it reaches the MCP server.
    executor = recording_executor({})

    with pytest.raises(ForbiddenArgumentError):
        await run_turn(
            "what did I order?",
            model_call=scripted_model(
                FakeMessage(
                    tool_calls=[
                        FakeToolCall("call_1", "get_orders", '{"user_id":"u1","limit":3}')
                    ]
                )
            ),
            execute_tool=executor,
        )

    assert executor.calls == []


async def test_the_graph_compiles():
    # It must compile without a checkpointer: this turn never pauses.
    assert build_graph() is not None
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent_loop.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'agent.loop'`.

- [x] **Step 3: Write `agent/loop.py`**

```python
"""One turn of the agent: utterance in, tool calls out, answer back.

LangGraph owns the graph and the state; the OpenAI SDK is called directly
from a plain node, with no model-wrapper library in between, so request
parameters stay under this project's control. Two nodes and a router:

    call_model -> (tool calls? -> execute_tools -> call_model) : END

`model_call` and `execute_tool` are injected rather than imported, because
what is worth testing here is the loop's own behaviour -- does it execute
what was asked, feed results back in the shape the API wants, stop when
the model stops, and refuse an identity argument -- not the model's
judgement, which is the eval harness's job.

Read-only tools only, per M4 Task 2. The cart writes and the cancellation
arrive with the approval machinery that guards them.
"""

import json
import operator
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agent.tools import (
    READ_ONLY_TOOLS,
    build_transport,
    list_openai_tools,
    reject_forbidden_arguments,
)

ModelCall = Callable[[list[dict], list[dict]], Awaitable[Any]]
ToolExecutor = Callable[[str, dict], Awaitable[Any]]


class TurnState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tools: list[dict]
    answer: str | None


def _tool_calls_of(message: dict) -> list[dict]:
    return message.get("tool_calls") or []


def build_graph(
    model_call: ModelCall | None = None,
    execute_tool: ToolExecutor | None = None,
):
    """The turn as a graph. Compiles without a checkpointer: nothing pauses."""

    async def call_model(state: TurnState) -> dict:
        message = await model_call(state["messages"], state.get("tools", []))
        dumped = message.model_dump(exclude_none=True)
        # content is None on a tool turn; the API rejects a null content
        # field on the way back in.
        dumped.setdefault("role", "assistant")

        return {"messages": [dumped], "answer": message.content}

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

    def route(state: TurnState) -> str:
        return "tools" if _tool_calls_of(state["messages"][-1]) else END

    graph = StateGraph(TurnState)
    graph.add_node("model", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    return graph.compile()


async def run_turn(
    utterance: str,
    *,
    model_call: ModelCall,
    execute_tool: ToolExecutor,
    tools: list[dict] | None = None,
) -> TurnState:
    """One turn, start to finish."""
    app = build_graph(model_call, execute_tool)

    return await app.ainvoke(
        {
            "messages": [{"role": "user", "content": utterance}],
            "tools": tools or [],
            "answer": None,
        }
    )


def openai_model_call(model: str | None = None) -> ModelCall:
    """The real model call. Kept separate so the loop stays testable."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    chosen = model or config.OPENAI_MODEL

    async def call(messages: list[dict], tools: list[dict]):
        response = await client.chat.completions.create(
            model=chosen,
            max_completion_tokens=1024,
            messages=messages,
            tools=tools or None,
        )
        return response.choices[0].message

    return call


def mcp_tool_executor(token: str, url: str | None = None) -> ToolExecutor:
    """The real tool execution, one MCP session per batch of calls."""
    from fastmcp import Client

    async def execute(name: str, arguments: dict) -> Any:
        transport = build_transport(url or config.MCP_SERVER_URL, token)

        async with Client(transport) as client:
            result = await client.call_tool(name, arguments)
            return json.loads(result.content[0].text) if result.content else None

    return execute


async def answer(utterance: str, token: str, *, model: str | None = None) -> TurnState:
    """The whole thing wired to the real model and the real MCP server."""
    tools = await list_openai_tools(token, only=READ_ONLY_TOOLS)

    return await run_turn(
        utterance,
        model_call=openai_model_call(model),
        execute_tool=mcp_tool_executor(token),
        tools=tools,
    )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_agent_loop.py -q`
Expected: 6 passed.

- [x] **Step 5: Full suite, then commit**

Run: `.venv/Scripts/python -m pytest -q`
Expected: 140 passed.

```bash
git add agent/loop.py tests/test_agent_loop.py
git commit -m "feat: the agent's single turn, read-only tools, on LangGraph

Two nodes and a router: call the model, execute whatever it asked for,
feed the results back, stop when it stops. LangGraph owns the graph and
the state; the OpenAI SDK is called directly from a plain node so
request parameters stay under our control, and the model and executor
are injected so the loop's own behaviour is what the tests exercise.

Identity arguments are refused before execution - the MUST PROVE of this
task. Identity comes from the bearer token, and a model can invent a key
that was never in the schema.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The live gate — workflow 1, end to end

**Files:** none modified. This is the task's real proof.

- [x] **Step 1: Run workflow 1 against the live MCP server and a real model**

The demo customer's credentials are the seeded fixture in the storefront's
`signin-form.tsx` (`customer@example.com` / `demo1234`), and a bearer is
minted the same way `scripts/sweep.py` does it.

Run:
```bash
.venv/Scripts/python -c "
import asyncio, httpx, json
from agent.loop import answer

API = 'https://web-production-bb55d.up.railway.app'

async def main():
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(f'{API}/api/v1/auth/token',
            json={'email':'customer@example.com','password':'demo1234','ttlSeconds':900})
        r.raise_for_status()
        token = r.json()['data']['token']

    state = await answer('What did I order recently?', token)
    print('ANSWER:', state['answer'])
    print()
    for m in state['messages']:
        role = m.get('role')
        if role == 'assistant' and m.get('tool_calls'):
            print('  called:', [c['function']['name'] for c in m['tool_calls']])
        elif role == 'tool':
            print('  result:', m['content'][:120])

asyncio.run(main())
"
```

Expected: the agent calls `get_orders` (possibly then `get_order`), and the
answer describes the demo customer's actual orders — the three seeded ones,
all `CANCELLED` since the Phase 1 verification work consumed them.

- [x] **Step 2: Confirm the two MUST PROVEs**

- Workflow 1 ran end to end against the live MCP server: the answer names
  real order data that could only have come from the API.
- No user id was sent: the guard raises before execution, and the printed
  `called:` lines show only schema arguments.

If the model answers without calling a tool, that is a failure of this
gate — it means the tools were not passed or the surface was empty. Check
`tools` is non-empty before blaming the model.

No commit — nothing under version control changed.

---

### Task 5: Amend `PLAN_M4_AGENT.txt` for the provider switch

**Files:** `docs/PLAN_M4_AGENT.txt`

§3 is entirely Claude-specific and would be actively wrong left as-is.

- [x] **Step 1: Rewrite §3's header block**

Find:
```
    Model            claude-opus-5
    Thinking         adaptive - {"type": "adaptive"}. It is on by default
                     on this model; do not disable it. If cost needs
                     trimming, lower effort instead.
    Effort           output_config: {"effort": "high"} to begin. Measure
                     before changing it; "high" is often the sweet spot.
    Streaming        Required for anything with a long output, and the chat
                     UI wants tokens as they arrive.
    max_tokens       Generous. A truncated turn costs a retry.
```

Replace with:
```
    Provider         OpenAI. Chosen 2026-09-02 because this project has
                     OpenAI credits and no Anthropic key. LangGraph was
                     picked partly for not being tied to one vendor, and
                     the MCP server, the nine tools, the approval
                     architecture and the risk tiers are all unchanged by
                     the switch.
    Model            gpt-4.1, via OPENAI_MODEL. One line to change.
    API              Chat Completions. The tool loop is the well-trodden
                     path there, and the SDK is called directly from a
                     LangGraph node rather than through a model-wrapper
                     library, so request parameters stay ours.
    Streaming        Not yet. The chat UI will want tokens as they arrive;
                     add it when the storefront's event stream exists.
    max_completion_tokens
                     Generous. A truncated turn costs a retry. Note the
                     parameter name: max_tokens is the older spelling.
```

- [x] **Step 2: Replace the five Claude API facts**

Find:
```
FIVE API FACTS WORTH PINNING DOWN NOW, because a stale assumption here is
expensive later:

  * budget_tokens IS REJECTED on this model. The fixed thinking-budget idea
    is gone; adaptive thinking plus effort replaces it.
  * ASSISTANT PREFILL IS REJECTED. Shape output with structured outputs or
    the system prompt, not by pre-writing the reply.
  * PARSE TOOL INPUTS AS JSON. Never string-match the serialised input;
    escaping varies.
  * RETURN ALL tool_result BLOCKS IN ONE user MESSAGE. Splitting them
    across messages silently teaches the model to stop calling tools in
    parallel.
  * ALWAYS CHECK stop_reason BEFORE READING CONTENT. A safety refusal
    arrives as a normal 200 response.

Verify these against the SDK at implementation time rather than trusting
this list - that is the habit that has paid off twice already.
```

Replace with:
```
FOUR API FACTS, EACH VERIFIED BY A LIVE gpt-4.1 CALL ON 2026-09-02 rather
than recalled:

  * PARSE TOOL ARGUMENTS AS JSON. tool_calls[i].function.arguments is a
    STRING, not a dict, and the escaping varies. Never string-match it.
  * THE MCP SCHEMA PASSES THROUGH UNCHANGED as function.parameters,
    anyOf/default and all. Translation is a re-nesting, not a rewrite.
  * finish_reason == "tool_calls" IS THE LOOP SIGNAL, and message.content
    is None on that turn. Check the reason before reading content.
  * ONE tool MESSAGE PER CALL, each keyed by tool_call_id, appended after
    the assistant turn that requested them.

Verify against the SDK at implementation time rather than trusting this
list - that habit has now paid off three times.
```

- [x] **Step 3: Note the switch in §1 Decision A**

Find:
```
(A) WHICH HARNESS RUNS THE LOOP?
```

Replace with:
```
(A) WHICH HARNESS RUNS THE LOOP?    [SETTLED: LangGraph, on OpenAI]

    Settled 2026-09-02: LangGraph, with the OpenAI SDK called directly
    from a plain node. The options below are kept for the reasoning; note
    that options 2 and 3 named Anthropic surfaces and no longer apply.
```

- [x] **Step 4: Verify no stale Claude references remain in the model section**

Run: `grep -n -i "claude\|anthropic\|budget_tokens\|adaptive" docs/PLAN_M4_AGENT.txt`
Expected: only the historical mentions inside §1 Decision A's preserved
option list. If §3 still names Claude anywhere, fix it.

- [x] **Step 5: Commit**

```bash
git add docs/PLAN_M4_AGENT.txt
git commit -m "docs: retarget the M4 agent plan from Claude to OpenAI

Section 3 was entirely Claude-specific and would have been actively
wrong. Replaces it with the OpenAI settings actually in use, and swaps
the five recalled Claude API facts for four verified against a live
gpt-4.1 call. Decision A is marked settled.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Push

- [x] **Step 1: Review pending commits**

Run: `git log --oneline origin/main..HEAD`
Expected: the three commits from Tasks 2, 3, and 5.

- [x] **Step 2: Push, after user confirmation**

```bash
git push origin main
```
