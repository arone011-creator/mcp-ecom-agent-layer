# M4 Task 1 — MCP Client and Tool Translation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect to the MCP server, list its tools, and translate the schemas into Claude tool definitions — `PLAN_M4_AGENT.txt` Task 1. MUST PROVE: exactly nine tools translate; an unknown tool is refused rather than passed through; the bearer token travels on every call.

**Architecture:** A new `agent/` package (this repo hosts both the MCP server and, from M4, the agent service — see `PLAN_M4_AGENT.txt` §2). `agent/tools.py` holds two pure functions (`to_claude_tool`, `translate_tools`) plus one thin async wrapper that opens an MCP session. Pure functions carry the test weight; a live check against production is the real gate, matching this repo's established habit.

**Tech Stack:** Python 3.11+, `fastmcp` (already a dependency), `anthropic` (new — for `ToolParam`, the SDK's own tool-definition type). No LangGraph: this task never touches the agent loop.

**Written against the real library, at the time** — as `PLAN_M4_AGENT.txt` requires. Everything below was verified by probing the live MCP server on 2026-09-02, not recalled:

- `client.list_tools()` returns `list[mcp.types.Tool]`, a Pydantic model.
- Its fields are `.name` (str), `.description` (str | None), `.inputSchema` (dict) — **camelCase**. Claude's tool definition needs **snake_case** `input_schema`. The rename is the translation.
- All nine schemas are `{"properties": {...}, "type": "object"}`, some with `required`. `get_cart` has empty properties and no `required` key at all.
- Optional parameters use Pydantic's `anyOf: [{type}, {type: "null"}]` + `default` shape.
- `StreamableHttpTransport` stores its headers on `self.headers` (confirmed in the installed source), which is the seam the bearer-token test uses.

One thing below is *not* verified, because `anthropic` is not installed yet:
the exact import path for the SDK's tool-definition type (`from
anthropic.types import ToolParam`). It is written from the SDK's documented
Python naming, and the first test run will say if it is wrong — fix it from
the error rather than researching ahead of the failure. `ToolParam` is a
TypedDict, so it is a plain `dict` at runtime either way and the assertions
hold regardless.

**Executed 2026-09-02. Outcome:**

- The flagged uncertainty resolved in our favour: `from anthropic.types
  import ToolParam` is correct. `anthropic 1.3.0` installed (it brings
  `httpx2` alongside the existing `httpx`; both coexist, existing suite
  unaffected).
- One environment deviation: this venv's `pip.exe` shim is broken (exits 1,
  prints nothing, including for `pip --version`). `.venv/Scripts/python -m
  pip` works normally and was used instead. Unrelated to this task, but
  worth knowing before the next install.
- All 11 tests green; full suite 129 passed (118 + 11, as predicted).
- Live gate passed: 9 tools translated from production, `cancel_order`
  advertising `order_id` only.
- The known `test_the_token_does_not_leak_what_it_authorises` flake fired
  once mid-task and passed 3/3 on rerun — still unrelated, still tracked
  separately.

---

## File Structure

| File | Change |
|---|---|
| `agent/__init__.py` | Create — empty, marks the package |
| `agent/tools.py` | Create — translation + session helper |
| `tests/test_agent_tools.py` | Create — the tests |
| `config.py` | Modify — add `MCP_SERVER_URL`; fix one now-stale comment |
| `requirements.txt` | Modify — add `anthropic` |

## Two design decisions, both from the probe

**1. `approval_token` is stripped from `cancel_order`'s advertised schema.** The live schema exposes it as a model-fillable string. The architecture is explicit that the storefront injects that token after a human clicks and the agent never produces one (`PLAN_M4_AGENT.txt` §1 Decision C, §2, Task 5). A forged token would be rejected by the HMAC check server-side, so this is not an exploitable hole — but advertising the field invites the model to fill it, and Task 5's "a resumed call uses the arguments the human saw" is easier to hold if the model was never offered the field. Code injects it at call time; the model never sees it.

**2. `strict: true` is out of scope.** Claude's strict mode requires `additionalProperties: false` and a `required` array on every schema. The MCP schemas have neither, and `get_cart` has no properties at all. Adding strict means rewriting nine schemas for a guarantee nothing yet needs. Revisit if the model starts producing invalid arguments.

---

### Task 1: Add the dependency and config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`

- [x] **Step 1: Add `anthropic` to `requirements.txt`**

Find:
```
fastmcp>=2.3,<3
httpx>=0.27,<0.29
pydantic>=2.7,<3
```

Replace with:
```
anthropic>=1,<2
fastmcp>=2.3,<3
httpx>=0.27,<0.29
pydantic>=2.7,<3
```

- [x] **Step 2: Install it**

Run: `.venv/Scripts/pip install -r requirements-dev.txt`
Expected: `anthropic` installs. Confirm with
`.venv/Scripts/python -c "import anthropic; print(anthropic.__version__)"` — prints a 1.x version.

If a 1.x does not exist yet and pip resolves nothing, pin whatever major version pip does offer and note the deviation; do not silently widen the range.

- [x] **Step 3: Add `MCP_SERVER_URL` to `config.py`, and fix the stale comment**

Find:
```python
# The storefront's public domain by default. Switch to the private-network
# address once it is verified -- see Task 10 of the M3 plan.
ECOMMERCE_API_BASE_URL = os.environ.get(
    "ECOMMERCE_API_BASE_URL", "https://web-production-bb55d.up.railway.app"
)
```

Replace with:
```python
# The storefront's public domain by default. The deployed service overrides
# this with the Railway private-network address (verified 2026-09-02); the
# public default is what a local run and the test suite get.
ECOMMERCE_API_BASE_URL = os.environ.get(
    "ECOMMERCE_API_BASE_URL", "https://web-production-bb55d.up.railway.app"
)

# Where the agent finds the MCP server. Read by agent/, not by the server
# itself -- the server does not call itself.
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL", "https://mcp-production-e344.up.railway.app/mcp"
)
```

- [x] **Step 4: Commit**

```bash
git add requirements.txt config.py
git commit -m "chore: add the anthropic SDK and the agent's MCP server URL

Task 1 of M4 needs the SDK's ToolParam type and somewhere to read the
MCP endpoint from. Also refreshes a config comment that went stale when
the private-network switch actually happened.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The translation, test-first

**Files:**
- Create: `tests/test_agent_tools.py`
- Create: `agent/__init__.py`, `agent/tools.py`

- [x] **Step 1: Write the failing tests**

Create `tests/test_agent_tools.py`:

```python
# tests/test_agent_tools.py
#
# The MCP server advertises tools; Claude needs them in its own shape. The
# rename is real (inputSchema -> input_schema), and two things must not pass
# through unexamined: a tool this agent does not know about, and
# cancel_order's approval_token, which only the storefront may supply.

import pytest
from mcp.types import Tool

from agent.tools import (
    KNOWN_TOOLS,
    UnknownToolError,
    build_transport,
    to_claude_tool,
    translate_tools,
)


def mcp_tool(name: str, schema: dict | None = None, description: str | None = "d") -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=schema if schema is not None else {"properties": {}, "type": "object"},
    )


def all_nine() -> list[Tool]:
    return [mcp_tool(name) for name in sorted(KNOWN_TOOLS)]


def test_the_nine_known_tools_are_the_nine_the_server_advertises():
    # A tool surface that silently shrinks is how a capability disappears
    # without anyone noticing.
    assert len(KNOWN_TOOLS) == 9


def test_translation_renames_the_schema_key_to_claude_spelling():
    schema = {"properties": {"product_id": {"type": "string"}}, "required": ["product_id"], "type": "object"}

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


def test_all_nine_translate():
    translated = translate_tools(all_nine())

    assert len(translated) == 9
    assert {t["name"] for t in translated} == KNOWN_TOOLS


def test_an_unknown_tool_is_refused_rather_than_passed_through():
    # The MCP server is a separate deployment. If it ever advertises
    # something this agent was not built for, that is a refusal, not a
    # capability the model silently gains.
    tools = all_nine() + [mcp_tool("wire_money")]

    with pytest.raises(UnknownToolError) as caught:
        translate_tools(tools)

    assert "wire_money" in str(caught.value)


def test_a_missing_tool_is_refused_too():
    tools = [t for t in all_nine() if t.name != "cancel_order"]

    with pytest.raises(UnknownToolError):
        translate_tools(tools)


def test_cancel_order_does_not_advertise_approval_token_to_the_model():
    # The storefront injects the approval token after a human clicks. A
    # model that can see the field is a model that can invent a value for
    # it; the server would reject a forged one, but the field has no
    # business being in the model's schema at all.
    schema = {
        "properties": {
            "order_id": {"type": "string"},
            "approval_token": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["order_id"],
        "type": "object",
    }

    translated = to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" not in translated["input_schema"]["properties"]
    assert "order_id" in translated["input_schema"]["properties"]
    assert translated["input_schema"]["required"] == ["order_id"]


def test_stripping_does_not_mutate_the_caller_s_schema():
    schema = {
        "properties": {"order_id": {"type": "string"}, "approval_token": {"type": "string"}},
        "type": "object",
    }

    to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" in schema["properties"]


def test_other_tools_keep_every_property():
    schema = {
        "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
        "required": ["product_id", "quantity"],
        "type": "object",
    }

    translated = to_claude_tool(mcp_tool("add_to_cart", schema))

    assert set(translated["input_schema"]["properties"]) == {"product_id", "quantity"}


def test_the_bearer_token_travels_on_every_call():
    transport = build_transport("https://mcp.test/mcp", "tok-123")

    assert transport.headers["authorization"] == "Bearer tok-123"


def test_a_blank_token_is_refused_rather_than_sent_empty():
    # An empty bearer is worse than no bearer: it looks like a credential
    # and authenticates nobody.
    with pytest.raises(ValueError):
        build_transport("https://mcp.test/mcp", "")
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'agent'`.

- [x] **Step 3: Write the implementation**

Create `agent/__init__.py` (empty file).

Create `agent/tools.py`:

```python
"""The MCP tool surface, translated into what Claude expects.

Two shapes for the same nine capabilities. MCP advertises `inputSchema`;
Claude's tool definition wants `input_schema`. The rename is most of the
work, and the two things that are NOT a rename are the point of this
module:

  - an unknown tool is refused. The MCP server is a separate deployment
    that could grow a tool this agent was never built for, and a
    capability the model gains silently is one nobody reviewed.

  - cancel_order's approval_token is stripped before the model ever sees
    it. The storefront mints that token after a human clicks and code
    injects it at call time; a field the model cannot see is a field it
    cannot invent a value for.
"""

from typing import Any

from anthropic.types import ToolParam
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool

import config

# The surface this agent was built against. Asserted, not discovered: a
# tool list that silently shrinks is how a capability disappears without
# anyone noticing, and one that silently grows is an unreviewed capability.
KNOWN_TOOLS = frozenset(
    {
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
)

# Arguments supplied by code, never by the model. Stripped from the schema
# the model is shown, and injected at call time.
INJECTED_ARGUMENTS: dict[str, frozenset[str]] = {
    "cancel_order": frozenset({"approval_token"}),
}


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


def build_transport(url: str, token: str) -> StreamableHttpTransport:
    """The transport every MCP call rides on, carrying this caller's token.

    One transport per caller, never shared: the token IS the identity, and
    a transport held between customers is an ambient identity by another
    name -- the same rule clients/ecommerce_api.py follows.
    """
    if not token.strip():
        raise ValueError("A bearer token is required")

    return StreamableHttpTransport(url, headers={"authorization": f"Bearer {token}"})


def _without_injected_arguments(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    injected = INJECTED_ARGUMENTS.get(name)
    if not injected:
        return schema

    # Copied rather than edited: the caller's Tool object is not ours to
    # mutate, and a shared nested dict would leak the edit anyway.
    copy = dict(schema)
    copy["properties"] = {
        key: value
        for key, value in schema.get("properties", {}).items()
        if key not in injected
    }
    if "required" in schema:
        copy["required"] = [key for key in schema["required"] if key not in injected]

    return copy


async def list_claude_tools(token: str, url: str | None = None) -> list[ToolParam]:
    """Connect, list, translate. The agent's whole view of what it can do."""
    from fastmcp import Client

    transport = build_transport(url or config.MCP_SERVER_URL, token)

    async with Client(transport) as client:
        return translate_tools(await client.list_tools())
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_agent_tools.py -v`
Expected: 11 passed.

- [x] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all green (118 existing + 11 new = 129). If `test_the_token_does_not_leak_what_it_authorises` fails, that is the known pre-existing flake — re-run once to confirm, it is unrelated.

- [x] **Step 6: Commit**

```bash
git add agent/ tests/test_agent_tools.py
git commit -m "feat: translate the MCP tool surface into Claude tool definitions

Renames inputSchema to input_schema, refuses a tool surface that does
not match the nine this agent was built against, and strips
cancel_order's approval_token before the model sees it - the storefront
mints that token after a human clicks, so a field the model cannot see
is a field it cannot invent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Prove it against the live server

**Files:** none modified — this is the real gate, matching how every M3 claim was verified.

- [x] **Step 1: Translate the production tool surface**

Run:
```bash
.venv/Scripts/python -c "
import asyncio, json
from agent.tools import list_claude_tools
tools = asyncio.run(list_claude_tools('placeholder'))
print(len(tools), 'tools')
for t in tools:
    print(' -', t['name'], sorted(t['input_schema'].get('properties', {})))
"
```

Expected: `9 tools`, and `cancel_order` listing `['order_id']` only — no `approval_token`.

- [x] **Step 2: If the count or the surface differs, stop**

A mismatch means the deployed MCP server is not the one this agent was built against. Report it; do not widen `KNOWN_TOOLS` to make the check pass — that check exists precisely to catch this.

No commit — nothing under version control changed.
