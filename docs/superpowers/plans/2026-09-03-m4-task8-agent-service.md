# M4 Task 8 - The Agent Service, and Deploying It

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent an HTTP surface the storefront can call, and run it as its own
Railway service.

**Architecture:** A small Starlette app (`agent_server.py`) beside the existing `server.py`,
deployed from the same repository as a third Railway service with a different start command.
`POST /turn` streams the frozen v1 events as SSE; a high-risk call pauses the stream and is
resumed by `POST /turn/{id}/decision`.

**Tech Stack:** Python 3.12, Starlette (already present via fastmcp), uvicorn, Railway.

---

## Task 8 says "deploy". There is nothing to deploy yet.

`server.py` serves the MCP tools. `agent/` is a **library** — its entry point is
`await answer(utterance, token)`, a Python function. No process serves it, no port, no route.
So "a third Railway service, or an extension of this one" is a decision about something that
does not exist.

Building that surface is most of this task, and its shape is consumed by the storefront's
Task 3 (the bridge route), so it needs agreeing in one place — the same problem the event
contract and the metrics file already had, handled the same way.

## Decision: a third service, not an extension

The plan says to decide on memory. Measured, over three hours of production:

```
mcp   MEMORY_USAGE_GB  avg 0.101  max 0.187   limit 8
      CPU_USAGE        avg 0.012
```

So memory says the agent would fit ten times over. **Memory is not the argument.** Two
better ones:

1. **Secrets.** The agent needs `OPENAI_API_KEY`. The MCP server does not, and today holds
   only `ECOMMERCE_API_BASE_URL`, `MCP_APPROVAL_SECRET`, `MCP_APPROVAL_TTL_SECONDS`.
   Co-locating them puts the model key inside the process that any customer's bearer token
   can reach.

2. **The approval boundary should be a process boundary.** `approvals.py` validates in the
   MCP server; the agent must never mint. Today that is asserted by a test scanning the
   `agent` package for the route. In one process it is one import away from being false. Two
   processes make it structural, which is what the rest of this design has consistently
   preferred.

Same repository, same build, different start command — Railway already does this.

## The HTTP contract

`POST /turn`
- Headers: `Authorization: Bearer <customer token>`, `X-Agent-Key: <shared secret>`
- Body: `{"utterance": "..."}`
- Returns: `text/event-stream`

Two SSE event channels, and the distinction is load-bearing:

| SSE `event:` | Contents | Who consumes it |
|---|---|---|
| `assistant` | one v1 event from `contracts/assistant-events.v1.json` | forwarded to the browser |
| `control` | `{"turn_id": "...", "session_id": "..."}` | the bridge route only, **never forwarded** |

The `control` frame is how the session id reaches the storefront without entering the event
contract. Task 5 established that the storefront must mint against the **agent's** MCP
session id, and Task 4 established that the id has no business in a stream that reaches a
browser. SSE's own `event:` field separates them at the transport, so the frozen contract
does not change and the bridge route simply does not forward `control`.

`POST /turn/{turn_id}/decision`
- Body: `{"approved": true, "token": "..."}` or `{"approved": false}`
- Resumes the paused turn. The `/turn` stream continues and completes.

`GET /health` — unauthenticated, for the readiness check.

## Why a shared secret, and why not private-only yet

An unauthenticated endpoint that calls OpenAI is a bill anyone can run up. `X-Agent-Key`
(`AGENT_SERVICE_KEY`) is checked before a single token is spent.

Private-network-only would be structurally better and is where this should end up — the
storefront calls it server-side, so it never needs a public address. It is **not** done in
this task because Task 8's MUST PROVE requires verifying the new deployment from outside,
and a private-only service cannot be verified until the bridge route exists. Recorded as a
follow-up for after storefront Task 3, when the caller exists.

## The state this holds, and what that costs

A paused turn holds a LangGraph thread (`InMemorySaver`) **and** an open MCP session, across
two HTTP requests. Both live in process memory, so:

- **one replica only.** The same limit `approvals.py` already documents for its spent-nonce
  set, and `InMemorySaver` for checkpoints. Recorded, not solved.
- **a turn abandoned mid-pause leaks a session** until its deadline. The 300-second approval
  timeout from Task 5 is what bounds this, and it is now bounding a real resource rather than
  a Python task.

---

## File Structure

- **Create** `agent_server.py` — the HTTP surface. At the repo root beside `server.py`,
  because they are siblings: two processes, one library.
- **Create** `tests/test_agent_server.py`.
- **Modify** `config.py` — `AGENT_SERVICE_KEY`.
- **Modify** `requirements.txt` — uvicorn, if not already pulled in.
- **Modify** `docs/PLAN_M4_AGENT.txt`, `../mcp-ecom-web-app/docs/PLAN_M4_STOREFRONT.txt`.

---

### Task 8.1: The turn registry

**Files:** Create `agent_server.py`, `tests/test_agent_server.py`

A paused turn must be findable by the decision request. That is a small amount of state with
a large number of ways to get wrong, so it is its own unit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_server.py
import asyncio

import pytest

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
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `TurnRegistry`** in `agent_server.py`: a dict of
  `turn_id -> {"session_id": str, "future": asyncio.Future}`. `open()` returns
  `uuid4().hex`; `decide()` sets the future, raising `KeyError` for an unknown turn and
  `ValueError` for one already decided; `wait_for_decision()` wraps `asyncio.wait_for` and
  returns the same `{"approved": False, "reason": "expired"}` shape `_decide` already uses in
  `agent/loop.py`, so the loop's declined branch needs no new case.

- [ ] **Step 4: Run to verify it passes. Step 5: Commit.**

---

### Task 8.2: The routes

**Files:** Modify `agent_server.py`, `tests/test_agent_server.py`

- [ ] **Step 1: Write the failing tests** (Starlette's `TestClient`, no network):

```python
# append to tests/test_agent_server.py
from starlette.testclient import TestClient

from agent_server import app


def test_health_needs_no_credentials():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_a_turn_without_the_service_key_is_refused_before_spending_anything():
    # An open endpoint that calls OpenAI is a bill anyone can run up.
    with TestClient(app) as client:
        response = client.post(
            "/turn",
            json={"utterance": "hi"},
            headers={"authorization": "Bearer customer-token"},
        )

    assert response.status_code == 401


def test_a_turn_without_a_customer_token_is_refused(monkeypatch):
    monkeypatch.setattr("config.AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn", json={"utterance": "hi"}, headers={"x-agent-key": "k"}
        )

    assert response.status_code == 401


def test_a_decision_for_an_unknown_turn_is_a_404(monkeypatch):
    monkeypatch.setattr("config.AGENT_SERVICE_KEY", "k")

    with TestClient(app) as client:
        response = client.post(
            "/turn/nope/decision",
            json={"approved": False},
            headers={"x-agent-key": "k"},
        )

    assert response.status_code == 404
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement the routes.** `/turn` must:

1. check `X-Agent-Key` against `config.AGENT_SERVICE_KEY`, then the customer bearer;
2. open a `session_scoped_executor`, register the turn, emit the `control` frame carrying
   `turn_id` and `session_id`;
3. run the turn with an `approve` callback that emits the `approval_required` event and then
   awaits `registry.wait_for_decision`;
4. stream each v1 event as `event: assistant`;
5. `registry.close(turn_id)` in a `finally`, so an aborted stream does not leak.

Events must reach the stream **as they happen**, not at the end. `run_turn` returns
accumulated state, so the SSE generator needs each event as it is appended — use LangGraph's
`astream` over state updates, or an `asyncio.Queue` the nodes publish to. Prefer the queue:
it is a smaller change than reworking `run_turn`'s return contract, and Task 4 deliberately
left the transport unbuilt so this choice stayed open.

- [ ] **Step 4: Run to verify it passes. Step 5: Commit.**

---

### Task 8.3: Deploy

- [ ] **Step 1: Add `AGENT_SERVICE_KEY` to `config.py`**, defaulting to `""`, and make
  `agent_server.py` refuse to boot without it — the same call `server.py` already makes for
  `MCP_APPROVAL_SECRET`. Better loudly broken than quietly open.

- [ ] **Step 2: Create the Railway service** `agent`, from
  `arone011-creator/mcp-ecom-agent-layer`, branch `main`, start command
  `python agent_server.py`.

- [ ] **Step 3: Set its variables:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `MCP_SERVER_URL`
  (the private address, `http://mcp.railway.internal:8080`), `AGENT_SERVICE_KEY`.
  **Do not** set `MCP_APPROVAL_SECRET` on this service — it has no business minting, and
  absence is a stronger statement than discipline.

- [ ] **Step 4: Generate a domain and verify.**

**MUST PROVE: the readiness check asserts something ONLY the new deployment can satisfy.**
Twice on this project a deploy was verified against the container being replaced, so this is
not a formality. `/health` returns the git SHA it was built from, and the check asserts that
SHA equals the commit just pushed. A `200 OK` proves a container is up; it does not prove it
is *this* container.

- [ ] **Step 5: Run one live turn through the deployed service** end to end — a read-only
  utterance, over SSE, asserting the `control` frame carries a session id and the
  `assistant` frames replay to the same conversation.

- [ ] **Step 6: Record it**, including the follow-up to move the service to private-only once
  the storefront's bridge route exists.

---

### Task 8.4: Tell the storefront

- [ ] **Amend `PLAN_M4_STOREFRONT.txt` Task 3** with the contract above: the two SSE
  channels, the rule that `control` is never forwarded to the browser, the decision endpoint,
  and `AGENT_SERVICE_KEY`. Without it the bridge route will invent its own shape.

---

## Self-Review

**Spec coverage.** Task 8's two requirements — decide third-service-vs-extension, and prove
the readiness check is specific to the new deployment — are the decision section and 8.3
step 4. The unstated prerequisite is 8.1 and 8.2.

**Placeholders.** 8.2 step 3 gives the route's five obligations and names the streaming
choice rather than writing every line, because it is glue over parts tested in 8.1 and
already proven in `agent/`.

**Type consistency.** `TurnRegistry.wait_for_decision` returns the same dict shape
`agent/loop.py::_decide` produces, so `execute_tools`' declined branch is unchanged.

**What this task does not do.** It does not make the agent reachable by a browser, and must
not: the storefront's bridge route is the only intended caller. And it leaves the service
publicly addressable behind a shared key rather than private-only, because Task 8's own
MUST PROVE cannot be satisfied against a service nothing can yet reach.
