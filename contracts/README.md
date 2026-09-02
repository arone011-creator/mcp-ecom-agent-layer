# The assistant event contract

**This directory is canonical.** The chat UI, this agent, and Phase 3's pause-and-resume
payload all depend on one event shape, and the design documents on both sides say the same
thing about it: it is the cheapest thing to get wrong.

Two implementations exist, and neither is the source of truth:

| Where | What it is |
|---|---|
| `agent/events.py` (this repo) | The Python emitter and the reference `replay()` reducer. |
| `lib/assistant/events.ts` (storefront) | The TypeScript parser and the UI's reducer. |
| `contracts/assistant-events.v1.json` | The golden stream. **Both** must replay it to the same conversation. |

## How agreement is actually enforced

The two halves live in separate git repositories, in different languages, so "define it once
and import it" cannot mean a shared module. It means a shared *artefact*.

`assistant-events.v1.json` holds a recorded stream covering all five event types and, beside
it, the `expected` conversation that replaying it must produce. This repository asserts that
in `tests/test_agent_events.py`; the storefront vendors the same file and asserts the same
thing about its own parser. Change the shape on either side and a test fails on that side —
which is the entire mechanism. Two prose descriptions of one contract drift; one file that
both sides execute against does not.

**Changing the contract means changing that file, and expecting both repositories to go red
until both are updated.** That is not an inconvenience to route around; it is the contract
working.

## The envelope

```json
{"v": 1, "seq": 0, "type": "tool_started", "data": {}}
```

| Field | Meaning |
|---|---|
| `v` | Schema version, `1` from the first commit. Phase 3 consumes this too. |
| `seq` | Monotonic from 0, no gaps. Makes replay deterministic and a dropped event detectable. |
| `type` | One of the five below. |
| `data` | Payload, shape determined by `type`. |

The payload is nested rather than flattened so that an envelope key and a future payload key
can never collide.

## The five events

### `message`
Assistant prose — the only event whose content the model authored, and therefore the only
one a UI must treat as untrusted text.
```json
{"text": "Your most recent order is ORD-1042, still pending."}
```

### `tool_started`
`arguments` is a structured object, never a rendered string. A UI that wants to show
"checking stock for 3" composes that itself from these fields.
```json
{"call_id": "call_1", "tool": "check_inventory", "arguments": {"product_id": "p1"}}
```

### `tool_completed`
Paired to its start by `call_id`. `result` carries the tool result itself, because product
and order cards are rendered from tool results rather than from agent prose. On failure `ok`
is false, `result` is absent, and `error` carries the storefront's own message **verbatim** —
a 409 contains the number that IS available, and parsing or rewording it anywhere else would
be a second implementation of a rule the storefront owns.
```json
{"call_id": "call_1", "tool": "check_inventory", "ok": true, "result": {"available": 17}}
{"call_id": "call_2", "tool": "add_to_cart", "ok": false, "error": "409: Only 17 available"}
```

Every `tool_started` gets a `tool_completed`, including a call the agent refused to run. A
start with no completion is a chip that spins forever.

### `approval_required`
Names WHICH action is waiting, in structured arguments.
```json
{"call_id": "call_3", "tool": "cancel_order", "arguments": {"order_id": "ord_9"}}
```

### `error`
The turn itself failed, as distinct from a tool failing.
```json
{"message": "The assistant could not reach the shop.", "retryable": true}
```

## Two deliberate departures from the plan text

Recorded here because a reader of `PLAN_M4_STOREFRONT.txt` section 3 will otherwise
implement something this contract does not send.

**1. `approval_required` carries no token handle.** The plan's summary line lists one, but
its own Task 5 has the approve route mint the token after a human click, from the arguments
in the event. A handle minted by the agent would be the agent participating in its own
approval — the exact thing the risk-tier design exists to prevent. `call_id` is the
correlation key, and it is sufficient: the storefront looks the order up fresh by id anyway,
so the event names *which* order and the lookup answers *what is true* about it.

**2. `tool_completed` carries no cache-invalidation hint.** Rule 3 requires a chat-driven
cart change to invalidate the same query the cart page and header badge already use. Which
cache key a given tool touches is the storefront's knowledge; the event carries `tool` and
`ok`, and the storefront maps. An agent shipping cache keys would be this repository
asserting facts about a UI it cannot see.

## Rules a consumer must honour

1. **Render the approval card from `arguments`, never from `message` text.** Product
   descriptions and reviews are written by strangers and reach the agent's context. If the
   agent writes the words next to the button, an injected review can write them too.
2. **Ignore an unknown `type` rather than failing.** Forward compatibility runs in the
   direction that actually happens: a newer agent deployed against an older UI must not
   crash it.
3. **Report a `seq` gap; do not smooth over it.** A dropped event means what is on screen is
   not what happened, and silence is the worse failure.
4. **Nothing in a stream is a secret.** These events leave the process for a browser. A test
   in this repository asserts no bearer token or authorization header ever appears in one.

## What is not here yet

Streaming transport. These events accumulate in turn state; delivering them over the wire as
they occur is the bridge route's job (storefront Task 3) against an agent HTTP surface that
does not exist yet (agent Task 8).

`approval_required` **is** emitted, as of agent Task 5, and proved live.

## Where the session id lives, and why it is not in the event

The storefront mints an approval by calling the MCP server's `POST /approvals` with the id
of the session the resumed call will actually ride. A token minted against any other session
is rejected — the binding is deliberate, and it is what stops an approval crossing
conversations.

That id is **not** in the `approval_required` event. It travels one layer below, as a field
on the argument the agent hands its `approve` callback — the server-side seam the bridge
route already sits on. The event reaches a browser; the session id has no business being
there, and the frozen contract above stays unchanged because of it.

The callback's argument is therefore:

```json
{"call_id": "call_3", "tool": "cancel_order", "arguments": {"order_id": "ord_9"},
 "session_id": "3be90c18607f4441a8cde0b706182628"}
```

and its answer is `{"approved": true, "token": "..."}`, `{"approved": false}`, or — when a
deadline passes rather than a person deciding — `{"approved": false, "reason": "expired"}`.
The UI should say which of the last two happened: "nobody answered" and "you said no" are
different facts about the same unchanged order.
