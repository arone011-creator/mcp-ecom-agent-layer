# The assistant event contract

**This directory is canonical.** The chat UI, this agent, and Phase 3's pause-and-resume
payload all depend on one event shape, and the design documents on both sides say the same
thing about it: it is the cheapest thing to get wrong.

Two implementations exist, and neither is the source of truth:

| Where | What it is |
|---|---|
| `agent/events.py` (this repo) | The Python emitter and the reference `replay()` reducer, including the ordered timeline. |
| `lib/assistant/events.ts` (storefront) | The TypeScript parser and the UI's reducer. |
| `contracts/assistant-events.v1.json` | The golden stream. **Both** must replay it to the same conversation. |

## How agreement is actually enforced

The two halves live in separate git repositories, in different languages, so "define it once
and import it" cannot mean a shared module. It means a shared *artefact*.

`assistant-events.v1.json` holds a recorded stream covering all six event types and, beside
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
| `seq` | Monotonic from 0, no gaps — **or `-1`**, see below. Makes replay deterministic and a dropped event detectable. |
| `type` | One of the six below. |
| `data` | Payload, shape determined by `type`. |

### `seq: -1` means out of band

Some events are produced *beside* the turn rather than by it: a live rendering hint, or a
notice the HTTP surface raises before the graph blocks on a human. They carry `-1`, they are
**excluded from gap accounting**, and there may be any number of them. Counting them would
drag the low end of the range down and invent gaps that never happened.

Two things carry it today: `message_delta`, and the `approval_required` that
`agent_server.py` emits ahead of the pause. Everything else is numbered from 0.

The payload is nested rather than flattened so that an envelope key and a future payload key
can never collide.

## The six events

### `message`
Assistant prose — the only event whose content the model authored, and therefore the only
one a UI must treat as untrusted text. **Authoritative**: this is the redacted text, and
where it differs from the fragments that preceded it, this is the one that stays on screen.
```json
{"text": "Your most recent order is ORD-1042, still pending."}
```

### `message_delta`
A fragment of that prose, while it is still being written. Always `seq: -1`.

Additive and optional, in the strict sense: a reader that has never heard of this type
ignores it and shows the finished `message`, which is exactly the behaviour this contract had
before deltas existed. That is what made it safe to add to a frozen v1.
```json
{"text": "Your most recent order "}
```

A reducer must **accumulate fragments into a pending buffer and let the next `message`
replace it, not join it** — otherwise the customer reads the answer twice. A run that no
`message` ever closes (the turn is still in flight, or it was cut off) survives as partial
text: those words were already on the screen, and erasing them is the worse lie.

The fragments are redacted chunk by chunk and flushed only at whitespace boundaries, because
a URL contains no whitespace and so is always complete inside one chunk. The `message` is
redacted over the whole answer. The two agree in practice; where they cannot, the `message`
wins by the rule above, which is why the ordering is not merely cosmetic.

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

## The ordered view

`replay()` also returns `timeline`: the same conversation, in the order it
happened. The three parallel lists cannot express that, and a chat transcript
is nothing but ordering — without it a UI renders every question, then every
tool chip, then every answer, which looks right for one exchange and is wrong
for two.

    {"kind": "text",  "text": "Your most recent order is ORD-1042."}
    {"kind": "tool",  "call_id": "call_1"}
    {"kind": "error", "message": "...", "retryable": true}

A tool is **named, not embedded.** Its state changes after it first appears, so
an embedded snapshot would be captured as "working" and stay that way; the
timeline says where it sits and `tools` says what became of it.

One call is one item. A high-risk call emits `approval_required`, then
`tool_started`, then `tool_completed` under a single `call_id`, and that is one
thing on screen.

A `message` **replaces its fragments in place** rather than appending, so an
answer keeps its position relative to tool calls that followed it.

## Rules a consumer must honour

1. **Render the approval card from `arguments`, never from `message` text.** Product
   descriptions and reviews are written by strangers and reach the agent's context. If the
   agent writes the words next to the button, an injected review can write them too.
2. **Ignore an unknown `type` rather than failing.** Forward compatibility runs in the
   direction that actually happens: a newer agent deployed against an older UI must not
   crash it.
3. **Report a `seq` gap; do not smooth over it.** A dropped event means what is on screen is
   not what happened, and silence is the worse failure. `seq: -1` is not a gap and is not
   counted; see above.
4. **Nothing in a stream is a secret.** These events leave the process for a browser. A test
   in this repository asserts no bearer token or authorization header ever appears in one.

## Streaming

Delivered, and it took three separate pieces — worth listing, because the first two were done
before the third and the result still arrived in one lump, which is how the gap was found.

1. **The transport.** One turn is one SSE stream (`agent_server.py`), forwarded frame by
   frame by the storefront's bridge route.
2. **Publishing as the graph runs.** `run_turn` drives the graph with `astream`, publishing
   after each step. It used to use `ainvoke`, which returns only when the whole turn is
   over — so every event, tool chips included, was back-filled at the end.
3. **Fragments of prose.** The model call streams, and each fragment becomes a
   `message_delta`.

Miss any one and the customer waits for the whole answer. All three together are what makes
the chat look like a chat.

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
