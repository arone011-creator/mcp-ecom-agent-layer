"""The event contract between this agent and the storefront's chat UI.

CANONICAL. The storefront's lib/assistant/events.ts is a translation of
this file, and contracts/assistant-events.v1.json is the golden stream
both sides test against. Change the shape here and that fixture fails on
both sides, which is the point: two prose descriptions of one contract
drift, one shared artefact does not.

Six types, one envelope, versioned from the first commit because Phase
3's interrupt payload consumes it too. replay() returns both the three
parallel lists callers already use and an ordered `timeline`, which is
what a chat transcript is rendered from.
"""

from typing import Any

SCHEMA_VERSION = 1

EVENT_TYPES = frozenset(
    {
        "message",
        "message_delta",
        "tool_started",
        "tool_completed",
        "approval_required",
        "error",
    }
)

# The sequence number of an event that is NOT part of the numbered record:
# a live rendering hint produced beside the turn rather than by it. Two
# things carry it -- message_delta, and the approval_required that
# agent_server.py emits before the graph blocks -- and neither may consume
# a number the record needs.
OUT_OF_BAND = -1


def _envelope(seq: int, type_: str, data: dict[str, Any]) -> dict[str, Any]:
    # Payload nested rather than flattened: an envelope key and a future
    # payload key can then never collide.
    return {"v": SCHEMA_VERSION, "seq": seq, "type": type_, "data": data}


def message(seq: int, text: str) -> dict[str, Any]:
    """Assistant prose. The only event whose content the model authored."""
    return _envelope(seq, "message", {"text": text})


def message_delta(text: str) -> dict[str, Any]:
    """A fragment of assistant prose, while it is still being written.

    ADDITIVE AND OPTIONAL. The `message` event that follows carries the
    whole answer and remains authoritative; these exist so the customer
    watches it arrive instead of staring at a spinner. A reader that has
    never heard of this type ignores it and shows the finished message,
    which is precisely the behaviour this contract had before.

    Takes no sequence number. Deltas are out of band by definition, and a
    caller that could pick a number could pick one the record needs.
    """
    return _envelope(OUT_OF_BAND, "message_delta", {"text": text})


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
        # verbatim: the storefront writes these sentences to be acted on.
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
    # Prose that has arrived in fragments and has not yet been closed by
    # the authoritative message. Held apart from `text` so the message can
    # replace it rather than land beside it as a duplicate.
    pending = ""
    # The ordered view. text/tools/errors are three parallel lists and so
    # cannot say what came before what, which is all a transcript is.
    timeline: list[dict[str, Any]] = []
    # Index of the text item still being written, or None.
    open_text: int | None = None
    # call_ids already placed. One call can emit approval_required,
    # tool_started AND tool_completed, and it is one thing on screen.
    charted: set[str] = set()

    for event in events:
        if event.get("v") != SCHEMA_VERSION:
            raise ValueError(
                f"Event schema v{event.get('v')} cannot be replayed by a "
                f"v{SCHEMA_VERSION} reader"
            )

        # An out-of-band event is not part of the numbered record. Counted
        # here it would drag the low end of the range down and invent gaps
        # that never happened.
        if event["seq"] != OUT_OF_BAND:
            seen.append(event["seq"])

        type_ = event["type"]
        data = event.get("data", {})

        if type_ == "message_delta":
            pending += data["text"]
            if open_text is None:
                timeline.append({"kind": "text", "text": ""})
                open_text = len(timeline) - 1
            timeline[open_text]["text"] += data["text"]

        elif type_ == "message":
            # The message wins. Its text is redacted over the whole answer
            # and may legitimately differ from the sum of the fragments;
            # when it does, the redacted one is what the customer keeps.
            text.append(data["text"])
            pending = ""
            if open_text is not None:
                # In place, so the answer keeps its position relative to
                # tool calls that came after it.
                timeline[open_text]["text"] = data["text"]
                open_text = None
            else:
                timeline.append({"kind": "text", "text": data["text"]})

        elif type_ in ("tool_started", "approval_required"):
            open_text = None
            if data["call_id"] not in charted:
                charted.add(data["call_id"])
                timeline.append({"kind": "tool", "call_id": data["call_id"]})

            call_id = data["call_id"]
            # One call, not two. An approved high-risk call emits
            # approval_required and then tool_started under the SAME
            # call_id -- listing it twice drew two chips for one
            # cancellation, which the live approval gate caught.
            if call_id not in tools:
                order.append(call_id)
                tools[call_id] = {
                    "call_id": call_id,
                    "tool": data["tool"],
                    "arguments": data["arguments"],
                }
            if type_ == "approval_required":
                tools[call_id]["awaiting_approval"] = True
            else:
                # It started, so it is no longer waiting on anyone.
                tools[call_id].pop("awaiting_approval", None)

        elif type_ == "tool_completed":
            # A completion without its start still records: half a pair is
            # a symptom worth seeing, not one worth swallowing.
            open_text = None
            if data["call_id"] not in charted:
                charted.add(data["call_id"])
                timeline.append({"kind": "tool", "call_id": data["call_id"]})

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
            open_text = None
            timeline.append({"kind": "error", **data})

        # Any other type is ignored on purpose. A newer agent must not be
        # able to crash an older reader.

    # A run of fragments no message ever closed: the turn was cut off. The
    # words were on the customer's screen, so dropping them now would erase
    # something they read -- a worse lie than an unfinished sentence.
    if pending:
        text.append(pending)

    expected = range(min(seen), max(seen) + 1) if seen else []

    return {
        "text": text,
        "tools": [tools[call_id] for call_id in order],
        "errors": failures,
        "gaps": [seq for seq in expected if seq not in set(seen)],
        "timeline": timeline,
    }
