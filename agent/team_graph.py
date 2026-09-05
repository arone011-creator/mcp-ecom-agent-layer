"""The supervisor, and the node that runs a specialist.

    START -> supervisor -> (delegating? -> delegate -> supervisor) : END

The supervisor holds no shop tools. Its tools ARE the specialists: it
delegates, gets an answer back as an ordinary tool result, may delegate
again, and then writes the reply.

ONE DELEGATE NODE, NOT ONE NODE PER SPECIALIST. Two specialists reached
in the same assistant message would be two parallel nodes writing to one
state, and their event numbering would have to be reconciled afterwards.
Handling every delegation in one node, in order, makes the numbering
arithmetic trivially correct instead of merely careful.

A SPECIALIST IS build_graph() UNCHANGED. That is the point of the design:
the loop that already works -- tool calls, repeat guard, forbidden
arguments, the approval interrupt -- is the same loop at both levels, with
a shorter tool list and a different prompt.
"""

import json

from langgraph.graph import END, START, StateGraph

from agent.delegation import is_delegation, member_for_tool
from agent.events import message as message_event
from agent.events import tool_completed, tool_started
from agent.loop import (
    TurnState,
    _next_seq,
    _tool_calls_of,
    _tool_message,
    build_graph,
)
from agent.prompt import redact_untrusted_urls, untrusted_urls
from agent.tools import restricted_executor

# What a specialist is allowed to spend before it has to answer with
# whatever it has. Lower than the turn's own ceiling: a specialist that
# cannot finish in this many steps is lost, and the supervisor asking a
# different one is a better outcome than a longer wander.
SPECIALIST_MAX_STEPS = 12


def _delegations(message: dict) -> list[tuple[str, str, str]]:
    """(call_id, tool_name, request) for each delegation in this message."""
    found = []

    for call in _tool_calls_of(message):
        name = call["function"]["name"]
        if not is_delegation(name):
            continue

        # Arguments arrive as a JSON string and the escaping varies.
        # Parse; never string-match.
        arguments = json.loads(call["function"]["arguments"] or "{}")
        found.append((call["id"], name, arguments.get("request", "")))

    return found


def build_team_graph(
    model_call,
    execute_tool,
    checkpointer=None,
    specialist_model_call=None,
    on_event=None,
):
    """The supervisor's graph, with specialists as subgraphs.

    A checkpointer is required for the same reason build_graph needs one:
    interrupt() writes the paused state through it, and here that pause
    starts one level down, inside a specialist.

    A SPECIALIST GETS ITS OWN MODEL CALL, and that is not a detail. The
    server's model call carries on_delta, which pushes prose fragments
    STRAIGHT to the browser as they are written -- bypassing the graph and
    every filter in it. Driving a specialist with it streams its private
    working answer into the customer's chat, ahead of the supervisor's
    real one. Passing a call without on_delta is the only thing that stops
    that; there is nothing downstream to catch it.

    Defaults to `model_call` so a test can pass one function and mean it.
    """
    specialist_model_call = specialist_model_call or model_call

    async def supervisor(state: TurnState) -> dict:
        message = await model_call(state["messages"], state.get("tools", []))
        dumped = message.model_dump(exclude_none=True)
        dumped.setdefault("role", "assistant")

        # The same backstop the single agent has, over a UNION rather than
        # over this state's messages alone -- and that difference is the
        # whole of it. The supervisor never reads a product description
        # itself; the delegate node hands it the specialist's finished
        # answer, so untrusted_urls() over its own transcript finds
        # nothing. A mutation run proved that: deleting this line broke no
        # test, because as written over one state it could never fire.
        answer = redact_untrusted_urls(
            message.content,
            untrusted_urls(state["messages"])
            | set(state.get("untrusted_seen", [])),
        )
        dumped["content"] = answer

        # A tool turn has no prose, and an empty message event would be a
        # blank bubble in the chat.
        events = [message_event(_next_seq(state), answer)] if answer else []

        if events and on_event is not None:
            on_event(events[0])

        return {"messages": [dumped], "answer": answer, "events": events}

    async def delegate(state: TurnState) -> dict:
        requests = _delegations(state["messages"][-1])

        messages = []
        events = []
        # What the specialist read that the supervisor never will.
        seen: set[str] = set()

        def record(event):
            """Append it AND hand it over now, rather than on return."""
            events.append(event)

            if on_event is not None:
                on_event(event)

            return event

        for call_id, tool_name, request in requests:
            member = member_for_tool(tool_name)

            # Derived from state, not counted, so this node produces the
            # same numbers on the re-run that follows an approval pause.
            base = _next_seq(state) + len(events)

            record(tool_started(base, call_id, tool_name, {"request": request}))

            # The specialist's chips as they happen, not when it finishes.
            # Without this the whole delegation -- the hand-off, every
            # tool inside it, and its completion -- reached the browser in
            # one burst when this node returned, so four chips appeared
            # at once and every one of them already said "done".
            #
            # The SAME renumbering the authoritative list below uses, so
            # the two agree and the sweep deduplicates them by seq. That
            # keeps the sweep as a real backstop: a node that ever forgets
            # to emit still reaches the customer, one step later.
            forwarded = []

            def collect(event, _kept=forwarded, _base=base):
                if event["type"] == "message":
                    return

                _kept.append(event)

                if on_event is not None:
                    on_event(dict(event, seq=_base + len(_kept)))

            specialist = build_graph(
                specialist_model_call,
                restricted_executor(execute_tool, member.tools),
                checkpointer=checkpointer,
                on_event=collect,
            )

            result = await specialist.ainvoke(
                {
                    # A FRESH TRANSCRIPT. The specialist sees its own
                    # prompt and the request, never the supervisor's
                    # conversation -- which is what keeps a delegation a
                    # bounded question rather than a second agent
                    # inheriting everything.
                    "messages": [
                        {"role": "system", "content": member.prompt},
                        {"role": "user", "content": request},
                    ],
                    "tools": state.get("specialist_tools", {}).get(member.name, []),
                    "answer": None,
                    "failed": [],
                    "events": [],
                    "seeded": 1,
                    # Its events slot into the supervisor's stream rather
                    # than starting again at zero.
                    "seq_base": base + 1,
                },
                config={"recursion_limit": SPECIALIST_MAX_STEPS},
            )

            # READ FROM THE SPECIALIST'S OWN TRANSCRIPT, which is the only
            # place the raw tool result exists. Carried up as data, never
            # as text: putting the untrusted block itself into the
            # supervisor's messages would hand the attacker a second
            # reader rather than protect the first.
            seen |= untrusted_urls(result.get("messages", []))

            # A SPECIALIST'S PROSE IS NOT SHOWN TO THE CUSTOMER. It is
            # answering the SUPERVISOR, and that answer travels back as
            # the tool result below. Forwarding its `message` event too
            # put a second bubble in the chat saying the same thing in
            # different words -- and put it there BEFORE the delegation
            # chip resolved, because the specialist's last event is
            # numbered ahead of the tool_completed that closes it.
            #
            # Its TOOL events are kept: "Searching products" inside
            # "Asking the product specialist" is exactly the visible
            # working the chips are for.
            #
            # Renumbered rather than left with a hole, because the
            # specialist numbered its events expecting all of them to be
            # forwarded. Pairing is by call_id, so renumbering is safe.
            sub_events = [
                dict(event, seq=base + 1 + offset)
                for offset, event in enumerate(
                    e for e in result.get("events", []) if e["type"] != "message"
                )
            ]
            answer = result.get("answer") or (
                "The specialist finished without an answer."
            )

            events.extend(sub_events)
            record(
                tool_completed(
                    base + 1 + len(sub_events),
                    call_id,
                    tool_name,
                    result={"answer": answer},
                )
            )
            messages.append(_tool_message(call_id, {"answer": answer}))

        return {
            "messages": messages,
            "events": events,
            "untrusted_seen": sorted(seen),
        }

    def route(state: TurnState) -> str:
        return "delegate" if _delegations(state["messages"][-1]) else END

    graph = StateGraph(TurnState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("delegate", delegate)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route, {"delegate": "delegate", END: END}
    )
    graph.add_edge("delegate", "supervisor")

    return graph.compile(checkpointer=checkpointer)
