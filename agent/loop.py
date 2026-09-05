"""One turn of the agent: utterance in, tool calls out, answer back.

LangGraph owns the graph and the state; the OpenAI SDK is called directly
from a plain node, with no model-wrapper library in between, so request
parameters stay under this project's control. Two nodes and a router:

    call_model -> (tool calls? -> execute_tools -> call_model) : END

`model_call`, `execute_tool` and `approve` are injected rather than
imported, because what is worth testing here is the loop's own behaviour
-- does it execute what was asked, feed results back in the shape the API
wants, stop when the model stops, refuse an identity argument, and wait
for a human before an irreversible action -- not the model's judgement,
which is the eval harness's job.

All nine tools, as of M4 Task 5. A high-risk call interrupts the graph
and resumes only with a token some human caused to be minted; the turn
holds ONE MCP session throughout, because that token is bound to the
session id and a session per call would invalidate it.
"""

import asyncio
import json
import operator
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from fastmcp.exceptions import ToolError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import config
from agent.events import approval_required as approval_required_event
from agent.events import message as message_event
from agent.events import tool_completed, tool_started
from agent.history import sanitise_history
from agent.prompt import (
    SYSTEM_PROMPT,
    StreamingRedactor,
    redact_untrusted_urls,
    untrusted_urls,
)
from agent.tools import (
    AGENT_TOOLS,
    HIGH_RISK_TOOLS,
    build_transport,
    list_openai_tools,
    reject_forbidden_arguments,
)

ModelCall = Callable[[list[dict], list[dict]], Awaitable[Any]]
ToolExecutor = Callable[[str, dict], Awaitable[Any]]
# Takes the interrupt payload (call_id, tool, arguments) and answers
# {"approved": bool, "token": str | None, "reason": str | None}.
ApprovalCallback = Callable[[dict], Awaitable[dict]]


class TurnState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tools: list[dict]
    answer: str | None
    # Every (tool, arguments) pair that has already failed this turn.
    # Accumulated so the loop can refuse an identical retry.
    failed: Annotated[list[str], operator.add]
    # The stream the chat UI is built from. Accumulated by the same
    # reducer as messages, so a node returns only what it added. The
    # shape is agent/events.py, which the storefront also implements.
    events: Annotated[list[dict], operator.add]
    # How many messages the turn was SEEDED with -- the system prompt plus
    # any replayed history. Written once by run_turn and returned by no
    # node, so it still describes the seed after the graph has appended to
    # `messages` many times. agent/history.py::exportable_context uses it
    # to send out only what this turn added.
    seeded: int
    # Where this state's event numbering starts. Zero for a turn; for a
    # specialist running as a subgraph it is the supervisor's next number,
    # so one ordered stream comes out of several states.
    #
    # A BASE RATHER THAN A SHARED COUNTER, deliberately. _next_seq is
    # derived from state so that a node re-run after an approval pause
    # produces the same numbers on both passes; a mutable counter would
    # hand out different ones the second time.
    seq_base: int
    # Each specialist's OpenAI tool schemas, keyed by member name. Listed
    # once per turn by the caller rather than per delegation: listing
    # tools opens an MCP connection, and doing that inside the graph would
    # put a network round trip in the middle of every hand-off.
    specialist_tools: dict[str, list[dict]]
    # Every URL a SPECIALIST read out of untrusted content, carried up to
    # the supervisor.
    #
    # Without this the supervisor's redaction backstop is inert, and a
    # mutation run proved it: the supervisor is handed the specialist's
    # finished answer, never the raw tool result, so untrusted_urls() over
    # its own messages finds nothing to match. Defence in depth that
    # cannot fire is not defence in depth.
    untrusted_seen: Annotated[list[str], operator.add]


def _tool_calls_of(message: dict) -> list[dict]:
    return message.get("tool_calls") or []


def _signature(name: str, arguments: dict) -> str:
    """Identifies one exact call, so a repeat of it can be recognised."""
    return json.dumps({"tool": name, "args": arguments}, sort_keys=True)


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
        return [{"approved": False, "reason": "expired"} for _ in requests]


def _emitter(on_event):
    """One place an event becomes visible to the caller, and exactly once.

    DEDUPLICATED BY SEQ RATHER THAN COUNTED, and the reason is the whole
    of this design. Events are now handed over as a node PRODUCES them --
    otherwise a tool chip could only ever be drawn already resolved,
    because the state a node returns arrives all at once -- and they also
    accumulate in state, where the sweep below would send them a second
    time.

    seq is derived from state rather than counted, so the node re-run
    that follows an approval pause produces the SAME numbers on both
    passes. That makes it the one identifier a re-run cannot duplicate,
    which a running count could not survive.
    """
    seen: set[int] = set()

    def emit(event) -> None:
        seq = event.get("seq")

        if seq in seen:
            return

        seen.add(seq)

        if on_event is not None:
            on_event(event)

    return emit


def _publish(emit, state: TurnState) -> None:
    """The sweep. Catches anything a node produced without emitting it.

    Belt and braces on purpose: a node that forgets to emit still reaches
    the caller here, one step later, rather than never.
    """
    for event in state.get("events", []):
        emit(event)


async def _drive(app, payload, settings, emit):
    """Run the graph to its next stopping point, publishing as it goes.

    astream rather than ainvoke, and that is the whole point. ainvoke
    returns only when the turn is over, so events could not be handed over
    until then -- the customer watched a spinner and then received the
    tool chips and the answer together, every chip already resolved. A
    stream that arrives all at once is not a stream.

    stream_mode="values" yields the accumulated state after each step, and
    the last one carries __interrupt__ when the graph pauses, so the
    approval loop below reads exactly what it read before.
    """
    state = None

    async for step in app.astream(payload, config=settings, stream_mode="values"):
        state = step
        _publish(emit, step)

    return state


def _next_seq(state: TurnState) -> int:
    """The next sequence number, derived from what has already been emitted.

    LangGraph nodes return only their additions, so a counter held in a
    node would restart on the next pass through it. The accumulated
    length is the one number both nodes can agree on without sharing
    state of their own -- and being derived rather than counted is also
    what makes it stable across the node re-run that follows an approval.

    `seq_base` offsets the whole run, so a specialist's events land in the
    supervisor's stream without colliding with it.
    """
    return state.get("seq_base", 0) + len(state.get("events", []))


def _tool_message(call_id: str, payload) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, default=str),
    }


def build_graph(
    model_call: ModelCall | None = None,
    execute_tool: ToolExecutor | None = None,
    checkpointer=None,
    on_event=None,
):
    """The turn as a graph.

    A checkpointer is required for the approval pause -- interrupt()
    writes the paused state through it. InMemorySaver is honest about the
    current deployment (one replica, one process), and is the same
    limitation approvals.py already documents for its spent-nonce set.
    """

    async def call_model(state: TurnState) -> dict:
        message = await model_call(state["messages"], state.get("tools", []))
        dumped = message.model_dump(exclude_none=True)
        # content is None on a tool turn; the API rejects a null content
        # field on the way back in.
        dumped.setdefault("role", "assistant")

        # The system prompt forbids repeating a URL out of untrusted
        # content, and in testing the model obeys. This is the backstop
        # for the turn it does not -- applied here so the redaction is in
        # the text the UI renders, not only in a return value.
        answer = redact_untrusted_urls(
            message.content, untrusted_urls(state["messages"])
        )
        dumped["content"] = answer

        # A tool turn has no prose, and an empty message event would be a
        # blank bubble in the chat.
        events = [message_event(_next_seq(state), answer)] if answer else []

        if events and on_event is not None:
            on_event(events[0])

        return {"messages": [dumped], "answer": answer, "events": events}

    async def execute_tools(state: TurnState) -> dict:
        parsed = []
        for call in _tool_calls_of(state["messages"][-1]):
            name = call["function"]["name"]
            # Arguments arrive as a JSON string and the escaping varies.
            # Parse; never string-match.
            arguments = json.loads(call["function"]["arguments"] or "{}")

            # Not a recoverable failure: identity is never the model's to
            # assert, so this refuses the turn rather than inviting a retry.
            reject_forbidden_arguments(name, arguments)

            # A field code injects is never one the model may pre-fill.
            arguments.pop("approval_token", None)

            parsed.append((call["id"], name, arguments))

        # This node emits several events; each needs its own number, and
        # they cannot come from _next_seq because state does not change
        # until the node returns.
        seq = _next_seq(state)

        # EVERY interrupt happens before ANY execution. LangGraph re-runs
        # this node from the top when the thread resumes, so a side effect
        # placed before the interrupt would happen twice. Collecting the
        # approvals first makes that re-run free.
        events = []
        decisions = {}

        def record(event):
            """Append it AND hand it over now.

            Now, rather than when this node returns, because a chip only
            ever drawn after its tool has finished is a chip only ever
            drawn done.
            """
            events.append(event)

            if on_event is not None:
                on_event(event)

            return event

        for call_id, name, arguments in parsed:
            if name not in HIGH_RISK_TOOLS:
                continue

            record(approval_required_event(seq, call_id, name, arguments))
            seq += 1

            # Pauses the thread on the first pass; returns the resume
            # value on the second.
            decisions[call_id] = interrupt(
                {"call_id": call_id, "tool": name, "arguments": arguments}
            )

        results = []
        newly_failed = []
        already_failed = set(state.get("failed", []))

        for call_id, name, arguments in parsed:
            signature = _signature(name, arguments)

            decision = decisions.get(call_id)

            if decision is not None and not decision.get("approved"):
                declined = (
                    "The approval request expired before anyone answered it. "
                    "Nothing was sent."
                    if decision.get("reason") == "expired"
                    else "The customer declined this action. Nothing was sent."
                )
                results.append(_tool_message(call_id, {"error": declined}))
                # No tool_started: it never started. The completion still
                # fires so the UI resolves the card rather than leaving it
                # open forever.
                record(tool_completed(seq, call_id, name, error=declined))
                seq += 1
                continue

            record(tool_started(seq, call_id, name, arguments))
            seq += 1

            if signature in already_failed:
                refusal = (
                    "This exact call was already tried this turn and failed. "
                    "Read the earlier error and change the arguments rather "
                    "than repeating it."
                )
                results.append(_tool_message(call_id, {"error": refusal}))
                # Emitted even though nothing ran: a start without a
                # completion is a chip that spins forever.
                record(tool_completed(seq, call_id, name, error=refusal))
                seq += 1
                continue

            # Injected here, by code, from the decision a human caused.
            # The field is stripped from the schema the model is shown, so
            # this is never a value the model could have produced.
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
                record(tool_completed(seq, call_id, name, error=str(failure)))
                seq += 1
                continue

            results.append(_tool_message(call_id, result))
            record(tool_completed(seq, call_id, name, result=result))
            seq += 1

        return {"messages": results, "failed": newly_failed, "events": events}

    def route(state: TurnState) -> str:
        return "tools" if _tool_calls_of(state["messages"][-1]) else END

    graph = StateGraph(TurnState)
    graph.add_node("model", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer)


async def run_turn(
    utterance: str,
    *,
    model_call: ModelCall,
    execute_tool: ToolExecutor,
    tools: list[dict] | None = None,
    history: list[dict] | None = None,
    max_steps: int = 25,
    approve: ApprovalCallback | None = None,
    approval_timeout_seconds: float = 300.0,
    session_id: str | None = None,
    on_event=None,
    build=None,
    system_prompt: str | None = None,
    specialist_tools: dict[str, list[dict]] | None = None,
) -> TurnState:
    """One turn, start to finish, pausing for approval where required.

    `session_id` is the MCP session this turn's calls ride on. It is
    passed to `approve` because a token is only valid on the session it
    was minted against, and it travels there -- a server-side callback
    argument -- rather than in the event stream, which reaches a browser.

    `history` is the earlier turns of this conversation, replayed by the
    storefront -- which owns the conversation, because this service holds
    the model key and must not also hold customer data. It is checked
    here as well as at the HTTP boundary: a caller inside this process
    (the eval harness, a future script) never touches the route, and the
    guarantee that nothing can seed a `system` message from stored data
    has to hold for them too.

    `on_event` receives each event as the graph appends it. Optional: the
    tests and the eval harness read the accumulated state instead. The
    HTTP surface needs them live, because a stream that only arrives once
    the turn is over is not a stream, and the pause in the middle is
    exactly when a customer most needs to see something.
    """
    # WHICH GRAPH, not which loop. Everything below -- the drive, the
    # publish accounting, the interrupt loop that waits for a human -- is
    # identical for one agent and for a team, and the approval pause is
    # the last thing in this system that should exist in two copies.
    emit = _emitter(on_event)
    app = (build or build_graph)(
        model_call, execute_tool, checkpointer=InMemorySaver(), on_event=emit
    )

    # Refused, not filtered, and refused BEFORE the graph is built: a turn
    # that has already begun cannot un-send a message it was seeded with.
    replayed = sanitise_history(history)

    settings = {
        # A confused agent stops rather than looping. The repeat guard
        # already blocks the obvious case; this bounds the rest, and is
        # the first half of the cost ceiling Decision D calls for.
        "recursion_limit": max_steps,
        # One thread per turn. Nothing outlives a turn today; Phase 3 is
        # where a thread becomes a conversation.
        "configurable": {"thread_id": uuid.uuid4().hex},
    }

    state = await _drive(
        app,
        {
            "messages": [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                # Between the prompt and the new message, in that order.
                # The prompt stays first whatever the storefront sends,
                # and the customer's actual question stays last so it is
                # not read as a continuation of something older.
                *replayed,
                {"role": "user", "content": utterance},
            ],
            "tools": tools or [],
            "specialist_tools": specialist_tools or {},
            "answer": None,
            "failed": [],
            "events": [],
            "seeded": 1 + len(replayed),
        },
        settings,
        emit,
    )

    while state.get("__interrupt__"):
        requests = [
            {**item.value, "session_id": session_id} for item in state["__interrupt__"]
        ]

        if approve is None:
            # The safe default. A turn with no way to reach a human must
            # not proceed as though one had answered.
            decisions = [{"approved": False} for _ in requests]
        else:
            decisions = await _decide(approve, requests, approval_timeout_seconds)

        state = await _drive(
            app,
            Command(resume=decisions[0] if len(decisions) == 1 else decisions),
            settings,
            emit,
        )

    return state


def _openai_client():
    """Seam. Tests replace the transport under this, not this.

    The same shape as _client_for below, and for a sharper reason. The
    first version of the streaming call was tested against a hand-made
    stand-in for the whole client, which meant the SDK's own request
    validation never ran in any test -- and the SDK is what rejected the
    call in production, on the first tool, before a token was read. A
    seam here lets a test drive the real SDK over a fake network, which
    is where the bug actually was.
    """
    from openai import AsyncOpenAI

    # The SDK defaults to a 600-second timeout and retries, so one stalled
    # request can hold a turn for half an hour. Observed: an eval sweep sat
    # for 36 minutes on 34 seconds of CPU. A turn that cannot answer in a
    # minute has already failed the customer waiting for it.
    return AsyncOpenAI(timeout=config.OPENAI_TIMEOUT_SECONDS)


def openai_model_call(
    model: str | None = None, on_usage=None, on_delta=None
) -> ModelCall:
    """The real model call. Kept separate so the loop stays testable.

    `on_usage` receives this request's token usage. Optional, because the
    loop does not care what a turn cost -- the eval harness does, and so
    will the cost ceiling in Decision D. A callback keeps that concern
    out of the graph rather than threading usage through the state, which
    every node would then have to carry and none would use.

    `on_delta` receives the answer in fragments as the model writes them.
    Also optional: the tests and the eval harness read the finished
    message, and only the chat UI needs to watch it arrive. The fragments
    are a rendering hint and nothing more -- the message this returns is
    unchanged and remains what the graph, the state and the record are
    built from.

    THE FRAGMENTS ARE REDACTED ON THE WAY OUT. call_model redacts the
    finished answer, which is no defence here: a fragment released
    unchecked has already been read by the time the finished answer
    exists. StreamingRedactor applies the same rule to text that has not
    finished arriving.

    NOT client.chat.completions.stream(). That helper auto-parses tool
    arguments and therefore refuses any tool that is not `strict`, and
    these tools come from the MCP server's own schemas, which are not --
    to_openai_tool re-nests them and deliberately does not rewrite them.
    It raises ValueError on the first tool before a single token is read.
    So: the plain streaming request, with the SDK's own accumulator
    assembling the chunks. ChatCompletionStreamState is given no tools
    for exactly the same reason, and needs none -- it concatenates
    tool-call argument fragments regardless.
    """
    from openai.lib.streaming.chat import ChatCompletionStreamState

    client = _openai_client()
    chosen = model or config.OPENAI_MODEL

    async def call(messages: list[dict], tools: list[dict]):
        redactor = StreamingRedactor(untrusted_urls(messages))
        state = ChatCompletionStreamState()

        stream = await client.chat.completions.create(
            model=chosen,
            max_completion_tokens=1024,
            messages=messages,
            tools=tools or None,
            stream=True,
            # A streamed request reports no usage unless asked. Omitting
            # this would not break a turn -- it would quietly zero the
            # eval harness's cost column, which is worse.
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            state.handle_chunk(chunk)

            if on_delta is None:
                continue

            for choice in chunk.choices:
                # Tool-call fragments arrive on this same stream and are
                # the accumulator's business, not the customer's. Only
                # prose is shown as it is written.
                text = choice.delta.content
                if not text:
                    continue

                fragment = redactor.push(text)
                if fragment:
                    on_delta(fragment)

        if on_delta is not None:
            tail = redactor.finish()
            if tail:
                on_delta(tail)

        response = state.get_final_completion()

        if on_usage is not None and response.usage is not None:
            on_usage(response.usage)

        return response.choices[0].message

    return call


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
        # Cached while the transport is live. Read after it closes it is
        # None, and the storefront would mint against nothing.
        self._session_id = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def execute(self, name: str, arguments: dict) -> Any:
        result = await self._client.call_tool(name, arguments)
        return json.loads(result.content[0].text) if result.content else None


def _client_for(url: str, token: str):
    """Seam. Tests replace this rather than the transport machinery."""
    from fastmcp import Client

    return Client(build_transport(url, token))


@asynccontextmanager
async def session_scoped_executor(token: str, url: str | None = None):
    """Hold one MCP session for the life of a turn.

    Held rather than reopened because an approval pause happens in the
    middle of a turn, and the token minted during that pause is only
    valid on the session it was minted against.
    """
    client = _client_for(url or config.MCP_SERVER_URL, token)

    async with client:
        session = McpSession(client)
        session._session_id = client.transport.get_session_id()
        yield session


async def answer(
    utterance: str,
    token: str,
    *,
    model: str | None = None,
    approve: ApprovalCallback | None = None,
    approval_timeout_seconds: float = 300.0,
) -> TurnState:
    """The whole thing wired to the real model and the real MCP server.

    One session for the turn, because an approval pause happens in the
    middle of one and the token is minted against the session id.

    `approve` is how a human answers. It is supplied by the caller -- the
    storefront's bridge route, which is already holding the connection it
    streams events over -- and never constructed here: an agent that
    could produce its own approvals would make the whole gate decoration.
    Omitting it means every high-risk call is refused.
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
            session_id=session.session_id,
        )

    # The id the storefront must mint against. Server-side only: it is
    # deliberately NOT in the event stream, which reaches a browser.
    state["session_id"] = session.session_id
    return state
