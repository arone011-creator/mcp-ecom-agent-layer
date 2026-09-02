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


def _next_seq(state: TurnState) -> int:
    """The next sequence number, derived from what has already been emitted.

    LangGraph nodes return only their additions, so a counter held in a
    node would restart on the next pass through it. The accumulated
    length is the one number both nodes can agree on without sharing
    state of their own.
    """
    return len(state.get("events", []))


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

        # A tool turn has no prose, and an empty message event would be a
        # blank bubble in the chat.
        events = [message_event(_next_seq(state), message.content)] if message.content else []

        return {"messages": [dumped], "answer": message.content, "events": events}

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

        for call_id, name, arguments in parsed:
            if name not in HIGH_RISK_TOOLS:
                continue

            events.append(approval_required_event(seq, call_id, name, arguments))
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
                # Emitted even though nothing ran: a start without a
                # completion is a chip that spins forever.
                events.append(tool_completed(seq, call_id, name, error=refusal))
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
                events.append(tool_completed(seq, call_id, name, error=str(failure)))
                seq += 1
                continue

            results.append(_tool_message(call_id, result))
            events.append(tool_completed(seq, call_id, name, result=result))
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
    max_steps: int = 25,
    approve: ApprovalCallback | None = None,
    approval_timeout_seconds: float = 300.0,
) -> TurnState:
    """One turn, start to finish, pausing for approval where required."""
    app = build_graph(model_call, execute_tool, checkpointer=InMemorySaver())

    settings = {
        # A confused agent stops rather than looping. The repeat guard
        # already blocks the obvious case; this bounds the rest, and is
        # the first half of the cost ceiling Decision D calls for.
        "recursion_limit": max_steps,
        # One thread per turn. Nothing outlives a turn today; Phase 3 is
        # where a thread becomes a conversation.
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
        config=settings,
    )

    while state.get("__interrupt__"):
        requests = [item.value for item in state["__interrupt__"]]

        if approve is None:
            # The safe default. A turn with no way to reach a human must
            # not proceed as though one had answered.
            decisions = [{"approved": False} for _ in requests]
        else:
            decisions = await _decide(approve, requests, approval_timeout_seconds)

        state = await app.ainvoke(
            Command(resume=decisions[0] if len(decisions) == 1 else decisions),
            config=settings,
        )

    return state


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


async def answer(utterance: str, token: str, *, model: str | None = None) -> TurnState:
    """The whole thing wired to the real model and the real MCP server.

    One session for the turn, because an approval pause happens in the
    middle of one and the token is minted against the session id.
    """
    tools = await list_openai_tools(token, only=AGENT_TOOLS)

    async with session_scoped_executor(token) as session:
        state = await run_turn(
            utterance,
            model_call=openai_model_call(model),
            execute_tool=session.execute,
            tools=tools,
        )

    # The id the storefront must mint against. Server-side only: it is
    # deliberately NOT in the event stream, which reaches a browser.
    state["session_id"] = session.session_id
    return state
