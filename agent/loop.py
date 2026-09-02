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
    """The real tool execution, one MCP session per call."""
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
