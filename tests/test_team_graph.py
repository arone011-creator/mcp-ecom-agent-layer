"""The supervisor delegating to specialists.

model_call and execute_tool are injected, as everywhere else in this
codebase: what is worth testing is the delegation's own behaviour -- does
the right specialist run, does its answer come back as a tool result, do
the events come out in one order -- not the model's judgement.
"""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.team_graph import build_team_graph


def _assistant(tool_name: str, request: str):
    """One assistant message asking for a delegation."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"request": request}),
                },
            }
        ],
    }


class _Message:
    """The shape openai_model_call returns: something with .model_dump()."""

    def __init__(self, payload):
        self._payload = payload
        self.content = payload.get("content")

    def model_dump(self, exclude_none=False):
        return {
            key: value
            for key, value in self._payload.items()
            if not exclude_none or value is not None
        }


def _scripted(replies):
    """A model that says the next scripted thing each time it is called."""
    remaining = list(replies)

    async def model_call(messages, tools):
        return _Message(remaining.pop(0))

    return model_call


@pytest.mark.asyncio
async def test_a_delegation_runs_the_named_specialist():
    ran = []

    async def execute_tool(name, arguments):
        ran.append(name)
        return {"products": [{"id": "p1", "name": "Laptop"}]}

    # Supervisor delegates; product specialist searches then answers;
    # supervisor writes the final reply.
    supervisor = _scripted(
        [
            _assistant("ask_product", "find laptops"),
            {"role": "assistant", "content": "We have a Laptop."},
        ]
    )
    specialist = _scripted(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_s1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": json.dumps({"query": "laptops"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Found one laptop."},
        ]
    )

    async def model_call(messages, tools):
        # The supervisor's toolbox is delegation tools; a specialist's is
        # shop tools. That is how this stand-in knows which is calling.
        names = {tool["function"]["name"] for tool in tools}
        if any(name.startswith("ask_") for name in names):
            return await supervisor(messages, tools)
        return await specialist(messages, tools)

    app = build_team_graph(model_call, execute_tool, checkpointer=InMemorySaver())

    state = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": "find laptops"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_product",
                        "description": "products",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "specialist_tools": {
                "product": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "description": "search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
            "events": [],
            "failed": [],
            "seeded": 1,
        },
        config={"configurable": {"thread_id": "t1"}},
    )

    assert ran == ["search_products"]
    assert state["answer"] == "We have a Laptop."


@pytest.mark.asyncio
async def test_the_specialists_answer_comes_back_as_a_tool_result():
    """Not as prose the supervisor has to parse.

    The supervisor sees the specialist's answer the same way it would see
    any tool result, which is what keeps the supervisor's transcript the
    ordinary shape the storefront already stores.
    """

    async def execute_tool(name, arguments):
        return {"cart": {"items": []}}

    supervisor = _scripted(
        [
            _assistant("ask_cart", "what is in the cart"),
            {"role": "assistant", "content": "Your cart is empty."},
        ]
    )
    specialist = _scripted([{"role": "assistant", "content": "The cart is empty."}])

    async def model_call(messages, tools):
        names = {tool["function"]["name"] for tool in tools}
        if any(name.startswith("ask_") for name in names):
            return await supervisor(messages, tools)
        return await specialist(messages, tools)

    app = build_team_graph(model_call, execute_tool, checkpointer=InMemorySaver())
    state = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": "what is in my cart"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_cart",
                        "description": "cart",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "events": [],
            "failed": [],
            "seeded": 1,
        },
        config={"configurable": {"thread_id": "t2"}},
    )

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert tool_messages, "the specialist's answer never reached the supervisor"
    assert "empty" in json.loads(tool_messages[-1]["content"])["answer"]


@pytest.mark.asyncio
async def test_the_events_come_out_in_one_unbroken_sequence():
    """THE MUST PROVE for numbering.

    The supervisor and the specialist keep separate states with separate
    event lists, and the storefront orders the whole transcript by seq.
    Duplicated or out-of-order numbers would scramble the chat.
    """

    async def execute_tool(name, arguments):
        return {"products": []}

    supervisor = _scripted(
        [
            _assistant("ask_product", "find laptops"),
            {"role": "assistant", "content": "Nothing found."},
        ]
    )
    specialist = _scripted(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_s1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Nothing found."},
        ]
    )

    async def model_call(messages, tools):
        names = {tool["function"]["name"] for tool in tools}
        if any(name.startswith("ask_") for name in names):
            return await supervisor(messages, tools)
        return await specialist(messages, tools)

    app = build_team_graph(model_call, execute_tool, checkpointer=InMemorySaver())
    state = await app.ainvoke(
        {
            "messages": [{"role": "user", "content": "find laptops"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_product",
                        "description": "products",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "specialist_tools": {
                "product": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "description": "search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
            "events": [],
            "failed": [],
            "seeded": 1,
        },
        config={"configurable": {"thread_id": "t3"}},
    )

    numbers = [event["seq"] for event in state["events"]]

    assert numbers == sorted(numbers), f"out of order: {numbers}"
    assert len(numbers) == len(set(numbers)), f"duplicates: {numbers}"
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), (
        f"gaps: {numbers}"
    )
