"""Does interrupt() survive being raised inside a subgraph?

THE WHOLE MULTI-AGENT DESIGN RESTS ON THIS. Specialists run as subgraphs
invoked from a parent node, and cancel_order pauses for a human with
interrupt(). If the pause does not reach the parent's stream -- and the
resume does not reach back down -- the approval boundary is gone, and it
is gone silently: the cancel just never pauses.

Kept in the suite rather than thrown away, so a LangGraph upgrade that
changes this fails here instead of in front of a customer.
"""

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def _child_graph(checkpointer):
    """A subgraph that stops and asks."""

    async def ask(state: State) -> dict:
        decision = interrupt({"question": "may I?"})
        return {"log": [f"child-resumed:{decision}"]}

    graph = StateGraph(State)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_interrupt_from_a_subgraph_reaches_the_parent_stream():
    saver = InMemorySaver()
    child = _child_graph(saver)
    side_effects = []

    async def parent_node(state: State) -> dict:
        # Deliberately BEFORE the subgraph call: LangGraph re-runs a node
        # from the top on resume, so this records how many times that
        # happened. The real delegate node must be safe under a re-run.
        side_effects.append("parent-ran")
        result = await child.ainvoke({"log": []})
        return {"log": ["parent-done", *result["log"]]}

    graph = StateGraph(State)
    graph.add_node("parent", parent_node)
    graph.add_edge(START, "parent")
    graph.add_edge("parent", END)
    app = graph.compile(checkpointer=saver)

    settings = {"configurable": {"thread_id": "spike-1"}}

    state = None
    async for step in app.astream({"log": []}, config=settings, stream_mode="values"):
        state = step

    # THE PAUSE REACHED THE TOP.
    assert state.get("__interrupt__"), "the subgraph's interrupt never surfaced"
    assert state["__interrupt__"][0].value == {"question": "may I?"}

    # THE RESUME REACHED BACK DOWN.
    async for step in app.astream(
        Command(resume="yes"), config=settings, stream_mode="values"
    ):
        state = step

    assert "child-resumed:yes" in state["log"]
    assert "parent-done" in state["log"]


@pytest.mark.asyncio
async def test_a_resumed_parent_node_re_runs_from_the_top():
    """Pins the re-run, because the delegate node must tolerate it.

    Whatever this number is, the delegate node must produce the same
    events and the same sequence numbers on both passes -- which is why
    it derives them from state rather than from a counter it holds.
    """
    saver = InMemorySaver()
    child = _child_graph(saver)
    runs = []

    async def parent_node(state: State) -> dict:
        runs.append(1)
        result = await child.ainvoke({"log": []})
        return {"log": ["parent-done", *result["log"]]}

    graph = StateGraph(State)
    graph.add_node("parent", parent_node)
    graph.add_edge(START, "parent")
    graph.add_edge("parent", END)
    app = graph.compile(checkpointer=saver)
    settings = {"configurable": {"thread_id": "spike-2"}}

    async for _ in app.astream({"log": []}, config=settings, stream_mode="values"):
        pass
    async for _ in app.astream(
        Command(resume="yes"), config=settings, stream_mode="values"
    ):
        pass

    # Two passes: once to the pause, once after it.
    assert len(runs) == 2
