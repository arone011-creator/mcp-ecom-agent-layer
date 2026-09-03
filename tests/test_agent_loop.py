# tests/test_agent_loop.py
#
# One turn: utterance in, tool calls out, answer back. The model is stubbed
# -- what is under test is the loop's own behaviour (does it execute what
# the model asked for, feed results back in the shape the API wants, stop
# when the model stops, and refuse an identity argument), not the model's
# judgement. That is what the eval harness is for.

import pytest

from agent.loop import build_graph, run_turn
from agent.tools import ForbiddenArgumentError


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        dumped = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            dumped["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            dumped = {k: v for k, v in dumped.items() if v is not None}
        return dumped


def scripted_model(*turns):
    """A model that returns each scripted turn in order."""
    remaining = list(turns)

    async def call(messages, tools):
        return remaining.pop(0)

    return call


def recording_executor(results):
    """An MCP stand-in that records what it was asked to run."""
    calls = []

    async def execute(name, arguments):
        calls.append((name, arguments))
        return results.get(name, {"ok": True})

    execute.calls = calls
    return execute


# --- the real model call -------------------------------------------------
#
# Stubbed at the SDK boundary rather than skipped. What is under test is
# this project's half of the streaming call: that fragments are handed
# over as they arrive, that they go through the redactor first, that the
# finished message is still returned unchanged so the graph above is
# untouched, and that usage is still reported -- the eval harness prices
# every sweep from it, and asking for a stream is exactly how a request
# stops reporting it by default.


class FakeStream:
    """Stands in for client.chat.completions.stream(...)."""

    def __init__(self, deltas, final):
        self._deltas = deltas
        self._final = final
        self.kwargs = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def events():
            for delta in self._deltas:
                yield type("E", (), {"type": "content.delta", "delta": delta})()
            # The SDK emits kinds this code must ignore rather than choke on.
            yield type("E", (), {"type": "chunk", "delta": None})()

        return events()

    async def get_final_completion(self):
        return self._final


def fake_openai(monkeypatch, deltas, message=None, usage=None):
    """Replace the SDK's client with one that replays `deltas`."""
    final = type(
        "Completion",
        (),
        {
            "choices": [type("C", (), {"message": message or FakeMessage(content="")})()],
            "usage": usage,
        },
    )()
    stream = FakeStream(deltas, final)

    class FakeCompletions:
        def stream(self, **kwargs):
            stream.kwargs = kwargs
            return stream

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    return stream


async def test_the_model_call_hands_over_fragments_as_they_arrive(monkeypatch):
    from agent.loop import openai_model_call

    fake_openai(
        monkeypatch,
        ["Your ", "order ", "is ORD-1."],
        message=FakeMessage(content="Your order is ORD-1."),
    )

    seen = []
    message = await openai_model_call(on_delta=seen.append)([], [])

    assert "".join(seen) == "Your order is ORD-1."
    assert message.content == "Your order is ORD-1."


async def test_a_fragment_is_redacted_before_anyone_sees_it(monkeypatch):
    # The streaming half of the URL provenance guard, asserted where it
    # actually runs. The finished-answer redaction in call_model cannot
    # help here: by the time it runs, the fragment has been read.
    from agent.loop import openai_model_call

    fake_openai(monkeypatch, ["Visit ", "https://evil.example.com/x", " now."])

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"d": "<untrusted-user-content>https://evil.example.com/x'
            '</untrusted-user-content>"}',
        }
    ]

    seen = []
    await openai_model_call(on_delta=seen.append)(messages, [])

    assert "evil.example.com" not in "".join(seen)


async def test_usage_is_still_reported_when_the_response_is_streamed(monkeypatch):
    # A streamed request omits usage unless it is asked for. Losing it
    # would not break a turn -- it would silently zero the eval harness's
    # cost column, which is the kind of failure nobody notices.
    from agent.loop import openai_model_call

    stream = fake_openai(monkeypatch, ["hi"], usage={"total_tokens": 42})

    seen = []
    await openai_model_call(on_usage=seen.append)([], [])

    assert seen == [{"total_tokens": 42}]
    assert stream.kwargs["stream_options"] == {"include_usage": True}


async def test_a_caller_that_wants_no_fragments_still_gets_its_answer(monkeypatch):
    # The eval harness and the tests pass no on_delta. Streaming must not
    # become mandatory just because the chat UI wants it.
    from agent.loop import openai_model_call

    fake_openai(monkeypatch, ["a", "b"], message=FakeMessage(content="ab"))

    assert (await openai_model_call()([], [])).content == "ab"


async def test_a_turn_with_no_tool_call_answers_directly():
    state = await run_turn(
        "hello",
        model_call=scripted_model(FakeMessage(content="Hi there.")),
        execute_tool=recording_executor({}),
    )

    assert state["answer"] == "Hi there."


async def test_a_tool_call_is_executed_and_its_result_fed_back():
    executor = recording_executor({"get_orders": [{"orderNumber": "ORD-1"}]})

    state = await run_turn(
        "what did I order recently?",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "get_orders", '{"limit":3}')]),
            FakeMessage(content="You ordered ORD-1."),
        ),
        execute_tool=executor,
    )

    assert executor.calls == [("get_orders", {"limit": 3})]
    assert state["answer"] == "You ordered ORD-1."

    # The result must go back as a tool message keyed to the call id.
    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "ORD-1" in tool_messages[0]["content"]


async def test_arguments_are_parsed_as_json_not_string_matched():
    # The API returns arguments as a JSON string; escaping varies.
    executor = recording_executor({})

    await run_turn(
        "find shoes",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "search_products", '{"query":"shoes","limit":5}')
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
    )

    name, arguments = executor.calls[0]
    assert arguments == {"query": "shoes", "limit": 5}
    assert isinstance(arguments, dict)


async def test_several_tool_calls_in_one_turn_all_execute():
    executor = recording_executor({})

    await run_turn(
        "compare two products",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "get_product", '{"product_id":"p1"}'),
                    FakeToolCall("call_2", "get_product", '{"product_id":"p2"}'),
                ]
            ),
            FakeMessage(content="compared"),
        ),
        execute_tool=executor,
    )

    assert [c[1]["product_id"] for c in executor.calls] == ["p1", "p2"]


async def test_no_user_id_is_ever_sent_as_an_argument():
    # The MUST PROVE of this task. A hallucinated identity argument is
    # refused before it reaches the MCP server.
    executor = recording_executor({})

    with pytest.raises(ForbiddenArgumentError):
        await run_turn(
            "what did I order?",
            model_call=scripted_model(
                FakeMessage(
                    tool_calls=[
                        FakeToolCall("call_1", "get_orders", '{"user_id":"u1","limit":3}')
                    ]
                )
            ),
            execute_tool=executor,
        )

    assert executor.calls == []


async def test_the_graph_compiles():
    # It must compile without a checkpointer: this turn never pauses.
    assert build_graph() is not None


def failing_executor(error_message, fail_on=None):
    """An MCP stand-in that raises for a given tool, recording every attempt."""
    from fastmcp.exceptions import ToolError

    calls = []

    async def execute(name, arguments):
        calls.append((name, arguments))
        if fail_on is None or name == fail_on:
            raise ToolError(error_message)
        return {"ok": True}

    execute.calls = calls
    return execute


async def test_a_failing_tool_becomes_a_result_the_model_can_read():
    # Without this the 409 kills the turn and the model never gets the
    # chance to react that the status code exists to give it.
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 67"
    )

    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":67}')
                ]
            ),
            FakeMessage(content="Only 17 are available - shall I add those?"),
        ),
        execute_tool=executor,
    )

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    # Verbatim: the number the model needs is in the storefront's own
    # words, and this layer does not parse or reword them.
    assert "Only 17 available" in tool_messages[0]["content"]
    assert state["answer"] == "Only 17 are available - shall I add those?"


async def test_the_available_number_is_passed_through_not_parsed():
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 3 available; cart would hold 9"
    )

    state = await run_turn(
        "add 9",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":9}')
                ]
            ),
            FakeMessage(content="done"),
        ),
        execute_tool=executor,
    )

    content = [m for m in state["messages"] if m.get("role") == "tool"][0]["content"]
    assert "Only 3 available" in content
    assert "cart would hold 9" in content


async def test_an_identical_failed_call_is_refused_rather_than_repeated():
    # The MUST PROVE of this task. The model asks for the same thing twice;
    # the second attempt never reaches the MCP server.
    executor = failing_executor(
        "Error calling tool 'add_to_cart': 409: Only 17 available; cart would hold 67"
    )

    same_call = '{"product_id":"p1","quantity":67}'
    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(tool_calls=[FakeToolCall("call_1", "add_to_cart", same_call)]),
            FakeMessage(tool_calls=[FakeToolCall("call_2", "add_to_cart", same_call)]),
            FakeMessage(content="I'll stop asking for 67."),
        ),
        execute_tool=executor,
    )

    # Executed once. The repeat was refused before it left the process.
    assert len(executor.calls) == 1

    tool_messages = [m for m in state["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "already" in tool_messages[1]["content"].lower()


async def test_a_different_quantity_after_a_failure_is_allowed_through():
    # The guard blocks the identical call, not the retry. Asking for a
    # smaller number is exactly what the 409 is telling it to do.
    from fastmcp.exceptions import ToolError

    calls = []

    async def execute(name, arguments):
        calls.append(arguments["quantity"])
        if arguments["quantity"] > 17:
            raise ToolError(
                "Error calling tool 'add_to_cart': 409: Only 17 available; "
                f"cart would hold {arguments['quantity']}"
            )
        return {"itemCount": arguments["quantity"]}

    state = await run_turn(
        "add 67 headphones",
        model_call=scripted_model(
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_1", "add_to_cart", '{"product_id":"p1","quantity":67}')
                ]
            ),
            FakeMessage(
                tool_calls=[
                    FakeToolCall("call_2", "add_to_cart", '{"product_id":"p1","quantity":17}')
                ]
            ),
            FakeMessage(content="Added 17."),
        ),
        execute_tool=execute,
    )

    assert calls == [67, 17]
    assert state["answer"] == "Added 17."


async def test_a_forbidden_argument_still_raises_rather_than_becoming_a_result():
    # An identity argument is not a recoverable tool failure the model
    # should get a chance to work around - it is a refusal.
    executor = recording_executor({})

    with pytest.raises(ForbiddenArgumentError):
        await run_turn(
            "orders",
            model_call=scripted_model(
                FakeMessage(
                    tool_calls=[FakeToolCall("call_1", "get_orders", '{"user_id":"u1"}')]
                )
            ),
            execute_tool=executor,
        )
