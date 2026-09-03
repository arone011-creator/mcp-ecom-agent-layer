# tests/test_agent_loop.py
#
# One turn: utterance in, tool calls out, answer back. The model is stubbed
# -- what is under test is the loop's own behaviour (does it execute what
# the model asked for, feed results back in the shape the API wants, stop
# when the model stops, and refuse an identity argument), not the model's
# judgement. That is what the eval harness is for.

import json

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
# DRIVEN THROUGH THE REAL SDK, OVER A FAKE NETWORK. The first version of
# these tests replaced the whole client with a stand-in, which meant the
# SDK's own request validation never ran -- and the SDK is exactly what
# refused the call in production:
#
#   ValueError: `search_products` is not strict.
#               Only `strict` function tools can be auto-parsed
#
# raised by client.chat.completions.stream() on the first tool, before a
# token was read. Four green tests said the streaming call worked. They
# were testing my idea of the SDK, not the SDK.
#
# So the seam is now the transport: httpx.MockTransport serves the wire
# format, and everything above it -- validation, chunk parsing, tool-call
# assembly, usage -- is the real library.

import httpx  # noqa: E402

from agent.tools import to_openai_tool  # noqa: E402


def sse(**fields) -> str:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1",
        "choices": [],
    }
    body.update(fields)
    return "data: " + json.dumps(body) + "\n\n"


def prose(text: str, first: bool = False) -> str:
    delta = {"content": text}
    # The real API names the role once, in the first chunk only.
    if first:
        delta["role"] = "assistant"
    return sse(choices=[{"index": 0, "delta": delta, "finish_reason": None}])


def tool_fragment(call_id=None, name=None, arguments=None) -> str:
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments

    call = {"index": 0, "type": "function", "function": function}
    if call_id:
        call["id"] = call_id

    return sse(
        choices=[
            {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}
        ]
    )


def usage_frame(total=15) -> str:
    return sse(
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": total}
    )


DONE = "data: [DONE]\n\n"


# The production tool shape, built by the production translator. It has no
# `strict` key -- to_openai_tool re-nests the MCP server's schema and
# deliberately does not rewrite it -- which is precisely what the parsing
# helper rejected. Passing these through the request is the regression.
def real_tools():
    from mcp.types import Tool

    return [
        to_openai_tool(
            Tool(
                name="search_products",
                description="Search the catalogue",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        )
    ]


def openai_over(monkeypatch, wire: str):
    """Point the real SDK at a fake network serving `wire`."""
    import agent.loop as loop_module
    from openai import AsyncOpenAI

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=wire.encode(),
        )

    def client():
        return AsyncOpenAI(
            api_key="test-key-not-a-real-one",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(loop_module, "_openai_client", client)
    return seen


async def test_the_production_tool_schemas_are_accepted_by_the_streaming_call(
    monkeypatch,
):
    # THE REGRESSION. This exact call raised ValueError in production on
    # the parsing helper, so the whole turn died before the first token
    # and the customer saw an empty panel.
    from agent.loop import openai_model_call

    seen = openai_over(monkeypatch, prose("Found some.", first=True) + DONE)

    message = await openai_model_call()([{"role": "user", "content": "shoes"}], real_tools())

    assert message.content == "Found some."
    # And the tools really did travel, rather than being dropped on the
    # way and making the acceptance meaningless.
    assert seen["body"]["tools"][0]["function"]["name"] == "search_products"
    assert "strict" not in seen["body"]["tools"][0]["function"]


async def test_the_model_call_hands_over_fragments_as_they_arrive(monkeypatch):
    from agent.loop import openai_model_call

    openai_over(
        monkeypatch,
        prose("Your ", first=True) + prose("order ") + prose("is ORD-1.") + DONE,
    )

    seen = []
    message = await openai_model_call(on_delta=seen.append)([], real_tools())

    assert len(seen) > 1, "arrived in one lump, which is the bug this replaced"
    assert "".join(seen) == "Your order is ORD-1."
    assert message.content == "Your order is ORD-1."


async def test_a_tool_call_split_across_chunks_is_reassembled(monkeypatch):
    # Arguments arrive a few characters at a time. Assembling them is the
    # accumulator's job, and getting it wrong would break every tool call
    # rather than merely the display.
    from agent.loop import openai_model_call

    openai_over(
        monkeypatch,
        tool_fragment(call_id="call_1", name="get_orders", arguments='{"lim')
        + tool_fragment(arguments='it":3}')
        + sse(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        + DONE,
    )

    message = await openai_model_call()([], real_tools())

    assert [(c.id, c.function.name, c.function.arguments) for c in message.tool_calls] == [
        ("call_1", "get_orders", '{"limit":3}')
    ]


async def test_the_assembled_message_says_it_is_from_the_assistant(monkeypatch):
    # The loop feeds this message straight back to the API. A role field
    # the accumulator built wrong would be rejected on the next request,
    # one step later and nowhere near the cause.
    from agent.loop import openai_model_call

    openai_over(monkeypatch, prose("hi", first=True) + prose(" there") + DONE)

    message = await openai_model_call()([], real_tools())

    assert message.model_dump(exclude_none=True)["role"] == "assistant"


async def test_a_fragment_is_redacted_before_anyone_sees_it(monkeypatch):
    # The streaming half of the URL provenance guard, asserted where it
    # actually runs. The finished-answer redaction in call_model cannot
    # help here: by the time it runs, the fragment has been read.
    from agent.loop import openai_model_call

    openai_over(
        monkeypatch,
        prose("Visit ", first=True)
        + prose("https://evil.example.com/x")
        + prose(" now.")
        + DONE,
    )

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"d": "<untrusted-user-content>https://evil.example.com/x'
            '</untrusted-user-content>"}',
        }
    ]

    seen = []
    await openai_model_call(on_delta=seen.append)(messages, real_tools())

    assert "evil.example.com" not in "".join(seen)


async def test_usage_is_still_reported_when_the_response_is_streamed(monkeypatch):
    # A streamed request omits usage unless it is asked for. Losing it
    # would not break a turn -- it would silently zero the eval harness's
    # cost column, which is the kind of failure nobody notices.
    from agent.loop import openai_model_call

    seen_request = openai_over(
        monkeypatch, prose("hi", first=True) + usage_frame(42) + DONE
    )

    reported = []
    await openai_model_call(on_usage=reported.append)([], real_tools())

    assert [u.total_tokens for u in reported] == [42]
    assert seen_request["body"]["stream_options"] == {"include_usage": True}


async def test_a_caller_that_wants_no_fragments_still_gets_its_answer(monkeypatch):
    # The eval harness and the tests pass no on_delta. Streaming must not
    # become mandatory just because the chat UI wants it.
    from agent.loop import openai_model_call

    openai_over(monkeypatch, prose("ab", first=True) + DONE)

    assert (await openai_model_call()([], real_tools())).content == "ab"


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
