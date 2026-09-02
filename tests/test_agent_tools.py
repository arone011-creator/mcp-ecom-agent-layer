# tests/test_agent_tools.py
#
# The MCP server advertises tools; Claude needs them in its own shape. The
# rename is real (inputSchema -> input_schema), and two things must not pass
# through unexamined: a tool this agent does not know about, and
# cancel_order's approval_token, which only the storefront may supply.

import pytest
from mcp.types import Tool

from agent.tools import (
    FORBIDDEN_ARGUMENTS,
    KNOWN_TOOLS,
    READ_ONLY_TOOLS,
    ForbiddenArgumentError,
    UnknownToolError,
    build_transport,
    reject_forbidden_arguments,
    to_openai_tool,
    translate_tools,
)


def mcp_tool(name: str, schema: dict | None = None, description: str | None = "d") -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=schema if schema is not None else {"properties": {}, "type": "object"},
    )


def all_nine() -> list[Tool]:
    return [mcp_tool(name) for name in sorted(KNOWN_TOOLS)]


def test_the_nine_known_tools_are_the_nine_the_server_advertises():
    # A tool surface that silently shrinks is how a capability disappears
    # without anyone noticing.
    assert len(KNOWN_TOOLS) == 9


def test_translation_nests_the_schema_the_way_openai_wants_it():
    # Verified against a live gpt-4.1 call: the MCP schema is accepted as
    # `parameters` unchanged, anyOf/default and all. This is a re-nesting,
    # not a schema rewrite.
    schema = {
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
        "type": "object",
    }

    translated = to_openai_tool(mcp_tool("get_product", schema))

    assert translated == {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "d",
            "parameters": schema,
        },
    }
    assert "inputSchema" not in translated["function"]


def test_a_missing_description_becomes_empty_rather_than_none():
    translated = to_openai_tool(mcp_tool("get_cart", description=None))

    assert translated["function"]["description"] == ""


def test_all_nine_translate():
    translated = translate_tools(all_nine())

    assert len(translated) == 9
    assert {t["function"]["name"] for t in translated} == KNOWN_TOOLS


def test_an_unknown_tool_is_refused_rather_than_passed_through():
    # The MCP server is a separate deployment. If it ever advertises
    # something this agent was not built for, that is a refusal, not a
    # capability the model silently gains.
    tools = all_nine() + [mcp_tool("wire_money")]

    with pytest.raises(UnknownToolError) as caught:
        translate_tools(tools)

    assert "wire_money" in str(caught.value)


def test_a_missing_tool_is_refused_too():
    tools = [t for t in all_nine() if t.name != "cancel_order"]

    with pytest.raises(UnknownToolError):
        translate_tools(tools)


def test_cancel_order_does_not_advertise_approval_token_to_the_model():
    # The storefront injects the approval token after a human clicks. A
    # model that can see the field is a model that can invent a value for
    # it; the server would reject a forged one, but the field has no
    # business being in the model's schema at all.
    schema = {
        "properties": {
            "order_id": {"type": "string"},
            "approval_token": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["order_id"],
        "type": "object",
    }

    translated = to_openai_tool(mcp_tool("cancel_order", schema))

    params = translated["function"]["parameters"]
    assert "approval_token" not in params["properties"]
    assert "order_id" in params["properties"]
    assert params["required"] == ["order_id"]


def test_stripping_does_not_mutate_the_caller_s_schema():
    schema = {
        "properties": {"order_id": {"type": "string"}, "approval_token": {"type": "string"}},
        "type": "object",
    }

    to_openai_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" in schema["properties"]


def test_other_tools_keep_every_property():
    schema = {
        "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
        "required": ["product_id", "quantity"],
        "type": "object",
    }

    translated = to_openai_tool(mcp_tool("add_to_cart", schema))

    assert set(translated["function"]["parameters"]["properties"]) == {
        "product_id",
        "quantity",
    }


def test_the_bearer_token_travels_on_every_call():
    transport = build_transport("https://mcp.test/mcp", "tok-123")

    assert transport.headers["authorization"] == "Bearer tok-123"


def test_a_blank_token_is_refused_rather_than_sent_empty():
    # An empty bearer is worse than no bearer: it looks like a credential
    # and authenticates nobody.
    with pytest.raises(ValueError):
        build_transport("https://mcp.test/mcp", "")


def test_the_read_only_surface_is_the_six_low_risk_tools():
    # Task 2 wires only the tools that cannot change anything. add_to_cart
    # and remove_from_cart are Medium; cancel_order is High and needs an
    # approval this task does not build.
    assert READ_ONLY_TOOLS == {
        "search_products",
        "get_product",
        "check_inventory",
        "get_orders",
        "get_order",
        "get_cart",
    }
    assert READ_ONLY_TOOLS < KNOWN_TOOLS


def test_translate_can_narrow_to_the_read_only_surface():
    translated = translate_tools(all_nine(), only=READ_ONLY_TOOLS)

    assert {t["function"]["name"] for t in translated} == READ_ONLY_TOOLS


def test_a_user_id_argument_is_refused():
    # Identity comes from the bearer token, resolved by the API's own
    # whoami. A user id in the arguments is the model asserting who the
    # caller is, which is the one thing it must never do -- and a model
    # can hallucinate a key that was never in the schema.
    for key in FORBIDDEN_ARGUMENTS:
        with pytest.raises(ForbiddenArgumentError):
            reject_forbidden_arguments("get_orders", {key: "u1"})


def test_ordinary_arguments_pass_the_guard():
    reject_forbidden_arguments("get_order", {"order_id": "o1"})
    reject_forbidden_arguments("search_products", {"query": "shoes", "limit": 5})


def test_the_guard_names_the_offending_key():
    with pytest.raises(ForbiddenArgumentError) as caught:
        reject_forbidden_arguments("get_orders", {"user_id": "u1", "limit": 5})

    assert "user_id" in str(caught.value)


def test_the_medium_risk_surface_is_the_two_cart_writes():
    from agent.tools import MEDIUM_RISK_TOOLS

    assert MEDIUM_RISK_TOOLS == {"add_to_cart", "remove_from_cart"}


def test_the_agent_surface_is_read_only_plus_medium_but_not_cancel():
    # cancel_order is High risk and needs the approval machinery that
    # arrives in Task 5. Until then the agent is not offered it at all.
    from agent.tools import AGENT_TOOLS

    assert AGENT_TOOLS == READ_ONLY_TOOLS | {"add_to_cart", "remove_from_cart"}
    assert "cancel_order" not in AGENT_TOOLS
    assert AGENT_TOOLS < KNOWN_TOOLS


def test_the_agent_surface_translates_to_eight_tools():
    from agent.tools import AGENT_TOOLS

    translated = translate_tools(all_nine(), only=AGENT_TOOLS)

    assert len(translated) == 8
    assert "cancel_order" not in {t["function"]["name"] for t in translated}
