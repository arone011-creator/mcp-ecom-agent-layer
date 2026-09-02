# tests/test_agent_tools.py
#
# The MCP server advertises tools; Claude needs them in its own shape. The
# rename is real (inputSchema -> input_schema), and two things must not pass
# through unexamined: a tool this agent does not know about, and
# cancel_order's approval_token, which only the storefront may supply.

import pytest
from mcp.types import Tool

from agent.tools import (
    KNOWN_TOOLS,
    UnknownToolError,
    build_transport,
    to_claude_tool,
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


def test_translation_renames_the_schema_key_to_claude_spelling():
    schema = {
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
        "type": "object",
    }

    translated = to_claude_tool(mcp_tool("get_product", schema))

    assert translated == {
        "name": "get_product",
        "description": "d",
        "input_schema": schema,
    }
    assert "inputSchema" not in translated


def test_a_missing_description_becomes_empty_rather_than_none():
    translated = to_claude_tool(mcp_tool("get_cart", description=None))

    assert translated["description"] == ""


def test_all_nine_translate():
    translated = translate_tools(all_nine())

    assert len(translated) == 9
    assert {t["name"] for t in translated} == KNOWN_TOOLS


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

    translated = to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" not in translated["input_schema"]["properties"]
    assert "order_id" in translated["input_schema"]["properties"]
    assert translated["input_schema"]["required"] == ["order_id"]


def test_stripping_does_not_mutate_the_caller_s_schema():
    schema = {
        "properties": {"order_id": {"type": "string"}, "approval_token": {"type": "string"}},
        "type": "object",
    }

    to_claude_tool(mcp_tool("cancel_order", schema))

    assert "approval_token" in schema["properties"]


def test_other_tools_keep_every_property():
    schema = {
        "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
        "required": ["product_id", "quantity"],
        "type": "object",
    }

    translated = to_claude_tool(mcp_tool("add_to_cart", schema))

    assert set(translated["input_schema"]["properties"]) == {"product_id", "quantity"}


def test_the_bearer_token_travels_on_every_call():
    transport = build_transport("https://mcp.test/mcp", "tok-123")

    assert transport.headers["authorization"] == "Bearer tok-123"


def test_a_blank_token_is_refused_rather_than_sent_empty():
    # An empty bearer is worse than no bearer: it looks like a credential
    # and authenticates nobody.
    with pytest.raises(ValueError):
        build_transport("https://mcp.test/mcp", "")
