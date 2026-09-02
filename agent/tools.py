"""The MCP tool surface, translated into what Claude expects.

Two shapes for the same nine capabilities. MCP advertises `inputSchema`;
Claude's tool definition wants `input_schema`. The rename is most of the
work, and the two things that are NOT a rename are the point of this
module:

  - an unknown tool is refused. The MCP server is a separate deployment
    that could grow a tool this agent was never built for, and a
    capability the model gains silently is one nobody reviewed.

  - cancel_order's approval_token is stripped before the model ever sees
    it. The storefront mints that token after a human clicks and code
    injects it at call time; a field the model cannot see is a field it
    cannot invent a value for.
"""

from typing import Any

from anthropic.types import ToolParam
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool

import config

# The surface this agent was built against. Asserted, not discovered: a
# tool list that silently shrinks is how a capability disappears without
# anyone noticing, and one that silently grows is an unreviewed capability.
KNOWN_TOOLS = frozenset(
    {
        "search_products",
        "get_product",
        "check_inventory",
        "get_orders",
        "get_order",
        "get_cart",
        "add_to_cart",
        "remove_from_cart",
        "cancel_order",
    }
)

# Arguments supplied by code, never by the model. Stripped from the schema
# the model is shown, and injected at call time.
INJECTED_ARGUMENTS: dict[str, frozenset[str]] = {
    "cancel_order": frozenset({"approval_token"}),
}


class UnknownToolError(Exception):
    """The MCP server's tool surface is not the one this agent expects."""


def to_claude_tool(tool: Tool) -> ToolParam:
    """One MCP tool as a Claude tool definition."""
    schema = _without_injected_arguments(tool.name, tool.inputSchema)

    return {
        "name": tool.name,
        # Claude tolerates a missing description; an explicit empty string
        # is clearer than None travelling through the request builder.
        "description": tool.description or "",
        "input_schema": schema,
    }


def translate_tools(tools: list[Tool]) -> list[ToolParam]:
    """Every advertised tool, or an error naming what did not match."""
    advertised = {tool.name for tool in tools}

    unknown = advertised - KNOWN_TOOLS
    if unknown:
        raise UnknownToolError(
            f"MCP server advertises tools this agent does not know: {sorted(unknown)}"
        )

    missing = KNOWN_TOOLS - advertised
    if missing:
        raise UnknownToolError(
            f"MCP server is not advertising expected tools: {sorted(missing)}"
        )

    return [to_claude_tool(tool) for tool in tools]


def build_transport(url: str, token: str) -> StreamableHttpTransport:
    """The transport every MCP call rides on, carrying this caller's token.

    One transport per caller, never shared: the token IS the identity, and
    a transport held between customers is an ambient identity by another
    name -- the same rule clients/ecommerce_api.py follows.
    """
    if not token.strip():
        raise ValueError("A bearer token is required")

    return StreamableHttpTransport(url, headers={"authorization": f"Bearer {token}"})


def _without_injected_arguments(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    injected = INJECTED_ARGUMENTS.get(name)
    if not injected:
        return schema

    # Copied rather than edited: the caller's Tool object is not ours to
    # mutate, and a shared nested dict would leak the edit anyway.
    copy = dict(schema)
    copy["properties"] = {
        key: value
        for key, value in schema.get("properties", {}).items()
        if key not in injected
    }
    if "required" in schema:
        copy["required"] = [key for key in schema["required"] if key not in injected]

    return copy


async def list_claude_tools(token: str, url: str | None = None) -> list[ToolParam]:
    """Connect, list, translate. The agent's whole view of what it can do."""
    from fastmcp import Client

    transport = build_transport(url or config.MCP_SERVER_URL, token)

    async with Client(transport) as client:
        return translate_tools(await client.list_tools())
