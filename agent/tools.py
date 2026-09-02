"""The MCP tool surface, translated into what Claude expects.

Two shapes for the same nine capabilities. MCP advertises `inputSchema`;
OpenAI's tool definition wants it nested as `function.parameters`. The
re-nesting is most of the work, and the things that are NOT a re-nesting
are the point of this module:

  - an unknown tool is refused. The MCP server is a separate deployment
    that could grow a tool this agent was never built for, and a
    capability the model gains silently is one nobody reviewed.

  - cancel_order's approval_token is stripped before the model ever sees
    it. The storefront mints that token after a human clicks and code
    injects it at call time; a field the model cannot see is a field it
    cannot invent a value for.
"""

from typing import Any, Iterable

from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import Tool
from openai.types.chat import ChatCompletionToolParam

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


# The tools that cannot change anything. Task 2 of the M4 plan wires only
# these; Medium-risk cart writes and the High-risk cancellation arrive with
# the approval machinery that guards them.
READ_ONLY_TOOLS = frozenset(
    {
        "search_products",
        "get_product",
        "check_inventory",
        "get_orders",
        "get_order",
        "get_cart",
    }
)

# Identity is never an argument. It is resolved from the bearer token by
# the API's own whoami, and a model supplying one of these is the model
# asserting who the caller is. None of the nine schemas contain these --
# the guard exists because a model can invent a key that was never offered.
FORBIDDEN_ARGUMENTS = frozenset(
    {"user_id", "userId", "customer_id", "customerId", "email", "user", "customer"}
)


class UnknownToolError(Exception):
    """The MCP server's tool surface is not the one this agent expects."""


class ForbiddenArgumentError(Exception):
    """A tool call carried an argument the model has no business supplying."""


def to_openai_tool(tool: Tool) -> ChatCompletionToolParam:
    """One MCP tool as an OpenAI tool definition.

    Verified against a live call: the MCP schema is accepted as
    `parameters` unchanged. This re-nests; it does not rewrite.
    """
    schema = _without_injected_arguments(tool.name, tool.inputSchema)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            # An explicit empty string is clearer than None travelling
            # through the request builder.
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def translate_tools(
    tools: list[Tool], only: Iterable[str] | None = None
) -> list[ChatCompletionToolParam]:
    """Every advertised tool, or an error naming what did not match.

    `only` narrows the surface handed to the model without weakening the
    check: the full nine must still be advertised, so a tool going missing
    is still caught even when this turn does not use it.
    """
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

    wanted = set(only) if only is not None else KNOWN_TOOLS

    return [to_openai_tool(tool) for tool in tools if tool.name in wanted]


def reject_forbidden_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Raise if a tool call carries an identity argument.

    Called before every execution, not just the risky ones -- a read tool
    scoped to the wrong customer is the leak, not the write.
    """
    offending = sorted(set(arguments) & FORBIDDEN_ARGUMENTS)
    if offending:
        raise ForbiddenArgumentError(
            f"{name} was called with identity arguments the model may not supply: "
            f"{offending}"
        )


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


async def list_openai_tools(
    token: str, url: str | None = None, only: Iterable[str] | None = None
) -> list[ChatCompletionToolParam]:
    """Connect, list, translate. The agent's whole view of what it can do."""
    from fastmcp import Client

    transport = build_transport(url or config.MCP_SERVER_URL, token)

    async with Client(transport) as client:
        return translate_tools(await client.list_tools(), only=only)
