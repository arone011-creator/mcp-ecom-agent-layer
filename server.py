"""The MCP server.

HTTP transport with per-request auth, deliberately not stdio. A stdio
process carries one ambient identity, and this is a multi-user app: every
caller would share whichever token the process started with. Over HTTP,
"the caller" is a fact about the request rather than about the process.

The tools here are thin wrappers. Everything they do lives in tools/*.py,
which is also the split Phase 3 uses to hand each file to a specialist
agent.
"""

from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

import approvals
import config
from clients.ecommerce_api import EcommerceApi
from tools import cart as cart_tools
from tools import orders as order_tools
from tools import products as product_tools

mcp = FastMCP("mcp-ecom")

# The only tools an approval can be minted for. A low-risk tool does not
# need one, and an unknown name should not be mintable at all -- otherwise
# the mint route becomes a way to probe what exists.
HIGH_RISK_TOOLS = {"cancel_order"}


class MissingCredential(Exception):
    """The request carried no usable identity."""


def api_for_headers(headers: dict[str, str]) -> EcommerceApi:
    """One client per request, carrying that request's token.

    Never cached and never shared. A client held between requests is an
    ambient identity by another name, and the bug it produces -- one user
    reading another's cart -- is exactly the one this design exists to
    make impossible.
    """
    header = headers.get("authorization", "")
    scheme, _, value = header.partition(" ")

    # Case-insensitive per RFC 7235; clients differ on capitalisation and
    # refusing "bearer" would be our bug, not theirs.
    if scheme.lower() != "bearer" or not value.strip():
        raise MissingCredential("A bearer token is required")

    return EcommerceApi(token=value.strip())


def session_id_for_headers(headers: dict[str, str]) -> str:
    """The conversation an approval is scoped to.

    Refused rather than defaulted: one shared default would put every
    caller into the same approval scope, so a token minted in someone
    else's conversation would validate in yours.
    """
    session = headers.get("mcp-session-id", "").strip()
    if not session:
        raise MissingCredential("A session id is required")
    return session


def request_headers() -> dict[str, str]:
    """This request's headers, including the ones FastMCP hides by default.

    include_all matters. The default call strips `mcp-session-id` as an
    "MCP-related header", and that header is the session identity this
    server scopes approvals to -- the transport assigns it at initialize,
    so it is a real per-connection fact rather than something a caller
    hands us. Without include_all every cancel_order fails with "a session
    id is required", while every unit test still passes, because the pure
    functions were never the problem.

    Nothing is forwarded downstream wholesale; callers pick the one or two
    headers they need.
    """
    return get_http_headers(include_all=True)


def _request_api() -> EcommerceApi:
    return api_for_headers(request_headers())


@mcp.tool
async def search_products(
    query: str = "",
    limit: int | None = None,
    page: int | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
) -> Any:
    """Search the product catalogue.

    An empty query lists everything published. Filters are optional and are
    only applied when given. Results are paginated; ask for a page rather
    than expecting the whole catalogue.
    """
    result = await product_tools.search_products(
        _request_api(),
        query=query,
        limit=limit,
        page=page,
        category=category,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
    )
    return result.model_dump(by_alias=True)


@mcp.tool
async def get_product(product_id: str) -> Any:
    """Full detail for one product, by id."""
    return (await product_tools.get_product(_request_api(), product_id)).model_dump(
        by_alias=True
    )


@mcp.tool
async def check_inventory(product_id: str) -> Any:
    """How many of a product are available to buy right now.

    Never cached. Use `available`, not `quantity` -- quantity counts stock
    that is already reserved for other orders.
    """
    return (await product_tools.check_inventory(_request_api(), product_id)).model_dump(
        by_alias=True
    )


@mcp.tool
async def get_orders(limit: int | None = None) -> Any:
    """The signed-in customer's own orders, newest first."""
    orders = await order_tools.get_orders(_request_api(), limit=limit)
    return [order.model_dump(by_alias=True) for order in orders]


@mcp.tool
async def get_order(order_id: str) -> Any:
    """One of the signed-in customer's orders, by id."""
    return (await order_tools.get_order(_request_api(), order_id)).model_dump(
        by_alias=True
    )


@mcp.tool
async def get_cart() -> Any:
    """The signed-in customer's cart, with item count and subtotal."""
    return (await cart_tools.get_cart(_request_api())).model_dump(by_alias=True)


@mcp.tool
async def add_to_cart(product_id: str, quantity: int, mode: str = "add") -> Any:
    """Add a product to the cart.

    'add' increases an existing line by the quantity given; 'set' replaces
    it. Asking for more than is available fails with the number that is
    available -- retry with that, do not retry the same request.
    """
    headers = request_headers()

    view = await cart_tools.add_to_cart(
        api_for_headers(headers),
        product_id=product_id,
        quantity=quantity,
        mode=mode,
        request_id=headers.get("mcp-session-id"),
    )
    return view.model_dump(by_alias=True)


@mcp.tool
async def remove_from_cart(product_id: str | None = None) -> Any:
    """Remove one product from the cart, or empty it when none is named."""
    return (
        await cart_tools.remove_from_cart(_request_api(), product_id=product_id)
    ).model_dump(by_alias=True)


@mcp.tool
async def cancel_order(order_id: str, approval_token: str | None = None) -> Any:
    """Cancel an order the customer placed.

    Requires an approval token issued for this exact order. The token is
    not something to invent, guess or reuse: it is minted only by the
    confirmation flow, for one order, once. Without a valid one this call
    fails and nothing is cancelled.
    """
    headers = request_headers()

    return await order_tools.cancel_order(
        api_for_headers(headers),
        order_id=order_id,
        session_id=session_id_for_headers(headers),
        approval_token=approval_token,
    )


def registered_tool_names() -> list[str]:
    """The tool surface, for the tests that assert what is exposed."""
    import asyncio

    return list(asyncio.run(mcp.get_tools()).keys())


def mint_approval_for(
    headers: dict[str, str], body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Mint an approval token. **Never exposed as an MCP tool.**

    Kept as a plain function so it can be tested without a server, and
    reachable only over HTTP so the Phase 2 chat UI -- non-LLM code, behind
    a human pressing a button -- can call it while the agent's MCP
    connection cannot.

    The credential is required even though it is not used to mint: minting
    for an unauthenticated caller would let anyone who can reach the
    service manufacture approvals.
    """
    try:
        api_for_headers(headers)
        session = session_id_for_headers(headers)
    except MissingCredential as error:
        return 401, {"error": str(error)}

    tool = body.get("tool")

    if tool not in HIGH_RISK_TOOLS:
        return 400, {"error": "Not a tool that takes an approval"}

    args = body.get("args") or {}

    return 200, {
        "token": approvals.mint(session, tool, args),
        "expiresIn": config.APPROVAL_TTL_SECONDS,
    }


@mcp.custom_route("/approvals", methods=["POST"])
async def approvals_route(request):  # pragma: no cover - shell over the above
    from starlette.responses import JSONResponse

    headers = {key.lower(): value for key, value in request.headers.items()}

    try:
        body = await request.json()
    except Exception:
        body = {}

    status, payload = mint_approval_for(headers, body)
    return JSONResponse(payload, status_code=status)


if __name__ == "__main__":
    # Fail at boot rather than at the first cancellation attempt.
    if not config.APPROVAL_SECRET:
        raise SystemExit("MCP_APPROVAL_SECRET is required")

    mcp.run(transport="http", host="0.0.0.0", port=config.PORT)
