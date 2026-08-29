"""Cart capabilities. Phase 3 hands this file to the Cart agent.

Always the cart belonging to the token's user, never one named in the
request. The API enforces that; this file simply never offers a way to ask
for another.

get_cart is low risk. add_to_cart and remove_from_cart are Medium: they
execute without blocking, and the agent surfaces them as informational
events ("Added to cart") rather than as a prompt. Medium is not a weaker
kind of approval -- it is no approval, with a visible trace.
"""

import hashlib
import json

from clients.ecommerce_api import EcommerceApi
from models.schemas import CartView


def _idempotency_key(operation: str, args: dict, request_id: str | None) -> str:
    """Stable for one logical call, different for a different one.

    Derived from the arguments rather than randomly generated, so a retry
    of the *same* call reuses the key and replays, while a genuinely
    different call cannot accidentally replay the previous result.

    `request_id` scopes it to one conversation turn: without it, "add one
    more" tomorrow would hash identically to "add one more" today and the
    second would silently replay the first instead of adding anything.
    """
    payload = json.dumps(
        {"op": operation, "args": args, "request": request_id or ""},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


async def get_cart(api: EcommerceApi) -> CartView:
    """The caller's cart, with item count and subtotal already computed.

    The totals come from the API on purpose, so no agent ever does money
    arithmetic on values it may have rounded on the way in.
    """
    return CartView.model_validate(await api.get("/api/v1/cart"))


async def add_to_cart(
    api: EcommerceApi,
    product_id: str,
    quantity: int,
    mode: str = "add",
    request_id: str | None = None,
) -> CartView:
    """Add a product to the caller's cart.

    `mode='add'` increments an existing line, which is what the storefront
    does; `mode='set'` replaces the quantity outright. The default matches
    the browser deliberately -- a cart assembled by an agent and one
    assembled by a person are the same cart and must behave the same way.

    More than available stock comes back as 409 carrying the number that
    *is* available, which is the signal to retry smaller rather than to
    rewrite the request.
    """
    body = {"productId": product_id, "quantity": quantity, "mode": mode}

    return CartView.model_validate(
        await api.post(
            "/api/v1/cart",
            body,
            idempotency_key=_idempotency_key("cart:add", body, request_id),
        )
    )


async def remove_from_cart(
    api: EcommerceApi, product_id: str | None = None
) -> CartView:
    """Remove one product from the cart, or empty it when none is named.

    Medium risk, by symmetry with add_to_cart: it is reversible -- re-add
    the line -- and the API scopes its delete to the caller's own cart id,
    so a product id can only ever reach their own lines.

    No idempotency key: the operation is already idempotent. Removing a
    line that is gone leaves the cart in the state the caller asked for,
    and a replay of that is indistinguishable from doing it once.
    """
    return CartView.model_validate(
        await api.delete("/api/v1/cart", {"productId": product_id})
    )
