"""Order capabilities. Phase 3 hands this file to the Order agent.

Identity is never an argument here. /api/v1/orders filters by the user its
own session helper verified; accepting an id would mean trusting one a
model produced, which is exactly what §1.4 forbids.
"""

import hashlib
import json

import approvals
from clients.ecommerce_api import EcommerceApi
from models.schemas import OrderSummary


async def get_orders(api: EcommerceApi, limit: int | None = None) -> list[OrderSummary]:
    """The caller's own orders, newest first."""
    data = await api.get("/api/v1/orders", {"limit": limit})
    return [OrderSummary.model_validate(order) for order in data.get("orders", [])]


async def get_order(api: EcommerceApi, order_id: str) -> OrderSummary:
    """One order, if the caller placed it.

    An order belonging to someone else answers 404, identically to one that
    does not exist. That is deliberate on the API's side -- a 403 would
    confirm the id is real -- and it is passed through unchanged rather
    than being translated into something more "helpful".
    """
    return OrderSummary.model_validate(await api.get(f"/api/v1/orders/{order_id}"))


async def cancel_order(
    api: EcommerceApi,
    order_id: str,
    session_id: str,
    approval_token: str | None,
) -> dict:
    """Cancel an order. **High risk** -- enforced, not merely conventional.

    The approval is checked before anything leaves this process. An agent
    that has been prompt-injected into calling this cannot proceed by
    intending to: without a token minted for this session, this tool and
    this order id, the request is never sent. That is the difference
    between a risk tier and a security boundary.

    Whether the order *may* be cancelled is not decided here. cancelOrderFor
    owns that rule -- only PENDING and PROCESSING orders qualify -- and
    answers 409 when it says no. An order belonging to someone else answers
    404, because an approval is not ownership.
    """
    approvals.validate(
        approval_token or "", session_id, "cancel_order", {"order_id": order_id}
    )

    # Derived from the order and the session rather than the token, so a
    # network retry of the same cancellation replays instead of cancelling
    # twice. The token cannot be part of this: it is single-use, so a
    # retry necessarily carries a different one.
    key = hashlib.sha256(
        json.dumps(
            {"op": "order:cancel", "id": order_id, "sid": session_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:32]

    return await api.post(f"/api/v1/orders/{order_id}/cancel", {}, idempotency_key=key)
