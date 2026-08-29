# tests/test_tools_orders.py
#
# The read side only. cancel_order is Task 8, because it cannot be written
# correctly until approval tokens exist.

import httpx
import pytest
import respx

from clients.ecommerce_api import ApiError, EcommerceApi
from tools import orders

BASE = "https://api.test"


def api() -> EcommerceApi:
    return EcommerceApi(base_url=BASE, token="tok")


def order(**overrides):
    base = {
        "id": "o1",
        "orderNumber": "ORD-1",
        "status": "PENDING",
        "total": "59.98",
        "orderItems": [],
    }
    base.update(overrides)
    return base


@respx.mock
async def test_get_orders_never_sends_a_user_id():
    # /api/v1/orders scopes to the user its own session helper verified.
    # A user_id argument here would be an LLM-supplied identity, which is
    # the one thing §1.4 forbids.
    route = respx.get(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(200, json={"data": {"orders": []}})
    )

    await orders.get_orders(api(), limit=5)

    params = route.calls.last.request.url.params
    assert params["limit"] == "5"
    assert "userId" not in params
    assert "user_id" not in params
    assert "email" not in params


@respx.mock
async def test_get_orders_models_each_order():
    respx.get(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "orders": [
                        order(
                            orderItems=[
                                {
                                    "productId": "p1",
                                    "productName": "Runner",
                                    "quantity": 2,
                                    "price": "29.99",
                                }
                            ]
                        )
                    ]
                }
            },
        )
    )

    result = await orders.get_orders(api())

    assert result[0].order_number == "ORD-1"
    assert result[0].items[0].product_name == "Runner"


@respx.mock
async def test_no_orders_is_an_empty_list_not_an_error():
    respx.get(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(200, json={"data": {"orders": []}})
    )

    assert await orders.get_orders(api()) == []


@respx.mock
async def test_get_orders_omits_an_unspecified_limit():
    route = respx.get(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(200, json={"data": {"orders": []}})
    )

    await orders.get_orders(api())

    assert "limit" not in route.calls.last.request.url.params


@respx.mock
async def test_get_order_returns_one_order():
    respx.get(f"{BASE}/api/v1/orders/o1").mock(
        return_value=httpx.Response(200, json={"data": order(status="DELIVERED")})
    )

    found = await orders.get_order(api(), order_id="o1")

    assert found.id == "o1"
    assert found.status == "DELIVERED"


@respx.mock
async def test_someone_elses_order_is_indistinguishable_from_a_missing_one():
    respx.get(f"{BASE}/api/v1/orders/o9").mock(
        return_value=httpx.Response(404, json={"error": "Order not found"})
    )

    with pytest.raises(ApiError) as caught:
        await orders.get_order(api(), order_id="o9")

    # 404, not 403 -- and this layer must not "helpfully" turn it into one,
    # because a 403 confirms the id is real, which is all an enumeration
    # attack needs.
    assert caught.value.status == 404
    assert caught.value.message == "Order not found"


@respx.mock
async def test_an_unauthenticated_read_surfaces_as_401():
    respx.get(f"{BASE}/api/v1/orders").mock(
        return_value=httpx.Response(401, json={"error": "Authentication required"})
    )

    with pytest.raises(ApiError) as caught:
        await orders.get_orders(EcommerceApi(base_url=BASE))

    assert caught.value.status == 401


# --- High risk: cancel_order. Enforced, not merely conventional.
#
# The assertions that matter are the ones where the API is NEVER CALLED.
# A tier the agent is trusted to respect is a convention; a tier the
# server refuses to execute without a token is a boundary.

import approvals
from approvals import ApprovalError


@pytest.fixture(autouse=True)
def _approval_secret(monkeypatch):
    monkeypatch.setattr(approvals.config, "APPROVAL_SECRET", "test-secret")
    approvals.reset_spent_nonces()


@respx.mock
async def test_cancel_without_an_approval_token_never_reaches_the_api():
    route = respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    with pytest.raises(ApprovalError):
        await orders.cancel_order(
            api(), order_id="o1", session_id="s1", approval_token=None
        )

    # Intent is irrelevant. The call does not happen.
    assert not route.called


@respx.mock
async def test_cancel_with_a_token_for_another_order_never_reaches_the_api():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o3"})
    route = respx.post(f"{BASE}/api/v1/orders/o7/cancel").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    with pytest.raises(ApprovalError, match="different arguments"):
        await orders.cancel_order(
            api(), order_id="o7", session_id="s1", approval_token=token
        )

    assert not route.called


@respx.mock
async def test_cancel_with_a_token_from_another_session_never_reaches_the_api():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o1"})
    route = respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    with pytest.raises(ApprovalError, match="another session"):
        await orders.cancel_order(
            api(), order_id="o1", session_id="s2", approval_token=token
        )

    assert not route.called


@respx.mock
async def test_a_garbage_token_never_reaches_the_api():
    route = respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    with pytest.raises(ApprovalError):
        await orders.cancel_order(
            api(), order_id="o1", session_id="s1", approval_token="not-a-token"
        )

    assert not route.called


@respx.mock
async def test_cancel_with_a_valid_token_calls_the_api_once():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o1"})
    route = respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(
            200, json={"data": {"orderId": "o1", "status": "CANCELLED"}}
        )
    )

    result = await orders.cancel_order(
        api(), order_id="o1", session_id="s1", approval_token=token
    )

    assert result == {"orderId": "o1", "status": "CANCELLED"}
    assert len(route.calls) == 1
    # A network retry must replay rather than cancel a second time.
    assert route.calls.last.request.headers.get("idempotency-key")


@respx.mock
async def test_the_token_is_burnt_even_though_the_api_call_follows():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o1"})
    respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(
            200, json={"data": {"orderId": "o1", "status": "CANCELLED"}}
        )
    )

    await orders.cancel_order(
        api(), order_id="o1", session_id="s1", approval_token=token
    )

    with pytest.raises(ApprovalError, match="already been used"):
        await orders.cancel_order(
            api(), order_id="o1", session_id="s1", approval_token=token
        )


@respx.mock
async def test_an_uncancellable_order_surfaces_as_409():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o1"})
    respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(409, json={"error": "Order cannot be cancelled"})
    )

    with pytest.raises(ApiError) as caught:
        await orders.cancel_order(
            api(), order_id="o1", session_id="s1", approval_token=token
        )

    # Eligibility lives in cancelOrderFor, not here. This layer reports the
    # answer; it does not compute it.
    assert caught.value.status == 409


@respx.mock
async def test_cancelling_someone_elses_order_is_still_a_404():
    token = approvals.mint("s1", "cancel_order", {"order_id": "o9"})
    respx.post(f"{BASE}/api/v1/orders/o9/cancel").mock(
        return_value=httpx.Response(404, json={"error": "Order not found"})
    )

    with pytest.raises(ApiError) as caught:
        await orders.cancel_order(
            api(), order_id="o9", session_id="s1", approval_token=token
        )

    # An approval is not ownership. The API still refuses, and still
    # refuses without confirming the id exists.
    assert caught.value.status == 404
    assert caught.value.message == "Order not found"


@respx.mock
async def test_the_same_cancellation_reuses_its_idempotency_key():
    route = respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(
            200, json={"data": {"orderId": "o1", "status": "CANCELLED"}}
        )
    )

    for _ in range(2):
        token = approvals.mint("s1", "cancel_order", {"order_id": "o1"})
        await orders.cancel_order(
            api(), order_id="o1", session_id="s1", approval_token=token
        )

    keys = [call.request.headers["idempotency-key"] for call in route.calls]
    assert keys[0] == keys[1]
