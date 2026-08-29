# tests/test_tools_cart.py
#
# get_cart only. The mutations are Task 7.

import httpx
import pytest
import respx

from clients.ecommerce_api import ApiError, EcommerceApi
from tools import cart

BASE = "https://api.test"


def api() -> EcommerceApi:
    return EcommerceApi(base_url=BASE, token="tok")


def cart_body(**overrides):
    base = {
        "items": [
            {
                "id": "ci1",
                "quantity": 2,
                "productId": "p1",
                "product": {
                    "id": "p1",
                    "name": "Runner",
                    "slug": "runner",
                    "price": "29.99",
                },
            }
        ],
        "itemCount": 2,
        "subtotal": "59.98",
    }
    base.update(overrides)
    return base


@respx.mock
async def test_get_cart_returns_the_computed_totals():
    respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    view = await cart.get_cart(api())

    # Totals come from the API so an agent never does money arithmetic.
    assert view.subtotal == "59.98"
    assert view.item_count == 2
    assert view.items[0].product.name == "Runner"


@respx.mock
async def test_an_empty_cart_is_not_an_error():
    respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(
            200, json={"data": {"items": [], "itemCount": 0, "subtotal": "0.00"}}
        )
    )

    view = await cart.get_cart(api())

    assert view.item_count == 0
    assert view.items == []


@respx.mock
async def test_get_cart_never_names_a_cart():
    # The API returns the cart belonging to the token's user. Naming one
    # would be asking for someone else's.
    route = respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.get_cart(api())

    assert str(route.calls.last.request.url).endswith("/api/v1/cart")


@respx.mock
async def test_an_unauthenticated_cart_read_surfaces_as_401():
    respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(401, json={"error": "Authentication required"})
    )

    with pytest.raises(ApiError) as caught:
        await cart.get_cart(EcommerceApi(base_url=BASE))

    assert caught.value.status == 401


# --- Medium risk: the mutations. Executed, then surfaced -- not blocked --
# but they carry an idempotency key so a retry after a timeout does not
# double-apply.

import json


@respx.mock
async def test_add_to_cart_increments_by_default():
    # server/actions/cart.ts increments an existing line, and a cart built
    # by an agent must behave like one built in the browser -- it is the
    # same cart.
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.add_to_cart(api(), product_id="p1", quantity=2)

    assert json.loads(route.calls.last.request.content) == {
        "productId": "p1",
        "quantity": 2,
        "mode": "add",
    }


@respx.mock
async def test_add_to_cart_can_set_an_exact_quantity():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.add_to_cart(api(), product_id="p1", quantity=2, mode="set")

    assert json.loads(route.calls.last.request.content)["mode"] == "set"


@respx.mock
async def test_add_to_cart_returns_the_cart_it_just_changed():
    # The API answers with the whole cart so no follow-up read is needed.
    # An agent that must make a second call to see what it did sometimes
    # will not.
    respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    view = await cart.add_to_cart(api(), product_id="p1", quantity=2)

    assert view.item_count == 2
    assert view.subtotal == "59.98"


@respx.mock
async def test_add_to_cart_sends_an_idempotency_key():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.add_to_cart(api(), product_id="p1", quantity=1)

    assert route.calls.last.request.headers.get("idempotency-key")


@respx.mock
async def test_the_same_call_reuses_its_key_and_a_different_one_does_not():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.add_to_cart(api(), product_id="p1", quantity=1, request_id="r1")
    await cart.add_to_cart(api(), product_id="p1", quantity=1, request_id="r1")
    await cart.add_to_cart(api(), product_id="p1", quantity=2, request_id="r1")

    keys = [call.request.headers["idempotency-key"] for call in route.calls]
    # A retry of the same call replays; a different call must not.
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


@respx.mock
async def test_the_same_call_in_another_turn_is_not_a_replay():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": cart_body()})
    )

    await cart.add_to_cart(api(), product_id="p1", quantity=1, request_id="r1")
    await cart.add_to_cart(api(), product_id="p1", quantity=1, request_id="r2")

    keys = [call.request.headers["idempotency-key"] for call in route.calls]
    # "Add one more" tomorrow is not a replay of "add one more" today.
    assert keys[0] != keys[1]


@respx.mock
async def test_over_stock_is_a_409_the_agent_can_act_on():
    respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(
            409, json={"error": "Only 3 available; cart would hold 5"}
        )
    )

    with pytest.raises(ApiError) as caught:
        await cart.add_to_cart(api(), product_id="p1", quantity=5)

    # 409 means "re-read stock and try a smaller number", not "rewrite the
    # request". The message carries the number to retry with.
    assert caught.value.status == 409
    assert "Only 3 available" in caught.value.message


@respx.mock
async def test_an_unpublished_product_is_a_404_like_any_other():
    respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(404, json={"error": "Product not found"})
    )

    with pytest.raises(ApiError) as caught:
        await cart.add_to_cart(api(), product_id="p9", quantity=1)

    assert caught.value.status == 404


@respx.mock
async def test_remove_from_cart_narrows_to_one_product():
    route = respx.delete(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(
            200, json={"data": {"items": [], "itemCount": 0, "subtotal": "0.00"}}
        )
    )

    await cart.remove_from_cart(api(), product_id="p1")

    # "Remove the shoes" must not empty the cart.
    assert route.calls.last.request.url.params["productId"] == "p1"


@respx.mock
async def test_remove_from_cart_can_empty_it():
    route = respx.delete(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(
            200, json={"data": {"items": [], "itemCount": 0, "subtotal": "0.00"}}
        )
    )

    await cart.remove_from_cart(api())

    assert "productId" not in route.calls.last.request.url.params


@respx.mock
async def test_remove_from_cart_returns_the_remaining_cart():
    respx.delete(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(
            200, json={"data": {"items": [], "itemCount": 0, "subtotal": "0.00"}}
        )
    )

    view = await cart.remove_from_cart(api(), product_id="p1")

    assert view.item_count == 0
