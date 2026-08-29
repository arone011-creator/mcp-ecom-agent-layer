# tests/test_ecommerce_api.py
#
# The client is the only thing in this service that speaks HTTP to
# /api/v1. Everything it knows about the envelope is asserted here, so a
# change to the API's shape breaks one file rather than nine tools.

import httpx
import pytest
import respx

from clients.ecommerce_api import ApiError, EcommerceApi

BASE = "https://api.test"


def client() -> EcommerceApi:
    return EcommerceApi(base_url=BASE, token="tok_abc")


@respx.mock
async def test_unwraps_the_data_envelope():
    respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {"products": []}})
    )

    assert await client().get("/api/v1/products") == {"products": []}


@respx.mock
async def test_sends_the_bearer_token():
    route = respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().get("/api/v1/cart")

    assert route.calls.last.request.headers["authorization"] == "Bearer tok_abc"


@respx.mock
async def test_raises_with_the_api_error_message_and_status():
    respx.post(f"{BASE}/api/v1/orders/o1/cancel").mock(
        return_value=httpx.Response(409, json={"error": "Order cannot be cancelled"})
    )

    with pytest.raises(ApiError) as caught:
        await client().post("/api/v1/orders/o1/cancel", {})

    # The status is what tells an agent whether to retry or give up, so it
    # has to survive the trip rather than collapsing into "it failed".
    assert caught.value.status == 409
    assert caught.value.message == "Order cannot be cancelled"


@respx.mock
async def test_reports_a_non_json_body_without_leaking_it():
    respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(502, text="<html>gateway</html>")
    )

    with pytest.raises(ApiError) as caught:
        await client().get("/api/v1/cart")

    # A proxy's HTML error page is attacker-adjacent text. Passing it
    # through would put arbitrary upstream content into agent context.
    assert caught.value.status == 502
    assert caught.value.message == "Upstream returned an unreadable response"
    assert "html" not in caught.value.message


@respx.mock
async def test_forwards_an_idempotency_key_when_given_one():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().post("/api/v1/cart", {"productId": "p1"}, idempotency_key="k1")

    assert route.calls.last.request.headers["idempotency-key"] == "k1"


@respx.mock
async def test_omits_the_idempotency_header_when_there_is_none():
    route = respx.post(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().post("/api/v1/cart", {"productId": "p1"})

    assert "idempotency-key" not in route.calls.last.request.headers


@respx.mock
async def test_never_sends_a_cookie():
    # An ambient cookie would be a second identity, and requireApiUser
    # accepts one. The bearer token is the only credential this service has
    # any business presenting.
    route = respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().get("/api/v1/cart")

    assert "cookie" not in route.calls.last.request.headers


@respx.mock
async def test_drops_query_params_that_were_not_given():
    # A defaulted filter is a filter. minRating=0 would silently exclude
    # every product nobody has reviewed.
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().get("/api/v1/products", {"q": "shoes", "minRating": None})

    params = route.calls.last.request.url.params
    assert params["q"] == "shoes"
    assert "minRating" not in params


@respx.mock
async def test_whoami_returns_the_verified_caller():
    respx.get(f"{BASE}/api/v1/auth/whoami").mock(
        return_value=httpx.Response(
            200, json={"data": {"id": "u1", "email": "a@x.com", "role": "USER"}}
        )
    )

    assert (await client().whoami())["id"] == "u1"


@respx.mock
async def test_delete_passes_a_query_param():
    route = respx.delete(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await client().delete("/api/v1/cart", {"productId": "p1"})

    assert route.calls.last.request.url.params["productId"] == "p1"


@respx.mock
async def test_an_error_body_without_a_message_still_raises_usefully():
    respx.get(f"{BASE}/api/v1/cart").mock(
        return_value=httpx.Response(401, json={"unexpected": "shape"})
    )

    with pytest.raises(ApiError) as caught:
        await client().get("/api/v1/cart")

    assert caught.value.status == 401
    assert caught.value.message == "Request failed"


@respx.mock
async def test_a_client_without_a_token_sends_no_authorization():
    # The product routes are public by design; presenting an empty bearer
    # would be worse than presenting none.
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )

    await EcommerceApi(base_url=BASE).get("/api/v1/products")

    assert "authorization" not in route.calls.last.request.headers
