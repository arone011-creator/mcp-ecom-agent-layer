# tests/test_tools_products.py
#
# Tools are business capabilities, not HTTP endpoints. What is asserted
# here is the mapping from a question an agent would ask to the request the
# API actually wants -- and the places where a plausible-looking default
# would quietly give a wrong answer.

import httpx
import pytest
import respx

from clients.ecommerce_api import ApiError, EcommerceApi
from tools import products

BASE = "https://api.test"


def api() -> EcommerceApi:
    return EcommerceApi(base_url=BASE, token="tok")


def product(**overrides):
    base = {"id": "p1", "name": "Runner", "slug": "runner", "price": "29.99"}
    base.update(overrides)
    return base


@respx.mock
async def test_search_passes_the_query_and_filters_through():
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(
            200, json={"data": {"products": [], "pagination": {}}}
        )
    )

    await products.search_products(
        api(), query="shoes", limit=5, page=2, min_price=10, max_price=50, min_rating=4.0
    )

    params = route.calls.last.request.url.params
    assert params["q"] == "shoes"
    assert params["limit"] == "5"
    assert params["page"] == "2"
    assert params["minPrice"] == "10"
    assert params["maxPrice"] == "50"
    assert params["minRating"] == "4.0"


@respx.mock
async def test_search_omits_absent_filters_rather_than_defaulting_them():
    # A defaulted minRating=0 reads as a filter and silently drops every
    # product nobody has reviewed.
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {"products": []}})
    )

    await products.search_products(api(), query="shoes")

    params = route.calls.last.request.url.params
    assert "minRating" not in params
    assert "minPrice" not in params
    assert "category" not in params
    assert "sort" not in params


@respx.mock
async def test_an_empty_query_is_the_plain_catalogue_listing():
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {"products": []}})
    )

    await products.search_products(api())

    # q="" matches everything published, which is how "show me what you
    # have" works without a second endpoint.
    assert route.calls.last.request.url.params["q"] == ""


@respx.mock
async def test_search_returns_modelled_products():
    respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "products": [product()],
                    "pagination": {"page": 1, "limit": 20, "total": 1, "pages": 1},
                }
            },
        )
    )

    result = await products.search_products(api(), query="shoes")

    assert result.products[0].price == "29.99"
    assert result.pagination.total == 1


@respx.mock
async def test_search_does_not_require_a_token():
    # The catalogue is the same data the storefront shows anyone.
    route = respx.get(f"{BASE}/api/v1/products").mock(
        return_value=httpx.Response(200, json={"data": {"products": []}})
    )

    await products.search_products(EcommerceApi(base_url=BASE), query="shoes")

    assert route.called


@respx.mock
async def test_get_product_returns_the_detail_shape():
    respx.get(f"{BASE}/api/v1/products/p1").mock(
        return_value=httpx.Response(
            200, json={"data": product(comparePrice="39.99", description="Fast")}
        )
    )

    found = await products.get_product(api(), product_id="p1")

    assert found.compare_price == "39.99"
    assert found.description == "Fast"


@respx.mock
async def test_get_product_surfaces_a_404_as_a_tool_error():
    # An unpublished product answers exactly like an absent one, so this
    # cannot be used to probe for what is coming.
    respx.get(f"{BASE}/api/v1/products/nope").mock(
        return_value=httpx.Response(404, json={"error": "Product not found"})
    )

    with pytest.raises(ApiError) as caught:
        await products.get_product(api(), product_id="nope")

    assert caught.value.status == 404


@respx.mock
async def test_check_inventory_answers_can_i_buy_this():
    respx.get(f"{BASE}/api/v1/products/p1/inventory").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "productId": "p1",
                    "quantity": 10,
                    "reserved": 3,
                    "available": 7,
                    "inStock": True,
                }
            },
        )
    )

    inventory = await products.check_inventory(api(), product_id="p1")

    # `available` is what checkout decrements against, so it is the number
    # that answers the question -- not `quantity`, which counts stock that
    # is already spoken for.
    assert inventory.available == 7
    assert inventory.quantity == 10
    assert inventory.in_stock is True


@respx.mock
async def test_check_inventory_reports_a_product_with_no_record():
    respx.get(f"{BASE}/api/v1/products/p9/inventory").mock(
        return_value=httpx.Response(
            404, json={"error": "No inventory record for that product"}
        )
    )

    with pytest.raises(ApiError) as caught:
        await products.check_inventory(api(), product_id="p9")

    assert caught.value.status == 404
