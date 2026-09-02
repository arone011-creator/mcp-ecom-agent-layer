# tests/test_schemas.py
#
# Mirrors of the real /api/v1 shapes, checked against responses actually
# returned by the deployed API rather than against a document describing
# it.
#
# Money is a string. respond.ts stringifies every Prisma Decimal so "10.50"
# keeps its scale, and the API's product view now guarantees the same on
# the search path -- a float would reintroduce exactly the precision loss
# both went out of their way to avoid.

import pytest
from pydantic import ValidationError

from models.schemas import (
    CartView,
    InventoryView,
    OrderSummary,
    ProductSearchResult,
    ProductSummary,
)


def test_product_price_stays_a_string():
    product = ProductSummary.model_validate(
        {"id": "p1", "name": "Runner", "slug": "runner", "price": "29.99"}
    )

    assert product.price == "29.99"
    assert isinstance(product.price, str)


def test_a_price_that_arrives_as_a_number_is_rejected_not_silently_coerced():
    # The API guarantees strings. If a float ever shows up again, that is a
    # regression on the API worth failing loudly for, not papering over
    # here -- silent coercion is how the two-types bug survived unnoticed.
    with pytest.raises(ValidationError):
        ProductSummary.model_validate(
            {"id": "p1", "name": "R", "slug": "r", "price": 29.99}
        )


def test_product_tolerates_the_fields_the_detail_route_omits():
    # GET /api/v1/products/{id} returns no images key at all.
    product = ProductSummary.model_validate(
        {"id": "p1", "name": "R", "slug": "r", "price": "1.00"}
    )

    assert product.images == []
    assert product.description is None
    assert product.compare_price is None


def test_product_ignores_fields_the_api_adds_later():
    # An additive API change must not take every tool down.
    product = ProductSummary.model_validate(
        {"id": "p1", "name": "R", "slug": "r", "price": "1.00", "somethingNew": 42}
    )

    assert product.id == "p1"


def test_product_reads_the_api_camel_case_names():
    product = ProductSummary.model_validate(
        {
            "id": "p1",
            "name": "R",
            "slug": "r",
            "price": "999.99",
            "comparePrice": "1099.99",
            "categoryId": "c1",
            "category": {"id": "c1", "name": "Smartphones", "slug": "smartphones"},
            "images": [{"url": "/a.svg", "altText": "A"}],
        }
    )

    assert product.compare_price == "1099.99"
    assert product.category_id == "c1"
    assert product.category.name == "Smartphones"
    assert product.images[0].alt_text == "A"


def test_product_description_is_marked_as_untrusted_content():
    product = ProductSummary.model_validate(
        {"id": "p1", "name": "R", "slug": "r", "price": "1.00", "description": "Nice shoes."}
    )

    assert product.description == "<untrusted-user-content>Nice shoes.</untrusted-user-content>"


def test_a_missing_description_is_not_wrapped():
    product = ProductSummary.model_validate(
        {"id": "p1", "name": "R", "slug": "r", "price": "1.00"}
    )

    assert product.description is None


def test_a_bidi_override_in_a_description_does_not_survive_parsing():
    product = ProductSummary.model_validate(
        {
            "id": "p1",
            "name": "R",
            "slug": "r",
            "price": "1.00",
            "description": "safe\u202edesc",
        }
    )

    assert "\u202e" not in product.description


def test_search_result_carries_pagination():
    result = ProductSearchResult.model_validate(
        {
            "products": [{"id": "p1", "name": "R", "slug": "r", "price": "1.00"}],
            "pagination": {"page": 1, "limit": 1, "total": 1, "pages": 1},
        }
    )

    assert result.pagination.total == 1
    assert len(result.products) == 1


def test_an_empty_search_is_valid():
    result = ProductSearchResult.model_validate({"products": [], "pagination": None})

    assert result.products == []


def test_cart_view_carries_the_computed_totals():
    cart = CartView.model_validate(
        {
            "items": [
                {
                    "id": "ci1",
                    "quantity": 2,
                    "productId": "p1",
                    "product": {
                        "id": "p1",
                        "name": "R",
                        "slug": "r",
                        "price": "29.99",
                    },
                }
            ],
            "itemCount": 2,
            "subtotal": "59.98",
        }
    )

    # Totals come from the API so no agent ever does money arithmetic.
    assert cart.item_count == 2
    assert cart.subtotal == "59.98"
    assert cart.items[0].product.name == "R"


def test_empty_cart_is_valid():
    cart = CartView.model_validate({"items": [], "itemCount": 0, "subtotal": "0.00"})

    assert cart.item_count == 0


def test_a_cart_line_whose_product_vanished_still_parses():
    # cartView selects the product relation; a deleted product would leave
    # the line without one, and losing the whole cart over that would be
    # worse than showing an incomplete line.
    cart = CartView.model_validate(
        {
            "items": [{"id": "ci1", "quantity": 1, "productId": "p1", "product": None}],
            "itemCount": 1,
            "subtotal": "0.00",
        }
    )

    assert cart.items[0].product is None


def test_inventory_reports_availability():
    inventory = InventoryView.model_validate(
        {
            "productId": "p1",
            "quantity": 10,
            "reserved": 3,
            "available": 7,
            "inStock": True,
        }
    )

    # `available` is what checkout decrements against, so it is the number
    # that answers "can I buy this" -- not `quantity`.
    assert inventory.available == 7
    assert inventory.in_stock is True


def test_order_summary_reads_the_api_field_names():
    order = OrderSummary.model_validate(
        {
            "id": "o1",
            "orderNumber": "ORD-1",
            "status": "PENDING",
            "total": "59.98",
            "currency": "USD",
            "createdAt": "2026-08-29T10:00:00.000Z",
            "orderItems": [
                {
                    "productId": "p1",
                    "productName": "R",
                    "quantity": 2,
                    "price": "29.99",
                }
            ],
        }
    )

    assert order.order_number == "ORD-1"
    assert order.items[0].product_name == "R"
    assert order.items[0].price == "29.99"


def test_order_without_items_selected_is_valid():
    order = OrderSummary.model_validate(
        {"id": "o1", "orderNumber": "ORD-1", "status": "CANCELLED", "total": "0.00"}
    )

    assert order.items == []
    assert order.cancelled_at is None


def test_timestamps_stay_strings_rather_than_becoming_local_datetimes():
    # respond.ts emits ISO 8601 in UTC. Parsing to datetime here would
    # invite a timezone conversion on the way back out.
    order = OrderSummary.model_validate(
        {
            "id": "o1",
            "orderNumber": "ORD-1",
            "status": "PENDING",
            "total": "1.00",
            "createdAt": "2026-08-29T10:00:00.000Z",
        }
    )

    assert order.created_at == "2026-08-29T10:00:00.000Z"


def test_models_round_trip_back_to_the_api_field_names():
    # The tools hand these to MCP clients, and a client that sees
    # item_count where the API says itemCount has been given a third
    # spelling of the same thing.
    cart = CartView.model_validate({"items": [], "itemCount": 0, "subtotal": "0.00"})

    assert cart.model_dump(by_alias=True) == {
        "items": [],
        "itemCount": 0,
        "subtotal": "0.00",
    }
