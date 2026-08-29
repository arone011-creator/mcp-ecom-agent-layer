"""Product capabilities. Phase 3 hands this file to the Product agent.

Low risk throughout: the catalogue is the same data the storefront shows
to anyone, and /api/v1/products resolves no caller by design.
"""

from clients.ecommerce_api import EcommerceApi
from models.schemas import InventoryView, ProductSearchResult, ProductSummary


async def search_products(
    api: EcommerceApi,
    query: str = "",
    limit: int | None = None,
    page: int | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    sort: str | None = None,
) -> ProductSearchResult:
    """Search the catalogue. An empty query lists everything published.

    `sort='price'` is always *descending* -- the ascending path is
    unreachable through this API. Do not describe this tool as answering
    "cheapest first"; it cannot.

    Every filter is passed only when given. A defaulted `min_rating` of 0
    would read as a filter and silently drop every product nobody has
    reviewed.
    """
    data = await api.get(
        "/api/v1/products",
        {
            "q": query,
            "limit": limit,
            "page": page,
            "category": category,
            "minPrice": min_price,
            "maxPrice": max_price,
            "minRating": min_rating,
            "sort": sort,
        },
    )
    return ProductSearchResult.model_validate(data)


async def get_product(api: EcommerceApi, product_id: str) -> ProductSummary:
    """One product in full.

    An unpublished product answers 404 exactly like an absent one, so this
    cannot be used to discover what is coming.
    """
    return ProductSummary.model_validate(await api.get(f"/api/v1/products/{product_id}"))


async def check_inventory(api: EcommerceApi, product_id: str) -> InventoryView:
    """Stock as it is right now.

    Separate from get_product because this is the one piece of product data
    that must never be cached: an agent deciding whether it can add three
    of something needs the number as it is, not as it was.
    """
    return InventoryView.model_validate(
        await api.get(f"/api/v1/products/{product_id}/inventory")
    )
