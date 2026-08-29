"""Pydantic mirrors of the real /api/v1 response shapes.

Written against responses the deployed API actually returned, not against
a description of them. Two properties follow from respond.ts and must not
drift:

  - every money field is a *string* ("10.50"). A float loses the scale the
    API deliberately preserves, and 10.50 becomes 10.5, which is not a
    price. Typed strictly as `str`: if a number ever appears again that is
    a regression on the API worth failing loudly for, since silent
    coercion is how the same bug went unnoticed the first time.
  - every timestamp is an ISO 8601 string in UTC. Left as a string rather
    than parsed to a datetime, which would invite a timezone conversion on
    the way back out.

Extra fields are ignored rather than rejected. The API may grow a field
before this service knows about it, and failing hard there would take
every tool down for an additive change.
"""

from pydantic import BaseModel, ConfigDict, Field


class Base(BaseModel):
    # populate_by_name lets these be built from either spelling; the alias
    # is what goes back out, so a client never sees a third name for a
    # field the API already named.
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ImageView(Base):
    url: str
    alt_text: str | None = Field(default=None, alias="altText")


class CategoryView(Base):
    id: str
    name: str | None = None
    slug: str | None = None


class ProductSummary(Base):
    id: str
    name: str | None = None
    slug: str | None = None
    price: str | None = None
    compare_price: str | None = Field(default=None, alias="comparePrice")
    description: str | None = None
    status: str | None = None
    sku: str | None = None
    tags: list[str] = Field(default_factory=list)
    category_id: str | None = Field(default=None, alias="categoryId")
    category: CategoryView | None = None
    # The detail route returns no images key at all, so this cannot be
    # required without breaking get_product.
    images: list[ImageView] = Field(default_factory=list)
    created_at: str | None = Field(default=None, alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")


class Pagination(Base):
    page: int | None = None
    limit: int | None = None
    total: int | None = None
    pages: int | None = None


class ProductSearchResult(Base):
    products: list[ProductSummary] = Field(default_factory=list)
    pagination: Pagination | None = None


class InventoryView(Base):
    product_id: str = Field(alias="productId")
    quantity: int
    reserved: int
    # What checkout decrements against, and so the number that answers
    # "can I buy this" -- `quantity` does not.
    available: int
    in_stock: bool = Field(alias="inStock")


class CartLine(Base):
    id: str | None = None
    quantity: int
    product_id: str | None = Field(default=None, alias="productId")
    # A deleted product leaves the line without one. Losing the whole cart
    # over that would be worse than showing an incomplete line.
    product: ProductSummary | None = None


class CartView(Base):
    items: list[CartLine] = Field(default_factory=list)
    item_count: int = Field(alias="itemCount")
    subtotal: str


class OrderLine(Base):
    product_id: str = Field(alias="productId")
    product_name: str = Field(alias="productName")
    product_sku: str | None = Field(default=None, alias="productSku")
    quantity: int
    price: str


class OrderSummary(Base):
    id: str
    order_number: str = Field(alias="orderNumber")
    status: str
    total: str
    subtotal: str | None = None
    tax: str | None = None
    shipping: str | None = None
    currency: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    cancelled_at: str | None = Field(default=None, alias="cancelledAt")
    items: list[OrderLine] = Field(default_factory=list, alias="orderItems")
