"""Who is on the team, and what each member may touch.

THIS IS THE FILE YOU CHANGE TO MOVE THIS PATTERN SOMEWHERE ELSE. The
supervisor's delegation tools are generated from it and the graph is
compiled from it, so a different domain is a different tuple here and
nothing else.

Each member is a name, a description the supervisor routes on, a prompt,
and a set of tool names. The prompt is COMPOSED from SHARED_RULES rather
than written out, so the untrusted-content boundary and the link rule
cannot be omitted from a new specialist by forgetting to copy them.

The split is by domain rather than by risk, which is what the roadmap
called for -- and it happens to give the security property anyway: the
product specialist is the one that reads text written by strangers, and
it is read-only by construction.
"""

from dataclasses import dataclass

from agent.prompt import SHARED_RULES


@dataclass(frozen=True)
class Member:
    """One specialist.

    Frozen because the team is a declaration, not a thing to mutate at
    run time. A member that could be edited mid-turn would make the tool
    restriction a suggestion rather than a boundary.
    """

    name: str
    # What the supervisor reads to decide where a request goes. This is
    # the routing logic: there is no classifier, the description IS the
    # rule, and a vague one produces vague routing.
    description: str
    prompt: str
    tools: frozenset[str]


def _prompt(role: str) -> str:
    return f"{role}\n\n{SHARED_RULES}"


PRODUCT = Member(
    name="product",
    description=(
        "The catalogue: searching for products, product details and "
        "descriptions, prices, and stock levels. Use for any question "
        "about what the shop sells or whether something is available."
    ),
    prompt=_prompt(
        "You are the product specialist for an online storefront. You "
        "answer questions about the catalogue using your tools, and you "
        "answer only what was asked. You have no access to the "
        "customer's cart or orders; if a request needs one, say so "
        "plainly and stop rather than guessing."
    ),
    tools=frozenset({"search_products", "get_product", "check_inventory"}),
)

ORDER = Member(
    name="order",
    description=(
        "The customer's own orders: listing them, opening one to see its "
        "details and status, and cancelling one. Use for anything about "
        "an order that has already been placed."
    ),
    prompt=_prompt(
        "You are the order specialist for an online storefront. You look "
        "up the signed-in customer's own orders and can cancel one, "
        "which requires the customer's approval. You have no access to "
        "the catalogue or the cart; if a request needs one, say so "
        "plainly and stop rather than guessing."
    ),
    tools=frozenset({"get_orders", "get_order", "cancel_order"}),
)

CART = Member(
    name="cart",
    description=(
        "The shopping cart: seeing what is in it, adding a product to "
        "it, and removing a product from it. Use for anything about what "
        "the customer intends to buy but has not yet ordered."
    ),
    prompt=_prompt(
        "You are the cart specialist for an online storefront. You read "
        "and change the signed-in customer's cart. You cannot search the "
        "catalogue: if you are asked to add something and were not given "
        "a product id, say which product id you need and stop rather "
        "than guessing at one."
    ),
    tools=frozenset({"add_to_cart", "remove_from_cart", "get_cart"}),
)

TEAM: tuple[Member, ...] = (PRODUCT, ORDER, CART)


def member_named(name: str) -> Member:
    """The member with this name, or KeyError.

    Raising rather than returning None: a lookup that misses means the
    supervisor asked for a specialist that does not exist, and continuing
    with nothing would turn that into a turn that quietly does less than
    it said.
    """
    for member in TEAM:
        if member.name == name:
            return member

    raise KeyError(f"No team member named {name!r}")
