"""The supervisor's toolbox, generated from the team.

The supervisor holds no shop tools at all. Its tools ARE the specialists,
which is what makes the delegation visible in the same event stream as
everything else -- and therefore free of any storefront change.
"""

import pytest

from agent.delegation import (
    DELEGATION_PREFIX,
    delegation_tool_name,
    delegation_tools,
    member_for_tool,
)
from agent.team import TEAM


def test_there_is_one_tool_per_member():
    tools = delegation_tools()

    assert len(tools) == len(TEAM)
    assert {tool["function"]["name"] for tool in tools} == {
        f"{DELEGATION_PREFIX}{member.name}" for member in TEAM
    }


def test_a_tool_carries_its_member_description():
    """The description IS the routing rule -- there is no classifier."""
    tools = {tool["function"]["name"]: tool for tool in delegation_tools()}
    product = tools["ask_product"]

    assert "catalogue" in product["function"]["description"]


def test_a_delegation_tool_takes_exactly_one_string():
    """A self-contained request, because the specialist sees nothing else."""
    product = delegation_tools()[0]["function"]
    params = product["parameters"]

    assert params["required"] == ["request"]
    assert params["properties"]["request"]["type"] == "string"
    assert list(params["properties"]) == ["request"]


def test_no_shop_tool_is_offered_to_the_supervisor():
    """THE MUST PROVE.

    The supervisor delegates and composes; it never touches the shop. A
    shop tool appearing here would put cancel_order back on every turn,
    which is the thing this whole change is for.
    """
    names = {tool["function"]["name"] for tool in delegation_tools()}

    for name in names:
        assert name.startswith(DELEGATION_PREFIX)


def test_a_tool_name_maps_back_to_its_member():
    assert member_for_tool("ask_order").name == "order"
    assert delegation_tool_name("order") == "ask_order"


def test_an_unknown_tool_name_is_refused_rather_than_guessed():
    with pytest.raises(KeyError):
        member_for_tool("ask_nobody")

    with pytest.raises(KeyError):
        member_for_tool("search_products")
