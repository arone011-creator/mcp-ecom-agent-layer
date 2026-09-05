"""Who is on the team, and what each one may touch.

These are invariants rather than examples. The partition test is what
catches a tenth tool being added that no specialist can reach -- a silent
loss of capability that no other test would notice.
"""

from agent.prompt import UNTRUSTED_TAG
from agent.team import TEAM, member_named
from agent.tools import AGENT_TOOLS, HIGH_RISK_TOOLS


def test_the_team_covers_every_tool_the_agent_has():
    """THE MUST PROVE. A tool no member holds is a capability lost."""
    covered = set()
    for member in TEAM:
        covered |= set(member.tools)

    assert covered == set(AGENT_TOOLS)


def test_no_tool_belongs_to_two_members():
    """Overlap makes routing ambiguous and the security claim untrue."""
    seen = set()
    for member in TEAM:
        clash = seen & set(member.tools)
        assert not clash, f"{member.name} shares {clash}"
        seen |= set(member.tools)


def test_member_names_are_unique():
    names = [member.name for member in TEAM]
    assert len(names) == len(set(names))


def test_the_product_specialist_cannot_reach_a_high_risk_tool():
    """THE SECURITY MUST PROVE.

    The product specialist is the one that reads product descriptions and
    reviews -- text written by strangers. It must not hold a tool that
    changes anything, so a successful injection has nothing to reach for.
    """
    product = member_named("product")

    assert not (set(product.tools) & set(HIGH_RISK_TOOLS))
    assert product.tools == frozenset(
        {"search_products", "get_product", "check_inventory"}
    )


def test_every_member_carries_the_security_rules():
    """A specialist written without these is a hole in the boundary."""
    for member in TEAM:
        assert f"<{UNTRUSTED_TAG}>" in member.prompt, member.name
        assert "identity is not yours to assert" in member.prompt, member.name


def test_every_member_describes_itself_for_the_supervisor():
    """The description is what routing is decided from. Empty is useless."""
    for member in TEAM:
        assert len(member.description) > 40, member.name
