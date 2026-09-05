"""The team, expressed as tools the supervisor can call.

DELEGATION IS AN ORDINARY TOOL CALL, and that is the whole trick. The
frozen event contract already has tool_started and tool_completed; the
storefront already renders them as chips and step lists; replay() already
orders them. Inventing a `delegation` event would have meant changing a
contract that exists in two languages with a golden fixture between them,
and changing the storefront to render it. Reusing tool events costs
nothing and the storefront needs no change at all.
"""

from agent.team import TEAM, Member

# Every delegation tool starts with this, which is how the supervisor's
# graph tells a delegation apart from anything else without a lookup.
DELEGATION_PREFIX = "ask_"


def delegation_tool_name(member_name: str) -> str:
    return f"{DELEGATION_PREFIX}{member_name}"


def delegation_tools(team: tuple[Member, ...] = TEAM) -> list[dict]:
    """One function schema per member, generated rather than written.

    Generated so that adding a specialist to TEAM is the only edit
    required. A hand-maintained list beside the team is a second
    implementation of the same fact, and it drifts the first time
    somebody renames a member.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": delegation_tool_name(member.name),
                "description": member.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": (
                                "A complete, self-contained instruction for "
                                "this specialist. It cannot see the "
                                "conversation, so include every identifier "
                                "and detail it needs."
                            ),
                        }
                    },
                    "required": ["request"],
                    "additionalProperties": False,
                },
            },
        }
        for member in team
    ]


def is_delegation(tool_name: str) -> bool:
    return tool_name.startswith(DELEGATION_PREFIX)


def member_for_tool(tool_name: str, team: tuple[Member, ...] = TEAM) -> Member:
    """The member a delegation tool routes to, or KeyError.

    Raising rather than returning None: a name that matches nothing means
    the model invented a specialist, and treating that as "no delegation"
    would turn an invented capability into a turn that silently did less
    than it claimed.
    """
    if not is_delegation(tool_name):
        raise KeyError(f"{tool_name!r} is not a delegation tool")

    wanted = tool_name[len(DELEGATION_PREFIX) :]

    for member in team:
        if member.name == wanted:
            return member

    raise KeyError(f"No team member named {wanted!r}")
