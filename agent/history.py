"""Replayed conversation history: what may come back in, what goes out.

Phase 5 of the chat-persistence roadmap. The storefront owns the
conversation; this agent stays stateless and is handed the earlier turns
as a request parameter. Two directions, two functions:

    sanitise_history()    what the storefront sends BACK IN, checked
    exportable_context()  what this turn added, sent out to be stored

WHY THE INCOMING SIDE IS CHECKED AT ALL. The blob was written by this
agent, into the storefront's own database, and returns over the
service-key channel. Three things would each have to be wrong before a
hostile message arrived -- and if all three are, the consequence is that
somebody else writes this agent's system prompt. A dictionary lookup per
message is not a cost worth saving against that.

AN ALLOWLIST, NOT A BAN ON `system`. The roadmap says "refuse any system
role". Written literally that is a rule about one string, and the OpenAI
API already has a second role carrying the same authority -- `developer`
-- which such a rule waves straight through. A turn produces exactly
three roles. Exactly three are accepted.
"""

# The only roles a turn of this agent can produce: the customer's message,
# the model's replies, and the tool results fed back to it.
REPLAYABLE_ROLES = frozenset({"user", "assistant", "tool"})


class UnsafeHistory(ValueError):
    """Replayed history contained something a turn could not have produced."""


def sanitise_history(raw: object) -> list[dict]:
    """Check what the storefront sent back, or refuse the lot.

    Refuses rather than filters. A row carrying a role a turn cannot
    produce has been written by something other than a turn, and the
    messages sitting next to it are not more trustworthy for the company
    they keep. The storefront drops such a row before it ever gets here
    (its own Task 4); this is the layer that does not depend on that one
    being right.
    """
    if raw is None:
        return []

    if not isinstance(raw, list):
        raise UnsafeHistory("Replayed history must be a list of messages")

    for message in raw:
        if not isinstance(message, dict):
            raise UnsafeHistory("Every replayed entry must be a message object")

        if message.get("role") not in REPLAYABLE_ROLES:
            # The offending value is deliberately NOT quoted into the
            # message: this goes to a log, and it came out of stored data.
            raise UnsafeHistory("Replayed history carries a role a turn cannot produce")

    return list(raw)


def exportable_context(state: dict) -> list[dict]:
    """This turn's own messages, for the storefront to store.

    Everything the turn was SEEDED with is dropped:

      - the system prompt, which the agent builds fresh every turn and
        which must never exist in a row that something could later edit;
      - the replayed history, which the storefront already has. Without
        this half, turn 3's row would contain turns 1 and 2 again, turn
        4's would contain three copies of turn 1, and replay would feed
        the model the same exchange once for every turn that followed it.

    `seeded` is written into the state once, by run_turn, and no node
    returns it -- so it still says how long the seed was after the graph
    has appended to `messages` many times over.
    """
    messages = state.get("messages", [])
    # Default 1: a state that somehow lost the count exports slightly too
    # much rather than exporting the system prompt.
    seeded = state.get("seeded", 1)

    return list(messages[seeded:])
