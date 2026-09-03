"""Marks free text that reaches an agent's context as untrusted data.

Any field an admin or (eventually) a customer can write, that flows into a
tool's response, is an injection vector: text a model reads as part of its
context, indistinguishable from an instruction unless something marks the
boundary. This module is that boundary -- applied once, at the data
contract in models/schemas.py, so no call site has to remember it.

Structural hygiene only. No phrase-based denylisting (blocking strings
like "ignore previous instructions") -- bypassable, and it buys false
confidence. See docs/superpowers/specs/2026-09-02-prompt-injection-design-
pass.md for the full threat model and the pieces this does not cover
(those wait for an agent to exist).
"""

import re

# C0 controls except \t \n \r (legitimate in prose), plus DEL, plus the
# Unicode bidi-override block -- a documented technique for visually
# disguising injected text (e.g. making it read differently on screen
# than it parses).
_STRIP_PATTERN = re.compile(
    "["
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    "\u200e\u200f"
    "\u202a-\u202e"
    "\u2066-\u2069"
    "]"
)

# Bounds worst-case payload size and the context cost of a single field.
# Generous for a product description; not a limit anything legitimate
# should ever hit.
MAX_LENGTH = 4000
_TRUNCATION_MARKER = "...[truncated]"


def mark_untrusted(text: str | None) -> str | None:
    """Wrap admin/customer-authored free text as an inert data boundary.

    Strips control and bidi-override characters, caps length, then wraps
    the result in an XML-style tag. The tag is inert on its own: it means
    something only because the agent's system prompt says what it means
    (agent/prompt.py), and the technique is not specific to any one model
    -- this agent runs on gpt-4.1 as of M4 Task 2. Every free-text field
    that reaches an agent through this server must be run through this,
    including reviews when that creation path ships.
    """
    if text is None:
        return None

    cleaned = _STRIP_PATTERN.sub("", text)

    if len(cleaned) > MAX_LENGTH:
        cleaned = cleaned[:MAX_LENGTH] + _TRUNCATION_MARKER

    return f"<untrusted-user-content>{cleaned}</untrusted-user-content>"
