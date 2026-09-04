"""Naming a conversation, and what a name may contain.

Phase 4 of the chat-persistence roadmap. One model call, once, after a
chat's first exchange. The storefront falls back to the customer's own
first message whenever this fails, so nothing here is allowed to be
load-bearing -- a refusal returns None and the chat keeps a usable name.

WHAT A NAME MAY CONTAIN IS THE PART WORTH READING. It is model-written
text derived from an exchange that may have carried untrusted product
copy, and it is rendered in a list, a browser tab and a log line rather
than in a message bubble. So: no URLs, no control characters, one line,
and short enough that the list truncates titles and fallbacks at the same
width.
"""

import re

import config

# The same 60 the storefront truncates a fallback name at, so a title and
# a fallback are cut to the same width and the list stays even. Three
# constants with one value, in three modules that must not import each
# other -- the shared reason is written at each of them.
TITLE_LIMIT = 60

_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

SYSTEM_PROMPT = """\
You name conversations. You are given the first thing a customer asked a \
shopping assistant and the assistant's reply.

Answer with a short label for that conversation - at most six words, no \
quotes, no full stop, no URLs. Describe the SUBJECT, not the assistant: \
"Recent order history", not "Assistant explains orders".

Answer with the label and nothing else."""


def clean_title(raw: object) -> str | None:
    """A usable name, or None if there is not one in here.

    None on anything doubtful rather than a best effort: the storefront
    has a perfectly good fallback -- the customer's own words -- and a
    mangled model answer is worse than that, not better.
    """
    if not isinstance(raw, str):
        return None

    # URLs first: stripping them can empty the string, and everything
    # after this treats an empty string as "no title".
    title = _URL.sub(" ", raw)
    title = _CONTROL.sub(" ", title)
    # One line, one space. A newline in a single-line list silently loses
    # everything after it.
    title = " ".join(title.split())
    # Models answer a naming question in quotes, and a label is not a
    # sentence.
    title = title.strip("\"'").strip().rstrip(".").strip()

    if not title:
        return None

    return title[:TITLE_LIMIT]


def _openai_client():
    """Seam. Tests replace this rather than the SDK underneath it."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(timeout=config.OPENAI_TIMEOUT_SECONDS)


async def name_conversation(utterance: str, answer: str) -> str | None:
    """Ask the model for a label. Raw -- the caller cleans it.

    BOTH HALVES OF THE EXCHANGE. The subject of a chat is often not in
    the question: "and the second one?" names nothing on its own, and the
    opening question of a chat is frequently that vague.

    NO TOOLS, NO SESSION, NO CUSTOMER TOKEN. Naming a conversation is a
    text task. Giving this endpoint tools would put a second, quieter
    path to the customer's orders beside the one the approval design
    guards.
    """
    client = _openai_client()

    response = await client.chat.completions.create(
        model=config.OPENAI_MODEL,
        # A label. Room for the model to be a little verbose before the
        # cap trims it, and not a token more.
        max_completion_tokens=32,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Customer asked: {utterance}\n\nAssistant replied: {answer}",
            },
        ],
    )

    return response.choices[0].message.content
