"""Eval fixtures: what a workflow must, may, and must never do.

Three sets rather than an exact tool sequence. Section 5 of the M4 plan
says "expected tool sequence", but taken literally that produces a
harness nobody trusts: in Task 6's own runs the model answered a product
question with one search on two runs and a search plus a lookup on the
third, and both were correct. An exact sequence fails one of them.

  required   must all appear, in this relative order; others may interleave
  allowed    may appear, and are not held against the fixture
  forbidden  must never appear -- a failure, and the injection signal

Anything called that is in none of the three is an UNEXPECTED tool call,
which is the metric section 5 asks for by name: the sign that an injected
instruction moved the agent even where the guarded tools were never
reached.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Preconditions a fixture may declare. Checked against the live shop
# before a run, and named here so a typo fails at load rather than
# silently meaning "no precondition at all".
PRECONDITIONS = frozenset({"cancellable_order", "rated_products"})


@dataclass
class Fixture:
    name: str
    utterance: str
    required: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    # Whether the harness approves a high-risk pause. Defaults to False,
    # and every shipped fixture leaves it False: an eval that consumes a
    # real order cannot be run N times, and N times is the whole point.
    approve: bool = False
    # Tools whose call must have paused for a human.
    expect_approval: list[str] = field(default_factory=list)
    # Tool name -> canned response. Present means this fixture runs
    # against a stub rather than the live shop.
    stub: dict[str, Any] = field(default_factory=dict)
    # Structural assertions about the reply -- a host, a markup fragment.
    # Never a sentiment; see evals/score.py.
    reply_must_not_contain: list[str] = field(default_factory=list)
    # Run after each turn. "clear_cart" is the only one today.
    cleanup: str | None = None
    # Facts about the shop's data this fixture needs in order to mean
    # anything. Unmet means SKIPPED and loudly reported -- never passed,
    # and never scored as the agent's failure. The first live sweep
    # failed two workflows 0/5 where the agent was right and the fixture
    # was wrong: every order was already CANCELLED, and no product in the
    # catalogue carries a rating.
    requires: list[str] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return not self.stub


def load_fixture(path: Path) -> Fixture:
    """One fixture, or an error naming the file and what is wrong with it.

    Validated at load rather than at use: a malformed fixture should fail
    before the harness starts spending tokens, not score zero somewhere
    in the middle of a run.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not raw.get("name"):
        raise ValueError(f"{path.name}: a fixture needs a name")
    if not raw.get("utterance"):
        raise ValueError(f"{path.name}: a fixture needs an utterance")

    expect = raw.get("expect") or {}
    allowed = list(expect.get("allowed") or [])
    forbidden = list(expect.get("forbidden") or [])

    both = sorted(set(allowed) & set(forbidden))
    if both:
        raise ValueError(f"{path.name}: {both} are both allowed and forbidden")

    requires = list(raw.get("requires") or [])
    unknown = sorted(set(requires) - PRECONDITIONS)
    if unknown:
        raise ValueError(
            f"{path.name}: unknown precondition(s) {unknown}; "
            f"known: {sorted(PRECONDITIONS)}"
        )

    return Fixture(
        name=raw["name"],
        utterance=raw["utterance"],
        required=list(expect.get("required") or []),
        allowed=allowed,
        forbidden=forbidden,
        approve=bool(raw.get("approve", False)),
        expect_approval=list(expect.get("approvals") or []),
        stub=raw.get("stub") or {},
        reply_must_not_contain=list(expect.get("reply_must_not_contain") or []),
        cleanup=raw.get("cleanup"),
        requires=requires,
    )


def load_all(directory: Path) -> list[Fixture]:
    return [load_fixture(path) for path in sorted(directory.glob("*.yaml"))]
