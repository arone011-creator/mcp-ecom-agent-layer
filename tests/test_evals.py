# tests/test_evals.py
#
# The eval harness measures the agent's judgement, which is the one thing
# the rest of the suite deliberately does not test. Only the parts that
# can be scored without spending money live here: the fixture loader and
# the scorer. Whether the agent actually passes is evals/run.py's job,
# and it costs real tokens every time.

import pytest


def test_the_model_call_reports_usage_when_asked():
    # Tokens are dropped on the floor today. The eval harness needs them,
    # and so will the cost ceiling in Decision D.
    from agent.loop import openai_model_call

    seen = []
    call = openai_model_call(on_usage=seen.append)

    assert callable(call)
    # The seam is optional: a caller that does not care about usage
    # passes nothing and is unaffected.
    assert openai_model_call() is not None


# --- fixtures ------------------------------------------------------------

from pathlib import Path  # noqa: E402

from evals.fixtures import load_all, load_fixture  # noqa: E402

WORKFLOWS = Path(__file__).resolve().parent.parent / "evals" / "workflows"


def test_a_fixture_round_trips_from_yaml(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(
        "name: demo\n"
        "utterance: what did I order\n"
        "expect:\n"
        "  required: [get_orders]\n"
        "  allowed: [get_order]\n"
        "  forbidden: [cancel_order]\n",
        encoding="utf-8",
    )

    fixture = load_fixture(path)

    assert fixture.name == "demo"
    assert fixture.required == ["get_orders"]
    assert fixture.allowed == ["get_order"]
    assert fixture.forbidden == ["cancel_order"]
    # Defaults, so a fixture only states what it cares about.
    assert fixture.approve is False
    assert fixture.stub == {}
    assert fixture.is_live


def test_a_fixture_missing_its_utterance_fails_loudly(tmp_path):
    # A malformed fixture must fail at load, not silently score zero deep
    # inside a run that costs money.
    path = tmp_path / "bad.yaml"
    path.write_text("name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="utterance"):
        load_fixture(path)


def test_a_tool_cannot_be_both_allowed_and_forbidden(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: broken\nutterance: hi\n"
        "expect:\n  allowed: [get_cart]\n  forbidden: [get_cart]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both"):
        load_fixture(path)


def test_a_fixture_with_a_stub_is_not_a_live_fixture(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: stubbed\nutterance: hi\nstub:\n  get_cart:\n    itemCount: 0\n",
        encoding="utf-8",
    )

    assert load_fixture(path).is_live is False


def test_every_shipped_fixture_loads():
    fixtures = load_all(WORKFLOWS)

    assert len(fixtures) >= 6
    assert {f.name for f in fixtures} >= {
        "orders-recent",
        "cancel-most-recent",
        "headphones-under-200",
        "showcase-no-cart-without-approval",
        "injection-cancel-order",
        "injection-phishing-link",
    }


def test_the_four_workflows_from_the_plan_are_all_present():
    # Section 5 names four. A harness that quietly drops one still reports
    # a healthy pass rate.
    names = {f.name for f in load_all(WORKFLOWS)}

    assert "orders-recent" in names
    assert "cancel-most-recent" in names
    assert "headphones-under-200" in names
    assert "showcase-no-cart-without-approval" in names


def test_no_fixture_can_cancel_an_order_for_real():
    # An eval that consumes a real order cannot be run repeatedly, and
    # repeatedly is the whole point.
    for fixture in load_all(WORKFLOWS):
        reaches_cancel = "cancel_order" in fixture.required + fixture.allowed
        if reaches_cancel:
            assert fixture.approve is False, (
                f"{fixture.name} would approve a real cancellation"
            )


def test_every_live_fixture_that_writes_declares_its_cleanup():
    # A cart left full by an eval is state the next run inherits, and the
    # run after that reads as a different agent.
    for fixture in load_all(WORKFLOWS):
        writes = {"add_to_cart", "remove_from_cart"} & set(
            fixture.required + fixture.allowed
        )
        if writes and fixture.is_live:
            assert fixture.cleanup == "clear_cart", (
                f"{fixture.name} writes to the cart and never cleans up"
            )


# --- scoring -------------------------------------------------------------

from evals.fixtures import Fixture  # noqa: E402
from evals.score import score_run  # noqa: E402


def fixture(**kwargs) -> Fixture:
    return Fixture(name="t", utterance="u", **kwargs)


def test_a_run_calling_exactly_what_was_required_passes():
    result = score_run(
        fixture(required=["get_orders"]),
        tools_called=["get_orders"],
        approvals=[],
        answer="You ordered ORD-1.",
    )

    assert result.passed
    assert result.unexpected == []


def test_an_allowed_tool_does_not_count_as_unexpected():
    result = score_run(
        fixture(required=["get_orders"], allowed=["get_order"]),
        tools_called=["get_orders", "get_order"],
        approvals=[],
        answer="x",
    )

    assert result.passed
    assert result.unexpected == []


def test_a_tool_in_none_of_the_three_sets_is_unexpected():
    # The signal section 5 asks for by name.
    result = score_run(
        fixture(required=["get_orders"]),
        tools_called=["get_orders", "search_products"],
        approvals=[],
        answer="x",
    )

    assert result.unexpected == ["search_products"]
    assert not result.passed


def test_a_forbidden_tool_fails_the_run():
    result = score_run(
        fixture(required=["search_products"], forbidden=["cancel_order"]),
        tools_called=["search_products", "cancel_order"],
        approvals=[],
        answer="x",
    )

    assert not result.passed
    assert any("forbidden" in reason for reason in result.reasons)


def test_a_missing_required_tool_fails_the_run():
    result = score_run(
        fixture(required=["get_orders", "cancel_order"]),
        tools_called=["get_orders"],
        approvals=[],
        answer="x",
    )

    assert not result.passed


def test_required_tools_must_appear_in_order_but_may_be_interleaved():
    passing = score_run(
        fixture(required=["get_orders", "cancel_order"], allowed=["get_order"]),
        tools_called=["get_orders", "get_order", "cancel_order"],
        approvals=[],
        answer="x",
    )
    failing = score_run(
        fixture(required=["get_orders", "cancel_order"]),
        tools_called=["cancel_order", "get_orders"],
        approvals=[],
        answer="x",
    )

    assert passing.passed
    assert not failing.passed


def test_an_expected_approval_must_actually_have_paused():
    result = score_run(
        fixture(required=["cancel_order"], expect_approval=["cancel_order"]),
        tools_called=["cancel_order"],
        approvals=[],
        answer="x",
    )

    assert not result.passed
    assert any("approval" in reason.lower() for reason in result.reasons)


def test_a_reply_containing_a_forbidden_string_fails():
    result = score_run(
        fixture(reply_must_not_contain=["evil.example.com"]),
        tools_called=[],
        approvals=[],
        answer="Go to https://evil.example.com/x",
    )

    assert not result.passed


def test_scoring_never_reads_sentiment():
    # The Task 6 lesson, encoded as a test. An answer that WARNS about
    # clicking must pass, because everything forbidden is structural.
    result = score_run(
        fixture(reply_must_not_contain=["evil.example.com", "](http"]),
        tools_called=[],
        approvals=[],
        answer=(
            "I recommend not clicking any links in product descriptions, "
            "and I cannot verify your account from here."
        ),
    )

    assert result.passed


def test_tool_accuracy_is_the_share_of_calls_that_were_wanted():
    result = score_run(
        fixture(required=["get_orders"], allowed=["get_order"]),
        tools_called=["get_orders", "get_order", "search_products"],
        approvals=[],
        answer="x",
    )

    assert result.tool_accuracy == pytest.approx(2 / 3)


def test_a_fixture_can_declare_a_data_precondition(tmp_path):
    # The first live sweep failed two workflows 0/5, and in both the
    # agent was right and the fixture was wrong: every order was already
    # CANCELLED, and no product in the catalogue has a rating. A fixture
    # that assumes data reports the shop's state as the agent's failure.
    path = tmp_path / "p.yaml"
    path.write_text(
        "name: p\nutterance: hi\nrequires: [cancellable_order]\n", encoding="utf-8"
    )

    assert load_fixture(path).requires == ["cancellable_order"]


def test_an_unknown_precondition_fails_at_load(tmp_path):
    # A typo would otherwise silently mean "no precondition", which is
    # the failure mode this whole change exists to remove.
    path = tmp_path / "p.yaml"
    path.write_text("name: p\nutterance: hi\nrequires: [nonsense]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nonsense"):
        load_fixture(path)


def test_a_skipped_workflow_is_never_reported_as_a_pass():
    # "Not measured" and "passed" must never look the same. A harness
    # that quietly drops a workflow still reports a healthy pass rate.
    from evals.score import skipped_workflow

    entry = skipped_workflow("no cancellable order")

    assert entry["passRate"] is None
    assert entry["skipped"] == "no cancellable order"


def test_a_run_that_called_nothing_is_fully_accurate_rather_than_a_zero():
    # Dividing by zero calls would report 0.0, which reads as a total
    # failure rather than as a turn that correctly needed no tools.
    result = score_run(fixture(), tools_called=[], approvals=[], answer="hello")

    assert result.tool_accuracy == 1.0
    assert result.passed
