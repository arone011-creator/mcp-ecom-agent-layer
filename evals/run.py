"""Run every eval fixture N times against the live agent and report.

Closes a risk carried since the original design: there has never been a
way to measure tool selection. Everything else in this repository proves
a mechanism -- the loop loops, the pause pauses, the prompt holds. None
of it says how often the agent picks the right tools for a real request,
and a model that gets a question right four times in five is a different
fact from one that gets it right.

Costs real tokens on every run. Not part of the pytest suite for exactly
that reason.

    python -m evals.run --runs 5
    python -m evals.run --runs 3 --only orders-recent
    python -m evals.run --gate          # exit 1 if any pass rate < 1
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mcp.types import Tool

import config
from agent.loop import (
    openai_model_call,
    run_turn,
    session_scoped_executor,
)
from agent.tools import AGENT_TOOLS, list_openai_tools, to_openai_tool
from evals.fixtures import Fixture, load_all
from evals.score import RunResult, score_run

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "evals" / "workflows"
OUTPUT = ROOT / "metrics" / "agent-evals.json"

# The demo fixture account. Hardcoded in the storefront's own sign-in
# form so visitors can reach the order flow; not a secret.
DEMO_EMAIL = "customer@example.com"
DEMO_PASSWORD = "demo1234"

# Schemas for stubbed runs. A stub that only offers the tools the right
# answer needs cannot detect the wrong one, so a stubbed fixture is shown
# the SAME surface a live one is -- including the tools it must not call.
STUB_SCHEMAS = {
    "search_products": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "max_price": {"type": "number"},
            "min_rating": {"type": "number"},
        },
    },
    "get_product": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    "check_inventory": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    "get_orders": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    "get_order": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "get_cart": {"type": "object", "properties": {}},
    "add_to_cart": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "quantity": {"type": "integer"},
        },
        "required": ["product_id"],
    },
    "remove_from_cart": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
    },
    "cancel_order": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}

STUB_DESCRIPTIONS = {
    "search_products": "Search the product catalogue.",
    "get_product": "Get one product by id.",
    "check_inventory": "Check stock for one product.",
    "get_orders": "List the customer's recent orders.",
    "get_order": "Get one order by id.",
    "get_cart": "Read the customer's cart.",
    "add_to_cart": "Add a product to the cart.",
    "remove_from_cart": "Remove a product from the cart.",
    "cancel_order": "Cancel an order the customer placed. High risk.",
}


def percentile(values: list[float], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((p / 100) * len(ordered) + 0.999) - 1))
    return round(ordered[index])


async def bearer_token() -> str:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(
            f"{config.ECOMMERCE_API_BASE_URL}/api/v1/auth/token",
            json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
        )
        response.raise_for_status()
        return response.json()["data"]["token"]


def stub_tools() -> list[dict]:
    return [
        to_openai_tool(
            Tool(
                name=name,
                description=STUB_DESCRIPTIONS[name],
                inputSchema=STUB_SCHEMAS[name],
            )
        )
        for name in sorted(AGENT_TOOLS)
    ]


def stub_executor(stub: dict):
    async def execute(name: str, arguments: dict):
        return stub.get(name, {"ok": True})

    return execute


def approver(fixture: Fixture):
    """How this fixture answers a pause.

    None when no pause is expected, so an unexpected one is refused by
    the loop's own safe default rather than silently granted here.
    """
    if not fixture.expect_approval:
        return None

    async def approve(request):
        return {"approved": fixture.approve}

    return approve


async def clear_cart(token: str) -> None:
    """Empty the cart, so the next run does not inherit this one's state."""
    async with session_scoped_executor(token) as session:
        await session.execute("remove_from_cart", {})


def tools_of(state) -> list[str]:
    return [
        event["data"]["tool"]
        for event in state["events"]
        if event["type"] == "tool_started"
    ]


def approvals_of(state) -> list[str]:
    return [
        event["data"]["tool"]
        for event in state["events"]
        if event["type"] == "approval_required"
    ]


async def one_run(fixture: Fixture, token: str, live_tools: list[dict]):
    usage: list = []
    started = time.monotonic()

    if fixture.is_live:
        async with session_scoped_executor(token) as session:
            state = await run_turn(
                fixture.utterance,
                model_call=openai_model_call(on_usage=usage.append),
                execute_tool=session.execute,
                tools=live_tools,
                approve=approver(fixture),
                session_id=session.session_id,
            )
    else:
        state = await run_turn(
            fixture.utterance,
            model_call=openai_model_call(on_usage=usage.append),
            execute_tool=stub_executor(fixture.stub),
            tools=stub_tools(),
            approve=approver(fixture),
        )

    elapsed_ms = (time.monotonic() - started) * 1000

    if fixture.cleanup == "clear_cart":
        await clear_cart(token)

    result = score_run(
        fixture, tools_of(state), approvals_of(state), state.get("answer")
    )

    return result, elapsed_ms, usage, state.get("answer") or ""


async def run_fixture(fixture: Fixture, token: str, live_tools, runs: int) -> dict:
    print(f"\n===== {fixture.name} ({'live' if fixture.is_live else 'stubbed'}) =====")
    print(f"  {fixture.utterance.strip()}")

    results: list[RunResult] = []
    timings: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0

    for index in range(runs):
        result, elapsed_ms, usage, answer = await one_run(fixture, token, live_tools)

        results.append(result)
        timings.append(elapsed_ms)
        prompt_tokens += sum(u.prompt_tokens for u in usage)
        completion_tokens += sum(u.completion_tokens for u in usage)

        verdict = "PASS" if result.passed else "FAIL"
        print(f"\n  run {index + 1}: {verdict}  {round(elapsed_ms)}ms")
        for reason in result.reasons:
            print(f"    ! {reason}")
        # The answer prints beside every verdict, pass or fail. Task 6's
        # first detector scored a correct answer as a failure; a harness
        # nobody can check is one nobody trusts.
        print(f"    > {answer[:200].strip()}")

    passed = sum(1 for r in results if r.passed)
    print(f"\n  {fixture.name}: {passed}/{runs} passed")

    return {
        "passRate": passed / runs,
        "toolAccuracy": round(sum(r.tool_accuracy for r in results) / runs, 3),
        "unexpectedToolCalls": sum(len(r.unexpected) for r in results),
        "p50": percentile(timings, 50),
        "p95": percentile(timings, 95),
        "promptTokens": round(prompt_tokens / runs),
        "completionTokens": round(completion_tokens / runs),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--only", default=None, help="one fixture name")
    parser.add_argument("--gate", action="store_true", help="exit 1 on any passRate < 1")
    args = parser.parse_args()

    fixtures = load_all(WORKFLOWS)
    if args.only:
        fixtures = [f for f in fixtures if f.name == args.only]
        if not fixtures:
            print(f"no fixture named {args.only}")
            return 2

    token = await bearer_token()
    live_tools = await list_openai_tools(token, only=AGENT_TOOLS)

    workflows = {}
    for fixture in fixtures:
        workflows[fixture.name] = await run_fixture(
            fixture, token, live_tools, args.runs
        )

    report = {
        "generatedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model": config.OPENAI_MODEL,
        "runs": args.runs,
        "workflows": workflows,
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")

    failing = [name for name, w in workflows.items() if w["passRate"] < 1]
    print("\n===== SUMMARY =====")
    for name, w in workflows.items():
        print(
            f"  {name:<38} pass {w['passRate']:.2f}  "
            f"acc {w['toolAccuracy']:.2f}  p50 {w['p50']}ms  "
            f"tok {w['promptTokens']}+{w['completionTokens']}"
        )

    if failing:
        print(f"\nBELOW 1.0: {failing}")
        # A workflow that works four times in five is broken, not slow --
        # the same call the scorecard makes for an MCP success rate.
        if args.gate:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
