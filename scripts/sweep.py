"""Exercise every tool through a bare MCP client, with no agent involved.

This is what isolates "is it the MCP layer or the agent" when something
breaks in Phase 2. It also produces the per-tool latency floor that Phase 2
must not regress below.

Queries are varied on purpose. searchProducts sits behind a 300-second
unstable_cache, so repeating one term measures the cache rather than the
system, and an agent issuing real questions will never see those numbers.

Two modes:

  Without credentials -- the public tools, plus every refusal that does not
  need a signed-in customer. Honest but partial: four tools need a bearer
  token and are skipped, and only the *refusal* half of cancel_order can be
  checked.

  With --email/--password -- the full surface, including cancelling a real
  order behind a real approval.

  python scripts/sweep.py --url https://mcp-production-e344.up.railway.app/mcp
  python scripts/sweep.py --url ... --api https://... --email demo@x.com --password '...'
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

TERMS = ["shoes", "iphone", "shirt", "laptop", "headphones", "cotton", "galaxy", "air"]
SAMPLES = 8


def percentile(values: list[float], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((p / 100) * len(ordered) + 0.999) - 1))
    return round(ordered[index])


class Results:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def record(self, tool: str, elapsed_ms: float, ok: bool) -> None:
        entry = self.data.setdefault(tool, {"timings": [], "ok": 0, "n": 0})
        entry["timings"].append(elapsed_ms)
        entry["ok"] += int(ok)
        entry["n"] += 1

    def summary(self) -> dict[str, dict]:
        return {
            tool: {
                "p50": percentile(entry["timings"], 50),
                "p95": percentile(entry["timings"], 95),
                "successRate": round(entry["ok"] / entry["n"], 3),
            }
            for tool, entry in self.data.items()
        }


async def timed(client: Client, results: Results, name: str, args: dict):
    started = time.perf_counter()
    try:
        result = await client.call_tool(name, args)
        results.record(name, (time.perf_counter() - started) * 1000, True)
        return result
    except Exception as error:  # noqa: BLE001 - a failure is a data point
        results.record(name, (time.perf_counter() - started) * 1000, False)
        print(f"    {name} failed: {str(error).splitlines()[0][:110]}")
        return None


def payload(result):
    return json.loads(result.content[0].text) if result else None


async def get_token(api: str, email: str, password: str) -> str:
    async with httpx.AsyncClient(timeout=20) as http:
        response = await http.post(
            f"{api}/api/v1/auth/token",
            # Short-lived on purpose: these JWTs cannot be revoked.
            json={"email": email, "password": password, "ttlSeconds": 900},
        )
        response.raise_for_status()
        return response.json()["data"]["token"]


async def mint_approval(url: str, token: str, session: str, order_id: str) -> str:
    base = url.rsplit("/mcp", 1)[0]
    async with httpx.AsyncClient(timeout=20) as http:
        response = await http.post(
            f"{base}/approvals",
            json={"tool": "cancel_order", "args": {"order_id": order_id}},
            headers={"authorization": f"Bearer {token}", "mcp-session-id": session},
        )
        response.raise_for_status()
        return response.json()["token"]


async def run(args) -> int:
    failures: list[str] = []
    results = Results()

    authenticated = bool(args.email and args.password)
    token = "placeholder"

    if authenticated:
        token = await get_token(args.api, args.email, args.password)
        print("  authenticated: yes")
    else:
        # The product routes are public on the API side, so a placeholder
        # still exercises the whole path for those. Everything needing a
        # real customer is skipped and reported as skipped.
        print("  authenticated: no -- cart and order tools will be skipped")

    transport = StreamableHttpTransport(args.url, headers={"authorization": f"Bearer {token}"})

    async with Client(transport) as client:
        advertised = sorted(tool.name for tool in await client.list_tools())
        print(f"  tools advertised: {len(advertised)}")
        if len(advertised) != 9:
            failures.append(f"expected 9 tools, found {len(advertised)}: {advertised}")

        print("\n  reads")
        for index in range(SAMPLES):
            await timed(
                client,
                results,
                "search_products",
                {"query": TERMS[index % len(TERMS)], "limit": 5},
            )

        found = payload(
            await timed(client, results, "search_products", {"query": "", "limit": 1})
        )
        product_id = None
        if found and found.get("products"):
            product_id = found["products"][0]["id"]

        if not product_id:
            failures.append("no product found; the catalogue may be empty")
        else:
            for _ in range(SAMPLES):
                await timed(client, results, "get_product", {"product_id": product_id})
                await timed(client, results, "check_inventory", {"product_id": product_id})

        order_id = None

        if authenticated:
            print("\n  customer reads and writes")
            for _ in range(SAMPLES):
                await timed(client, results, "get_orders", {"limit": 5})
                await timed(client, results, "get_cart", {})

            if product_id:
                for _ in range(SAMPLES):
                    await timed(
                        client,
                        results,
                        "add_to_cart",
                        {"product_id": product_id, "quantity": 1},
                    )
                    await timed(
                        client, results, "remove_from_cart", {"product_id": product_id}
                    )

            orders = payload(await timed(client, results, "get_orders", {"limit": 20})) or []
            cancellable = [o for o in orders if o["status"] in ("PENDING", "PROCESSING")]
            if cancellable:
                order_id = cancellable[0]["id"]
            else:
                print("    no cancellable order; place one to sweep cancel_order fully")
        else:
            print("\n  customer reads and writes: SKIPPED (no credentials)")

        # The security assertion, run against the live server. This half
        # needs no credentials: a cancel with no approval must fail even
        # though the caller is otherwise well-formed.
        print("\n  security")
        try:
            await client.call_tool("cancel_order", {"order_id": order_id or "probe"})
            failures.append("cancel_order SUCCEEDED without an approval token")
            print("    cancel_order without approval: SUCCEEDED  <-- FAIL")
        except Exception as error:  # noqa: BLE001
            print(f"    cancel_order without approval: refused ({str(error).splitlines()[0][-60:]})")

        try:
            await client.call_tool(
                "cancel_order", {"order_id": order_id or "probe", "approval_token": "forged.token"}
            )
            failures.append("cancel_order SUCCEEDED with a forged approval token")
            print("    cancel_order with a forged token: SUCCEEDED  <-- FAIL")
        except Exception as error:  # noqa: BLE001
            print(f"    cancel_order with a forged token: refused ({str(error).splitlines()[0][-60:]})")

        if authenticated and order_id:
            session = transport.__dict__.get("_session_id") or "sweep"
            try:
                approval = await mint_approval(args.url, token, session, order_id)
                await timed(
                    client,
                    results,
                    "cancel_order",
                    {"order_id": order_id, "approval_token": approval},
                )
                print(f"    cancel_order with a valid approval: order {order_id} cancelled")
            except Exception as error:  # noqa: BLE001
                failures.append(f"approved cancel_order failed: {error}")
        else:
            print("    cancel_order with a valid approval: SKIPPED")

    summary = results.summary()

    print()
    for tool, timing in sorted(summary.items()):
        print(
            f"  {tool:20} p50 {timing['p50']:>5}ms  p95 {timing['p95']:>5}ms  "
            f"ok {timing['successRate']}"
        )

    missing = sorted(set(advertised) - set(summary))
    if missing:
        print(f"\n  not measured: {', '.join(missing)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n  wrote {out}")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="MCP endpoint, ending in /mcp")
    parser.add_argument("--api", help="Storefront API root, required with credentials")
    parser.add_argument("--email")
    parser.add_argument("--password")
    parser.add_argument("--out", default="metrics/mcp-latency.json")
    args = parser.parse_args()

    if (args.email or args.password) and not (args.email and args.password and args.api):
        parser.error("--email, --password and --api must be given together")

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
