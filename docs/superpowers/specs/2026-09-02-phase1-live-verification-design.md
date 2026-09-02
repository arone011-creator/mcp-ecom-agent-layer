# Phase 1 Live Verification — Design

## Problem

Six of the nine MCP tools — `get_orders`, `get_order`, `get_cart`, `add_to_cart`,
`remove_from_cart`, and `cancel_order`'s success path — are verified against
mocked HTTP responses only (see `docs/TECHNICAL_SNAPSHOT.txt` §12). Everything
needing a signed-in customer falls in this bucket, because no demo credentials
existed when M3 closed.

This matters because it has already happened on this project: four separate
bugs were invisible to a fully green mock/unit test suite and were only ever
caught by a live check. `docs/PLAN_M4_AGENT.txt` §7 lists this as a risk that
should close before M4 builds an agent on top of these tools.

## Approach

No new code is required. `scripts/sweep.py` (built during M3) already exercises
all nine tools through a bare MCP client — including the authenticated cart,
order, and approval-gated cancellation flows — when run with
`--email/--password/--api`. The gap is operational (missing demo credentials),
not technical.

Creating an account and entering a password to authenticate are actions Claude
does not perform under any circumstances, including for a disposable demo/test
account. The account signup, order placement, and authenticated sweep run are
therefore human-only steps.

## Plan

1. **Baseline (Claude, no credentials needed).** Run `sweep.py` unauthenticated
   against the production MCP endpoint. Confirms the 3 already-verified tools
   and the two refusal checks (missing approval, forged approval) still pass,
   and that exactly 9 tools are advertised.
2. **Create the demo account (user).** Sign up a demo customer through the live
   storefront's normal signup flow, then place one order for any product (no
   real payment is ever taken — checkout only writes an order row) so it is
   left `PENDING`/`PROCESSING` and cancellable.
3. **Run the authenticated sweep (user, own terminal).**
   ```bash
   python scripts/sweep.py \
     --url https://mcp-production-e344.up.railway.app/mcp \
     --api https://web-production-bb55d.up.railway.app \
     --email <demo email> --password <demo password>
   ```
4. **Share the output (user → Claude).** The script's stdout summary (pass/fail
   list + per-tool latency table) contains no credentials and is safe to paste
   back.
5. **Update documentation (Claude).**
   - `docs/TECHNICAL_SNAPSHOT.txt` §12: move all 6 tools from
     "VERIFIED AGAINST MOCKS ONLY" to "VERIFIED AGAINST PRODUCTION".
   - `docs/PLAN_M4_AGENT.txt` §7: drop the "six of the nine tools are still
     unverified" risk item.
   - `README.md`: update if it references verification status.
6. **`.gitignore` the `metrics/` output directory.** `scripts/sweep.py` writes
   a point-in-time latency snapshot (`metrics/mcp-latency.json`) that goes
   stale immediately; it is not currently ignored. Track the qualitative
   pass/fail status in prose (step 5), not a generated snapshot in git.
7. **Commit and push** the documentation updates, after user review.

## Out of scope

- Building new test infrastructure — `sweep.py` already covers this.
- The other four Phase 1 gaps from `docs/mcp/open-questions.md` (prompt
  injection design pass, the in-memory nonce store, auth/rate-limiting/
  observability hardening, the Railway private-network workaround) — each is
  independent and gets its own design when picked up.
- Any change to production code, the storefront, or the MCP server.

## Division of labor

| Step | Who |
|---|---|
| Unauthenticated baseline sweep | Claude |
| Demo account signup | User |
| Placing a cancellable demo order | User |
| Authenticated sweep run | User |
| Sharing sweep output | User |
| Documentation updates | Claude |
| `.gitignore` update | Claude |
| Commit / push | Claude (with user review) |
