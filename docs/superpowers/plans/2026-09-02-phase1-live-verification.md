# Phase 1 Live Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the "6 of 9 MCP tools verified against mocks only" gap recorded in `docs/TECHNICAL_SNAPSHOT.txt` §12 and `docs/PLAN_M4_AGENT.txt` §7, by running the existing `scripts/sweep.py` against production with a real demo account, then updating those two docs to match the result.

**Architecture:** No new code. `scripts/sweep.py` already exercises all nine MCP tools through a bare client. Phase A (Claude, no credentials) runs the unauthenticated half now as a regression baseline. Phase B (human-only, per the account-creation/password boundary) creates a demo account, places a cancellable order, and runs the authenticated half. Phase C (Claude) updates documentation from the real result — never from an assumption.

**Tech Stack:** Python 3.11+, the existing `.venv`, `scripts/sweep.py` (FastMCP client + httpx), the deployed production MCP server (`https://mcp-production-e344.up.railway.app`) and storefront API (`https://web-production-bb55d.up.railway.app`).

**Spec:** `docs/superpowers/specs/2026-09-02-phase1-live-verification-design.md`

---

## File Structure

| File | Change |
|---|---|
| `.gitignore` | Add `metrics/` — the sweep writes a point-in-time latency snapshot that goes stale immediately and should not be tracked |
| `metrics/mcp-latency.json` | `git rm --cached` only — stays on disk, stops being tracked |
| `docs/TECHNICAL_SNAPSHOT.txt` | §12: merge "VERIFIED AGAINST MOCKS ONLY" into "VERIFIED AGAINST PRODUCTION"; update the plain-terms paragraph |
| `docs/PLAN_M4_AGENT.txt` | §7: remove the "six of nine tools unverified" risk bullet; fix the "Five known problems" count |

No new files. No source code changes — `scripts/sweep.py` already covers this; this plan only runs it and records the result.

**Note on Tasks 3 and 4:** they require a human browser session and a human-entered password. Per the project's operating rules, account creation and password entry are never done by Claude, even for a disposable demo account. If executing this plan with `subagent-driven-development`, do not dispatch a subagent for Tasks 3–4 — pause the main session and wait for the user to report completion before dispatching Task 5.

---

### Task 1: Stop tracking the generated metrics snapshot

**Files:**
- Modify: `.gitignore`
- Untrack: `metrics/mcp-latency.json`

- [ ] **Step 1: Add `metrics/` to `.gitignore`**

Current file:
```
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.env
.env.local
```

New file:
```
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.env
.env.local
metrics/
```

- [ ] **Step 2: Untrack the existing snapshot (keep it on disk)**

Run: `git rm --cached metrics/mcp-latency.json`
Expected: `rm 'metrics/mcp-latency.json'`. The file still exists on disk (verify with `cat metrics/mcp-latency.json` — unchanged).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git rm --cached metrics/mcp-latency.json
git commit -m "chore: stop tracking the generated sweep-metrics snapshot

It's a point-in-time latency reading that goes stale immediately;
scripts/sweep.py regenerates it locally on every run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Run the unauthenticated baseline sweep

**Files:** none modified — this is a verification run, not a code change.

- [ ] **Step 1: Confirm the venv has the runtime deps**

Run: `.venv/Scripts/python -c "import fastmcp, httpx; print('ok')"`
Expected: `ok`. (If it errors, run `.venv/Scripts/pip install -r requirements.txt` first — `fastmcp`, `httpx` and `pydantic` are all `sweep.py` needs; the dev-only deps in `requirements-dev.txt` are not required for this script.)

- [ ] **Step 2: Run the sweep against production, unauthenticated**

Run:
```bash
.venv/Scripts/python scripts/sweep.py --url https://mcp-production-e344.up.railway.app/mcp
```

Expected, in order:
- `authenticated: no -- cart and order tools will be skipped`
- `tools advertised: 9` (if this number is anything else, STOP — that alone is a regression worth investigating before continuing to Task 3)
- A `reads` section with `search_products`, `get_product`, `check_inventory` timing lines and no `failed` messages under them
- `customer reads and writes: SKIPPED (no credentials)`
- `cancel_order without approval: refused (...)` — must say "refused", not "SUCCEEDED"
- `cancel_order with a forged token: refused (...)` — must say "refused", not "SUCCEEDED"
- `cancel_order with a valid approval: SKIPPED`
- A per-tool `p50`/`p95`/`ok` summary table
- `wrote metrics/mcp-latency.json`
- Exit code `0` (check with `echo $?` on the line after)

- [ ] **Step 3: If anything failed, stop here**

If exit code is non-zero, or either `cancel_order` refusal line says `SUCCEEDED`, this is a live regression in already-shipped behavior — stop, report it to the user, and do not proceed to Task 3 until it's resolved. This plan assumes the baseline is green.

No commit for this task — nothing under version control changed (Task 1 already made the output file untracked).

---

### Task 3: [HUMAN CHECKPOINT] Create a demo account and place a cancellable order

**This task cannot be performed by Claude.** Creating an account and entering a password to authenticate are actions Claude never takes, including for a disposable demo account — see `docs/superpowers/specs/2026-09-02-phase1-live-verification-design.md`.

- [ ] **Step 1 (user): Sign up a demo customer**

Go to `https://web-production-bb55d.up.railway.app` and sign up through the normal storefront signup flow. Use any email/password you're comfortable sharing for this test (it's a demo project — no real payment is ever taken).

- [ ] **Step 2 (user): Place one order**

Add any product to the cart and check out. Checkout only writes an order row — there is no real payment step. Confirm in the UI (or via `GET /api/v1/orders`) that the order's status is `PENDING` or `PROCESSING` (i.e., not already cancelled or delivered) so it's eligible for `cancel_order`.

- [ ] **Step 3 (user): Report back**

Tell Claude the task is done. You do not need to share the password in chat if you'd rather run Task 4 yourself entirely (see Task 4's alternative).

---

### Task 4: [HUMAN CHECKPOINT] Run the authenticated sweep

**This task requires entering a password and should be run by the user, not Claude**, either directly or by exporting credentials as environment variables Claude never sees the value of.

- [ ] **Step 1 (user): Run the full sweep**

From the `mcp-ecom-agent-layer` directory:
```bash
.venv/Scripts/python scripts/sweep.py \
  --url https://mcp-production-e344.up.railway.app/mcp \
  --api https://web-production-bb55d.up.railway.app \
  --email <your demo email> --password <your demo password>
```

Expected, in addition to everything from Task 2's baseline:
- `authenticated: yes`
- A `customer reads and writes` section with `get_orders`, `get_cart`, `add_to_cart`, `remove_from_cart` timing lines and no `failed` messages
- Either `cancel_order with a valid approval: order <id> cancelled`, or — if no cancellable order was found — `no cancellable order; place one to sweep cancel_order fully` (if you see this, Task 3 Step 2 didn't leave an order in a cancellable state; place one and re-run)
- Exit code `0`

- [ ] **Step 2 (user): Share the result**

Paste the full stdout (the summary table plus the pass/fail lines) back to Claude. It contains no credentials — the script never prints the email, password, or bearer token.

- [ ] **Step 3 (Claude): Judge the result**

- If exit code was `0`, every listed tool shows in the summary with `ok 1.0` (or close to it — a single transient timeout is not a correctness failure, but re-run once if you see one), and `cancel_order with a valid approval` printed an order id being cancelled: proceed to Task 5.
- If anything failed: STOP. Do not proceed to Task 5/6. Report the specific failure to the user — this is a real bug the mocks were hiding, which is exactly the scenario this whole exercise exists to catch. Fixing it is a separate, unplanned task.

---

### Task 5: Update `docs/TECHNICAL_SNAPSHOT.txt` — only if Task 4 passed cleanly

**Files:**
- Modify: `docs/TECHNICAL_SNAPSHOT.txt`

- [ ] **Step 1: Replace the verification section**

Find (around line 416–424):
```
VERIFIED AGAINST PRODUCTION
    search_products, get_product, check_inventory - and cancel_order refusing
    both a missing and a forged approval.

VERIFIED AGAINST MOCKS ONLY
    get_orders, get_order, get_cart, add_to_cart, remove_from_cart, and
    cancel_order's SUCCESS path. These need a signed-in customer, and no demo
    credentials were available for the sweep. One run with credentials closes
    the gap.
```

Replace with:
```
VERIFIED AGAINST PRODUCTION
    All nine tools. search_products, get_product and check_inventory were
    verified first; get_orders, get_order, get_cart, add_to_cart,
    remove_from_cart and cancel_order's SUCCESS path were closed by a sweep
    run against a live demo account. cancel_order was also confirmed to
    refuse both a missing and a forged approval.
```

- [ ] **Step 2: Update the plain-terms paragraph**

Find (around line 442–449):
```
    IN PLAIN TERMS
    Most of it is proven against the real, live shop. Six capabilities are
    proven only against a simulation, because testing them needs a real
    customer login that was not available - that is a known, written-down
    gap, not an oversight. The most serious open issue: product reviews are
    written by the public and the AI reads them, so a carefully worded review
    could try to give the AI instructions. That must be solved before this is
    used for real.
```

Replace with:
```
    IN PLAIN TERMS
    All of it is now proven against the real, live shop, not just a
    simulation. The most serious open issue left: product reviews are
    written by the public and the AI reads them, so a carefully worded review
    could try to give the AI instructions. That must be solved before this is
    used for real.
```

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -n "MOCKS ONLY\|Six capabilities" docs/TECHNICAL_SNAPSHOT.txt`
Expected: no output.

---

### Task 6: Update `docs/PLAN_M4_AGENT.txt` — only if Task 4 passed cleanly

**Files:**
- Modify: `docs/PLAN_M4_AGENT.txt`

- [ ] **Step 1: Remove the resolved risk bullet**

Find (around line 337–344):
```
    * SINGLE-USE APPROVALS HOLD FOR ONE REPLICA ONLY. The spent-nonce set
      is in process memory. Before this scales, it needs a shared store.

    * SIX OF THE NINE TOOLS ARE STILL UNVERIFIED AGAINST PRODUCTION. They
      pass against mocks. Four bugs on this project were invisible to
      mocks. Run the Phase 1 sweep with real credentials before trusting
      them under an agent.

    * NO TIMEOUT ON A PAUSED CONVERSATION. Flagged in the design document
```

Replace with:
```
    * SINGLE-USE APPROVALS HOLD FOR ONE REPLICA ONLY. The spent-nonce set
      is in process memory. Before this scales, it needs a shared store.

    * NO TIMEOUT ON A PAUSED CONVERSATION. Flagged in the design document
```

- [ ] **Step 2: Fix the risk count**

Find (around line 353):
```
    Five known problems carried into this stage. Two are worth watching
```

Replace with:
```
    Four known problems carried into this stage. Two are worth watching
```

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -n "SIX OF THE NINE\|Five known problems" docs/PLAN_M4_AGENT.txt`
Expected: no output.

---

### Task 7: Commit and push the documentation update

**Files:**
- `docs/TECHNICAL_SNAPSHOT.txt`
- `docs/PLAN_M4_AGENT.txt`

- [ ] **Step 1: Review the diff**

Run: `git diff docs/TECHNICAL_SNAPSHOT.txt docs/PLAN_M4_AGENT.txt`
Confirm only the intended sections changed.

- [ ] **Step 2: Confirm `README.md` needs no matching update**

Run: `grep -n -i "verif\|unverified\|mocks only" README.md`
Expected: no output — `README.md`'s "Known limitations" section doesn't currently list the tool-verification gap, so nothing there needs changing. If this now prints a match, update it to match Task 5/6's wording before committing.

- [ ] **Step 3: Commit**

```bash
git add docs/TECHNICAL_SNAPSHOT.txt docs/PLAN_M4_AGENT.txt
git commit -m "docs: record all nine MCP tools as verified against production

The six tools needing a signed-in customer were closed by a sweep run
against a live demo account (including cancel_order's approved success
path), not just the mocked test suite.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Push, after user confirmation**

```bash
git push origin main
```
