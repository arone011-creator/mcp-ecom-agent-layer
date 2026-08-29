# Technical Snapshot — mcp-ecom-agent-layer

**As of 29 August 2026. Phase 1 shipped; no agent exists yet.**

A complete picture of what this service is, what it is built from, and how
it connects to everything else. Every section has an **In plain terms**
note, so a non-technical reader can follow the whole document by reading
only those.

> **In plain terms:** This is a translator. It sits between an AI assistant
> and an online shop, and it gives the AI a small, safe set of things it is
> allowed to do — search products, check stock, look at orders, use the
> cart, and cancel an order. This document explains how it works and what
> stops it doing anything worse.

---

## 1. What this is

An **MCP server**. MCP — Model Context Protocol — is a standard way to
describe capabilities ("tools") to an AI client so it can call them.

This server exposes nine capabilities from the
[mcp-ecom](https://github.com/arone011-creator/mcp-ecom) storefront.

| | |
|---|---|
| Language | Python 3.11+ (3.12.7 in development) |
| Framework | FastMCP 2.14.7 |
| HTTP client | httpx 0.28.1 |
| Validation | Pydantic 2.13.5 |
| Tests | pytest + respx — 108 tests |
| Source size | ~920 lines, plus a 263-line sweep script |
| Live at | `mcp-production-e344.up.railway.app` |

**There is no AI in this repository.** That is the point of Phase 1:
building the tools and the reasoning at the same time would confound two
unknowns — *do my tools work* and *does my orchestration work*. The agent
arrives in Phase 2, on top of a tool layer already proven.

> **In plain terms:** This is the set of buttons an AI is allowed to press.
> The AI itself comes later, on purpose. If you build the buttons and the
> AI at once and something breaks, you cannot tell which half is at fault.

---

## 2. The core principle

```
MCP client  →  this server  →  mcp-ecom /api/v1  →  PostgreSQL

NOT:        →  this server  →  PostgreSQL
```

This server is an **adapter**. It never touches the database, and it never
re-implements a business rule. "May this order be cancelled?" is answered
by the storefront, not here.

The value of an adapter is that there is exactly one implementation of each
rule. A second copy would drift, and the drift would show up as an AI doing
something the website forbids.

> **In plain terms:** This service is not allowed to make decisions. When an
> AI asks to cancel an order, this service does not decide whether that is
> allowed — it forwards the question to the shop and reports the answer. It
> is a messenger, not a manager.

---

## 3. The nine tools

| Tool | Risk | Backing endpoint |
|---|---|---|
| `search_products` | Low | `GET /api/v1/products` |
| `get_product` | Low | `GET /api/v1/products/{id}` |
| `check_inventory` | Low | `GET /api/v1/products/{id}/inventory` |
| `get_orders` | Low | `GET /api/v1/orders` |
| `get_order` | Low | `GET /api/v1/orders/{id}` |
| `get_cart` | Low | `GET /api/v1/cart` |
| `add_to_cart` | Medium | `POST /api/v1/cart` |
| `remove_from_cart` | Medium | `DELETE /api/v1/cart` |
| `cancel_order` | **High** | `POST /api/v1/orders/{id}/cancel` |

**What the tiers mean, and who enforces them:**

| Tier | The AI's behaviour | This server's behaviour |
|---|---|---|
| Low | Runs it, says nothing | Executes |
| Medium | Runs it, then mentions it ("Added to cart") | Executes |
| High | Must stop and ask a human first | **Refuses without a valid approval token** |

The two columns are the whole design. A tier the AI is merely *trusted* to
respect is a convention. A tier the server *refuses to execute* is a
boundary. Only the second survives an AI that has been misled.

Tools are business capabilities, not a mirror of the API. There is no
one-to-one wrapper for every endpoint, and `POST /api/v1/auth/token` is
deliberately **not** a tool — exposing it would hand the AI the credential
exchange itself.

> **In plain terms:** Nine things the AI can do, sorted by how much damage a
> mistake would cause. Searching products is harmless. Adding to a cart is
> easily undone. Cancelling an order is not — so that one is blocked at the
> door unless a human has specifically approved that exact cancellation.

---

## 4. Identity — who the AI is acting for

**The server never trusts a user id supplied by the AI.**

Every request carries a bearer token in its `Authorization` header. The
server forwards that token to the storefront and asks
`GET /api/v1/auth/whoami` who it belongs to. When the AI calls
`get_orders`, there is no customer argument to get wrong — the storefront
returns the orders of whoever the *token* names.

**Why it asks instead of reading the token:** NextAuth v4 does not sign
tokens, it *encrypts* them (a JWE, AES-256-GCM, with a key derived through
HKDF-SHA256). Reading the user out of one in Python would mean a second
implementation of that cryptography — which would break silently the day
NextAuth changed a detail. Asking keeps one implementation of identity.

Every request also builds **its own HTTP client**, never a shared one, and
sends `cookies=None`. A client held between requests is an ambient identity
by another name, and the bug that produces — one customer seeing another's
cart — is the exact failure this design exists to prevent.

> **In plain terms:** The AI cannot say "show me Alice's orders". It can
> only say "show me my orders", and who *my* refers to is decided by the
> sign-in pass attached to the request, not by anything the AI writes. Each
> request is handled with its own fresh connection so two customers can
> never get mixed up.

---

## 5. Approval tokens — the security boundary

The problem being solved: an AI reads a product review that contains hidden
instructions, is talked into cancelling a different order, and reuses an
approval the customer granted for something else.

**Checking that a token is merely *present* does not stop this** — the AI
has a valid token either way. So each approval is cryptographically bound
to the specific call:

```
HMAC-SHA256 over (session, tool name, hash of the arguments, nonce, expiry)
```

Each part earns its place:

| Bound to | Stops |
|---|---|
| session | An approval leaking between conversations |
| tool name | Approval for one capability being used for another |
| **argument hash** | **Approval to cancel order #3 being spent on order #7** |
| nonce + single use | A captured token being replayed |
| expiry (5 minutes) | A token left in a transcript staying live |

Approvals are minted only by `POST /approvals` — a plain HTTP route that is
**deliberately not an MCP tool**. An AI that can approve itself makes the
mechanism theatre. Two tests exist purely to keep it off the tool surface,
and one asserts no tool name even contains the string "approv".

The session identity is the MCP protocol's own `mcp-session-id`, assigned
by the server when a client connects — not something the caller can choose.

**Verified by deliberately breaking it:** replacing the binding with a
simple presence check fails four tests, including the one that tries to
spend an approval for order `o3` on order `o7`.

### What this does not defend

Anyone holding a valid bearer token can still call the storefront's cancel
endpoint directly. That is unchanged and by design — the storefront's own
defence is ownership and order-status rules. This mechanism protects the
*AI* path, and it lives where the AI's calls arrive.

> **In plain terms:** Before the AI can cancel an order, a human must
> approve it — and that approval is stamped with the exact order number. If
> the AI is tricked into cancelling a different order, the stamp does not
> match and the request is refused. Each approval works once, and expires
> after five minutes. Critically, the AI cannot issue approvals to itself:
> only the confirmation button a person clicks can do that.

---

## 6. Transport — why HTTP and not stdio

MCP servers commonly run over "stdio" — the AI program starts this program
as a child process and they talk through pipes.

**That would be wrong here, for a security reason rather than a deployment
one.** A stdio process carries one ambient identity: whichever token it was
started with. Every customer using the chat would share it. Over HTTP,
identity is a fact about each individual request.

This server runs **streamable HTTP** on `$PORT`, with per-request
authentication.

> **In plain terms:** The usual way to run this kind of service assumes one
> user. This is a shop with many customers, so it runs as a proper web
> service where every request says who it is for.

---

## 7. Repository layout

```
approvals.py              Mint, validate and burn approval tokens (173 lines)
server.py                 FastMCP app, the nine tools, /approvals (267 lines)
config.py                 Environment variables (18 lines)
clients/ecommerce_api.py  The only thing that speaks HTTP to /api/v1 (120)
models/schemas.py         Pydantic mirrors of the real API shapes (117)
tools/products.py         search_products, get_product, check_inventory
tools/orders.py           get_orders, get_order, cancel_order
tools/cart.py             get_cart, add_to_cart, remove_from_cart
scripts/sweep.py          Exercises every tool against a live deployment
tests/                    108 tests, one module per source module
```

**The `tools/` split is deliberate and gets reused.** In Phase 3 each file
becomes one specialist agent's toolbox — `products.py` for a Product agent,
and so on. It is not an arbitrary tidy-up and should not be reorganised for
convenience.

`clients/ecommerce_api.py` is the single place that knows the API's
envelope. If the storefront changes its response shape, exactly one file
breaks instead of nine.

> **In plain terms:** How the code is organised. The grouping into
> products / orders / cart is planned ahead: later, each group becomes one
> specialised AI assistant's area of responsibility.

---

## 8. Data handling

Responses are validated into Pydantic models before any tool returns them.
Two rules are strict on purpose:

- **Money is a `str`**, and a number is **rejected**, not quietly
  converted. The storefront guarantees strings; if a float ever reappears
  that is a regression worth failing loudly for. Silent conversion is
  precisely how a two-type price bug survived unnoticed until M3.
- **Timestamps stay strings.** Parsing them into datetimes would invite a
  timezone conversion on the way back out of a value that was already
  correct.

Unknown fields are **ignored, not rejected** — the storefront may grow a
field before this service knows about it, and failing hard there would take
all nine tools down for a harmless addition.

> **In plain terms:** Prices are handled as text, never as decimal numbers,
> so £10.50 cannot silently become £10.5. And if the shop starts sending
> some new piece of information this service has not been told about, it
> ignores it rather than falling over.

---

## 9. Retry safety

If a request times out, the caller cannot tell whether the work happened.
Retrying might do it twice; not retrying might do it never.

- `add_to_cart` sends an **idempotency key** derived from its arguments plus
  the conversation id. A retry of the same call replays the first result; a
  genuinely different call cannot. The conversation scoping matters: without
  it, "add one more" tomorrow would hash identically to "add one more"
  today, and the second would silently replay the first — the customer asks
  for another item and gets nothing, with a cheerful success message.
- `cancel_order` sends a key derived from the order and session — **not**
  from the approval token, which is single-use and therefore different on
  every retry.
- `remove_from_cart` sends **no** key. It is already idempotent: removing a
  line that is gone leaves the cart exactly as asked.

The storefront honours these keys with a database table added in M3.

> **In plain terms:** If the connection drops halfway through adding
> something to a cart, retrying is safe — the shop recognises it as the same
> request and does not add the item twice.

---

## 10. Deployment

| | |
|---|---|
| Platform | Railway, project `mcp_ecom`, service `mcp` |
| Source | GitHub `main`, repository root |
| Builder | RAILPACK (`.python-version` pins 3.11) |
| Start | `python server.py` |
| Endpoint | `https://mcp-production-e344.up.railway.app/mcp` |

| Variable | Required | Purpose |
|---|---|---|
| `MCP_APPROVAL_SECRET` | **yes** | Signs approval tokens. Unrelated to the storefront's secret — this authorises one action, it does not authenticate a person |
| `ECOMMERCE_API_BASE_URL` | no | Storefront API root |
| `MCP_APPROVAL_TTL_SECONDS` | no | Approval lifetime, default 300 |
| `PORT` | no | Injected by Railway |

The server **refuses to start** without `MCP_APPROVAL_SECRET`, rather than
booting and minting tokens that any other misconfigured instance would
accept.

**Do not add a `Dockerfile` at the repository root.** Railway silently
switches its builder when it finds one, which caused a failed production
build in the storefront repo.

**Private networking does not currently work.** Railway offers an internal
address, but the storefront's middleware upgrades HTTP to HTTPS whenever
`x-forwarded-proto` is absent — which is exactly the case internally — so
every call redirects to an address with no TLS. Traffic therefore goes over
the public domain. The fix is a condition in the storefront's middleware.

> **In plain terms:** Pushing to the main branch publishes it automatically.
> One setting is essential — the secret key used to stamp approvals — and
> the service deliberately refuses to start without it, rather than running
> in a state where approvals could be forged.

---

## 11. Testing

108 tests, no network access, one module per source module:

| Module | Tests | Covers |
|---|---|---|
| `test_server.py` | 20 | Per-request identity, the exposed tool surface, minting |
| `test_approvals.py` | 16 | Every way an approval must fail |
| `test_tools_orders.py` | 16 | Order reads, and cancellation refusals |
| `test_tools_cart.py` | 15 | Cart reads, mutations, idempotency keys |
| `test_schemas.py` | 15 | The real response shapes |
| `test_ecommerce_api.py` | 12 | Envelope, errors, headers |
| `test_tools_products.py` | 9 | Search filters, inventory |

HTTP is mocked with `respx`. **Every important behaviour was verified by
mutation** — deliberately breaking the code to confirm the tests notice.
The security tests were checked this way specifically: replacing the
argument binding with a presence check fails four of them.

### The sweep

`scripts/sweep.py` drives a real deployment through a bare MCP client, with
no AI involved. It varies its search terms deliberately, because the
storefront caches searches for 300 seconds and repeating one term would
measure the cache rather than the system.

```bash
python scripts/sweep.py --url https://mcp-production-e344.up.railway.app/mcp
python scripts/sweep.py --url ... --api ... --email ... --password ...   # full
```

It exits non-zero if a cancellation succeeds without a valid approval.

**A green test suite has never been the gate on this project.** Four bugs
have been found by live checks that the tests could not see, including one
in this service: FastMCP strips the session header by default, which broke
every cancellation while all 19 unit tests passed.

> **In plain terms:** The code is tested by pretending to be the shop and
> checking the answers. On top of that, the tests themselves are tested —
> the code is deliberately broken to confirm the tests actually notice. And
> because tests can only check what someone thought to ask, everything is
> also run against the real, live system.

---

## 12. Current status and known limits

**Shipped:** all nine tools live and callable; the high-risk refusal
enforced in production.

**Verified against production:** `search_products`, `get_product`,
`check_inventory` — and `cancel_order` refusing both a missing and a forged
approval.

**Verified against mocks only:** `get_orders`, `get_order`, `get_cart`,
`add_to_cart`, `remove_from_cart`, and `cancel_order`'s *success* path.
These need a signed-in customer, and no demo credentials were available for
the sweep. One run with credentials closes the gap.

Known limits, each real:

- **The spent-nonce set is in process.** Single-use holds for one replica.
  Scale out and a token becomes replayable within its five-minute life. A
  shared store is required first.
- **Prompt injection is unmitigated.** Product descriptions and review text
  are written by strangers and flow into AI context through the low-risk
  tools. Approval tokens close the *execution* path; they do nothing about
  information leaking out through an innocent-looking search.
- **Approvals defend the AI path, not the API** (see §5).
- **Sign-in tokens cannot be revoked.** Short lifetimes bound the exposure;
  nothing removes it.

> **In plain terms:** Most of it is proven against the real, live shop. Six
> capabilities are proven only against a simulation, because testing them
> needs a real customer login that was not available — that is a known,
> written-down gap, not an oversight. The most serious open issue: product
> reviews are written by the public and the AI reads them, so a carefully
> worded review could try to give the AI instructions. That must be solved
> before this is used for real.

---

## 13. What comes next

| Phase | Delivers | Status |
|---|---|---|
| 1 | This MCP server. No agent. | **Shipped** |
| 2 | One agent, the full toolbox, a chat UI with approval prompts | Not started |
| 3 | A supervisor agent coordinating product / order / cart specialists | Not started |

Phase 2 already owes two things, both decided during Phase 1:

- **A token refresh mechanism.** Sign-in tokens are deliberately short-lived
  because they cannot be revoked, but a conversation outlives fifteen
  minutes and this server never sees a password.
- **Approval prompts rendered from structured arguments, never from AI
  prose.** The confirmation gate is only as trustworthy as the text beside
  the button; if the AI writes that text, an injected review can write it
  too.

The design documents for all three phases live in the storefront
repository under
[`docs/mcp/`](https://github.com/arone011-creator/mcp-ecom/tree/main/docs/mcp).

> **In plain terms:** The buttons exist and work. Next comes the AI that
> presses them, then a version where several specialised AIs divide the work
> between them. Two safety items are already booked in for the next stage.

---

## 14. Running it locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Scripts/ on Windows, bin/ elsewhere
.venv/Scripts/python -m pytest
```

```bash
MCP_APPROVAL_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(32))") \
  .venv/Scripts/python server.py
```

Serves on `http://localhost:8000/mcp`. Point any MCP client at it with an
`Authorization: Bearer <token>` header, where the token comes from the
storefront's `POST /api/v1/auth/token`.

> **In plain terms:** The steps to run a copy on your own computer. You need
> Python and a sign-in token from the shop.
