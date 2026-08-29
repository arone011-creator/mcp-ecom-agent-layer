# mcp-ecom-agent-layer

The AI-access layer for the [mcp-ecom](https://github.com/arone011-creator/mcp-ecom)
storefront: an MCP server exposing nine e-commerce capabilities to agents,
and — from Phase 2 onward — the agents themselves.

**Status: Phase 1 (MCP server) complete. No agent exists yet.**

**[→ Technical Snapshot](docs/TECHNICAL_SNAPSHOT.txt)** — the whole service
in one document: architecture, the nine tools, the security model,
deployment and known limits. Every section has a plain-language summary,
so it reads for technical and non-technical audiences alike.

## What this is

```
MCP Client  ->  this server  ->  mcp-ecom /api/v1  ->  Postgres
```

An adapter, not a second implementation. Business logic — may this order be
cancelled, is there stock, who owns this cart — stays in the storefront's
API. This server never touches the database and never re-implements a rule,
because the value of an adapter is that there is exactly one implementation
of each rule.

## The tools

| Tool | Risk | What it does |
|---|---|---|
| `search_products` | Low | Search the catalogue by keyword, category, price, rating |
| `get_product` | Low | Full detail for one product |
| `check_inventory` | Low | Stock available right now, never cached |
| `get_orders` | Low | The caller's own orders |
| `get_order` | Low | One of the caller's orders |
| `get_cart` | Low | The caller's cart, with totals computed server-side |
| `add_to_cart` | Medium | Add a product; increments by default, `mode='set'` replaces |
| `remove_from_cart` | Medium | Remove one product, or empty the cart |
| `cancel_order` | **High** | Cancel an order — requires an approval token |

Low executes silently. Medium executes and is surfaced as an informational
event. High is **refused at the server** without a valid approval token,
regardless of what the caller intended.

## Two things worth understanding before changing anything

**Identity is never an argument.** The server derives the caller from the
bearer token by asking `GET /api/v1/auth/whoami`. It does not decode the
token itself: NextAuth v4 mints an encrypted JWE, so reading a subject out
of one in Python would mean a second implementation of NextAuth's key
derivation, which would break silently the day that changes.

**Approval tokens are bound, not merely present.** Each is HMAC-signed over
`(session, tool, argument hash, nonce, expiry)` and single-use. The
argument hash is the part that matters: without it, approval to cancel
order #3 is approval to cancel order #7, and a prompt-injected agent can
talk itself past a confirmation step. Approvals are minted only by the
`POST /approvals` HTTP route, which is deliberately **not** an MCP tool —
an agent that can approve itself makes the mechanism theatre.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt      # Scripts/ on Windows, bin/ elsewhere
.venv/Scripts/python -m pytest
```

```bash
MCP_APPROVAL_SECRET=<32 random bytes> .venv/Scripts/python server.py
```

Serves streamable HTTP on `$PORT` (default 8000). Deliberately not stdio: a
stdio process carries one ambient identity, which is wrong for a
multi-user app — every caller would share whichever token the process
started with.

| Variable | Required | Purpose |
|---|---|---|
| `MCP_APPROVAL_SECRET` | yes | Signs approval tokens. Unrelated to `NEXTAUTH_SECRET` — this authorises one call, it does not authenticate a person |
| `ECOMMERCE_API_BASE_URL` | no | Storefront API root; defaults to the deployed instance |
| `MCP_APPROVAL_TTL_SECONDS` | no | Approval lifetime, default 300 |
| `PORT` | no | Listen port, default 8000 |

## Known limitations

These are honest gaps, not oversights. Each needs closing before this is
more than a demo.

- **Prompt injection via tool output is unmitigated.** Product descriptions
  and review text are attacker-controllable and flow into agent context
  through low-risk tools. Approval tokens close the *execution* path only.
- **The spent-nonce set is in process.** Single-use holds for one replica;
  scale out and a token becomes replayable within its TTL.
- **Approvals defend the agent path, not the API.** Anyone holding a bearer
  token can still call the cancel endpoint directly. That is unchanged and
  by design — the API's own defence is ownership plus status rules.
- **Session JWTs cannot be revoked.** Short lifetimes bound the exposure;
  nothing shortens it to zero.

## Provenance

Split out of the `mcp-ecom` monorepo, where it was built as `apps/mcp`
during milestone M3. The design documents, phase plans and the reasoning
behind each decision live there under `docs/mcp/`.
