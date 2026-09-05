"""Environment the MCP server reads, resolved once at import."""

import os

# The storefront's public domain by default. The deployed service overrides
# this with the Railway private-network address (verified 2026-09-02); the
# public default is what a local run and the test suite get.
ECOMMERCE_API_BASE_URL = os.environ.get(
    "ECOMMERCE_API_BASE_URL", "https://web-production-bb55d.up.railway.app"
)

# Where the agent finds the MCP server. Read by agent/, not by the server
# itself -- the server does not call itself.
MCP_SERVER_URL = os.environ.get(
    "MCP_SERVER_URL", "https://mcp-production-e344.up.railway.app/mcp"
)

# The model behind the agent. OPENAI_API_KEY is read by the SDK itself and
# deliberately not mirrored here -- one fewer place a key can be logged.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")

# One model request. The SDK's own default is 600 seconds with retries on
# top, which let a stalled request hold an eval sweep for 36 minutes on 34
# seconds of CPU. A turn that cannot answer inside a minute has already
# failed the customer waiting on it.
OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))

# Signs approval tokens. Deliberately unrelated to NEXTAUTH_SECRET: this key
# authorises one tool call, it does not authenticate a person, and the two
# should be rotatable independently.
APPROVAL_SECRET = os.environ.get("MCP_APPROVAL_SECRET", "")

APPROVAL_TTL_SECONDS = int(os.environ.get("MCP_APPROVAL_TTL_SECONDS", "300"))

# Guards the agent service's /turn route. Deliberately NOT the approval
# secret: this one says "you may spend model tokens", that one says "this
# action was confirmed by a human", and an endpoint that calls a paid
# model with no key at all is a bill anyone who finds the URL can run up.
AGENT_SERVICE_KEY = os.environ.get("AGENT_SERVICE_KEY", "")

# How long a turn holds an approval open. Longer than a person needs to
# read a card and click, short enough that an abandoned tab does not hold
# an MCP session all afternoon.
APPROVAL_WAIT_SECONDS = float(os.environ.get("AGENT_APPROVAL_WAIT_SECONDS", "300"))

# Set by Railway on every deploy. Reported by /health so a readiness check
# can assert it reached the container it just built, rather than the one
# being replaced -- a mistake made twice on this project.
RAILWAY_GIT_COMMIT_SHA = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")
HTTP_TIMEOUT_SECONDS = float(os.environ.get("MCP_HTTP_TIMEOUT_SECONDS", "10"))
PORT = int(os.environ.get("PORT", "8000"))

# Which agent architecture a turn uses.
#
# DEFAULTS TO single, deliberately. The single-agent path is verified
# live and working; the team path replaces it only once it has been
# verified the same way. A routing mistake should not take a working demo
# down, and the two share build_graph, so keeping both is cheap.
AGENT_MODE = os.environ.get("AGENT_MODE", "single")
