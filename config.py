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
HTTP_TIMEOUT_SECONDS = float(os.environ.get("MCP_HTTP_TIMEOUT_SECONDS", "10"))
PORT = int(os.environ.get("PORT", "8000"))
