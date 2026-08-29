"""Environment the MCP server reads, resolved once at import."""

import os

# The storefront's public domain by default. Switch to the private-network
# address once it is verified -- see Task 10 of the M3 plan.
ECOMMERCE_API_BASE_URL = os.environ.get(
    "ECOMMERCE_API_BASE_URL", "https://web-production-bb55d.up.railway.app"
)

# Signs approval tokens. Deliberately unrelated to NEXTAUTH_SECRET: this key
# authorises one tool call, it does not authenticate a person, and the two
# should be rotatable independently.
APPROVAL_SECRET = os.environ.get("MCP_APPROVAL_SECRET", "")

APPROVAL_TTL_SECONDS = int(os.environ.get("MCP_APPROVAL_TTL_SECONDS", "300"))
HTTP_TIMEOUT_SECONDS = float(os.environ.get("MCP_HTTP_TIMEOUT_SECONDS", "10"))
PORT = int(os.environ.get("PORT", "8000"))
