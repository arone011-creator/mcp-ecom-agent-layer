"""The only thing in this service that speaks HTTP to /api/v1.

Every v1 response is `{"data": ...}` or `{"error": "..."}`. Unwrapping in
one place means the tools deal in values, and a change to the envelope
breaks one file rather than nine.

This client never re-implements a business rule. If a question can be
answered by the API -- may this order be cancelled, is there stock -- it is
asked, not decided here.
"""

from typing import Any

import httpx

import config


class ApiError(Exception):
    """A non-2xx from /api/v1, carrying the status the agent should act on.

    The status is the useful part: 409 means "re-read and try something
    smaller", 404 means "give up". Collapsing both into "it failed" would
    throw away the only signal that distinguishes them.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class EcommerceApi:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.ECOMMERCE_API_BASE_URL).rstrip("/")
        self.token = token
        self.timeout = timeout or config.HTTP_TIMEOUT_SECONDS

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        headers = {"accept": "application/json"}
        # Omitted rather than sent empty when there is no token: the product
        # routes are public, and an empty bearer is worse than no bearer.
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        if idempotency_key:
            headers["idempotency-key"] = idempotency_key
        return headers

    @staticmethod
    def _params(params: dict[str, Any] | None) -> dict[str, Any]:
        """Drop absent values. A defaulted filter is still a filter."""
        return {k: v for k, v in (params or {}).items() if v is not None}

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        try:
            body = response.json()
        except ValueError:
            # The body may be an HTML error page from a proxy. Reporting it
            # verbatim would put arbitrary upstream text into agent context.
            raise ApiError(
                response.status_code, "Upstream returned an unreadable response"
            )

        if response.status_code >= 400:
            message = body.get("error") if isinstance(body, dict) else None
            raise ApiError(response.status_code, message or "Request failed")

        return body.get("data") if isinstance(body, dict) else body

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        # cookies=None keeps this from ever presenting a second identity;
        # requireApiUser accepts a session cookie, and the bearer token is
        # the only credential this service should be offering.
        async with httpx.AsyncClient(timeout=self.timeout, cookies=None) as http:
            response = await http.get(
                f"{self.base_url}{path}",
                params=self._params(params),
                headers=self._headers(),
            )
        return self._unwrap(response)

    async def post(
        self,
        path: str,
        body: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout, cookies=None) as http:
            response = await http.post(
                f"{self.base_url}{path}",
                json=body,
                headers=self._headers(idempotency_key),
            )
        return self._unwrap(response)

    async def delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout, cookies=None) as http:
            response = await http.request(
                "DELETE",
                f"{self.base_url}{path}",
                params=self._params(params),
                headers=self._headers(),
            )
        return self._unwrap(response)

    async def whoami(self) -> dict[str, Any]:
        """The verified caller, as /api/v1 understands them.

        The MCP server has no other way to know. NextAuth v4 mints an
        encrypted JWE, so the token is opaque to a Python process, and
        re-implementing that decryption here would be a second copy of a
        rule that already has one.
        """
        return await self.get("/api/v1/auth/whoami")
