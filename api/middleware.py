"""HTTP middlewares for security hardening."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_audit_logger = logging.getLogger("api.audit")
_AUDIT_SKIP_PATHS = {"/metrics", "/health"}

_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Cache-Control": "no-store",
}


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds the limit.

    Requests without Content-Length (e.g. chunked transfer encoding) are
    rejected as well to keep the bound enforceable without streaming.
    """

    def __init__(self, app, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length is None:
                return JSONResponse(
                    {"detail": "Content-Length header is required"},
                    status_code=411,
                )
            try:
                declared = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length"},
                    status_code=400,
                )
            if declared > self.max_bytes:
                return JSONResponse(
                    {"detail": f"Request body too large (max {self.max_bytes} bytes)"},
                    status_code=413,
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach a baseline set of security headers to every response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Structured access log for every request.

    Emits one JSON record per request via the dedicated `api.audit` logger:
        - method, path, status
        - duration_ms
        - client IP
        - api_key_id (12-char SHA-256 prefix of the X-API-Key header value,
          or "anonymous" when the header is absent)
        - user_agent (truncated)

    Health and metrics endpoints are skipped to keep the audit stream
    focused on actual user / client traffic.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _AUDIT_SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)

        raw_key = request.headers.get("x-api-key")
        api_key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:12] if raw_key else "anonymous"

        client_host = request.client.host if request.client else "-"
        ua = request.headers.get("user-agent", "")[:128]

        _audit_logger.info(
            "audit",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": client_host,
                "api_key_id": api_key_id,
                "user_agent": ua,
            },
        )
        return response
