"""HTTP middlewares for security hardening."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

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
