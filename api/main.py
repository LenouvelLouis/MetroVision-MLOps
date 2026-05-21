"""MetroVision API — FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.logging_config import setup_json_logging
from api.middleware import (
    AuditLogMiddleware,
    MaxBodySizeMiddleware,
    SecurityHeadersMiddleware,
)
from api.model_manager import model_manager
from api.rate_limit import limiter
from api.routes import drift, health, metrics, predict, version
from api.routes.metrics import ERROR_REASON, REQUEST_COUNT
from api.tracing import setup_tracing

setup_json_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load models at startup, cleanup at shutdown."""
    # Re-apply: uvicorn sets up its own loggers after our import-time call,
    # so we override them again once the lifespan starts.
    setup_json_logging()
    logger.info("api_starting")
    model_manager.load()
    logger.info("api_ready")
    yield
    logger.info("api_shutting_down")


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom 429 handler that bumps the error counters before responding."""
    ERROR_REASON.labels(reason="rate_limited").inc()
    REQUEST_COUNT.labels(status="error").inc()

    response = JSONResponse(
        {"detail": f"Rate limit exceeded: {exc.detail}"},
        status_code=429,
    )
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if view_rate_limit is not None:
        response = request.app.state.limiter._inject_headers(response, view_rate_limit)
    return response


app = FastAPI(
    title="MetroVision API",
    description="Production REST API for Paris Metro pictogram detection",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiter — exposed on app.state so the @limiter.limit decorator can find it.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Middlewares are stacked outermost-last. Order applied:
#   request:  MaxBodySize -> SecurityHeaders -> AuditLog -> app
#   response: app -> AuditLog -> SecurityHeaders -> MaxBodySize
app.add_middleware(AuditLogMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MaxBodySizeMiddleware)

# OpenTelemetry: wraps the FastAPI app last so spans cover the whole stack.
setup_tracing(app)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(version.router)
app.include_router(drift.router)
