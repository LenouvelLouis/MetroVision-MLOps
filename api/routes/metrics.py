"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)

router = APIRouter()

# ── Metric definitions (importable by other modules) ─────────────────────
REQUEST_COUNT = Counter(
    "metrovision_requests_total",
    "Total prediction requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "metrovision_request_duration_seconds",
    "Prediction request latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_LINES = Counter(
    "metrovision_predicted_lines_total",
    "Count of predicted metro lines",
    ["line"],
)


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
)
def metrics() -> str:
    """Return Prometheus-format metrics."""
    return generate_latest().decode("utf-8")
