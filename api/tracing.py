"""OpenTelemetry tracing setup (phase 7.5).

Auto-instruments the FastAPI app to emit OTLP traces. The OTLP endpoint
defaults to `http://tempo:4318` (in-cluster name) and can be overridden
via `OTEL_EXPORTER_OTLP_ENDPOINT`. When unset (e.g. in tests) the
exporter is silently disabled — `setup_tracing` becomes a no-op.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(app) -> None:
    """Wire up OTel auto-instrumentation on the given FastAPI app."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("tracing_disabled", extra={"reason": "no_endpoint"})
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("tracing_unavailable", extra={"err": str(exc)})
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "metrovision-api")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info("tracing_enabled", extra={"endpoint": endpoint, "service": service_name})
