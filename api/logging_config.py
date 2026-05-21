"""Structured JSON logging for the MetroVision API.

Produces one JSON object per log record on stdout, suitable for ingestion
by Loki / ELK / Datadog without further parsing.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


_UVICORN_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")


def setup_json_logging(level: int = logging.INFO) -> None:
    """Replace the root logger handlers with a single JSON stdout handler.

    Also normalizes uvicorn's own loggers so its access / error lines flow
    through the same JSON formatter instead of uvicorn's default text format.
    Safe to call multiple times (idempotent).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Strip uvicorn's own handlers and let its records propagate to root.
    for name in _UVICORN_LOGGERS:
        uv_logger = logging.getLogger(name)
        for existing in list(uv_logger.handlers):
            uv_logger.removeHandler(existing)
        uv_logger.propagate = True
        uv_logger.setLevel(level)
