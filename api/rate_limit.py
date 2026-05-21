"""Per-IP rate limiter shared across the API.

Defined in its own module so both `api.main` (handler/middleware wiring)
and `api.routes.predict` (per-endpoint decorator) can import the same
Limiter instance without circular imports.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

PREDICT_RATE_LIMIT = os.getenv("METROVISION_PREDICT_RATE_LIMIT", "10/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[])
