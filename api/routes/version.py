"""Version endpoint."""

from __future__ import annotations

import sys

from fastapi import APIRouter

from api.model_manager import model_manager
from api.schemas import VersionResponse

router = APIRouter()

APP_VERSION = "0.1.0"


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Application version and model metadata",
)
def version() -> VersionResponse:
    """Return application version, Python version, and model file info."""
    return VersionResponse(
        version=APP_VERSION,
        python_version=sys.version,
        models=model_manager.model_info(),
    )
