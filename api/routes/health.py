"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api.model_manager import model_manager
from api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
)
def health() -> HealthResponse:
    """Return service health status and whether models are loaded."""
    loaded = model_manager.is_loaded
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        models_loaded=loaded,
    )
