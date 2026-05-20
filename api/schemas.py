"""Pydantic response models for the MetroVision API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected metro pictogram."""

    y1: int = Field(description="Top boundary (pixels)")
    y2: int = Field(description="Bottom boundary (pixels)")
    x1: int = Field(description="Left boundary (pixels)")
    x2: int = Field(description="Right boundary (pixels)")
    line: int = Field(description="Predicted metro line number")


class PredictResponse(BaseModel):
    """Response from the /predict endpoint."""

    detections: list[Detection]
    count: int = Field(description="Number of detections")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(description="healthy or unhealthy")
    models_loaded: bool


class VersionResponse(BaseModel):
    """Response from the /version endpoint."""

    version: str
    python_version: str
    models: dict[str, str]
