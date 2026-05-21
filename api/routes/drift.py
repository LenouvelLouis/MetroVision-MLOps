"""Push endpoint for the Evidently drift job.

Phase 5.4: the drift batch job runs offline and POSTs its score here so
it can be scraped by Prometheus and alerted on. The endpoint is gated
by the same X-API-Key dependency as /predict — only the trusted batch
runner should be able to update the gauge.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.routes.metrics import DRIFT_SCORE

logger = logging.getLogger(__name__)

router = APIRouter()


class DriftReport(BaseModel):
    feature_set: str = Field(
        description="Logical name of the feature set ('hog', 'pixel_stats', ...)",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
    )
    score: float = Field(ge=0.0, le=1.0, description="Drift score in [0, 1]")


@router.post(
    "/internal/drift-score",
    summary="Publish a drift score from the Evidently batch job",
    dependencies=[Depends(require_api_key)],
    status_code=204,
)
def publish_drift_score(report: DriftReport) -> None:
    DRIFT_SCORE.labels(feature_set=report.feature_set).set(report.score)
    logger.info(
        "drift_score_updated",
        extra={"feature_set": report.feature_set, "score": report.score},
    )
