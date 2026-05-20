"""Prediction endpoint — accepts an image, returns detected metro lines."""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from PIL import Image

from api.model_manager import model_manager
from api.routes.metrics import PREDICTION_LINES, REQUEST_COUNT, REQUEST_LATENCY
from api.schemas import Detection, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Detect metro pictograms in an uploaded image",
)
async def predict(
    file: UploadFile = File(description="Image file (JPEG/PNG)"),
    resize_factor: float = Query(
        default=1.0, ge=0.5, le=1.5, description="Image resize factor"
    ),
) -> PredictResponse:
    """Run the full detection pipeline on the uploaded image.

    1. Read and decode image
    2. Call the academic processOneMetroImage function
    3. Return structured JSON with bounding boxes and line predictions
    """
    start = time.perf_counter()

    if not model_manager.is_loaded:
        REQUEST_COUNT.labels(status="error").inc()
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Models not loaded yet")

    contents = await file.read()
    image = Image.open(
        __import__("io").BytesIO(contents)
    ).convert("RGB")
    im_array = np.array(image)

    from myMetroProcessing import processOneMetroImage

    _, bd = processOneMetroImage("api_request", im_array, 0, resize_factor)

    detections: list[Detection] = []
    if bd is not None and bd.shape[0] > 0:
        for row in bd:
            _, y1, y2, x1, x2, line = row
            detections.append(
                Detection(y1=int(y1), y2=int(y2), x1=int(x1), x2=int(x2), line=int(line))
            )
            PREDICTION_LINES.labels(line=str(int(line))).inc()

    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.observe(elapsed)
    REQUEST_COUNT.labels(status="success").inc()

    logger.info("Prediction completed in %.3fs — %d detections", elapsed, len(detections))

    return PredictResponse(detections=detections, count=len(detections))
