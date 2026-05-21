"""Prediction endpoint — accepts an image, returns detected metro lines."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from api.model_manager import model_manager
from api.rate_limit import PREDICT_RATE_LIMIT, limiter
from api.routes.metrics import (
    ERROR_REASON,
    PREDICTION_LINES,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from api.schemas import Detection, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter()

PROCESSING_TIMEOUT_S = 30.0

# Magic bytes for the two formats we accept on /predict.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _detect_image_type(data: bytes) -> str | None:
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    return None


def _fail(reason: str, status: int, detail: str) -> HTTPException:
    ERROR_REASON.labels(reason=reason).inc()
    REQUEST_COUNT.labels(status="error").inc()
    return HTTPException(status_code=status, detail=detail)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Detect metro pictograms in an uploaded image",
)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(
    request: Request,
    file: UploadFile,
    resize_factor: float = Query(default=1.0, ge=0.5, le=1.5, description="Image resize factor"),
) -> PredictResponse:
    """Run the full detection pipeline on the uploaded image.

    1. Validate the upload (magic bytes, models loaded)
    2. Decode the image
    3. Run the academic processOneMetroImage in a thread pool with a timeout
    4. Return structured JSON with bounding boxes and line predictions
    """
    start = time.perf_counter()

    if not model_manager.is_loaded:
        raise _fail("models_not_loaded", 503, "Models not loaded yet")

    try:
        contents = await file.read()
    except Exception as exc:
        logger.warning("upload_read_failed", extra={"err": str(exc)})
        raise _fail("read_failed", 400, "Could not read uploaded file") from exc

    detected_type = _detect_image_type(contents)
    if detected_type is None:
        raise _fail("invalid_type", 415, "Unsupported file type; expected JPEG or PNG")

    file_id = hashlib.sha256(contents).hexdigest()[:12]
    logger.info(
        "predict_received",
        extra={
            "file_id": file_id,
            "file_size_bytes": len(contents),
            "content_type": detected_type,
            "resize_factor": resize_factor,
        },
    )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        im_array = np.array(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("decode_failed", extra={"file_id": file_id, "err": str(exc)})
        raise _fail("decode_failed", 400, "Could not decode image") from exc

    from myMetroProcessing import processOneMetroImage

    loop = asyncio.get_running_loop()
    try:
        _, bd = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                processOneMetroImage,
                "api_request",
                im_array,
                0,
                resize_factor,
            ),
            timeout=PROCESSING_TIMEOUT_S,
        )
    except TimeoutError as exc:
        logger.warning("processing_timeout", extra={"file_id": file_id})
        raise _fail("processing_timeout", 504, "Processing timed out") from exc
    except Exception as exc:
        logger.exception("processing_error", extra={"file_id": file_id})
        raise _fail("processing_error", 500, "Processing error") from exc

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

    logger.info(
        "predict_completed",
        extra={
            "file_id": file_id,
            "duration_s": round(elapsed, 3),
            "detection_count": len(detections),
        },
    )

    return PredictResponse(detections=detections, count=len(detections))
