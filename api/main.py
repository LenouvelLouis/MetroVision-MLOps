"""MetroVision API — FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.model_manager import model_manager
from api.routes import health, metrics, predict, version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load models at startup, cleanup at shutdown."""
    logger.info("Starting MetroVision API — loading models...")
    model_manager.load()
    logger.info("Models loaded. API ready.")
    yield
    logger.info("Shutting down MetroVision API.")


app = FastAPI(
    title="MetroVision API",
    description="Production REST API for Paris Metro pictogram detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(version.router)
