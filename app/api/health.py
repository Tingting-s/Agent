from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.response import HealthResponse
from app.utils.logger import get_logger


router = APIRouter(tags=["system"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    logger.info("Health check requested.")
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
    )
