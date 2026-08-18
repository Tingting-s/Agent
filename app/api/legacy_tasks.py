from __future__ import annotations

"""Legacy compatibility API for the historical /tasks/execute endpoint."""

from fastapi import APIRouter

from app.schemas.tasks import TaskRequest, TaskResponse
from app.services.legacy_task_service import LegacyTaskService
from app.utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


def _task_service() -> LegacyTaskService:
    from app.api import routes

    return routes.task_service


@router.post("/tasks/execute", response_model=TaskResponse, tags=["tasks"])
def execute_task(request: TaskRequest) -> TaskResponse:
    logger.info(
        "Task execution requested. task_type=%s input_preview=%s",
        request.task_type,
        request.user_input[:80],
    )
    return _task_service().handle_request(request)
