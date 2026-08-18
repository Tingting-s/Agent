from __future__ import annotations

from fastapi import APIRouter

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.legacy_tasks import router as legacy_tasks_router
from app.api.logs import router as logs_router
from app.api.qa import router as qa_router
from app.services.document_qa_service import DocumentQAService
from app.services.legacy_task_service import LegacyTaskService


router = APIRouter()
document_service = DocumentQAService()
task_service = LegacyTaskService(document_service=document_service)

router.include_router(health_router)
router.include_router(legacy_tasks_router)
router.include_router(documents_router)
router.include_router(qa_router)
router.include_router(logs_router)
