from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.logs import LogListApiResponse, QALogListResponse
from app.services.document_qa_service import DocumentQAService


router = APIRouter()


def _document_service() -> DocumentQAService:
    from app.api import routes

    return routes.document_service


@router.get("/qa/logs", response_model=QALogListResponse, tags=["logs"])
def list_qa_logs(limit: int = Query(default=50, ge=1, le=200)) -> QALogListResponse:
    return _document_service().list_logs(limit=limit)


@router.get("/api/logs", response_model=LogListApiResponse, tags=["api-logs"])
def api_list_logs(limit: int = Query(default=50, ge=1, le=200)) -> LogListApiResponse:
    return _document_service().list_logs_api(limit=limit)
