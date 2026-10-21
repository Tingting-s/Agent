from __future__ import annotations

import logging
from contextlib import ExitStack
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.documents import (
    BatchDeleteDocumentsRequest,
    BatchDeleteDocumentsResponse,
    DocumentDeleteApiResponse,
    DocumentIngestionTaskApiResponse,
    DocumentListApiResponse,
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentReindexApiResponse,
    DocumentUploadResponse,
    SummaryApiRequest,
    SummaryApiResponse,
    UploadApiResponse,
)
from app.schemas.qa import DocumentQARequest, DocumentQAResponse
from app.services.document_qa_service import DocumentQAService
from app.services.ingestion_task_service import (
    DocumentOperationConflictError,
    IngestionTaskService,
)
from app.tasks.document_ingestion import enqueue_document_ingestion


router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _document_service() -> DocumentQAService:
    from app.api import routes

    return routes.document_service


def _ingestion_task_service() -> IngestionTaskService:
    return IngestionTaskService()


@router.post("/documents/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    try:
        return await _document_service().upload_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/parse", response_model=DocumentParseResponse, tags=["documents"])
def parse_document(request: DocumentParseRequest) -> DocumentParseResponse:
    try:
        return _document_service().parse_document(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/ask", response_model=DocumentQAResponse, tags=["documents"])
def ask_document(request: DocumentQARequest) -> DocumentQAResponse:
    try:
        return _document_service().ask_document(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/documents/upload", response_model=UploadApiResponse, tags=["api-documents"])
async def api_upload_document(file: UploadFile = File(...)) -> UploadApiResponse:
    return await _document_service().upload_document_api(file)


@router.post(
    "/api/documents/upload/async",
    response_model=DocumentIngestionTaskApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["api-documents"],
)
async def api_upload_document_async(
    file: UploadFile = File(...),
) -> DocumentIngestionTaskApiResponse | JSONResponse:
    request_id = str(uuid4())
    start = perf_counter()
    document_service = _document_service()
    settings = get_settings()
    try:
        document = await document_service.save_document_for_ingestion(
            file,
            max_size_bytes=max(settings.MAX_ASYNC_UPLOAD_SIZE_MB, 1) * 1024 * 1024,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = str(document["document_id"])
    task_service = _ingestion_task_service()
    task = task_service.create_task(document_id, "UPLOAD")
    try:
        enqueue_document_ingestion(str(task["task_id"]))
    except Exception as exc:
        failed_task = _record_dispatch_failure(task_service, task, exc)
        document_service.storage_service.update_index_status(
            document_id,
            status="error",
            vector_indexed=False,
            error_message=str(failed_task["error_message"]),
        )
        response = _task_response(
            failed_task,
            request_id=request_id,
            start=start,
            message="Document was saved, but task submission failed. Retry with async reindex.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return _task_response(
        task,
        request_id=request_id,
        start=start,
        message="Document ingestion task submitted.",
    )


@router.get("/api/documents", response_model=DocumentListApiResponse, tags=["api-documents"])
def api_list_documents() -> DocumentListApiResponse:
    return _document_service().list_documents_api()


@router.get(
    "/api/documents/tasks/{task_id}",
    response_model=DocumentIngestionTaskApiResponse,
    tags=["api-documents"],
)
def api_get_document_ingestion_task(task_id: str) -> DocumentIngestionTaskApiResponse:
    request_id = str(uuid4())
    start = perf_counter()
    task = _ingestion_task_service().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Ingestion task not found: {task_id}")
    return _task_response(
        task,
        request_id=request_id,
        start=start,
        message=f"Document ingestion task is {task['status']}.",
    )


@router.post("/api/documents/{document_id}/reindex", response_model=DocumentReindexApiResponse, tags=["api-documents"])
def api_reindex_document(document_id: str) -> DocumentReindexApiResponse:
    try:
        with _ingestion_task_service().document_operation_lock(document_id, "SYNC_REINDEX"):
            return _document_service().reindex_document(document_id)
    except DocumentOperationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/api/documents/{document_id}/reindex/async",
    response_model=DocumentIngestionTaskApiResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["api-documents"],
)
def api_reindex_document_async(
    document_id: str,
) -> DocumentIngestionTaskApiResponse | JSONResponse:
    request_id = str(uuid4())
    start = perf_counter()
    document = _document_service().storage_service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    task_service = _ingestion_task_service()
    requested_task_id = str(uuid4())
    try:
        task = task_service.create_task(
            document_id,
            "REINDEX",
            task_id=requested_task_id,
        )
    except DocumentOperationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if task["task_id"] != requested_task_id:
        return _task_response(
            task,
            request_id=request_id,
            start=start,
            message="An active document ingestion task already exists.",
        )

    try:
        enqueue_document_ingestion(str(task["task_id"]))
    except Exception as exc:
        failed_task = _record_dispatch_failure(task_service, task, exc)
        response = _task_response(
            failed_task,
            request_id=request_id,
            start=start,
            message="Document reindex task submission failed. Submit async reindex again to retry.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return _task_response(
        task,
        request_id=request_id,
        start=start,
        message="Document reindex task submitted.",
    )


@router.delete("/api/documents/batch", response_model=BatchDeleteDocumentsResponse, tags=["api-documents"])
def api_batch_delete_documents(request: BatchDeleteDocumentsRequest) -> BatchDeleteDocumentsResponse:
    try:
        with ExitStack() as locks:
            task_service = _ingestion_task_service()
            for document_id in sorted(set(request.document_ids)):
                locks.enter_context(
                    task_service.document_operation_lock(document_id, "DELETE")
                )
            return _document_service().batch_delete_documents(request.document_ids)
    except DocumentOperationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/api/documents/{document_id}", response_model=DocumentDeleteApiResponse, tags=["api-documents"])
def api_delete_document(document_id: str) -> DocumentDeleteApiResponse:
    try:
        with _ingestion_task_service().document_operation_lock(document_id, "DELETE"):
            return _document_service().delete_document_api(document_id)
    except DocumentOperationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/documents/summary", response_model=SummaryApiResponse, tags=["api-documents"])
def api_summarize_document(request: SummaryApiRequest) -> SummaryApiResponse:
    return _document_service().summarize_document_api(request.document_id)


def _record_dispatch_failure(
    task_service: IngestionTaskService,
    task: dict[str, object],
    exc: Exception,
) -> dict[str, object]:
    task_id = str(task["task_id"])
    error_message = f"Celery task submission failed: {type(exc).__name__}: {exc}"
    LOGGER.exception(
        "Celery task submission failed: task_id=%s document_id=%s operation=%s",
        task_id,
        task.get("document_id"),
        task.get("operation"),
    )
    current = task_service.get_task(task_id)
    if current is None:
        raise RuntimeError(f"Ingestion task disappeared after creation: {task_id}") from exc
    if current["status"] == "PENDING":
        current = task_service.mark_processing(task_id)
    if current["status"] == "PROCESSING":
        current = task_service.record_failure(
            task_id,
            error_message,
            result={
                "stage": "broker_dispatch",
                "recoverable": True,
                "error_type": type(exc).__name__,
            },
        )
    return current


def _task_response(
    task: dict[str, object],
    *,
    request_id: str,
    start: float,
    message: str,
) -> DocumentIngestionTaskApiResponse:
    return DocumentIngestionTaskApiResponse(
        request_id=request_id,
        task_id=str(task["task_id"]),
        document_id=str(task["document_id"]),
        operation=str(task["operation"]),
        status=str(task["status"]),
        message=message,
        elapsed_time=round(perf_counter() - start, 6),
        error_message=str(task.get("error_message") or ""),
        retry_count=int(task.get("retry_count") or 0),
        chunk_count=int(task.get("chunk_count") or 0),
        vector_indexed=bool(task.get("vector_indexed")),
        created_time=str(task.get("created_time") or ""),
        updated_time=str(task.get("updated_time") or ""),
    )
