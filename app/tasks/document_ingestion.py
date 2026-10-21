"""Celery task for idempotent background document ingestion."""

from __future__ import annotations

import logging
from typing import Any

from celery import Task

from app.celery_app import celery_app
from app.config import get_settings
from app.services.document_qa_service import DocumentQAService
from app.services.ingestion_task_service import (
    IngestionTaskNotFoundError,
    IngestionTaskService,
)


LOGGER = logging.getLogger(__name__)


class PermanentIngestionError(RuntimeError):
    """An ingestion error that cannot be fixed by retrying the same input."""


class RecoverableIngestionError(RuntimeError):
    """An ingestion error that may succeed on a later attempt."""


PERMANENT_ERROR_MARKERS = (
    "document not found",
    "document file was not found",
    "file does not exist",
    "unsupported document",
    "only .txt",
    "uploaded document is empty",
    "produced no chunks",
    "document parsing failed",
    "failed to read document content",
)


def run_document_ingestion(
    task_id: str,
    *,
    retry_count: int = 0,
    task_service: IngestionTaskService | None = None,
    document_service: DocumentQAService | None = None,
) -> dict[str, Any]:
    """Run one idempotent ingestion attempt using fresh standalone services."""
    task_service = task_service or IngestionTaskService()
    task_record = task_service.get_task(task_id)
    if task_record is None:
        raise IngestionTaskNotFoundError(f"Ingestion task not found: {task_id}")

    status = str(task_record["status"])
    if status == "SUCCESS":
        return _task_result(task_record)
    if status == "FAILURE":
        raise PermanentIngestionError(
            f"Ingestion task is already in FAILURE state: {task_id}"
        )
    if status == "PENDING":
        task_record = task_service.mark_processing(task_id)

    persisted_retry_count = int(task_record["retry_count"])
    if retry_count > persisted_retry_count:
        task_record = task_service.update_retry_count(task_id, retry_count)

    document_id = str(task_record["document_id"])
    operation = str(task_record["operation"])
    document_service = document_service or DocumentQAService()
    response = document_service.reindex_document(document_id)

    if response.status != "success" or not response.vector_indexed:
        error_message = str(
            response.error_message or response.message or "Document ingestion failed."
        )
        if _is_permanent_message(error_message):
            raise PermanentIngestionError(error_message)
        raise RecoverableIngestionError(error_message)

    document = document_service.storage_service.get_document(document_id)
    if document is None:
        _cleanup_orphan_vectors(document_service, document_id)
        raise PermanentIngestionError(
            f"Document was deleted while ingestion was running: {document_id}"
        )

    count_reader = getattr(document_service, "get_document_vector_count", None)
    if callable(count_reader):
        indexed_count = int(count_reader(document_id))
        if indexed_count != int(response.chunk_count):
            raise RecoverableIngestionError(
                "Vector count verification failed: "
                f"expected={int(response.chunk_count)} actual={indexed_count}"
            )

    completed = task_service.record_success(
        task_id,
        chunk_count=int(response.chunk_count),
        vector_indexed=True,
        result={
            "document_id": document_id,
            "operation": operation,
            "chunk_count": int(response.chunk_count),
            "vector_indexed": True,
        },
    )
    return _task_result(completed)


@celery_app.task(
    bind=True,
    name="app.tasks.document_ingestion.ingest_document",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=get_settings().CELERY_TASK_MAX_RETRIES,
)
def ingest_document(self: Task, task_id: str) -> dict[str, Any]:
    """Execute ingestion, retry transient failures, and persist final state."""
    retry_count = int(self.request.retries or 0)
    try:
        return run_document_ingestion(task_id, retry_count=retry_count)
    except PermanentIngestionError as exc:
        _record_terminal_failure(
            task_id,
            exc,
            retry_count=retry_count,
            recoverable=False,
        )
        LOGGER.error(
            "Document ingestion failed permanently: task_id=%s retry_count=%s error_type=%s error=%s",
            task_id,
            retry_count,
            type(exc).__name__,
            exc,
        )
        raise
    except Exception as exc:
        max_retries = max(int(self.max_retries or 0), 0)
        if retry_count < max_retries:
            LOGGER.warning(
                "Document ingestion will be retried: task_id=%s retry_count=%s max_retries=%s error_type=%s error=%s",
                task_id,
                retry_count,
                max_retries,
                type(exc).__name__,
                exc,
            )
            raise self.retry(
                exc=exc,
                countdown=max(get_settings().CELERY_TASK_RETRY_DELAY, 0),
                max_retries=max_retries,
            )

        _record_terminal_failure(
            task_id,
            exc,
            retry_count=retry_count,
            recoverable=True,
        )
        LOGGER.exception(
            "Document ingestion exhausted retries: task_id=%s retry_count=%s error_type=%s error=%s",
            task_id,
            retry_count,
            type(exc).__name__,
            exc,
        )
        raise


def enqueue_document_ingestion(task_id: str) -> Any:
    """Publish an ingestion job using the SQLite task ID as the Celery ID."""
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        raise ValueError("task_id must not be empty")
    return ingest_document.apply_async(
        args=[normalized_task_id],
        task_id=normalized_task_id,
    )


def _record_terminal_failure(
    task_id: str,
    exc: Exception,
    *,
    retry_count: int,
    recoverable: bool,
) -> None:
    task_service = IngestionTaskService()
    task_record = task_service.get_task(task_id)
    if task_record is None or task_record["status"] in {"SUCCESS", "FAILURE"}:
        return
    if task_record["status"] == "PENDING":
        task_service.mark_processing(task_id)
    if retry_count > int(task_record["retry_count"]):
        task_service.update_retry_count(task_id, retry_count)
    task_service.record_failure(
        task_id,
        str(exc),
        result={
            "error_type": type(exc).__name__,
            "recoverable": recoverable,
            "retry_count": retry_count,
        },
    )


def _is_permanent_message(message: str) -> bool:
    normalized = message.strip().lower()
    return any(marker in normalized for marker in PERMANENT_ERROR_MARKERS)


def _cleanup_orphan_vectors(
    document_service: DocumentQAService,
    document_id: str,
) -> None:
    try:
        from app.rag.vector_store import get_vector_store

        get_vector_store().delete_document(document_id)
    except Exception:
        LOGGER.exception(
            "Failed to clean vectors for a document deleted during ingestion: document_id=%s",
            document_id,
        )


def _task_result(task_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task_record["task_id"]),
        "document_id": str(task_record["document_id"]),
        "operation": str(task_record["operation"]),
        "status": str(task_record["status"]),
        "retry_count": int(task_record["retry_count"]),
        "chunk_count": int(task_record["chunk_count"]),
        "vector_indexed": bool(task_record["vector_indexed"]),
        "result": dict(task_record.get("result") or {}),
    }


__all__ = [
    "PermanentIngestionError",
    "RecoverableIngestionError",
    "enqueue_document_ingestion",
    "ingest_document",
    "run_document_ingestion",
]
