from __future__ import annotations

import asyncio
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import UploadFile

from app.celery_app import celery_app
from app.rag.vector_store import (
    ChromaVectorStore,
    HashEmbeddingFunction,
    LocalVectorCollection,
)
from app.services.document_qa_service import DocumentQAService
from app.services.document_storage_service import DocumentStorageService
from app.services.ingestion_task_service import IngestionTaskService
from app.tasks import document_ingestion as task_module
from app.tasks.document_ingestion import (
    enqueue_document_ingestion,
    ingest_document,
    run_document_ingestion,
)


@pytest.fixture(autouse=True)
def eager_celery(monkeypatch: pytest.MonkeyPatch):
    previous_eager = celery_app.conf.task_always_eager
    previous_propagates = celery_app.conf.task_eager_propagates
    previous_store_eager = celery_app.conf.task_store_eager_result
    previous_max_retries = ingest_document.max_retries
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=False,
        task_store_eager_result=False,
    )
    ingest_document.max_retries = 2
    monkeypatch.setattr(
        task_module,
        "get_settings",
        lambda: SimpleNamespace(CELERY_TASK_RETRY_DELAY=0),
    )
    yield
    celery_app.conf.update(
        task_always_eager=previous_eager,
        task_eager_propagates=previous_propagates,
        task_store_eager_result=previous_store_eager,
    )
    ingest_document.max_retries = previous_max_retries


def test_eager_task_ingests_document_successfully(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "background-document"
    _create_document(isolated_runtime_paths, document_id, "Startup requires a power and guard check.")
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task(document_id, "UPLOAD", task_id="background-task")

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.successful()
    persisted = task_service.get_task(task["task_id"])
    assert persisted is not None
    assert persisted["status"] == "SUCCESS"
    assert persisted["chunk_count"] > 0
    assert persisted["vector_indexed"] is True
    assert vector_store.get_document_chunk_count(document_id) == persisted["chunk_count"]


def test_task_is_processing_before_document_work(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task("document-1", "UPLOAD", task_id="task-1")

    class InspectingDocumentService:
        storage_service = SimpleNamespace(get_document=lambda document_id: {"document_id": document_id})

        def reindex_document(self, document_id: str):
            current = task_service.get_task(task["task_id"])
            assert current is not None
            assert current["status"] == "PROCESSING"
            return _success_response(chunk_count=2)

    result = run_document_ingestion(
        task["task_id"],
        task_service=task_service,
        document_service=InspectingDocumentService(),
    )

    assert result["status"] == "SUCCESS"


def test_recoverable_error_retries_and_succeeds(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task("document-1", "UPLOAD", task_id="retry-task")
    calls = 0

    class FlakyDocumentService:
        storage_service = SimpleNamespace(get_document=lambda document_id: {"document_id": document_id})

        def reindex_document(self, document_id: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary vector store timeout")
            return _success_response(chunk_count=3)

    monkeypatch.setattr(task_module, "DocumentQAService", FlakyDocumentService)

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.successful()
    persisted = task_service.get_task(task["task_id"])
    assert calls == 2
    assert persisted is not None
    assert persisted["status"] == "SUCCESS"
    assert persisted["retry_count"] == 1


def test_permanent_error_fails_without_retry(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task("missing-document", "UPLOAD", task_id="permanent-task")
    calls = 0

    class MissingFileDocumentService:
        storage_service = SimpleNamespace(get_document=lambda document_id: None)

        def reindex_document(self, document_id: str):
            nonlocal calls
            calls += 1
            return _error_response("Document file was not found.")

    monkeypatch.setattr(task_module, "DocumentQAService", MissingFileDocumentService)

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.failed()
    persisted = task_service.get_task(task["task_id"])
    assert calls == 1
    assert persisted is not None
    assert persisted["status"] == "FAILURE"
    assert persisted["retry_count"] == 0
    assert "not found" in persisted["error_message"].lower()


def test_retry_exhaustion_records_failure(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task("document-1", "UPLOAD", task_id="exhausted-task")
    calls = 0
    ingest_document.max_retries = 1

    class UnavailableDocumentService:
        def __init__(self) -> None:
            self.storage_service = SimpleNamespace(
                get_document=lambda document_id: {"document_id": document_id}
            )

        def reindex_document(self, document_id: str):
            nonlocal calls
            calls += 1
            raise ConnectionError("temporary Chroma connection failure")

    monkeypatch.setattr(task_module, "DocumentQAService", UnavailableDocumentService)

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.failed()
    persisted = task_service.get_task(task["task_id"])
    assert calls == 2
    assert persisted is not None
    assert persisted["status"] == "FAILURE"
    assert persisted["retry_count"] == 1
    assert persisted["result"]["recoverable"] is True


def test_duplicate_delivery_does_not_duplicate_vectors(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "duplicate-document"
    _create_document(isolated_runtime_paths, document_id, "One stable chunk for duplicate delivery.")
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task(document_id, "UPLOAD", task_id="duplicate-task")

    first = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])
    first_ids = _document_vector_ids(vector_store, document_id)

    monkeypatch.setattr(
        task_module,
        "DocumentQAService",
        lambda: pytest.fail("A completed task must not execute ingestion again"),
    )
    second = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert first.successful()
    assert second.successful()
    assert _document_vector_ids(vector_store, document_id) == first_ids


def test_reindex_replaces_old_chunks(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "reindex-document"
    file_path = _create_document(
        isolated_runtime_paths,
        document_id,
        "Long original content. " * 500,
    )
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    upload_task = task_service.create_task(document_id, "UPLOAD", task_id="initial-task")
    assert ingest_document.apply(args=[upload_task["task_id"]]).successful()
    old_ids = _document_vector_ids(vector_store, document_id)
    assert len(old_ids) > 1

    file_path.write_text("Short replacement content.", encoding="utf-8")
    reindex_task = task_service.create_task(document_id, "REINDEX", task_id="reindex-task")
    result = ingest_document.apply(args=[reindex_task["task_id"]], task_id=reindex_task["task_id"])
    new_ids = _document_vector_ids(vector_store, document_id)

    assert result.successful()
    assert len(new_ids) == 1
    assert len(new_ids) < len(old_ids)
    persisted = task_service.get_task(reindex_task["task_id"])
    assert persisted is not None
    assert persisted["operation"] == "REINDEX"
    assert persisted["chunk_count"] == 1


def test_status_write_failure_after_vector_success_recovers(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = "status-recovery-document"
    _create_document(isolated_runtime_paths, document_id, "Recoverable state write document.")
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = task_service.create_task(document_id, "UPLOAD", task_id="status-recovery-task")
    original_record_success = IngestionTaskService.record_success
    calls = 0
    ingest_document.max_retries = 1

    def flaky_record_success(self, *args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is temporarily locked")
        return original_record_success(self, *args, **kwargs)

    monkeypatch.setattr(IngestionTaskService, "record_success", flaky_record_success)

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.successful()
    persisted = task_service.get_task(task["task_id"])
    assert calls == 2
    assert persisted is not None
    assert persisted["status"] == "SUCCESS"
    assert persisted["retry_count"] == 1
    assert vector_store.get_document_chunk_count(document_id) == persisted["chunk_count"]


def test_worker_uses_independent_task_service_instance(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creator = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    task = creator.create_task("document-1", "UPLOAD", task_id="independent-task")
    constructed_services: list[IngestionTaskService] = []
    original_service_class = IngestionTaskService

    def service_factory(*args: Any, **kwargs: Any) -> IngestionTaskService:
        service = original_service_class(*args, **kwargs)
        constructed_services.append(service)
        return service

    class SuccessfulDocumentService:
        storage_service = SimpleNamespace(get_document=lambda document_id: {"document_id": document_id})

        def reindex_document(self, document_id: str):
            return _success_response(chunk_count=1)

    monkeypatch.setattr(task_module, "IngestionTaskService", service_factory)
    monkeypatch.setattr(task_module, "DocumentQAService", SuccessfulDocumentService)

    result = ingest_document.apply(args=[task["task_id"]], task_id=task["task_id"])

    assert result.successful()
    assert constructed_services
    assert all(service is not creator for service in constructed_services)
    reader = original_service_class(isolated_runtime_paths["document_db_path"])
    assert reader.get_task(task["task_id"])["status"] == "SUCCESS"


def test_existing_synchronous_upload_still_indexes_document(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    service = DocumentQAService()
    upload = UploadFile(
        filename="synchronous.txt",
        file=BytesIO(b"Synchronous upload still indexes this document."),
    )

    response = asyncio.run(service.upload_document(upload))

    assert response.status == "success"
    assert response.vector_indexed is True
    results = vector_store.search(
        document_id=response.document_id,
        question="What does the synchronous upload do?",
        top_k=1,
    )
    assert results
    assert results[0]["document_id"] == response.document_id


def test_enqueue_uses_sqlite_task_id_as_celery_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_apply_async(*, args: list[str], task_id: str):
        captured["args"] = args
        captured["task_id"] = task_id
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(ingest_document, "apply_async", fake_apply_async)

    result = enqueue_document_ingestion("sqlite-task-id")

    assert result.id == "sqlite-task-id"
    assert captured == {
        "args": ["sqlite-task-id"],
        "task_id": "sqlite-task-id",
    }


def _create_document(
    paths: dict[str, Path],
    document_id: str,
    content: str,
) -> Path:
    file_path = paths["upload_dir"] / f"{document_id}_document.txt"
    file_path.write_text(content, encoding="utf-8")
    DocumentStorageService(paths["document_db_path"]).upsert_document(
        document_id=document_id,
        original_file_name="document.txt",
        stored_file_name=file_path.name,
        file_path=str(file_path),
        file_type="txt",
        status="success",
        chunk_count=0,
        vector_indexed=False,
    )
    return file_path


def _patch_hash_vector_store(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> ChromaVectorStore:
    embedding = HashEmbeddingFunction()
    vector_store = ChromaVectorStore(
        persist_directory=paths["chroma_dir"],
        collection_name="test_document_ingestion_task",
        embedding_function=embedding,
    )
    vector_store._collection = LocalVectorCollection(embedding_function=embedding)
    monkeypatch.setattr(
        "app.services.document_qa_service.get_vector_store",
        lambda: vector_store,
    )
    return vector_store


def _document_vector_ids(
    vector_store: ChromaVectorStore,
    document_id: str,
) -> set[str]:
    result = vector_store.collection.get(where={"document_id": document_id}, include=[])
    return set(result.get("ids") or [])


def _success_response(chunk_count: int):
    return SimpleNamespace(
        status="success",
        message="Document reindexed successfully.",
        error_message="",
        chunk_count=chunk_count,
        vector_indexed=True,
    )


def _error_response(message: str):
    return SimpleNamespace(
        status="error",
        message="Document reindex failed.",
        error_message=message,
        chunk_count=0,
        vector_indexed=False,
    )
