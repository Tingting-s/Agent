from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.api import routes
from app.main import app
from app.rag.vector_store import (
    ChromaVectorStore,
    HashEmbeddingFunction,
    LocalVectorCollection,
)
from app.services.document_storage_service import DocumentStorageService
from app.services.ingestion_task_service import IngestionTaskService


client = TestClient(app)


@pytest.fixture
def dispatched_task_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    dispatched: list[str] = []
    monkeypatch.setattr(
        documents_api,
        "enqueue_document_ingestion",
        lambda task_id: dispatched.append(task_id),
    )
    return dispatched


def test_async_upload_returns_202_and_task_id(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
) -> None:
    response = _async_upload("async-policy.txt", "Async policy document.")

    assert response.status_code == 202
    payload = response.json()
    assert payload["task_id"]
    assert payload["document_id"]
    assert payload["status"] == "PENDING"
    assert payload["operation"] == "UPLOAD"
    assert dispatched_task_ids == [payload["task_id"]]


def test_async_upload_does_not_run_indexing_in_request(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routes.document_service,
        "_index_uploaded_document",
        lambda **kwargs: pytest.fail("Async upload must not index in the request"),
    )

    response = _async_upload("no-inline-index.txt", "Index this in the worker.")

    assert response.status_code == 202
    document = routes.document_service.storage_service.get_document(
        response.json()["document_id"]
    )
    assert document is not None
    assert document["status"] == "pending"
    assert document["chunk_count"] == 0
    assert document["vector_indexed"] is False


def test_async_upload_passes_sqlite_task_id_to_celery(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
) -> None:
    response = _async_upload("task-id.txt", "Use one task identifier.")
    payload = response.json()
    persisted = IngestionTaskService(
        isolated_runtime_paths["document_db_path"]
    ).get_task(payload["task_id"])

    assert persisted is not None
    assert persisted["task_id"] == dispatched_task_ids[0]
    assert persisted["document_id"] == payload["document_id"]


def test_task_status_query_returns_pending(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    task = _create_task(isolated_runtime_paths, status="PENDING")

    response = client.get(f"/api/documents/tasks/{task['task_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PENDING"
    assert payload["operation"] == "UPLOAD"
    assert payload["created_time"]
    assert payload["updated_time"]


def test_task_status_query_returns_processing(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    task = _create_task(isolated_runtime_paths, status="PROCESSING")

    response = client.get(f"/api/documents/tasks/{task['task_id']}")

    assert response.status_code == 200
    assert response.json()["status"] == "PROCESSING"


def test_task_status_query_returns_success_and_chunk_count(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    task = _create_task(isolated_runtime_paths, status="SUCCESS", chunk_count=4)

    response = client.get(f"/api/documents/tasks/{task['task_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "SUCCESS"
    assert payload["chunk_count"] == 4
    assert payload["vector_indexed"] is True


def test_task_status_query_returns_failure_reason(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    task = _create_task(
        isolated_runtime_paths,
        status="FAILURE",
        error_message="unsupported document content",
    )

    response = client.get(f"/api/documents/tasks/{task['task_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "FAILURE"
    assert payload["error_message"] == "unsupported document content"


def test_missing_task_status_returns_404() -> None:
    response = client.get("/api/documents/tasks/missing-task")

    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


def test_duplicate_async_reindex_returns_existing_active_task(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
) -> None:
    document_id = _create_document(isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    active = task_service.create_task(document_id, "REINDEX", task_id="active-task")

    response = client.post(f"/api/documents/{document_id}/reindex/async")

    assert response.status_code == 202
    assert response.json()["task_id"] == active["task_id"]
    assert dispatched_task_ids == []
    assert len(task_service.list_history(document_id)) == 1


def test_completed_task_allows_new_async_reindex(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
) -> None:
    document_id = _create_document(isolated_runtime_paths)
    task_service = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    old = task_service.create_task(document_id, "UPLOAD", task_id="completed-task")
    task_service.mark_processing(old["task_id"])
    task_service.record_success(old["task_id"], chunk_count=1)

    response = client.post(f"/api/documents/{document_id}/reindex/async")

    assert response.status_code == 202
    payload = response.json()
    assert payload["task_id"] != old["task_id"]
    assert payload["operation"] == "REINDEX"
    assert dispatched_task_ids == [payload["task_id"]]


def test_redis_unavailable_marks_upload_task_failure(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dispatch(task_id: str) -> None:
        raise ConnectionError("Redis broker unavailable")

    monkeypatch.setattr(documents_api, "enqueue_document_ingestion", fail_dispatch)

    response = _async_upload("redis-down.txt", "Persist before broker dispatch.")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "FAILURE"
    assert "Redis broker unavailable" in payload["error_message"]
    task = IngestionTaskService(isolated_runtime_paths["document_db_path"]).get_task(
        payload["task_id"]
    )
    assert task is not None
    assert task["status"] == "FAILURE"
    assert task["result"]["stage"] == "broker_dispatch"
    document = routes.document_service.storage_service.get_document(payload["document_id"])
    assert document is not None
    assert document["status"] == "error"
    assert Path(str(document["file_path"])).exists()


def test_dispatch_failure_can_recover_through_async_reindex(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        documents_api,
        "enqueue_document_ingestion",
        lambda task_id: (_ for _ in ()).throw(ConnectionError("Redis unavailable")),
    )
    failed_response = _async_upload("recover.txt", "Retry this saved document.")
    failed_payload = failed_response.json()
    dispatched: list[str] = []
    monkeypatch.setattr(
        documents_api,
        "enqueue_document_ingestion",
        lambda task_id: dispatched.append(task_id),
    )

    retry_response = client.post(
        f"/api/documents/{failed_payload['document_id']}/reindex/async"
    )

    assert failed_response.status_code == 503
    assert retry_response.status_code == 202
    retry_payload = retry_response.json()
    assert retry_payload["status"] == "PENDING"
    assert retry_payload["operation"] == "REINDEX"
    assert retry_payload["task_id"] != failed_payload["task_id"]
    assert dispatched == [retry_payload["task_id"]]


def test_sync_reindex_conflicts_with_active_background_task(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = _create_document(isolated_runtime_paths)
    IngestionTaskService(isolated_runtime_paths["document_db_path"]).create_task(
        document_id,
        "REINDEX",
        task_id="active-reindex",
    )
    monkeypatch.setattr(
        routes.document_service,
        "reindex_document",
        lambda document_id: pytest.fail("Conflicting sync reindex must not run"),
    )

    response = client.post(f"/api/documents/{document_id}/reindex")

    assert response.status_code == 409
    assert "active ingestion task" in response.json()["message"].lower()


def test_delete_conflicts_with_active_background_task(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = _create_document(isolated_runtime_paths)
    IngestionTaskService(isolated_runtime_paths["document_db_path"]).create_task(
        document_id,
        "UPLOAD",
        task_id="active-upload",
    )
    monkeypatch.setattr(
        routes.document_service,
        "delete_document_api",
        lambda document_id: pytest.fail("Conflicting delete must not run"),
    )

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 409
    assert routes.document_service.storage_service.get_document(document_id) is not None


def test_batch_delete_conflicts_with_active_background_task(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    first_document_id = _create_document(isolated_runtime_paths, "batch-first")
    second_document_id = _create_document(isolated_runtime_paths, "batch-second")
    IngestionTaskService(isolated_runtime_paths["document_db_path"]).create_task(
        second_document_id,
        "UPLOAD",
        task_id="active-batch-upload",
    )

    response = client.request(
        "DELETE",
        "/api/documents/batch",
        json={"document_ids": [first_document_id, second_document_id]},
    )

    assert response.status_code == 409
    assert routes.document_service.storage_service.get_document(first_document_id) is not None
    assert routes.document_service.storage_service.get_document(second_document_id) is not None


def test_existing_sync_upload_reindex_and_qa_remain_available(
    isolated_runtime_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_store = _patch_hash_vector_store(monkeypatch, isolated_runtime_paths)
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "sync-policy.txt",
                b"Security reviews are required every quarter.",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    reindex_response = client.post(f"/api/documents/{document_id}/reindex")
    qa_response = client.post(
        "/api/qa/ask",
        json={
            "document_id": document_id,
            "question": "How often are security reviews required?",
        },
    )

    assert upload_response.status_code == 200
    assert reindex_response.status_code == 200
    assert reindex_response.json()["status"] == "success"
    assert qa_response.status_code == 200
    assert qa_response.json()["status"] == "success"
    assert qa_response.json()["source_chunks"]
    assert vector_store.get_document_chunk_count(document_id) > 0


def test_new_service_instance_reads_api_task_history(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
) -> None:
    response = _async_upload("persistent-task.txt", "Persist this task state.")
    payload = response.json()

    recreated = IngestionTaskService(isolated_runtime_paths["document_db_path"])
    restored = recreated.get_task(payload["task_id"])

    assert restored is not None
    assert restored["document_id"] == payload["document_id"]
    assert restored["status"] == "PENDING"


def test_async_upload_rejects_oversized_file_before_dispatch(
    isolated_runtime_paths: dict[str, Path],
    dispatched_task_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        documents_api,
        "get_settings",
        lambda: SimpleNamespace(MAX_ASYNC_UPLOAD_SIZE_MB=1),
    )

    response = client.post(
        "/api/documents/upload/async",
        files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 400
    assert "size limit" in response.json()["message"].lower()
    assert dispatched_task_ids == []
    assert routes.document_service.storage_service.list_documents() == []


def _async_upload(file_name: str, content: str):
    return client.post(
        "/api/documents/upload/async",
        files={"file": (file_name, content.encode("utf-8"), "text/plain")},
    )


def _create_task(
    paths: dict[str, Path],
    *,
    status: str,
    chunk_count: int = 0,
    error_message: str = "",
) -> dict[str, Any]:
    task_service = IngestionTaskService(paths["document_db_path"])
    task = task_service.create_task(
        f"document-{status.lower()}",
        "UPLOAD",
        task_id=f"task-{status.lower()}",
    )
    if status != "PENDING":
        task = task_service.mark_processing(task["task_id"])
    if status == "SUCCESS":
        task = task_service.record_success(
            task["task_id"],
            chunk_count=chunk_count,
            vector_indexed=True,
        )
    elif status == "FAILURE":
        task = task_service.record_failure(task["task_id"], error_message)
    return task


def _create_document(
    paths: dict[str, Path],
    document_id: str = "document-1",
) -> str:
    file_path = paths["upload_dir"] / f"{document_id}_document.txt"
    file_path.write_text("Document content for asynchronous reindex.", encoding="utf-8")
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
    return document_id


def _patch_hash_vector_store(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> ChromaVectorStore:
    embedding = HashEmbeddingFunction()
    vector_store = ChromaVectorStore(
        persist_directory=paths["chroma_dir"],
        collection_name="test_document_ingestion_tasks_api",
        embedding_function=embedding,
    )
    vector_store._collection = LocalVectorCollection(embedding_function=embedding)
    monkeypatch.setattr(
        "app.services.document_qa_service.get_vector_store",
        lambda: vector_store,
    )
    monkeypatch.setattr(
        "app.agents.document_agent.get_vector_store",
        lambda: vector_store,
    )
    monkeypatch.setattr(
        "app.rag.retrieval.get_vector_store",
        lambda: vector_store,
    )
    return vector_store
