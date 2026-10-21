from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.services.ingestion_task_service import (
    DocumentOperationConflictError,
    IngestionTaskService,
    InvalidTaskTransitionError,
)


@pytest.fixture
def service(tmp_path):
    return IngestionTaskService(tmp_path / "ingestion_tasks.db")


def test_create_task(service):
    task = service.create_task("document-1", "UPLOAD", task_id="task-1")

    assert task["task_id"] == "task-1"
    assert task["document_id"] == "document-1"
    assert task["operation"] == "UPLOAD"
    assert task["status"] == "PENDING"
    assert task["retry_count"] == 0
    assert task["chunk_count"] == 0
    assert task["vector_indexed"] is False
    assert task["result"] == {}


def test_query_task_status(service):
    service.create_task("document-1", "UPLOAD", task_id="task-1")

    task = service.get_task("task-1")

    assert task is not None
    assert task["status"] == "PENDING"
    assert service.get_task("missing") is None


def test_pending_to_processing_to_success(service):
    service.create_task("document-1", "UPLOAD", task_id="task-1")

    processing = service.mark_processing("task-1")
    successful = service.record_success(
        "task-1",
        chunk_count=4,
        vector_indexed=True,
        result={"collection": "documents"},
    )

    assert processing["status"] == "PROCESSING"
    assert successful["status"] == "SUCCESS"
    assert successful["chunk_count"] == 4
    assert successful["vector_indexed"] is True
    assert successful["result"] == {"collection": "documents"}


def test_pending_to_processing_to_failure(service):
    service.create_task("document-1", "UPLOAD", task_id="task-1")
    service.mark_processing("task-1")

    failed = service.record_failure("task-1", "embedding unavailable")

    assert failed["status"] == "FAILURE"
    assert failed["error_message"] == "embedding unavailable"
    assert failed["vector_indexed"] is False


def test_retry_count_updates_only_while_processing(service):
    service.create_task("document-1", "UPLOAD", task_id="task-1")
    service.mark_processing("task-1")

    updated = service.update_retry_count("task-1", 2)
    incremented = service.increment_retry_count("task-1")

    assert updated["retry_count"] == 2
    assert incremented["retry_count"] == 3


def test_duplicate_submission_returns_existing_active_task(service):
    first = service.create_task("document-1", "UPLOAD", task_id="task-1")
    duplicate = service.create_task("document-1", "REINDEX", task_id="task-2")

    assert duplicate["task_id"] == first["task_id"]
    assert len(service.list_history("document-1")) == 1


def test_concurrent_creation_produces_one_active_task(service):
    worker_count = 8
    barrier = Barrier(worker_count)

    def create(index):
        barrier.wait()
        return service.create_task(
            "document-1", "UPLOAD", task_id=f"concurrent-task-{index}"
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        tasks = list(executor.map(create, range(worker_count)))

    assert len({task["task_id"] for task in tasks}) == 1
    history = service.list_history("document-1")
    assert len(history) == 1
    assert history[0]["status"] == "PENDING"


def test_completed_task_allows_new_reindex_task(service):
    service.create_task("document-1", "UPLOAD", task_id="upload-task")
    service.mark_processing("upload-task")
    service.record_success("upload-task", chunk_count=2)

    reindex = service.create_task(
        "document-1", "REINDEX", task_id="reindex-task"
    )

    assert reindex["task_id"] == "reindex-task"
    assert reindex["operation"] == "REINDEX"
    assert reindex["status"] == "PENDING"
    assert len(service.list_history("document-1")) == 2


def test_invalid_status_transitions_are_rejected(service):
    service.create_task("document-1", "UPLOAD", task_id="task-1")

    with pytest.raises(InvalidTaskTransitionError):
        service.update_status("task-1", "SUCCESS")

    service.mark_processing("task-1")
    service.record_success("task-1", chunk_count=1)

    with pytest.raises(InvalidTaskTransitionError):
        service.update_status("task-1", "PROCESSING")


def test_history_survives_service_recreation(tmp_path):
    db_path = tmp_path / "persistent_tasks.db"
    first_service = IngestionTaskService(db_path)
    first_service.create_task("document-1", "UPLOAD", task_id="task-1")
    first_service.mark_processing("task-1")
    first_service.record_success("task-1", chunk_count=3)

    recreated_service = IngestionTaskService(db_path)
    restored = recreated_service.get_task("task-1")

    assert restored is not None
    assert restored["status"] == "SUCCESS"
    assert restored["chunk_count"] == 3


def test_different_documents_have_independent_active_tasks(service):
    first = service.create_task("document-1", "UPLOAD", task_id="task-1")
    second = service.create_task("document-2", "UPLOAD", task_id="task-2")

    assert first["task_id"] != second["task_id"]
    assert service.get_active_task("document-1")["task_id"] == "task-1"
    assert service.get_active_task("document-2")["task_id"] == "task-2"


def test_failure_reason_and_result_are_persisted(service):
    service.create_task("document-1", "REINDEX", task_id="task-1")
    service.mark_processing("task-1")

    failed = service.record_failure(
        "task-1",
        "vector store write failed",
        result={"stage": "vector_index", "retryable": True},
    )
    restored = service.get_task("task-1")

    assert failed["error_message"] == "vector store write failed"
    assert restored is not None
    assert restored["result"] == {"stage": "vector_index", "retryable": True}


def test_document_operation_lock_blocks_task_creation_until_released(service):
    with service.document_operation_lock("document-1", "DELETE"):
        with pytest.raises(DocumentOperationConflictError):
            service.create_task("document-1", "REINDEX", task_id="blocked-task")

    created = service.create_task("document-1", "REINDEX", task_id="allowed-task")

    assert created["task_id"] == "allowed-task"


def test_active_task_blocks_document_operation_lock(service):
    service.create_task("document-1", "UPLOAD", task_id="active-task")

    with pytest.raises(DocumentOperationConflictError):
        with service.document_operation_lock("document-1", "DELETE"):
            pytest.fail("The operation lock must not be acquired")
