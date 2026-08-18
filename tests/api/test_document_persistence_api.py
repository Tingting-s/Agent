from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.rag.vector_store import ChromaVectorStore, HashEmbeddingFunction, get_vector_store
from app.services.document_qa_service import DocumentQAService
from app.services.qa_log_service import QALogService


client = TestClient(app)


@pytest.fixture()
def isolated_document_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentQAService:
    log_service = QALogService(log_path=tmp_path / "logs" / "qa_logs.jsonl")
    service = DocumentQAService(storage_dir=tmp_path / "uploads", log_service=log_service)
    monkeypatch.setattr(routes, "document_service", service)
    return service


def test_upload_document_writes_sqlite_metadata(
    isolated_document_service: DocumentQAService,
) -> None:
    payload = _upload_text_document("persistent-policy.txt", "Persistent Policy\nReviews are quarterly.")

    record = isolated_document_service.storage_service.get_document(str(payload["document_id"]))

    assert record is not None
    assert record["document_id"] == payload["document_id"]
    assert record["original_file_name"] == "persistent-policy.txt"
    assert record["stored_file_name"] == payload["file_name"]
    assert record["status"] == "success"
    assert record["chunk_count"] > 0
    assert record["vector_indexed"] is True
    assert Path(record["file_path"]).exists()


def test_list_documents_api_returns_persisted_documents(
    isolated_document_service: DocumentQAService,
) -> None:
    first = _upload_text_document("first.txt", "First document.")
    second = _upload_text_document("second.txt", "Second document.")

    response = client.get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["request_id"]
    assert payload["error_message"] == ""
    document_ids = [item["document_id"] for item in payload["items"]]
    assert second["document_id"] in document_ids
    assert first["document_id"] in document_ids
    item = next(item for item in payload["items"] if item["document_id"] == second["document_id"])
    assert item["original_file_name"] == "second.txt"
    assert item["stored_file_name"] == second["file_name"]
    assert item["upload_time"]
    assert item["chunk_count"] > 0
    assert item["vector_indexed"] is True


def test_streamlit_document_loader_uses_documents_api(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_payload = _upload_text_document("streamlit-list.txt", "Streamlit list document.")
    list_payload = client.get("/api/documents").json()

    from web import streamlit_app

    def fake_api_get(path: str, *, api_base_url: str, timeout: int = 20) -> tuple[dict[str, Any], str]:
        assert path == "/api/documents"
        assert api_base_url == "http://testserver"
        return list_payload, ""

    monkeypatch.setattr(streamlit_app, "api_get", fake_api_get)

    documents, error = streamlit_app.fetch_documents("http://testserver")
    normalized = [streamlit_app._normalize_document_entry(document) for document in documents]

    assert error == ""
    assert normalized
    loaded = next(document for document in normalized if document["document_id"] == upload_payload["document_id"])
    assert loaded["display_name"] == "streamlit-list.txt"
    assert loaded["stored_file_name"] == upload_payload["file_name"]


def test_streamlit_qa_request_uses_document_id_not_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web import streamlit_app

    captured: dict[str, Any] = {}
    document_id = str(uuid4())

    def fake_api_post(path: str, *, api_base_url: str, timeout: int = 20, **kwargs: Any) -> tuple[dict[str, Any], str]:
        captured["path"] = path
        captured["api_base_url"] = api_base_url
        captured["json"] = kwargs.get("json")
        return {"status": "success"}, ""

    monkeypatch.setattr(streamlit_app, "api_post", fake_api_post)

    payload, error = streamlit_app.ask_question(
        "http://testserver",
        document_id,
        "What should be checked?",
    )

    assert error == ""
    assert payload == {"status": "success"}
    assert captured["path"] == "/api/qa/ask"
    assert captured["api_base_url"] == "http://testserver"
    assert captured["json"] == {
        "document_id": document_id,
        "question": "What should be checked?",
    }


def test_streamlit_qa_resolves_display_name_to_document_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web import streamlit_app

    document_id = str(uuid4())
    documents = [
        {
            "document_id": document_id,
            "display_name": "sample_device_manual.txt",
            "original_file_name": "sample_device_manual.txt",
            "stored_file_name": f"{document_id}_sample_device_manual.txt",
            "file_name": f"{document_id}_sample_device_manual.txt",
        }
    ]
    monkeypatch.setattr(streamlit_app, "_document_options", lambda: documents)

    resolved_document_id, selected_document = streamlit_app._resolve_qa_document_id("sample_device_manual.txt")

    assert resolved_document_id == document_id
    assert selected_document["display_name"] == "sample_device_manual.txt"


def test_uploaded_document_qa_prefers_chroma_retrieval(
    isolated_document_service: DocumentQAService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_uploaded_document_qa_prefers_chroma",
        embedding_function=HashEmbeddingFunction(),
    )
    monkeypatch.setattr("app.services.document_qa_service.get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.agents.document_agent.get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.rag.retrieval.get_vector_store", lambda: vector_store)

    upload_payload = _upload_text_document(
        "startup-checks.txt",
        "Device startup checks include power supply, emergency stop button, and guard door.",
    )
    document_id = str(upload_payload["document_id"])

    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": document_id,
            "question": "What should be checked before device startup?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_source"] == "chroma"
    assert payload["source_chunks"]
    assert all(chunk["document_id"] == document_id for chunk in payload["source_chunks"])
    assert all(chunk["retrieval_source"] == "chroma" for chunk in payload["source_chunks"])


def test_delete_document_removes_metadata_file_and_vectors(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document(
        "delete-vector.txt",
        "设备启动前需要检查环境、电源、急停按钮和防护门。",
    )
    document_id = str(upload_payload["document_id"])
    stored_file = isolated_document_service.storage_dir / str(upload_payload["file_name"])

    assert get_vector_store().get_document_chunk_count(document_id) > 0

    delete_response = client.delete(f"/api/documents/{document_id}")

    assert delete_response.status_code == 200
    delete_payload = delete_response.json()
    assert delete_payload["status"] == "success"
    assert delete_payload["document_id"] == document_id
    assert delete_payload["file_deleted"] is True
    assert delete_payload["vector_deleted"] is True
    assert isolated_document_service.storage_service.get_document(document_id) is None
    assert not stored_file.exists()
    assert get_vector_store().get_document_chunk_count(document_id) == 0


def test_delete_non_indexed_document_does_not_require_chroma(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_file = isolated_document_service.storage_dir / "manual-non-indexed.txt"
    stored_file.write_text("Document that has not been indexed.", encoding="utf-8")
    isolated_document_service.storage_service.upsert_document(
        document_id="non-indexed-doc",
        original_file_name="manual-non-indexed.txt",
        stored_file_name=stored_file.name,
        file_path=str(stored_file),
        file_type="txt",
        status="success",
        chunk_count=0,
        vector_indexed=False,
    )

    def fail_vector_store():  # type: ignore[no-untyped-def]
        raise RuntimeError("Chroma initialization failed in test")

    monkeypatch.setattr("app.services.document_qa_service.get_vector_store", fail_vector_store)

    response = client.delete("/api/documents/non-indexed-doc")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["file_deleted"] is True
    assert payload["vector_deleted"] is False
    assert "vector cleanup failed or was not required" in payload["message"]
    assert isolated_document_service.storage_service.get_document("non-indexed-doc") is None
    assert not stored_file.exists()


def test_delete_missing_document_returns_clear_error(
    isolated_document_service: DocumentQAService,
) -> None:
    response = client.delete("/api/documents/missing-document")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["document_id"] == "missing-document"
    assert payload["file_deleted"] is False
    assert payload["vector_deleted"] is False
    assert "Document not found" in payload["error_message"]


def test_reindex_missing_document_returns_clear_error(
    isolated_document_service: DocumentQAService,
) -> None:
    response = client.post("/api/documents/missing-document/reindex")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["document_id"] == "missing-document"
    assert payload["chunk_count"] == 0
    assert payload["vector_indexed"] is False
    assert "Document not found" in payload["error_message"]


def test_parse_document_uses_current_upload_when_stored_path_is_stale(
    isolated_document_service: DocumentQAService,
    tmp_path: Path,
) -> None:
    document_id = "moved-project-doc"
    stored_file = isolated_document_service.storage_dir / f"{document_id}_manual.txt"
    stored_file.write_text("Moved project document.\nStartup checks include power.", encoding="utf-8")
    stale_path = tmp_path / "old-project" / "uploads" / stored_file.name
    isolated_document_service.storage_service.upsert_document(
        document_id=document_id,
        original_file_name="manual.txt",
        stored_file_name=stored_file.name,
        file_path=str(stale_path),
        file_type="txt",
        status="success",
        chunk_count=1,
        vector_indexed=True,
    )

    response = client.post("/documents/parse", json={"document_id": document_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["file_path"] == str(stored_file)

    record = isolated_document_service.storage_service.get_document(document_id)
    assert record is not None
    assert record["file_path"] == str(stored_file)


def test_reindex_existing_document_rebuilds_vectors_and_deletes_old_vectors(
    isolated_document_service: DocumentQAService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_calls: list[str] = []

    class SpyVectorStore(ChromaVectorStore):
        def delete_document(self, document_id: str) -> int:
            delete_calls.append(document_id)
            return super().delete_document(document_id)

    vector_store = SpyVectorStore(
        persist_directory=tmp_path / "chroma-reindex",
        collection_name="test_reindex_existing_document",
        embedding_function=HashEmbeddingFunction(),
    )
    monkeypatch.setattr("app.services.document_qa_service.get_vector_store", lambda: vector_store)

    upload_payload = _upload_text_document(
        "reindex-policy.txt",
        "Reindex policy document. Emergency stop and guard door checks are required before startup.",
    )
    document_id = str(upload_payload["document_id"])
    isolated_document_service.storage_service.update_index_status(
        document_id,
        chunk_count=0,
        vector_indexed=False,
        status="error",
        error_message="Previous indexing failed.",
    )

    response = client.post(f"/api/documents/{document_id}/reindex")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["document_id"] == document_id
    assert payload["chunk_count"] > 0
    assert payload["vector_indexed"] is True
    assert payload["error_message"] == ""
    assert delete_calls == [document_id]

    record = isolated_document_service.storage_service.get_document(document_id)
    assert record is not None
    assert record["chunk_count"] > 0
    assert record["vector_indexed"] is True
    assert record["status"] == "success"
    assert record["error_message"] == ""

    assert vector_store.get_document_chunk_count(document_id) > 0
    search_results = vector_store.search(document_id=document_id, question="What checks are required?", top_k=3)
    assert search_results
    assert all(result["document_id"] == document_id for result in search_results)


def test_reindex_missing_file_returns_clear_error(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document("missing-file.txt", "This document file will be removed.")
    document_id = str(upload_payload["document_id"])
    stored_file = isolated_document_service.storage_dir / str(upload_payload["file_name"])
    stored_file.unlink()

    response = client.post(f"/api/documents/{document_id}/reindex")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["document_id"] == document_id
    assert payload["chunk_count"] == 0
    assert payload["vector_indexed"] is False
    assert "Document file was not found" in payload["error_message"]

    record = isolated_document_service.storage_service.get_document(document_id)
    assert record is not None
    assert record["status"] == "error"
    assert record["chunk_count"] == 0
    assert record["vector_indexed"] is False
    assert "Document file was not found" in record["error_message"]


def test_qa_missing_file_marks_document_metadata_error(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document("qa-missing-file.txt", "This document file will be removed before QA.")
    document_id = str(upload_payload["document_id"])
    stored_file = isolated_document_service.storage_dir / str(upload_payload["file_name"])
    stored_file.unlink()

    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": document_id,
            "question": "What does the document say?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "File does not exist" in payload["error_message"]

    record = isolated_document_service.storage_service.get_document(document_id)
    assert record is not None
    assert record["status"] == "error"
    assert record["vector_indexed"] is False
    assert "File does not exist" in record["error_message"]


def test_batch_delete_documents_removes_metadata_files_and_vectors(
    isolated_document_service: DocumentQAService,
) -> None:
    first = _upload_text_document("batch-first.txt", "First batch document.")
    second = _upload_text_document("batch-second.txt", "Second batch document.")
    first_id = str(first["document_id"])
    second_id = str(second["document_id"])
    first_file = isolated_document_service.storage_dir / str(first["file_name"])
    second_file = isolated_document_service.storage_dir / str(second["file_name"])

    assert get_vector_store().get_document_chunk_count(first_id) > 0
    assert get_vector_store().get_document_chunk_count(second_id) > 0

    response = client.request(
        "DELETE",
        "/api/documents/batch",
        json={"document_ids": [first_id, second_id]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["deleted_count"] == 2
    assert payload["failed_count"] == 0
    assert {item["document_id"] for item in payload["results"]} == {first_id, second_id}
    assert all(item["status"] == "success" for item in payload["results"])
    assert isolated_document_service.storage_service.get_document(first_id) is None
    assert isolated_document_service.storage_service.get_document(second_id) is None
    assert not first_file.exists()
    assert not second_file.exists()
    assert get_vector_store().get_document_chunk_count(first_id) == 0
    assert get_vector_store().get_document_chunk_count(second_id) == 0

    list_payload = client.get("/api/documents").json()
    remaining_ids = {item["document_id"] for item in list_payload["items"]}
    assert first_id not in remaining_ids
    assert second_id not in remaining_ids


def test_batch_delete_reports_missing_document_and_continues(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document("batch-existing.txt", "Existing document.")
    document_id = str(upload_payload["document_id"])
    stored_file = isolated_document_service.storage_dir / str(upload_payload["file_name"])

    response = client.request(
        "DELETE",
        "/api/documents/batch",
        json={"document_ids": [document_id, "missing-document"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["deleted_count"] == 1
    assert payload["failed_count"] == 1
    results = {item["document_id"]: item for item in payload["results"]}
    assert results[document_id]["status"] == "success"
    assert results["missing-document"]["status"] == "failed"
    assert results["missing-document"]["message"] == "文档不存在"
    assert isolated_document_service.storage_service.get_document(document_id) is None
    assert not stored_file.exists()

    list_payload = client.get("/api/documents").json()
    assert document_id not in {item["document_id"] for item in list_payload["items"]}


def test_batch_delete_chroma_failure_does_not_block_metadata_or_file_delete(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_payload = _upload_text_document("batch-vector-warning.txt", "Vector warning document.")
    document_id = str(upload_payload["document_id"])
    stored_file = isolated_document_service.storage_dir / str(upload_payload["file_name"])
    delete_calls: list[str] = []

    class FailingVectorStore:
        def delete_document(self, target_document_id: str) -> int:
            delete_calls.append(target_document_id)
            raise RuntimeError("Chroma delete failed in test")

    monkeypatch.setattr("app.services.document_qa_service.get_vector_store", lambda: FailingVectorStore())

    response = client.request(
        "DELETE",
        "/api/documents/batch",
        json={"document_ids": [document_id]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["deleted_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["results"] == [
        {
            "document_id": document_id,
            "status": "success",
            "message": "删除成功",
        }
    ]
    assert delete_calls == [document_id]
    assert isolated_document_service.storage_service.get_document(document_id) is None
    assert not stored_file.exists()


def test_upload_vector_index_failure_keeps_document_record(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingVectorStore:
        def add_chunks(self, **kwargs: Any) -> int:
            raise RuntimeError("Chroma initialization failed in test")

    monkeypatch.setattr("app.services.document_qa_service.get_vector_store", lambda: FailingVectorStore())

    payload = _upload_text_document("vector-failure.txt", "Vector Failure\nThis document still parses.")
    record = isolated_document_service.storage_service.get_document(str(payload["document_id"]))

    assert payload["status"] == "success"
    assert payload["error_message"] == "Chroma initialization failed in test"
    assert record is not None
    assert record["chunk_count"] > 0
    assert record["vector_indexed"] is False


def _upload_text_document(file_name: str, content: str) -> dict[str, Any]:
    response = client.post(
        "/api/documents/upload",
        files={"file": (file_name, content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["document_id"]
    assert payload["file_name"]
    return payload
