from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_parse_document_endpoint_returns_chunks() -> None:
    upload_payload = _upload_legacy_document(
        "project-notes.txt",
        (DATA_DIR / "project_notes.txt").read_text(encoding="utf-8"),
    )

    response = client.post(
        "/documents/parse",
        json={"document_id": upload_payload["document_id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["chunks"]
    assert payload["summary"]


def test_ask_document_endpoint_returns_answer_and_log_id() -> None:
    upload_payload = _upload_legacy_document(
        "project-notes.txt",
        (DATA_DIR / "project_notes.txt").read_text(encoding="utf-8"),
    )

    response = client.post(
        "/documents/ask",
        json={
            "document_id": upload_payload["document_id"],
            "question": "What project phase is described?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["sources"]
    assert payload["validation"]["is_valid"] is True
    assert payload["log_id"]


def test_parse_document_rejects_direct_file_path_outside_uploads() -> None:
    response = client.post(
        "/documents/parse",
        json={"file_path": str(DATA_DIR / "project_notes.txt")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error_code"] == "http_exception"
    assert "uploads directory" in payload["message"]


def test_upload_document_endpoint_stores_file() -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"Enterprise policy document\nAccess reviews are required.", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["document_id"]
    assert payload["file_type"] == "txt"


def test_api_document_upload_summary_ask_and_logs() -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "handbook.txt",
                b"Company Handbook\nAccess reviews are required every quarter.\nOwners must approve exceptions.",
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    assert upload_payload["status"] == "success"
    assert upload_payload["request_id"]
    assert upload_payload["document_id"]
    assert upload_payload["error_message"] == ""

    summary_response = client.post(
        "/api/documents/summary",
        json={"document_id": upload_payload["document_id"]},
    )

    assert summary_response.status_code == 200
    summary_payload = summary_response.json()
    assert summary_payload["status"] == "success"
    assert summary_payload["request_id"]
    assert summary_payload["summary"]
    assert summary_payload["error_message"] == ""

    qa_response = client.post(
        "/api/qa/ask",
        json={
            "document_id": upload_payload["document_id"],
            "question": "How often are access reviews required?",
        },
    )

    assert qa_response.status_code == 200
    qa_payload = qa_response.json()
    assert qa_payload["status"] == "success"
    assert qa_payload["request_id"]
    assert qa_payload["answer"]
    assert qa_payload["source_chunks"]
    assert qa_payload["retrieval_source"] in {"chroma", "fallback"}
    assert qa_payload["source_chunks"][0]["retrieval_source"] == qa_payload["retrieval_source"]
    assert qa_payload["error_message"] == ""

    logs_response = client.get("/api/logs?limit=5")

    assert logs_response.status_code == 200
    logs_payload = logs_response.json()
    assert logs_payload["status"] == "success"
    assert logs_payload["request_id"]
    assert isinstance(logs_payload["items"], list)
    assert logs_payload["items"]
    log_item = logs_payload["items"][0]
    assert log_item["request_id"]
    assert log_item["task_type"] == "document_qa"
    assert log_item["document_id"] == upload_payload["document_id"]
    assert log_item["file_name"]
    assert log_item["question"]
    assert log_item["retrieved_chunks"]
    assert log_item["retrieval_source"] == qa_payload["retrieval_source"]
    assert log_item["answer"]
    assert log_item["validation_result"]
    assert log_item["status"] == "success"
    assert "timestamp" in log_item


def test_api_qa_ask_writes_jsonl_log_for_every_call() -> None:
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "audit.txt",
                b"Audit Policy\nAccess reviews are required every quarter.",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    qa_response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "What is required every quarter?"},
    )

    assert qa_response.status_code == 200
    qa_payload = qa_response.json()
    assert set(qa_payload) == {
        "request_id",
        "status",
        "message",
        "answer",
        "source_chunks",
        "retrieval_source",
        "elapsed_time",
        "error_message",
    }
    assert qa_payload["status"] == "success"

    logs_response = client.get("/api/logs?limit=1")
    log_item = logs_response.json()["items"][0]
    assert log_item["request_id"] == qa_payload["request_id"]
    assert log_item["task_type"] == "document_qa"
    assert log_item["document_id"] == document_id
    assert log_item["validation_result"]["is_valid"] is True


def test_api_document_summary_returns_error_for_unknown_document() -> None:
    response = client.post(
        "/api/documents/summary",
        json={"document_id": "missing-document"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["request_id"]
    assert payload["summary"] == ""
    assert payload["error_message"]


def _upload_legacy_document(file_name: str, content: str) -> dict[str, str]:
    response = client.post(
        "/documents/upload",
        files={"file": (file_name, content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    return payload
