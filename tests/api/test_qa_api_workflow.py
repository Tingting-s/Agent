from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE
from app.services.document_qa_service import DocumentQAService
from app.services.llm_service import LLMRequestError, LLMTimeoutError, LLM_TIMEOUT_MESSAGE
from app.services.qa_log_service import QALogService


client = TestClient(app)

QA_RESPONSE_KEYS = {
    "request_id",
    "status",
    "message",
    "answer",
    "source_chunks",
    "retrieval_source",
    "elapsed_time",
    "error_message",
}

LOG_REQUIRED_KEYS = {
    "request_id",
    "task_type",
    "document_id",
    "file_name",
    "question",
    "retrieved_chunks",
    "answer",
    "validation_result",
    "status",
    "elapsed_time",
    "error_message",
    "retrieval_source",
    "timestamp",
}


@pytest.fixture()
def isolated_document_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentQAService:
    log_service = QALogService(log_path=tmp_path / "logs" / "qa_logs.jsonl")
    service = DocumentQAService(storage_dir=tmp_path / "uploads", log_service=log_service)
    monkeypatch.setattr(routes, "document_service", service)
    return service


def test_api_upload_document_stores_file_with_document_id(
    isolated_document_service: DocumentQAService,
) -> None:
    payload = _upload_text_document(
        "access-policy.txt",
        "Enterprise Access Policy\nAccess reviews are required every quarter.",
    )

    stored_file = isolated_document_service.storage_dir / str(payload["file_name"])
    assert stored_file.exists()
    assert str(payload["document_id"]) in stored_file.name
    assert "Access reviews" in stored_file.read_text(encoding="utf-8")


def test_api_qa_ask_answers_from_uploaded_document_and_writes_jsonl_log(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document(
        "security-handbook.txt",
        "Security Handbook\nAccess reviews are required every quarter. Owners approve exceptions.",
    )
    document_id = str(upload_payload["document_id"])

    qa_response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "How often are access reviews required?"},
    )

    assert qa_response.status_code == 200
    payload = qa_response.json()
    assert set(payload) == QA_RESPONSE_KEYS
    assert payload["status"] == "success"
    assert payload["request_id"]
    assert "quarter" in payload["answer"].lower()
    assert payload["source_chunks"]
    assert payload["retrieval_source"] in {"chroma", "fallback"}
    assert payload["source_chunks"][0]["retrieval_source"] == payload["retrieval_source"]
    assert payload["error_message"] == ""

    log_entry = _find_log_entry(isolated_document_service.log_service.log_path, payload["request_id"])
    assert LOG_REQUIRED_KEYS.issubset(log_entry)
    assert log_entry["task_type"] == "document_qa"
    assert log_entry["document_id"] == document_id
    assert log_entry["question"] == "How often are access reviews required?"
    assert log_entry["retrieved_chunks"]
    assert log_entry["retrieval_source"] == payload["retrieval_source"]
    assert log_entry["metadata"]["retrieval_source"] == payload["retrieval_source"]
    assert log_entry["request_id"] == payload["request_id"]
    assert log_entry["answer"] == payload["answer"]
    assert log_entry["status"] == "success"
    assert log_entry["validation_result"]["is_valid"] is True
    assert log_entry["source_chunks_count"] == len(log_entry["retrieved_chunks"])
    assert log_entry["total_elapsed"] >= 0
    assert log_entry["retrieval_elapsed"] is not None
    assert log_entry["validation_elapsed"] is not None
    assert log_entry["llm_model"]
    assert log_entry["embedding_model"]
    assert log_entry["validation_status"] == "success"


def test_api_qa_ask_returns_no_answer_prompt_and_logs_error_status(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document(
        "cafeteria.txt",
        "Cafeteria Menu\nSoup, salad, coffee, and tea are available at lunch.",
    )
    document_id = str(upload_payload["document_id"])

    qa_response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "What is the quantum encryption budget?"},
    )

    assert qa_response.status_code == 200
    payload = qa_response.json()
    assert set(payload) == QA_RESPONSE_KEYS
    assert payload["status"] == "error"
    assert payload["answer"] == NO_RETRIEVAL_CONTENT_MESSAGE
    assert payload["source_chunks"] == []
    assert payload["retrieval_source"] == "none"
    assert payload["error_message"] == "no_retrieved_chunks"

    log_entry = _find_log_entry(isolated_document_service.log_service.log_path, payload["request_id"])
    assert log_entry["status"] == "error"
    assert log_entry["retrieved_chunks"] == []
    assert log_entry["retrieval_source"] == "none"
    assert log_entry["metadata"]["retrieval_source"] == "none"
    assert log_entry["answer"] == NO_RETRIEVAL_CONTENT_MESSAGE
    assert "no_retrieved_chunks" in log_entry["validation_result"]["issues"]


def test_api_qa_ask_returns_public_error_for_missing_document(
    isolated_document_service: DocumentQAService,
) -> None:
    response = client.post(
        "/api/qa/ask",
        json={"document_id": "missing-document", "question": "What is the policy?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == QA_RESPONSE_KEYS
    assert payload["status"] == "error"
    assert payload["answer"] == ""
    assert payload["source_chunks"] == []
    assert payload["error_message"]
    assert "Traceback" not in payload["error_message"]
    assert "\n" not in payload["error_message"]
    assert not Path(isolated_document_service.storage_dir / "missing-document").exists()


def test_api_upload_document_returns_public_error_for_empty_file(
    isolated_document_service: DocumentQAService,
) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["document_id"] == ""
    assert payload["file_name"] == "empty.txt"
    assert payload["error_message"] == "Uploaded document is empty."
    assert "Traceback" not in payload["error_message"]


def test_qa_log_write_failure_does_not_break_successful_answer(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_payload = _upload_text_document(
        "operations.txt",
        "Operations Guide\nIncident reviews must be completed within two business days.",
    )
    document_id = str(upload_payload["document_id"])

    def fail_to_record(*, state: dict[str, Any]) -> dict[str, Any]:
        raise OSError("disk full")

    monkeypatch.setattr(isolated_document_service.graph.log_agent, "record", fail_to_record)

    response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "When must incident reviews be completed?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == QA_RESPONSE_KEYS
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["source_chunks"]
    assert payload["error_message"] == ""


def test_api_qa_narrow_question_does_not_return_unrelated_sections(
    isolated_document_service: DocumentQAService,
) -> None:
    document_text = (
        "Startup checks require power supply, emergency stop, guard door, and site environment.\n"
        "Shutdown procedure requires stopping feed, low-speed idle, stop button, and main power off.\n"
        "Daily maintenance requires cleaning filters and checking lubrication points."
    )
    upload_payload = _upload_text_document("device-narrow.txt", document_text)

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "What is required for startup checks?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["source_chunks"]) <= 3
    assert all(len(chunk["text"]) <= 600 for chunk in payload["source_chunks"])
    answer = payload["answer"].lower()
    assert "power" in answer
    assert "emergency" in answer


def test_api_qa_uses_chroma_for_uploaded_document(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document(
        "device-chroma.txt",
        "Startup checks require power supply, emergency stop, guard door, and site environment.\n"
        "During operation, record temperature, pressure, current, and speed.",
    )

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "What is required for startup checks?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["retrieval_source"] == "chroma"
    assert payload["source_chunks"]
    assert payload["source_chunks"][0]["text"]
    assert payload["source_chunks"][0]["retrieval_source"] == "chroma"


def test_api_logs_reads_legacy_entries(
    isolated_document_service: DocumentQAService,
) -> None:
    legacy_entry = {
        "log_id": "legacy-log-1",
        "request_id": "legacy-request-1",
        "task_type": "document_qa",
        "timestamp": "2026-05-18T00:00:00+00:00",
        "document_id": "legacy-document",
        "file_name": "legacy.txt",
        "file_path": "uploads/legacy.txt",
        "question": "Legacy question?",
        "retrieved_chunks": [],
        "answer": "Legacy answer.",
        "validation_result": {"is_valid": True, "confidence": 0.8, "issues": []},
        "status": "success",
        "elapsed_time": 0.12,
        "error_message": "",
        "metadata": {"retrieval_source": "fallback"},
    }
    log_path = isolated_document_service.log_service.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(legacy_entry, ensure_ascii=False) + "\n", encoding="utf-8")

    response = client.get("/api/logs?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    item = payload["items"][0]
    assert item["request_id"] == "legacy-request-1"
    assert item["retrieval_source"] == "fallback"
    assert item["total_elapsed"] == 0.12
    assert item["source_chunks_count"] == 0


def test_api_qa_unanswerable_question_does_not_emit_document_content(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_text_document(
        "device-no-price.txt",
        "Startup checks require power supply and guard door checks.\n"
        "During operation, record temperature and pressure.",
    )

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "What is the equipment purchase price?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["answer"] == NO_RETRIEVAL_CONTENT_MESSAGE
    assert payload["source_chunks"] == []
    assert payload["retrieval_source"] == "none"
    assert "temperature" not in payload["answer"].lower()


def test_api_qa_returns_timeout_when_llm_times_out(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutLLM:
        def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, object]:
            raise LLMTimeoutError(LLM_TIMEOUT_MESSAGE)

    monkeypatch.setattr("app.agents.answer_agent.get_qwen_llm_service", lambda: TimeoutLLM())

    upload_payload = _upload_text_document(
        "device-timeout.txt",
        "Startup checks require power supply, emergency stop, guard door, and site environment.",
    )

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "What is required for startup checks?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "timeout"
    assert payload["answer"] == LLM_TIMEOUT_MESSAGE
    assert payload["error_message"] == LLM_TIMEOUT_MESSAGE
    assert payload["retrieval_source"] == "chroma"
    assert payload["source_chunks"]

    log_entry = _find_log_entry(isolated_document_service.log_service.log_path, payload["request_id"])
    assert log_entry["status"] == "timeout"
    assert log_entry["error_message"] == LLM_TIMEOUT_MESSAGE
    assert log_entry["failed_stage"] == "llm_generate"
    assert log_entry["total_elapsed"] >= 0
    assert log_entry["retrieval_source"] == "chroma"


def test_api_qa_uses_fallback_when_llm_fails(
    isolated_document_service: DocumentQAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLLM:
        def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, object]:
            raise LLMRequestError("model gateway unavailable")

    monkeypatch.setattr("app.agents.answer_agent.get_qwen_llm_service", lambda: FailingLLM())

    upload_payload = _upload_text_document(
        "device-llm-failure.txt",
        "Startup checks require power supply, emergency stop, guard door, and site environment.",
    )

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "What is required for startup checks?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["error_message"] == ""
    assert payload["answer"]
    assert "power" in payload["answer"].lower()


def _upload_text_document(file_name: str, content: str) -> dict[str, Any]:
    response = client.post(
        "/api/documents/upload",
        files={"file": (file_name, content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["request_id"]
    assert payload["document_id"]
    assert payload["file_name"]
    assert payload["error_message"] == ""
    return payload


def _find_log_entry(log_path: Path, request_id: str) -> dict[str, Any]:
    entries = _read_jsonl(log_path)
    for entry in entries:
        if entry.get("request_id") == request_id:
            return entry
    raise AssertionError(f"Log entry not found for request_id={request_id}")


def _read_jsonl(log_path: Path) -> list[dict[str, Any]]:
    assert log_path.exists()
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
