from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE


client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _uploads_dir() -> Path:
    return Path(os.environ["UPLOAD_DIR"])


def _log_path() -> Path:
    return Path(os.environ["LOG_DIR"]) / "qa_logs.jsonl"


def _upload_text_document(file_name: str, content: str) -> dict[str, object]:
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


def test_document_upload_stores_file_and_returns_document_id() -> None:
    payload = _upload_text_document(
        "security_policy.txt",
        "Security Policy\nAccess reviews are required every quarter.",
    )

    stored_file = _uploads_dir() / str(payload["file_name"])
    assert stored_file.exists()
    assert str(payload["document_id"]) in stored_file.name


def test_document_qa_returns_answer_sources_and_writes_jsonl_log() -> None:
    upload_payload = _upload_text_document(
        "security_policy.txt",
        "Security Policy\nAccess reviews are required every quarter.\nOwners approve exceptions.",
    )

    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": upload_payload["document_id"],
            "question": "How often are access reviews required?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["request_id"]
    assert payload["answer"]
    assert payload["source_chunks"]
    assert payload["retrieval_source"] in {"chroma", "fallback"}
    assert payload["elapsed_time"] >= 0
    assert payload["error_message"] == ""

    assert _log_path().exists()
    log_lines = [line for line in _log_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    assert log_lines
    log_entry = json.loads(log_lines[-1])
    assert log_entry["request_id"] == payload["request_id"]
    assert log_entry["task_type"] == "document_qa"
    assert log_entry["document_id"] == upload_payload["document_id"]
    assert log_entry["file_name"]
    assert log_entry["question"] == "How often are access reviews required?"
    assert log_entry["retrieved_chunks"]
    assert log_entry["retrieval_source"] == payload["retrieval_source"]
    assert log_entry["answer"] == payload["answer"]
    assert log_entry["validation_result"]["is_valid"] is True
    assert log_entry["status"] == "success"
    assert log_entry["elapsed_time"] >= 0
    assert log_entry["error_message"] == ""
    assert log_entry["timestamp"]


def test_document_qa_returns_no_answer_message_when_no_relevant_chunk() -> None:
    upload_payload = _upload_text_document(
        "lunch_menu.txt",
        "Lunch Menu\nVegetarian noodles are served on Wednesday.",
    )

    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": upload_payload["document_id"],
            "question": "What is the quantum encryption budget?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["answer"] == NO_RETRIEVAL_CONTENT_MESSAGE
    assert payload["source_chunks"] == []
    assert payload["retrieval_source"] == "none"
    assert payload["error_message"]


def test_document_qa_returns_clear_error_for_missing_document() -> None:
    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": "missing-document",
            "question": "What does this document say?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["request_id"]
    assert payload["answer"] == ""
    assert payload["source_chunks"] == []
    assert "Traceback" not in payload["error_message"]
    assert payload["error_message"]


def test_upload_rejects_unsupported_file_with_clear_error() -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("payload.json", b'{"hello": "world"}', "application/json")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["document_id"] == ""
    assert "Traceback" not in payload["error_message"]
    assert "supported" in payload["error_message"]


def test_log_write_failure_does_not_break_qa_response(monkeypatch) -> None:
    from app.api import routes

    upload_payload = _upload_text_document(
        "resilience_policy.txt",
        "Resilience Policy\nBackups are tested every month.",
    )

    def raise_log_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk unavailable")

    monkeypatch.setattr(routes.document_service.graph.log_agent, "record", raise_log_error)

    response = client.post(
        "/api/qa/ask",
        json={
            "document_id": upload_payload["document_id"],
            "question": "How often are backups tested?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["error_message"] == ""


def test_streamlit_qa_request_timeout_message(monkeypatch) -> None:
    from web import streamlit_app

    def timeout_request(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(streamlit_app.requests, "request", timeout_request)

    payload, error = streamlit_app.ask_question("http://testserver", "doc-1", "What is required?")

    assert payload is None
    assert error == "问答请求超时，请检查后端日志或稍后重试。"
