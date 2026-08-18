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
from app.services.qa_log_service import QALogService


client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
QA_TEST_CASES_PATH = FIXTURES_DIR / "qa_test_cases.json"
FIXTURE_DOCUMENTS = [
    "sample_device_manual.txt",
    "sample_training_material.md",
    "sample_faq.txt",
]
QA_RESPONSE_REQUIRED_FIELDS = {
    "request_id",
    "status",
    "answer",
    "source_chunks",
    "retrieval_source",
    "elapsed_time",
    "error_message",
}


@pytest.fixture()
def isolated_document_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DocumentQAService:
    log_service = QALogService(log_path=tmp_path / "logs" / "qa_logs.jsonl")
    service = DocumentQAService(storage_dir=tmp_path / "uploads", log_service=log_service)
    monkeypatch.setattr(routes, "document_service", service)
    return service


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"]
    assert payload["environment"]


def test_qa_fixture_files_are_complete() -> None:
    device_manual = (FIXTURES_DIR / "sample_device_manual.txt").read_text(encoding="utf-8")
    training_material = (FIXTURES_DIR / "sample_training_material.md").read_text(encoding="utf-8")
    faq = (FIXTURES_DIR / "sample_faq.txt").read_text(encoding="utf-8")
    cases = _load_qa_cases()

    assert len(device_manual) >= 800
    assert "设备启动前检查" in device_manual
    assert "设备报警处理步骤" in device_manual
    assert "日常维护要求" in device_manual

    assert len(training_material) >= 800
    assert "培训目标" in training_material
    assert "培训对象" in training_material
    assert "考核标准" in training_material

    assert faq.count("Q") >= 10
    assert faq.count("A") >= 10

    assert len(cases) >= 20
    for case in cases:
        assert set(case) == {
            "document_file",
            "question",
            "expected_type",
            "expected_keywords",
            "description",
        }
        assert case["document_file"] in FIXTURE_DOCUMENTS
        assert case["expected_type"] in {"answerable", "unanswerable"}
        assert isinstance(case["expected_keywords"], list)
        assert case["expected_keywords"]


def test_api_documents_upload_accepts_all_fixture_documents(
    isolated_document_service: DocumentQAService,
) -> None:
    uploaded_documents = _upload_all_fixture_documents()

    assert set(uploaded_documents) == set(FIXTURE_DOCUMENTS)
    for document_file, payload in uploaded_documents.items():
        assert payload["status"] == "success"
        assert payload["request_id"]
        assert payload["document_id"]
        assert payload["file_name"].endswith(document_file)
        assert payload["error_message"] == ""
        assert (isolated_document_service.storage_dir / payload["file_name"]).exists()


def test_api_qa_answerable_cases_return_answers_and_source_chunks(
    isolated_document_service: DocumentQAService,
) -> None:
    uploaded_documents = _upload_all_fixture_documents()
    answerable_cases = [case for case in _load_qa_cases() if case["expected_type"] == "answerable"]

    assert answerable_cases
    for case in answerable_cases:
        response = client.post(
            "/api/qa/ask",
            json={
                "document_id": uploaded_documents[case["document_file"]]["document_id"],
                "question": case["question"],
            },
        )

        assert response.status_code == 200, case["description"]
        payload = response.json()
        _assert_qa_response_fields(payload)
        assert payload["status"] == "success", case["description"]
        assert payload["answer"], case["description"]
        assert payload["source_chunks"], case["description"]
        assert payload["retrieval_source"] in {"chroma", "fallback"}
        assert payload["source_chunks"][0]["retrieval_source"] == payload["retrieval_source"]
        assert payload["elapsed_time"] >= 0
        assert payload["error_message"] == ""

        combined_text = _response_text_with_sources(payload)
        assert any(keyword in combined_text for keyword in case["expected_keywords"]), case["description"]


def test_api_qa_unanswerable_cases_return_no_answer_prompt(
    isolated_document_service: DocumentQAService,
) -> None:
    uploaded_documents = _upload_all_fixture_documents()
    unanswerable_cases = [case for case in _load_qa_cases() if case["expected_type"] == "unanswerable"]

    assert unanswerable_cases
    for case in unanswerable_cases:
        response = client.post(
            "/api/qa/ask",
            json={
                "document_id": uploaded_documents[case["document_file"]]["document_id"],
                "question": case["question"],
            },
        )

        assert response.status_code == 200, case["description"]
        payload = response.json()
        _assert_qa_response_fields(payload)
        assert payload["status"] == "error", case["description"]
        assert payload["source_chunks"] == []
        assert payload["retrieval_source"] == "none"
        assert (
            NO_RETRIEVAL_CONTENT_MESSAGE in payload["answer"]
            or "未检索到相关内容" in payload["message"]
            or "no_retrieved_chunks" in payload["error_message"]
        )


def test_api_qa_empty_question_returns_clear_validation_error(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_fixture_document("sample_faq.txt")

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": ""},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error_code"] == "request_validation_error"
    assert payload["message"] == "Request validation failed."
    assert payload["details"]["errors"]


def test_api_qa_invalid_document_id_returns_clear_error(
    isolated_document_service: DocumentQAService,
) -> None:
    response = client.post(
        "/api/qa/ask",
        json={"document_id": "invalid-document-id", "question": "设备启动前需要检查什么？"},
    )

    assert response.status_code == 200
    payload = response.json()
    _assert_qa_response_fields(payload)
    assert payload["status"] == "error"
    assert payload["answer"] == ""
    assert payload["source_chunks"] == []
    assert payload["error_message"]
    assert "Traceback" not in payload["error_message"]
    assert "\n" not in payload["error_message"]


def test_api_qa_writes_jsonl_log_file(
    isolated_document_service: DocumentQAService,
) -> None:
    upload_payload = _upload_fixture_document("sample_faq.txt")

    response = client.post(
        "/api/qa/ask",
        json={"document_id": upload_payload["document_id"], "question": "账号无法登录时如何处理？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"

    log_path = isolated_document_service.log_service.log_path
    assert log_path.parent.name == "logs"
    assert log_path.name == "qa_logs.jsonl"
    assert log_path.exists()

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries
    log_entry = next(entry for entry in entries if entry["request_id"] == payload["request_id"])
    assert log_entry["task_type"] == "document_qa"
    assert log_entry["document_id"] == upload_payload["document_id"]
    assert log_entry["question"] == "账号无法登录时如何处理？"
    assert log_entry["retrieved_chunks"]
    assert log_entry["retrieval_source"] == payload["retrieval_source"]
    assert log_entry["answer"] == payload["answer"]
    assert log_entry["status"] == "success"
    assert log_entry["elapsed_time"] >= 0
    assert log_entry["timestamp"]


def _load_qa_cases() -> list[dict[str, Any]]:
    return json.loads(QA_TEST_CASES_PATH.read_text(encoding="utf-8"))


def _upload_all_fixture_documents() -> dict[str, dict[str, Any]]:
    return {document_file: _upload_fixture_document(document_file) for document_file in FIXTURE_DOCUMENTS}


def _upload_fixture_document(document_file: str) -> dict[str, Any]:
    file_path = FIXTURES_DIR / document_file
    media_type = "text/markdown" if file_path.suffix == ".md" else "text/plain"
    with file_path.open("rb") as file:
        response = client.post(
            "/api/documents/upload",
            files={"file": (document_file, file, media_type)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    return payload


def _assert_qa_response_fields(payload: dict[str, Any]) -> None:
    assert QA_RESPONSE_REQUIRED_FIELDS.issubset(payload)
    assert payload["request_id"]
    assert isinstance(payload["source_chunks"], list)
    assert isinstance(payload["elapsed_time"], int | float)


def _response_text_with_sources(payload: dict[str, Any]) -> str:
    source_text = "\n".join(str(chunk.get("text") or "") for chunk in payload["source_chunks"])
    return f"{payload['answer']}\n{source_text}"
