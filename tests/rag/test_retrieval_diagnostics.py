from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE
from app.rag.ingestion import load_document
from app.rag.vector_store import get_vector_store


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
client = TestClient(app)


@pytest.fixture(autouse=True)
def local_answer_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class LocalLLM:
        def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, object]:
            if "异常处理原则" in question and "先保证人员安全" in context:
                return {
                    "answer": "设备异常处理原则是先保证人员安全、再保护设备、最后恢复生产；异常未确认前不得继续运行。",
                    "confidence": 0.8,
                    "source": "local_retrieval_diagnostic",
                }
            return {
                "answer": NO_RETRIEVAL_CONTENT_MESSAGE,
                "confidence": 0.0,
                "source": "local_retrieval_diagnostic",
            }

    monkeypatch.setattr("app.agents.answer_agent.get_qwen_llm_service", lambda: LocalLLM())


def test_docx_loader_extracts_table_rows(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "table_manual.docx"
    document = docx.Document()
    document.add_paragraph("设备表格手册")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "要求"
    table.rows[1].cells[0].text = "异常处理原则"
    table.rows[1].cells[1].text = "先保证人员安全、再保护设备、最后恢复生产"
    document.save(path)

    result = load_document(str(path))

    assert result["status"] == "success"
    assert "设备表格手册" in result["text"]
    assert "项目：异常处理原则" in result["text"]
    assert "要求：先保证人员安全、再保护设备、最后恢复生产" in result["text"]


def test_rag_device_manual_retrieves_abnormal_handling_principle_from_chroma() -> None:
    document_id = _upload_fixture("rag_device_manual.txt")

    documents = client.get("/api/documents").json()["items"]
    current = next(item for item in documents if item["document_id"] == document_id)
    assert current["chunk_count"] > 0
    assert current["vector_indexed"] is True

    raw_results = get_vector_store().search(
        document_id=document_id,
        question="异常处理原则是什么？",
        top_k=5,
    )
    assert len(raw_results) > 0

    response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "异常处理原则是什么？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["retrieval_source"] == "chroma"
    source_text = "\n".join(str(chunk.get("text") or "") for chunk in payload["source_chunks"])
    assert "先保证人员安全、再保护设备、最后恢复生产" in source_text


def test_rag_device_manual_rejects_purchase_price_without_unrelated_chunks() -> None:
    document_id = _upload_fixture("rag_device_manual.txt")

    response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": "设备采购价格是多少？"},
    )

    assert response.status_code == 200
    payload = response.json()
    answer = str(payload.get("answer") or "")
    assert payload["status"] == "error"
    assert NO_RETRIEVAL_CONTENT_MESSAGE in answer or NO_RETRIEVAL_CONTENT_MESSAGE in str(payload.get("message") or "")
    assert payload["source_chunks"] == []
    assert payload["retrieval_source"] == "none"
    assert "启动前检查" not in answer
    assert "报警处理" not in answer
    assert "日常维护" not in answer


def _upload_fixture(file_name: str) -> str:
    file_path = FIXTURES_DIR / file_name
    with file_path.open("rb") as file:
        response = client.post(
            "/api/documents/upload",
            files={"file": (file_name, file, "text/plain")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    return str(payload["document_id"])
