from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
client = TestClient(app)


@pytest.fixture(autouse=True)
def local_baseline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class LocalBaselineLLM:
        def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, object]:
            if "异常处理原则" in question and "先保证人员安全" in context:
                answer = "设备异常处理原则是先保证人员安全、再保护设备、最后恢复生产；异常未确认前不得继续运行。"
            elif "启动前" in question and "环境检查" in context:
                answer = "设备启动前需要完成环境检查、外观检查、电源检查、介质检查和安全状态检查。"
            elif "报警" in question and "报警信息" in context:
                answer = "设备报警后应确认报警信息，查看报警代码，必要时降低负载，通知维护人员，并在需要时停机。"
            elif "停机流程" in question and "停止进料" in context:
                answer = "设备停机流程包括停止进料或任务，等待当前周期完成，按停机按钮，关闭辅助系统，切断主电源，并记录停机状态。"
            else:
                answer = NO_RETRIEVAL_CONTENT_MESSAGE
            return {"answer": answer, "confidence": 0.8, "source": "local_baseline_mock"}

    monkeypatch.setattr("app.agents.answer_agent.get_qwen_llm_service", lambda: LocalBaselineLLM())


@pytest.fixture()
def uploaded_device_manual() -> dict[str, Any]:
    file_path = FIXTURES_DIR / "rag_device_manual.txt"
    with file_path.open("rb") as file:
        response = client.post(
            "/api/documents/upload",
            files={"file": (file_path.name, file, "text/plain")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    documents = list_response.json()["items"]
    current = next(item for item in documents if item["document_id"] == payload["document_id"])
    assert current["chunk_count"] > 0
    assert current["vector_indexed"] is True
    return payload


@pytest.mark.parametrize(
    ("question", "expected_keywords"),
    [
        (
            "异常处理原则是什么？",
            ["先保证人员安全", "再保护设备", "最后恢复生产"],
        ),
        (
            "设备启动前需要检查什么？",
            ["环境检查", "外观检查", "电源检查", "安全状态检查"],
        ),
        (
            "设备报警后应该如何处理？",
            ["报警信息", "报警代码", "通知维护人员", "必要时停机"],
        ),
        (
            "设备停机流程是什么？",
            ["停止进料", "按停机按钮", "切断主电源", "记录停机状态"],
        ),
    ],
)
def test_rag_device_manual_answerable_questions_retrieve_chroma_sources(
    uploaded_device_manual: dict[str, Any],
    question: str,
    expected_keywords: list[str],
) -> None:
    response = client.post(
        "/api/qa/ask",
        json={"document_id": uploaded_device_manual["document_id"], "question": question},
    )

    assert response.status_code == 200
    payload = response.json()
    source_chunks = list(payload.get("source_chunks") or [])
    source_text = "\n".join(str(chunk.get("text") or "") for chunk in source_chunks)

    assert payload["status"] == "success"
    assert payload["retrieval_source"] == "chroma"
    assert 1 <= len(source_chunks) <= 3
    assert all(chunk.get("retrieval_source") == "chroma" for chunk in source_chunks)
    assert all(float(chunk.get("rerank_score") or 0.0) > 0.0 for chunk in source_chunks)
    assert _is_descending([float(chunk.get("rerank_score") or 0.0) for chunk in source_chunks])
    for keyword in expected_keywords:
        assert keyword in source_text


def test_rag_device_manual_unanswerable_question_is_rejected(uploaded_device_manual: dict[str, Any]) -> None:
    response = client.post(
        "/api/qa/ask",
        json={"document_id": uploaded_device_manual["document_id"], "question": "设备采购价格是多少？"},
    )

    assert response.status_code == 200
    payload = response.json()
    answer = str(payload.get("answer") or "")

    assert payload["status"] == "error"
    assert payload["retrieval_source"] == "none"
    assert payload["source_chunks"] == []
    assert NO_RETRIEVAL_CONTENT_MESSAGE in answer or NO_RETRIEVAL_CONTENT_MESSAGE in str(payload.get("message") or "")
    for forbidden in ("启动前检查", "报警处理", "日常维护", "停机流程"):
        assert forbidden not in answer


def _is_descending(values: list[float]) -> bool:
    return all(left >= right for left, right in zip(values, values[1:]))
