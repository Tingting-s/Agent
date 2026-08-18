from pathlib import Path

from app.graphs.document_qa_graph import DocumentQAGraph
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE
from app.services.qa_log_service import QALogService


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_document_qa_graph_answers_from_document(tmp_path) -> None:
    graph = DocumentQAGraph(log_service=QALogService(tmp_path / "qa_logs.jsonl"))

    result = graph.invoke(
        {
            "file_path": str(DATA_DIR / "project_notes.txt"),
            "question": "What is the project phase?",
            "top_k": 2,
        }
    )

    response = result["response"]
    assert response["status"] == "success"
    assert response["answer"]
    assert response["sources"]
    assert response["validation"]["is_valid"] is True
    assert response["log_id"]


def test_document_qa_prompt_uses_only_final_retrieved_chunks(tmp_path) -> None:
    class FakeRetrievalAgent:
        def retrieve_with_source(self, *, question, chunks, top_k):  # type: ignore[no-untyped-def]
            return {
                "retrieval_source": "chroma",
                "chunks": [
                    {
                        "document_id": "doc-1",
                        "file_name": "prompt-check.txt",
                        "chunk_id": "selected-chunk",
                        "text": "Access reviews are required every quarter.",
                        "start_index": 0,
                        "end_index": 44,
                        "score": 1.0,
                        "rerank_score": 3.0,
                        "retrieval_source": "chroma",
                        "metadata": {"retrieval_source": "chroma"},
                    }
                ],
            }

    file_path = tmp_path / "prompt-check.txt"
    file_path.write_text(
        "Access reviews are required every quarter.\n"
        "FULL_TEXT_MARKER_UNRELATED should not enter the QA prompt.",
        encoding="utf-8",
    )
    graph = DocumentQAGraph(
        retrieval_agent=FakeRetrievalAgent(),  # type: ignore[arg-type]
        log_service=QALogService(tmp_path / "qa_logs.jsonl"),
    )

    result = graph.invoke(
        {
            "document_id": "doc-1",
            "file_path": str(file_path),
            "question": "How often are access reviews required?",
            "task_type": "document_qa",
        }
    )

    assert "selected-chunk" in result["prompt"]
    assert "FULL_TEXT_MARKER_UNRELATED" not in result["prompt"]
    assert len(result["retrieved_chunks"]) == 1


def test_document_qa_graph_returns_no_answer_without_sources(tmp_path) -> None:
    class NoAnswerRetrievalAgent:
        def retrieve_with_source(self, *, question, chunks, top_k):  # type: ignore[no-untyped-def]
            return {
                "retrieval_source": "none",
                "fallback_reason": "below_relevance_threshold",
                "chunks": [],
            }

    file_path = tmp_path / "device.txt"
    file_path.write_text(
        "设备启动前需要进行外观检查，确认管路接口正常。运行维护和停机流程另行执行。",
        encoding="utf-8",
    )
    graph = DocumentQAGraph(
        retrieval_agent=NoAnswerRetrievalAgent(),  # type: ignore[arg-type]
        log_service=QALogService(tmp_path / "qa_logs.jsonl"),
    )

    result = graph.invoke(
        {
            "document_id": "doc-1",
            "file_path": str(file_path),
            "question": "设备采购价格是多少？",
            "task_type": "document_qa",
        }
    )
    response = result["response"]

    assert response["status"] == "error"
    assert response["answer"] == NO_RETRIEVAL_CONTENT_MESSAGE
    assert response["sources"] == []
    assert response["metadata"]["retrieval_source"] == "none"
    assert response["validation"]["is_valid"] is False
    assert "no_retrieved_chunks" in response["validation"]["issues"]
