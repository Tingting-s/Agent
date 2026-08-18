from app.agents.validation import DocumentValidatorAgent
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE


def test_validator_agent_accepts_grounded_document_answer() -> None:
    result = DocumentValidatorAgent().validate_document_answer(
        answer="设备报警后应先查看报警代码并暂停设备。",
        retrieved_chunks=[{"chunk_id": "chunk-1", "text": "设备报警后应查看报警代码并暂停设备。", "score": 2.0}],
        task_type="document_qa",
    )

    assert result["is_valid"] is True
    assert result["issues"] == []
    assert result["confidence"] > 0


def test_validator_agent_rejects_document_answer_without_sources() -> None:
    result = DocumentValidatorAgent().validate_document_answer(
        answer=NO_RETRIEVAL_CONTENT_MESSAGE,
        retrieved_chunks=[],
        task_type="document_qa",
    )

    assert result["is_valid"] is False
    assert "no_retrieved_chunks" in result["issues"]


def test_validator_agent_rejects_broad_answer_that_repeats_sources() -> None:
    source_text = "设备启动前需要检查环境、电源、急停按钮和防护门。" * 20
    result = DocumentValidatorAgent().validate_document_answer(
        answer=source_text,
        retrieved_chunks=[{"chunk_id": "chunk-1", "text": source_text, "score": 2.0}],
        task_type="document_qa",
    )

    assert result["is_valid"] is False
    assert "answer_repeats_source_chunks" in result["issues"]
