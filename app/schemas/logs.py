from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import AnswerValidation, ApiResponseBase, DocumentChunk


class QALogEntry(BaseModel):
    log_id: str = Field(..., description="Log identifier.")
    request_id: str | None = Field(default=None, description="Request identifier.")
    task_type: str = Field(default="", description="Task type.")
    timestamp: str = Field(..., description="UTC timestamp.")
    document_id: str | None = Field(default=None, description="Document identifier.")
    file_name: str = Field(default="", description="Document file name.")
    file_path: str = Field(default="", description="Document path.")
    question: str = Field(..., description="Question.")
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list, description="Retrieved chunks.")
    answer: str = Field(default="", description="Answer.")
    validation_result: AnswerValidation = Field(..., description="Validation result.")
    status: str = Field(..., description="QA status.")
    elapsed_time: float = Field(default=0.0, description="Elapsed time in seconds.")
    error_message: str = Field(default="", description="Error message if any.")
    retrieval_source: str = Field(default="", description="Retrieval source: chroma, fallback, or none.")
    fallback_reason: str | None = Field(default=None, description="Reason for falling back or returning no retrieval.")
    source_chunks_count: int | None = Field(default=None, description="Number of source chunks used for the answer.")
    top_score: float | None = Field(default=None, description="Top retrieved chunk score.")
    top_distance: float | None = Field(default=None, description="Top retrieved chunk vector distance.")
    top_chunk_id: str | None = Field(default=None, description="Top retrieved chunk identifier.")
    rerank_enabled: bool | None = Field(default=None, description="Whether rerank was enabled.")
    rerank_top_n: int | None = Field(default=None, description="Number of chunks kept after rerank.")
    top_rerank_score: float | None = Field(default=None, description="Top rerank score.")
    retrieval_elapsed: float | None = Field(default=None, description="Retrieval elapsed time in seconds.")
    chroma_elapsed: float | None = Field(default=None, description="Chroma retrieval elapsed time in seconds.")
    rerank_elapsed: float | None = Field(default=None, description="Rerank elapsed time in seconds.")
    prompt_build_elapsed: float | None = Field(default=None, description="Prompt build elapsed time in seconds.")
    llm_elapsed: float | None = Field(default=None, description="LLM generation elapsed time in seconds.")
    validation_elapsed: float | None = Field(default=None, description="Answer validation elapsed time in seconds.")
    total_elapsed: float | None = Field(default=None, description="Total QA elapsed time in seconds.")
    log_record_elapsed: float | None = Field(default=None, description="Log record elapsed time in seconds.")
    embedding_model: str | None = Field(default=None, description="Embedding model used for retrieval.")
    llm_model: str | None = Field(default=None, description="LLM model used for answer generation.")
    llm_timeout: float | None = Field(default=None, description="Configured LLM timeout in seconds.")
    validation_status: str | None = Field(default=None, description="Validation status.")
    validation_message: str | None = Field(default=None, description="Validation message.")
    failed_stage: str | None = Field(default=None, description="Failed QA stage if any.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Log metadata.")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_entry(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("request_id", None)
        normalized.setdefault("task_type", "")
        normalized.setdefault("document_id", None)
        normalized.setdefault("file_name", "")
        normalized.setdefault("file_path", "")
        normalized.setdefault("question", "")
        normalized.setdefault("retrieved_chunks", [])
        normalized.setdefault("answer", "")
        normalized.setdefault(
            "validation_result",
            {"is_valid": False, "confidence": 0.0, "issues": [], "error_message": ""},
        )
        normalized.setdefault("status", "unknown")
        normalized.setdefault("elapsed_time", 0.0)
        normalized.setdefault("error_message", "")
        normalized.setdefault("retrieval_source", "")
        normalized.setdefault("metadata", {})
        return normalized


class QALogListResponse(BaseModel):
    items: list[QALogEntry] = Field(default_factory=list, description="Recent QA log entries.")


class LogListApiResponse(ApiResponseBase):
    items: list[QALogEntry] = Field(default_factory=list, description="Recent QA log entries.")
