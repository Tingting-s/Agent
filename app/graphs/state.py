from __future__ import annotations

from typing import Any, TypedDict


class DocumentQAState(TypedDict, total=False):
    request_id: str
    task_type: str
    document_id: str | None
    file_path: str
    file_name: str
    question: str
    parsed_text: str
    chunks: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    prompt: str
    answer: str
    source_chunks: list[dict[str, Any]]
    validation_result: dict[str, Any]
    status: str
    error_message: str
    elapsed_time: float
    retry_count: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    summary: str
    key_points: list[str]
    risks: list[str]
    log_id: str | None
    start_time: float
    metadata: dict[str, Any]
    response: dict[str, Any]
