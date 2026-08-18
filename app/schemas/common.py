from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiResponseBase(BaseModel):
    request_id: str = Field(..., description="Request identifier.")
    status: str = Field(..., description="Business status.")
    message: str = Field(..., description="Human-readable message.")
    elapsed_time: float = Field(default=0.0, description="Elapsed time in seconds.")
    error_message: str = Field(default="", description="Error message when status is error.")


class DocumentChunk(BaseModel):
    document_id: str | None = Field(default=None, description="Document identifier.")
    file_name: str = Field(default="", description="Document file name.")
    chunk_id: str = Field(..., description="Chunk identifier.")
    text: str = Field(..., description="Chunk text.")
    start_index: int = Field(..., description="Start character index in the source document.")
    end_index: int = Field(..., description="End character index in the source document.")
    score: float = Field(default=0.0, description="Retrieval score when used as a source.")
    rerank_score: float = Field(default=0.0, description="Rerank score after candidate reordering.")
    retrieval_source: str = Field(default="", description="Retrieval source: chroma, fallback, or none.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata.")


class AnswerValidation(BaseModel):
    is_valid: bool = Field(..., description="Whether the answer is grounded in the document.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation confidence.")
    issues: list[str] = Field(default_factory=list, description="Validation issues.")
