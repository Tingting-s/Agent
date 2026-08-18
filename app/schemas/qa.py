from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import AnswerValidation, ApiResponseBase, DocumentChunk
from app.schemas.documents import DocumentParseRequest


class AskApiRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Uploaded document identifier.")
    question: str = Field(..., min_length=1, description="Question to answer from the document.")


class DocumentQARequest(DocumentParseRequest):
    question: str = Field(..., min_length=1, description="Question to answer from the document.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks used for answering.")


class DocumentQAResponse(BaseModel):
    document_id: str | None = Field(default=None, description="Document identifier when available.")
    file_name: str = Field(default="", description="Document file name.")
    file_path: str = Field(default="", description="Document path.")
    status: str = Field(..., description="QA status.")
    message: str = Field(..., description="QA result message.")
    question: str = Field(..., description="Original user question.")
    answer: str = Field(default="", description="Grounded answer.")
    summary: str = Field(default="", description="Document summary.")
    sources: list[DocumentChunk] = Field(default_factory=list, description="Source chunks used for the answer.")
    validation: AnswerValidation = Field(..., description="Answer validation result.")
    log_id: str | None = Field(default=None, description="QA log identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata.")


class AskApiResponse(ApiResponseBase):
    answer: str = Field(default="", description="Generated answer.")
    source_chunks: list[DocumentChunk] = Field(default_factory=list, description="Source chunks used by the answer.")
    retrieval_source: str = Field(default="", description="Retrieval source: chroma, fallback, or none.")
