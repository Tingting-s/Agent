from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ApiResponseBase, DocumentChunk


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Uploaded document identifier.")
    file_name: str = Field(..., description="Stored document file name.")
    file_path: str = Field(..., description="Stored document file path.")
    file_type: str = Field(..., description="Document extension without leading dot.")
    status: str = Field(..., description="Upload status.")
    message: str = Field(..., description="Upload result message.")
    chunk_count: int = Field(default=0, description="Parsed chunk count after upload indexing.")
    vector_indexed: bool = Field(default=False, description="Whether chunks were indexed after upload.")
    vector_error: str = Field(default="", description="Vector indexing error when upload succeeds but indexing fails.")


class UploadApiResponse(ApiResponseBase):
    document_id: str = Field(default="", description="Uploaded document identifier.")
    file_name: str = Field(default="", description="Stored document file name.")


class DocumentListItem(BaseModel):
    document_id: str = Field(..., description="Uploaded document identifier.")
    original_file_name: str = Field(..., description="Original user uploaded file name.")
    stored_file_name: str = Field(..., description="Stored file name on disk.")
    upload_time: str = Field(..., description="Upload timestamp.")
    status: str = Field(..., description="Document status.")
    chunk_count: int = Field(default=0, description="Current parsed chunk count.")
    vector_indexed: bool = Field(default=False, description="Whether chunks have been indexed into the vector store.")
    error_message: str = Field(default="", description="Last document indexing error message.")
    updated_time: str = Field(default="", description="Last metadata update timestamp.")


class DocumentListApiResponse(ApiResponseBase):
    items: list[DocumentListItem] = Field(default_factory=list, description="Persisted uploaded documents.")


class DocumentDeleteApiResponse(ApiResponseBase):
    document_id: str = Field(default="", description="Deleted document identifier.")
    file_deleted: bool = Field(default=False, description="Whether the stored file was deleted.")
    vector_deleted: bool = Field(default=False, description="Whether vector data deletion was requested successfully.")


class DocumentReindexApiResponse(ApiResponseBase):
    document_id: str = Field(default="", description="Reindexed document identifier.")
    chunk_count: int = Field(default=0, description="Parsed chunk count after reindexing.")
    vector_indexed: bool = Field(default=False, description="Whether chunks were indexed into the vector store.")


class DocumentIngestionTaskApiResponse(ApiResponseBase):
    task_id: str = Field(..., description="SQLite and Celery ingestion task identifier.")
    document_id: str = Field(..., description="Document associated with the ingestion task.")
    operation: str = Field(..., description="Ingestion operation: UPLOAD or REINDEX.")
    retry_count: int = Field(default=0, ge=0, description="Completed retry attempts.")
    chunk_count: int = Field(default=0, ge=0, description="Indexed chunk count.")
    vector_indexed: bool = Field(default=False, description="Whether vector indexing completed.")
    created_time: str = Field(default="", description="Task creation timestamp.")
    updated_time: str = Field(default="", description="Task update timestamp.")


class BatchDeleteDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list, description="Document identifiers to delete.")


class BatchDeleteDocumentResult(BaseModel):
    document_id: str = Field(..., description="Document identifier.")
    status: str = Field(..., description="Deletion status for this document.")
    message: str = Field(..., description="Deletion result message.")


class BatchDeleteDocumentsResponse(ApiResponseBase):
    deleted_count: int = Field(default=0, description="Number of documents deleted successfully.")
    failed_count: int = Field(default=0, description="Number of documents that failed to delete.")
    results: list[BatchDeleteDocumentResult] = Field(
        default_factory=list,
        description="Per-document deletion results.",
    )


class SummaryApiRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Uploaded document identifier.")


class SummaryApiResponse(ApiResponseBase):
    summary: str = Field(default="", description="Generated document summary.")


class DocumentReferenceRequest(BaseModel):
    document_id: str | None = Field(default=None, description="Document id returned by upload.")
    file_path: str | None = Field(default=None, description="Local document path.")

    @model_validator(mode="after")
    def require_document_reference(self) -> "DocumentReferenceRequest":
        if not self.document_id and not self.file_path:
            raise ValueError("document_id or file_path is required.")
        return self


class DocumentParseRequest(DocumentReferenceRequest):
    chunk_size: int = Field(default=800, ge=100, le=4000, description="Maximum characters per chunk.")
    chunk_overlap: int = Field(default=120, ge=0, le=1000, description="Characters repeated between chunks.")


class DocumentParseResponse(BaseModel):
    document_id: str | None = Field(default=None, description="Document identifier when available.")
    file_name: str = Field(..., description="Document file name.")
    file_path: str = Field(..., description="Document path.")
    file_type: str = Field(..., description="Document type.")
    status: str = Field(..., description="Parse status.")
    message: str = Field(..., description="Parse result message.")
    summary: str = Field(default="", description="Document summary.")
    key_points: list[str] = Field(default_factory=list, description="Summary key points.")
    risks: list[str] = Field(default_factory=list, description="Detected document risks.")
    chunks: list[DocumentChunk] = Field(default_factory=list, description="Document text chunks.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata.")
