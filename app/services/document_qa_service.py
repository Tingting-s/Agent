from __future__ import annotations

import logging
import re
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import UploadFile

from app.config import get_upload_dir
from app.graphs.document_qa_graph import DocumentQAGraph
from app.rag.generation import summarize_document_text
from app.rag.ingestion import SUPPORTED_DOCUMENT_EXTENSIONS, load_document, split_text
from app.schemas.documents import (
    BatchDeleteDocumentResult,
    BatchDeleteDocumentsResponse,
    DocumentDeleteApiResponse,
    DocumentListApiResponse,
    DocumentListItem,
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentReindexApiResponse,
    DocumentUploadResponse,
    SummaryApiResponse,
    UploadApiResponse,
)
from app.schemas.logs import LogListApiResponse, QALogListResponse
from app.schemas.qa import AskApiResponse, DocumentQARequest, DocumentQAResponse
from app.rag.vector_store import get_vector_store
from app.services.document_storage_service import DocumentStorageService
from app.services.qa_log_service import QALogService


SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS
LOGGER = logging.getLogger(__name__)


class DocumentQAService:
    def __init__(
        self,
        *,
        storage_dir: str | Path | None = None,
        graph: DocumentQAGraph | None = None,
        log_service: QALogService | None = None,
        storage_service: DocumentStorageService | None = None,
    ) -> None:
        using_default_storage_dir = storage_dir is None
        self.storage_dir = Path(storage_dir or get_upload_dir())
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_service = log_service or QALogService()
        self.graph = graph or DocumentQAGraph(log_service=self.log_service)
        self.storage_service = storage_service or DocumentStorageService(
            None if using_default_storage_dir else self.storage_dir.parent / "data" / "documents.db"
        )

    async def upload_document(self, upload: UploadFile) -> DocumentUploadResponse:
        original_name = Path(upload.filename or "document.txt").name.strip() or "document.txt"
        safe_name = self._sanitize_filename(original_name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only .txt, .md, .docx, and text-based .pdf files are supported.")

        content = await upload.read()
        if not content:
            raise ValueError("Uploaded document is empty.")

        document_id = str(uuid4())
        stored_name = f"{document_id}_{safe_name}"
        file_path = self.storage_dir / stored_name
        file_path.write_bytes(content)
        self.storage_service.upsert_document(
            document_id=document_id,
            original_file_name=original_name,
            stored_file_name=stored_name,
            file_path=str(file_path),
            file_type=suffix.lstrip("."),
            status="success",
            chunk_count=0,
            vector_indexed=False,
        )
        chunk_count, vector_indexed, vector_error = self._index_uploaded_document(
            document_id=document_id,
            file_path=file_path,
        )
        self.storage_service.update_index_status(
            document_id,
            chunk_count=chunk_count,
            vector_indexed=vector_indexed,
            status="success",
            error_message=vector_error,
        )
        message = "Document uploaded successfully."
        if vector_error:
            message = "Document uploaded successfully. Vector indexing failed."
        return DocumentUploadResponse(
            document_id=document_id,
            file_name=stored_name,
            file_path=str(file_path),
            file_type=suffix.lstrip("."),
            status="success",
            message=message,
            chunk_count=chunk_count,
            vector_indexed=vector_indexed,
            vector_error=vector_error,
        )

    async def upload_document_api(self, upload: UploadFile) -> UploadApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            result = await self.upload_document(upload)
            return UploadApiResponse(
                request_id=request_id,
                status=result.status,
                message=result.message,
                elapsed_time=self._elapsed(start),
                error_message=result.vector_error,
                document_id=result.document_id,
                file_name=result.file_name,
            )
        except Exception as exc:
            return UploadApiResponse(
                request_id=request_id,
                status="error",
                message="Document upload failed.",
                elapsed_time=self._elapsed(start),
                error_message=self._public_error(exc, fallback="Document upload failed."),
                document_id="",
                file_name=upload.filename or "",
            )

    def parse_document(self, request: DocumentParseRequest) -> DocumentParseResponse:
        file_path = self._resolve_file_path(request.document_id, request.file_path)
        result = load_document(str(file_path))
        if result["status"] == "error":
            suffix = file_path.suffix.lower()
            is_unsupported = suffix not in SUPPORTED_EXTENSIONS
            error_message = str(result["error_message"])
            if is_unsupported:
                error_message = "Only .txt, .md, .docx, and text-based .pdf files are supported."
            message = "Unsupported document type." if is_unsupported else "Failed to read document content."
            if not file_path.exists():
                message = "Document file was not found."
            return DocumentParseResponse(
                document_id=request.document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower().lstrip("."),
                status="error",
                message=message,
                metadata={"error": error_message},
            )

        content = str(result.get("text") or "")
        file_name = str(result.get("file_name") or file_path.name)
        chunks = split_text(
            document_id=request.document_id,
            file_name=file_name,
            text=content,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        summary = summarize_document_text(content)
        return DocumentParseResponse(
            document_id=request.document_id,
            file_name=file_name,
            file_path=str(result.get("file_path") or file_path),
            file_type=str(result.get("file_type") or file_path.suffix.lower().lstrip(".")),
            status="success",
            message="Document parsed successfully.",
            summary=str(summary.get("summary") or ""),
            key_points=list(summary.get("key_points") or []),
            risks=list(summary.get("risks") or []),
            chunks=chunks,
            metadata={
                "line_count": result.get("line_count", 0),
                "char_count": result.get("char_count", 0),
                "chunk_count": len(chunks),
            },
        )

    def ask_document(self, request: DocumentQARequest) -> DocumentQAResponse:
        file_path = self._resolve_file_path(request.document_id, request.file_path)
        state = self.graph.invoke(
            {
                "document_id": request.document_id,
                "file_path": str(file_path),
                "question": request.question,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "top_k": request.top_k,
                "metadata": {"entrypoint": "documents.ask"},
            }
        )
        return DocumentQAResponse.model_validate(state["response"])

    def summarize_document_api(self, document_id: str) -> SummaryApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            file_path = self._resolve_file_path(document_id, None)
            state = self.graph.invoke(
                {
                    "request_id": request_id,
                    "document_id": document_id,
                    "file_path": str(file_path),
                    "task_type": "document_summary",
                    "question": "",
                    "metadata": {"entrypoint": "api.documents.summary"},
                }
            )
            response = state.get("response", {})
            metadata = dict(response.get("metadata") or {})
            self._update_document_metadata_after_graph(
                document_id=document_id,
                metadata=metadata,
                response_status=str(response.get("status") or ""),
                error_message=str(metadata.get("error_message") or response.get("message") or ""),
            )
            return SummaryApiResponse(
                request_id=str(metadata.get("request_id") or request_id),
                status=str(response.get("status") or "error"),
                message=str(response.get("message") or ""),
                elapsed_time=float(metadata.get("elapsed_time") or self._elapsed(start)),
                error_message=str(metadata.get("error_message") or ""),
                summary=str(response.get("answer") or response.get("summary") or ""),
            )
        except Exception as exc:
            return SummaryApiResponse(
                request_id=request_id,
                status="error",
                message="Document summary failed.",
                elapsed_time=self._elapsed(start),
                error_message=self._public_error(exc, fallback="Document summary failed."),
                summary="",
            )

    def ask_document_api(self, *, document_id: str, question: str) -> AskApiResponse:
        document_id = str(document_id or "").strip()
        request_id = str(uuid4())
        start = perf_counter()
        LOGGER.info(
            "QA stage started: request_id=%s document_id=%s question=%s stage_name=received_question elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
            request_id,
            document_id,
            question,
            0.0,
            "",
            0,
        )
        try:
            file_path = self._resolve_file_path(document_id, None)
            state = self.graph.invoke(
                {
                    "request_id": request_id,
                    "document_id": document_id,
                    "file_path": str(file_path),
                    "question": question,
                    "task_type": "document_qa",
                    "metadata": {"entrypoint": "api.qa.ask"},
                }
            )
            response = state.get("response", {})
            metadata = dict(response.get("metadata") or {})
            self._update_document_metadata_after_graph(
                document_id=document_id,
                metadata=metadata,
                response_status=str(response.get("status") or ""),
                error_message=str(metadata.get("error_message") or response.get("message") or ""),
            )
            LOGGER.info(
                "QA stage completed: request_id=%s document_id=%s question=%s stage_name=api_total elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
                str(metadata.get("request_id") or request_id),
                document_id,
                question,
                self._elapsed(start),
                str(metadata.get("retrieval_source") or "none"),
                len(list(response.get("sources") or [])),
            )
            return AskApiResponse(
                request_id=str(metadata.get("request_id") or request_id),
                status=str(response.get("status") or "error"),
                message=str(response.get("message") or ""),
                elapsed_time=float(metadata.get("elapsed_time") or self._elapsed(start)),
                error_message=str(metadata.get("error_message") or ""),
                answer=str(response.get("answer") or ""),
                source_chunks=list(response.get("sources") or []),
                retrieval_source=str(metadata.get("retrieval_source") or "none"),
            )
        except Exception as exc:
            elapsed = self._elapsed(start)
            LOGGER.warning(
                "QA stage failed: request_id=%s document_id=%s question=%s stage_name=api_total elapsed_time=%s retrieval_source=%s source_chunks_count=%s error_type=%s error=%s",
                request_id,
                document_id,
                question,
                elapsed,
                "none",
                0,
                type(exc).__name__,
                exc,
            )
            return AskApiResponse(
                request_id=request_id,
                status="error",
                message="Document QA failed.",
                elapsed_time=elapsed,
                error_message=self._public_error(exc, fallback="Document QA failed."),
                answer="",
                source_chunks=[],
                retrieval_source="none",
            )

    def list_documents_api(self) -> DocumentListApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            documents = [
                DocumentListItem(
                    document_id=str(item.get("document_id") or ""),
                    original_file_name=str(item.get("original_file_name") or ""),
                    stored_file_name=str(item.get("stored_file_name") or ""),
                    upload_time=str(item.get("upload_time") or ""),
                    status=str(item.get("status") or ""),
                    chunk_count=int(item.get("chunk_count") or 0),
                    vector_indexed=bool(item.get("vector_indexed")),
                    error_message=str(item.get("error_message") or ""),
                    updated_time=str(item.get("updated_time") or ""),
                )
                for item in self.storage_service.list_documents()
            ]
            return DocumentListApiResponse(
                request_id=request_id,
                status="success",
                message="Documents loaded successfully.",
                elapsed_time=self._elapsed(start),
                error_message="",
                items=documents,
            )
        except Exception as exc:
            return DocumentListApiResponse(
                request_id=request_id,
                status="error",
                message="Failed to load documents.",
                elapsed_time=self._elapsed(start),
                error_message=self._public_error(exc, fallback="Failed to load documents."),
                items=[],
            )

    def delete_document_api(self, document_id: str) -> DocumentDeleteApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            document = self.storage_service.get_document(document_id)
            if not document:
                return DocumentDeleteApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document not found.",
                    elapsed_time=self._elapsed(start),
                    error_message=f"Document not found: {document_id}",
                    document_id=document_id,
                    file_deleted=False,
                    vector_deleted=False,
                )

            self.storage_service.delete_document_metadata(document_id)
            file_deleted = self._delete_stored_file(str(document.get("file_path") or ""))
            vector_deleted, vector_error = self._cleanup_document_vectors(document_id=document_id, document=document)
            message = "Document deleted successfully."
            if vector_error or not vector_deleted:
                message = "Document record and file deleted; vector cleanup failed or was not required."
            return DocumentDeleteApiResponse(
                request_id=request_id,
                status="success",
                message=message,
                elapsed_time=self._elapsed(start),
                error_message="",
                document_id=document_id,
                file_deleted=file_deleted,
                vector_deleted=vector_deleted,
            )
        except Exception as exc:
            return DocumentDeleteApiResponse(
                request_id=request_id,
                status="error",
                message="Document deletion failed.",
                elapsed_time=self._elapsed(start),
                error_message=self._public_error(exc, fallback="Document deletion failed."),
                document_id=document_id,
                file_deleted=False,
                vector_deleted=False,
            )

    def reindex_document(self, document_id: str) -> DocumentReindexApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        document_id = str(document_id or "").strip()
        try:
            document = self.storage_service.get_document(document_id)
            if not document:
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document not found.",
                    elapsed_time=self._elapsed(start),
                    error_message=f"Document not found: {document_id}",
                    document_id=document_id,
                    chunk_count=0,
                    vector_indexed=False,
                )

            file_path = self._stored_document_path(document)
            if not file_path.exists():
                error_message = f"Document file was not found: {file_path}"
                self.storage_service.update_index_status(
                    document_id,
                    chunk_count=0,
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document file was not found.",
                    elapsed_time=self._elapsed(start),
                    error_message=error_message,
                    document_id=document_id,
                    chunk_count=0,
                    vector_indexed=False,
                )

            parse_result = load_document(str(file_path))
            if parse_result["status"] == "error":
                error_message = str(parse_result.get("error_message") or "Document parsing failed.")
                self.storage_service.update_index_status(
                    document_id,
                    chunk_count=0,
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document parsing failed.",
                    elapsed_time=self._elapsed(start),
                    error_message=error_message,
                    document_id=document_id,
                    chunk_count=0,
                    vector_indexed=False,
                )

            content = str(parse_result.get("text") or "")
            file_name = str(parse_result.get("file_name") or file_path.name)
            chunks = split_text(
                document_id=document_id,
                file_name=file_name,
                text=content,
                chunk_size=800,
                chunk_overlap=100,
            )
            if not chunks:
                error_message = "Document parsed successfully but produced no chunks."
                self.storage_service.update_index_status(
                    document_id,
                    chunk_count=0,
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document reindex failed.",
                    elapsed_time=self._elapsed(start),
                    error_message=error_message,
                    document_id=document_id,
                    chunk_count=0,
                    vector_indexed=False,
                )

            vector_store = get_vector_store()
            delete_warning = ""
            try:
                vector_store.delete_document(document_id)
            except Exception as exc:
                delete_warning = self._public_error(exc, fallback="Chroma old vector deletion failed.")
                LOGGER.warning(
                    "Chroma old vector deletion failed during reindex: document_id=%s error=%s",
                    document_id,
                    delete_warning,
                    exc_info=True,
                )

            try:
                indexed_count = vector_store.add_chunks(
                    document_id=document_id,
                    file_name=file_name,
                    chunks=chunks,
                )
            except Exception as exc:
                error_message = self._public_error(exc, fallback="Chroma vector indexing failed.")
                self.storage_service.update_index_status(
                    document_id,
                    chunk_count=len(chunks),
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document reindex failed.",
                    elapsed_time=self._elapsed(start),
                    error_message=error_message,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    vector_indexed=False,
                )

            vector_indexed = indexed_count > 0
            if not vector_indexed:
                error_message = "Vector store returned zero indexed chunks."
                self.storage_service.update_index_status(
                    document_id,
                    chunk_count=len(chunks),
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
                return DocumentReindexApiResponse(
                    request_id=request_id,
                    status="error",
                    message="Document reindex failed.",
                    elapsed_time=self._elapsed(start),
                    error_message=error_message,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    vector_indexed=False,
                )

            self.storage_service.update_index_status(
                document_id,
                chunk_count=len(chunks),
                vector_indexed=True,
                status="success",
                error_message="",
            )
            return DocumentReindexApiResponse(
                request_id=request_id,
                status="success",
                message="文档重新入库成功",
                elapsed_time=self._elapsed(start),
                error_message="",
                document_id=document_id,
                chunk_count=len(chunks),
                vector_indexed=True,
            )
        except Exception as exc:
            error_message = self._public_error(exc, fallback="Document reindex failed.")
            if document_id and self.storage_service.get_document(document_id):
                self.storage_service.update_index_status(
                    document_id,
                    vector_indexed=False,
                    status="error",
                    error_message=error_message,
                )
            return DocumentReindexApiResponse(
                request_id=request_id,
                status="error",
                message="Document reindex failed.",
                elapsed_time=self._elapsed(start),
                error_message=error_message,
                document_id=document_id,
                chunk_count=0,
                vector_indexed=False,
            )

    def batch_delete_documents(self, document_ids: list[str]) -> BatchDeleteDocumentsResponse:
        request_id = str(uuid4())
        start = perf_counter()
        results: list[BatchDeleteDocumentResult] = []

        for document_id in document_ids:
            try:
                delete_result = self.delete_document_api(document_id)
                if delete_result.status == "success":
                    results.append(
                        BatchDeleteDocumentResult(
                            document_id=document_id,
                            status="success",
                            message="删除成功",
                        )
                    )
                else:
                    message = delete_result.error_message or delete_result.message or "删除失败"
                    if "Document not found" in message:
                        message = "文档不存在"
                    results.append(
                        BatchDeleteDocumentResult(
                            document_id=document_id,
                            status="failed",
                            message=message,
                        )
                    )
            except Exception as exc:
                results.append(
                    BatchDeleteDocumentResult(
                        document_id=document_id,
                        status="failed",
                        message=self._public_error(exc, fallback="删除失败"),
                    )
                )

        deleted_count = sum(1 for result in results if result.status == "success")
        failed_count = len(results) - deleted_count
        return BatchDeleteDocumentsResponse(
            request_id=request_id,
            status="success",
            message="Batch document deletion completed.",
            elapsed_time=self._elapsed(start),
            error_message="",
            deleted_count=deleted_count,
            failed_count=failed_count,
            results=results,
        )

    def list_logs(self, limit: int = 50) -> QALogListResponse:
        return QALogListResponse(items=self.log_service.list_recent(limit=limit))

    def list_logs_api(self, limit: int = 50) -> LogListApiResponse:
        request_id = str(uuid4())
        start = perf_counter()
        try:
            return LogListApiResponse(
                request_id=request_id,
                status="success",
                message="QA logs loaded successfully.",
                elapsed_time=self._elapsed(start),
                error_message="",
                items=self.log_service.list_recent(limit=limit),
            )
        except Exception as exc:
            return LogListApiResponse(
                request_id=request_id,
                status="error",
                message="Failed to load QA logs.",
                elapsed_time=self._elapsed(start),
                error_message=self._public_error(exc, fallback="Failed to load QA logs."),
                items=[],
            )

    def _resolve_file_path(self, document_id: str | None, file_path: str | None) -> Path:
        if file_path:
            return self._resolve_direct_file_path(file_path)

        if not document_id:
            raise ValueError("document_id or file_path is required.")

        document = self.storage_service.get_document(document_id)
        if document:
            return self._stored_document_path(document)

        matches = sorted(self.storage_dir.glob(f"{document_id}_*"))
        if matches:
            return matches[0]

        return self.storage_dir / document_id

    def _resolve_direct_file_path(self, file_path: str) -> Path:
        target = Path(file_path)
        try:
            resolved_target = target.resolve()
            resolved_storage_dir = self.storage_dir.resolve()
        except OSError as exc:
            raise ValueError("Document file path could not be resolved.") from exc
        if not resolved_target.is_relative_to(resolved_storage_dir):
            raise ValueError("Direct file_path must point to a file in the uploads directory.")
        return target

    def _stored_document_path(self, document: dict[str, object]) -> Path:
        document_id = str(document.get("document_id") or "").strip()
        file_path = str(document.get("file_path") or "").strip()
        if file_path:
            path = Path(file_path)
            if path.exists() and self._is_path_in_storage(path):
                self._sync_document_file_path(document_id=document_id, file_path=path)
                return path

        stored_file_name = str(document.get("stored_file_name") or "").strip()
        if stored_file_name:
            path = self.storage_dir / stored_file_name
            if path.exists():
                self._sync_document_file_path(document_id=document_id, file_path=path)
                return path
            return path

        return Path(file_path) if file_path else self.storage_dir / str(document.get("document_id") or "")

    def _index_uploaded_document(self, *, document_id: str, file_path: Path) -> tuple[int, bool, str]:
        try:
            result = load_document(str(file_path))
            if result["status"] == "error":
                error_message = str(result.get("error_message") or "Document parsing failed.")
                LOGGER.warning(
                    "Upload indexing skipped after parse failure: document_id=%s file_path=%s error=%s",
                    document_id,
                    file_path,
                    error_message,
                )
                return 0, False, error_message

            content = str(result.get("text") or "")
            file_name = str(result.get("file_name") or file_path.name)
            chunks = split_text(
                document_id=document_id,
                file_name=file_name,
                text=content,
                chunk_size=800,
                chunk_overlap=100,
            )
            if not chunks:
                return 0, False, "Document parsed successfully but produced no chunks."

            try:
                indexed_count = get_vector_store().add_chunks(
                    document_id=document_id,
                    file_name=file_name,
                    chunks=chunks,
                )
            except Exception as exc:
                error_message = self._public_error(exc, fallback="Vector indexing failed.")
                LOGGER.warning(
                    "Upload vector indexing failed after parsing: document_id=%s file_path=%s chunk_count=%s error=%s",
                    document_id,
                    file_path,
                    len(chunks),
                    error_message,
                    exc_info=True,
                )
                return len(chunks), False, error_message
            vector_indexed = indexed_count > 0
            vector_error = "" if vector_indexed else "Vector store returned zero indexed chunks."
            return len(chunks), vector_indexed, vector_error
        except Exception as exc:
            error_message = self._public_error(exc, fallback="Vector indexing failed.")
            LOGGER.warning(
                "Upload vector indexing failed: document_id=%s file_path=%s error=%s",
                document_id,
                file_path,
                error_message,
                exc_info=True,
            )
            return 0, False, error_message

    def _cleanup_document_vectors(self, *, document_id: str, document: dict[str, object]) -> tuple[bool, str]:
        if not bool(document.get("vector_indexed")):
            LOGGER.info("Skipping Chroma vector cleanup for non-indexed document: document_id=%s", document_id)
            return False, ""
        try:
            get_vector_store().delete_document(document_id)
            return True, ""
        except Exception as exc:
            error_message = self._public_error(exc, fallback="Chroma vector cleanup failed.")
            LOGGER.warning(
                "Chroma vector cleanup failed after document deletion: document_id=%s error=%s",
                document_id,
                error_message,
                exc_info=True,
            )
            return False, error_message

    def _update_document_metadata_after_graph(
        self,
        *,
        document_id: str,
        metadata: dict[str, object],
        response_status: str,
        error_message: str = "",
    ) -> None:
        if not document_id:
            return
        if not self.storage_service.get_document(document_id):
            return

        if response_status.lower() != "success":
            safe_error = str(error_message or metadata.get("error_message") or "").strip()
            self.storage_service.update_index_status(
                document_id,
                vector_indexed=False,
                status="error",
                error_message=safe_error[:240],
            )
            return

        chunk_count = int(metadata.get("chunk_count") or 0)
        indexed_chunks = int(metadata.get("indexed_chunks") or 0)
        vector_store_status = str(metadata.get("vector_store_status") or "")
        vector_indexed = vector_store_status == "success" or indexed_chunks > 0
        self.storage_service.update_index_status(
            document_id,
            chunk_count=chunk_count,
            vector_indexed=vector_indexed,
            status="success",
        )

    def _delete_stored_file(self, file_path: str) -> bool:
        if not file_path:
            return False
        target = Path(file_path)
        try:
            resolved_target = target.resolve()
            resolved_storage_dir = self.storage_dir.resolve()
        except OSError:
            return False
        if not resolved_target.is_relative_to(resolved_storage_dir):
            raise ValueError("Stored document file path is outside the uploads directory.")
        if not target.exists():
            return False
        target.unlink()
        return True

    def _is_path_in_storage(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self.storage_dir.resolve())
        except OSError:
            return False

    def _sync_document_file_path(self, *, document_id: str, file_path: Path) -> None:
        if not document_id:
            return
        document = self.storage_service.get_document(document_id)
        if not document:
            return
        current_path = str(document.get("file_path") or "")
        if Path(current_path) == file_path:
            return
        self.storage_service.update_file_path(document_id, file_path)

    def _sanitize_filename(self, filename: str) -> str:
        name = Path(filename).name.strip() or "document.txt"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    def _elapsed(self, start: float) -> float:
        return round(perf_counter() - start, 6)

    def _public_error(self, exc: Exception, *, fallback: str) -> str:
        message = str(exc).strip()
        if not message:
            return fallback
        if "\n" in message or "Traceback" in message:
            return fallback
        return message[:240]
