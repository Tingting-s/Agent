from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.rag.generation import summarize_document_text
from app.rag.ingestion import load_document, split_text
from app.rag.vector_store import ChromaVectorStore, get_vector_store


LOGGER = logging.getLogger(__name__)


class DocumentAgent:
    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store

    def parse(
        self,
        *,
        file_path: str,
        document_id: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> dict[str, Any]:
        result = load_document(file_path)
        if result["status"] == "error":
            return {
                "status": "error",
                "error_message": result["error_message"],
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "parsed_text": "",
                "chunks": [],
            }

        parsed_text = str(result.get("text") or "")
        file_name = str(result.get("file_name") or Path(file_path).name)
        chunks = split_text(
            document_id=document_id,
            file_name=file_name,
            text=parsed_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        vector_metadata = self._index_chunks(
            document_id=document_id,
            file_name=file_name,
            chunks=chunks,
        )
        summary = summarize_document_text(parsed_text)
        return {
            "status": "success",
            "file_path": str(result.get("file_path") or file_path),
            "file_name": file_name,
            "file_type": str(result.get("file_type") or Path(file_path).suffix.lower().lstrip(".")),
            "parsed_text": parsed_text,
            "chunks": chunks,
            "summary": str(summary.get("summary") or ""),
            "key_points": list(summary.get("key_points") or []),
            "risks": list(summary.get("risks") or []),
            "metadata": {
                "line_count": result.get("line_count", 0),
                "char_count": result.get("char_count", 0),
                "chunk_count": len(chunks),
                **vector_metadata,
            },
        }

    def _index_chunks(
        self,
        *,
        document_id: str | None,
        file_name: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not document_id or not chunks:
            LOGGER.info(
                "Chroma indexing skipped: document_id=%s file_name=%s chunk_count=%s",
                document_id,
                file_name,
                len(chunks),
            )
            return {"vector_store_status": "skipped", "vector_store_error": ""}
        try:
            vector_store = self.vector_store or get_vector_store()
            self.vector_store = vector_store
            indexed_count = vector_store.add_chunks(
                document_id=document_id,
                file_name=file_name,
                chunks=chunks,
            )
            LOGGER.info(
                "Chroma indexing succeeded: document_id=%s file_name=%s indexed_chunks=%s",
                document_id,
                file_name,
                indexed_count,
            )
            return {
                "vector_store_status": "success",
                "vector_store": "chroma",
                "indexed_chunks": indexed_count,
                "vector_store_error": "",
            }
        except Exception as exc:
            LOGGER.exception(
                "Chroma indexing failed: document_id=%s file_name=%s chunk_count=%s",
                document_id,
                file_name,
                len(chunks),
            )
            return {
                "vector_store_status": "error",
                "vector_store": "chroma",
                "indexed_chunks": 0,
                "vector_store_error": str(exc) or "Chroma indexing failed.",
            }
