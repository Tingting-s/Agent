from __future__ import annotations

from typing import Any

from app.services.qa_log_service import QALogService


class LogAgent:
    def __init__(self, log_service: QALogService | None = None) -> None:
        self.log_service = log_service or QALogService()

    def record(self, *, state: dict[str, Any]) -> dict[str, Any]:
        validation = dict(
            state.get("validation_result")
            or {"is_valid": False, "confidence": 0.0, "issues": []}
        )
        metadata = dict(state.get("metadata") or {})
        retrieval_source = self._retrieval_source(
            metadata=metadata,
            retrieved_chunks=list(state.get("retrieved_chunks") or []),
        )
        metadata["retrieval_source"] = retrieval_source
        entry = self.log_service.append(
            request_id=str(state.get("request_id") or ""),
            task_type=str(state.get("task_type") or ""),
            document_id=state.get("document_id"),
            file_name=str(state.get("file_name") or ""),
            file_path=str(state.get("file_path") or ""),
            question=str(state.get("question") or ""),
            retrieved_chunks=list(state.get("retrieved_chunks") or []),
            answer=str(state.get("answer") or ""),
            validation_result=validation,
            status=str(state.get("status") or "unknown"),
            elapsed_time=float(state.get("elapsed_time") or 0.0),
            error_message=str(state.get("error_message") or ""),
            retrieval_source=retrieval_source,
            metadata=metadata,
        )
        return {"log_id": entry.log_id}

    def _retrieval_source(
        self,
        *,
        metadata: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        metadata_source = str(metadata.get("retrieval_source") or "").strip()
        if metadata_source:
            return metadata_source
        for chunk in retrieved_chunks:
            chunk_source = str(chunk.get("retrieval_source") or "").strip()
            if chunk_source:
                return chunk_source
            chunk_metadata = chunk.get("metadata")
            if isinstance(chunk_metadata, dict):
                metadata_chunk_source = str(chunk_metadata.get("retrieval_source") or "").strip()
                if metadata_chunk_source:
                    return metadata_chunk_source
        return "none"
