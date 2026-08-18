from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.config import get_log_dir, get_settings
from app.schemas.common import AnswerValidation
from app.schemas.logs import QALogEntry


class QALogService:
    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path or get_log_dir() / "qa_logs.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        request_id: str | None = None,
        task_type: str = "",
        document_id: str | None,
        file_name: str = "",
        file_path: str,
        question: str,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        answer: str,
        validation_result: dict[str, Any],
        status: str,
        elapsed_time: float = 0.0,
        error_message: str = "",
        retrieval_source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> QALogEntry:
        append_start = perf_counter()
        safe_metadata = dict(metadata or {})
        safe_retrieval_source = retrieval_source or str(safe_metadata.get("retrieval_source") or "")
        if not safe_retrieval_source:
            safe_retrieval_source = "none"
        safe_metadata.setdefault("retrieval_source", safe_retrieval_source)
        safe_metadata.setdefault("log_record_elapsed", round(perf_counter() - append_start, 6))
        safe_chunks = retrieved_chunks or []
        fields = self._structured_fields(
            retrieved_chunks=safe_chunks,
            validation_result=validation_result,
            elapsed_time=elapsed_time,
            error_message=error_message,
            retrieval_source=safe_retrieval_source,
            metadata=safe_metadata,
        )
        entry = QALogEntry(
            log_id=str(uuid4()),
            request_id=request_id,
            task_type=task_type,
            timestamp=datetime.now(UTC).isoformat(),
            document_id=document_id,
            file_name=file_name,
            file_path=file_path,
            question=question,
            retrieved_chunks=safe_chunks,
            answer=answer,
            validation_result=AnswerValidation.model_validate(validation_result),
            status=status,
            elapsed_time=elapsed_time,
            error_message=error_message,
            retrieval_source=safe_retrieval_source,
            **fields,
            metadata=safe_metadata,
        )
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
        return entry

    def list_recent(self, limit: int = 50) -> list[QALogEntry]:
        if not self.log_path.exists():
            return []

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        entries: list[QALogEntry] = []
        for line in reversed(lines[-limit:]):
            if not line.strip():
                continue
            try:
                raw_entry = json.loads(line)
                if isinstance(raw_entry, dict):
                    raw_entry = self._normalize_legacy_entry(raw_entry)
                entries.append(QALogEntry.model_validate(raw_entry))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries

    def _structured_fields(
        self,
        *,
        retrieved_chunks: list[dict[str, Any]],
        validation_result: dict[str, Any],
        elapsed_time: float,
        error_message: str,
        retrieval_source: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        settings = get_settings()
        top_chunk = self._top_chunk(retrieved_chunks)
        validation_message = str(
            validation_result.get("error_message")
            or "; ".join(str(item) for item in validation_result.get("issues", []) if item)
        )
        is_valid = bool(validation_result.get("is_valid"))
        failed_stage = str(metadata.get("failed_stage") or "")
        if not failed_stage and error_message:
            failed_stage = "unknown"
        return {
            "fallback_reason": self._none_if_empty(metadata.get("fallback_reason")),
            "source_chunks_count": len(retrieved_chunks),
            "top_score": self._safe_float(top_chunk.get("score")) if top_chunk else None,
            "top_distance": self._safe_float(top_chunk.get("distance")) if top_chunk else None,
            "top_chunk_id": self._none_if_empty(top_chunk.get("chunk_id")) if top_chunk else None,
            "rerank_enabled": self._safe_bool(metadata.get("rerank_enabled")),
            "rerank_top_n": self._safe_int(metadata.get("rerank_top_n")),
            "top_rerank_score": self._top_rerank_score(retrieved_chunks),
            "retrieval_elapsed": self._safe_float(metadata.get("retrieval_elapsed")),
            "chroma_elapsed": self._safe_float(metadata.get("chroma_elapsed")),
            "rerank_elapsed": self._safe_float(metadata.get("rerank_elapsed")),
            "prompt_build_elapsed": self._safe_float(metadata.get("prompt_build_elapsed")),
            "llm_elapsed": self._safe_float(metadata.get("llm_elapsed")),
            "validation_elapsed": self._safe_float(metadata.get("validation_elapsed")),
            "total_elapsed": self._safe_float(metadata.get("total_elapsed")) or round(float(elapsed_time or 0.0), 6),
            "log_record_elapsed": self._safe_float(metadata.get("log_record_elapsed")),
            "embedding_model": self._none_if_empty(metadata.get("embedding_model")) or settings.EMBEDDING_MODEL_NAME,
            "llm_model": self._none_if_empty(metadata.get("llm_model")) or settings.QWEN_MODEL,
            "llm_timeout": self._safe_float(metadata.get("llm_timeout")) or float(settings.LLM_TIMEOUT),
            "validation_status": "success" if is_valid else "error",
            "validation_message": validation_message or None,
            "failed_stage": failed_stage or None,
        }

    def _normalize_legacy_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(entry.get("metadata") or {})
        retrieved_chunks = entry.get("retrieved_chunks")
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []
        validation = entry.get("validation_result")
        if not isinstance(validation, dict):
            validation = {"is_valid": False, "confidence": 0.0, "issues": [], "error_message": ""}
        entry = dict(entry)
        entry["metadata"] = metadata
        entry["retrieved_chunks"] = retrieved_chunks
        entry["validation_result"] = validation
        entry.setdefault("retrieval_source", metadata.get("retrieval_source") or "none")
        structured = self._structured_fields(
            retrieved_chunks=retrieved_chunks,
            validation_result=validation,
            elapsed_time=float(entry.get("elapsed_time") or 0.0),
            error_message=str(entry.get("error_message") or ""),
            retrieval_source=str(entry.get("retrieval_source") or "none"),
            metadata=metadata,
        )
        for key, value in structured.items():
            entry.setdefault(key, value)
        return entry

    def _top_chunk(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            return {}
        return max(chunks, key=lambda chunk: float(chunk.get("score") or 0.0))

    def _top_rerank_score(self, chunks: list[dict[str, Any]]) -> float | None:
        scores = [self._safe_float(chunk.get("rerank_score")) for chunk in chunks if "rerank_score" in chunk]
        scores = [score for score in scores if score is not None]
        return max(scores) if scores else None

    def _none_if_empty(self, value: Any) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    def _safe_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
