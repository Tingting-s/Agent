from __future__ import annotations

import re
from typing import Any

from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE


class DocumentValidatorAgent:
    """Validates answers produced by the document QA LangGraph flow."""

    def validate_document_answer(
        self,
        *,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        task_type: str,
        retrieval_source: str = "",
        relevance_sufficient: bool = True,
    ) -> dict[str, Any]:
        issues: list[str] = []
        normalized_answer = answer.strip()
        is_no_answer = NO_RETRIEVAL_CONTENT_MESSAGE in normalized_answer

        if not normalized_answer:
            issues.append("answer_empty")
        if task_type == "document_qa":
            if not retrieved_chunks:
                issues.append("no_retrieved_chunks")
                if normalized_answer and not is_no_answer:
                    issues.append("answer_without_sources")
            elif is_no_answer:
                issues.append("answer_not_usable")
            if (retrieval_source == "none" or not relevance_sufficient) and not is_no_answer:
                issues.append("answer_without_relevant_sources")
            if self._looks_too_broad(answer=normalized_answer, question="", retrieved_chunks=retrieved_chunks):
                issues.append("answer_too_broad")
            if self._repeats_sources(answer=normalized_answer, retrieved_chunks=retrieved_chunks):
                issues.append("answer_repeats_source_chunks")
        if any(
            marker in normalized_answer.lower()
            for marker in (
                "could not find enough relevant content",
                "document could not be summarized",
                "i don't know",
            )
        ):
            issues.append("answer_not_usable")

        score_total = sum(float(chunk.get("score") or 0.0) for chunk in retrieved_chunks)
        confidence = min(1.0, score_total / max(len(retrieved_chunks), 1))
        if normalized_answer:
            confidence = max(confidence, 0.45)
        if task_type == "document_summary" and normalized_answer:
            confidence = max(confidence, 0.7)
        if issues:
            confidence = min(confidence, 0.35)

        return {
            "is_valid": not issues,
            "confidence": round(confidence, 2),
            "issues": issues,
            "error_message": "; ".join(issues),
        }

    def _looks_too_broad(
        self,
        *,
        answer: str,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> bool:
        if not answer or NO_RETRIEVAL_CONTENT_MESSAGE in answer:
            return False
        bullet_like_lines = [
            line
            for line in answer.splitlines()
            if line.strip().startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6."))
        ]
        if len(answer) <= 800 and len(answer.splitlines()) <= 8 and len(bullet_like_lines) <= 5:
            return False
        return bool(retrieved_chunks)

    def _repeats_sources(self, *, answer: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
        if not answer or NO_RETRIEVAL_CONTENT_MESSAGE in answer:
            return False
        compact_answer = re.sub(r"\s+", "", answer)
        if len(compact_answer) < 300:
            return False
        for chunk in retrieved_chunks:
            compact_chunk = re.sub(r"\s+", "", str(chunk.get("text") or ""))
            if len(compact_chunk) >= 300 and compact_chunk[:300] in compact_answer:
                return True
        return False
