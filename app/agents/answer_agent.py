from __future__ import annotations

import re
import logging
from time import perf_counter
from typing import Any

from app.rag.generation import (
    MAX_PROMPT_CHUNK_CHARS,
    MAX_PROMPT_CHUNKS,
    NO_RETRIEVAL_CONTENT_MESSAGE,
    build_document_qa_prompt,
)
from app.services.llm_service import LLM_TIMEOUT_MESSAGE, LLMTimeoutError, get_qwen_llm_service


LOGGER = logging.getLogger(__name__)


class AnswerGenerationTimeout(RuntimeError):
    """Raised when answer generation exceeds the LLM timeout."""


class AnswerAgent:
    def build_prompt(
        self,
        *,
        question: str,
        context_chunks: list[dict[str, Any]],
        summary: str = "",
    ) -> str:
        return build_document_qa_prompt(question=question, retrieved_chunks=context_chunks)

    def generate_answer(
        self,
        *,
        question: str,
        context_chunks: list[dict[str, Any]],
        summary: str = "",
        request_id: str = "",
        document_id: str = "",
        retrieval_source: str = "",
    ) -> dict[str, Any]:
        effective_chunks = self._effective_chunks(context_chunks)
        context = self._build_context(effective_chunks)
        if not context:
            return {
                "answer": NO_RETRIEVAL_CONTENT_MESSAGE,
                "confidence": 0.0,
                "source": "fallback",
            }

        try:
            llm_start = perf_counter()
            LOGGER.info(
                "QA stage started: request_id=%s document_id=%s question=%s stage_name=llm_generate retrieval_source=%s source_chunks_count=%s",
                request_id,
                document_id,
                question,
                retrieval_source,
                len(effective_chunks),
            )
            result = get_qwen_llm_service().answer_document_question(
                question=question,
                context=context,
                summary="",
            )
            if isinstance(result, dict) and str(result.get("answer") or "").strip():
                result["answer"] = self._limit_answer(str(result.get("answer") or ""))
                LOGGER.info(
                    "QA stage completed: request_id=%s document_id=%s question=%s stage_name=llm_generate elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
                    request_id,
                    document_id,
                    question,
                    round(perf_counter() - llm_start, 6),
                    retrieval_source,
                    len(effective_chunks),
                )
                return result
        except LLMTimeoutError as exc:
            LOGGER.warning(
                "QA stage timeout: request_id=%s document_id=%s question=%s stage_name=llm_generate error_message=%s retrieval_source=%s source_chunks_count=%s",
                request_id,
                document_id,
                question,
                exc,
                retrieval_source,
                len(effective_chunks),
            )
            raise AnswerGenerationTimeout(LLM_TIMEOUT_MESSAGE) from exc
        except Exception as exc:
            if self._is_timeout_exception(exc):
                LOGGER.warning(
                    "QA stage timeout: request_id=%s document_id=%s question=%s stage_name=llm_generate error_type=%s error=%s retrieval_source=%s source_chunks_count=%s",
                    request_id,
                    document_id,
                    question,
                    type(exc).__name__,
                    exc,
                    retrieval_source,
                    len(effective_chunks),
                )
                raise AnswerGenerationTimeout(LLM_TIMEOUT_MESSAGE) from exc
            LOGGER.warning(
                "LLM generation unavailable; using extractive fallback: request_id=%s document_id=%s error_type=%s error=%s",
                request_id,
                document_id,
                type(exc).__name__,
                exc,
            )

        return {
            "answer": self._fallback_answer(question=question, context_chunks=effective_chunks),
            "confidence": 0.45,
            "source": "extractive_fallback",
        }

    def generate_summary_answer(self, *, summary: str, key_points: list[str], risks: list[str]) -> dict[str, Any]:
        sections = []
        if summary:
            sections.append(summary)
        if key_points:
            sections.append("Key points: " + "; ".join(str(item) for item in key_points[:5]))
        if risks:
            sections.append("Risks: " + "; ".join(str(item) for item in risks[:5]))
        answer = "\n".join(sections).strip() or "The document could not be summarized."
        return {"answer": answer, "confidence": 0.7 if summary else 0.2, "source": "summary"}

    def _build_context(self, context_chunks: list[dict[str, Any]]) -> str:
        sections = []
        for index, chunk in enumerate(context_chunks, start=1):
            text = str(chunk.get("text") or "").strip()
            if text:
                sections.append(f"[{index}] {text}")
        return "\n\n".join(sections)

    def _fallback_answer(self, *, question: str, context_chunks: list[dict[str, Any]]) -> str:
        query_terms = set(self._tokenize(question))
        ranked_sentences: list[tuple[int, str]] = []
        for chunk in context_chunks:
            for sentence in self._split_sentences(str(chunk.get("text") or "")):
                if not self._sentence_matches_required_scope(question=question, sentence=sentence):
                    continue
                sentence_terms = set(self._tokenize(sentence))
                overlap = len(query_terms.intersection(sentence_terms))
                overlap_ratio = overlap / max(len(query_terms), 1)
                if overlap >= 2 or overlap_ratio >= 0.3:
                    ranked_sentences.append((overlap, sentence))

        if not ranked_sentences:
            return NO_RETRIEVAL_CONTENT_MESSAGE

        ranked_sentences.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        max_overlap = ranked_sentences[0][0]
        min_overlap = max(2, int(max_overlap * 0.6))
        selected: list[str] = []
        seen: set[str] = set()
        for overlap, sentence in ranked_sentences:
            if overlap < min_overlap:
                continue
            normalized = sentence.strip()
            if normalized and normalized not in seen:
                selected.append(normalized)
                seen.add(normalized)
            if len(selected) >= 5:
                break

        if not selected:
            return NO_RETRIEVAL_CONTENT_MESSAGE
        if len(selected) == 1:
            return self._limit_answer(selected[0])
        return self._limit_answer("\n".join(f"- {item}" for item in selected))

    def _effective_chunks(self, context_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        effective_chunks: list[dict[str, Any]] = []
        for chunk in context_chunks:
            text = str(chunk.get("text") or "").strip()
            if not text or str(chunk.get("retrieval_source") or "").strip() == "none":
                continue
            limited_chunk = dict(chunk)
            limited_chunk["text"] = text[:MAX_PROMPT_CHUNK_CHARS]
            effective_chunks.append(limited_chunk)
            if len(effective_chunks) >= MAX_PROMPT_CHUNKS:
                break
        return effective_chunks

    def _split_sentences(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?；;])\s*|\n+", normalized)
            if sentence.strip()
        ]

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_chars = [cjk_text[index : index + 2] for index in range(max(len(cjk_text) - 1, 0))]
        words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized)
        return cjk_chars + words

    def _sentence_matches_required_scope(self, *, question: str, sentence: str) -> bool:
        normalized_question = question.lower()
        normalized_sentence = sentence.lower()
        required_phrases = ("启动前", "停机", "运行维护", "采购", "价格", "费用", "售后", "保修", "工资", "奖金", "地址", "电话")
        for phrase in required_phrases:
            if phrase in normalized_question and phrase not in normalized_sentence:
                return False
        return True

    def _limit_answer(self, answer: str, max_chars: int = 900) -> str:
        normalized = answer.strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip()

    def _is_timeout_exception(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return "timeout" in name or "timed out" in message or LLM_TIMEOUT_MESSAGE in str(exc)
