from __future__ import annotations

import logging
import re
from collections import Counter
from time import perf_counter
from typing import Any

from app.rag.vector_store import ChromaVectorStore, get_vector_store


LOGGER = logging.getLogger(__name__)
DEFAULT_RERANK_TOP_N = 3
DEFAULT_MAX_RERANK_TEXT_CHARS = 800
RETRIEVAL_SOURCE_CHROMA = "chroma"
RETRIEVAL_SOURCE_FALLBACK = "fallback"
RETRIEVAL_SOURCE_NONE = "none"
DEFAULT_CHROMA_CANDIDATE_K = 10
DEFAULT_MIN_RERANK_SCORE = 1.2
DEFAULT_MIN_KEYWORD_OVERLAP = 2
MAX_SOURCE_CHUNK_TEXT_CHARS = 600
SENSITIVE_QUERY_TERM_GROUPS = (
    ("价格", "费用", "金额", "采购", "多少钱", "售价", "报价", "price", "cost", "fee", "purchase"),
    ("售后", "保修", "质保", "sla", "warranty", "after-sales"),
    ("年终奖", "工资", "薪酬", "奖金", "bonus", "salary", "compensation"),
    ("地址", "电话", "邮箱", "联系方式", "address", "phone", "email", "contact"),
)
STOP_WORDS = {
    "the",
    "what",
    "which",
    "when",
    "where",
    "how",
    "why",
    "are",
    "is",
    "was",
    "were",
    "for",
    "and",
    "or",
    "with",
    "about",
    "required",
    "require",
    "need",
    "needs",
    "should",
    "does",
    "do",
    "document",
    "policy",
    "设备",
    "需要",
    "什么",
    "哪些",
    "如何",
    "怎么",
    "应该",
    "进行",
    "包括",
}


class RuleBasedReranker:
    """Lightweight local reranker for Chroma candidates."""

    def rerank(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        query_terms = self._tokenize(question)
        query_keywords = set(self._keywords(question))
        query_core_terms = self._core_terms(question)
        query_phrases = self._phrases(question)
        query_cjk_chars = set(self._cjk_chars(question))
        query_counter = Counter(query_terms)
        query_text = question.strip().lower()
        scored_chunks: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "")
            visible_text = text[:DEFAULT_MAX_RERANK_TEXT_CHARS]
            metadata = dict(chunk.get("metadata") or {})
            text_terms = self._tokenize(text)
            text_keywords = set(self._keywords(text))
            text_counter = Counter(text_terms)
            overlap = sum(min(query_counter[term], text_counter[term]) for term in query_counter)
            coverage = overlap / max(len(query_counter), 1)
            density = overlap / max(len(text_terms), 1)
            keyword_overlap = len(query_keywords.intersection(text_keywords))
            keyword_coverage = keyword_overlap / max(len(query_keywords), 1)
            core_coverage = self._coverage(query_core_terms, self._core_terms(text))
            cjk_overlap = self._coverage(query_cjk_chars, set(self._cjk_chars(text)))
            phrase_hits = self._phrase_hits(query_phrases, text)
            title_or_lead_hit = self._title_or_lead_hit(query_core_terms, visible_text, metadata)
            exact_bonus = 1.0 if query_text and query_text in text.lower() else 0.0
            vector_score = float(chunk.get("score") or 0.0)
            distance_score = self._distance_score(chunk.get("distance"))
            length_penalty = self._length_penalty(text)
            high_frequency_penalty = self._high_frequency_penalty(text_keywords, query_keywords, len(text))
            rerank_score = (
                (keyword_coverage * 3.2)
                + (min(keyword_overlap, 6) * 0.42)
                + (core_coverage * 2.3)
                + (coverage * 1.1)
                + (density * 0.7)
                + (cjk_overlap * 1.1)
                + (phrase_hits * 0.75)
                + title_or_lead_hit
                + exact_bonus
                + (self._bounded_vector_score(vector_score) * 0.45)
                + (distance_score * 0.25)
                - high_frequency_penalty
            ) * length_penalty

            enriched = dict(chunk)
            enriched["rerank_score"] = round(rerank_score, 6)
            enriched["_rerank_original_index"] = index
            scored_chunks.append(enriched)

        scored_chunks.sort(
            key=lambda item: (
                float(item.get("rerank_score") or 0.0),
                float(item.get("score") or 0.0),
                -int(item.get("_rerank_original_index") or 0),
            ),
            reverse=True,
        )

        reranked = scored_chunks[: max(1, top_n)]
        for chunk in reranked:
            chunk.pop("_rerank_original_index", None)
        return reranked

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_bigrams = [cjk_text[index : index + 2] for index in range(max(len(cjk_text) - 1, 0))]
        words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized)
        return cjk_bigrams + words

    def _keywords(self, text: str) -> list[str]:
        normalized = text.lower()
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_terms = [cjk_text[index : index + 2] for index in range(max(len(cjk_text) - 1, 0))]
        words = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", normalized)
        return [term for term in cjk_terms + words if term not in STOP_WORDS]

    def _core_terms(self, text: str) -> set[str]:
        keywords = self._keywords(text)
        return {
            term
            for term in keywords
            if term not in STOP_WORDS and (self._is_cjk(term) or len(term) >= 4)
        }

    def _phrases(self, text: str) -> list[str]:
        normalized = text.lower()
        words = [
            word
            for word in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", normalized)
            if word not in STOP_WORDS
        ]
        word_phrases = [" ".join(words[index : index + 2]) for index in range(max(len(words) - 1, 0))]
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_phrases = [cjk_text[index : index + 3] for index in range(max(len(cjk_text) - 2, 0))]
        return [phrase for phrase in word_phrases + cjk_phrases if phrase.strip()]

    def _phrase_hits(self, query_phrases: list[str], text: str) -> int:
        normalized = text.lower()
        return min(sum(1 for phrase in query_phrases if phrase and phrase in normalized), 4)

    def _cjk_chars(self, text: str) -> list[str]:
        return re.findall(r"[\u4e00-\u9fff]", text)

    def _coverage(self, query_items: set[str], text_items: set[str]) -> float:
        if not query_items:
            return 0.0
        return len(query_items.intersection(text_items)) / max(len(query_items), 1)

    def _title_or_lead_hit(self, query_core_terms: set[str], text: str, metadata: dict[str, Any]) -> float:
        if not query_core_terms:
            return 0.0
        title_parts = [
            str(metadata.get("title") or ""),
            str(metadata.get("heading") or ""),
            str(metadata.get("section") or ""),
            str(metadata.get("file_name") or ""),
        ]
        title_text = " ".join(title_parts).lower()
        lead_text = text[:160].lower()
        title_hits = sum(1 for term in query_core_terms if term.lower() in title_text)
        lead_hits = sum(1 for term in query_core_terms if term.lower() in lead_text)
        return min(title_hits * 0.55 + lead_hits * 0.28, 1.5)

    def _length_penalty(self, text: str) -> float:
        length = len(text.strip())
        if length <= DEFAULT_MAX_RERANK_TEXT_CHARS:
            return 1.0
        if length <= DEFAULT_MAX_RERANK_TEXT_CHARS * 2:
            return 0.86
        if length <= DEFAULT_MAX_RERANK_TEXT_CHARS * 4:
            return 0.72
        return 0.58

    def _high_frequency_penalty(self, text_keywords: set[str], query_keywords: set[str], text_length: int) -> float:
        if not text_keywords or text_length < DEFAULT_MAX_RERANK_TEXT_CHARS * 2:
            return 0.0
        unrelated_terms = len(text_keywords - query_keywords)
        return min(unrelated_terms / max(len(query_keywords), 1) * 0.08, 1.2)

    def _bounded_vector_score(self, value: float) -> float:
        return max(0.0, min(float(value or 0.0), 1.0))

    def _distance_score(self, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        try:
            distance = max(float(value), 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 / (1.0 + distance)

    def _is_cjk(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)


class BasicRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        reranker: RuleBasedReranker | None = None,
    ) -> None:
        self.embedding_model_name: str | None = "local-hash-embedding"
        self.vector_store: ChromaVectorStore | None = vector_store
        self.reranker: RuleBasedReranker = reranker or RuleBasedReranker()
        self._last_fallback_reason = ""

    def retrieve(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        result = self.retrieve_with_source(question=question, chunks=chunks, top_k=top_k)
        return list(result.get("chunks") or [])

    def retrieve_with_source(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        request_id: str = "",
    ) -> dict[str, Any]:
        retrieval_start = perf_counter()
        self._last_fallback_reason = ""
        self._last_chroma_elapsed: float | None = None
        self._last_rerank_elapsed: float | None = None
        self._last_rerank_enabled = False
        self._last_rerank_top_n: int | None = None
        self._required_group = self._required_term_group(question)
        self._current_question = question
        self._request_id = request_id
        vector_results = self._vector_retrieve(
            question=question,
            chunks=chunks,
            top_k=self._chroma_candidate_k(top_k),
        )
        if vector_results:
            rerank_start = perf_counter()
            rerank_top_n = self._rerank_top_n(top_k)
            self._last_rerank_enabled = True
            self._last_rerank_top_n = rerank_top_n
            self._log_stage(
                "rerank",
                "started",
                question=question,
                document_id=self._infer_document_id(chunks),
                retrieval_source=RETRIEVAL_SOURCE_CHROMA,
                source_chunks_count=len(vector_results),
            )
            reranked_results = self._rerank_or_original(
                question=question,
                chunks=vector_results,
                top_n=rerank_top_n,
            )
            self._log_rerank_candidates(reranked_results)
            self._last_rerank_elapsed = round(perf_counter() - rerank_start, 6)
            self._log_stage(
                "rerank",
                "completed",
                question=question,
                document_id=self._infer_document_id(chunks),
                elapsed_time=self._last_rerank_elapsed,
                retrieval_source=RETRIEVAL_SOURCE_CHROMA,
                source_chunks_count=len(reranked_results),
            )
            relevant_results = self._filter_relevant_chunks(reranked_results)
            if not relevant_results:
                self._last_fallback_reason = "chroma_results_below_relevance_threshold"
                LOGGER.info(
                    "No answer after Chroma rerank; fallback is not used because Chroma returned candidates: reason=%s top_score=%s",
                    self._last_fallback_reason,
                    self._best_rerank_score(reranked_results),
                )
                self._log_retrieval_source(RETRIEVAL_SOURCE_NONE, [])
                return {
                    "chunks": [],
                    "retrieval_source": RETRIEVAL_SOURCE_NONE,
                    "fallback_reason": self._last_fallback_reason,
                    **self._runtime_metadata(retrieval_start),
                }
            tagged_results = self._with_retrieval_source(relevant_results, RETRIEVAL_SOURCE_CHROMA)
            self._log_retrieval_source(RETRIEVAL_SOURCE_CHROMA, tagged_results)
            return {
                "chunks": tagged_results,
                "retrieval_source": RETRIEVAL_SOURCE_CHROMA,
                "fallback_reason": "",
                **self._runtime_metadata(retrieval_start),
            }

        fallback_reason = self._last_fallback_reason or "chroma_no_results"
        LOGGER.info("Falling back to keyword retrieval: fallback_reason=%s", fallback_reason)
        fallback_results = self._keyword_retrieve(question=question, chunks=chunks, top_k=top_k)
        relevant_results = self._filter_relevant_chunks(fallback_results[: self._rerank_top_n(top_k)])
        tagged_results = self._with_retrieval_source(relevant_results, RETRIEVAL_SOURCE_FALLBACK)
        retrieval_source = RETRIEVAL_SOURCE_FALLBACK if tagged_results else RETRIEVAL_SOURCE_NONE
        self._log_retrieval_source(retrieval_source, tagged_results)
        return {
            "chunks": tagged_results,
            "retrieval_source": retrieval_source,
            "fallback_reason": fallback_reason,
            **self._runtime_metadata(retrieval_start),
        }

    def _keyword_retrieve(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_terms = self._tokenize(question)
        if not query_terms:
            return []
        required_group = self._required_term_group(question)

        query_counter = Counter(query_terms)
        scored_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str(chunk.get("text") or "")
            if required_group and not self._contains_any(text, required_group):
                continue
            chunk_terms = self._tokenize(text)
            if not chunk_terms:
                continue

            chunk_counter = Counter(chunk_terms)
            overlap_score = sum(min(query_counter[term], chunk_counter[term]) for term in query_counter)
            exact_bonus = 2 if question.strip().lower() in text.lower() else 0
            score = float(overlap_score + exact_bonus)
            if score <= 0:
                continue

            enriched = dict(chunk)
            enriched["score"] = score
            enriched["rerank_score"] = score
            scored_chunks.append(enriched)

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        return scored_chunks[:top_k]

    def _vector_retrieve(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        document_id = self._infer_document_id(chunks)
        if not document_id:
            self._last_fallback_reason = "missing_document_id"
            LOGGER.info("Chroma retrieval skipped: reason=%s", self._last_fallback_reason)
            return []

        try:
            vector_store = self.vector_store or get_vector_store()
            self.vector_store = vector_store
            chroma_start = perf_counter()
            self._log_stage(
                "chroma_retrieve",
                "started",
                question=question,
                document_id=document_id,
                retrieval_source=RETRIEVAL_SOURCE_CHROMA,
                source_chunks_count=0,
            )
            LOGGER.info(
                "Chroma retrieval requested: request_id=%s document_id=%s question=%s top_k=%s",
                getattr(self, "_request_id", ""),
                document_id,
                question,
                top_k,
            )
            results = vector_store.search(document_id=document_id, question=question, top_k=top_k)
            self._last_chroma_elapsed = round(perf_counter() - chroma_start, 6)
            self._log_stage(
                "chroma_retrieve",
                "completed",
                question=question,
                document_id=document_id,
                elapsed_time=self._last_chroma_elapsed,
                retrieval_source=RETRIEVAL_SOURCE_CHROMA,
                source_chunks_count=len(results),
            )
        except Exception as exc:
            self._last_chroma_elapsed = round(perf_counter() - chroma_start, 6) if "chroma_start" in locals() else None
            self._last_fallback_reason = f"chroma_query_error: {exc}"
            LOGGER.exception(
                "Chroma retrieval failed: document_id=%s question=%s top_k=%s",
                document_id,
                question,
                top_k,
            )
            return []
        if not results:
            self._last_fallback_reason = "chroma_no_results"
            try:
                indexed_count = vector_store.get_document_chunk_count(document_id)
            except Exception as exc:
                indexed_count = -1
                LOGGER.warning(
                    "Failed to inspect Chroma document chunk count after empty query: document_id=%s error=%s",
                    document_id,
                    exc,
                    exc_info=True,
                )
            LOGGER.info(
                "Chroma retrieval returned no results: document_id=%s question=%s top_k=%s indexed_count=%s fallback_reason=%s",
                document_id,
                question,
                top_k,
                indexed_count,
                self._last_fallback_reason,
            )
            return []

        LOGGER.info(
            "Chroma retrieval candidate counts: document_id=%s raw_count=%s returned_count=%s",
            document_id,
            len(results),
            min(len(results), top_k),
        )
        return results[:top_k]

    def bge_m3_retrieve_placeholder(self) -> None:
        raise NotImplementedError("bge-m3 retrieval is reserved for a later phase.")

    def chroma_retrieve_placeholder(self) -> None:
        raise NotImplementedError("Chroma retrieval is handled by ChromaVectorStore.")

    def faiss_retrieve_placeholder(self) -> None:
        raise NotImplementedError("FAISS retrieval is reserved for a later phase.")

    def rerank_placeholder(self) -> None:
        raise NotImplementedError("Rerank is reserved for a later phase.")

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        cjk_bigrams = [cjk_text[index : index + 2] for index in range(max(len(cjk_text) - 1, 0))]
        words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized)
        return cjk_bigrams + words

    def _infer_document_id(self, chunks: list[dict[str, Any]]) -> str:
        for chunk in chunks:
            document_id = str(chunk.get("document_id") or "").strip()
            if document_id:
                return document_id
        return ""

    def _rerank_or_original(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        try:
            return self.reranker.rerank(question=question, chunks=chunks, top_n=top_n)
        except Exception:
            LOGGER.exception("Rerank failed; using original Chroma ordering.")
            return chunks[:top_n]

    def _chroma_candidate_k(self, top_k: int) -> int:
        return max(DEFAULT_CHROMA_CANDIDATE_K, int(top_k or 0))

    def _rerank_top_n(self, top_k: int) -> int:
        return max(1, min(DEFAULT_RERANK_TOP_N, int(top_k or DEFAULT_RERANK_TOP_N)))

    def _runtime_metadata(self, retrieval_start: float) -> dict[str, Any]:
        return {
            "retrieval_elapsed": round(perf_counter() - retrieval_start, 6),
            "chroma_elapsed": self._last_chroma_elapsed,
            "rerank_elapsed": self._last_rerank_elapsed,
            "rerank_enabled": self._last_rerank_enabled,
            "rerank_top_n": self._last_rerank_top_n,
            "embedding_model": self._embedding_model_label(),
        }

    def _embedding_model_label(self) -> str:
        vector_store = self.vector_store
        embedding_function = getattr(vector_store, "embedding_function", None)
        model_name = getattr(embedding_function, "model_name", None)
        if model_name:
            return str(model_name)
        name = getattr(embedding_function, "name", None)
        if callable(name):
            try:
                return str(name())
            except Exception:
                return self.embedding_model_name or "unknown"
        return self.embedding_model_name or "unknown"

    def _filter_relevant_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if chunks and self._missing_required_topic(chunks):
            self._log_filter_decision(chunks, "filtered", "missing_required_topic")
            return []
        overlap_filtered_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            if self._has_minimum_overlap(chunk):
                overlap_filtered_chunks.append(chunk)
            else:
                self._log_filter_decision([chunk], "filtered", "minimum_keyword_overlap_not_met")
        chunks = overlap_filtered_chunks
        scored_chunks = [chunk for chunk in chunks if "rerank_score" in chunk]
        if not scored_chunks:
            self._log_filter_decision(chunks, "kept", "no_rerank_score")
            return chunks

        best_score = max(float(chunk.get("rerank_score") or 0.0) for chunk in scored_chunks)
        if best_score < DEFAULT_MIN_RERANK_SCORE:
            self._log_filter_decision(
                chunks,
                "filtered",
                f"best_rerank_score_below_threshold:{round(best_score, 6)}<{DEFAULT_MIN_RERANK_SCORE}",
            )
            return []
        kept_chunks = [
            chunk
            for chunk in chunks
            if float(chunk.get("rerank_score") or 0.0) >= DEFAULT_MIN_RERANK_SCORE
        ]
        for chunk in chunks:
            if chunk in kept_chunks:
                self._log_filter_decision([chunk], "kept", "rerank_score_meets_threshold")
            else:
                self._log_filter_decision([chunk], "filtered", "rerank_score_below_threshold")
        return kept_chunks

    def _best_rerank_score(self, chunks: list[dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        return max(float(chunk.get("rerank_score") or 0.0) for chunk in chunks)

    def _has_minimum_overlap(self, chunk: dict[str, Any]) -> bool:
        question = str(getattr(self, "_current_question", "") or "")
        query_terms = [term for term in self._tokenize(question) if term not in self._generic_terms()]
        if not query_terms:
            return True

        text = str(chunk.get("text") or "")
        text_terms = set(self._tokenize(text))
        overlap = len(set(query_terms).intersection(text_terms))
        if overlap >= DEFAULT_MIN_KEYWORD_OVERLAP:
            return True

        # Chroma can return a semantically close chunk with sparse lexical overlap.
        # Keep it only when the vector score is very strong and at least one focused term matches.
        vector_score = float(chunk.get("score") or 0.0)
        return vector_score >= 0.88 and overlap >= 1

    def _required_term_group(self, question: str) -> tuple[str, ...]:
        normalized = question.lower()
        for group in SENSITIVE_QUERY_TERM_GROUPS:
            if self._contains_any(normalized, group):
                return group
        return ()

    def _missing_required_topic(self, chunks: list[dict[str, Any]]) -> bool:
        required_group = getattr(self, "_required_group", ())
        if not required_group:
            return False
        combined_text = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
        return not self._contains_any(combined_text, required_group)

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        normalized = text.lower()
        return any(term in normalized for term in terms)

    def _with_retrieval_source(
        self,
        chunks: list[dict[str, Any]],
        retrieval_source: str,
    ) -> list[dict[str, Any]]:
        tagged_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            tagged_chunk = dict(chunk)
            tagged_chunk["text"] = self._truncate_text(str(tagged_chunk.get("text") or ""))
            metadata = dict(tagged_chunk.get("metadata") or {})
            metadata["retrieval_source"] = retrieval_source
            tagged_chunk["retrieval_source"] = retrieval_source
            tagged_chunk["metadata"] = metadata
            tagged_chunks.append(tagged_chunk)
        return tagged_chunks

    def _log_retrieval_source(self, retrieval_source: str, chunks: list[dict[str, Any]]) -> None:
        LOGGER.info(
            "retrieval_source=%s result_count=%s request_id=%s",
            retrieval_source,
            len(chunks),
            getattr(self, "_request_id", ""),
        )

    def _log_rerank_candidates(self, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks[:10]:
            LOGGER.info(
                "Rerank candidate: request_id=%s chunk_id=%s score=%s distance=%s rerank_score=%s text_length=%s",
                getattr(self, "_request_id", ""),
                chunk.get("chunk_id"),
                chunk.get("score"),
                chunk.get("distance"),
                chunk.get("rerank_score"),
                len(str(chunk.get("text") or "")),
            )

    def _log_filter_decision(self, chunks: list[dict[str, Any]], decision: str, reason: str) -> None:
        for chunk in chunks[:10]:
            LOGGER.info(
                "Retrieval filter decision: request_id=%s chunk_id=%s decision=%s reason=%s score=%s distance=%s rerank_score=%s",
                getattr(self, "_request_id", ""),
                chunk.get("chunk_id"),
                decision,
                reason,
                chunk.get("score"),
                chunk.get("distance"),
                chunk.get("rerank_score"),
            )

    def _log_stage(
        self,
        stage_name: str,
        event: str,
        *,
        question: str,
        document_id: str,
        elapsed_time: float = 0.0,
        retrieval_source: str = "",
        source_chunks_count: int = 0,
    ) -> None:
        LOGGER.info(
            "QA stage %s: request_id=%s document_id=%s question=%s stage_name=%s elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
            event,
            getattr(self, "_request_id", ""),
            document_id,
            question,
            stage_name,
            elapsed_time,
            retrieval_source,
            source_chunks_count,
        )

    def _truncate_text(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= MAX_SOURCE_CHUNK_TEXT_CHARS:
            return normalized
        start = self._focused_window_start(normalized)
        return normalized[start : start + MAX_SOURCE_CHUNK_TEXT_CHARS].strip()

    def _focused_window_start(self, text: str) -> int:
        question = str(getattr(self, "_current_question", "") or "")
        terms = self._focus_terms(question)
        match_indexes = [text.find(term) for term in terms if term and text.find(term) >= 0]
        if not match_indexes:
            return 0
        return max(0, min(match_indexes) - 80)

    def _focus_terms(self, question: str) -> list[str]:
        terms = [term for term in self._tokenize(question) if term not in self._generic_terms()]
        required_group = self._required_term_group(question)
        return list(required_group) + terms

    def _generic_terms(self) -> set[str]:
        return {
            "设备",
            "需要",
            "什么",
            "哪些",
            "如何",
            "怎么",
            "应该",
            "分别",
            "进行",
            "包括",
            "多少",
            "是多少",
            "the",
            "what",
            "which",
            "how",
            "are",
            "is",
        }


def retrieve_top_k(
    *,
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return BasicRetriever().retrieve(question=question, chunks=chunks, top_k=top_k)
