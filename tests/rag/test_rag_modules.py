from pathlib import Path

from app.agents.answer_agent import AnswerAgent
from app.rag.generation import NO_RETRIEVAL_CONTENT_MESSAGE, build_document_qa_prompt
from app.rag.ingestion import load_document, split_text
from app.rag.retrieval import BasicRetriever, RuleBasedReranker, retrieve_top_k


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_document_loader_loads_text_file() -> None:
    result = load_document(str(DATA_DIR / "project_notes.txt"))

    assert result["status"] == "success"
    assert result["file_type"] == "txt"
    assert "Project Notes" in result["text"]


def test_document_loader_returns_error_for_unsupported_file() -> None:
    result = load_document(str(DATA_DIR / "raw_payload.json"))

    assert result["status"] == "error"
    assert "supported" in result["error_message"]


def test_text_splitter_adds_document_metadata() -> None:
    chunks = split_text(
        document_id="doc-1",
        file_name="policy.txt",
        text="abcdef" * 100,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks
    assert chunks[0]["document_id"] == "doc-1"
    assert chunks[0]["file_name"] == "policy.txt"
    assert chunks[0]["chunk_id"] == "chunk-1"
    assert chunks[0]["text"]


def test_retriever_returns_matching_top_k_chunks() -> None:
    chunks = [
        {"chunk_id": "a", "text": "Access reviews happen quarterly."},
        {"chunk_id": "b", "text": "Lunch menu is published weekly."},
    ]

    results = retrieve_top_k(question="access reviews", chunks=chunks, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"
    assert results[0]["score"] > 0
    assert results[0]["retrieval_source"] == "fallback"
    assert results[0]["metadata"]["retrieval_source"] == "fallback"


def test_retriever_marks_chroma_source_when_vector_results_match() -> None:
    class FakeVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "chunk_id": "chunk-1",
                    "text": "Access reviews are required every quarter.",
                    "start_index": 0,
                    "end_index": 44,
                    "score": 0.9,
                    "distance": 0.1,
                }
            ]

    retriever = BasicRetriever(vector_store=FakeVectorStore())  # type: ignore[arg-type]
    result = retriever.retrieve_with_source(
        question="access reviews",
        chunks=[{"document_id": "doc-1", "chunk_id": "chunk-1", "text": "Access reviews."}],
        top_k=1,
    )

    assert result["retrieval_source"] == "chroma"
    assert result["chunks"][0]["retrieval_source"] == "chroma"
    assert result["chunks"][0]["metadata"]["retrieval_source"] == "chroma"
    assert result["chunks"][0]["rerank_score"] > 0


def test_retriever_reranks_chroma_candidates_and_uses_default_candidate_count() -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.top_k = 0

        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            self.top_k = top_k
            return [
                {
                    "document_id": document_id,
                    "chunk_id": "low-lexical-match",
                    "text": "Completely unrelated operational note.",
                    "start_index": 0,
                    "end_index": 38,
                    "score": 0.95,
                    "distance": 0.1,
                },
                {
                    "document_id": document_id,
                    "chunk_id": "strong-lexical-match",
                    "text": "Access reviews are required every quarter.",
                    "start_index": 39,
                    "end_index": 82,
                    "score": 0.3,
                    "distance": 0.9,
                },
            ]

    vector_store = FakeVectorStore()
    result = BasicRetriever(vector_store=vector_store).retrieve_with_source(
        question="access reviews",
        chunks=[{"document_id": "doc-1", "chunk_id": "chunk-1", "text": "Access reviews."}],
        top_k=5,
    )

    assert vector_store.top_k == 10
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["chunk_id"] == "strong-lexical-match"
    assert result["chunks"][0]["rerank_score"] >= 0.6


def test_retriever_limits_reranked_chroma_results_to_top_n() -> None:
    class FakeVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "chunk_id": f"chunk-{index}",
                    "text": f"Access reviews candidate {index}.",
                    "start_index": index,
                    "end_index": index + 20,
                    "score": 1.0 / index,
                    "distance": 0.1,
                }
                for index in range(1, 8)
            ]

    result = BasicRetriever(vector_store=FakeVectorStore()).retrieve_with_source(
        question="access reviews",
        chunks=[{"document_id": "doc-1", "chunk_id": "chunk-1", "text": "Access reviews."}],
        top_k=10,
    )

    assert result["retrieval_source"] == "chroma"
    assert len(result["chunks"]) == 3
    assert all("rerank_score" in chunk for chunk in result["chunks"])


def test_rule_based_reranker_places_relevant_chunk_first() -> None:
    chunks = [
        {
            "chunk_id": "unrelated",
            "text": "Lunch menu and office holiday schedule are published weekly.",
            "score": 0.95,
            "distance": 0.05,
        },
        {
            "chunk_id": "relevant",
            "text": "Access reviews are required every quarter and owners approve exceptions.",
            "score": 0.4,
            "distance": 0.7,
        },
    ]

    results = RuleBasedReranker().rerank(question="How often are access reviews required?", chunks=chunks, top_n=2)

    assert results[0]["chunk_id"] == "relevant"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_rule_based_reranker_penalizes_long_weakly_related_chunk() -> None:
    long_noise = " ".join(
        ["access"] + [f"unrelated operational appendix {index}" for index in range(220)]
    )
    chunks = [
        {
            "chunk_id": "long-weak",
            "text": long_noise,
            "score": 0.9,
            "distance": 0.05,
        },
        {
            "chunk_id": "short-strong",
            "text": "Access reviews are required every quarter.",
            "score": 0.45,
            "distance": 0.5,
        },
    ]

    results = RuleBasedReranker().rerank(question="access reviews every quarter", chunks=chunks, top_n=2)

    assert results[0]["chunk_id"] == "short-strong"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_rule_based_reranker_keeps_source_fields_and_score_order() -> None:
    chunks = [
        {
            "chunk_id": "best",
            "text": "Emergency stop checks must be completed before startup.",
            "score": 0.6,
            "distance": 0.2,
            "retrieval_source": "chroma",
        },
        {
            "chunk_id": "second",
            "text": "Startup preparation includes reviewing the shift handover.",
            "score": 0.5,
            "distance": 0.4,
            "retrieval_source": "chroma",
        },
        {
            "chunk_id": "third",
            "text": "Cafeteria service is available at lunch.",
            "score": 0.95,
            "distance": 0.05,
            "retrieval_source": "chroma",
        },
    ]

    results = RuleBasedReranker().rerank(question="emergency stop before startup", chunks=chunks, top_n=2)

    assert len(results) == 2
    assert [chunk["rerank_score"] for chunk in results] == sorted(
        [chunk["rerank_score"] for chunk in results],
        reverse=True,
    )
    assert results[0]["chunk_id"] == "best"
    assert results[0]["retrieval_source"] == "chroma"
    assert results[0]["score"] == 0.6
    assert results[0]["distance"] == 0.2


def test_retriever_uses_original_chroma_order_when_rerank_fails() -> None:
    class FakeVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "chunk_id": "first",
                    "text": "Access reviews first candidate.",
                    "start_index": 0,
                    "end_index": 31,
                    "score": 0.8,
                    "distance": 0.2,
                },
                {
                    "document_id": document_id,
                    "chunk_id": "second",
                    "text": "Access reviews second candidate.",
                    "start_index": 32,
                    "end_index": 64,
                    "score": 0.7,
                    "distance": 0.3,
                },
            ]

    class FailingReranker:
        def rerank(self, *, question: str, chunks: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
            raise RuntimeError("rerank failed")

    result = BasicRetriever(
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        reranker=FailingReranker(),  # type: ignore[arg-type]
    ).retrieve_with_source(
        question="access reviews",
        chunks=[{"document_id": "doc-1", "chunk_id": "chunk-1", "text": "Access reviews."}],
        top_k=5,
    )

    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["first", "second"]
    assert result["chunks"][0]["retrieval_source"] == "chroma"


def test_retriever_does_not_fallback_when_chroma_returns_low_relevance_candidates() -> None:
    class FakeVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return [
                {
                    "document_id": document_id,
                    "chunk_id": "semantic-only",
                    "text": "Inspect power, emergency stop, and guard door before activation.",
                    "start_index": 0,
                    "end_index": 63,
                    "score": 0.2,
                    "distance": 0.9,
                }
            ]

    result = BasicRetriever(vector_store=FakeVectorStore()).retrieve_with_source(  # type: ignore[arg-type]
        question="startup checklist",
        chunks=[
            {
                "document_id": "doc-1",
                "chunk_id": "fallback-match",
                "text": "startup checklist fallback text should not override chroma candidates",
            }
        ],
        top_k=3,
    )

    assert result["retrieval_source"] == "none"
    assert result["chunks"] == []
    assert result["fallback_reason"] == "chroma_results_below_relevance_threshold"


def test_retriever_marks_fallback_source_when_chroma_has_no_results() -> None:
    class EmptyVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return []

    retriever = BasicRetriever(vector_store=EmptyVectorStore())  # type: ignore[arg-type]
    result = retriever.retrieve_with_source(
        question="access reviews",
        chunks=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "Access reviews are required every quarter.",
            }
        ],
        top_k=1,
    )

    assert result["retrieval_source"] == "fallback"
    assert result["chunks"][0]["retrieval_source"] == "fallback"
    assert result["chunks"][0]["metadata"]["retrieval_source"] == "fallback"


def test_retriever_marks_fallback_source_when_chroma_raises() -> None:
    class FailingVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            raise RuntimeError("chroma unavailable")

    retriever = BasicRetriever(vector_store=FailingVectorStore())  # type: ignore[arg-type]
    result = retriever.retrieve_with_source(
        question="access reviews",
        chunks=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "Access reviews are required every quarter.",
            }
        ],
        top_k=1,
    )

    assert result["retrieval_source"] == "fallback"
    assert "chroma_query_error" in result["fallback_reason"]
    assert result["chunks"][0]["retrieval_source"] == "fallback"
    assert result["chunks"][0]["metadata"]["retrieval_source"] == "fallback"


def test_fallback_relevance_filter_rejects_unanswered_price_question() -> None:
    class FailingVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            raise RuntimeError("chroma unavailable")

    result = BasicRetriever(vector_store=FailingVectorStore()).retrieve_with_source(  # type: ignore[arg-type]
        question="设备采购价格是多少？",
        chunks=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "设备启动前需要检查环境、电源、急停按钮和防护门。",
            }
        ],
        top_k=3,
    )

    assert result["retrieval_source"] == "none"
    assert result["chunks"] == []
    assert "chroma_query_error" in result["fallback_reason"]


def test_price_question_returns_no_answer_without_unrelated_device_content() -> None:
    class EmptyVectorStore:
        def search(self, *, document_id: str, question: str, top_k: int) -> list[dict[str, object]]:
            return []

    retrieval_result = BasicRetriever(vector_store=EmptyVectorStore()).retrieve_with_source(  # type: ignore[arg-type]
        question="设备采购价格是多少？",
        chunks=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "设备启动前需要进行外观检查，确认管路接口正常。运行维护和停机流程另行执行。",
            }
        ],
        top_k=3,
    )
    answer = AnswerAgent().generate_answer(
        question="设备采购价格是多少？",
        context_chunks=list(retrieval_result["chunks"]),
    )["answer"]

    assert retrieval_result["retrieval_source"] == "none"
    assert answer == NO_RETRIEVAL_CONTENT_MESSAGE
    assert "外观检查" not in answer
    assert "管路接口" not in answer
    assert "运行维护" not in answer
    assert "停机流程" not in answer


def test_startup_question_answers_only_startup_checks() -> None:
    answer = AnswerAgent().generate_answer(
        question="设备启动前需要检查什么？",
        context_chunks=[
            {
                "retrieval_source": "chroma",
                "text": (
                    "设备启动前需要检查电源、急停按钮、防护门和现场环境。"
                    "运行维护应定期清洁过滤器。"
                    "停机流程需要先卸载压力。"
                ),
            }
        ],
    )["answer"]

    assert "启动前" in answer
    assert "电源" in answer
    assert "急停按钮" in answer
    assert "运行维护" not in answer
    assert "停机流程" not in answer


def test_retriever_marks_none_source_when_no_chunks_match() -> None:
    result = BasicRetriever(vector_store=None).retrieve_with_source(
        question="security",
        chunks=[{"chunk_id": "a", "text": "Lunch menu is published weekly."}],
        top_k=1,
    )

    assert result["retrieval_source"] == "none"
    assert result["chunks"] == []


def test_retriever_returns_empty_when_no_match() -> None:
    results = retrieve_top_k(
        question="security",
        chunks=[{"chunk_id": "a", "text": "Lunch menu is published weekly."}],
    )

    assert results == []


def test_prompt_builder_handles_empty_retrieval() -> None:
    prompt = build_document_qa_prompt(question="What is required?", retrieved_chunks=[])

    assert prompt == NO_RETRIEVAL_CONTENT_MESSAGE


def test_prompt_builder_constrains_answer_to_retrieved_chunks() -> None:
    prompt = build_document_qa_prompt(
        question="设备启动前需要检查什么？",
        retrieved_chunks=[
            {
                "file_name": "device.txt",
                "chunk_id": "chunk-1",
                "text": "设备启动前需要检查环境、电源、急停按钮和防护门。",
            }
        ],
    )

    assert "只能根据【检索片段】回答用户问题" in prompt
    assert "不要总结全文" in prompt
    assert "不要输出与问题无关的文档内容" in prompt
    assert NO_RETRIEVAL_CONTENT_MESSAGE in prompt


def test_prompt_builder_limits_chunks_and_does_not_include_full_parsed_text() -> None:
    huge_text = "相关片段。" + ("A" * 900) + "FULL_PARSED_TEXT_MARKER"
    prompt = build_document_qa_prompt(
        question="设备启动前需要检查什么？",
        retrieved_chunks=[
            {"file_name": "device.txt", "chunk_id": f"chunk-{index}", "text": huge_text}
            for index in range(1, 6)
        ],
    )

    assert "chunk-1" in prompt
    assert "chunk-3" in prompt
    assert "chunk-4" not in prompt
    assert "FULL_PARSED_TEXT_MARKER" not in prompt
