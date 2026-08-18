from typing import Any

from app.rag.retrieval import BasicRetriever


class RetrievalAgent:
    """Document retrieval using Chroma first and keyword matching as fallback."""

    def __init__(self) -> None:
        self.retriever = BasicRetriever()
        self.embedding_model_name: str | None = self.retriever.embedding_model_name
        self.vector_store: Any | None = self.retriever.vector_store

    def retrieve(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(question=question, chunks=chunks, top_k=top_k)

    def retrieve_with_source(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        request_id: str = "",
    ) -> dict[str, Any]:
        return self.retriever.retrieve_with_source(question=question, chunks=chunks, top_k=top_k, request_id=request_id)

    def keyword_retrieve(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.retriever.retrieve(question=question, chunks=chunks, top_k=top_k)

    def vector_retrieve_placeholder(self) -> None:
        """Reserved for bge-m3 + Chroma/FAISS integration."""
        self.retriever.bge_m3_retrieve_placeholder()
