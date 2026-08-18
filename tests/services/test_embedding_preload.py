from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.rag import vector_store
from app.rag.vector_store import (
    ChromaVectorStore,
    HashEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
    preload_embedding_model,
)


def test_sentence_transformer_preload_reuses_loaded_model() -> None:
    calls: list[tuple[str, str | None]] = []

    def model_factory(model_name: str, device: str | None):  # type: ignore[no-untyped-def]
        calls.append((model_name, device))
        return SimpleNamespace(encode=lambda *args, **kwargs: [[1.0, 0.0]])

    embedding = SentenceTransformerEmbeddingFunction(
        "test-model",
        device="cpu",
        model_factory=model_factory,
    )

    first_model = embedding.preload()
    second_model = embedding.preload()

    assert first_model is second_model
    assert calls == [("test-model", "cpu")]


def test_preload_embedding_model_failure_does_not_raise(monkeypatch) -> None:
    class FailingEmbedding:
        model_name = "failing-model"
        device = "cpu"

        def preload(self) -> None:
            raise RuntimeError("model load failed")

    store = ChromaVectorStore(
        persist_directory="unused",
        embedding_function=FailingEmbedding(),
    )
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: store)

    assert preload_embedding_model() is False


def test_fastapi_startup_survives_embedding_preload_failure(monkeypatch) -> None:
    monkeypatch.setattr("app.main.preload_embedding_model", lambda: False)

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_vector_store_remains_available_after_preload(monkeypatch) -> None:
    store = ChromaVectorStore(
        persist_directory="unused",
        embedding_function=HashEmbeddingFunction(),
    )
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: store)

    assert preload_embedding_model() is True
    assert vector_store.get_vector_store() is store
