from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from uuid import uuid4

from app.rag.vector_store import (
    ChromaVectorStore,
    HashEmbeddingFunction,
    LocalVectorCollection,
    SentenceTransformerEmbeddingFunction,
)


def _store(tmp_path):
    return ChromaVectorStore(
        persist_directory=tmp_path / "tmp_chroma_db",
        collection_name=f"test_enterprise_documents_{uuid4().hex}",
    )


def test_vector_store_adds_and_searches_document_chunks(tmp_path) -> None:
    store = _store(tmp_path)
    chunks = [
        {
            "document_id": "doc-1",
            "file_name": "policy.txt",
            "chunk_id": "chunk-1",
            "text": "Access reviews are required every quarter.",
            "start_index": 0,
            "end_index": 44,
        }
    ]

    indexed_count = store.add_chunks(document_id="doc-1", file_name="policy.txt", chunks=chunks)
    results = store.search(document_id="doc-1", question="access reviews", top_k=3)

    assert indexed_count == 1
    assert results
    assert results[0]["text"]
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["file_name"] == "policy.txt"


def test_vector_store_filters_by_document_id(tmp_path) -> None:
    store = _store(tmp_path)
    store.add_chunks(
        document_id="doc-1",
        file_name="a.txt",
        chunks=[
            {
                "document_id": "doc-1",
                "file_name": "a.txt",
                "chunk_id": "chunk-1",
                "text": "Access reviews are required every quarter.",
            }
        ],
    )
    store.add_chunks(
        document_id="doc-2",
        file_name="b.txt",
        chunks=[
            {
                "document_id": "doc-2",
                "file_name": "b.txt",
                "chunk_id": "chunk-1",
                "text": "Training attendance is reviewed monthly.",
            }
        ],
    )

    results = store.search(document_id="doc-2", question="access reviews", top_k=5)

    assert results
    assert {result["document_id"] for result in results} == {"doc-2"}
    assert all("Access reviews" not in result["text"] for result in results)


def test_vector_store_metadata_document_id_matches_query_document_id(tmp_path) -> None:
    store = _store(tmp_path)
    store.add_chunks(
        document_id="metadata-doc",
        file_name="manual.txt",
        chunks=[
            {
                "document_id": "metadata-doc",
                "file_name": "manual.txt",
                "chunk_id": "chunk-1",
                "text": "Device startup checks include power and emergency stop.",
                "start_index": 0,
                "end_index": 56,
            }
        ],
    )

    results = store.search(document_id="metadata-doc", question="startup checks", top_k=3)

    assert results
    assert {result["document_id"] for result in results} == {"metadata-doc"}


def test_vector_store_search_supports_chroma_embed_query_interface(tmp_path) -> None:
    class FakeModel:
        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):  # type: ignore[no-untyped-def]
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [[1.0, 0.0] for _ in texts]

    class EmbedQueryCollection:
        def __init__(self, embedding_function) -> None:  # type: ignore[no-untyped-def]
            self.embedding_function = embedding_function
            self.documents: list[str] = []
            self.metadatas: list[dict[str, object]] = []

        def upsert(self, *, ids, documents, metadatas):  # type: ignore[no-untyped-def]
            assert self.embedding_function.embed_documents(documents)
            self.documents = list(documents)
            self.metadatas = [dict(metadata) for metadata in metadatas]

        def query(self, *, query_texts, n_results, where):  # type: ignore[no-untyped-def]
            assert self.embedding_function.embed_query(query_texts[0])
            document_id = where.get("document_id")
            selected = [
                (document, metadata)
                for document, metadata in zip(self.documents, self.metadatas, strict=False)
                if metadata.get("document_id") == document_id
            ][:n_results]
            return {
                "documents": [[document for document, _ in selected]],
                "metadatas": [[metadata for _, metadata in selected]],
                "distances": [[0.1 for _ in selected]],
            }

        def get(self, *, where, include=None):  # type: ignore[no-untyped-def]
            document_id = where.get("document_id")
            return {
                "ids": [
                    str(index)
                    for index, metadata in enumerate(self.metadatas)
                    if metadata.get("document_id") == document_id
                ]
            }

    class FakeClient:
        def __init__(self, *, path: str) -> None:
            self.path = path

        def get_or_create_collection(self, *, name: str, embedding_function, metadata):  # type: ignore[no-untyped-def]
            return EmbedQueryCollection(embedding_function)

    embedding = SentenceTransformerEmbeddingFunction(
        "BAAI/bge-small-zh-v1.5",
        device="cpu",
        model_factory=lambda model_name, device: FakeModel(),
    )
    store = ChromaVectorStore(
        persist_directory=tmp_path / "tmp_chroma_db",
        collection_name=f"test_enterprise_documents_{uuid4().hex}",
        embedding_function=embedding,
    )
    store._import_chromadb = lambda: SimpleNamespace(  # type: ignore[method-assign]
        __version__="test-version",
        PersistentClient=FakeClient,
    )

    store.add_chunks(
        document_id="doc-embed-query",
        file_name="manual.txt",
        chunks=[{"chunk_id": "chunk-1", "text": "Device startup checks include power."}],
    )
    results = store.search(document_id="doc-embed-query", question="startup checks", top_k=5)

    assert results
    assert results[0]["retrieval_source"] == "chroma"
    assert results[0]["metadata"]["document_id"] == "doc-embed-query"


def test_vector_store_counts_and_deletes_document_chunks(tmp_path) -> None:
    store = _store(tmp_path)
    store.add_chunks(
        document_id="doc-1",
        file_name="policy.txt",
        chunks=[
            {"chunk_id": "chunk-1", "text": "First policy chunk."},
            {"chunk_id": "chunk-2", "text": "Second policy chunk."},
        ],
    )

    assert store.get_document_chunk_count("doc-1") == 2

    store.delete_document("doc-1")

    assert store.get_document_chunk_count("doc-1") == 0


def test_sentence_transformer_embedding_uses_model_factory() -> None:
    class FakeModel:
        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):  # type: ignore[no-untyped-def]
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [[float(index), 1.0] for index, _ in enumerate(texts, start=1)]

    calls: list[tuple[str, str | None]] = []

    def model_factory(model_name: str, device: str | None):  # type: ignore[no-untyped-def]
        calls.append((model_name, device))
        return FakeModel()

    embedding = SentenceTransformerEmbeddingFunction(
        "BAAI/bge-small-zh-v1.5",
        device="cpu",
        model_factory=model_factory,
    )

    assert embedding(["hello", "world"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert embedding.embed_documents(["document"]) == [[1.0, 1.0]]
    assert embedding.embed_query("query") == [1.0, 1.0]
    assert embedding.embed_query(["query", "another query"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert calls == [("BAAI/bge-small-zh-v1.5", "cpu")]
    assert embedding.name() == "sentence-transformers:BAAI/bge-small-zh-v1.5"


def test_hash_embedding_exposes_stable_chroma_name() -> None:
    assert HashEmbeddingFunction(dimension=8).name() == "hash-embedding:8"


def test_sentence_transformer_embedding_falls_back_to_hash(caplog) -> None:
    def model_factory(model_name: str, device: str | None):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"cannot load {model_name}")

    embedding = SentenceTransformerEmbeddingFunction(
        "BAAI/bge-small-zh-v1.5",
        fallback_embedding=HashEmbeddingFunction(dimension=8),
        model_factory=model_factory,
    )

    caplog.set_level(logging.WARNING, logger="app.rag.vector_store")
    vectors = embedding(["fallback text"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 8
    assert "falling back to hash embedding" in caplog.text


def test_sentence_transformer_embedding_uses_local_files_only_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, device: str, local_files_only: bool) -> None:
            calls["model_name"] = model_name
            calls["device"] = device
            calls["local_files_only"] = local_files_only

        def encode(self, texts, normalize_embeddings: bool, show_progress_bar: bool):  # type: ignore[no-untyped-def]
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embedding = SentenceTransformerEmbeddingFunction("local-model", device="cpu")

    assert embedding(["hello"]) == [[1.0, 0.0]]
    assert calls == {
        "model_name": "local-model",
        "device": "cpu",
        "local_files_only": True,
    }


def test_vector_store_initializes_chroma_when_package_is_available(tmp_path, caplog) -> None:
    collection = LocalVectorCollection(embedding_function=HashEmbeddingFunction())

    class FakeClient:
        def __init__(self, *, path: str) -> None:
            self.path = path

        def get_or_create_collection(self, *, name: str, embedding_function, metadata):  # type: ignore[no-untyped-def]
            assert name.startswith("test_enterprise_documents_")
            assert metadata == {"hnsw:space": "cosine"}
            assert embedding_function
            return collection

    store = _store(tmp_path)
    store._import_chromadb = lambda: SimpleNamespace(  # type: ignore[method-assign]
        __version__="test-version",
        PersistentClient=FakeClient,
    )

    caplog.set_level(logging.INFO, logger="app.rag.vector_store")

    assert store.collection is collection
    assert (tmp_path / "tmp_chroma_db").exists()
    assert "Chroma initialized" in caplog.text
    assert "test-version" in caplog.text


def test_vector_store_import_failure_uses_local_collection_and_logs_reason(tmp_path, caplog) -> None:
    store = _store(tmp_path)
    store._import_chromadb = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        ValueError("onnxruntime DLL initialization failed")
    )

    caplog.set_level(logging.WARNING, logger="app.rag.vector_store")

    indexed_count = store.add_chunks(
        document_id="doc-import-failure",
        file_name="fallback.txt",
        chunks=[{"chunk_id": "chunk-1", "text": "Fallback collection remains usable."}],
    )
    results = store.search(document_id="doc-import-failure", question="Fallback collection", top_k=1)

    assert indexed_count == 1
    assert results
    assert results[0]["document_id"] == "doc-import-failure"
    assert "chromadb package import failed" in caplog.text
    assert "onnxruntime DLL initialization failed" in caplog.text


def test_vector_store_initialization_failure_logs_concrete_reason(tmp_path, caplog) -> None:
    class FailingClient:
        def __init__(self, *, path: str) -> None:
            raise RuntimeError("permission denied for chroma path")

    store = _store(tmp_path)
    fake_chromadb = SimpleNamespace(
        __version__="test-version",
        PersistentClient=FailingClient,
    )
    store._import_chromadb = lambda: fake_chromadb  # type: ignore[method-assign]

    caplog.set_level(logging.ERROR, logger="app.rag.vector_store")

    try:
        _ = store.collection
    except RuntimeError as exc:
        # Local chroma errors are wrapped as VectorStoreError, which inherits RuntimeError.
        assert "permission denied for chroma path" in str(exc)
    else:  # pragma: no cover - this branch means the test did not exercise the failure path.
        raise AssertionError("Expected Chroma initialization to fail.")

    log_text = caplog.text
    assert "Chroma initialization failed" in log_text
    assert "permission denied for chroma path" in log_text
    assert str(tmp_path / "tmp_chroma_db") in log_text


def test_vector_store_initialization_explains_incompatible_chroma_store(tmp_path) -> None:
    class FailingClient:
        def __init__(self, *, path: str) -> None:
            self.path = path

        def get_or_create_collection(self, *, name: str, embedding_function, metadata):  # type: ignore[no-untyped-def]
            raise KeyError("_type")

    store = _store(tmp_path)
    store._import_chromadb = lambda: SimpleNamespace(  # type: ignore[method-assign]
        __version__="test-version",
        PersistentClient=FailingClient,
    )

    try:
        _ = store.collection
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - this branch means the test did not exercise the failure path.
        raise AssertionError("Expected Chroma initialization to fail.")

    assert "Existing chroma_db data appears incompatible" in message
    assert "reindex uploaded documents" in message
