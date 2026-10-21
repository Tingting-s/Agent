from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from app.config import get_chroma_dir, get_settings


DEFAULT_CHROMA_DIR = get_chroma_dir()
DEFAULT_COLLECTION_NAME = "enterprise_documents"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-zh-v1.5"
CHROMADB_RUNTIME_CHECK_TIMEOUT_SECONDS = 5
LOGGER = logging.getLogger(__name__)
_CHROMADB_RUNTIME_STATUS: tuple[bool, str] | None = None


class VectorStoreError(RuntimeError):
    """Raised when the local vector store is unavailable or fails."""


class HashEmbeddingFunction:
    """Small deterministic embedding function for local Chroma demos.

    This avoids downloading external models during development. It is not a
    semantic embedding replacement for bge-m3 or sentence-transformers, but it
    provides a stable local vector representation for Chroma persistence and
    retrieval while keeping the system self-contained.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma expects this parameter name.
        return self.embed_documents(input)

    def name(self) -> str:
        return f"hash-embedding:{self.dimension}"

    def preload(self) -> None:
        LOGGER.info("Reusing embedding model: model=%s device=%s", self.name(), "local-hash")

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            return self._embed_text(input)
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [self._embed_text(str(text or "")) for text in input]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = text.lower().strip()
        if not normalized:
            return vector

        tokens = self._tokens(normalized)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], byteorder="little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        words = [token for token in text.replace("\n", " ").split(" ") if token]
        cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
        return words + cjk_chars


class SentenceTransformerEmbeddingFunction:
    """Sentence-transformers embedding function with hash fallback.

    The same callable is passed to Chroma for upsert and query, so uploaded
    chunks and questions use the same embedding model when sentence-transformers
    is available. If the package or model cannot be loaded, the deterministic
    hash embedding keeps local development and tests usable.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        fallback_embedding: HashEmbeddingFunction | None = None,
        model_factory: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        self.device = device if device is not None else settings.EMBEDDING_DEVICE.strip()
        self.local_files_only = settings.EMBEDDING_LOCAL_FILES_ONLY
        self.fallback_embedding = fallback_embedding or HashEmbeddingFunction()
        self._model_factory = model_factory
        self._model: Any | None = None
        self._fallback_logged = False

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma expects this parameter name.
        return self.embed_documents(input)

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            vectors = self._embed_texts([input])
            return vectors[0] if vectors else []
        return self._embed_texts(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed_texts(input)

    def preload(self) -> Any:
        return self._get_model()

    def _embed_texts(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        texts = [str(text or "") for text in input]
        if not texts:
            return []

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return self._to_vectors(embeddings)
        except Exception as exc:
            self._log_fallback(exc)
            return self.fallback_embedding(texts)

    def _get_model(self) -> Any:
        if self._model is not None:
            LOGGER.info(
                "Reusing embedding model: model=%s device=%s",
                self.model_name,
                self.device or "auto",
            )
            return self._model

        if self._model_factory is not None:
            self._model = self._model_factory(self.model_name, self.device)
            return self._model

        from sentence_transformers import SentenceTransformer

        device = self.device
        if not device or device.lower() == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            local_files_only=self.local_files_only,
        )
        self.device = device
        LOGGER.info(
            "SentenceTransformer embedding model loaded: model=%s device=%s local_files_only=%s",
            self.model_name,
            self.device,
            self.local_files_only,
        )
        return self._model




    def _to_vectors(self, embeddings: Any) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in vector] for vector in embeddings]

    def _log_fallback(self, exc: Exception) -> None:
        if self._fallback_logged:
            return
        self._fallback_logged = True
        LOGGER.warning(
            "SentenceTransformer embedding unavailable; falling back to hash embedding: model=%s error_type=%s error=%s",
            self.model_name,
            type(exc).__name__,
            exc,
            exc_info=True,
        )


class ChromaVectorStore:
    def __init__(
        self,
        *,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any | None = None,
        write_lock_timeout_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.persist_directory = Path(persist_directory or get_chroma_dir())
        self.collection_name = collection_name
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction()
        self.write_lock_timeout_seconds = max(
            int(
                write_lock_timeout_seconds
                if write_lock_timeout_seconds is not None
                else settings.CHROMA_WRITE_LOCK_TIMEOUT_SECONDS
            ),
            1,
        )
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self._get_collection()
        return self._collection

    def add_chunks(self, *, document_id: str, file_name: str, chunks: list[dict[str, Any]]) -> int:
        usable_chunks = [
            chunk
            for chunk in chunks
            if str(chunk.get("text") or "").strip() and str(chunk.get("chunk_id") or "").strip()
        ]
        if not document_id or not usable_chunks:
            return 0

        ids = [self._chunk_uid(document_id, str(chunk["chunk_id"])) for chunk in usable_chunks]
        documents = [str(chunk.get("text") or "") for chunk in usable_chunks]
        metadatas = [
            self._metadata(document_id=document_id, file_name=file_name, chunk=chunk)
            for chunk in usable_chunks
        ]
        collection = self.collection
        with self._write_lock():
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            document_chunk_count = self.get_document_chunk_count(document_id)
            collection_total_count = self._collection_total_count()
        LOGGER.info(
            "Chroma upsert completed: document_id=%s file_name=%s chunk_count=%s document_chunk_count=%s collection_total_count=%s collection=%s",
            document_id,
            file_name,
            len(usable_chunks),
            document_chunk_count,
            collection_total_count,
            self.collection_name,
        )
        return len(usable_chunks)

    def search(self, *, document_id: str, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        LOGGER.info(
            "vector_store.search started: document_id=%s top_k=%s collection=%s",
            document_id,
            top_k,
            self.collection_name,
        )
        if not document_id or not question.strip():
            return []

        LOGGER.info(
            "Chroma query started: document_id=%s question=%s top_k=%s collection=%s where=%s",
            document_id,
            question,
            top_k,
            self.collection_name,
            {"document_id": document_id},
        )
        try:
            result = self.collection.query(
                query_texts=[question],
                n_results=max(1, top_k),
                where={"document_id": document_id},
            )
        except Exception:
            LOGGER.exception(
                "Chroma query failed: document_id=%s top_k=%s collection=%s",
                document_id,
                top_k,
                self.collection_name,
            )
            raise
        chunks = self._format_query_result(result=result, document_id=document_id, top_k=top_k)
        raw_count = len((result.get("documents") or [[]])[0] or []) if isinstance(result, dict) else 0
        LOGGER.info(
            "Chroma query completed: document_id=%s result_count=%s formatted_count=%s collection=%s where=%s",
            document_id,
            raw_count,
            len(chunks),
            self.collection_name,
            {"document_id": document_id},
        )
        for chunk in chunks[:10]:
            LOGGER.info(
                "Chroma query candidate: document_id=%s chunk_id=%s score=%s distance=%s text_length=%s",
                chunk.get("document_id"),
                chunk.get("chunk_id"),
                chunk.get("score"),
                chunk.get("distance"),
                len(str(chunk.get("text") or "")),
            )
        if raw_count == 0:
            indexed_count = self.get_document_chunk_count(document_id)
            LOGGER.info(
                "Chroma query returned zero results: document_id=%s chunk_count=%s collection=%s",
                document_id,
                indexed_count,
                self.collection_name,
            )
        return chunks

    def delete_document(self, document_id: str) -> int:
        if not document_id:
            return 0
        collection = self.collection
        with self._write_lock():
            before_count = self.get_document_chunk_count(document_id)
            collection.delete(where={"document_id": document_id})
            after_count = self.get_document_chunk_count(document_id)
        deleted_count = max(before_count - after_count, 0)
        LOGGER.info(
            "Chroma document vectors deleted: document_id=%s deleted_count=%s collection=%s",
            document_id,
            deleted_count,
            self.collection_name,
        )
        return deleted_count

    def get_document_chunk_count(self, document_id: str) -> int:
        if not document_id:
            return 0
        result = self.collection.get(where={"document_id": document_id}, include=[])
        return len(result.get("ids") or [])

    def _collection_total_count(self) -> int | None:
        count = getattr(self.collection, "count", None)
        if callable(count):
            try:
                return int(count())
            except Exception:
                return None
        items = getattr(self.collection, "_items", None)
        if isinstance(items, dict):
            return len(items)
        try:
            result = self.collection.get(include=[])
            return len(result.get("ids") or [])
        except Exception:
            return None

    def _get_collection(self) -> Any:
        persist_directory_exists = self.persist_directory.exists()
        parent_exists = self.persist_directory.parent.exists()
        LOGGER.info(
            "Chroma initialization requested: persist_directory=%s exists=%s parent=%s parent_exists=%s collection=%s",
            self.persist_directory,
            persist_directory_exists,
            self.persist_directory.parent,
            parent_exists,
            self.collection_name,
        )
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.exception(
                "Failed to prepare Chroma directory: persist_directory=%s parent=%s parent_exists=%s error_type=%s error=%s",
                self.persist_directory,
                self.persist_directory.parent,
                parent_exists,
                type(exc).__name__,
                exc,
            )
            raise VectorStoreError(
                f"Failed to prepare Chroma directory {self.persist_directory}: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            chromadb = self._import_chromadb()
        except Exception as exc:
            LOGGER.warning(
                "chromadb package import failed; using in-memory LocalVectorCollection: "
                "persist_directory=%s collection=%s error_type=%s error=%s",
                self.persist_directory,
                self.collection_name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return LocalVectorCollection(embedding_function=self.embedding_function)

        try:
            with self._write_lock():
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
                collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
            LOGGER.info(
                "Chroma initialized: persist_directory=%s exists=%s collection=%s chromadb_version=%s",
                self.persist_directory,
                self.persist_directory.exists(),
                self.collection_name,
                getattr(chromadb, "__version__", "unknown"),
            )
            return collection
        except Exception as exc:  # pragma: no cover - exact Chroma errors differ by version.
            LOGGER.exception(
                "Chroma initialization failed: persist_directory=%s exists=%s is_dir=%s collection=%s chromadb_version=%s error_type=%s error=%s",
                self.persist_directory,
                self.persist_directory.exists(),
                self.persist_directory.is_dir(),
                self.collection_name,
                getattr(chromadb, "__version__", "unknown"),
                type(exc).__name__,
                exc,
            )
            raise VectorStoreError(self._initialization_error_message(exc)) from exc

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Serialize local Chroma mutations across API and worker processes."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.persist_directory / ".chroma-write-lock.sqlite3"
        timeout = self.write_lock_timeout_seconds
        connection = sqlite3.connect(
            str(lock_path),
            timeout=timeout,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _import_chromadb(self) -> Any:
        self._validate_chromadb_runtime()
        import chromadb

        return chromadb

    def _initialization_error_message(self, exc: Exception) -> str:
        message = (
            f"Failed to initialize Chroma vector store at {self.persist_directory}: "
            f"{type(exc).__name__}: {exc}"
        )
        if self._looks_like_incompatible_chroma_store(exc):
            return (
                f"{message}. Existing chroma_db data appears incompatible with the installed "
                "chromadb version. Back up or remove chroma_db, then reindex uploaded documents."
            )
        return message

    def _looks_like_incompatible_chroma_store(self, exc: Exception) -> bool:
        return isinstance(exc, KeyError) and str(exc).strip("\"'") == "_type"

    def _validate_chromadb_runtime(self) -> None:
        global _CHROMADB_RUNTIME_STATUS

        if _CHROMADB_RUNTIME_STATUS is not None:
            is_usable, message = _CHROMADB_RUNTIME_STATUS
            if is_usable:
                return
            raise ImportError(message)

        if importlib.util.find_spec("chromadb") is None:
            message = "chromadb package is not installed."
            _CHROMADB_RUNTIME_STATUS = (False, message)
            raise ImportError(message)

        command = [
            sys.executable,
            "-c",
            "import chromadb, sys; sys.stdout.write(getattr(chromadb, '__version__', 'unknown'))",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=CHROMADB_RUNTIME_CHECK_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as exc:
            message = f"chromadb runtime check failed before in-process import: {type(exc).__name__}: {exc}"
            _CHROMADB_RUNTIME_STATUS = (False, message)
            raise ImportError(message) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"process exited with code {result.returncode}"
            message = f"chromadb runtime check failed before in-process import: {details}"
            _CHROMADB_RUNTIME_STATUS = (False, message)
            raise ImportError(message)

        _CHROMADB_RUNTIME_STATUS = (True, (result.stdout or "").strip())

    def _format_query_result(
        self,
        *,
        result: dict[str, Any],
        document_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []

        chunks: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            if metadata.get("document_id") != document_id:
                continue
            distance = float(distances[index]) if index < len(distances) and distances[index] is not None else 0.0
            chunk = {
                "document_id": str(metadata.get("document_id") or document_id),
                "file_name": str(metadata.get("file_name") or ""),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "text": str(text or ""),
                "start_index": int(metadata.get("start_index") or 0),
                "end_index": int(metadata.get("end_index") or 0),
                "score": round(1.0 / (1.0 + max(distance, 0.0)), 6),
                "distance": distance,
                "retrieval_source": "chroma",
                "metadata": {**metadata, "retrieval_source": "chroma"},
            }
            chunks.append(chunk)
        return chunks[:top_k]

    def _metadata(self, *, document_id: str, file_name: str, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "file_name": str(chunk.get("file_name") or file_name),
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "start_index": int(chunk.get("start_index") or 0),
            "end_index": int(chunk.get("end_index") or 0),
        }

    def _chunk_uid(self, document_id: str, chunk_id: str) -> str:
        return f"{document_id}_{chunk_id}"


class LocalVectorCollection:
    """Minimal collection fallback used when chromadb is unavailable.

    Runtime retrieval still falls back to the keyword retriever if Chroma fails.
    This object keeps local development and unit tests usable in environments
    where the Chroma package cannot be installed immediately.
    """

    def __init__(self, *, embedding_function: Any) -> None:
        self.embedding_function = embedding_function
        self._items: dict[str, dict[str, Any]] = {}

    def upsert(self, *, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        embeddings = self.embedding_function(documents)
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=False):
            self._items[item_id] = {
                "document": document,
                "metadata": dict(metadata),
                "embedding": embedding,
            }

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_text = query_texts[0] if query_texts else ""
        query_embedding = self.embedding_function([query_text])[0]
        document_id = (where or {}).get("document_id")
        candidates = [
            item
            for item in self._items.values()
            if not document_id or item["metadata"].get("document_id") == document_id
        ]
        scored = [
            (self._cosine_distance(query_embedding, item["embedding"]), item)
            for item in candidates
        ]
        scored.sort(key=lambda value: value[0])
        selected = scored[:n_results]
        return {
            "documents": [[item["document"] for _, item in selected]],
            "metadatas": [[item["metadata"] for _, item in selected]],
            "distances": [[distance for distance, _ in selected]],
        }

    def delete(self, *, where: dict[str, Any]) -> None:
        document_id = where.get("document_id")
        if not document_id:
            return
        for item_id in [
            item_id
            for item_id, item in self._items.items()
            if item["metadata"].get("document_id") == document_id
        ]:
            del self._items[item_id]

    def get(self, *, where: dict[str, Any], include: list[str] | None = None) -> dict[str, Any]:
        document_id = where.get("document_id")
        ids = [
            item_id
            for item_id, item in self._items.items()
            if not document_id or item["metadata"].get("document_id") == document_id
        ]
        return {"ids": ids}

    def _cosine_distance(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0 or right_norm <= 0:
            return 1.0
        return 1.0 - (dot / (left_norm * right_norm))


_DEFAULT_VECTOR_STORE: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _DEFAULT_VECTOR_STORE
    if _DEFAULT_VECTOR_STORE is None:
        _DEFAULT_VECTOR_STORE = ChromaVectorStore()
    return _DEFAULT_VECTOR_STORE


def preload_embedding_model() -> bool:
    settings = get_settings()
    if not settings.EMBEDDING_PRELOAD:
        LOGGER.info("embedding model preloading skipped: EMBEDDING_PRELOAD=false")
        return False

    start = perf_counter()
    vector_store = get_vector_store()
    embedding_function = vector_store.embedding_function
    model_name = getattr(embedding_function, "model_name", None)
    if not model_name and hasattr(embedding_function, "name"):
        model_name = embedding_function.name()
    device = getattr(embedding_function, "device", "") or "auto"
    LOGGER.info(
        "embedding model preloading started: embedding_model_name=%s device=%s",
        model_name or "unknown",
        device,
    )
    try:
        preload = getattr(embedding_function, "preload", None)
        if callable(preload):
            preload()
        else:
            embedding_function(["preload"])
        elapsed_time = round(perf_counter() - start, 6)
        LOGGER.info(
            "embedding model preloaded successfully: embedding_model_name=%s device=%s preload_elapsed_time=%s",
            model_name or "unknown",
            getattr(embedding_function, "device", "") or device,
            elapsed_time,
        )
        return True
    except Exception as exc:
        elapsed_time = round(perf_counter() - start, 6)
        LOGGER.warning(
            "embedding preload failed: embedding_model_name=%s device=%s preload_elapsed_time=%s error_type=%s error=%s",
            model_name or "unknown",
            device,
            elapsed_time,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return False
