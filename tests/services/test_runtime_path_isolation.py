from __future__ import annotations

import os
from pathlib import Path

from app.config import BASE_DIR
from app.rag.vector_store import ChromaVectorStore, HashEmbeddingFunction
from app.services.document_qa_service import DocumentQAService
from app.services.document_storage_service import DocumentStorageService
from app.services.qa_log_service import QALogService


def test_default_runtime_paths_are_redirected_to_pytest_tmp(
    isolated_runtime_paths: dict[str, Path],
) -> None:
    project_data_db = BASE_DIR / "data" / "documents.db"
    project_upload_dir = BASE_DIR / "uploads"
    project_chroma_dir = BASE_DIR / "chroma_db"
    project_log_dir = BASE_DIR / "logs"

    storage = DocumentStorageService()
    service = DocumentQAService()
    logs = QALogService()
    vector_store = ChromaVectorStore(embedding_function=HashEmbeddingFunction())

    assert storage.db_path == isolated_runtime_paths["document_db_path"]
    assert service.storage_dir == isolated_runtime_paths["upload_dir"]
    assert logs.log_path == isolated_runtime_paths["log_dir"] / "qa_logs.jsonl"
    assert vector_store.persist_directory == isolated_runtime_paths["chroma_dir"]

    assert storage.db_path != project_data_db
    assert service.storage_dir != project_upload_dir
    assert vector_store.persist_directory != project_chroma_dir
    assert logs.log_path.parent != project_log_dir

    storage.upsert_document(
        document_id="test-isolation-doc",
        original_file_name="isolation.txt",
        stored_file_name="test-isolation-doc_isolation.txt",
        file_path=str(service.storage_dir / "test-isolation-doc_isolation.txt"),
        file_type="txt",
    )
    assert isolated_runtime_paths["document_db_path"].exists()
    assert Path(os.environ["DOCUMENT_DB_PATH"]).exists()
