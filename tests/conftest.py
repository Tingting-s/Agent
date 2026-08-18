from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SESSION_RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="enterprise_doc_qa_pytest_"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["APP_NAME"] = "enterprise-document-qa-agent"
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DOCUMENT_DB_PATH"] = str(_SESSION_RUNTIME_DIR / "documents_test.db")
os.environ["UPLOAD_DIR"] = str(_SESSION_RUNTIME_DIR / "uploads")
os.environ["CHROMA_DIR"] = str(_SESSION_RUNTIME_DIR / "chroma_db")
os.environ["LOG_DIR"] = str(_SESSION_RUNTIME_DIR / "logs")
os.environ["PYTEST_RUNTIME_DIR"] = str(_SESSION_RUNTIME_DIR)

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class OpenAI:  # pragma: no cover - test bootstrap shim
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    openai_stub.OpenAI = OpenAI
    sys.modules["openai"] = openai_stub


@pytest.fixture(autouse=True)
def isolated_runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    runtime_dir = tmp_path / "runtime"
    paths = {
        "runtime_dir": runtime_dir,
        "document_db_path": runtime_dir / "documents_test.db",
        "upload_dir": runtime_dir / "uploads",
        "chroma_dir": runtime_dir / "chroma_db",
        "log_dir": runtime_dir / "logs",
    }
    for directory in (paths["upload_dir"], paths["chroma_dir"], paths["log_dir"]):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DOCUMENT_DB_PATH", str(paths["document_db_path"]))
    monkeypatch.setenv("UPLOAD_DIR", str(paths["upload_dir"]))
    monkeypatch.setenv("CHROMA_DIR", str(paths["chroma_dir"]))
    monkeypatch.setenv("LOG_DIR", str(paths["log_dir"]))
    monkeypatch.setenv("PYTEST_RUNTIME_DIR", str(runtime_dir))

    from app.api import routes
    from app.config import get_settings
    from app.rag import vector_store
    from app.services.document_qa_service import DocumentQAService
    from app.services.document_storage_service import DocumentStorageService
    from app.services.legacy_task_service import LegacyTaskService
    from app.services.qa_log_service import QALogService

    get_settings.cache_clear()
    vector_store._DEFAULT_VECTOR_STORE = None  # type: ignore[attr-defined]
    log_service = QALogService(log_path=paths["log_dir"] / "qa_logs.jsonl")
    document_service = DocumentQAService(
        storage_dir=paths["upload_dir"],
        log_service=log_service,
        storage_service=DocumentStorageService(paths["document_db_path"]),
    )
    monkeypatch.setattr(routes, "document_service", document_service)
    monkeypatch.setattr(routes, "task_service", LegacyTaskService(document_service=document_service))

    yield paths

    vector_store._DEFAULT_VECTOR_STORE = None  # type: ignore[attr-defined]
    get_settings.cache_clear()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_SESSION_RUNTIME_DIR, ignore_errors=True)
