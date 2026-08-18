from __future__ import annotations

import json
import importlib.util
import logging
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_chroma_dir, get_document_db_path, get_log_dir, get_settings, get_upload_dir
from app.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    CHROMADB_RUNTIME_CHECK_TIMEOUT_SECONDS,
    HashEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)


def main() -> int:
    logging.disable(logging.CRITICAL)
    checks: list[tuple[str, str]] = []
    warnings: list[str] = []

    settings = get_settings()
    upload_dir = get_upload_dir()
    log_dir = get_log_dir()
    chroma_dir = get_chroma_dir()
    db_path = get_document_db_path()
    data_dir = db_path.parent

    checks.append(("uploads", "存在" if upload_dir.exists() else "不存在"))
    checks.append(("logs", "存在" if log_dir.exists() else "不存在"))
    checks.append(("chroma_db", "存在" if chroma_dir.exists() else "不存在"))
    checks.append(("data", "存在" if data_dir.exists() else "不存在"))

    sqlite_status, sqlite_warning = check_sqlite(db_path)
    checks.append(("SQLite", sqlite_status))
    if sqlite_warning:
        warnings.append(sqlite_warning)

    document_file_status, document_file_warning = check_document_files(db_path, upload_dir)
    checks.append(("文档文件", document_file_status))
    if document_file_warning:
        warnings.append(document_file_warning)

    chroma_status, chroma_warning = check_chroma(chroma_dir)
    checks.append(("Chroma", chroma_status))
    if chroma_warning:
        warnings.append(chroma_warning)

    checks.append(
        (
            "Embedding 配置",
            (
                f"{settings.EMBEDDING_MODEL_NAME}，"
                f"device={settings.EMBEDDING_DEVICE or 'auto'}，"
                f"preload={settings.EMBEDDING_PRELOAD}，"
                f"local_files_only={settings.EMBEDDING_LOCAL_FILES_ONLY}"
            ),
        )
    )
    embedding_status, embedding_warning = check_embedding()
    checks.append(("Embedding 加载", embedding_status))
    if embedding_warning:
        warnings.append(embedding_warning)

    log_status, log_warning = check_recent_logs(log_dir)
    checks.append(("最近日志", log_status))
    if log_warning:
        warnings.append(log_warning)

    overall_status = "可运行" if not warnings else "可运行，但存在需要关注的项目"

    print("企业文档问答系统自检结果：")
    for name, status in checks:
        print(f"- {name}：{status}")
    if warnings:
        print("- 异常汇总：")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"- 总体状态：{overall_status}")
    return 0


def check_sqlite(db_path: Path) -> tuple[str, str]:
    if not db_path.exists():
        return "未发现 documents.db，当前为空数据状态", ""

    try:
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
            ).fetchone()
            if row is None:
                return "已存在，但未发现 documents 表", "SQLite 中缺少 documents 表。"
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return f"正常，文档数量 {int(count)}", ""
    except Exception as exc:
        return "异常，无法读取文档库", f"SQLite 检查失败：{type(exc).__name__}: {exc}"


def check_document_files(db_path: Path, upload_dir: Path) -> tuple[str, str]:
    if not db_path.exists():
        return "无文档记录", ""

    try:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT document_id, stored_file_name, file_path
                FROM documents
                ORDER BY upload_time DESC
                """
            ).fetchall()
    except Exception as exc:
        return "未知", f"文档文件检查失败：{type(exc).__name__}: {exc}"

    if not rows:
        return "无文档记录", ""

    available_count = 0
    stale_path_count = 0
    missing_ids: list[str] = []
    for document_id, stored_file_name, file_path in rows:
        stored_file_name_value = str(stored_file_name or "").strip()
        recorded_path = Path(str(file_path or ""))
        stored_path = upload_dir / stored_file_name_value
        recorded_path_exists = recorded_path.exists() and is_path_in_directory(recorded_path, upload_dir)
        stored_path_exists = bool(stored_file_name_value) and stored_path.exists()
        if recorded_path_exists or stored_path_exists:
            available_count += 1
            if stored_path_exists and not same_path(recorded_path, stored_path):
                stale_path_count += 1
        else:
            missing_ids.append(str(document_id or ""))

    status = f"{available_count}/{len(rows)} 个上传文件可用"
    if stale_path_count:
        status += f"，{stale_path_count} 条记录可自动迁移到当前 uploads"

    warnings: list[str] = []
    if missing_ids:
        warnings.append(
            "找不到部分文档上传文件："
            f"{', '.join(missing_ids[:3])}"
            f"{' 等' if len(missing_ids) > 3 else ''}。"
        )
    if stale_path_count:
        warnings.append("部分 SQLite 记录仍保存旧项目路径，接口会在访问文档时自动修正。")
    return status, " ".join(warnings)


def check_chroma(chroma_dir: Path) -> tuple[str, str]:
    init_status, init_warning = check_chroma_initialization()
    vector_count_status, vector_count_warning = count_existing_chroma_vectors(chroma_dir)
    warning = "; ".join(item for item in [init_warning, vector_count_warning] if item)
    return f"{init_status}，{vector_count_status}", warning


def check_chroma_initialization() -> tuple[str, str]:
    if importlib.util.find_spec("chromadb") is None:
        return "初始化异常", "chromadb package is not installed."

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import chromadb, sys; sys.stdout.write(getattr(chromadb, '__version__', 'unknown'))",
            ],
            capture_output=True,
            text=True,
            timeout=CHROMADB_RUNTIME_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:
        return "初始化异常", f"chromadb runtime check failed: {type(exc).__name__}: {exc}"

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip() or f"exit code {result.returncode}"
        return "初始化异常", f"chromadb runtime check failed: {details}"

    temp_dir = Path(tempfile.mkdtemp(prefix="enterprise_doc_qa_chroma_check_"))
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(temp_dir))
        collection = client.get_or_create_collection(
            name=DEFAULT_COLLECTION_NAME,
            embedding_function=HashEmbeddingFunction(),
        )
        _ = collection.count()
        version = (result.stdout or "").strip() or "unknown"
        return f"初始化正常，chromadb={version}", ""
    except Exception as exc:
        return "初始化异常", f"Chroma 初始化检查失败：{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def count_existing_chroma_vectors(chroma_dir: Path) -> tuple[str, str]:
    if not chroma_dir.exists():
        return "当前向量数量 0（chroma_db 不存在）", ""

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(
            name=DEFAULT_COLLECTION_NAME,
            embedding_function=HashEmbeddingFunction(),
        )
        count = collection.count()
        return f"当前向量数量 {int(count)}", ""
    except ValueError:
        return "当前向量数量 0（collection 尚未创建）", ""
    except Exception as exc:
        if looks_like_incompatible_chroma_store(exc):
            return (
                "当前向量数量未知（旧索引不兼容）",
                "现有 chroma_db 与当前 chromadb 版本不兼容。建议备份或清空 chroma_db 后重新入库文档。",
            )
        return "当前向量数量未知", f"Chroma 向量数量统计失败：{type(exc).__name__}: {exc}"


def check_embedding() -> tuple[str, str]:
    try:
        embedding = SentenceTransformerEmbeddingFunction()
        vector = embedding.embed_query("企业文档问答系统自检")
        dimension = len(vector) if isinstance(vector, list) else 0
        if dimension <= 0:
            return "异常，未返回有效向量", "Embedding function 返回了空向量。"
        if getattr(embedding, "_model", None) is None:
            return f"可加载（当前使用 hash fallback），向量维度 {dimension}", ""
        return f"可加载，向量维度 {dimension}", ""
    except Exception as exc:
        return "异常，无法加载", f"Embedding 加载失败：{type(exc).__name__}: {exc}"


def check_recent_logs(log_dir: Path) -> tuple[str, str]:
    log_path = log_dir / "qa_logs.jsonl"
    if not log_path.exists():
        return "0 条", ""

    try:
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid_count = 0
        for line in lines[-50:]:
            try:
                raw: Any = json.loads(line)
                if isinstance(raw, dict):
                    valid_count += 1
            except json.JSONDecodeError:
                continue
        return f"{len(lines)} 条（最近 50 条中可解析 {valid_count} 条）", ""
    except Exception as exc:
        return "未知", f"日志读取失败：{type(exc).__name__}: {exc}"


def is_path_in_directory(path: Path, directory: Path) -> bool:
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except OSError:
        return False


def same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def looks_like_incompatible_chroma_store(exc: Exception) -> bool:
    return isinstance(exc, KeyError) and str(exc).strip("\"'") == "_type"


if __name__ == "__main__":
    raise SystemExit(main())
