from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_chroma_dir
from app.rag import vector_store as vector_store_module
from app.services.document_qa_service import DocumentQAService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Back up the local Chroma index and rebuild it from uploaded documents.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    args = parser.parse_args(argv)

    service = DocumentQAService()
    documents = service.storage_service.list_documents()
    if not documents:
        print("没有发现文档记录，无需重建 Chroma 索引。")
        return 0

    chroma_dir = get_chroma_dir()
    backup_dir: Path | None = None
    if chroma_dir.exists():
        ensure_under_project(chroma_dir)
        backup_dir = next_backup_path(chroma_dir)
        print("即将备份并重建 Chroma 索引：")
        print(f"- 当前索引：{chroma_dir}")
        print(f"- 备份位置：{backup_dir}")
        print(f"- 待重建文档数：{len(documents)}")
        if not args.yes:
            confirm = input("确认继续请输入 YES，其他输入将取消：").strip()
            if confirm != "YES":
                print("已取消，未修改任何索引数据。")
                return 0
        shutil.move(str(chroma_dir), str(backup_dir))
        print(f"旧 Chroma 索引已备份到：{backup_dir}")
    else:
        print(f"未发现现有 Chroma 索引目录，将新建：{chroma_dir}")

    vector_store_module._DEFAULT_VECTOR_STORE = None  # type: ignore[attr-defined]

    success_count = 0
    failed_count = 0
    for document in documents:
        document_id = str(document.get("document_id") or "")
        file_name = str(document.get("original_file_name") or document.get("stored_file_name") or "")
        result = service.reindex_document(document_id)
        if result.status == "success":
            success_count += 1
            print(f"- {document_id} {file_name}：成功，chunks={result.chunk_count}")
        else:
            failed_count += 1
            print(f"- {document_id} {file_name}：失败，{result.error_message or result.message}")

    print(f"重建完成：成功 {success_count}，失败 {failed_count}。")
    if backup_dir:
        print(f"旧索引备份仍保留在：{backup_dir}")
    return 0 if failed_count == 0 else 1


def next_backup_path(chroma_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup_dir = chroma_dir.with_name(f"{chroma_dir.name}_backup_{timestamp}")
    counter = 1
    while backup_dir.exists():
        backup_dir = chroma_dir.with_name(f"{chroma_dir.name}_backup_{timestamp}_{counter}")
        counter += 1
    ensure_under_project(backup_dir.parent)
    return backup_dir


def ensure_under_project(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = ROOT_DIR.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"拒绝处理项目目录外路径：{resolved_path}")


if __name__ == "__main__":
    raise SystemExit(main())
