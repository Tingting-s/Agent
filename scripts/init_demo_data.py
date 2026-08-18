from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from starlette.datastructures import UploadFile


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.services.document_qa_service import DocumentQAService


DEMO_FILES = [
    "sample_device_manual.txt",
    "sample_training_material.md",
    "sample_faq.txt",
]


async def main() -> int:
    logging.disable(logging.CRITICAL)
    settings = get_settings()
    fixtures_dir = ROOT_DIR / "tests" / "fixtures"
    service = DocumentQAService()
    existing_names = {
        str(document.get("original_file_name") or "")
        for document in service.storage_service.list_documents()
    }

    print("企业文档问答系统演示数据初始化：")
    print(
        "- Embedding 首次加载可能较慢，当前配置："
        f"{settings.EMBEDDING_MODEL_NAME}，device={settings.EMBEDDING_DEVICE or 'auto'}"
    )

    imported_count = 0
    skipped_count = 0
    failed_count = 0

    for file_name in DEMO_FILES:
        fixture_path = fixtures_dir / file_name
        if not fixture_path.exists():
            failed_count += 1
            print(f"- {file_name}：失败，样本文档不存在：{fixture_path}")
            continue

        if file_name in existing_names:
            skipped_count += 1
            print(f"- {file_name}：已存在，跳过导入")
            continue

        try:
            with fixture_path.open("rb") as file:
                upload = UploadFile(filename=file_name, file=file)
                result = await service.upload_document(upload)
            imported_count += 1
            print(
                f"- {file_name}：导入成功，"
                f"document_id={result.document_id}，"
                f"chunk_count={result.chunk_count}，"
                f"vector_indexed={result.vector_indexed}"
            )
            if result.vector_error:
                print(f"  - 向量入库提示：{result.vector_error}")
        except Exception as exc:
            failed_count += 1
            print(f"- {file_name}：导入失败，{type(exc).__name__}: {exc}")

    print(
        "初始化完成："
        f"导入 {imported_count} 个，跳过 {skipped_count} 个，失败 {failed_count} 个。"
    )
    print("可以启动 Streamlit 演示页面：streamlit run web/streamlit_app.py")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
