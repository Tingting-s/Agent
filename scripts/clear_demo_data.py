from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_chroma_dir, get_document_db_path, get_log_dir, get_upload_dir


def main() -> int:
    targets = [
        ("uploads", get_upload_dir()),
        ("logs", get_log_dir()),
        ("chroma_db", get_chroma_dir()),
        ("data/documents.db", get_document_db_path()),
    ]

    print("企业文档问答系统演示数据清理")
    print("即将清理以下本地运行数据：")
    for label, path in targets:
        print(f"- {label}: {path}")
    print("不会删除 tests/fixtures/、.env、.env.example、README、PROJECT_DEMO、源码或测试文件。")

    confirm = input("确认删除请输入 YES，其他输入将取消：").strip()
    if confirm != "YES":
        print("已取消，未删除任何数据。")
        return 0

    for label, path in targets:
        try:
            delete_target(path)
            print(f"- {label}：已清理")
        except Exception as exc:
            print(f"- {label}：清理失败，{type(exc).__name__}: {exc}")

    print("清理完成。上述目录或数据库文件会在下次运行服务或接口调用时重新生成。")
    return 0


def delete_target(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT_DIR.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"拒绝删除项目目录外路径：{resolved}")

    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
