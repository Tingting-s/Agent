from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_document_db_path


DEFAULT_DOCUMENT_DB_PATH = get_document_db_path()


class DocumentStorageService:
    """SQLite-backed metadata store for uploaded documents."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or get_document_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def upsert_document(
        self,
        *,
        document_id: str,
        original_file_name: str,
        stored_file_name: str,
        file_path: str,
        file_type: str,
        status: str = "success",
        chunk_count: int = 0,
        vector_indexed: bool = False,
        error_message: str = "",
        upload_time: str | None = None,
    ) -> dict[str, Any]:
        upload_time_value = upload_time or datetime.now(UTC).isoformat()
        updated_time_value = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    original_file_name,
                    stored_file_name,
                    file_path,
                    file_type,
                    upload_time,
                    status,
                    chunk_count,
                    vector_indexed,
                    error_message,
                    updated_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    original_file_name = excluded.original_file_name,
                    stored_file_name = excluded.stored_file_name,
                    file_path = excluded.file_path,
                    file_type = excluded.file_type,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    vector_indexed = excluded.vector_indexed,
                    error_message = excluded.error_message,
                    updated_time = excluded.updated_time
                """,
                (
                    document_id,
                    original_file_name,
                    stored_file_name,
                    file_path,
                    file_type,
                    upload_time_value,
                    status,
                    int(chunk_count),
                    1 if vector_indexed else 0,
                    error_message,
                    updated_time_value,
                ),
            )
        document = self.get_document(document_id)
        return document or {}

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    original_file_name,
                    stored_file_name,
                    file_path,
                    file_type,
                    upload_time,
                    status,
                    chunk_count,
                    vector_indexed,
                    error_message,
                    updated_time
                FROM documents
                ORDER BY upload_time DESC
                """
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        if not document_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    original_file_name,
                    stored_file_name,
                    file_path,
                    file_type,
                    upload_time,
                    status,
                    chunk_count,
                    vector_indexed,
                    error_message,
                    updated_time
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def update_index_status(
        self,
        document_id: str,
        *,
        chunk_count: int | None = None,
        vector_indexed: bool | None = None,
        status: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any] | None:
        if not document_id:
            return None

        assignments: list[str] = []
        values: list[Any] = []
        if chunk_count is not None:
            assignments.append("chunk_count = ?")
            values.append(max(0, int(chunk_count)))
        if vector_indexed is not None:
            assignments.append("vector_indexed = ?")
            values.append(1 if vector_indexed else 0)
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if error_message is not None:
            assignments.append("error_message = ?")
            values.append(error_message)
        if not assignments:
            return self.get_document(document_id)

        assignments.append("updated_time = ?")
        values.append(datetime.now(UTC).isoformat())
        values.append(document_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE document_id = ?",
                values,
            )
        return self.get_document(document_id)

    def update_file_path(self, document_id: str, file_path: str | Path) -> dict[str, Any] | None:
        if not document_id:
            return None

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET file_path = ?, updated_time = ?
                WHERE document_id = ?
                """,
                (
                    str(file_path),
                    datetime.now(UTC).isoformat(),
                    document_id,
                ),
            )
        return self.get_document(document_id)

    def delete_document_metadata(self, document_id: str) -> bool:
        if not document_id:
            return False
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_file_name TEXT NOT NULL,
                    stored_file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    vector_indexed INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    updated_time TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._ensure_column(connection, "error_message", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "updated_time", "TEXT NOT NULL DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(self, connection: sqlite3.Connection, name: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        if name not in columns:
            connection.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")

    def _row_to_document(self, row: sqlite3.Row) -> dict[str, Any]:
        document = dict(row)
        document["chunk_count"] = int(document.get("chunk_count") or 0)
        document["vector_indexed"] = bool(document.get("vector_indexed"))
        document["error_message"] = str(document.get("error_message") or "")
        document["updated_time"] = str(document.get("updated_time") or "")
        return document
