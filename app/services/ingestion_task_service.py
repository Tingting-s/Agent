"""Persistent lifecycle management for document ingestion tasks."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from app.config import get_document_db_path


TASK_OPERATIONS = frozenset({"UPLOAD", "REINDEX"})
TASK_STATUSES = frozenset({"PENDING", "PROCESSING", "SUCCESS", "FAILURE"})
TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILURE"})
ALLOWED_TRANSITIONS = {
    "PENDING": frozenset({"PROCESSING"}),
    "PROCESSING": TERMINAL_STATUSES,
    "SUCCESS": frozenset(),
    "FAILURE": frozenset(),
}


class IngestionTaskNotFoundError(LookupError):
    """Raised when an ingestion task cannot be found."""


class InvalidTaskTransitionError(ValueError):
    """Raised when an ingestion task lifecycle transition is invalid."""


class DocumentOperationConflictError(RuntimeError):
    """Raised when a document already has an active or locked operation."""


class IngestionTaskService:
    """Store and update document ingestion task state in SQLite."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        busy_timeout_ms: int = 5000,
        operation_lock_timeout_seconds: int = 3600,
    ):
        self.db_path = Path(db_path or get_document_db_path())
        self.busy_timeout_ms = max(int(busy_timeout_ms), 1)
        self.operation_lock_timeout_seconds = max(int(operation_lock_timeout_seconds), 1)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_ingestion_tasks (
                    task_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    operation TEXT NOT NULL
                        CHECK (operation IN ('UPLOAD', 'REINDEX')),
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (status IN ('PENDING', 'PROCESSING', 'SUCCESS', 'FAILURE')),
                    retry_count INTEGER NOT NULL DEFAULT 0
                        CHECK (retry_count >= 0),
                    chunk_count INTEGER NOT NULL DEFAULT 0
                        CHECK (chunk_count >= 0),
                    vector_indexed INTEGER NOT NULL DEFAULT 0
                        CHECK (vector_indexed IN (0, 1)),
                    error_message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_time TEXT NOT NULL,
                    updated_time TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    ux_document_ingestion_tasks_active_document
                ON document_ingestion_tasks(document_id)
                WHERE status IN ('PENDING', 'PROCESSING')
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_document_ingestion_tasks_history
                ON document_ingestion_tasks(document_id, created_time DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_operation_locks (
                    document_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL UNIQUE,
                    operation TEXT NOT NULL,
                    created_time TEXT NOT NULL,
                    expires_time TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_operation(operation: str) -> str:
        normalized = str(operation).strip().upper()
        if normalized not in TASK_OPERATIONS:
            raise ValueError(f"Unsupported ingestion operation: {operation}")
        return normalized

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status).strip().upper()
        if normalized not in TASK_STATUSES:
            raise ValueError(f"Unsupported ingestion task status: {status}")
        return normalized

    @staticmethod
    def _serialize_result(result: Mapping[str, Any] | None) -> str:
        return json.dumps(dict(result or {}), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _row_to_task(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        task = dict(row)
        task["retry_count"] = int(task["retry_count"])
        task["chunk_count"] = int(task["chunk_count"])
        task["vector_indexed"] = bool(task["vector_indexed"])
        try:
            task["result"] = json.loads(task["result_json"])
        except (TypeError, json.JSONDecodeError):
            task["result"] = {}
        return task

    @staticmethod
    def _select_active(
        connection: sqlite3.Connection, document_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT *
            FROM document_ingestion_tasks
            WHERE document_id = ? AND status IN ('PENDING', 'PROCESSING')
            ORDER BY created_time DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()

    def create_task(
        self,
        document_id: str,
        operation: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a task or return the document's existing active task."""
        normalized_document_id = str(document_id).strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")
        normalized_operation = self._normalize_operation(operation)
        requested_task_id = str(task_id or uuid.uuid4())
        timestamp = self._now()

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._delete_expired_operation_locks(connection, timestamp)
                operation_lock = connection.execute(
                    """
                    SELECT operation
                    FROM document_operation_locks
                    WHERE document_id = ?
                    """,
                    (normalized_document_id,),
                ).fetchone()
                if operation_lock is not None:
                    raise DocumentOperationConflictError(
                        "Document operation is already in progress: "
                        f"document_id={normalized_document_id} "
                        f"operation={operation_lock['operation']}"
                    )
                connection.execute(
                    """
                    INSERT INTO document_ingestion_tasks (
                        task_id, document_id, operation, status, retry_count,
                        chunk_count, vector_indexed, error_message, result_json,
                        created_time, updated_time
                    ) VALUES (?, ?, ?, 'PENDING', 0, 0, 0, '', '{}', ?, ?)
                    """,
                    (
                        requested_task_id,
                        normalized_document_id,
                        normalized_operation,
                        timestamp,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                active_row = self._select_active(connection, normalized_document_id)
                if active_row is not None:
                    connection.commit()
                    task = self._row_to_task(active_row)
                    assert task is not None
                    return task

                existing_id_row = connection.execute(
                    "SELECT * FROM document_ingestion_tasks WHERE task_id = ?",
                    (requested_task_id,),
                ).fetchone()
                connection.rollback()
                if existing_id_row is not None:
                    existing_task = self._row_to_task(existing_id_row)
                    assert existing_task is not None
                    if (
                        existing_task["document_id"] == normalized_document_id
                        and existing_task["operation"] == normalized_operation
                    ):
                        return existing_task
                    raise ValueError(
                        f"task_id already belongs to another task: {requested_task_id}"
                    ) from exc
                raise

            created_row = connection.execute(
                "SELECT * FROM document_ingestion_tasks WHERE task_id = ?",
                (requested_task_id,),
            ).fetchone()
            connection.commit()

        task = self._row_to_task(created_row)
        assert task is not None
        return task

    def acquire_document_operation_lock(
        self,
        document_id: str,
        operation: str,
        owner_id: str | None = None,
    ) -> dict[str, str]:
        normalized_document_id = str(document_id).strip()
        normalized_operation = str(operation).strip().upper()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")
        if not normalized_operation:
            raise ValueError("operation must not be empty")

        normalized_owner_id = str(owner_id or uuid.uuid4())
        created_time = self._now()
        expires_time = (
            datetime.now(timezone.utc)
            + timedelta(seconds=self.operation_lock_timeout_seconds)
        ).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._delete_expired_operation_locks(connection, created_time)
                active_task = self._select_active(connection, normalized_document_id)
                if active_task is not None:
                    raise DocumentOperationConflictError(
                        "Document has an active ingestion task: "
                        f"document_id={normalized_document_id} "
                        f"task_id={active_task['task_id']}"
                    )
                connection.execute(
                    """
                    INSERT INTO document_operation_locks (
                        document_id, owner_id, operation, created_time, expires_time
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_document_id,
                        normalized_owner_id,
                        normalized_operation,
                        created_time,
                        expires_time,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise DocumentOperationConflictError(
                    "Document operation is already in progress: "
                    f"document_id={normalized_document_id}"
                ) from exc
            except Exception:
                connection.rollback()
                raise
        return {
            "document_id": normalized_document_id,
            "owner_id": normalized_owner_id,
            "operation": normalized_operation,
            "created_time": created_time,
            "expires_time": expires_time,
        }

    def release_document_operation_lock(self, document_id: str, owner_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM document_operation_locks
                WHERE document_id = ? AND owner_id = ?
                """,
                (str(document_id), str(owner_id)),
            )
        return cursor.rowcount > 0

    @contextmanager
    def document_operation_lock(
        self,
        document_id: str,
        operation: str,
    ) -> Iterator[dict[str, str]]:
        lock = self.acquire_document_operation_lock(document_id, operation)
        try:
            yield lock
        finally:
            self.release_document_operation_lock(document_id, lock["owner_id"])

    @staticmethod
    def _delete_expired_operation_locks(
        connection: sqlite3.Connection,
        current_time: str,
    ) -> None:
        connection.execute(
            "DELETE FROM document_operation_locks WHERE expires_time <= ?",
            (current_time,),
        )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM document_ingestion_tasks WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        return self._row_to_task(row)

    def get_active_task(self, document_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = self._select_active(connection, str(document_id))
        return self._row_to_task(row)

    def _require_task(
        self, connection: sqlite3.Connection, task_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM document_ingestion_tasks WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
        if row is None:
            raise IngestionTaskNotFoundError(f"Ingestion task not found: {task_id}")
        return row

    @staticmethod
    def _validate_transition(current_status: str, new_status: str) -> None:
        if new_status not in ALLOWED_TRANSITIONS[current_status]:
            raise InvalidTaskTransitionError(
                f"Invalid ingestion task transition: {current_status} -> {new_status}"
            )

    def update_status(self, task_id: str, status: str) -> dict[str, Any]:
        """Apply one allowed lifecycle transition without changing result fields."""
        new_status = self._normalize_status(status)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_task(connection, task_id)
                self._validate_transition(current["status"], new_status)
                connection.execute(
                    """
                    UPDATE document_ingestion_tasks
                    SET status = ?, updated_time = ?
                    WHERE task_id = ? AND status = ?
                    """,
                    (new_status, self._now(), str(task_id), current["status"]),
                )
                updated = self._require_task(connection, task_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        task = self._row_to_task(updated)
        assert task is not None
        return task

    def mark_processing(self, task_id: str) -> dict[str, Any]:
        return self.update_status(task_id, "PROCESSING")

    def update_retry_count(self, task_id: str, retry_count: int) -> dict[str, Any]:
        normalized_count = int(retry_count)
        if normalized_count < 0:
            raise ValueError("retry_count must not be negative")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_task(connection, task_id)
                if current["status"] != "PROCESSING":
                    raise InvalidTaskTransitionError(
                        "retry_count can only be updated while a task is PROCESSING"
                    )
                connection.execute(
                    """
                    UPDATE document_ingestion_tasks
                    SET retry_count = ?, updated_time = ?
                    WHERE task_id = ? AND status = 'PROCESSING'
                    """,
                    (normalized_count, self._now(), str(task_id)),
                )
                updated = self._require_task(connection, task_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        task = self._row_to_task(updated)
        assert task is not None
        return task

    def increment_retry_count(self, task_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_task(connection, task_id)
                if current["status"] != "PROCESSING":
                    raise InvalidTaskTransitionError(
                        "retry_count can only be updated while a task is PROCESSING"
                    )
                connection.execute(
                    """
                    UPDATE document_ingestion_tasks
                    SET retry_count = retry_count + 1, updated_time = ?
                    WHERE task_id = ? AND status = 'PROCESSING'
                    """,
                    (self._now(), str(task_id)),
                )
                updated = self._require_task(connection, task_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        task = self._row_to_task(updated)
        assert task is not None
        return task

    def record_success(
        self,
        task_id: str,
        chunk_count: int,
        vector_indexed: bool = True,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_chunk_count = int(chunk_count)
        if normalized_chunk_count < 0:
            raise ValueError("chunk_count must not be negative")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_task(connection, task_id)
                self._validate_transition(current["status"], "SUCCESS")
                connection.execute(
                    """
                    UPDATE document_ingestion_tasks
                    SET status = 'SUCCESS', chunk_count = ?, vector_indexed = ?,
                        error_message = '', result_json = ?, updated_time = ?
                    WHERE task_id = ? AND status = 'PROCESSING'
                    """,
                    (
                        normalized_chunk_count,
                        int(bool(vector_indexed)),
                        self._serialize_result(result),
                        self._now(),
                        str(task_id),
                    ),
                )
                updated = self._require_task(connection, task_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        task = self._row_to_task(updated)
        assert task is not None
        return task

    def record_failure(
        self,
        task_id: str,
        error_message: str,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_task(connection, task_id)
                self._validate_transition(current["status"], "FAILURE")
                connection.execute(
                    """
                    UPDATE document_ingestion_tasks
                    SET status = 'FAILURE', vector_indexed = 0,
                        error_message = ?, result_json = ?, updated_time = ?
                    WHERE task_id = ? AND status = 'PROCESSING'
                    """,
                    (
                        str(error_message),
                        self._serialize_result(result),
                        self._now(),
                        str(task_id),
                    ),
                )
                updated = self._require_task(connection, task_id)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        task = self._row_to_task(updated)
        assert task is not None
        return task

    def list_history(
        self, document_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        normalized_limit = int(limit)
        if normalized_limit <= 0:
            raise ValueError("limit must be greater than zero")

        with closing(self._connect()) as connection:
            if document_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM document_ingestion_tasks
                    ORDER BY created_time DESC
                    LIMIT ?
                    """,
                    (normalized_limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM document_ingestion_tasks
                    WHERE document_id = ?
                    ORDER BY created_time DESC
                    LIMIT ?
                    """,
                    (str(document_id), normalized_limit),
                ).fetchall()
        return [task for row in rows if (task := self._row_to_task(row)) is not None]
