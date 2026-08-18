from __future__ import annotations

"""Compatibility import for the historical TaskService name."""

from app.services.legacy_task_service import LegacyTaskService, TaskService


__all__ = ["LegacyTaskService", "TaskService"]
