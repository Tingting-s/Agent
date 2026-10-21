"""Celery application used by standalone document-ingestion workers."""

from __future__ import annotations

from celery import Celery

from app.config import get_settings


settings = get_settings()

celery_app = Celery(
    "enterprise_document_qa",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.document_ingestion"],
)
celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": max(settings.CELERY_VISIBILITY_TIMEOUT, 1),
    },
    enable_utc=True,
    result_expires=max(settings.CELERY_RESULT_EXPIRES, 1),
    result_serializer="json",
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    task_soft_time_limit=max(settings.CELERY_TASK_SOFT_TIME_LIMIT, 1),
    task_time_limit=max(settings.CELERY_TASK_TIME_LIMIT, 1),
    timezone="UTC",
    worker_prefetch_multiplier=1,
)


__all__ = ["celery_app"]
