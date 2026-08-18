from __future__ import annotations

"""Legacy /tasks/execute service adapter.

The main service path is the LangGraph document QA API. This module keeps the
historical /tasks/execute endpoint usable without mixing it into the primary
document QA service layer.
"""

from pydantic import ValidationError

from app.schemas.documents import DocumentParseRequest
from app.schemas.qa import DocumentQARequest
from app.schemas.tasks import TaskRequest, TaskResponse
from app.services.document_qa_service import DocumentQAService
from app.utils.logger import get_logger


logger = get_logger(__name__)


class LegacyTaskService:
    """Compatibility adapter for the historical /tasks/execute endpoint."""

    def __init__(self, document_service: DocumentQAService | None = None) -> None:
        self.document_service = document_service or DocumentQAService()

    def handle_request(self, request: TaskRequest) -> TaskResponse:
        logger.info(
            "Handling legacy task request as document workflow. task_type_hint=%s input_preview=%s",
            request.task_type,
            request.user_input[:80],
        )

        if request.task_type in {"weather_query", "email_draft", "meeting_extraction"}:
            return TaskResponse(
                task_type=str(request.task_type),
                status="error",
                message=(
                    "This task type has been deprecated. The service now supports enterprise document "
                    "upload, parsing, retrieval, QA, validation, and logging."
                ),
                structured_result={"deprecated": True},
                retry_count=0,
            )

        if request.task_type == "document_qa" or request.get_input_value("question"):
            return self._handle_document_qa(request)

        return self._handle_document_parse(request)

    def _handle_document_parse(self, request: TaskRequest) -> TaskResponse:
        document_id = request.get_input_value("document_id")
        file_path = request.get_input_value("file_path")
        if not document_id and not file_path:
            file_path = request.user_input
        try:
            parse_request = DocumentParseRequest(
                document_id=document_id,
                file_path=file_path,
            )
        except ValidationError as exc:
            return self._need_more_info("document_summary", exc)

        try:
            response = self.document_service.parse_document(parse_request)
        except ValueError as exc:
            return self._need_more_info("document_summary", exc)
        return TaskResponse(
            task_type="document_summary",
            status=response.status,
            message=response.message,
            structured_result=response.model_dump(),
            retry_count=0,
        )

    def _handle_document_qa(self, request: TaskRequest) -> TaskResponse:
        try:
            qa_request = DocumentQARequest(
                document_id=request.get_input_value("document_id"),
                file_path=request.get_input_value("file_path"),
                question=str(request.get_input_value("question") or request.user_input),
                top_k=int(request.get_input_value("top_k") or 3),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            return self._need_more_info("document_qa", exc)

        try:
            response = self.document_service.ask_document(qa_request)
        except ValueError as exc:
            return self._need_more_info("document_qa", exc)
        return TaskResponse(
            task_type="document_qa",
            status=response.status,
            message=response.message,
            structured_result=response.model_dump(),
            retry_count=0,
        )

    def _need_more_info(self, task_type: str, exc: Exception) -> TaskResponse:
        return TaskResponse(
            task_type=task_type,
            status="need_more_info",
            message="A document reference is required before this request can continue.",
            structured_result={"error": str(exc), "required": ["document_id", "file_path"]},
            retry_count=0,
        )


TaskService = LegacyTaskService
