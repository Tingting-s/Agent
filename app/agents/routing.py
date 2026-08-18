from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.prompts import ROUTER_SYSTEM_PROMPT
from app.schemas.tasks import TaskRequest


@dataclass(frozen=True)
class RouteDecision:
    task_type: str
    steps: tuple[str, ...]
    reason: str
    prompt: str = ROUTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class DocumentRouteDecision:
    task_type: str
    steps: tuple[str, ...]
    reason: str
    prompt: str = ROUTER_SYSTEM_PROMPT


class RouterAgent:
    """Legacy /tasks/execute routing adapter."""

    supported_task_types: ClassVar[set[str]] = {
        "document_summary",
        "document_qa",
    }
    deprecated_task_types: ClassVar[set[str]] = {
        "weather_query",
        "email_draft",
        "meeting_extraction",
    }
    keyword_rules: ClassVar[dict[str, tuple[str, ...]]] = {
        "document_qa": ("question", "ask", "qa", "answer", "query"),
        "document_summary": ("summary", "summarize", "document", "file", ".txt", ".md"),
    }

    def __init__(self, system_prompt: str = ROUTER_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def route(self, request: TaskRequest) -> RouteDecision:
        if request.task_type in self.deprecated_task_types:
            return RouteDecision(
                task_type=str(request.task_type),
                steps=("deprecated",),
                reason=f"Task type '{request.task_type}' is deprecated in the document QA service.",
            )

        if request.task_type in self.supported_task_types:
            return RouteDecision(
                task_type=str(request.task_type),
                steps=(str(request.task_type),),
                reason=f"Used explicit task_type '{request.task_type}'.",
            )

        text = self._build_search_text(request)
        for task_type, keywords in self.keyword_rules.items():
            matched_keyword = next((keyword for keyword in keywords if keyword in text), None)
            if matched_keyword:
                return RouteDecision(
                    task_type=task_type,
                    steps=(task_type,),
                    reason=f"Matched keyword '{matched_keyword}' and routed to {task_type}.",
                )

        if request.get_input_value("question"):
            return RouteDecision(
                task_type="document_qa",
                steps=("document_qa",),
                reason="Found question in structured inputs.",
            )

        return RouteDecision(
            task_type="document_summary",
            steps=("document_summary",),
            reason="Defaulted to document summary workflow.",
        )

    def _build_search_text(self, request: TaskRequest) -> str:
        parts = [
            request.user_input,
            request.task_type or "",
            str(request.get_input_value("question") or ""),
            str(request.get_input_value("file_path") or ""),
            str(request.get_input_value("document_id") or ""),
        ]
        return " ".join(part for part in parts if part).lower()


class DocumentRouterAgent:
    """Routes LangGraph document QA requests only."""

    supported_task_types = {"document_qa", "document_summary"}

    def __init__(self, system_prompt: str = ROUTER_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def route_document_request(
        self,
        *,
        question: str | None,
        file_path: str | None,
        document_id: str | None = None,
        task_type: str | None = None,
    ) -> DocumentRouteDecision:
        normalized_task_type = (task_type or "").strip().lower()
        normalized_question = (question or "").strip()
        normalized_file_path = (file_path or "").strip()
        normalized_document_id = (document_id or "").strip()

        if normalized_task_type and normalized_task_type not in self.supported_task_types:
            return DocumentRouteDecision(
                task_type="invalid_request",
                steps=("invalid_request",),
                reason=f"Unsupported document task type '{normalized_task_type}'.",
            )
        if not normalized_file_path and not normalized_document_id:
            return DocumentRouteDecision(
                task_type="invalid_request",
                steps=("invalid_request",),
                reason="A document_id or file_path is required.",
            )
        if normalized_task_type == "document_summary":
            return DocumentRouteDecision(
                task_type="document_summary",
                steps=("document_summary",),
                reason="Explicit document_summary request.",
            )
        if normalized_question:
            return DocumentRouteDecision(
                task_type="document_qa",
                steps=("document_qa",),
                reason="Question and document reference are present.",
            )
        return DocumentRouteDecision(
            task_type="document_summary",
            steps=("document_summary",),
            reason="No question was provided; routed to document_summary.",
        )
