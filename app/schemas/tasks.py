from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    user_input: str = Field(..., description="Raw user input for the task.")
    task_type: str | None = Field(default=None, description="Optional task type hint.")
    context: dict[str, Any] | None = Field(default=None, description="Optional execution context.")
    additional_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured inputs that take priority over free-form text parsing.",
    )

    def get_context(self) -> dict[str, Any]:
        return self.context or {}

    def get_additional_inputs(self) -> dict[str, Any]:
        return self.additional_inputs or {}

    def get_input_value(self, key: str) -> Any:
        additional_inputs = self.get_additional_inputs()
        if key in additional_inputs and additional_inputs[key] not in (None, ""):
            return additional_inputs[key]

        context = self.get_context()
        if key in context and context[key] not in (None, ""):
            return context[key]

        return None


class TaskResponse(BaseModel):
    task_type: str = Field(..., description="Resolved task type.")
    status: str = Field(..., description="Task execution status.")
    message: str = Field(..., description="High-level task message.")
    structured_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured result payload for the executed task.",
    )
    retry_count: int = Field(default=0, description="Retry count used during execution.")


class DocumentSummaryResult(BaseModel):
    summary: str = Field(..., description="Short document summary.")
    key_points: list[str] = Field(default_factory=list, description="Key points from the document.")
    risks: list[str] = Field(default_factory=list, description="Potential risks identified in the document.")


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="Tool identifier.")
    status: Literal["success", "error"] = Field(..., description="Tool execution status.")
    message: str = Field(..., description="High-level tool message.")
    output: dict[str, Any] = Field(default_factory=dict, description="Structured tool output payload.")
    error: str | None = Field(default=None, description="Tool error details when execution fails.")
