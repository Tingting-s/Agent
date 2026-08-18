from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health state.")
    service: str = Field(..., description="Application name.")
    environment: str = Field(..., description="Current runtime environment.")


class ErrorResponse(BaseModel):
    status: str = Field(..., description="Error status.")
    message: str = Field(..., description="High-level error message.")
    error_code: str = Field(..., description="Machine-readable error code.")
    details: dict[str, Any] = Field(default_factory=dict, description="Optional error details.")
