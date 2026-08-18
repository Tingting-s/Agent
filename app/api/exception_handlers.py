from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.response import ErrorResponse
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _build_error_response(
    *,
    message: str,
    error_code: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ErrorResponse(
        status="error",
        message=message,
        error_code=error_code,
        details=details or {},
    ).model_dump()


def register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning(
            "Request validation failed for %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content=_build_error_response(
                message="Request validation failed.",
                error_code="request_validation_error",
                details={
                    "errors": exc.errors(),
                },
            ),
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(
            "HTTP exception for %s %s: status=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_response(
                message=str(exc.detail),
                error_code="http_exception",
                details={"status_code": exc.status_code},
            ),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception for %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=_build_error_response(
                message="Internal server error.",
                error_code="internal_server_error",
                details={},
            ),
        )
