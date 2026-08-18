from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter

from app.schemas.qa import AskApiRequest, AskApiResponse
from app.services.document_qa_service import DocumentQAService


router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _document_service() -> DocumentQAService:
    from app.api import routes

    return routes.document_service


@router.post("/api/qa/ask", response_model=AskApiResponse, tags=["api-qa"])
def api_ask_document(request: AskApiRequest) -> AskApiResponse:
    start = perf_counter()
    LOGGER.info(
        "QA stage started: request_id=%s document_id=%s question=%s stage_name=api_receive elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
        "",
        request.document_id,
        request.question,
        0.0,
        "",
        0,
    )
    response = _document_service().ask_document_api(
        document_id=request.document_id,
        question=request.question,
    )
    LOGGER.info(
        "QA stage completed: request_id=%s document_id=%s question=%s stage_name=api_response elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
        response.request_id,
        request.document_id,
        request.question,
        round(perf_counter() - start, 6),
        response.retrieval_source,
        len(response.source_chunks),
    )
    return response
