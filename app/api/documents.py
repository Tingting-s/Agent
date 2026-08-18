from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.documents import (
    BatchDeleteDocumentsRequest,
    BatchDeleteDocumentsResponse,
    DocumentDeleteApiResponse,
    DocumentListApiResponse,
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentReindexApiResponse,
    DocumentUploadResponse,
    SummaryApiRequest,
    SummaryApiResponse,
    UploadApiResponse,
)
from app.schemas.qa import DocumentQARequest, DocumentQAResponse
from app.services.document_qa_service import DocumentQAService


router = APIRouter()


def _document_service() -> DocumentQAService:
    from app.api import routes

    return routes.document_service


@router.post("/documents/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    try:
        return await _document_service().upload_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/parse", response_model=DocumentParseResponse, tags=["documents"])
def parse_document(request: DocumentParseRequest) -> DocumentParseResponse:
    try:
        return _document_service().parse_document(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/ask", response_model=DocumentQAResponse, tags=["documents"])
def ask_document(request: DocumentQARequest) -> DocumentQAResponse:
    try:
        return _document_service().ask_document(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/documents/upload", response_model=UploadApiResponse, tags=["api-documents"])
async def api_upload_document(file: UploadFile = File(...)) -> UploadApiResponse:
    return await _document_service().upload_document_api(file)


@router.get("/api/documents", response_model=DocumentListApiResponse, tags=["api-documents"])
def api_list_documents() -> DocumentListApiResponse:
    return _document_service().list_documents_api()


@router.post("/api/documents/{document_id}/reindex", response_model=DocumentReindexApiResponse, tags=["api-documents"])
def api_reindex_document(document_id: str) -> DocumentReindexApiResponse:
    return _document_service().reindex_document(document_id)


@router.delete("/api/documents/batch", response_model=BatchDeleteDocumentsResponse, tags=["api-documents"])
def api_batch_delete_documents(request: BatchDeleteDocumentsRequest) -> BatchDeleteDocumentsResponse:
    return _document_service().batch_delete_documents(request.document_ids)


@router.delete("/api/documents/{document_id}", response_model=DocumentDeleteApiResponse, tags=["api-documents"])
def api_delete_document(document_id: str) -> DocumentDeleteApiResponse:
    return _document_service().delete_document_api(document_id)


@router.post("/api/documents/summary", response_model=SummaryApiResponse, tags=["api-documents"])
def api_summarize_document(request: SummaryApiRequest) -> SummaryApiResponse:
    return _document_service().summarize_document_api(request.document_id)
