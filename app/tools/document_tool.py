from __future__ import annotations

"""Legacy compatibility wrapper for document helpers.

New document loading, splitting, and summarization code should depend on
app.rag modules directly. This wrapper keeps older imports and tests working.
"""

from pathlib import Path
from typing import Any

from app.rag.generation import summarize_document_text as rag_summarize_document_text
from app.rag.ingestion import SUPPORTED_DOCUMENT_EXTENSIONS, load_document as rag_load_document, split_text
from app.schemas.tasks import TaskRequest, ToolResult


SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS


def load_document(file_path: str) -> ToolResult:
    result = rag_load_document(file_path)
    path = Path(file_path)
    if result["status"] == "error":
        error_message = str(result["error_message"])
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            error_message = "Only .txt, .md, .docx, and text-based .pdf files are supported."
        message = (
            "Unsupported document type."
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS
            else "Failed to read document content."
        )
        if not path.exists():
            message = "Document file was not found."
        return ToolResult(
            tool_name="document_tool",
            status="error",
            message=message,
            output={"file_path": str(path), "supported_extensions": sorted(SUPPORTED_EXTENSIONS)},
            error=error_message,
        )

    return ToolResult(
        tool_name="document_tool",
        status="success",
        message=f"Loaded document from {result['file_name']}.",
        output={
            "file_path": str(result["file_path"]),
            "file_name": str(result["file_name"]),
            "file_type": str(result["file_type"]),
            "content": str(result["text"]),
            "line_count": int(result["line_count"]),
            "char_count": int(result["char_count"]),
        },
    )


def split_document_text(
    content: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    return split_text(
        document_id=None,
        file_name="",
        text=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def summarize_document_text(content: str) -> dict[str, Any]:
    return rag_summarize_document_text(content)


def read_document(file_path: str) -> ToolResult:
    result = load_document(file_path)
    if result.status == "error":
        return result

    content = str(result.output.get("content") or "")
    summary = summarize_document_text(content)
    output = dict(result.output)
    output.update(summary)
    return ToolResult(
        tool_name="document_tool",
        status="success",
        message=f"Loaded and summarized document from {output['file_name']}.",
        output=output,
    )


class DocumentTool:
    name = "document_tool"

    def run(self, request: TaskRequest) -> ToolResult:
        file_path = str(request.get_input_value("file_path") or request.user_input)
        return read_document(file_path)
