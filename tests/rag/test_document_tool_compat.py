from pathlib import Path

import app.tools.document_tool as document_tool
import app.rag.generation as generation


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class FakeDocumentLLMService:
    def summarize_document(self, content: str) -> dict[str, object]:
        return {
            "summary": "The document outlines the current project status and next steps.",
            "key_points": [
                "The project is in the scaffolding phase.",
                "The next milestone is real orchestration.",
            ],
            "risks": [
                "Tool-layer coverage is still incomplete.",
            ],
        }


def test_read_document_supports_txt_file(monkeypatch) -> None:
    monkeypatch.setattr(generation, "get_qwen_llm_service", lambda: FakeDocumentLLMService())
    result = document_tool.read_document(str(DATA_DIR / "project_notes.txt"))

    assert result.status == "success"
    assert result.output["file_type"] == "txt"
    assert "Project Notes" in result.output["content"]
    assert result.output["summary"] == "The document outlines the current project status and next steps."
    assert result.output["key_points"]
    assert result.output["risks"]


def test_read_document_supports_markdown_file(monkeypatch) -> None:
    monkeypatch.setattr(generation, "get_qwen_llm_service", lambda: FakeDocumentLLMService())
    result = document_tool.read_document(str(DATA_DIR / "release_plan.md"))

    assert result.status == "success"
    assert result.output["file_type"] == "md"
    assert result.output["summary"] == "The document outlines the current project status and next steps."


def test_read_document_rejects_unsupported_files() -> None:
    result = document_tool.read_document(str(DATA_DIR / "raw_payload.json"))

    assert result.status == "error"
    assert result.error == "Only .txt, .md, .docx, and text-based .pdf files are supported."
