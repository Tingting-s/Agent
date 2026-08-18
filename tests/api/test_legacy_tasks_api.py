from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_legacy_tasks_execute_runs_document_summary() -> None:
    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "project-notes.txt",
                (DATA_DIR / "project_notes.txt").read_bytes(),
                "text/plain",
            )
        },
    )
    upload_payload = upload_response.json()

    response = client.post(
        "/tasks/execute",
        json={
            "user_input": "Please summarize this document",
            "task_type": "document_summary",
            "context": {"document_id": upload_payload["document_id"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "document_summary"
    assert payload["status"] == "success"
    assert payload["structured_result"]["file_type"] == "txt"


def test_legacy_tasks_execute_rejects_direct_file_path_outside_uploads() -> None:
    response = client.post(
        "/tasks/execute",
        json={
            "user_input": "Please summarize this document",
            "task_type": "document_summary",
            "context": {"file_path": str(DATA_DIR / "project_notes.txt")},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "document_summary"
    assert payload["status"] == "need_more_info"
    assert "uploads directory" in payload["structured_result"]["error"]


def test_legacy_tasks_execute_rejects_deprecated_weather() -> None:
    response = client.post(
        "/tasks/execute",
        json={"user_input": "Shanghai weather", "task_type": "weather_query"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["structured_result"]["deprecated"] is True
