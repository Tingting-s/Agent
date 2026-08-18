from fastapi.testclient import TestClient

from app.main import app


def test_unhandled_exception_returns_standard_error_payload(monkeypatch) -> None:
    from app.api import routes

    def raise_runtime_error(_request):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(routes.task_service, "handle_request", raise_runtime_error)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/tasks/execute",
        json={"user_input": "Shanghai weather today", "context": {"city": "Shanghai"}},
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error_code"] == "internal_server_error"
    assert payload["message"] == "Internal server error."
