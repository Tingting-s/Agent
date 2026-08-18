from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_service_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "enterprise-document-qa-agent"


def test_validation_error_returns_standard_error_payload() -> None:
    response = client.post("/tasks/execute", json={"task_type": "weather_query"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error_code"] == "request_validation_error"
    assert payload["message"] == "Request validation failed."
    assert payload["details"]["errors"]
