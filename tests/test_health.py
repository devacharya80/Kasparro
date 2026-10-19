from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_success() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_returns_expected_payload() -> None:
    response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "service": "kasparro-assessment-api",
    }