from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_classify_rejects_missing_fields():
    response = client.post("/classify", json={"answer_id": "A01"})
    assert response.status_code == 422


def test_classify_rejects_unsupported_agent_version():
    response = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "answer_text": "The speaker weighs 1 kg.",
            "agent_version": "v3",
        },
    )
    assert response.status_code == 422


def test_get_run_rejects_invalid_uuid_before_database_access():
    response = client.get("/evaluation-runs/not-a-uuid")
    assert response.status_code == 422
