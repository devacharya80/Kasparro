from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_classify_rejects_missing_fields():
    response = client.post("/classify", json={"answer_id": "A01"})
    assert response.status_code == 422


def test_classify_rejects_unsupported_agent_version():
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail",
        "question": "How heavy is it?",
        "answer_text": "The speaker weighs 1 kg.",
        "agent_version": "v3",
    })
    assert response.status_code == 422


def test_classify_rejects_unknown_fields():
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail",
        "question": "How heavy is it?",
        "answer_text": "The speaker weighs 1 kg.",
        "gold_claims": [],
    })
    assert response.status_code == 422


def test_classify_requires_question_context():
    response = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail",
        "answer_text": "The speaker weighs 1 kg.",
    })
    assert response.status_code == 422


def test_get_run_rejects_invalid_uuid_before_database_access():
    response = client.get("/evaluation-runs/not-a-uuid")
    assert response.status_code == 422


def test_health_endpoint_does_not_require_database():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_primary_brand_maps_to_404(monkeypatch):
    from app import main
    from app.agent.catalogue import UnknownBrandError

    def fail(**kwargs):
        raise UnknownBrandError("Unknown brand_id: not-real")

    monkeypatch.setattr(main, "classify_v1", fail)
    response = client.post("/classify", json={
        "answer_id": "A01",
        "brand_id": "not-real",
        "question": "Does it have Wi-Fi?",
        "answer_text": "The speaker has Wi-Fi.",
        "agent_version": "v1",
    })
    assert response.status_code == 404


def test_provider_timeout_maps_to_504(monkeypatch):
    from app import main

    def fail(**kwargs):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(main, "classify_v1", fail)
    response = client.post("/classify", json={
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "Does it have Wi-Fi?",
        "answer_text": "The speaker has Wi-Fi.",
        "agent_version": "v1",
    })
    assert response.status_code == 504


def test_invalid_model_output_maps_to_502(monkeypatch):
    from app import main
    from app.agent.agent_v1 import AgentOutputError

    def fail(**kwargs):
        raise AgentOutputError("not valid model output")

    monkeypatch.setattr(main, "classify_v1", fail)
    response = client.post("/classify", json={
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "Does it have Wi-Fi?",
        "answer_text": "The speaker has Wi-Fi.",
        "agent_version": "v1",
    })
    assert response.status_code == 502
