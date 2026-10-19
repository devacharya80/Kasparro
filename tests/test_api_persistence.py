from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import main
from app.db.models import Base
from app.db.session import get_db


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _test_data_dir(tmp_path: Path, split: str = "development") -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    answer_text = "ClearTrail Mini weighs 580 grams."
    _write_csv(
        data_dir / "answers.csv",
        ["answer_id", "split", "brand_id", "question", "ai_answer"],
        [{
            "answer_id": "A01",
            "split": split,
            "brand_id": "cleartrail",
            "question": "How much does it weigh?",
            "ai_answer": answer_text,
        }],
    )
    _write_csv(
        data_dir / "brand_facts.csv",
        ["fact_id", "brand_id", "model_id", "field", "value", "value_type", "unit", "condition", "valid_from", "valid_to"],
        [{
            "fact_id": "cleartrail.mini.weight",
            "brand_id": "cleartrail",
            "model_id": "mini",
            "field": "weight",
            "value": "580",
            "value_type": "number",
            "unit": "grams",
            "condition": "speaker only",
            "valid_from": "2026-01-01",
            "valid_to": "2026-12-31",
        }],
    )
    _write_csv(
        data_dir / "development_gold.csv",
        ["answer_id", "claim_id", "start_char", "end_char", "quote", "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags"],
        [{
            "answer_id": "A01",
            "claim_id": f"A01:0:{len(answer_text)}",
            "start_char": "0",
            "end_char": str(len(answer_text)),
            "quote": answer_text,
            "expected_label": "supported",
            "evidence_fact_ids": "cleartrail.mini.weight",
            "explanation": "The supplied fact confirms the mass.",
            "difficulty_tags": "unit",
        }],
    )
    return data_dir


def _install_test_database(monkeypatch, tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.sqlite3'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(main, "get_db", override_get_db)
    main.app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main, "DATA_DIR", _test_data_dir(tmp_path))
    monkeypatch.setattr(main, "_provider_details", lambda: ("fake", "deterministic-test-model"))
    monkeypatch.setattr(main, "_source_revision", lambda: "test-revision")
    monkeypatch.setattr(main, "_classify", lambda version, **kwargs: {
        "answer_id": kwargs["answer_id"],
        "claims": [{
            "claim_id": f"{kwargs['answer_id']}:0:{len(kwargs['answer_text'])}",
            "start_char": 0,
            "end_char": len(kwargs["answer_text"]),
            "quote": kwargs["answer_text"],
            "label": "supported",
            "fact_ids": ["cleartrail.mini.weight"],
            "reason": "The catalogue confirms the mass.",
        }],
    })


def test_evaluation_run_persists_results_and_returns_joined_records(monkeypatch, tmp_path):
    _install_test_database(monkeypatch, tmp_path)
    client = TestClient(main.app)
    created = client.post(
        "/evaluation-runs",
        json={"split": "development", "agent_version": "v1"},
    )
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["status"] == "completed"
    assert run["scheduled_answers"] == 1
    assert run["successful_answers"] == 1
    assert run["failed_answers"] == 0
    assert run["metrics"]["gold_claims"] == 1
    assert run["metrics"]["joint"]["f1"] == 1.0

    fetched = client.get(f"/evaluation-runs/{run['run_id']}")
    assert fetched.status_code == 200, fetched.text
    detail = fetched.json()
    assert detail["status"] == "completed"
    assert detail["answers"][0]["answer_id"] == "A01"
    assert detail["answers"][0]["status"] == "succeeded"
    assert detail["answers"][0]["claims"][0]["quote"] == "ClearTrail Mini weighs 580 grams."

    main.app.dependency_overrides.clear()


def test_check_run_persists_predictions_without_quality_metrics(monkeypatch, tmp_path):
    _install_test_database(monkeypatch, tmp_path)
    data_dir = tmp_path / "data"
    answers_path = data_dir / "answers.csv"
    rows = list(csv.DictReader(answers_path.open(encoding="utf-8", newline="")))
    rows[0]["split"] = "check"
    _write_csv(
        answers_path,
        ["answer_id", "split", "brand_id", "question", "ai_answer"],
        rows,
    )

    client = TestClient(main.app)
    created = client.post(
        "/evaluation-runs",
        json={"split": "check", "agent_version": "v2"},
    )
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["status"] == "completed"
    assert run["metrics"] is None
    assert run["successful_answers"] == 1

    fetched = client.get(f"/evaluation-runs/{run['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["metrics"] is None
    assert fetched.json()["answers"][0]["claims"]

    main.app.dependency_overrides.clear()
