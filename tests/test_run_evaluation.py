from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from app.evaluation import run_evaluation


ANSWER = "The speaker weighs 800 grams."
CLAIM = {
    "claim_id": f"A01:0:{len(ANSWER)}",
    "start_char": 0,
    "end_char": len(ANSWER),
    "quote": ANSWER,
    "label": "supported",
    "fact_ids": ["cleartrail.mini.weight"],
    "reason": "The catalogue confirms the mass.",
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_data(tmp_path: Path, split: str = "development") -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "answers.csv",
        ["answer_id", "split", "brand_id", "question", "ai_answer"],
        [{
            "answer_id": "A01",
            "split": split,
            "brand_id": "cleartrail",
            "question": "How heavy is the speaker?",
            "ai_answer": ANSWER,
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
            "value": "800",
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
            "claim_id": CLAIM["claim_id"],
            "start_char": "0",
            "end_char": str(len(ANSWER)),
            "quote": ANSWER,
            "expected_label": "supported",
            "evidence_fact_ids": "cleartrail.mini.weight",
            "explanation": "The fact establishes the mass.",
            "difficulty_tags": "basic",
        }],
    )
    return data_dir


def _configure_runner(monkeypatch, tmp_path: Path, split: str = "development") -> None:
    data_dir = _prepare_data(tmp_path, split)
    monkeypatch.setattr(run_evaluation, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_evaluation, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(run_evaluation, "source_revision", lambda: "test-revision")
    monkeypatch.setattr(run_evaluation, "get_provider_name", lambda: "fake")
    monkeypatch.setattr(run_evaluation, "get_model_name", lambda: "fake-model")


def test_development_cli_saves_metrics_outcomes_and_sanitized_exchange(monkeypatch, tmp_path, capsys):
    _configure_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_evaluation", "--agent", "v1", "--split", "development"])

    def fake_classify(*, answer_id, brand_id, question, answer_text, trace):
        assert answer_id == "A01"
        assert brand_id == "cleartrail"
        assert answer_text == ANSWER
        trace.append({
            "turn": 1,
            "tool_requests": [{"tool_name": "get_brand_facts", "arguments": {"brand_id": "cleartrail"}}],
            "tool_results": [{"brand_id": "cleartrail", "fact_version": "speaker-catalogue-v4", "facts_returned": 1, "fact_ids": ["cleartrail.mini.weight"]}],
        })
        return {"answer_id": answer_id, "claims": [CLAIM]}

    monkeypatch.setattr(run_evaluation, "classify_v1", fake_classify)
    assert run_evaluation.main() == 0
    report = json.loads((tmp_path / "artifacts/v1_development_run.json").read_text(encoding="utf-8"))
    assert report["scheduled_answers"] == 1
    assert report["successful_answers"] == 1
    assert report["failed_answers"] == 0
    assert report["metrics"]["joint"]["f1"] == 1.0
    assert report["sanitised_exchanges"][0]["turns"][0]["tool_results"][0]["fact_ids"] == ["cleartrail.mini.weight"]
    assert "OPENAI_API_KEY" not in json.dumps(report)
    outcomes = json.loads((tmp_path / "artifacts/v1_development_outcomes.json").read_text(encoding="utf-8"))
    assert outcomes["A01"]["status"] == "succeeded"
    assert capsys.readouterr().out.count("Report:") == 1


def test_failed_development_answer_has_no_predictions_and_gold_is_missed(monkeypatch, tmp_path):
    _configure_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_evaluation", "--agent", "v1", "--split", "development"])

    def fail(*, answer_id, brand_id, question, answer_text, trace):
        raise TimeoutError("sensitive provider details should not be persisted")

    monkeypatch.setattr(run_evaluation, "classify_v1", fail)
    assert run_evaluation.main() == 2
    report = json.loads((tmp_path / "artifacts/v1_development_run.json").read_text(encoding="utf-8"))
    assert report["failed_answers"] == 1
    assert report["successful_answers"] == 0
    assert report["accepted_predictions"] == 0
    assert report["metrics"]["missed_gold_claim_ids"] == [CLAIM["claim_id"]]
    assert "sensitive provider details" not in json.dumps(report)
    predictions = json.loads((tmp_path / "artifacts/v1_development_predictions.json").read_text(encoding="utf-8"))
    assert predictions == {}


def test_check_cli_never_writes_quality_metrics(monkeypatch, tmp_path):
    _configure_runner(monkeypatch, tmp_path, split="check")
    monkeypatch.setattr(sys, "argv", ["run_evaluation", "--agent", "v2", "--split", "check"])

    def fake_classify(*, answer_id, brand_id, question, answer_text, trace):
        trace.append({"turn": 1, "tool_requests": [], "tool_results": []})
        return {"answer_id": answer_id, "claims": [CLAIM]}

    monkeypatch.setattr(run_evaluation, "classify_v2", fake_classify)
    assert run_evaluation.main() == 0
    report = json.loads((tmp_path / "artifacts/v2_check_run.json").read_text(encoding="utf-8"))
    assert report["scheduled_answers"] == 1
    assert report["metrics"] is None
    assert "missed_gold_claim_ids" not in report
