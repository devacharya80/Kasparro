import pytest

from app.evaluation.matcher import match_claims
from app.evaluation.metrics import evaluate_claims


def gold(answer_id, start, end, label):
    return {
        "answer_id": answer_id,
        "claim_id": f"{answer_id}:{start}:{end}",
        "start_char": start,
        "end_char": end,
        "expected_label": label,
    }


def prediction(answer_id, start, end, label):
    return {
        "answer_id": answer_id,
        "claim_id": f"{answer_id}:{start}:{end}",
        "start_char": start,
        "end_char": end,
        "label": label,
    }


def test_exact_span_and_correct_label():
    result = evaluate_claims(
        [gold("A1", 0, 10, "supported")],
        [prediction("A1", 0, 10, "supported")],
    )
    assert result["extraction"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["joint"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["claim_coverage"] == 1.0


def test_official_arithmetic_example():
    gold_claims = [gold("A1", 0, i + 1, "supported") for i in range(10)]
    predicted = [prediction("A1", 0, i + 1, "supported") for i in range(8)]
    predicted.extend(prediction("A1", 20 + i, 21 + i, "unknown") for i in range(2))
    # Ensure exact spans are 8, only 7 with the correct label.
    predicted[7]["label"] = "contradicted"
    result = evaluate_claims(gold_claims, predicted)
    assert result["exact_span_matches"] == 8
    assert result["joint_correct_claims"] == 7
    assert result["extraction"] == {"precision": 0.8, "recall": 0.8, "f1": 0.8}
    assert result["joint"] == {"precision": 0.7, "recall": 0.7, "f1": 0.7}


def test_failure_answer_predictions_are_discarded_and_gold_still_missed():
    gold_claims = [
        gold("A1", 0, 10, "supported"),
        gold("A2", 0, 10, "unknown"),
    ]
    predictions = [
        prediction("A1", 0, 10, "supported"),
        prediction("A2", 0, 10, "unknown"),
    ]
    result = evaluate_claims(gold_claims, predictions, answer_errors={"A2": "provider timeout"})
    assert result["predicted_claims"] == 1
    assert result["gold_claims"] == 2
    assert result["answer_error_count"] == 1
    assert result["missed_gold_claim_ids"] == ["A2:0:10"]


def test_extra_predictions_and_wrong_label_are_diagnostic():
    gold_claims = [gold("A1", 0, 10, "supported")]
    predicted = [prediction("A1", 0, 10, "unknown"), prediction("A1", 20, 30, "supported")]
    result = evaluate_claims(gold_claims, predicted)
    assert result["unmatched_prediction_ids"] == ["A1:20:30"]
    assert result["matched_wrong_label_ids"][0]["predicted_label"] == "unknown"


def test_empty_evaluation_has_zero_scores():
    result = evaluate_claims([], [])
    assert result["extraction"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert result["joint"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_span_matcher_does_not_truncate_invalid_float_offsets():
    result = match_claims([gold("A1", 0, 10, "supported")], [
        {"answer_id":"A1", "start_char":0.5, "end_char":10, "label":"supported"}
    ])
    assert result["matches"] == []
    assert len(result["unmatched_predictions"]) == 1
