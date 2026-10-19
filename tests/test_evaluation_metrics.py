from app.evaluation.matcher import match_claims
from app.evaluation.metrics import evaluate_claims


def gold(answer_id, start, end, label):
    return {
        "answer_id": answer_id,
        "start_char": start,
        "end_char": end,
        "expected_label": label,
    }


def prediction(answer_id, start, end, label):
    return {
        "answer_id": answer_id,
        "start_char": start,
        "end_char": end,
        "label": label,
    }


def test_exact_span_and_correct_label():
    gold_claims = [gold("A1", 0, 10, "supported")]
    predictions = [prediction("A1", 0, 10, "supported")]

    result = evaluate_claims(gold_claims, predictions)

    assert result["exact_span_matches"] == 1
    assert result["label_correct_on_matched_spans"] == 1
    assert result["extraction"]["f1"] == 1.0
    assert result["joint"]["f1"] == 1.0


def test_wrong_label_has_extraction_match_but_no_joint_match():
    gold_claims = [gold("A1", 0, 10, "supported")]
    predictions = [prediction("A1", 0, 10, "contradicted")]

    result = evaluate_claims(gold_claims, predictions)

    assert result["exact_span_matches"] == 1
    assert result["classification_accuracy"] == 0.0
    assert result["extraction"]["f1"] == 1.0
    assert result["joint"]["f1"] == 0.0


def test_missing_and_extra_claims_are_counted():
    gold_claims = [
        gold("A1", 0, 10, "supported"),
        gold("A1", 20, 30, "unknown"),
    ]
    predictions = [
        prediction("A1", 0, 10, "supported"),
        prediction("A1", 40, 50, "unknown"),
    ]

    result = evaluate_claims(gold_claims, predictions)

    assert result["exact_span_matches"] == 1
    assert result["unmatched_predictions"] == 1
    assert result["missed_gold_claims"] == 1
    assert result["extraction"]["f1"] == 0.5


def test_duplicate_prediction_does_not_match_gold_twice():
    gold_claims = [gold("A1", 0, 10, "supported")]
    predictions = [
        prediction("A1", 0, 10, "supported"),
        prediction("A1", 0, 10, "supported"),
    ]

    result = evaluate_claims(gold_claims, predictions)

    assert result["exact_span_matches"] == 1
    assert result["unmatched_predictions"] == 1
