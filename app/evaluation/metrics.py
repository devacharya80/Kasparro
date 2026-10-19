from __future__ import annotations

from collections import Counter
from typing import Any

from app.evaluation.matcher import claim_label, match_claims


LABELS = ("supported", "contradicted", "unknown")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
    """Assessment-defined precision, recall and F1 using global counts."""
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, gold)
    f1 = _ratio(2 * tp, predicted + gold)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_claims(
    gold_claims: list[dict[str, Any]],
    predicted_claims: list[dict[str, Any]],
    *,
    answer_errors: dict[str, Any] | list[str] | set[str] | None = None,
    scheduled_answer_ids: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Score exact spans and labels, and include assessment-required diagnostics.

    Predictions for answer IDs with a recorded answer-level error are removed
    before scoring. All gold claims remain in the denominator, so a failed
    answer contributes missed gold claims but no accepted predictions.
    """
    if scheduled_answer_ids is not None:
        scheduled = {str(value) for value in scheduled_answer_ids}
        observed = [str(row.get("answer_id", "")) for row in gold_claims]
        if any(answer_id not in scheduled for answer_id in observed):
            raise ValueError("Gold claims include an answer outside the scheduled split.")

    if answer_errors is None:
        error_ids: set[str] = set()
    elif isinstance(answer_errors, dict):
        error_ids = {str(value) for value in answer_errors}
    else:
        error_ids = {str(value) for value in answer_errors}

    accepted_predictions = [
        claim for claim in predicted_claims
        if str(claim.get("answer_id", "")) not in error_ids
    ]
    matched = match_claims(gold_claims, accepted_predictions)
    pairs = matched["matches"]

    correct_label_pairs = [
        (gold, pred) for gold, pred in pairs
        if str(gold.get("expected_label", "")).strip().lower() == claim_label(pred)
    ]
    m = len(pairs)
    g = len(gold_claims)
    p = len(accepted_predictions)
    c = len(correct_label_pairs)

    matched_wrong_labels = [
        {
            "gold_claim_id": str(gold.get("claim_id", _claim_id(gold))),
            "predicted_claim_id": str(pred.get("claim_id", _claim_id(pred))),
            "answer_id": str(gold.get("answer_id", "")),
            "start_char": gold.get("start_char"),
            "end_char": gold.get("end_char"),
            "expected_label": str(gold.get("expected_label", "")).strip().lower(),
            "predicted_label": claim_label(pred),
        }
        for gold, pred in pairs
        if str(gold.get("expected_label", "")).strip().lower() != claim_label(pred)
    ]

    unmatched_prediction_ids = [
        str(item.get("claim_id", _claim_id(item)))
        for item in matched["unmatched_predictions"]
    ]
    missed_gold_ids = [
        str(item.get("claim_id", _claim_id(item)))
        for item in matched["unmatched_gold"]
    ]
    answer_ids_in_errors = sorted(error_ids)
    label_counts = Counter(claim_label(pred) for pred in accepted_predictions)

    return {
        "gold_claims": g,
        "predicted_claims": p,
        "exact_span_matches": m,
        "joint_correct_claims": c,
        "extraction": _prf(m, p, g),
        "joint": _prf(c, p, g),
        "claim_coverage": _ratio(c, g),
        "missed_gold_claim_ids": missed_gold_ids,
        "unmatched_prediction_ids": unmatched_prediction_ids,
        "matched_wrong_label_ids": matched_wrong_labels,
        "unmatched_predictions": len(matched["unmatched_predictions"]),
        "missed_gold_claims": len(matched["unmatched_gold"]),
        "answer_error_count": len(answer_ids_in_errors),
        "answer_error_ids": answer_ids_in_errors,
        "accepted_predictions_by_label": dict(sorted(label_counts.items())),
    }


def _claim_id(claim: dict[str, Any]) -> str:
    answer_id = str(claim.get("answer_id", ""))
    start = claim.get("start_char", "")
    end = claim.get("end_char", "")
    return f"{answer_id}:{start}:{end}"
