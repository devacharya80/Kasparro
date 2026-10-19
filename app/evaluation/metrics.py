from typing import Any

from app.evaluation.matcher import claim_label, match_claims


LABELS = ("supported", "contradicted", "unknown")


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _macro_f1(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> float:
    per_label_f1 = []

    for label in LABELS:
        tp = sum(
            1 for gold, pred in pairs
            if gold.get("expected_label") == label and claim_label(pred) == label
        )
        fp = sum(
            1 for gold, pred in pairs
            if gold.get("expected_label") != label and claim_label(pred) == label
        )
        fn = sum(
            1 for gold, pred in pairs
            if gold.get("expected_label") == label and claim_label(pred) != label
        )
        per_label_f1.append(_scores(tp, fp, fn)["f1"])

    return round(sum(per_label_f1) / len(per_label_f1), 4)


def evaluate_claims(
    gold_claims: list[dict[str, Any]],
    predicted_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    matched = match_claims(gold_claims, predicted_claims)
    pairs = matched["matches"]

    label_correct = sum(
        1
        for gold, pred in pairs
        if str(gold.get("expected_label", "")).strip().lower()
        == claim_label(pred)
    )

    extraction = _scores(
        tp=len(pairs),
        fp=len(matched["unmatched_predictions"]),
        fn=len(matched["unmatched_gold"]),
    )

    joint = _scores(
        tp=label_correct,
        fp=len(predicted_claims) - label_correct,
        fn=len(gold_claims) - label_correct,
    )

    return {
        "gold_claims": len(gold_claims),
        "predicted_claims": len(predicted_claims),
        "exact_span_matches": len(pairs),
        "label_correct_on_matched_spans": label_correct,
        "classification_accuracy": round(
            label_correct / len(pairs), 4
        ) if pairs else 0.0,
        "classification_macro_f1": _macro_f1(pairs),
        "extraction": extraction,
        "joint": joint,
        "unmatched_predictions": len(matched["unmatched_predictions"]),
        "missed_gold_claims": len(matched["unmatched_gold"]),
    }
