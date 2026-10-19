from collections import defaultdict
from typing import Any


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def span_key(claim: dict[str, Any]) -> tuple[str, int, int] | None:
    answer_id = claim.get("answer_id")
    start = _integer(claim.get("start_char"))
    end = _integer(claim.get("end_char"))

    if answer_id is None or start is None or end is None:
        return None

    return str(answer_id), start, end


def claim_label(claim: dict[str, Any]) -> str:
    value = (
        claim.get("label")
        or claim.get("predicted_label")
        or claim.get("expected_label")
        or claim.get("classification")
        or ""
    )
    return str(value).strip().lower()


def match_claims(
    gold_claims: list[dict[str, Any]],
    predicted_claims: list[dict[str, Any]],
) -> dict[str, list]:
    """Match claims by exact span while safely handling duplicate predictions."""
    gold_by_span: dict[tuple[str, int, int], list[int]] = defaultdict(list)

    for index, gold in enumerate(gold_claims):
        key = span_key(gold)
        if key is not None:
            gold_by_span[key].append(index)

    used_gold: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_predictions: list[dict[str, Any]] = []

    for prediction in predicted_claims:
        key = span_key(prediction)
        candidates = gold_by_span.get(key, []) if key is not None else []
        available = next(
            (index for index in candidates if index not in used_gold),
            None,
        )

        if available is None:
            unmatched_predictions.append(prediction)
            continue

        used_gold.add(available)
        matches.append((gold_claims[available], prediction))

    unmatched_gold = [
        gold for index, gold in enumerate(gold_claims)
        if index not in used_gold
    ]

    return {
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_gold": unmatched_gold,
    }
