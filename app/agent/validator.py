from __future__ import annotations

import re
from typing import Any, Iterable


VALID_LABELS = frozenset({"supported", "contradicted", "unknown"})
REQUIRED_CLAIM_FIELDS = frozenset({
    "claim_id", "start_char", "end_char", "quote", "label", "fact_ids", "reason"
})


class PredictionValidationError(ValueError):
    """Raised when an agent prediction violates the output contract."""


def validate_prediction(
    prediction: Any,
    *,
    answer_id: str,
    answer_text: str,
    valid_fact_ids: Iterable[str],
) -> dict[str, Any]:
    """Validate exact answer spans, labels, evidence references and ordering."""
    if not isinstance(answer_id, str) or not answer_id.strip():
        raise PredictionValidationError("answer_id must be a non-empty string.")
    if not isinstance(answer_text, str):
        raise PredictionValidationError("answer_text must be a string.")
    if not isinstance(prediction, dict):
        raise PredictionValidationError("Prediction must be a JSON object.")
    if set(prediction) != {"answer_id", "claims"}:
        raise PredictionValidationError("Prediction must contain only answer_id and claims.")
    if prediction.get("answer_id") != answer_id:
        raise PredictionValidationError("Prediction answer_id does not match the requested answer.")
    claims = prediction.get("claims")
    if not isinstance(claims, list):
        raise PredictionValidationError("Prediction claims must be a list.")

    catalogue_ids: set[str] = set()
    for fact_id in valid_fact_ids:
        if not isinstance(fact_id, str) or not fact_id:
            raise PredictionValidationError("The catalogue contains an invalid fact ID.")
        catalogue_ids.add(fact_id)

    seen_claim_ids: set[str] = set()
    seen_spans: set[tuple[int, int]] = set()
    previous_start = -1
    previous_end = -1

    for index, claim in enumerate(claims):
        location = f"claims[{index}]"
        if not isinstance(claim, dict):
            raise PredictionValidationError(f"{location} must be a JSON object.")
        if set(claim) != REQUIRED_CLAIM_FIELDS:
            missing = REQUIRED_CLAIM_FIELDS - claim.keys()
            extra = claim.keys() - REQUIRED_CLAIM_FIELDS
            details = []
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if extra:
                details.append("unexpected: " + ", ".join(sorted(extra)))
            raise PredictionValidationError(
                f"{location} has invalid fields ({'; '.join(details)})."
            )

        start = claim["start_char"]
        end = claim["end_char"]
        if (
            isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
        ):
            raise PredictionValidationError(f"{location} offsets must be integers.")
        if not 0 <= start < end <= len(answer_text):
            raise PredictionValidationError(f"{location} has invalid character offsets.")
        expected_claim_id = f"{answer_id}:{start}:{end}"
        if claim["claim_id"] != expected_claim_id:
            raise PredictionValidationError(f"{location} claim_id must be {expected_claim_id!r}.")
        if expected_claim_id in seen_claim_ids:
            raise PredictionValidationError(f"{location} duplicates claim ID {expected_claim_id!r}.")
        seen_claim_ids.add(expected_claim_id)

        if (start, end) in seen_spans:
            raise PredictionValidationError(f"{location} duplicates a claim span.")
        if start <= previous_start or start < previous_end:
            raise PredictionValidationError(
                f"{location} claims must be non-overlapping and in increasing source order."
            )
        seen_spans.add((start, end))
        previous_start, previous_end = start, end

        quote = claim["quote"]
        if not isinstance(quote, str):
            raise PredictionValidationError(f"{location} quote must be a string.")
        if quote != answer_text[start:end]:
            raise PredictionValidationError(
                f"{location} quote does not exactly match the original answer at the supplied offsets."
            )
        if quote != quote.strip():
            raise PredictionValidationError(f"{location} quote must not include leading or trailing whitespace.")

        label = claim["label"]
        if not isinstance(label, str) or label not in VALID_LABELS:
            raise PredictionValidationError(
                f"{location} label must be one of: supported, contradicted, unknown."
            )

        fact_ids = claim["fact_ids"]
        if not isinstance(fact_ids, list):
            raise PredictionValidationError(f"{location} fact_ids must be a list.")
        if len(fact_ids) != len(set(fact_ids)):
            raise PredictionValidationError(f"{location} fact_ids must not contain duplicates.")
        for fact_id in fact_ids:
            if not isinstance(fact_id, str):
                raise PredictionValidationError(f"{location} contains a non-string fact ID.")
            if fact_id not in catalogue_ids:
                raise PredictionValidationError(f"{location} references unknown catalogue fact ID {fact_id!r}.")

        reason = claim["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise PredictionValidationError(f"{location} reason must be a non-empty string.")
        sentence_parts = [part for part in re.split(r"(?<=[.!?])\s+", reason.strip()) if part]
        if len(sentence_parts) > 2:
            raise PredictionValidationError(f"{location} reason must be at most two sentences.")

    return prediction
