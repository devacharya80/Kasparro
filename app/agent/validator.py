
from __future__ import annotations

from typing import Any, Iterable


VALID_LABELS = frozenset(
    {
        "supported",
        "contradicted",
        "unknown",
    }
)

REQUIRED_CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "start_char",
        "end_char",
        "quote",
        "label",
        "fact_ids",
        "reason",
    }
)


class PredictionValidationError(ValueError):
    """Raised when an agent prediction violates the output contract."""


def validate_prediction(
    prediction: Any,
    *,
    answer_id: str,
    answer_text: str,
    valid_fact_ids: Iterable[str],
) -> dict[str, Any]:
    """
    Validate a prediction against the original answer and catalogue.

    Character offsets use Python string indexing:
    start_char is inclusive and end_char is exclusive.

    The prediction is returned unchanged if every check passes.
    Invalid predictions raise PredictionValidationError.
    """

    if not isinstance(answer_id, str) or not answer_id.strip():
        raise PredictionValidationError(
            "answer_id must be a non-empty string."
        )

    if not isinstance(answer_text, str):
        raise PredictionValidationError(
            "answer_text must be a string."
        )

    if not isinstance(prediction, dict):
        raise PredictionValidationError(
            "Prediction must be a JSON object."
        )

    if prediction.get("answer_id") != answer_id:
        raise PredictionValidationError(
            "Prediction answer_id does not match the requested answer."
        )

    claims = prediction.get("claims")

    if not isinstance(claims, list):
        raise PredictionValidationError(
            "Prediction claims must be a list."
        )

    catalogue_ids: set[str] = set()

    for fact_id in valid_fact_ids:
        if not isinstance(fact_id, str) or not fact_id:
            raise PredictionValidationError(
                "The catalogue contains an invalid fact ID."
            )
        catalogue_ids.add(fact_id)

    seen_claim_ids: set[str] = set()
    accepted_spans: list[tuple[int, int]] = []

    for index, claim in enumerate(claims):
        location = f"claims[{index}]"

        if not isinstance(claim, dict):
            raise PredictionValidationError(
                f"{location} must be a JSON object."
            )

        missing_fields = REQUIRED_CLAIM_FIELDS - claim.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise PredictionValidationError(
                f"{location} is missing required fields: {missing}."
            )

        start = claim["start_char"]
        end = claim["end_char"]

        # bool is a subclass of int in Python, so reject it explicitly.
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise PredictionValidationError(
                f"{location} offsets must be integers."
            )

        if not 0 <= start < end <= len(answer_text):
            raise PredictionValidationError(
                f"{location} has invalid character offsets."
            )

        expected_claim_id = f"{answer_id}:{start}:{end}"

        if claim["claim_id"] != expected_claim_id:
            raise PredictionValidationError(
                f"{location} claim_id must be "
                f"{expected_claim_id!r}."
            )

        if expected_claim_id in seen_claim_ids:
            raise PredictionValidationError(
                f"{location} duplicates claim ID "
                f"{expected_claim_id!r}."
            )

        seen_claim_ids.add(expected_claim_id)

        quote = claim["quote"]

        if not isinstance(quote, str):
            raise PredictionValidationError(
                f"{location} quote must be a string."
            )

        if quote != answer_text[start:end]:
            raise PredictionValidationError(
                f"{location} quote does not exactly match "
                "the original answer at the supplied offsets."
            )

        label = claim["label"]

        if not isinstance(label, str) or label not in VALID_LABELS:
            raise PredictionValidationError(
                f"{location} label must be one of: "
                "supported, contradicted, unknown."
            )

        fact_ids = claim["fact_ids"]

        if not isinstance(fact_ids, list):
            raise PredictionValidationError(
                f"{location} fact_ids must be a list."
            )

        for fact_id in fact_ids:
            if not isinstance(fact_id, str):
                raise PredictionValidationError(
                    f"{location} contains a non-string fact ID."
                )

            if fact_id not in catalogue_ids:
                raise PredictionValidationError(
                    f"{location} references unknown catalogue "
                    f"fact ID {fact_id!r}."
                )

        reason = claim["reason"]

        if not isinstance(reason, str) or not reason.strip():
            raise PredictionValidationError(
                f"{location} reason must be a non-empty string."
            )

        for previous_start, previous_end in accepted_spans:
            overlaps = start < previous_end and previous_start < end

            if overlaps:
                raise PredictionValidationError(
                    f"{location} overlaps another claim span."
                )

        accepted_spans.append((start, end))

    return prediction
