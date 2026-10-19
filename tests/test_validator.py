
import pytest

from app.agent.validator import (
    PredictionValidationError,
    validate_prediction,
)


ANSWER_ID = "A01"
ANSWER = "The speaker weighs 800 grams."
FACT_IDS = ["F001"]


def make_claim(
    answer: str = ANSWER,
    start: int = 0,
    end: int | None = None,
) -> dict:
    if end is None:
        end = len(answer)

    return {
        "claim_id": f"{ANSWER_ID}:{start}:{end}",
        "start_char": start,
        "end_char": end,
        "quote": answer[start:end],
        "label": "supported",
        "fact_ids": ["F001"],
        "reason": "The catalogue confirms the claim.",
    }


def make_prediction(
    answer: str = ANSWER,
    claims: list | None = None,
) -> dict:
    if claims is None:
        claims = [make_claim(answer)]

    return {
        "answer_id": ANSWER_ID,
        "claims": claims,
    }


def validate(prediction: dict, answer: str = ANSWER) -> dict:
    return validate_prediction(
        prediction,
        answer_id=ANSWER_ID,
        answer_text=answer,
        valid_fact_ids=FACT_IDS,
    )


def test_accepts_valid_prediction() -> None:
    prediction = make_prediction()
    assert validate(prediction) == prediction


def test_accepts_empty_claim_list() -> None:
    assert validate(make_prediction(claims=[]))["claims"] == []


def test_rejects_wrong_answer_id() -> None:
    prediction = make_prediction()
    prediction["answer_id"] = "A99"

    with pytest.raises(PredictionValidationError, match="answer_id"):
        validate(prediction)


def test_rejects_invalid_offsets() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["end_char"] = len(ANSWER) + 1

    with pytest.raises(PredictionValidationError, match="offsets"):
        validate(prediction)


def test_rejects_quote_mismatch() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["quote"] = "A different sentence."

    with pytest.raises(PredictionValidationError, match="quote"):
        validate(prediction)


def test_rejects_invalid_claim_id() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["claim_id"] = "wrong-id"

    with pytest.raises(PredictionValidationError, match="claim_id"):
        validate(prediction)


def test_rejects_unsupported_label() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["label"] = "maybe"

    with pytest.raises(PredictionValidationError, match="label"):
        validate(prediction)


def test_rejects_unknown_fact_id() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["fact_ids"] = ["F999"]

    with pytest.raises(PredictionValidationError, match="unknown catalogue"):
        validate(prediction)


def test_rejects_overlapping_claims() -> None:
    second = make_claim(start=10, end=20)
    prediction = make_prediction(claims=[make_claim(), second])

    with pytest.raises(PredictionValidationError, match="overlaps"):
        validate(prediction)


def test_rejects_missing_required_field() -> None:
    prediction = make_prediction()
    del prediction["claims"][0]["reason"]

    with pytest.raises(PredictionValidationError, match="missing required fields"):
        validate(prediction)


def test_rejects_non_integer_offsets() -> None:
    prediction = make_prediction()
    prediction["claims"][0]["start_char"] = True

    with pytest.raises(PredictionValidationError, match="integers"):
        validate(prediction)
