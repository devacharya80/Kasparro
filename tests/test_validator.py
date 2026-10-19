import pytest

from app.agent.validator import PredictionValidationError, validate_prediction

ANSWER_ID = "A01"
ANSWER = "The speaker weighs 800 grams.\nIt has Bluetooth."
FACT_IDS = ["F001", "F002"]


def claim(answer, start, end, *, label="supported", fact_ids=None, reason="The catalogue confirms this claim."):
    return {
        "claim_id": f"{ANSWER_ID}:{start}:{end}",
        "start_char": start,
        "end_char": end,
        "quote": answer[start:end],
        "label": label,
        "fact_ids": list(fact_ids if fact_ids is not None else ["F001"]),
        "reason": reason,
    }


def prediction(answer=ANSWER, claims=None):
    if claims is None:
        first_end = len("The speaker weighs 800 grams.")
        second_start = answer.index("It has Bluetooth.")
        claims = [
            claim(answer, 0, first_end),
            claim(answer, second_start, len(answer), fact_ids=["F002"]),
        ]
    return {"answer_id": ANSWER_ID, "claims": claims}


def validate(value, answer=ANSWER):
    return validate_prediction(value, answer_id=ANSWER_ID, answer_text=answer, valid_fact_ids=FACT_IDS)


def test_accepts_valid_ordered_prediction():
    result = prediction()
    assert validate(result) == result


def test_accepts_empty_claim_list():
    assert validate(prediction(claims=[]))["claims"] == []


def test_rejects_wrong_answer_id():
    value = prediction()
    value["answer_id"] = "A99"
    with pytest.raises(PredictionValidationError, match="answer_id"):
        validate(value)


def test_rejects_invalid_offsets():
    value = prediction()
    value["claims"][0]["end_char"] = len(ANSWER) + 1
    with pytest.raises(PredictionValidationError, match="offsets"):
        validate(value)


def test_rejects_quote_mismatch():
    value = prediction()
    value["claims"][0]["quote"] = "A different sentence."
    with pytest.raises(PredictionValidationError, match="quote"):
        validate(value)


def test_rejects_claim_id_mismatch():
    value = prediction()
    value["claims"][0]["claim_id"] = "wrong-id"
    with pytest.raises(PredictionValidationError, match="claim_id"):
        validate(value)


def test_rejects_unsupported_label():
    value = prediction()
    value["claims"][0]["label"] = "maybe"
    with pytest.raises(PredictionValidationError, match="label"):
        validate(value)


def test_rejects_unknown_fact_id():
    value = prediction()
    value["claims"][0]["fact_ids"] = ["F999"]
    with pytest.raises(PredictionValidationError, match="unknown catalogue"):
        validate(value)


def test_rejects_overlapping_claims():
    first = claim(ANSWER, 0, 20)
    second = claim(ANSWER, 10, 30)
    value = prediction(claims=[first, second])
    with pytest.raises(PredictionValidationError, match="source order"):
        validate(value)


def test_rejects_non_integer_offsets():
    value = prediction()
    value["claims"][0]["start_char"] = True
    with pytest.raises(PredictionValidationError, match="integers"):
        validate(value)


def test_rejects_claims_out_of_source_order():
    valid = prediction()["claims"]
    value = prediction(claims=list(reversed(valid)))
    with pytest.raises(PredictionValidationError, match="source order"):
        validate(value)


def test_rejects_duplicate_fact_ids():
    value = prediction()
    value["claims"][0]["fact_ids"] = ["F001", "F001"]
    with pytest.raises(PredictionValidationError, match="duplicates"):
        validate(value)


def test_rejects_reason_longer_than_two_sentences():
    value = prediction()
    value["claims"][0]["reason"] = "First sentence. Second sentence. Third sentence."
    with pytest.raises(PredictionValidationError, match="at most two"):
        validate(value)


def test_rejects_quote_with_leading_whitespace():
    answer = " The speaker weighs 800 grams."
    value = {
        "answer_id": ANSWER_ID,
        "claims": [{
            "claim_id": f"{ANSWER_ID}:0:{len(answer)}",
            "start_char": 0,
            "end_char": len(answer),
            "quote": answer,
            "label": "supported",
            "fact_ids": ["F001"],
            "reason": "The catalogue confirms this claim.",
        }],
    }
    # Exact-slice equality alone is not enough: the source quote itself may not have padding.
    with pytest.raises(PredictionValidationError, match="leading or trailing whitespace"):
        validate(value, answer=answer)
