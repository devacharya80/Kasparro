
import json
from types import SimpleNamespace

import pytest

from app.agent import agent_v1


ANSWER = "The speaker weighs 800 grams."


@pytest.fixture
def mock_dependencies(monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "execute_tool",
        lambda tool_name, arguments: {
            "brand_id": arguments["brand_id"],
            "catalogue_version": "test-v1",
            "facts": [
                {
                    "fact_id": "F001",
                    "value": "800 grams",
                }
            ],
        },
    )

    def fake_runner(messages, **kwargs):
        assert messages[0]["content"] == agent_v1.SYSTEM_PROMPT
        return SimpleNamespace(
            final_text=json.dumps(
                {
                    "claims": [
                        {
                            "quote": ANSWER,
                            "label": "supported",
                            "fact_ids": ["F001"],
                            "reason": "The catalogue confirms the weight.",
                        }
                    ]
                }
            ),
            model_turns=2,
            tool_calls=[],
        )

    monkeypatch.setattr(agent_v1, "run_tool_calling", fake_runner)


def test_classify_answer_returns_valid_prediction(mock_dependencies):
    result = agent_v1.classify_answer(
        answer_id="A01",
        brand_id="cleartrail",
        answer_text=ANSWER,
    )

    assert result["answer_id"] == "A01"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["quote"] == ANSWER
    assert result["claims"][0]["start_char"] == 0
    assert result["claims"][0]["end_char"] == len(ANSWER)
    assert result["claims"][0]["claim_id"] == f"A01:0:{len(ANSWER)}"


def test_rejects_claim_not_found_in_original_answer(
    mock_dependencies,
    monkeypatch,
):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text=json.dumps(
                {
                    "claims": [
                        {
                            "quote": "This sentence is not in the answer.",
                            "label": "supported",
                            "fact_ids": ["F001"],
                            "reason": "Test.",
                        }
                    ]
                }
            )
        ),
    )

    with pytest.raises(
        agent_v1.AgentOutputError,
        match="does not match the original answer",
    ):
        agent_v1.classify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            answer_text=ANSWER,
        )


def test_rejects_invalid_json(mock_dependencies, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text="not valid JSON"
        ),
    )

    with pytest.raises(agent_v1.AgentOutputError, match="valid JSON"):
        agent_v1.classify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            answer_text=ANSWER,
        )


def test_rejects_unknown_fact_id(mock_dependencies, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text=json.dumps(
                {
                    "claims": [
                        {
                            "quote": ANSWER,
                            "label": "supported",
                            "fact_ids": ["F999"],
                            "reason": "Test.",
                        }
                    ]
                }
            )
        ),
    )

    with pytest.raises(
        agent_v1.AgentOutputError,
        match="unknown catalogue",
    ):
        agent_v1.classify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            answer_text=ANSWER,
        )
