import json
from types import SimpleNamespace

import pytest

from app.agent import agent_v1

ANSWER = "The speaker weighs 800 grams."
FACT = {"fact_id": "F001", "brand_id": "cleartrail", "value": "800 grams"}


def tool_call():
    return SimpleNamespace(
        brand_id="cleartrail",
        facts=[FACT],
        facts_returned=1,
        tool_name="get_brand_facts",
    )


@pytest.fixture
def fake_runner(monkeypatch):
    captured = {}

    def runner(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            final_text=json.dumps({"claims": [{
                "quote": ANSWER,
                "label": "supported",
                "fact_ids": ["F001"],
                "reason": "The catalogue confirms the weight.",
            }]}),
            model_turns=2,
            tool_calls=[tool_call()],
            exchanges=[{"turn": 1, "tool_results": [{"brand_id": "cleartrail"}]}],
        )

    monkeypatch.setattr(agent_v1, "run_tool_calling", runner)
    return captured


def test_classify_answer_returns_exact_valid_prediction(fake_runner):
    trace = []
    result = agent_v1.classify_answer(
        answer_id="A01",
        brand_id="cleartrail",
        question="How heavy is the speaker?",
        answer_text=ANSWER,
        trace=trace,
    )
    assert result["answer_id"] == "A01"
    assert result["claims"][0]["quote"] == ANSWER
    assert result["claims"][0]["start_char"] == 0
    assert result["claims"][0]["end_char"] == len(ANSWER)
    assert result["claims"][0]["claim_id"] == f"A01:0:{len(ANSWER)}"
    sent = json.loads(fake_runner["messages"][1]["content"])
    assert sent["question"] == "How heavy is the speaker?"
    assert sent["ai_answer"] == ANSWER
    assert trace[0]["turn"] == 1


def test_rejects_quote_not_in_source(fake_runner, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text=json.dumps({"claims": [{
                "quote": "This sentence is not in the answer.",
                "label": "supported",
                "fact_ids": ["F001"],
                "reason": "Test.",
            }]}),
            tool_calls=[tool_call()],
            exchanges=[],
        ),
    )
    with pytest.raises(agent_v1.AgentOutputError, match="source order"):
        agent_v1.classify_answer(
            answer_id="A01", brand_id="cleartrail", answer_text=ANSWER
        )


def test_rejects_invalid_json(fake_runner, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text="not valid JSON", tool_calls=[tool_call()], exchanges=[]
        ),
    )
    with pytest.raises(agent_v1.AgentOutputError, match="valid JSON"):
        agent_v1.classify_answer(
            answer_id="A01", brand_id="cleartrail", answer_text=ANSWER
        )


def test_rejects_fact_id_not_returned_by_tool(fake_runner, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text=json.dumps({"claims": [{
                "quote": ANSWER,
                "label": "supported",
                "fact_ids": ["F999"],
                "reason": "Test.",
            }]}),
            tool_calls=[tool_call()],
            exchanges=[],
        ),
    )
    with pytest.raises(agent_v1.AgentOutputError, match="unknown catalogue"):
        agent_v1.classify_answer(
            answer_id="A01", brand_id="cleartrail", answer_text=ANSWER
        )


def test_requires_a_real_tool_call(fake_runner, monkeypatch):
    monkeypatch.setattr(
        agent_v1,
        "run_tool_calling",
        lambda messages, **kwargs: SimpleNamespace(
            final_text='{"claims":[]}', tool_calls=[], exchanges=[]
        ),
    )
    with pytest.raises(Exception, match="catalogue lookup"):
        agent_v1.classify_answer(
            answer_id="A01", brand_id="cleartrail", answer_text=ANSWER
        )
