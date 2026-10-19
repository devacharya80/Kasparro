from app.agent.agent_v2 import V2_PROMPT_ADDENDUM


def test_v2_prompt_addendum_targets_atomic_claim_evidence():
    prompt = V2_PROMPT_ADDENDUM.lower()
    assert "every material factual component" in prompt
    assert "comparison" in prompt
    assert "never transfer facts across models" in prompt
