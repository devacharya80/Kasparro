from __future__ import annotations

from typing import Any

from app.agent.agent_v1 import classify_answer


# Targeted v2 change: emphasize atomic claims and evidence for every
# qualifier. The current inference implementation reuses v1 validation
# and extraction plumbing so comparisons isolate the prompt change.
V2_PROMPT_ADDENDUM = """
V2 EVIDENCE CHECK:
Before assigning a label, decompose each sentence mentally into its
material assertions and conditions. A compound claim is supported only
when every material assertion is supported. Use contradicted when at
least one material assertion explicitly conflicts with catalogue data.
Otherwise use unknown when any material assertion cannot be established.
For comparisons, independently verify both values, the direction of the
comparison, the relevant dates/market, and included/excluded costs.
For a condition or warranty claim, require evidence for that exact
condition; do not transfer facts across regions, models, or policies.
"""


def classify_answer(
    *,
    answer_id: str,
    brand_id: str,
    answer_text: str,
) -> dict[str, Any]:
    """Classify an answer with the v2 targeted evidence prompt."""

    # V2 is kept as a separate entry point. Import locally to avoid
    # changing v1's public prompt or behavior.
    from app.agent import agent_v1

    original_prompt = agent_v1.SYSTEM_PROMPT
    try:
        agent_v1.SYSTEM_PROMPT = original_prompt + "\n" + V2_PROMPT_ADDENDUM
        return agent_v1.classify_answer(
            answer_id=answer_id,
            brand_id=brand_id,
            answer_text=answer_text,
        )
    finally:
        agent_v1.SYSTEM_PROMPT = original_prompt
