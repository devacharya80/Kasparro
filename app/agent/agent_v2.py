from __future__ import annotations

from typing import Any

from app.agent.agent_v1 import SYSTEM_PROMPT, classify_answer as _classify_v1


V2_PROMPT_ADDENDUM = """
V2 targeted change — stronger atomic evidence checking:
Evaluate every material assertion within a sentence. A compound claim is
supported only when every material assertion is supported. Mark it
contradicted if catalogue evidence explicitly conflicts with any material
part; otherwise mark it unknown when a material part remains unverified.
For comparisons, check both values, comparison direction, relevant date,
market, tax, and shipping assumptions. Do not transfer facts across models,
regions, dates, or warranty policies. Preserve exact complete source quotes.
"""


def classify_answer(
    *,
    answer_id: str,
    brand_id: str,
    answer_text: str,
) -> dict[str, Any]:
    """Run the same validated inference pipeline with the v2 prompt change."""
    return _classify_v1(
        answer_id=answer_id,
        brand_id=brand_id,
        answer_text=answer_text,
        system_prompt=f"{SYSTEM_PROMPT}\n{V2_PROMPT_ADDENDUM}",
    )
