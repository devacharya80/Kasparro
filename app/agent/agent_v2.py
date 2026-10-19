from __future__ import annotations

from typing import Any

from app.agent.agent_v1 import SYSTEM_PROMPT, classify_answer as _classify_v1


V2_PROMPT_ADDENDUM = """
V2 targeted change: enforce atomic evidence checking within compound sentences.
Evaluate every material factual component. A sentence is contradicted if the
catalogue explicitly refutes any material component; otherwise it is unknown
when a material component remains unverified. It is supported only when all
material components are established. Check each named brand/model in a
comparison, both values and comparison direction, dates, market, tax, shipping,
usage conditions and policy scope. Never transfer facts across models, markets,
dates or warranty policies. Preserve exact complete source sentences.
"""


def classify_answer(
    *,
    answer_id: str,
    brand_id: str,
    answer_text: str,
    question: str = "",
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the candidate prompt through the baseline's unchanged pipeline."""
    return _classify_v1(
        answer_id=answer_id,
        brand_id=brand_id,
        answer_text=answer_text,
        question=question,
        trace=trace,
        system_prompt=f"{SYSTEM_PROMPT}\n{V2_PROMPT_ADDENDUM}",
    )
