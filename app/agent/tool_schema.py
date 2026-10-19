
from __future__ import annotations

from typing import Any

from app.agent.catalogue import BrandFactCatalogue, TOOL_NAME


def build_brand_facts_tool_schema(
    catalogue: BrandFactCatalogue | None = None,
) -> list[dict[str, Any]]:
    """Build the function schema presented to the LLM."""

    if catalogue is None:
        catalogue = BrandFactCatalogue()

    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "Retrieve factual catalogue records for a known "
                    "speaker brand. Use these records to verify "
                    "product specifications, prices, capabilities, "
                    "accessories, policies, and comparisons. "
                    "Do not infer missing facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand_id": {
                            "type": "string",
                            "enum": list(catalogue.brand_ids),
                            "description": (
                                "Exact ID of a known brand in the "
                                "supplied product catalogue."
                            ),
                        }
                    },
                    "required": ["brand_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]
