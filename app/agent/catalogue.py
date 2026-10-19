from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


# Assessment-defined catalogue version.
FACT_VERSION = "speaker-catalogue-v4"

TOOL_NAME = "get_brand_facts"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOGUE_PATH = PROJECT_ROOT / "data" / "brand_facts.csv"

EXPECTED_COLUMNS = [
    "fact_id",
    "brand_id",
    "model_id",
    "field",
    "value",
    "value_type",
    "unit",
    "condition",
    "valid_from",
    "valid_to",
]

VALID_VALUE_TYPES = {"number", "boolean", "string"}


class CatalogueError(ValueError):
    """Base exception for catalogue lookup and validation errors."""


class UnknownBrandError(CatalogueError):
    """Raised when a requested brand does not exist in the catalogue."""


class UnknownToolError(CatalogueError):
    """Raised when an unsupported tool is requested."""


class BrandFactCatalogue:
    """Read-only catalogue loaded from the supplied CSV file."""

    def __init__(
        self,
        csv_path: Path = DEFAULT_CATALOGUE_PATH,
    ) -> None:
        self.csv_path = Path(csv_path)
        self._facts_by_brand: dict[
            str, list[dict[str, str]]
        ] = defaultdict(list)

        self._load()

    def _load(self) -> None:
        """Load catalogue records without changing their values."""

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Catalogue file not found: {self.csv_path}"
            )

        with self.csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != EXPECTED_COLUMNS:
                raise CatalogueError(
                    "Catalogue columns do not match the expected "
                    f"schema. Expected {EXPECTED_COLUMNS}, "
                    f"received {reader.fieldnames}."
                )

            seen_fact_ids: set[str] = set()

            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise CatalogueError(
                        f"Unexpected extra fields at CSV line "
                        f"{row_number}."
                    )

                if any(row.get(column) is None for column in EXPECTED_COLUMNS):
                    raise CatalogueError(
                        f"Missing field value at CSV line {row_number}."
                    )

                # The checks above establish that these fields are strings.
                fact: dict[str, str] = {
                    column: row[column]
                    for column in EXPECTED_COLUMNS
                }  # type: ignore[misc]

                fact_id = fact["fact_id"]
                brand_id = fact["brand_id"]
                value_type = fact["value_type"]

                if not fact_id or not brand_id:
                    raise CatalogueError(
                        f"Missing fact_id or brand_id at "
                        f"CSV line {row_number}."
                    )

                if fact_id in seen_fact_ids:
                    raise CatalogueError(
                        f"Duplicate fact ID: {fact_id}"
                    )

                seen_fact_ids.add(fact_id)

                if value_type not in VALID_VALUE_TYPES:
                    raise CatalogueError(
                        f"Invalid value_type {value_type!r} "
                        f"for fact {fact_id}."
                    )

                if (
                    value_type == "boolean"
                    and fact["value"] not in {"true", "false"}
                ):
                    raise CatalogueError(
                        f"Invalid Boolean value for fact {fact_id}: "
                        f"{fact['value']!r}"
                    )

                # Preserve the original catalogue values, including
                # empty unit/condition fields and historical dates.
                self._facts_by_brand[brand_id].append(fact)

        if not self._facts_by_brand:
            raise CatalogueError("The catalogue contains no facts.")

    @property
    def brand_ids(self) -> tuple[str, ...]:
        """Return the known brand IDs in stable sorted order."""

        return tuple(sorted(self._facts_by_brand))

    def get_brand_facts(
        self,
        brand_id: str,
    ) -> dict[str, Any]:
        """
        Return the catalogue records for exactly one known brand.

        The result includes the required catalogue version.
        """

        if not isinstance(brand_id, str) or not brand_id:
            raise CatalogueError(
                "brand_id must be a non-empty string."
            )

        if brand_id not in self._facts_by_brand:
            raise UnknownBrandError(
                f"Unknown brand_id: {brand_id}"
            )

        return {
            "fact_version": FACT_VERSION,
            "brand_id": brand_id,
            "facts": [
                dict(fact)
                for fact in self._facts_by_brand[brand_id]
            ],
        }


@lru_cache(maxsize=1)
def _get_default_catalogue() -> BrandFactCatalogue:
    """Load the project's catalogue once per process."""

    return BrandFactCatalogue()


def get_brand_facts(brand_id: str) -> dict[str, Any]:
    """
    Read-only tool function to retrieve one brand's catalogue facts.

    This function will later be connected to real LLM tool calling.
    """

    return _get_default_catalogue().get_brand_facts(brand_id)


def execute_tool(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate and execute an explicitly allowed tool request.

    This dispatcher never executes arbitrary function names or code.
    """

    if not isinstance(tool_name, str) or tool_name != TOOL_NAME:
        raise UnknownToolError(
            f"Unsupported tool: {tool_name!r}"
        )

    if not isinstance(arguments, Mapping):
        raise CatalogueError(
            "Tool arguments must be a JSON object."
        )

    if set(arguments.keys()) != {"brand_id"}:
        raise CatalogueError(
            "get_brand_facts accepts exactly one argument: brand_id."
        )

    brand_id = arguments["brand_id"]

    if not isinstance(brand_id, str) or not brand_id:
        raise CatalogueError(
            "brand_id must be a non-empty string."
        )

    return get_brand_facts(brand_id)