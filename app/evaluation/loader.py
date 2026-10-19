import csv
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV file without modifying the source data."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_answers(path: str | Path = "data/answers.csv") -> list[dict[str, Any]]:
    return read_csv(path)


def load_gold_claims(
    path: str | Path = "data/development_gold.csv",
) -> list[dict[str, Any]]:
    claims = read_csv(path)

    for row in claims:
        try:
            row["start_char"] = int(row["start_char"])
            row["end_char"] = int(row["end_char"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid character offsets in gold claim: {row}"
            ) from exc

        row["expected_label"] = str(
            row.get("expected_label", "")
        ).strip().lower()

    return claims


def load_development_data(
    data_dir: str | Path = "data",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return development answers and gold claims as separate datasets."""
    data_dir = Path(data_dir)
    answers = load_answers(data_dir / "answers.csv")
    gold_claims = load_gold_claims(data_dir / "development_gold.csv")

    answer_ids = {str(row["answer_id"]) for row in answers}
    unknown_ids = {
        str(row["answer_id"])
        for row in gold_claims
        if str(row["answer_id"]) not in answer_ids
    }

    if unknown_ids:
        raise ValueError(
            f"Gold claims reference unknown answer IDs: {sorted(unknown_ids)}"
        )

    return answers, gold_claims