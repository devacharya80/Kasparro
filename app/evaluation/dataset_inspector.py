from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

EXPECTED_COLUMNS = {
    "answers.csv": [
        "answer_id",
        "split",
        "brand_id",
        "question",
        "ai_answer",
    ],
    "brand_facts.csv": [
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
    ],
    "development_gold.csv": [
        "answer_id",
        "claim_id",
        "start_char",
        "end_char",
        "quote",
        "expected_label",
        "evidence_fact_ids",
        "explanation",
        "difficulty_tags",
    ],
}

EXPECTED_ROW_COUNTS = {
    "answers.csv": 50,
    "brand_facts.csv": 132,
    "development_gold.csv": 376,
}

VALID_LABELS = {
    "supported",
    "contradicted",
    "unknown",
}

VALID_VALUE_TYPES = {
    "number",
    "boolean",
    "string",
}


def read_csv_rows(
    path: Path,
    expected_columns: list[str],
) -> list[dict[str, str]]:
    """Read CSV records without modifying their field contents."""

    with path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")

        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Header mismatch in {path.name}.\n"
                f"Expected: {expected_columns}\n"
                f"Received: {reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []

        for record_number, row in enumerate(reader, start=1):
            # DictReader uses a None key when a row has extra fields.
            if None in row:
                raise ValueError(
                    f"Unexpected extra fields in {path.name}, "
                    f"record {record_number}"
                )

            missing_fields = [
                column
                for column in expected_columns
                if row.get(column) is None
            ]

            if missing_fields:
                raise ValueError(
                    f"Missing fields in {path.name}, "
                    f"record {record_number}: {missing_fields}"
                )

            # Keep values exactly as parsed. Do not strip, normalize,
            # rewrite, or otherwise alter the original answer text.
            rows.append(
                {
                    column: row[column]
                    for column in expected_columns
                }
            )

        return rows


def find_duplicates(values: list[str]) -> list[str]:
    """Return values that appear more than once."""

    counts = Counter(values)

    return sorted(
        value
        for value, count in counts.items()
        if count > 1
    )


def inspect_datasets() -> list[str]:
    """Inspect the supplied datasets without writing to them."""

    issues: list[str] = []

    datasets: dict[str, list[dict[str, str]]] = {}

    # Load only the exact, expected CSV files.
    for filename, columns in EXPECTED_COLUMNS.items():
        path = DATA_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing dataset: {path}\n"
                "Place the original assessment CSVs in the data/ folder."
            )

        datasets[filename] = read_csv_rows(path, columns)

    answers = datasets["answers.csv"]
    facts = datasets["brand_facts.csv"]
    gold = datasets["development_gold.csv"]

    answers_by_id = {
        row["answer_id"]: row
        for row in answers
    }

    fact_ids = {
        row["fact_id"]
        for row in facts
    }

    # ---------------------------------------------------------
    # 1. Dataset sizes and file hashes
    # ---------------------------------------------------------

    print("=" * 65)
    print("KASPARRO DATASET INSPECTION")
    print("=" * 65)
    print("Mode: read-only")
    print()

    for filename, rows in datasets.items():
        path = DATA_DIR / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        print(f"{filename}")
        print(f"  Records: {len(rows)}")
        print(f"  SHA-256: {digest}")

        expected_count = EXPECTED_ROW_COUNTS[filename]

        if len(rows) != expected_count:
            issues.append(
                f"{filename}: expected {expected_count} records, "
                f"found {len(rows)}"
            )

        print()

    # ---------------------------------------------------------
    # 2. Inspect answers.csv
    # ---------------------------------------------------------

    print("-" * 65)
    print("ANSWERS")
    print("-" * 65)

    answer_ids = [row["answer_id"] for row in answers]
    duplicates = find_duplicates(answer_ids)

    if duplicates:
        issues.append(f"Duplicate answer IDs: {duplicates}")

    split_counts = Counter(
        row["split"]
        for row in answers
    )

    print(f"Split counts: {dict(sorted(split_counts.items()))}")

    if split_counts != Counter(
        {"development": 40, "check": 10}
    ):
        issues.append(
            "Unexpected answer split counts: "
            f"{dict(split_counts)}"
        )

    if any(not row["ai_answer"] for row in answers):
        issues.append("One or more answers have an empty ai_answer.")

    answer_lengths = [
        len(row["ai_answer"])
        for row in answers
    ]

    embedded_newlines = sum(
        "\n" in row["ai_answer"]
        or "\r" in row["ai_answer"]
        for row in answers
    )

    print(
        "Answer character lengths: "
        f"min={min(answer_lengths)}, "
        f"max={max(answer_lengths)}"
    )
    print(
        "Answers containing embedded newlines: "
        f"{embedded_newlines}"
    )

    # ---------------------------------------------------------
    # 3. Inspect brand_facts.csv
    # ---------------------------------------------------------

    print()
    print("-" * 65)
    print("BRAND FACTS")
    print("-" * 65)

    fact_id_list = [
        row["fact_id"]
        for row in facts
    ]

    duplicates = find_duplicates(fact_id_list)

    if duplicates:
        issues.append(f"Duplicate fact IDs: {duplicates}")

    brand_counts = Counter(
        row["brand_id"]
        for row in facts
    )

    model_ids = {
        row["model_id"]
        for row in facts
    }

    value_type_counts = Counter(
        row["value_type"]
        for row in facts
    )

    print(f"Brands and fact counts: {dict(sorted(brand_counts.items()))}")
    print(f"Unique model IDs: {len(model_ids)}")
    print(f"Value types: {dict(sorted(value_type_counts.items()))}")

    invalid_types = sorted(
        set(value_type_counts) - VALID_VALUE_TYPES
    )

    if invalid_types:
        issues.append(
            f"Unsupported catalogue value types: {invalid_types}"
        )

    invalid_boolean_values = sorted(
        {
            row["value"]
            for row in facts
            if row["value_type"] == "boolean"
            and row["value"] not in {"true", "false"}
        }
    )

    if invalid_boolean_values:
        issues.append(
            "Invalid Boolean values: "
            f"{invalid_boolean_values}"
        )

    # ---------------------------------------------------------
    # 4. Inspect development_gold.csv
    # ---------------------------------------------------------

    print()
    print("-" * 65)
    print("DEVELOPMENT GOLD")
    print("-" * 65)

    label_counts = Counter(
        row["expected_label"]
        for row in gold
    )

    print(f"Claim label counts: {dict(sorted(label_counts.items()))}")
    print(f"Unique answer IDs in gold: {len(set(r['answer_id'] for r in gold))}")

    unsupported_labels = sorted(
        set(label_counts) - VALID_LABELS
    )

    if unsupported_labels:
        issues.append(
            f"Unsupported gold labels: {unsupported_labels}"
        )

    duplicate_spans = find_duplicates(
        [
            f"{row['answer_id']}:{row['start_char']}:{row['end_char']}"
            for row in gold
        ]
    )

    if duplicate_spans:
        issues.append(
            f"Duplicate gold claim spans: {duplicate_spans[:10]}"
        )

    valid_span_count = 0
    referenced_fact_count = 0
    invalid_fact_references: list[str] = []

    spans_by_answer: dict[
        str, list[tuple[int, int, str]]
    ] = defaultdict(list)

    for row in gold:
        answer_id = row["answer_id"]
        claim_id = row["claim_id"]

        answer = answers_by_id.get(answer_id)

        if answer is None:
            issues.append(
                f"{claim_id}: references unknown answer {answer_id}"
            )
            continue

        if answer["split"] != "development":
            issues.append(
                f"{claim_id}: gold label is attached to a "
                f"non-development answer ({answer_id})"
            )

        try:
            start = int(row["start_char"])
            end = int(row["end_char"])
        except ValueError:
            issues.append(
                f"{claim_id}: offsets are not integers"
            )
            continue

        expected_claim_id = f"{answer_id}:{start}:{end}"

        if claim_id != expected_claim_id:
            issues.append(
                f"{claim_id}: expected claim ID "
                f"{expected_claim_id}"
            )

        original_answer = answer["ai_answer"]

        if not (
            0 <= start < end <= len(original_answer)
        ):
            issues.append(
                f"{claim_id}: offsets are outside the answer bounds"
            )
            continue

        actual_quote = original_answer[start:end]

        if actual_quote != row["quote"]:
            issues.append(
                f"{claim_id}: quote does not match the original answer"
            )
            continue

        valid_span_count += 1

        spans_by_answer[answer_id].append(
            (start, end, claim_id)
        )

        evidence_ids = [
            value
            for value in row["evidence_fact_ids"].split("|")
            if value
        ]

        for evidence_id in evidence_ids:
            if evidence_id not in fact_ids:
                invalid_fact_references.append(
                    f"{claim_id} -> {evidence_id}"
                )
            else:
                referenced_fact_count += 1

    if invalid_fact_references:
        issues.append(
            "Unknown catalogue references in gold: "
            f"{invalid_fact_references[:10]}"
        )

    # Confirm gold spans are in source order and do not overlap.
    for answer_id, spans in spans_by_answer.items():
        starts = [start for start, _, _ in spans]

        if starts != sorted(starts):
            issues.append(
                f"{answer_id}: gold spans are not in source order"
            )

        sorted_spans = sorted(spans)

        for previous, current in zip(
            sorted_spans,
            sorted_spans[1:],
        ):
            previous_start, previous_end, previous_id = previous
            current_start, current_end, current_id = current

            if previous_end > current_start:
                issues.append(
                    f"Overlapping gold spans in {answer_id}: "
                    f"{previous_id} and {current_id}"
                )

    print(
        f"Gold spans with valid bounds and exact quotes: "
        f"{valid_span_count}/{len(gold)}"
    )
    print(
        f"Valid catalogue evidence references: "
        f"{referenced_fact_count}"
    )

    # ---------------------------------------------------------
    # 5. Final result
    # ---------------------------------------------------------

    print()
    print("=" * 65)

    if issues:
        print(f"INSPECTION FAILED: {len(issues)} issue(s)")
        print()

        for issue in issues:
            print(f"- {issue}")

        print("=" * 65)
        return issues

    print("INSPECTION PASSED")
    print("All dataset integrity checks completed successfully.")
    print("No CSV files were modified.")
    print("=" * 65)

    return []


def main() -> int:
    try:
        issues = inspect_datasets()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"DATASET INSPECTION ERROR: {error}")
        return 1

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())