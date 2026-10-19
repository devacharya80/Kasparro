import csv

import pytest

from app.evaluation.dataset_inspector import read_csv_rows


def test_csv_reader_preserves_commas_and_embedded_newlines(
    tmp_path,
) -> None:
    path = tmp_path / "sample.csv"

    original_answer = (
        'First sentence, with a comma.\n'
        'Second sentence with "quoted text".'
    )

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["answer_id", "ai_answer"],
        )

        writer.writeheader()
        writer.writerow(
            {
                "answer_id": "A01",
                "ai_answer": original_answer,
            }
        )

    rows = read_csv_rows(
        path,
        ["answer_id", "ai_answer"],
    )

    assert len(rows) == 1
    assert rows[0]["answer_id"] == "A01"
    assert rows[0]["ai_answer"] == original_answer


def test_csv_reader_rejects_unexpected_headers(
    tmp_path,
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(
        "answer_id,wrong_column\nA01,value\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Header mismatch"):
        read_csv_rows(
            path,
            ["answer_id", "ai_answer"],
        )


def test_csv_reader_rejects_extra_unquoted_fields(
    tmp_path,
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(
        "answer_id,ai_answer\nA01,first,extra\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra fields"):
        read_csv_rows(
            path,
            ["answer_id", "ai_answer"],
        )