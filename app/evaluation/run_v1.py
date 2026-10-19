import argparse
import json
from pathlib import Path
from typing import Any

from app.agent.agent_v1 import classify_answer
from app.evaluation.loader import load_development_data
from app.evaluation.metrics import evaluate_claims


ARTIFACTS = Path("artifacts")
CHECKPOINT = ARTIFACTS / "v1_development_predictions.json"
METRICS_FILE = ARTIFACTS / "v1_development_metrics.json"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def extract_claims(result: Any, answer_id: str) -> list[dict[str, Any]]:
    if isinstance(result, list):
        claims = result
    elif isinstance(result, dict):
        claims = result.get("claims")

        if claims is None:
            for key in ("predictions", "claim_predictions", "results"):
                if isinstance(result.get(key), list):
                    claims = result[key]
                    break

        if claims is None and "start_char" in result and "end_char" in result:
            claims = [result]

    else:
        raise ValueError(
            f"{answer_id}: unexpected agent result type: {type(result).__name__}"
        )

    if not isinstance(claims, list):
        raise ValueError(
            f"{answer_id}: agent result does not contain a claims list. "
            f"Returned keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
        )

    normalized = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"{answer_id}: claim {index} is not an object")

        item = dict(claim)
        item.setdefault("answer_id", answer_id)

        if str(item["answer_id"]) != answer_id:
            raise ValueError(
                f"{answer_id}: claim {index} has mismatched answer_id "
                f"{item['answer_id']!r}"
            )

        normalized.append(item)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent v1 on development data")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N development answers (0 means all)",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or greater")

    answers, gold_claims = load_development_data()
    answers = [
        row for row in answers
        if str(row.get("split", "")).strip().lower() == "development"
    ]

    if args.limit:
        answers = answers[:args.limit]

    if not answers:
        raise RuntimeError("No development answers found")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT.exists():
        completed = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise ValueError(f"Invalid checkpoint format: {CHECKPOINT}")
    else:
        completed = {}

    total = len(answers)

    for number, row in enumerate(answers, start=1):
        answer_id = str(row["answer_id"])

        if answer_id in completed:
            print(f"[{number}/{total}] {answer_id}: already saved, skipping")
            continue

        print(f"[{number}/{total}] {answer_id}: calling Agent v1...", flush=True)

        result = classify_answer(
            answer_id=answer_id,
            brand_id=str(row["brand_id"]),
            answer_text=str(row["ai_answer"]),
        )
        claims = extract_claims(result, answer_id)

        completed[answer_id] = claims
        save_json(CHECKPOINT, completed)

        print(
            f"[{number}/{total}] {answer_id}: saved {len(claims)} claims",
            flush=True,
        )

    selected_ids = {str(row["answer_id"]) for row in answers}
    selected_gold = [
        claim for claim in gold_claims
        if str(claim["answer_id"]) in selected_ids
    ]
    predicted_claims = [
        claim
        for answer_id in selected_ids
        for claim in completed.get(answer_id, [])
    ]

    metrics = evaluate_claims(selected_gold, predicted_claims)

    # Only publish official development metrics after all 40 answers are done.
    all_development_ids = {
        str(row["answer_id"])
        for row in load_development_data()[0]
        if str(row.get("split", "")).strip().lower() == "development"
    }

    if all_development_ids.issubset(completed.keys()):
        save_json(METRICS_FILE, metrics)
        print("\n=== AGENT V1 DEVELOPMENT METRICS ===")
        print(json.dumps(metrics, indent=2))
        print(f"\nPredictions: {CHECKPOINT}")
        print(f"Metrics:     {METRICS_FILE}")
    else:
        print("\nPartial run completed. Final metrics have NOT been published.")
        print(json.dumps(metrics, indent=2))
        print(f"Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    main()
