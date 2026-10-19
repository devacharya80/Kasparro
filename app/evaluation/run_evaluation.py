from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.agent.agent_v1 import classify_answer as classify_v1
from app.agent.agent_v2 import classify_answer as classify_v2
from app.catalogue_version import DATASET_VERSION, EVALUATOR_VERSION
from app.evaluation.loader import load_answers, load_gold_claims
from app.evaluation.metrics import evaluate_claims


ARTIFACTS = Path("artifacts")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an assessment agent on one dataset split.")
    parser.add_argument("--agent", choices=("v1", "v2"), required=True)
    parser.add_argument("--split", choices=("development", "check"), required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    answers = [
        row for row in load_answers()
        if str(row.get("split", "")).strip().lower() == args.split
    ]
    if args.limit:
        answers = answers[:args.limit]
    if not answers:
        parser.error("No answers found for the selected split.")

    classifier = classify_v1 if args.agent == "v1" else classify_v2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = ARTIFACTS / f"{args.agent}_{args.split}_predictions.json"
    predictions: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    latencies_ms: dict[str, int] = {}

    for index, row in enumerate(answers, start=1):
        answer_id = str(row["answer_id"])
        print(f"[{index}/{len(answers)}] {answer_id}: running {args.agent}", flush=True)
        started = perf_counter()
        try:
            result = classifier(
                answer_id=answer_id,
                brand_id=str(row["brand_id"]),
                answer_text=str(row["ai_answer"]),
            )
            predictions[answer_id] = [
                {**claim, "answer_id": answer_id} for claim in result["claims"]
            ]
        except Exception as error:
            errors[answer_id] = f"{type(error).__name__}: {str(error)[:500]}"
        finally:
            latencies_ms[answer_id] = round((perf_counter() - started) * 1000)
        # Save every iteration so a stopped live run leaves a useful checkpoint.
        save_json(output_path, predictions)

    report: dict[str, Any] = {
        "run_id": run_id,
        "agent_version": args.agent,
        "split": args.split,
        "provider": os.getenv("LLM_PROVIDER", "groq"),
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "source_revision": os.getenv("GIT_COMMIT_SHA"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_count": len(answers),
        "succeeded_answers": len(predictions),
        "failed_answers": len(errors),
        "errors": errors,
        "latency_ms": latencies_ms,
        "predictions_file": str(output_path).replace("\\", "/"),
    }

    # The check split must never be scored with development gold labels.
    if args.split == "development":
        gold = load_gold_claims()
        selected = {str(row["answer_id"]) for row in answers}
        selected_gold = [claim for claim in gold if str(claim["answer_id"]) in selected]
        predicted_claims = [claim for claims in predictions.values() for claim in claims]
        report["metrics"] = evaluate_claims(selected_gold, predicted_claims)
    else:
        report["metrics"] = None
        report["note"] = "Check-split scores are intentionally unavailable to the candidate."

    report_path = ARTIFACTS / f"{args.agent}_{args.split}_run.json"
    save_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
