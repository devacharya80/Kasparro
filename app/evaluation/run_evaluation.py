from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.agent.agent_v1 import classify_answer as classify_v1
from app.agent.agent_v2 import classify_answer as classify_v2
from app.agent.provider import REQUEST_TIMEOUT_SECONDS, get_model_name, get_provider_name
from app.catalogue_version import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION
from app.evaluation.loader import load_answers, load_gold_claims
from app.evaluation.metrics import evaluate_claims

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> str | None:
    if os.getenv("GIT_COMMIT_SHA"):
        return os.environ["GIT_COMMIT_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sanitise_error(error: Exception) -> str:
    name = type(error).__name__
    if "timeout" in name.lower() or isinstance(error, TimeoutError):
        return f"{name}: model request timed out (configured timeout {REQUEST_TIMEOUT_SECONDS:g}s)"
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]?\s*\S+",
        r"\1=[REDACTED]",
        str(error),
    )
    return f"{name}: {message[:240]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned live assessment evaluation.")
    parser.add_argument("--agent", choices=("v1", "v2"), required=True)
    parser.add_argument("--split", choices=("development", "check"), required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; omit for the full assessment.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    answers_path = DATA_DIR / "answers.csv"
    gold_path = DATA_DIR / "development_gold.csv"
    facts_path = DATA_DIR / "brand_facts.csv"
    answers = [
        row for row in load_answers(answers_path)
        if row.get("split") == args.split
    ]
    if args.limit:
        answers = answers[:args.limit]
    if not answers:
        parser.error("No answers found for the selected split.")

    scheduled_ids = [str(row["answer_id"]) for row in answers]
    if len(set(scheduled_ids)) != len(scheduled_ids):
        parser.error("Dataset has duplicate answer IDs; refusing to run.")

    classifier = classify_v1 if args.agent == "v1" else classify_v2
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    predictions: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    latencies_ms: dict[str, int] = {}
    exchanges: list[dict[str, Any]] = []

    for index, row in enumerate(answers, start=1):
        answer_id = str(row["answer_id"])
        print(f"[{index}/{len(answers)}] {answer_id}: running {args.agent}", flush=True)
        started = perf_counter()
        trace: list[dict[str, Any]] = []
        try:
            result = classifier(
                answer_id=answer_id,
                brand_id=str(row["brand_id"]),
                question=str(row["question"]),
                answer_text=str(row["ai_answer"]),
                trace=trace,
            )
            claims = result["claims"]
            predictions[answer_id] = [
                {**claim, "answer_id": answer_id} for claim in claims
            ]
            outcomes[answer_id] = {
                "status": "succeeded",
                "accepted_claim_count": len(claims),
                "error": None,
            }
        except Exception as error:
            errors[answer_id] = sanitise_error(error)
            predictions.pop(answer_id, None)
            outcomes[answer_id] = {
                "status": "failed",
                "accepted_claim_count": 0,
                "error": errors[answer_id],
            }
        finally:
            latencies_ms[answer_id] = round((perf_counter() - started) * 1000)

        exchanges.append({"answer_id": answer_id, "turns": trace})
        save_json(ARTIFACTS / f"{args.agent}_{args.split}_predictions.json", predictions)
        save_json(ARTIFACTS / f"{args.agent}_{args.split}_outcomes.json", outcomes)
        save_json(ARTIFACTS / f"{args.agent}_{args.split}_errors.json", errors)
        save_json(ARTIFACTS / f"{args.agent}_{args.split}_exchanges.json", exchanges)

    flat_predictions = [
        claim for answer_id in scheduled_ids for claim in predictions.get(answer_id, [])
    ]
    report: dict[str, Any] = {
        "run_id": run_id,
        "agent_version": args.agent,
        "split": args.split,
        "provider": get_provider_name(),
        "model": get_model_name(),
        "dataset_version": DATASET_VERSION,
        "fact_version": FACT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": f"{args.agent}-prompt-v1",
        "settings": {
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_model_turns": 3,
            "temperature": 0,
            "scheduled_answer_ids": scheduled_ids,
            "smoke_test_limit": args.limit or None,
        },
        "input_hashes": {
            "answers.csv": sha256_file(answers_path),
            "brand_facts.csv": sha256_file(facts_path),
            **({"development_gold.csv": sha256_file(gold_path)} if args.split == "development" else {}),
        },
        "source_revision": source_revision(),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scheduled_answers": len(answers),
        "successful_answers": sum(outcome["status"] == "succeeded" for outcome in outcomes.values()),
        "failed_answers": len(errors),
        "answer_outcomes": outcomes,
        "errors": errors,
        "latency_ms": latencies_ms,
        "accepted_predictions": len(flat_predictions),
        "predictions_file": f"artifacts/{args.agent}_{args.split}_predictions.json",
        "outcomes_file": f"artifacts/{args.agent}_{args.split}_outcomes.json",
        "errors_file": f"artifacts/{args.agent}_{args.split}_errors.json",
        "exchanges_file": f"artifacts/{args.agent}_{args.split}_exchanges.json",
        "sanitised_exchanges": exchanges,
    }

    if args.split == "development":
        gold = load_gold_claims(gold_path)
        scheduled_set = set(scheduled_ids)
        selected_gold = [claim for claim in gold if str(claim["answer_id"]) in scheduled_set]
        metrics = evaluate_claims(
            selected_gold,
            flat_predictions,
            answer_errors=errors,
            scheduled_answer_ids=scheduled_ids,
        )
        report["metrics"] = metrics
        report["wrong_label_claims"] = metrics["matched_wrong_label_ids"]
        report["missed_gold_claim_ids"] = metrics["missed_gold_claim_ids"]
        report["unmatched_prediction_ids"] = metrics["unmatched_prediction_ids"]
    else:
        report["metrics"] = None
        report["note"] = "Final-check quality is withheld for reviewer-only scoring."

    report_path = ARTIFACTS / f"{args.agent}_{args.split}_run.json"
    save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
