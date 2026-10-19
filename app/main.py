from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.agent.agent_v1 import AgentOutputError, classify_answer as classify_v1
from app.agent.agent_v2 import classify_answer as classify_v2
from app.agent.catalogue import UnknownBrandError
from app.agent.provider import REQUEST_TIMEOUT_SECONDS, get_groq_model
from app.agent.tool_runner import InvalidToolArgumentsError, ToolCallingError
from app.catalogue_version import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION
from app.db.models import AnswerResult, Base, ClaimResult, EvaluationRun
from app.db.session import engine, get_db
from app.evaluation.loader import load_answers, load_gold_claims
from app.evaluation.metrics import evaluate_claims

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
AgentVersion = Literal["v1", "v2"]
Split = Literal["development", "check"]

app = FastAPI(
    title="Kasparro Assessment API",
    description="Evidence-grounded factual claim extraction and evaluation.",
    version="1.1.0",
)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(min_length=1, max_length=16)
    brand_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1)
    answer_text: str
    agent_version: AgentVersion = "v1"

    @field_validator("answer_id", "brand_id", "question")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: Split
    agent_version: AgentVersion = "v1"


class ClaimResponse(BaseModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Literal["supported", "contradicted", "unknown"]
    fact_ids: list[str]
    reason: str


class ClassifyResponse(BaseModel):
    answer_id: str
    claims: list[ClaimResponse]


def _data_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("answers.csv", "brand_facts.csv", "development_gold.csv"):
        path = DATA_DIR / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _source_revision() -> str | None:
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


def _provider_details() -> tuple[str, str]:
    return "groq", get_groq_model()


def _classify(
    version: AgentVersion,
    *,
    answer_id: str,
    brand_id: str,
    question: str,
    answer_text: str,
) -> dict[str, Any]:
    classifier = classify_v1 if version == "v1" else classify_v2
    return classifier(
        answer_id=answer_id,
        brand_id=brand_id,
        question=question,
        answer_text=answer_text,
    )


def _sanitised_error(error: Exception) -> str:
    """Persist a safe error summary, not provider request/response bodies."""
    if isinstance(error, UnknownBrandError):
        return f"UnknownBrandError: {error}"
    name = type(error).__name__
    if isinstance(error, TimeoutError) or "timeout" in name.lower():
        return "ProviderTimeout: model request exceeded its configured timeout."
    if isinstance(error, (ToolCallingError, InvalidToolArgumentsError, AgentOutputError)):
        return f"{name}: {str(error)[:240]}"
    return f"{name}: provider or agent request failed."


def _http_model_error(error: Exception) -> HTTPException:
    from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    if isinstance(error, UnknownBrandError):
        return HTTPException(status_code=404, detail="Unknown primary brand.")
    if isinstance(error, (APITimeoutError, TimeoutError)) or "timeout" in type(error).__name__.lower():
        return HTTPException(status_code=504, detail="LLM provider timed out.")
    if isinstance(error, (APIConnectionError, APIStatusError, RateLimitError)):
        return HTTPException(status_code=502, detail="LLM provider request failed.")
    return HTTPException(
        status_code=502,
        detail=f"{type(error).__name__}: invalid model output or tool request.",
    )


@app.on_event("startup")
def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "kasparro-assessment-api"}


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(request: ClassifyRequest) -> dict[str, Any]:
    try:
        return _classify(
            request.agent_version,
            answer_id=request.answer_id,
            brand_id=request.brand_id,
            question=request.question,
            answer_text=request.answer_text,
        )
    except Exception as error:
        raise _http_model_error(error) from error


@app.post(
    "/evaluation-runs",
    status_code=status.HTTP_201_CREATED,
    tags=["Evaluation"],
)
def create_evaluation_run(
    request: EvaluationRunRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run the fixed split and save exactly one terminal result per answer."""
    try:
        answers = [
            row for row in load_answers(DATA_DIR / "answers.csv")
            if row.get("split") == request.split
        ]
        if not answers:
            raise HTTPException(status_code=422, detail="The selected split is empty.")

        scheduled_ids = [str(row["answer_id"]) for row in answers]
        if len(set(scheduled_ids)) != len(scheduled_ids):
            raise HTTPException(status_code=500, detail="Dataset contains duplicate answer IDs.")

        provider, model = _provider_details()
        run = EvaluationRun(
            dataset_split=request.split,
            agent_version=request.agent_version,
            provider=provider,
            model=model,
            dataset_version=DATASET_VERSION,
            fact_version=FACT_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            prompt_version=f"{request.agent_version}-prompt-v1",
            settings={
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "max_model_turns": 3,
                "temperature": 0,
                "answer_order": "source CSV order",
            },
            input_hashes=_data_hashes(),
            source_revision=_source_revision(),
            status="running",
            metrics=None,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
    except HTTPException:
        raise
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=503, detail="Unable to initialise evaluation run or database.") from error

    started_run = time.perf_counter()
    errors: dict[str, str] = {}
    predictions: list[dict[str, Any]] = []
    try:
        for row in answers:
            answer_id = str(row["answer_id"])
            answer_result = AnswerResult(
                run_id=run.id,
                answer_id=answer_id,
                status="running",
            )
            db.add(answer_result)
            db.flush()
            started = time.perf_counter()
            try:
                prediction = _classify(
                    request.agent_version,
                    answer_id=answer_id,
                    brand_id=str(row["brand_id"]),
                    question=str(row["question"]),
                    answer_text=str(row["ai_answer"]),
                )
                answer_result.status = "succeeded"
                answer_result.prediction = prediction
                predictions.append(prediction)
                for claim in prediction["claims"]:
                    db.add(ClaimResult(
                        answer_result_id=answer_result.id,
                        claim_id=claim["claim_id"],
                        start_char=claim["start_char"],
                        end_char=claim["end_char"],
                        quote=claim["quote"],
                        label=claim["label"],
                        fact_ids=claim["fact_ids"],
                        reason=claim["reason"],
                    ))
            except Exception as error:
                answer_result.status = "failed"
                answer_result.error_summary = _sanitised_error(error)
                errors[answer_id] = answer_result.error_summary
            finally:
                answer_result.latency_ms = round((time.perf_counter() - started) * 1000)
            db.commit()

        stored_ids = [
            answer.answer_id
            for answer in db.scalars(select(AnswerResult).where(AnswerResult.run_id == run.id)).all()
        ]
        if len(stored_ids) != len(scheduled_ids) or set(stored_ids) != set(scheduled_ids):
            raise RuntimeError("Persistence invariant failed: each scheduled answer must have one outcome.")

        if request.split == "development":
            gold = load_gold_claims(DATA_DIR / "development_gold.csv")
            scheduled_set = set(scheduled_ids)
            selected_gold = [claim for claim in gold if str(claim["answer_id"]) in scheduled_set]
            predicted_claims = [
                {**claim, "answer_id": prediction["answer_id"]}
                for prediction in predictions for claim in prediction["claims"]
            ]
            run.metrics = evaluate_claims(
                selected_gold,
                predicted_claims,
                answer_errors=errors,
                scheduled_answer_ids=scheduled_ids,
            )
        else:
            run.metrics = None

        run.status = "completed" if not errors else "completed_with_errors"
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(run)
    except Exception as error:
        db.rollback()
        run.status = "failed"
        run.error_summary = _sanitised_error(error)
        run.completed_at = datetime.now(timezone.utc)
        try:
            db.add(run)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
        raise HTTPException(status_code=500, detail="Evaluation run failed; inspect the persisted run status.") from error

    return {
        "run_id": str(run.id),
        "split": run.dataset_split,
        "agent_version": run.agent_version,
        "status": run.status,
        "scheduled_answers": len(answers),
        "successful_answers": len(answers) - len(errors),
        "failed_answers": len(errors),
        "answer_errors": errors,
        "metrics": run.metrics,
        "elapsed_seconds": round(time.perf_counter() - started_run, 3),
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@app.get("/evaluation-runs/{run_id}", tags=["Evaluation"])
def get_evaluation_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        parsed_id = uuid.UUID(run_id)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail="Invalid evaluation run UUID.") from error

    try:
        run = db.scalar(
            select(EvaluationRun)
            .options(selectinload(EvaluationRun.answers).selectinload(AnswerResult.claims))
            .where(EvaluationRun.id == parsed_id)
        )
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail="Database unavailable.") from error
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")

    return {
        "run_id": str(run.id),
        "split": run.dataset_split,
        "agent_version": run.agent_version,
        "provider": run.provider,
        "model": run.model,
        "dataset_version": run.dataset_version,
        "fact_version": run.fact_version,
        "evaluator_version": run.evaluator_version,
        "prompt_version": run.prompt_version,
        "settings": run.settings,
        "input_hashes": run.input_hashes,
        "source_revision": run.source_revision,
        "status": run.status,
        "metrics": run.metrics,
        "error_summary": run.error_summary,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "answers": [
            {
                "answer_id": answer.answer_id,
                "status": answer.status,
                "latency_ms": answer.latency_ms,
                "error_summary": answer.error_summary,
                "prediction": answer.prediction,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "start_char": claim.start_char,
                        "end_char": claim.end_char,
                        "quote": claim.quote,
                        "label": claim.label,
                        "fact_ids": claim.fact_ids,
                        "reason": claim.reason,
                    }
                    for claim in answer.claims
                ],
            }
            for answer in run.answers
        ],
    }
