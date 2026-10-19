from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agent.agent_v1 import classify_answer as classify_v1
from app.agent.agent_v2 import classify_answer as classify_v2
from app.catalogue_version import DATASET_VERSION, EVALUATOR_VERSION
from app.db.models import AnswerResult, Base, ClaimResult, EvaluationRun
from app.db.session import engine, get_db
from app.evaluation.loader import load_answers, load_gold_claims
from app.evaluation.metrics import evaluate_claims

app = FastAPI(
    title="Kasparro Assessment API",
    description="Evidence-grounded factual claim extraction and evaluation.",
    version="1.0.0",
)
AgentVersion = Literal["v1", "v2"]


class ClassifyRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=16)
    brand_id: str = Field(min_length=1, max_length=64)
    answer_text: str
    agent_version: AgentVersion = "v1"

    @field_validator("answer_id", "brand_id")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class EvaluationRunRequest(BaseModel):
    split: Literal["development", "check"]
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


def _provider_details() -> tuple[str, str]:
    from app.agent.provider import get_groq_model

    return "groq", get_groq_model()


def _classify(version: AgentVersion, *, answer_id: str, brand_id: str, answer_text: str):
    classifier = classify_v1 if version == "v1" else classify_v2
    return classifier(answer_id=answer_id, brand_id=brand_id, answer_text=answer_text)


def _http_model_error(error: Exception) -> HTTPException:
    from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    if isinstance(error, APITimeoutError):
        return HTTPException(status_code=504, detail="LLM provider timed out.")
    if isinstance(error, (APIConnectionError, APIStatusError, RateLimitError)):
        return HTTPException(status_code=502, detail="LLM provider request failed.")
    return HTTPException(status_code=502, detail="LLM response failed validation.")


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
    try:
        provider, model = _provider_details()
        run = EvaluationRun(
            dataset_split=request.split,
            agent_version=request.agent_version,
            provider=provider,
            model=model,
            dataset_version=DATASET_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            source_revision=os.getenv("GIT_COMMIT_SHA"),
            status="running",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Unable to initialise evaluation run or database.",
        ) from error

    try:
        answers = [
            row for row in load_answers()
            if str(row.get("split", "")).strip().lower() == request.split
        ]
        if not answers:
            run.status = "failed"
            run.error_summary = "The selected split contains no answers."
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
            raise HTTPException(status_code=422, detail=run.error_summary)

        predictions: list[dict[str, Any]] = []
        successful_answer_ids: set[str] = set()
        for row in answers:
            answer_id = str(row["answer_id"])
            started = time.perf_counter()
            answer_result = AnswerResult(run_id=run.id, answer_id=answer_id, status="running")
            db.add(answer_result)
            db.flush()
            try:
                prediction = _classify(
                    request.agent_version,
                    answer_id=answer_id,
                    brand_id=str(row["brand_id"]),
                    answer_text=str(row["ai_answer"]),
                )
                answer_result.status = "succeeded"
                answer_result.prediction = prediction
                successful_answer_ids.add(answer_id)
                predictions.append(prediction)
                for claim in prediction["claims"]:
                    db.add(
                        ClaimResult(
                            answer_result_id=answer_result.id,
                            claim_id=claim["claim_id"],
                            start_char=claim["start_char"],
                            end_char=claim["end_char"],
                            quote=claim["quote"],
                            label=claim["label"],
                            fact_ids=claim["fact_ids"],
                            reason=claim["reason"],
                        )
                    )
            except Exception as error:
                answer_result.status = "failed"
                answer_result.error_summary = f"{type(error).__name__}: {str(error)[:300]}"
            finally:
                answer_result.latency_ms = round((time.perf_counter() - started) * 1000)
            db.commit()

        run.status = (
            "completed" if len(successful_answer_ids) == len(answers)
            else "completed_with_errors"
        )
        run.completed_at = datetime.now(timezone.utc)
        if request.split == "development":
            gold = load_gold_claims()
            answer_ids = {str(row["answer_id"]) for row in answers}
            selected_gold = [claim for claim in gold if str(claim["answer_id"]) in answer_ids]
            predicted_claims = [
                {**claim, "answer_id": prediction["answer_id"]}
                for prediction in predictions for claim in prediction["claims"]
            ]
            run.metrics = evaluate_claims(selected_gold, predicted_claims)
        db.commit()
        db.refresh(run)
    except HTTPException:
        raise
    except Exception as error:
        db.rollback()
        try:
            run.status = "failed"
            run.error_summary = f"{type(error).__name__}: {str(error)[:300]}"
            run.completed_at = datetime.now(timezone.utc)
            db.add(run)
            db.commit()
        except SQLAlchemyError:
            db.rollback()
        raise HTTPException(status_code=500, detail="Evaluation run failed.") from error

    return {
        "id": str(run.id),
        "split": run.dataset_split,
        "agent_version": run.agent_version,
        "status": run.status,
        "answer_count": len(answers),
        "metrics": run.metrics,
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
        run = db.scalar(select(EvaluationRun).where(EvaluationRun.id == parsed_id))
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail="Database unavailable.") from error
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    answers = db.scalars(
        select(AnswerResult).where(AnswerResult.run_id == run.id)
    ).all()
    return {
        "id": str(run.id),
        "split": run.dataset_split,
        "agent_version": run.agent_version,
        "provider": run.provider,
        "model": run.model,
        "dataset_version": run.dataset_version,
        "evaluator_version": run.evaluator_version,
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
            }
            for answer in answers
        ],
    }
