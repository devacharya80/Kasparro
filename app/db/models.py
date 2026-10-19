from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Uuid, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_split: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_version: Mapped[str] = mapped_column(String(8), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    fact_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    settings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    input_hashes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_revision: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="running")
    metrics: Mapped[dict | None] = mapped_column(JSON)
    error_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    answers: Mapped[list["AnswerResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AnswerResult.answer_id"
    )


class AnswerResult(Base):
    __tablename__ = "answer_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", name="uq_answer_run_answer_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    answer_id: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_summary: Mapped[str | None] = mapped_column(Text)
    trace_reference: Mapped[str | None] = mapped_column(String(256))
    prediction: Mapped[dict | None] = mapped_column(JSON)
    run: Mapped["EvaluationRun"] = relationship(back_populates="answers")
    claims: Mapped[list["ClaimResult"]] = relationship(
        back_populates="answer_result",
        cascade="all, delete-orphan",
        order_by="ClaimResult.start_char",
    )


class ClaimResult(Base):
    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint(
            "answer_result_id", "start_char", "end_char",
            name="uq_claim_answer_span",
        ),
        UniqueConstraint(
            "answer_result_id", "claim_id",
            name="uq_claim_answer_claim_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    answer_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("answer_results.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    claim_id: Mapped[str] = mapped_column(String(80), nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    fact_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_match: Mapped[dict | None] = mapped_column(JSON)
    answer_result: Mapped["AnswerResult"] = relationship(back_populates="claims")
