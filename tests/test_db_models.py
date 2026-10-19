from app.db.models import AnswerResult, Base, ClaimResult, EvaluationRun


def test_expected_persistence_tables_exist():
    assert set(Base.metadata.tables) == {
        "evaluation_runs",
        "answer_results",
        "claim_results",
    }


def test_persistence_columns_cover_run_answer_and_claim_data():
    assert "metrics" in EvaluationRun.__table__.columns
    assert "prediction" in AnswerResult.__table__.columns
    assert "latency_ms" in AnswerResult.__table__.columns
    assert "fact_ids" in ClaimResult.__table__.columns
    assert "start_char" in ClaimResult.__table__.columns
    assert "end_char" in ClaimResult.__table__.columns
