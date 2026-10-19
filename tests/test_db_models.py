from app.db.models import AnswerResult, Base, ClaimResult, EvaluationRun


def test_expected_persistence_tables_exist():
    assert set(Base.metadata.tables) == {
        "evaluation_runs",
        "answer_results",
        "claim_results",
    }


def test_persistence_models_have_unique_answer_and_claim_constraints():
    answer_constraints = {
        constraint.name for constraint in AnswerResult.__table__.constraints
    }
    claim_constraints = {
        constraint.name for constraint in ClaimResult.__table__.constraints
    }
    assert "uq_answer_run_answer_id" in answer_constraints
    assert "uq_claim_answer_span" in claim_constraints
    assert "uq_claim_answer_claim_id" in claim_constraints


def test_run_metadata_includes_versions_and_input_hashes():
    columns = EvaluationRun.__table__.columns
    assert "fact_version" in columns
    assert "prompt_version" in columns
    assert "input_hashes" in columns
    assert "settings" in columns
    assert "metrics" in columns


def test_persistence_columns_cover_run_answer_and_claim_data():
    assert "prediction" in AnswerResult.__table__.columns
    assert "latency_ms" in AnswerResult.__table__.columns
    assert "fact_ids" in ClaimResult.__table__.columns
    assert "start_char" in ClaimResult.__table__.columns
    assert "end_char" in ClaimResult.__table__.columns
