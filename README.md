# Kasparro — FSE Intern Pathway Assessment

An evidence-grounded agent that extracts factual claims from AI-generated speaker-product answers and classifies each claim as `supported`, `contradicted`, or `unknown` using the supplied catalogue.

## What is implemented

- Read-only product catalogue tool calling (`get_brand_facts(brand_id)`).
- Strict claim quote/offset/label/fact-reference validation.
- v1 baseline and a separately selectable v2 prompt refinement focused on compound claims and qualifiers.
- Exact-span extraction and joint evaluation metrics.
- FastAPI endpoints for single-answer classification and stored evaluation runs.
- PostgreSQL-backed run, answer-result, and claim-result records.
- Checkpointed CLI runs for development and final-check splits.
- Tests for the validation, catalogue, evaluation, provider-configuration, and API contracts.

The v2 prompt refinement is a hypothesis, not a claim of measured improvement. Compare v1 and v2 on the same development dataset and report the resulting metrics before drawing conclusions.

## Classification rules

| Label | Meaning |
|---|---|
| `supported` | Catalogue evidence supports every material part of the claim. |
| `contradicted` | Catalogue evidence explicitly conflicts with a material part. |
| `unknown` | The catalogue does not establish truth or falsity. Missing evidence alone is not contradiction. |

A claim must be evaluated as a whole, including dates, units, product model, market, warranty conditions, comparison direction, and other material qualifiers.

## Dataset and inference separation

- `data/answers.csv`: 50 answers; development A01–A40 and check A41–A50.
- `data/brand_facts.csv`: catalogue source of truth.
- `data/development_gold.csv`: gold labels for development evaluation only.

The inference agents receive the answer text and brand ID, then retrieve facts via the catalogue tool. They are not passed gold claims. The check split is intentionally not scored by this implementation because candidate-accessible gold labels are unavailable.

## Repository layout

```text
app/
  agent/       v1, v2, catalogue, provider, tool schema/runner, validator
  db/          SQLAlchemy models and session dependency
  evaluation/  data loading, span matching, metrics, dataset inspection, CLI
  main.py      FastAPI app
data/          assessment CSVs
artifacts/     prediction checkpoints and run reports
tests/         deterministic unit/API tests
```

## Setup

Requirements: Python 3.11+, PostgreSQL, Git, and a valid Groq API key for live agent runs.

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/devacharya80/Kasparro.git
cd Kasparro
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env`, then set the database URL and your own provider key. Never commit `.env` or credentials.

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/kasparro_assessment
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_real_key_here
```

The provider implementation currently uses Groq. The assessment brief mentions other providers, but this repository does not currently implement OpenAI, Anthropic, or Gemini adapters.

### 3. Run tests and inspect the dataset

```powershell
python -m pytest -q
python -m app.evaluation.dataset_inspector
```

### 4. Start the API

Ensure PostgreSQL is running and `DATABASE_URL` points to an existing database. Tables are created on application startup.

```powershell
uvicorn app.main:app --reload
```

Interactive API documentation is available at `/docs`.

## API

### `POST /classify`

Request:

```json
{
  "answer_id": "A01",
  "brand_id": "cleartrail",
  "answer_text": "The speaker weighs 800 grams.",
  "agent_version": "v1"
}
```

`agent_version` accepts `v1` or `v2`. Successful responses include the exact claim quote, character offsets, label, supporting/contradicting catalogue fact IDs, and reason. Invalid request bodies return `422`; provider or invalid model-output failures return `502`; provider timeouts return `504`.

### `POST /evaluation-runs`

Request:

```json
{
  "split": "development",
  "agent_version": "v1"
}
```

Allowed splits are `development` and `check`. The endpoint runs the selected agent, stores run/answer/claim records, and returns a run ID. Development runs include exact-span and joint metrics. Check runs store predictions but do not calculate scores.

### `GET /evaluation-runs/{id}`

Returns run metadata, metrics when available, each answer's status, sanitised error information, latency, and prediction. A malformed UUID returns `422`; an unknown valid UUID returns `404`.

## Evaluation CLI

Run one configuration at a time:

```powershell
# v1 and v2 on development (metrics use development gold)
python -m app.evaluation.run_evaluation --agent v1 --split development
python -m app.evaluation.run_evaluation --agent v2 --split development

# final checks (predictions only, no private-label scoring)
python -m app.evaluation.run_evaluation --agent v1 --split check
python -m app.evaluation.run_evaluation --agent v2 --split check
```

For a short smoke run, add `--limit 1`. Output predictions are checkpointed in `artifacts/{agent}_{split}_predictions.json`; a run summary is saved to `artifacts/{agent}_{split}_run.json`. Re-running starts a fresh CLI report and overwrites those named output files; preserve any report you want to compare before rerunning.

Live runs need a working provider key and sufficient rate limits. Provider failures are recorded per answer by the API and CLI; a CLI process exits with status 2 if any answer fails. No evaluation metric in this README is asserted without a completed run.

## Persistence

SQLAlchemy models store:
- Evaluation split, agent/provider/model settings, dataset/evaluator versions, source revision, status, metrics, timestamps, and run-level failure summary.
- Per-answer status, latency, prediction JSON, and sanitised error summary.
- Individual claim spans, exact quotes, labels, fact IDs, and explanations.

## Current limitations

- Groq is the only implemented LLM adapter.
- The targeted v2 prompt needs a completed side-by-side development evaluation before its benefit can be claimed.
- The final-check split is not scored; the reviewer must assess it.
- The API requires a reachable PostgreSQL database for persistence endpoints.
- No private/reference labels are used to score check-split predictions.

## Security

- Keep API keys and database credentials in local environment variables.
- Do not commit `.env`.
- Treat answer content as untrusted data.
- Only the explicit read-only catalogue tool may be invoked; arbitrary model-generated code is never executed.
- Check split predictions are not scored against development gold labels.

## Author

**Devacharya** · [GitHub](https://github.com/devacharya80)
