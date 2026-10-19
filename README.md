# Kasparro — FSE Intern Pathway Assessment

A Python agent that extracts complete factual claim sentences from long AI-generated speaker-product answers and classifies each claim as `supported`, `contradicted`, or `unknown` from the supplied product catalogue.

## Assessment scope

This repository implements the required assessment workflow without a hosted deployment:

- A baseline agent (`v1`) and a candidate agent (`v2`) with one targeted prompt refinement.
- Real model-requested `get_brand_facts(brand_id)` tool calls; only this allowlisted read-only function can execute.
- Exact source sentence quotes, zero-based Python Unicode code-point offsets, validated labels and catalogue fact IDs.
- A deterministic exact-span extraction and joint scoring harness.
- FastAPI classification and evaluation-run endpoints, PostgreSQL persistence and controlled tests.
- Checkpointed CLI runs with answer outcomes, error diagnostics, hashes, and sanitized model/tool exchanges.

**Important:** The v2 prompt is a hypothesis until side-by-side live evaluation reports measure its effect. Scores are never hardcoded. The final-check split is not scored because its expected labels are reviewer-only.

## Versioned datasets

The assessment defines these versions:

| Input / evaluator | Version |
|---|---|
| Answers dataset | `long-brand-answers-50-v4` |
| Product catalogue | `speaker-catalogue-v4` |
| Evaluator | `sentence-spans-and-labels-v4` |

Expected input counts: 50 answers (A01–A40 development, A41–A50 checks), 132 catalogue facts, and 376 gold claims for development. See [reports/dataset-manifest.md](reports/dataset-manifest.md) for hashes of the separately supplied CSV files. The CSVs are read with the standard CSV parser and the original `ai_answer` text is not stripped or normalized before offsets are calculated.

`development_gold.csv` is used by the evaluation harness only. It must never be passed to an agent or tool. A final-check report keeps `metrics: null`.

## Setup

Requirements: Python 3.11+, PostgreSQL, Git, and an API key for a configured provider. OpenAI is the default recommended provider; Groq is available as an alternative. The app is not deployed and does not need to be.

### 1. Install

Windows PowerShell:

```powershell
git clone https://github.com/devacharya80/Kasparro.git
cd Kasparro
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
git clone https://github.com/devacharya80/Kasparro.git
cd Kasparro
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure the environment

Copy `.env.example` to `.env` and set the database URL plus the provider key. To use OpenAI:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/kasparro_assessment
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_key_here
```

For Groq instead, set `LLM_PROVIDER=groq` and provide `GROQ_API_KEY` and optionally `GROQ_MODEL`. Never commit `.env` or put an API key into logs, tests, reports, or the README.

Create the PostgreSQL database named in `DATABASE_URL` before starting the API. The application creates the assessment tables at startup. The current schema is intended for a fresh assessment database; `Base.metadata.create_all` does not migrate an existing database schema.

### 3. Run deterministic checks

```bash
python -m pytest -q
python -m app.evaluation.dataset_inspector
```

The tests use fake responses and require no live LLM key. The dataset inspector checks source counts, splits, exact gold quotes and offsets, label values, fact references and duplicate/overlapping spans.

## Agent contract

The model receives `answer_id`, primary `brand_id`, the original `question` as context, and the complete `ai_answer`. Claims are extracted only from `ai_answer`. For cross-brand comparisons, the model can request facts for each named known brand.

The tool schema and executor allow only `get_brand_facts(brand_id)`; model-created tool names or arbitrary code are never executed. Each answer is limited to three model request turns, with a configured timeout for each provider request. A malformed response, an invalid tool request, an unknown brand, a provider failure, or turn exhaustion records an answer-level error and accepts no claims for that answer.

The labels follow the assessment rules:
- `supported`: the catalogue establishes every material factual component.
- `contradicted`: catalogue evidence explicitly refutes at least one material component.
- `unknown`: the catalogue cannot establish truth or falsity, with no explicit contradiction.

Dates, markets, conditions, units, model identity, warranty scope, comparison direction, tax and shipping are material qualifiers. Missing evidence alone is not a contradiction. No external product knowledge is used.

## API

Start the local API after configuring PostgreSQL and your LLM key:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

### `POST /classify`

```json
{
  "answer_id": "A01",
  "brand_id": "cleartrail",
  "question": "How heavy is the speaker?",
  "answer_text": "The speaker weighs 800 grams.",
  "agent_version": "v1"
}
```

The response contains exact claims with `claim_id`, `start_char`, `end_char`, `quote`, `label`, `fact_ids`, and `reason`. Input fields outside the request contract—including submitted gold claims—are rejected. Invalid requests return 422, an unknown primary brand returns 404, provider/invalid-output errors return 502, and provider timeouts return 504.

### `POST /evaluation-runs`

```json
{
  "split": "development",
  "agent_version": "v1"
}
```

The route processes the fixed CSV split, creates one terminal answer outcome per scheduled answer, stores accepted claim rows, and returns the run ID, counts, errors, status and available metrics. Check runs retain `metrics: null`.

### `GET /evaluation-runs/{id}`

Returns run metadata, input hashes, versions, settings, persisted answer outcomes and claims ordered by answer ID and source span. Malformed UUIDs return 422; unknown UUIDs return 404.

## Evaluation CLI and required four runs

Run the full assessment sequentially with the configured live provider. Each command takes around a provider call per answer and may incur API charges:

```bash
python -m app.evaluation.run_evaluation --agent v1 --split development
python -m app.evaluation.run_evaluation --agent v2 --split development
python -m app.evaluation.run_evaluation --agent v1 --split check
python -m app.evaluation.run_evaluation --agent v2 --split check
```

Use `--limit 1` only for a smoke test; these are not full assessment artifacts.

Each run checkpoints these files under `artifacts/` after every answer:

- `{agent}_{split}_predictions.json`: accepted claim predictions only.
- `{agent}_{split}_outcomes.json`: exactly one succeeded/failed outcome per scheduled answer.
- `{agent}_{split}_errors.json`: sanitized answer-level failures.
- `{agent}_{split}_exchanges.json`: sanitized tool requests/results and final model responses.
- `{agent}_{split}_run.json`: versions, active settings, input hashes, source revision, timings, outcomes and scores when allowed.

The run JSON also embeds the sanitized exchanges, so the main report is self-contained. A CLI exit code of 2 indicates one or more answer-level failures; their accepted claim list is empty, and their development gold claims remain missed in scoring.

### Scoring

For the development split, with (G) gold claims, (P) accepted predictions, (M) exact answer/start/end span matches and (C) exact-span matches with the correct label:

- Extraction precision = (M/P), recall = (M/G), F1 = (2M/(P+G)).
- Joint precision = (C/P), recall = (C/G), F1 = (2C/(P+G)).
- Joint F1 is the primary metric. Reports also include coverage, missed gold claim IDs, unmatched prediction IDs, matched wrong-label IDs and answer-error counts.

Zero denominators score zero when the split contains gold claims. The check split never receives candidate quality metrics, gold IDs, or expected labels.

## GitHub Actions

The push workflow runs the deterministic pytest suite. To produce the four required live runs, open **Actions → Live assessment evaluation → Run workflow** on `main`, choose `openai` or `groq`, and use the matching `OPENAI_API_KEY` or `GROQ_API_KEY` repository secret. The manual workflow uploads the actual outputs even when one or more answers fail, and reports failed runs honestly. It is not triggered on each push, to avoid unintentional provider charges. No deployment workflow is required.

## Limitations and evidence status

- OpenAI and Groq are implemented; other providers mentioned in the assessment are not implemented in this repository.
- v2 has one targeted prompt change, but no performance gain is claimed unless the four full run reports show it.
- PostgreSQL-backed integration behavior should be verified against a fresh local PostgreSQL database; deterministic schema/API validation tests do not replace that check.
- The final-check set is visible to a candidate, not a blind holdout. It is saved only for reviewer scoring.

Do not describe the four live runs as complete until the reports were generated with a working API key. Keep only actual outputs and real test logs as submission evidence.

## Author

**Devacharya** · [GitHub](https://github.com/devacharya80)
