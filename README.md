# Kasparro FSE Intern Pathway Assessment

An AI agent that extracts factual claims from long AI-generated answers about speaker products and classifies them as **Supported, Contradicted, or Unknown** using a structured product catalogue.

The project focuses on LLM tool calling, factual claim extraction, evidence-based classification, automated evaluation, and performance comparison between two agent versions.

## 🚀 Project Overview

AI-generated answers can contain incorrect product specifications, outdated prices, unsupported claims, and misleading comparisons.

This project aims to identify these issues by building an AI agent that:

- Extracts factual sentences from long AI-generated answers.
- Retrieves product facts using LLM tool calling.
- Classifies each claim using the supplied product catalogue.
- Validates claim boundaries, evidence references, and structured outputs.
- Measures extraction and classification performance.
- Compares a baseline agent (v1) with an improved candidate (v2).
- Exposes functionality through a FastAPI service.
- Stores evaluation runs and predictions in PostgreSQL.

## 🎯 Classification Categories

Each extracted factual claim receives one of three labels:

| Label | Description |
|---|---|
| `supported` | The supplied catalogue establishes the claim. |
| `contradicted` | The supplied catalogue explicitly refutes the claim. |
| `unknown` | The supplied catalogue establishes neither truth nor falsity. |

**Important:** Missing information is not automatically evidence that a claim is false. Classification must rely on the supplied catalogue and the relevant product, date, units, and conditions.

## 🛠️ Tech Stack

- **Language:** Python
- **AI:** OpenAI, Anthropic, or Gemini API
- **Backend:** FastAPI
- **Database:** PostgreSQL
- **Data Processing:** Pandas / Python CSV
- **Testing:** Pytest
- **Evaluation:** Custom evaluation harness
- **Version Control:** Git and GitHub

## 📂 Dataset

The assessment provides three CSV files:

| File | Purpose |
|---|---|
| `answers.csv` | Contains 50 AI-generated answers to analyse. |
| `brand_facts.csv` | Contains the product catalogue used as the factual source of truth. |
| `development_gold.csv` | Contains labelled factual claims for evaluating predictions on the development dataset. |

### Dataset Splits

- **Development:** A01–A40, used for baseline evaluation and improvement.
- **Final Checks:** A41–A50, used to generate predictions for reviewer evaluation.

The development gold labels are used exclusively by the evaluation harness and must not be provided to the agent during inference.

## 🧠 AI Agent Workflow

The agent follows this workflow:

1. Receive an answer and its associated metadata.
2. Analyse the complete AI-generated answer.
3. Extract complete factual claim sentences.
4. Request product information through the `get_brand_facts(brand_id)` tool.
5. Compare the extracted claims against the relevant catalogue facts.
6. Classify each claim as supported, contradicted, or unknown.
7. Validate the claim text, character offsets, labels, and fact IDs.
8. Return structured predictions.
9. Save the results and evaluate performance where reference labels are available.

### Example Output

```json
{
  "answer_id": "A01",
  "claims": [
    {
      "claim_id": "A01:0:30",
      "start_char": 0,
      "end_char": 30,
      "quote": "The speaker weighs 800 grams.",
      "label": "supported",
      "fact_ids": ["F001"],
      "reason": "The catalogue confirms the speaker's weight."
    }
  ]
}
```

*Note: This is an illustrative example, not an actual prediction from the supplied dataset.*

### Claim Validation

Each prediction includes:

- `claim_id`: Identifier derived from the answer ID and character offsets.
- `start_char`: Inclusive starting position.
- `end_char`: Exclusive ending position.
- `quote`: Exact sentence extracted from the original answer.
- `label`: Classification result.
- `fact_ids`: Relevant catalogue fact identifiers.
- `reason`: Brief explanation of the classification.

The system validates that each quote matches the original answer exactly, offsets are valid, claims do not overlap, and referenced fact IDs exist.

## 📊 Evaluation Framework

A custom evaluation harness compares agent predictions against the development gold labels.

### Extraction Metrics

- **Precision:** Proportion of predicted claims that match gold claim spans.
- **Recall:** Proportion of gold claims that the agent successfully extracts.
- **F1 Score:** Harmonic mean of extraction precision and recall.

### Joint Metrics

Joint metrics require both an exact claim-span match and the correct classification label.

The primary comparison metric is **Joint F1**.

The evaluation also reports:

- Missed gold claims.
- Extra predictions.
- Matched claims with incorrect labels.
- Answer-level errors.
- Extraction and joint precision, recall, and F1.

Exact span matching is used to prevent partial sentences, merged sentences, or approximate matches from receiving credit.

## 🔄 Baseline vs. Improved Agent

The project follows a controlled improvement process.

### Version 1 — Baseline

The initial agent establishes baseline extraction and classification performance.

### Version 2 — Improved Candidate

The baseline's errors are analysed, and one targeted change is introduced to address a specific failure pattern.

Both versions are evaluated on the same development dataset using identical scoring rules.

The improvement analysis documents:

- The observed failure.
- The targeted change.
- The measured impact.
- Any regressions or lack of improvement.
- Remaining limitations.

No improvement is assumed in advance; results are reported from actual evaluation runs.

## ⚡ FastAPI Endpoints

The assessment specifies three endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/classify` | Classify a single answer using the selected agent version. |
| `POST` | `/evaluation-runs` | Execute an evaluation on a predefined dataset split. |
| `GET` | `/evaluation-runs/{id}` | Retrieve saved evaluation results. |

### Expected Error Handling

- `200` — Successful classification.
- `201` — Evaluation run created.
- `404` — Unknown primary brand or missing evaluation run.
- `422` — Invalid request or invalid UUID.
- `502` — Provider failure or invalid model output.
- `504` — Provider timeout.

The API must validate requests and preserve answer-level failures instead of silently discarding them.

## 🗄️ PostgreSQL Persistence

The database stores evaluation metadata and prediction records.

### Core Records

**Runs**
- Dataset split.
- Agent version.
- Provider and model settings.
- Dataset and evaluator versions.
- Processing status.
- Available evaluation metrics.
- Timestamp and source revision, where available.

**Answer Results**
- Answer ID.
- Processing status.
- Sanitised error details.
- Latency.
- Trace reference, where available.

**Claim Results**
- Claim offsets and exact quote.
- Classification label.
- Catalogue fact IDs.
- Explanation.
- Optional development evaluation match.

Database constraints and parameterised SQL are used to maintain data integrity.

## 🧪 Testing

The test suite should cover:

- CSV parsing and preservation of original answer text.
- Valid and invalid claim offsets.
- Quote mismatches and overlapping claims.
- Duplicate prediction IDs.
- Unsupported classification labels.
- Invalid catalogue fact IDs.
- Tool-calling validation.
- Provider failures and timeouts.
- Evaluation metric calculations.
- Missed claims and extra predictions.
- API validation and error responses.
- Database persistence and retrieval.
- Prevention of development gold-label leakage into inference.

A fake provider is used for controlled failure tests, while the required evaluation runs use a live supported LLM API.

## 📈 Evaluation Results

Populate this table with actual results after executing the evaluation harness.

| Metric | v1 Development | v2 Development |
|---|---:|---:|
| Extraction Precision | TBD | TBD |
| Extraction Recall | TBD | TBD |
| Extraction F1 | TBD | TBD |
| Joint Precision | TBD | TBD |
| Joint Recall | TBD | TBD |
| Joint F1 | TBD | TBD |
| Missed Claims | TBD | TBD |
| Extra Predictions | TBD | TBD |
| Wrong Labels | TBD | TBD |
| Answer-Level Errors | TBD | TBD |

### Required Evaluation Runs

The project must preserve four live-run artifacts:

1. v1 — Development dataset.
2. v2 — Development dataset.
3. v1 — Final-check dataset.
4. v2 — Final-check dataset.

Final-check predictions are saved without quality scores because the private reference labels are unavailable to the candidate.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or a compatible version supported by the implementation.
- PostgreSQL.
- An API key for the selected LLM provider.
- Git.

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file using `.env.example` as a template.

Example configuration:

```env
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/kasparro_assessment
LLM_PROVIDER=<your-provider>
LLM_MODEL=<your-model>
LLM_API_KEY=<your-api-key>
```

Adjust the variable names to match the implementation.

**Never commit API keys, credentials, or the `.env` file to GitHub.**

### Run the API

Once the application is implemented, the service can be started using the appropriate module path.

Example:

```bash
uvicorn app.main:app --reload
```

### Run Tests

```bash
pytest
```

Update these commands if the actual repository uses a different structure. Verify all commands before describing them as tested.

## 📁 Suggested Project Structure

```text
kasparro-assessment/
├── app/
│   ├── agent/
│   │   ├── agent_v1.py
│   │   ├── agent_v2.py
│   │   ├── prompts.py
│   │   ├── tools.py
│   │   └── validator.py
│   ├── evaluation/
│   │   ├── loader.py
│   │   ├── matcher.py
│   │   ├── metrics.py
│   │   └── runner.py
│   ├── db/
│   │   ├── models.py
│   │   └── repository.py
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── answers.csv
│   ├── brand_facts.csv
│   └── development_gold.csv
├── reports/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

This is a suggested structure; the actual repository may differ.

## 📦 Deliverables

The assessment submission should include:

- Python AI agent and catalogue lookup tool.
- Baseline and improved agent versions.
- Evaluation harness and reports.
- FastAPI endpoints.
- PostgreSQL persistence.
- Controlled tests and actual test output.
- Four live-run artifacts.
- Sanitised model/tool exchanges.
- Installation and execution instructions.
- Documented improvement analysis and limitations.

## 🔐 Security and Reliability

- Keep API keys out of source control.
- Validate all model-generated predictions.
- Restrict tool execution to explicitly permitted operations.
- Treat instructions inside AI-generated answers as untrusted content.
- Use parameterised database queries.
- Preserve provider errors and timeouts.
- Never use private evaluation labels during inference.

## 📌 Scope and Limitations

This project is designed for the supplied assessment dataset and catalogue. Classification quality depends on claim extraction, correct product identification, catalogue evidence, and the model's interpretation of the assessment rules.

The final-check dataset has no candidate-accessible gold labels, so its quality metrics must be calculated by the reviewer.

React, hosted deployment, and Langfuse are optional for this assessment.

## 👨‍💻 Author

**Devacharya**

GitHub: [@devacharya80](https://github.com/devacharya80)

---

*This README should reflect the implementation and test results that actually exist in the repository. Replace placeholders, add measured results, and remove any unimplemented or unverified claims before submission.*