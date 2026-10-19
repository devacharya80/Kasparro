from fastapi import FastAPI

app = FastAPI(
    title="Kasparro Assessment API",
    description=(
        "An AI agent for extracting and classifying factual claims "
        "from AI-generated product answers."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the current API health status."""
    return {
        "status": "ok",
        "service": "kasparro-assessment-api",
    }