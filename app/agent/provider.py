from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SECONDS = 30.0


def load_environment() -> None:
    """Load local environment settings without overriding existing variables."""
    load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )


def get_groq_api_key() -> str:
    """Return the configured Groq key without exposing it in logs."""

    load_environment()

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "replace-me":
        raise RuntimeError(
            "GROQ_API_KEY is missing or is still a placeholder. "
            "Set it in your local .env file."
        )

    return api_key


def get_groq_model() -> str:
    """Return the configured Groq model identifier."""

    load_environment()

    model = os.getenv(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    ).strip()

    if not model:
        raise RuntimeError(
            "GROQ_MODEL must be a non-empty string."
        )

    return model


def create_groq_client() -> Groq:
    """Create a Groq client with a finite request timeout."""

    return Groq(
        api_key=get_groq_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )