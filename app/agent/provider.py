from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
SUPPORTED_PROVIDERS = frozenset({"openai", "groq"})


def load_environment() -> None:
    """Load local environment settings without overriding configured variables."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def get_provider_name() -> str:
    load_environment()
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER {provider!r}; choose openai or groq."
        )
    return provider


def get_model_name() -> str:
    load_environment()
    provider = get_provider_name()
    env_name, default = (
        ("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        if provider == "openai"
        else ("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    )
    model = os.getenv(env_name, default).strip()
    if not model:
        raise RuntimeError(f"{env_name} must be a non-empty string.")
    return model


def get_openai_api_key() -> str:
    load_environment()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "replace-me":
        raise RuntimeError("OPENAI_API_KEY is missing or is still a placeholder. Set it in .env.")
    return api_key


def get_groq_api_key() -> str:
    """Legacy explicit Groq-key accessor, retained for existing local setups."""
    load_environment()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "replace-me":
        raise RuntimeError("GROQ_API_KEY is missing or is still a placeholder. Set it in .env.")
    return api_key


def get_groq_model() -> str:
    """Legacy Groq model accessor, retained for existing tests and scripts."""
    load_environment()
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    if not model:
        raise RuntimeError("GROQ_MODEL must be a non-empty string.")
    return model


def create_llm_client() -> Any:
    """Create the selected model client with finite request timeout."""
    provider = get_provider_name()
    if provider == "openai":
        return OpenAI(
            api_key=get_openai_api_key(),
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=1,
        )
    return Groq(
        api_key=get_groq_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def create_groq_client() -> Groq:
    """Create an explicit Groq client for the legacy smoke-test entry point."""
    return Groq(
        api_key=get_groq_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )
