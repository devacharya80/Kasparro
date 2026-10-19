import pytest

from app.agent.provider import (
    DEFAULT_GROQ_MODEL,
    create_groq_client,
    get_groq_api_key,
    get_groq_model,
)


def test_model_uses_default_when_not_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert get_groq_model() == DEFAULT_GROQ_MODEL


def test_model_uses_environment_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    assert get_groq_model() == "test-model"


def test_missing_api_key_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_groq_api_key()


def test_placeholder_api_key_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "replace-me")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_groq_api_key()


def test_client_can_be_created_without_network_request(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    client = create_groq_client()

    assert client is not None
    client.close()


def test_openai_provider_uses_configured_model(monkeypatch):
    from app.agent.provider import get_model_name, get_provider_name

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-model")
    assert get_provider_name() == "openai"
    assert get_model_name() == "test-openai-model"


def test_openai_provider_rejects_missing_key(monkeypatch):
    from app.agent.provider import get_openai_api_key

    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_openai_api_key()


def test_openai_client_can_be_created_without_network_request(monkeypatch):
    from app.agent.provider import create_llm_client

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = create_llm_client()
    assert client is not None
    client.close()
