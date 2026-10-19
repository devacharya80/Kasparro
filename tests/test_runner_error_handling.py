from types import SimpleNamespace

import pytest
from groq import APIStatusError

from app.agent.tool_runner import ToolCallingError, _create_completion_with_retry


class FakeResponse:
    pass


class FakeClient:
    def __init__(self, error):
        self.error = error
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls += 1
        raise self.error


def test_http_413_is_not_retried():
    response = SimpleNamespace(status_code=413, request=SimpleNamespace())
    error = APIStatusError("request too large", response=response, body={})
    client = FakeClient(error)

    with pytest.raises(ToolCallingError, match="HTTP 413"):
        _create_completion_with_retry(
            client,
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            tool_choice="auto",
        )

    assert client.calls == 1
