import json
from types import SimpleNamespace

import pytest

from app.agent.catalogue import UnknownToolError
from app.agent.tool_runner import (
    InvalidToolArgumentsError,
    TurnLimitExceeded,
    run_tool_calling,
)
from app.agent.tool_schema import build_brand_facts_tool_schema


def make_tool_response(
    tool_name: str,
    arguments: str,
    call_id: str = "call_001",
):
    tool_call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=tool_name,
            arguments=arguments,
        ),
    )

    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message),
        ]
    )


def make_final_response(content: str):
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message),
        ]
    )


class FakeClient:
    """Return predefined responses without making network requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create,
            )
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if not self.responses:
            raise AssertionError("Unexpected extra model request.")

        return self.responses.pop(0)


def test_tool_schema_contains_known_brands() -> None:
    schema = build_brand_facts_tool_schema()

    assert len(schema) == 1
    assert schema[0]["type"] == "function"

    function = schema[0]["function"]
    assert function["name"] == "get_brand_facts"

    parameters = function["parameters"]
    assert parameters["required"] == ["brand_id"]
    assert parameters["additionalProperties"] is False

    assert set(
        parameters["properties"]["brand_id"]["enum"]
    ) == {
        "cleartrail",
        "cloudnote",
        "stonewave",
    }


def test_tool_runner_executes_tool_and_returns_final_response() -> None:
    client = FakeClient(
        [
            make_tool_response(
                "get_brand_facts",
                json.dumps({"brand_id": "cleartrail"}),
            ),
            make_final_response(
                "cleartrail has 44 catalogue facts."
            ),
        ]
    )

    result = run_tool_calling(
        [
            {
                "role": "user",
                "content": "Look up cleartrail.",
            }
        ],
        client=client,
        tool_choice="required",
    )

    assert result.final_text == "cleartrail has 44 catalogue facts."
    assert result.model_turns == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].brand_id == "cleartrail"
    assert result.tool_calls[0].facts_returned == 44

    assert len(client.calls) == 2

    second_request_messages = client.calls[1]["messages"]

    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_001"
        for message in second_request_messages
    )


def test_tool_runner_rejects_unknown_tool() -> None:
    client = FakeClient(
        [
            make_tool_response(
                "run_arbitrary_code",
                json.dumps({"brand_id": "cleartrail"}),
            )
        ]
    )

    with pytest.raises(UnknownToolError):
        run_tool_calling(
            [{"role": "user", "content": "Test an invalid tool."}],
            client=client,
            max_turns=3,
        )


def test_tool_runner_rejects_malformed_json_arguments() -> None:
    client = FakeClient(
        [
            make_tool_response(
                "get_brand_facts",
                "{not-valid-json",
            )
        ]
    )

    with pytest.raises(InvalidToolArgumentsError):
        run_tool_calling(
            [{"role": "user", "content": "Test invalid arguments."}],
            client=client,
        )


@pytest.mark.parametrize("max_turns", [0, 4])
def test_tool_runner_rejects_invalid_turn_limits(
    max_turns: int,
) -> None:
    client = FakeClient([])

    with pytest.raises(ValueError, match="max_turns"):
        run_tool_calling(
            [{"role": "user", "content": "Test turn validation."}],
            client=client,
            max_turns=max_turns,
        )

    assert client.calls == []


def test_tool_runner_rejects_tool_call_on_final_turn() -> None:
    client = FakeClient(
        [
            make_tool_response(
                "get_brand_facts",
                json.dumps({"brand_id": "cleartrail"}),
            )
        ]
    )

    with pytest.raises(TurnLimitExceeded):
        run_tool_calling(
            [{"role": "user", "content": "Test turn exhaustion."}],
            client=client,
            max_turns=1,
        )

    assert len(client.calls) == 1