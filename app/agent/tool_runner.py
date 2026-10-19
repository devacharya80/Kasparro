from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.catalogue import execute_tool
from app.agent.provider import REQUEST_TIMEOUT_SECONDS, create_llm_client, get_model_name
from app.agent.tool_schema import build_brand_facts_tool_schema


MAX_MODEL_TURNS = 3
TOOL_NAME = "get_brand_facts"
MAX_RATE_LIMIT_RETRIES = 2
INITIAL_RETRY_WAIT_SECONDS = 3


class ToolCallingError(RuntimeError):
    """Base exception for tool-calling failures."""


class InvalidToolArgumentsError(ToolCallingError):
    """Raised when a model-generated tool request is invalid."""


class TurnLimitExceeded(ToolCallingError):
    """Raised when the model cannot finish within the turn limit."""


@dataclass
class ExecutedToolCall:
    """Sanitised record of an executed catalogue lookup."""
    tool_name: str
    brand_id: str
    facts_returned: int
    facts: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ToolCallingResult:
    """Final model result and safe audit exchange."""
    final_text: str
    model_turns: int
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    exchanges: list[dict[str, Any]] = field(default_factory=list)


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError) as error:
        raise InvalidToolArgumentsError("The model returned invalid JSON tool arguments.") from error
    if not isinstance(arguments, dict):
        raise InvalidToolArgumentsError("Tool arguments must be a JSON object.")
    return arguments


def _create_completion_with_retry(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
) -> Any:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            error_name = type(error).__name__.lower()
            if status_code == 413:
                raise ToolCallingError(
                    "Provider rejected the request with HTTP 413; the identical request was not retried."
                ) from error
            is_rate_limit = status_code == 429 or "ratelimit" in error_name
            if not is_rate_limit or attempt >= MAX_RATE_LIMIT_RETRIES:
                raise
            wait_seconds = INITIAL_RETRY_WAIT_SECONDS * (attempt + 1)
            print(
                f"Provider rate limit reached; retrying in {wait_seconds}s "
                f"(retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Completion retry loop ended unexpectedly.")


def run_tool_calling(
    messages: list[dict[str, Any]],
    *,
    client: Any | None = None,
    max_turns: int = MAX_MODEL_TURNS,
    tool_choice: str | dict[str, Any] = "auto",
) -> ToolCallingResult:
    """Run a bounded model/tool conversation. Only catalogue lookup is executable."""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= MAX_MODEL_TURNS:
        raise ValueError(f"max_turns must be between 1 and {MAX_MODEL_TURNS}.")
    if not messages:
        raise ValueError("At least one conversation message is required.")

    conversation = [dict(message) for message in messages]
    owns_client = client is None
    if client is None:
        client = create_llm_client()

    try:
        tool_schema = build_brand_facts_tool_schema()
        executed_calls: list[ExecutedToolCall] = []
        exchanges: list[dict[str, Any]] = []

        for turn_number in range(1, max_turns + 1):
            response = _create_completion_with_retry(
                client,
                model=get_model_name(),
                messages=conversation,
                tools=tool_schema,
                tool_choice=tool_choice if turn_number == 1 else "auto",
            )
            if not response.choices:
                raise ToolCallingError("The provider returned no completion choices.")

            assistant_message = response.choices[0].message
            requested_calls = assistant_message.tool_calls or []
            if not requested_calls:
                final_text = assistant_message.content or ""
                if not final_text.strip():
                    raise ToolCallingError("The provider returned an empty final response.")
                exchanges.append({
                    "turn": turn_number,
                    "assistant_final": final_text,
                    "tool_requests": [],
                    "tool_results": [],
                })
                return ToolCallingResult(final_text, turn_number, executed_calls, exchanges)

            if turn_number == max_turns:
                raise TurnLimitExceeded(
                    f"The model requested a tool call on turn {turn_number}, reaching the {max_turns}-turn limit."
                )

            assistant_entry: dict[str, Any] = {"role": "assistant", "tool_calls": []}
            if assistant_message.content is not None:
                assistant_entry["content"] = assistant_message.content
            for call in requested_calls:
                if call.type != "function":
                    raise InvalidToolArgumentsError("Only function tool calls are allowed.")
                if not call.id:
                    raise InvalidToolArgumentsError("The tool request has no call ID.")
                assistant_entry["tool_calls"].append({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                })
            conversation.append(assistant_entry)

            exchange: dict[str, Any] = {
                "turn": turn_number,
                "tool_requests": [],
                "tool_results": [],
            }
            # Execute every requested tool from the single assistant turn.
            for call in requested_calls:
                arguments = _parse_tool_arguments(call.function.arguments)
                if call.function.name != TOOL_NAME:
                    raise InvalidToolArgumentsError("only get_brand_facts is permitted.")
                result = execute_tool(call.function.name, arguments)
                facts = result["facts"]
                conversation.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                executed_calls.append(ExecutedToolCall(
                    tool_name=call.function.name,
                    brand_id=result["brand_id"],
                    facts_returned=len(facts),
                    facts=[dict(fact) for fact in facts],
                ))
                exchange["tool_requests"].append({
                    "tool_name": TOOL_NAME,
                    "arguments": {"brand_id": result["brand_id"]},
                })
                exchange["tool_results"].append({
                    "brand_id": result["brand_id"],
                    "fact_version": result["fact_version"],
                    "facts_returned": len(facts),
                    "fact_ids": [fact["fact_id"] for fact in facts],
                })
            exchanges.append(exchange)

        raise TurnLimitExceeded(f"The model did not finish within {max_turns} turns.")
    finally:
        if owns_client:
            client.close()


def main() -> int:
    """Run a live smoke check of model-requested catalogue access."""
    messages = [
        {"role": "system", "content": "Use get_brand_facts before answering. Report the brand, catalogue version and fact count."},
        {"role": "user", "content": "Retrieve records for brand_id 'cleartrail' with get_brand_facts."},
    ]
    client = None
    try:
        client = create_llm_client()
        result = run_tool_calling(
            messages,
            client=client,
            max_turns=MAX_MODEL_TURNS,
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        print(f"Model: {get_model_name()}")
        print(f"Model turns: {result.model_turns}")
        print(f"Tool calls executed: {len(result.tool_calls)}")
        for call in result.tool_calls:
            print(f"Executed {call.tool_name} for {call.brand_id}; facts returned: {call.facts_returned}")
        print(result.final_text)
        if not any(call.brand_id == "cleartrail" for call in result.tool_calls):
            print("ERROR: ClearTrail catalogue lookup was not executed.")
            return 1
        return 0
    except Exception as error:
        print(f"Tool-calling check failed: {type(error).__name__}: {error}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
