import asyncio
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from app.agent.messages import AIMessage, AgentMessage, ToolCall, to_chat_message
from app.config import Settings


T = TypeVar("T", bound=BaseModel)
ToolDefinition = dict[str, Any]


class LLMAdapter(ABC):
    """Provider-neutral interface used by the invoice agent loop."""

    @abstractmethod
    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        response_model: type[T] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AIMessage:
        raise NotImplementedError


class UnconfiguredAdapter(LLMAdapter):
    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        response_model: type[T] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AIMessage:
        raise RuntimeError(
            "No LLM endpoint is configured. Set LLM_PROVIDER, LLM_MODEL, "
            "LLM_API_KEY and LLM_CHAT_COMPLETIONS_URL in .env."
        )


class OpenAIChatCompletionsAdapter(LLMAdapter):
    """Calls an exact OpenAI-compatible /chat/completions URL."""

    def __init__(self, settings: Settings) -> None:
        self.url = settings.llm_chat_completions_url
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        self.max_retries = settings.llm_max_retries
        self.temperature = settings.llm_temperature
        try:
            extra_headers = json.loads(settings.llm_extra_headers_json)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM_EXTRA_HEADERS_JSON must be a JSON object.") from exc
        if not isinstance(extra_headers, dict):
            raise ValueError("LLM_EXTRA_HEADERS_JSON must be a JSON object.")
        self.extra_headers = {str(key): str(value) for key, value in extra_headers.items()}

    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        response_model: type[T] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AIMessage:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                *[to_chat_message(message) for message in messages],
            ],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False
        if response_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": strict_json_schema(response_model.model_json_schema()),
                },
            }

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        data = await self._post(payload, headers)
        try:
            raw_message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Chat Completions response: {data}") from exc

        if raw_message.get("refusal"):
            raise RuntimeError(f"LLM refused extraction: {raw_message['refusal']}")

        tool_calls: list[ToolCall] = []
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid tool arguments for {function.get('name')}: {exc}") from exc
            tool_calls.append(
                ToolCall(
                    id=raw_call["id"],
                    name=function["name"],
                    arguments=arguments,
                )
            )

        content = normalize_content(raw_message.get("content"))
        if response_model is not None and not tool_calls:
            if not content:
                raise RuntimeError("The LLM returned no structured extraction content.")
            validated = response_model.model_validate_json(content)
            content = validated.model_dump_json()
        return AIMessage(content=content, tool_calls=tool_calls)

    async def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.url, headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    raise RuntimeError(
                        f"Chat Completions endpoint returned {response.status_code}: {response.text[:1000]}"
                    )
                result = response.json()
                if not isinstance(result, dict):
                    raise RuntimeError("Chat Completions endpoint did not return a JSON object.")
                return result
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Chat Completions request failed: {last_error}") from last_error


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic JSON Schema to strict object form for compatible endpoints."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties", {})
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def normalize_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts) or None
    return str(content)


def create_llm_adapter(settings: Settings) -> LLMAdapter:
    provider = settings.llm_provider.lower()
    if provider == "unconfigured":
        return UnconfiguredAdapter()
    if provider in {"openai", "openai_compatible", "chat_completions"}:
        return OpenAIChatCompletionsAdapter(settings)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
