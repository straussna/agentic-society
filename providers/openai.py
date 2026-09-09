"""Direct OpenAI Responses API provider."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from .base import (ModelSpec, NormalizedTurn, PendingResponse, ProviderConfigurationError,
                   ProviderError, Refusal, ToolCall, ToolResult, ToolSpec, Usage, charge, field,
                   native_dict)


MODELS = {
    "gpt-5.6-sol": ModelSpec(
        "openai", "gpt-5.6-sol", 1_050_000,
        (("uncached_input", 400), ("cache_read", 40), ("cache_write", 500),
         ("output", 2000)), 128_000, 272_000, (2, 1), (3, 2),
        "2026-11-21", "Update the gpt-5.6-sol promotional rates before running."),
    "gpt-5.6-terra": ModelSpec(
        "openai", "gpt-5.6-terra", 1_050_000,
        (("uncached_input", 200), ("cache_read", 20), ("cache_write", 250),
         ("output", 1200)), 128_000, 272_000, (2, 1), (3, 2)),
    "gpt-5.6-luna": ModelSpec(
        "openai", "gpt-5.6-luna", 1_050_000,
        (("uncached_input", 20), ("cache_read", 2), ("cache_write", 25),
         ("output", 120)), 128_000, 272_000, (2, 1), (3, 2)),
}


def _error(error: Exception) -> ProviderError:
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    text = f"{name}: {error}"
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        category = "authentication"
    elif status in (408, 409, 429) or isinstance(status, int) and status >= 500:
        category = "retryable_api"
    elif status is not None:
        category = "permanent_api"
    else:
        category = "adapter"
    return ProviderError(text, category=category, provider="openai",
                         status_code=status, native_type=name)


def normalize(native: Any, requested_model: str) -> NormalizedTurn:
    texts, reasoning, calls, refusals = [], [], [], []
    for item in field(native, "output", []) or []:
        kind = field(item, "type")
        if kind == "function_call":
            arguments = field(item, "arguments", "{}") or "{}"
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ProviderError(f"invalid function arguments: {error}", category="adapter",
                                    provider="openai", native_type=type(error).__name__) from error
            if not isinstance(parsed, dict):
                raise ProviderError("function arguments are not a JSON object", category="adapter",
                                    provider="openai")
            calls.append(ToolCall(str(field(item, "call_id", field(item, "id", ""))),
                                  str(field(item, "name", "")), parsed))
        elif kind == "reasoning":
            for summary in field(item, "summary", []) or []:
                text = field(summary, "text")
                if text:
                    reasoning.append(str(text))
        elif kind == "message":
            for part in field(item, "content", []) or []:
                part_kind = field(part, "type")
                if part_kind == "output_text":
                    texts.append(str(field(part, "text", "")))
                elif part_kind == "refusal":
                    refusals.append(str(field(part, "refusal", "")))
    raw_usage = field(native, "usage", {}) or {}
    input_tokens = int(field(raw_usage, "input_tokens", 0) or 0)
    input_details = field(raw_usage, "input_tokens_details", {}) or {}
    cached = int(field(input_details, "cached_tokens", 0) or 0)
    cache_write = int(field(input_details, "cache_write_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached - cache_write)
    output = int(field(raw_usage, "output_tokens", 0) or 0)
    output_details = field(raw_usage, "output_tokens_details", {}) or {}
    reasoning_tokens = int(field(output_details, "reasoning_tokens", 0) or 0)
    usage = Usage(input_tokens, uncached, cached, cache_write, output, reasoning_tokens)
    spec = MODELS[requested_model]
    long = input_tokens > (spec.long_context_threshold or spec.context_window)
    input_multiplier = spec.long_context_input_multiplier if long else (1, 1)
    output_multiplier = spec.long_context_output_multiplier if long else (1, 1)
    charges = (
        charge("uncached_input", uncached, spec.rate("uncached_input"), input_multiplier),
        charge("cache_read", cached, spec.rate("cache_read"), input_multiplier),
        charge("cache_write", cache_write, spec.rate("cache_write"), input_multiplier),
        charge("output", output, spec.rate("output"), output_multiplier),
    )
    status = str(field(native, "status", ""))
    incomplete = field(native, "incomplete_details", {}) or {}
    native_stop = field(incomplete, "reason") or status or None
    if refusals:
        stop = "refusal"
    elif status == "incomplete" and native_stop in ("max_output_tokens", "max_tokens"):
        stop = "max_tokens"
    elif calls:
        stop = "tool_use"
    elif status == "completed":
        stop = "end_turn"
    else:
        stop = "other"
    refusal = Refusal("refusal", "\n".join(refusals)) if refusals else None
    stop_details = native_dict(incomplete) if incomplete else None
    return NormalizedTurn(str(field(native, "id", "")), "openai", requested_model,
                          str(field(native, "model", requested_model)), stop,
                          str(native_stop) if native_stop is not None else None,
                          tuple(texts), tuple(reasoning), tuple(calls), usage, charges,
                          refusal, stop_details)


class OpenAISession:
    provider = "openai"

    def __init__(self, client: Any, model: str, system: str, tools: tuple[ToolSpec, ...],
                 max_tokens: int):
        self.client = client
        self.requested_model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens
        self.items: list[dict[str, Any]] = []

    def request(self, content: str | tuple[ToolResult, ...]) -> PendingResponse:
        if isinstance(content, str):
            additions = [{"role": "user", "content": [{"type": "input_text", "text": content}]}]
        else:
            additions = [{"type": "function_call_output", "call_id": result.tool_call_id,
                          "output": result.content} for result in content]
        items = [*self.items, *additions]
        params: dict[str, Any] = {
            "model": self.requested_model,
            "input": items,
            "max_output_tokens": self.max_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "tools": [{"type": "function", "name": tool.name, "description": tool.description,
                       "parameters": tool.input_schema, "strict": True} for tool in self.tools],
        }
        if self.system:
            params["instructions"] = self.system
        try:
            response = self.client.responses.create(**params)
        except Exception as error:
            raise _error(error) from error
        raw = native_dict(response)

        def finish() -> NormalizedTurn:
            turn = normalize(response, self.requested_model)
            self.items = [*items, *[native_dict(item) for item in field(response, "output", []) or []]]
            return turn
        return PendingResponse(self.provider, raw, finish)


class OpenAIProvider:
    name = "openai"
    models = MODELS

    def __init__(self, client: Any | None = None):
        if os.environ.get("OPENAI_BASE_URL"):
            raise ProviderConfigurationError(
                "OPENAI_BASE_URL is not supported; provider adapters use first-party endpoints",
                provider=self.name)
        if client is None:
            try:
                import openai
                client = openai.OpenAI(max_retries=0)
            except Exception as error:
                raise _error(error) from error
        self.client = client

    def preflight(self, models: Iterable[str]) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set", category="authentication",
                                provider=self.name)
        for model in sorted(set(models)):
            try:
                self.client.models.retrieve(model)
            except Exception as error:
                raise _error(error) from error

    def open_session(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                     max_tokens: int) -> OpenAISession:
        return OpenAISession(self.client, model, system, tools, max_tokens)

    def provenance(self, model: str) -> dict[str, Any]:
        return {"name": self.name, "adapter": "openai-responses-v1",
                "endpoint": "first-party", "model_spec": self.models[model].as_dict(),
                "store": False, "reasoning_state": "encrypted"}
