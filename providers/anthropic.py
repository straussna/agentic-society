"""Direct Anthropic Messages API provider."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .base import (Charge, ModelSpec, NormalizedTurn, PendingResponse, ProviderConfigurationError,
                   ProviderError, Refusal, ToolCall, ToolResult, ToolSpec, Usage, charge, field,
                   native_dict)


MODELS = {
    name: ModelSpec("anthropic", name, window,
                    (("uncached_input", input_rate), ("cache_read", input_rate // 10),
                     ("cache_write_5m", input_rate * 5 // 4),
                     ("cache_write_1h", input_rate * 2), ("output", output_rate)),
                    long_context_threshold=200_000 if window > 200_000 else None,
                    long_context_input_multiplier=(2, 1) if window > 200_000 else (1, 1),
                    long_context_output_multiplier=(3, 2) if window > 200_000 else (1, 1))
    for name, input_rate, output_rate, window in (
        ("claude-fable-5", 1000, 5000, 1_000_000),
        ("claude-mythos-5", 1000, 5000, 1_000_000),
        ("claude-opus-5", 500, 2500, 1_000_000),
        ("claude-opus-4-8", 500, 2500, 1_000_000),
        ("claude-opus-4-5", 500, 2500, 200_000),
        ("claude-sonnet-5", 300, 1500, 1_000_000),
        ("claude-haiku-4-5", 100, 500, 200_000),
    )
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
    return ProviderError(text, category=category, provider="anthropic",
                         status_code=status, native_type=name)


def normalize(native: Any, requested_model: str) -> NormalizedTurn:
    content = field(native, "content", []) or []
    texts, reasoning, calls = [], [], []
    for block in content:
        kind = field(block, "type")
        if kind == "text":
            texts.append(str(field(block, "text", "")))
        elif kind in ("thinking", "redacted_thinking"):
            text = field(block, "thinking") or field(block, "data")
            if text:
                reasoning.append(str(text))
        elif kind == "tool_use":
            calls.append(ToolCall(str(field(block, "id", "")), str(field(block, "name", "")),
                                  dict(field(block, "input", {}) or {})))

    raw_usage = field(native, "usage", {}) or {}
    uncached = int(field(raw_usage, "input_tokens", 0) or 0)
    cache_read = int(field(raw_usage, "cache_read_input_tokens", 0) or 0)
    cache_5m = int(field(raw_usage, "cache_creation_input_tokens", 0) or 0)
    cache_creation = field(raw_usage, "cache_creation", {}) or {}
    if cache_creation:
        cache_5m = int(field(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
        cache_1h = int(field(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
    else:
        cache_1h = 0
    output = int(field(raw_usage, "output_tokens", 0) or 0)
    usage = Usage(uncached + cache_read + cache_5m + cache_1h, uncached, cache_read,
                  cache_5m + cache_1h, output, 0)
    spec = MODELS[requested_model]
    long = usage.prefix_tokens > (spec.long_context_threshold or spec.context_window)
    input_multiplier = spec.long_context_input_multiplier if long else (1, 1)
    output_multiplier = spec.long_context_output_multiplier if long else (1, 1)
    charges: tuple[Charge, ...] = (
        charge("uncached_input", uncached, spec.rate("uncached_input"), input_multiplier),
        charge("cache_read", cache_read, spec.rate("cache_read"), input_multiplier),
        charge("cache_write_5m", cache_5m, spec.rate("cache_write_5m"), input_multiplier),
        charge("cache_write_1h", cache_1h, spec.rate("cache_write_1h"), input_multiplier),
        charge("output", output, spec.rate("output"), output_multiplier),
    )
    native_stop = field(native, "stop_reason")
    if native_stop == "refusal" and not content:
        charges = ()
    stop = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
            "refusal": "refusal"}.get(native_stop, "other")
    stop_details = native_dict(field(native, "stop_details", {})) if field(native, "stop_details") else None
    refusal = None
    if native_stop == "refusal" or field(native, "refusal"):
        details = field(native, "refusal") or stop_details or {}
        refusal = Refusal(str(field(details, "type", "refusal")),
                          field(details, "explanation"), field(details, "recommended_model"),
                          native_dict(details) if details else None)
    return NormalizedTurn(str(field(native, "id", "")), "anthropic", requested_model,
                          str(field(native, "model", requested_model)), stop,
                          str(native_stop) if native_stop is not None else None, tuple(texts),
                          tuple(reasoning), tuple(calls), usage, charges, refusal, stop_details)


class AnthropicSession:
    provider = "anthropic"

    def __init__(self, client: Any, model: str, system: str, tools: tuple[ToolSpec, ...],
                 max_tokens: int):
        self.client = client
        self.requested_model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def request(self, content: str | tuple[ToolResult, ...]) -> PendingResponse:
        if isinstance(content, str):
            user = {"role": "user", "content": content}
        else:
            user = {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": result.tool_call_id,
                 "content": result.content, "is_error": result.is_error}
                for result in content]}
        messages = [*self.messages, user]
        params: dict[str, Any] = {
            "model": self.requested_model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tools": [{"name": tool.name, "description": tool.description,
                       "input_schema": tool.input_schema, "strict": True}
                      for tool in self.tools],
            "cache_control": {"type": "ephemeral"},
        }
        if self.system:
            params["system"] = self.system
        try:
            response = self.client.messages.create(**params)
        except Exception as error:
            raise _error(error) from error
        raw = native_dict(response)

        def finish() -> NormalizedTurn:
            turn = normalize(response, self.requested_model)
            response_content = field(response, "content", []) or []
            self.messages = [*messages]
            if response_content:
                self.messages.append({"role": "assistant", "content": response_content})
            return turn
        return PendingResponse(self.provider, raw, finish)


class AnthropicProvider:
    name = "anthropic"
    models = MODELS

    def __init__(self, client: Any | None = None):
        if os.environ.get("ANTHROPIC_BASE_URL"):
            raise ProviderConfigurationError(
                "ANTHROPIC_BASE_URL is not supported; provider adapters use first-party endpoints",
                provider=self.name)
        if client is None:
            try:
                import anthropic
                client = anthropic.Anthropic(max_retries=0)
            except Exception as error:
                raise _error(error) from error
        self.client = client

    def preflight(self, models: Iterable[str]) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set", category="authentication",
                                provider=self.name)
        for model in sorted(set(models)):
            try:
                self.client.models.retrieve(model_id=model)
            except Exception as error:
                raise _error(error) from error

    def open_session(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                     max_tokens: int) -> AnthropicSession:
        return AnthropicSession(self.client, model, system, tools, max_tokens)

    def provenance(self, model: str) -> dict[str, Any]:
        return {"name": self.name, "adapter": "anthropic-messages-v1",
                "endpoint": "first-party", "model_spec": self.models[model].as_dict()}
