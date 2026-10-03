"""Registered model providers."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .anthropic import AnthropicProvider
from .base import (
    Charge,
    FailureCategory,
    ModelProvider,
    ModelSession,
    ModelSpec,
    NormalizedTurn,
    PendingResponse,
    ProviderConfigurationError,
    ProviderError,
    ProviderRouter,
    Refusal,
    SessionContext,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSpec,
    USAGE_FIELDS,
    Usage,
)
from .human import HumanProvider
from .openai import OpenAIProvider


FACTORIES = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "human": HumanProvider}
CATALOGS = {name: factory.models for name, factory in FACTORIES.items()}


def provider_names() -> tuple[str, ...]:
    return tuple(FACTORIES)


def is_interactive(provider: str | None) -> bool:
    """Whether a person, not a model, answers this provider's turns.

    No one answers for a provider no adapter is registered under, or for none.
    """
    return provider in FACTORIES and FACTORIES[provider].interactive


def model_spec(provider: str, model: str) -> ModelSpec:
    if provider not in CATALOGS:
        raise ProviderConfigurationError(
            f"unknown provider {provider!r}; choose one of {list(provider_names())}", provider=provider
        )
    if model not in CATALOGS[provider]:
        raise ProviderConfigurationError(
            f"unknown {provider} model {model!r}; choose one of {list(CATALOGS[provider])}", provider=provider
        )
    return CATALOGS[provider][model]


def provenance(provider: str, model: str) -> dict[str, Any]:
    """The trace's provider record: the name, what the adapter declares, and the model."""
    spec = model_spec(provider, model)
    return {"name": provider, **FACTORIES[provider].provenance_facts, "model_spec": spec.as_dict()}


def lapsed_prices(requirements: Iterable[tuple[str, str]], today: dt.date | None = None) -> list[str]:
    today = today or dt.date.today()
    lapsed = []
    for provider, model in sorted(set(requirements)):
        spec = model_spec(provider, model)
        if spec.price_valid_through and today > dt.date.fromisoformat(spec.price_valid_through):
            lapsed.append(
                f"{provider}/{model}: rates expired after {spec.price_valid_through}. "
                f"{spec.price_replacement or 'Update the provider catalog.'}"
            )
    return lapsed


class DirectProviderRouter:
    def __init__(self, requirements: Iterable[tuple[str, str]], instances: dict[str, ModelProvider] | None = None):
        grouped: dict[str, set[str]] = defaultdict(set)
        for provider, model in requirements:
            model_spec(provider, model)
            grouped[provider].add(model)
        self.requirements = {name: frozenset(models) for name, models in grouped.items()}
        self.providers: dict[str, ModelProvider] = dict(instances or {})
        for name in self.requirements:
            if name not in self.providers:
                self.providers[name] = FACTORIES[name]()

    def preflight(self) -> None:
        for name, models in self.requirements.items():
            self.providers[name].preflight(models)

    def open_session(
        self,
        provider: str,
        model: str,
        system: str,
        tools: tuple[ToolSpec, ...],
        max_tokens: int,
        context: SessionContext,
    ) -> ModelSession:
        model_spec(provider, model)
        return self.providers[provider].open_session(model, system, tools, max_tokens, context)


__all__ = [
    "Charge",
    "DirectProviderRouter",
    "FailureCategory",
    "ModelProvider",
    "ModelSession",
    "ModelSpec",
    "NormalizedTurn",
    "PendingResponse",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRouter",
    "Refusal",
    "SessionContext",
    "StopReason",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "USAGE_FIELDS",
    "Usage",
    "is_interactive",
    "lapsed_prices",
    "model_spec",
    "provenance",
    "provider_names",
]
