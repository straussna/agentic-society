"""Canonical model-provider contracts and serializable value types."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable


USAGE_FIELDS = ("prefix_tokens", "uncached_input_tokens", "cache_read_tokens",
                "cache_write_tokens", "output_tokens", "reasoning_tokens")


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))


@dataclasses.dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Usage:
    prefix_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def zero(cls) -> "Usage":
        return cls()


@dataclasses.dataclass(frozen=True)
class Charge:
    kind: str
    tokens: int
    rate_centi_micros: int
    centi_micros: int
    multiplier_numerator: int = 1
    multiplier_denominator: int = 1

    def as_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Refusal:
    kind: str
    explanation: str | None = None
    recommended_model: str | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NormalizedTurn:
    id: str
    provider: str
    requested_model: str
    resolved_model: str
    stop_reason: str
    native_stop_reason: str | None
    text: tuple[str, ...]
    reasoning: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    charges: tuple[Charge, ...]
    refusal: Refusal | None = None
    native_stop_details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str
    context_window: int
    rates: tuple[tuple[str, int], ...]
    max_output_tokens: int | None = None
    long_context_threshold: int | None = None
    long_context_input_multiplier: tuple[int, int] = (1, 1)
    long_context_output_multiplier: tuple[int, int] = (1, 1)
    price_valid_through: str | None = None
    price_replacement: str | None = None

    def rate(self, kind: str) -> int:
        return dict(self.rates)[kind]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))


@dataclasses.dataclass(frozen=True)
class ProviderFailure:
    category: str
    provider: str
    message: str
    status_code: int | None = None
    native_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ProviderError(RuntimeError):
    """A provider failure classified without exposing SDK exception types."""

    def __init__(self, message: str, *, category: str, provider: str,
                 status_code: int | None = None, native_type: str | None = None):
        super().__init__(message)
        self.category = category
        self.provider = provider
        self.status_code = status_code
        self.native_type = native_type
        self.failure = ProviderFailure(category, provider, message, status_code, native_type)

    @property
    def retryable(self) -> bool:
        return self.category == "retryable_api"

    def as_dict(self) -> dict[str, Any]:
        return self.failure.as_dict()


class ProviderConfigurationError(ProviderError):
    def __init__(self, message: str, *, provider: str):
        super().__init__(message, category="adapter", provider=provider)


class PendingResponse:
    """A native response whose canonical form is produced only after raw logging."""

    def __init__(self, provider: str, native: dict[str, Any],
                 normalize: Callable[[], NormalizedTurn]):
        self.provider = provider
        self.native = native
        self._normalize = normalize
        self._normalized: NormalizedTurn | None = None

    def normalize(self) -> NormalizedTurn:
        if self._normalized is None:
            self._normalized = self._normalize()
        return self._normalized


@runtime_checkable
class ModelSession(Protocol):
    provider: str
    requested_model: str

    def request(self, content: str | tuple[ToolResult, ...]) -> PendingResponse: ...


@runtime_checkable
class ModelProvider(Protocol):
    name: str
    models: Mapping[str, ModelSpec]

    def preflight(self, models: Iterable[str]) -> None: ...
    def open_session(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                     max_tokens: int) -> ModelSession: ...
    def provenance(self, model: str) -> dict[str, Any]: ...


@runtime_checkable
class ProviderRouter(Protocol):
    def preflight(self) -> None: ...
    def open_session(self, provider: str, model: str, system: str,
                     tools: tuple[ToolSpec, ...], max_tokens: int) -> ModelSession: ...
    def provenance(self, provider: str, model: str) -> dict[str, Any]: ...


def native_dict(value: Any) -> dict[str, Any]:
    """Return the complete JSON-compatible representation of an SDK object."""
    if isinstance(value, dict):
        return json.loads(json.dumps(value, default=str))
    for method in ("model_dump", "to_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            return json.loads(json.dumps(fn(), default=str))
    if dataclasses.is_dataclass(value):
        return json.loads(json.dumps(dataclasses.asdict(value), default=str))
    return json.loads(json.dumps(value, default=lambda v: getattr(v, "__dict__", str(v))))


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def charge(kind: str, tokens: int, rate: int, multiplier: tuple[int, int] = (1, 1)) -> Charge:
    numerator, denominator = multiplier
    return Charge(kind, tokens, rate, tokens * rate * numerator // denominator,
                  numerator, denominator)
