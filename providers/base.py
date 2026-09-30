"""Canonical model-provider contracts and serializable value types."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


# What every adapter maps its native stop reasons onto, and the turn loop branches on.
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]

# How a provider failure is classified; retryable_api is the one a retry can cure.
FailureCategory = Literal["authentication", "retryable_api", "permanent_api", "adapter"]

# Exception classes, by name, of a request that got no answer: a dropped connection
# or a timeout. They carry no status code. A name anywhere in an exception's MRO
# counts, so the SDKs' subclasses of these and the standard library's do too.
RETRYABLE = {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"}


@dataclasses.dataclass(frozen=True)
class SessionContext:
    """Host context available to a provider for the lifetime of one episode."""

    agent: str
    label: str
    episode: int
    interaction_root: Path
    cancelled: Callable[[], bool] | None = None

    def is_cancelled(self) -> bool:
        return bool(self.cancelled and self.cancelled())


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


USAGE_FIELDS = tuple(f.name for f in dataclasses.fields(Usage))


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
    stop_reason: StopReason
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


class ProviderError(RuntimeError):
    """A provider failure classified without exposing SDK exception types."""

    def __init__(self, message: str, *, category: FailureCategory, provider: str,
                 status_code: int | None = None, native_type: str | None = None):
        super().__init__(message)
        self.category: FailureCategory = category
        self.provider = provider
        self.status_code = status_code
        self.native_type = native_type

    @property
    def retryable(self) -> bool:
        return self.category == "retryable_api"

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "provider": self.provider, "message": str(self),
                "status_code": self.status_code, "native_type": self.native_type}


class ProviderConfigurationError(ProviderError):
    def __init__(self, message: str, *, provider: str):
        super().__init__(message, category="adapter", provider=provider)


def classify_error(error: Exception, provider: str,
                   key_variable: str | None = None) -> ProviderError:
    """Classify an SDK exception by its status code and the names of its classes.

    401 and 403, or a class named for authentication or a denied permission, are
    authentication; 408, 409, 429 and 5xx are worth retrying, and any other status
    is permanent. With no status, a class named in RETRYABLE is a request that got
    no answer, and is retried; anything else is the adapter's. An authentication
    failure says where `key_variable` goes, when one is given.
    """
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    category: FailureCategory
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        category = "authentication"
    elif status in (408, 409, 429) or isinstance(status, int) and status >= 500:
        category = "retryable_api"
    elif status is not None:
        category = "permanent_api"
    elif any(cls.__name__ in RETRYABLE for cls in type(error).__mro__):
        category = "retryable_api"
    else:
        category = "adapter"
    message = f"{name}: {error}"
    if category == "authentication" and key_variable:
        message = f"{message} {where_key_goes(key_variable)}"
    return ProviderError(message, category=category, provider=provider,
                         status_code=status, native_type=name)


def refuse_custom_endpoint(variable: str, provider: str) -> None:
    """Refuse an endpoint redirected through `variable`: the adapters speak first-party only."""
    if os.environ.get(variable):
        raise ProviderConfigurationError(
            f"{variable} is not supported; provider adapters use first-party endpoints",
            provider=provider)


def where_key_goes(variable: str) -> str:
    return f"Set {variable} in the shell this experiment is launched from."


def require_key(variable: str, provider: str) -> None:
    """Refuse a provider whose API key is not in the environment, saying where it goes."""
    if not os.environ.get(variable):
        raise ProviderError(f"{variable} is not set. {where_key_goes(variable)}",
                            category="authentication", provider=provider)


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
    """A provider adapter, whose class declares what is recorded of it.

    `interactive` says a person, not a model, answers its turns. `provenance_facts`
    is the adapter's part of a trace's provider record, in the order the trace keeps
    it; providers.provenance() puts the name before it and the model spec after.
    """

    name: str
    interactive: bool

    @property
    def models(self) -> Mapping[str, ModelSpec]: ...
    @property
    def provenance_facts(self) -> Mapping[str, Any]: ...

    def preflight(self, models: Iterable[str]) -> None: ...
    def open_session(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                     max_tokens: int, context: SessionContext) -> ModelSession: ...


@runtime_checkable
class ProviderRouter(Protocol):
    def preflight(self) -> None: ...
    def open_session(self, provider: str, model: str, system: str,
                     tools: tuple[ToolSpec, ...], max_tokens: int,
                     context: SessionContext) -> ModelSession: ...


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
