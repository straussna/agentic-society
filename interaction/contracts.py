"""Serializable records exchanged by interactive providers and their clients."""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from providers.base import ToolCall, ToolSpec


VERSION = 1


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclasses.dataclass(frozen=True)
class InteractionRequest:
    version: int
    request_id: str
    agent: str
    label: str
    episode: int
    turn: int
    created_at: str
    system_prompt: str
    input: dict[str, Any]
    available_tools: tuple[ToolSpec, ...]
    status: str = "pending"

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["available_tools"] = [tool.as_dict() for tool in self.available_tools]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InteractionRequest":
        return cls(int(value["version"]), str(value["request_id"]), str(value["agent"]),
                   str(value["label"]), int(value["episode"]), int(value["turn"]),
                   str(value["created_at"]), str(value.get("system_prompt", "")),
                   dict(value["input"]), tuple(ToolSpec(str(tool["name"]),
                   str(tool.get("description", "")), dict(tool["input_schema"]))
                   for tool in value["available_tools"]), str(value["status"]))


@dataclasses.dataclass(frozen=True)
class Submission:
    version: int
    request_id: str
    submission_id: str
    action: str
    tool_calls: tuple[ToolCall, ...]
    submitted_at: str

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["tool_calls"] = [call.as_dict() for call in self.tool_calls]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Submission":
        return cls(int(value["version"]), str(value["request_id"]),
                   str(value["submission_id"]), str(value["action"]),
                   tuple(ToolCall(str(call["id"]), str(call["name"]), dict(call["input"]))
                         for call in value.get("tool_calls", [])),
                   str(value["submitted_at"]))
