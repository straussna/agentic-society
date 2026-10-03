"""Serializable records exchanged by interactive providers and their clients."""

from __future__ import annotations

import dataclasses
from typing import Any

from product import now
from providers.base import ToolCall, ToolSpec


VERSION = 1


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
        return cls(
            int(value["version"]),
            str(value["request_id"]),
            str(value["agent"]),
            str(value["label"]),
            int(value["episode"]),
            int(value["turn"]),
            str(value["created_at"]),
            str(value.get("system_prompt", "")),
            dict(value["input"]),
            tuple(
                ToolSpec(
                    str(tool["name"]),
                    str(tool.get("description", "")),
                    dict(tool["input_schema"]),
                )
                for tool in value["available_tools"]
            ),
            str(value["status"]),
        )


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
        return cls(
            int(value["version"]),
            str(value["request_id"]),
            str(value["submission_id"]),
            str(value["action"]),
            tuple(
                ToolCall(str(call["id"]), str(call["name"]), dict(call["input"]))
                for call in value.get("tool_calls", [])
            ),
            str(value["submitted_at"]),
        )

    @classmethod
    def parse(cls, value: dict[str, Any], request: InteractionRequest) -> "Submission":
        """A client's envelope as its submission to `request`, stamped now.

        Refused with a ValueError naming the first fault: a version other than VERSION,
        another request's id, no submission_id, an action other than tool_calls or
        end_turn, calls that do not fit the action, or a call to a tool not offered.
        """
        try:
            version = int(value.get("version", VERSION))
        except (TypeError, ValueError) as error:
            raise ValueError("malformed submission envelope") from error
        if version != VERSION:
            raise ValueError("unsupported submission version")
        if value.get("request_id", request.request_id) != request.request_id:
            raise ValueError("submission request_id does not match the URL")
        submission_id = value.get("submission_id")
        action = value.get("action")
        raw_calls = value.get("tool_calls", [])
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError("submission_id must be a non-empty string")
        if action not in ("tool_calls", "end_turn") or not isinstance(raw_calls, list):
            raise ValueError("action must be tool_calls or end_turn")
        calls = []
        for call in raw_calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("input"), dict)
            ):
                raise ValueError("tool calls require string id/name and object input")
            calls.append(ToolCall(call["id"], call["name"], call["input"]))
        if action == "end_turn" and calls:
            raise ValueError("end_turn cannot contain tool calls")
        if action == "tool_calls" and not calls:
            raise ValueError("tool_calls requires at least one call")
        names = {tool.name for tool in request.available_tools}
        if any(
            not call.id or not call.name or call.name not in names for call in calls
        ):
            raise ValueError("each call needs an id and a declared tool name")
        return cls(
            VERSION, request.request_id, submission_id, action, tuple(calls), now()
        )

    def same_as(self, other: "Submission") -> bool:
        """Whether `other` is this submission sent again: the same id, action and calls."""
        return (self.submission_id, self.action, self.tool_calls) == (
            other.submission_id,
            other.action,
            other.tool_calls,
        )
