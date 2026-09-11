"""Atomic filesystem coordination for interactive turns."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from providers.base import ToolCall, ToolResult, ToolSpec

from .contracts import InteractionRequest, Submission, VERSION, timestamp


class InteractionError(RuntimeError):
    pass


class InvalidSubmission(InteractionError):
    pass


class StaleRequest(InteractionError):
    pass


class InteractionConflict(InteractionError):
    pass


class InteractionCancelled(InteractionError):
    pass


class InteractionStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _agent(self, agent: str) -> str:
        if not agent or agent in (".", "..") or Path(agent).name != agent or \
                "/" in agent or "\\" in agent:
            raise InteractionError("invalid agent identifier")
        return agent

    def _request_path(self, agent: str, request_id: str) -> Path:
        self._agent(agent)
        if not request_id or Path(request_id).name != request_id:
            raise InteractionError("invalid request identifier")
        return self.root / "requests" / agent / f"{request_id}.json"

    def _pending_path(self, agent: str) -> Path:
        return self.root / "pending" / f"{self._agent(agent)}.json"

    def _submission_path(self, agent: str, request_id: str) -> Path:
        return self.root / "submissions" / self._agent(agent) / f"{request_id}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        for attempt in (1, 2, 3):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                if attempt == 3:
                    return None
                time.sleep(0.02)
        return None

    @staticmethod
    def _atomic(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def publish(self, agent: str, label: str, episode: int, turn: int, system_prompt: str,
                content: str | tuple[ToolResult, ...],
                tools: tuple[ToolSpec, ...]) -> InteractionRequest:
        previous = self.current(agent)
        if previous is not None:
            self.cancel(agent, previous.request_id)
        request_id = uuid.uuid4().hex
        supplied = ({"kind": "initial_observation", "text": content}
                    if isinstance(content, str) else
                    {"kind": "tool_results", "results": [item.as_dict() for item in content]})
        request = InteractionRequest(VERSION, request_id, agent, label, episode, turn,
                                     timestamp(), system_prompt, supplied, tools)
        self._atomic(self._request_path(agent, request_id), request.as_dict())
        self._atomic(self._pending_path(agent), {"request_id": request_id})
        return request

    def request(self, agent: str, request_id: str) -> InteractionRequest | None:
        value = self._read(self._request_path(agent, request_id))
        return InteractionRequest.from_dict(value) if value else None

    def current(self, agent: str) -> InteractionRequest | None:
        pointer = self._read(self._pending_path(agent))
        if not pointer or not isinstance(pointer.get("request_id"), str):
            return None
        request = self.request(agent, pointer["request_id"])
        return request if request and request.status == "pending" else None

    def pending(self) -> list[InteractionRequest]:
        directory = self.root / "pending"
        if not directory.is_dir():
            return []
        out = []
        for path in sorted(directory.glob("*.json")):
            try:
                request = self.current(path.stem)
            except InteractionError:
                continue
            if request is not None:
                out.append(request)
        return out

    def _finish(self, agent: str, request_id: str, status: str) -> None:
        request = self.request(agent, request_id)
        if request is None:
            return
        self._atomic(self._request_path(agent, request_id),
                     {**request.as_dict(), "status": status})
        pointer = self._read(self._pending_path(agent))
        if pointer and pointer.get("request_id") == request_id:
            self._pending_path(agent).unlink(missing_ok=True)

    def cancel(self, agent: str, request_id: str) -> None:
        self._finish(agent, request_id, "cancelled")

    def complete(self, agent: str, request_id: str) -> None:
        self._finish(agent, request_id, "completed")

    def submit(self, agent: str, request_id: str, value: dict[str, Any]) -> Submission:
        request = self.request(agent, request_id)
        if request is None:
            raise StaleRequest("the request is no longer pending")
        try:
            if int(value.get("version", VERSION)) != VERSION:
                raise InvalidSubmission("unsupported submission version")
            if value.get("request_id", request_id) != request_id:
                raise InvalidSubmission("submission request_id does not match the URL")
            submission_id = value["submission_id"]
            action = value["action"]
            raw_calls = value.get("tool_calls", [])
            if not isinstance(submission_id, str) or not submission_id:
                raise InvalidSubmission("submission_id must be a non-empty string")
            if action not in ("tool_calls", "end_turn") or not isinstance(raw_calls, list):
                raise InvalidSubmission("action must be tool_calls or end_turn")
            calls = []
            for call in raw_calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or \
                        not isinstance(call.get("name"), str) or not isinstance(call.get("input"), dict):
                    raise InvalidSubmission("tool calls require string id/name and object input")
                calls.append(ToolCall(call["id"], call["name"], call["input"]))
            calls = tuple(calls)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidSubmission("malformed submission envelope") from error
        if action == "end_turn" and calls:
            raise InvalidSubmission("end_turn cannot contain tool calls")
        if action == "tool_calls" and not calls:
            raise InvalidSubmission("tool_calls requires at least one call")
        names = {tool.name for tool in request.available_tools}
        if any(not call.id or not call.name or call.name not in names for call in calls):
            raise InvalidSubmission("each call needs an id and a declared tool name")
        submission = Submission(VERSION, request_id, submission_id, action, calls, timestamp())
        path = self._submission_path(agent, request_id)
        if existing_value := self._read(path):
            existing = Submission.from_dict(existing_value)
            if existing.submission_id == submission.submission_id and \
                    existing.action == submission.action and existing.tool_calls == submission.tool_calls:
                return existing
            raise InteractionConflict("a different submission already won this request")
        current = self.current(agent)
        if current is None or current.request_id != request_id:
            raise StaleRequest("the request is no longer pending")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self._atomic(temporary, submission.as_dict())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing_value = self._read(path)
            if not existing_value:
                raise InteractionConflict("a submission already exists")
            existing = Submission.from_dict(existing_value)
            if existing.submission_id != submission.submission_id or \
                    existing.action != submission.action or existing.tool_calls != submission.tool_calls:
                raise InteractionConflict("a different submission already won this request")
            return existing
        finally:
            temporary.unlink(missing_ok=True)
        return submission

    def wait(self, request: InteractionRequest, cancelled: Callable[[], bool],
             interval: float = 0.1) -> Submission:
        path = self._submission_path(request.agent, request.request_id)
        while True:
            if cancelled():
                self.cancel(request.agent, request.request_id)
                raise InteractionCancelled("interactive request cancelled")
            value = self._read(path)
            if value:
                submission = Submission.from_dict(value)
                self.complete(request.agent, request.request_id)
                return submission
            time.sleep(interval)

    def load_draft(self, agent: str, request_id: str) -> list[dict[str, Any]]:
        value = self._read(self.root / "drafts" / self._agent(agent) / f"{request_id}.json")
        return list(value.get("tool_calls", [])) if value else []

    def save_draft(self, agent: str, request_id: str, calls: list[dict[str, Any]]) -> None:
        current = self.current(agent)
        if current is None or current.request_id != request_id:
            raise StaleRequest("the request is no longer pending")
        self._atomic(self.root / "drafts" / self._agent(agent) / f"{request_id}.json",
                     {"version": VERSION, "request_id": request_id, "tool_calls": calls})
