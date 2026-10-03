"""Atomic filesystem coordination for interactive turns."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from product import atomic, now
from providers.base import ToolResult, ToolSpec

from .contracts import InteractionRequest, Submission, VERSION


T = TypeVar("T")


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


class UnreadableRecord(InteractionError):
    """A file the store keeps is there and does not hold the record it should."""


class InteractionStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _agent(self, agent: str) -> str:
        if (
            not agent
            or agent in (".", "..")
            or Path(agent).name != agent
            or "/" in agent
            or "\\" in agent
        ):
            raise InteractionError("invalid agent identifier")
        return agent

    def _requests(self, agent: str) -> Path:
        return self.root / "requests" / self._agent(agent)

    def _request_path(self, agent: str, request_id: str) -> Path:
        directory = self._requests(agent)
        if not request_id or Path(request_id).name != request_id:
            raise InteractionError("invalid request identifier")
        return directory / f"{request_id}.json"

    def _pending_path(self, agent: str) -> Path:
        return self.root / "pending" / f"{self._agent(agent)}.json"

    def _submission_path(self, agent: str, request_id: str) -> Path:
        return self.root / "submissions" / self._agent(agent) / f"{request_id}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        """The JSON object at `path`, or None when there is no file there.

        A file that is there and is not a JSON object is read three times, since on
        Windows a read can meet a rename in progress, and then raises UnreadableRecord.
        """
        for attempt in (1, 2, 3):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except OSError, ValueError:
                value = None
            if isinstance(value, dict):
                return value
            if attempt < 3:
                time.sleep(0.02)
        raise UnreadableRecord(f"{path} is not a readable JSON object")

    def _load(self, path: Path, record: Callable[[dict[str, Any]], T]) -> T | None:
        """The file at `path` as `record` reads it, or None when there is no file there."""
        value = self._read(path)
        if value is None:
            return None
        try:
            return record(value)
        except (KeyError, TypeError, ValueError) as error:
            raise UnreadableRecord(f"{path} does not hold a complete record") from error

    def publish(
        self,
        agent: str,
        label: str,
        episode: int,
        turn: int,
        system_prompt: str,
        content: str | tuple[ToolResult, ...],
        tools: tuple[ToolSpec, ...],
    ) -> InteractionRequest:
        """A new pending request for `agent`, which cancels the one it replaces.

        A pending pointer that does not read names nothing to cancel, and this one
        is written over it.
        """
        try:
            previous = self.current(agent)
        except UnreadableRecord:
            previous = None
        if previous is not None:
            self.cancel(agent, previous.request_id)
        request_id = uuid.uuid4().hex
        supplied = (
            {"kind": "initial_observation", "text": content}
            if isinstance(content, str)
            else {
                "kind": "tool_results",
                "results": [item.as_dict() for item in content],
            }
        )
        request = InteractionRequest(
            VERSION,
            request_id,
            agent,
            label,
            episode,
            turn,
            now(),
            system_prompt,
            supplied,
            tools,
        )
        atomic(self._request_path(agent, request_id), request.as_dict())
        atomic(self._pending_path(agent), {"request_id": request_id})
        return request

    def request(self, agent: str, request_id: str) -> InteractionRequest | None:
        return self._load(
            self._request_path(agent, request_id), InteractionRequest.from_dict
        )

    def current(self, agent: str) -> InteractionRequest | None:
        path = self._pending_path(agent)
        pointer = self._read(path)
        if pointer is None:
            return None
        if not isinstance(pointer.get("request_id"), str):
            raise UnreadableRecord(f"{path} names no request")
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

    def history(self, agent: str) -> list[tuple[InteractionRequest, Submission]]:
        """Every request of `agent`'s that a submission answered, with that submission,
        in episode and turn order.

        A pair either of whose files does not read is left out, as pending() leaves
        out a seat it cannot read.
        """
        directory = self._requests(agent)
        if not directory.is_dir():
            return []
        out = []
        for path in directory.glob("*.json"):
            try:
                submission = self._load(
                    self._submission_path(agent, path.stem), Submission.from_dict
                )
                request = None if submission is None else self.request(agent, path.stem)
            except UnreadableRecord:
                continue
            if request is not None and submission is not None:
                out.append((request, submission))
        return sorted(
            out, key=lambda pair: (pair[0].episode, pair[0].turn, pair[0].created_at)
        )

    def _finish(self, agent: str, request_id: str, status: str) -> None:
        """Mark the request `status` and take down the pending pointer if it names it.

        Either file is left as it is when it does not read: nothing can be rewritten
        from it, and the next publish writes over the pointer.
        """
        try:
            request = self.request(agent, request_id)
        except UnreadableRecord:
            request = None
        if request is not None:
            atomic(
                self._request_path(agent, request_id),
                {**request.as_dict(), "status": status},
            )
        try:
            pointer = self._read(self._pending_path(agent))
        except UnreadableRecord:
            return
        if pointer and pointer.get("request_id") == request_id:
            self._pending_path(agent).unlink(missing_ok=True)

    def cancel(self, agent: str, request_id: str) -> None:
        self._finish(agent, request_id, "cancelled")

    def complete(self, agent: str, request_id: str) -> None:
        self._finish(agent, request_id, "completed")

    def submit(self, agent: str, request_id: str, value: dict[str, Any]) -> Submission:
        """Answer a pending request with the envelope `value`; the first to land wins it.

        An envelope Submission.parse refuses raises InvalidSubmission, a request no
        longer pending StaleRequest, and any submission but the winner InteractionConflict;
        the winner sent again is returned. A request, pending pointer or submission that
        is there and does not read raises UnreadableRecord.
        """
        request = self.request(agent, request_id)
        if request is None:
            raise StaleRequest("the request is no longer pending")
        try:
            submission = Submission.parse(value, request)
        except ValueError as error:
            raise InvalidSubmission(str(error)) from error
        path = self._submission_path(agent, request_id)
        if (existing := self._load(path, Submission.from_dict)) is not None:
            return self._winner(existing, submission)
        current = self.current(agent)
        if current is None or current.request_id != request_id:
            raise StaleRequest("the request is no longer pending")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        atomic(temporary, submission.as_dict())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = self._load(path, Submission.from_dict)
            if existing is None:
                raise InteractionConflict("a submission already exists")
            return self._winner(existing, submission)
        finally:
            temporary.unlink(missing_ok=True)
        return submission

    @staticmethod
    def _winner(existing: Submission, submission: Submission) -> Submission:
        """The submission that won, when `submission` is it sent again."""
        if not existing.same_as(submission):
            raise InteractionConflict("a different submission already won this request")
        return existing

    def wait(
        self,
        request: InteractionRequest,
        cancelled: Callable[[], bool],
        interval: float = 0.1,
    ) -> Submission:
        """The submission that answers `request`, which it then completes.

        The request is cancelled instead when `cancelled` returns True, which raises
        InteractionCancelled, or when its submission is there and does not read, which
        raises UnreadableRecord: once a submission is there nothing can replace it.
        """
        path = self._submission_path(request.agent, request.request_id)
        while True:
            if cancelled():
                self.cancel(request.agent, request.request_id)
                raise InteractionCancelled("interactive request cancelled")
            try:
                submission = self._load(path, Submission.from_dict)
            except UnreadableRecord:
                self.cancel(request.agent, request.request_id)
                raise
            if submission is not None:
                self.complete(request.agent, request.request_id)
                return submission
            time.sleep(interval)

    def load_draft(self, agent: str, request_id: str) -> list[dict[str, Any]]:
        value = self._read(
            self.root / "drafts" / self._agent(agent) / f"{request_id}.json"
        )
        return list(value.get("tool_calls", [])) if value else []

    def save_draft(
        self, agent: str, request_id: str, calls: list[dict[str, Any]]
    ) -> None:
        current = self.current(agent)
        if current is None or current.request_id != request_id:
            raise StaleRequest("the request is no longer pending")
        atomic(
            self.root / "drafts" / self._agent(agent) / f"{request_id}.json",
            {"version": VERSION, "request_id": request_id, "tool_calls": calls},
        )
