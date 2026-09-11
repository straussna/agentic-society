"""Interactive provider backed by the durable interaction store."""

from __future__ import annotations

from collections.abc import Iterable

from interaction.store import InteractionCancelled, InteractionStore

from .base import (ModelSpec, NormalizedTurn, PendingResponse, SessionContext, ToolResult,
                   ToolSpec, Usage)


MODELS = {
    "interactive": ModelSpec("human", "interactive", 1_000_000,
                             (("uncached_input", 0), ("cache_read", 0),
                              ("cache_write", 0), ("output", 0)))
}


class HumanSession:
    provider = "human"

    def __init__(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                 context: SessionContext):
        self.requested_model = model
        self.system = system
        self.tools = tools
        self.context = context
        self.store = InteractionStore(context.interaction_root)
        self.turn = 0

    def request(self, content: str | tuple[ToolResult, ...]) -> PendingResponse:
        self.turn += 1
        request = self.store.publish(self.context.agent, self.context.label,
                                     self.context.episode, self.turn, self.system,
                                     content, self.tools)
        try:
            submission = self.store.wait(request, self.context.is_cancelled)
        except InteractionCancelled as error:
            raise KeyboardInterrupt from error
        calls = submission.tool_calls
        stop = "tool_use" if submission.action == "tool_calls" else "end_turn"
        native = {"id": submission.submission_id, "provider": self.provider,
                  "model": self.requested_model, "stop_reason": stop,
                  "request_id": request.request_id,
                  "content": [call.as_dict() for call in calls],
                  "usage": Usage.zero().as_dict()}
        turn = NormalizedTurn(submission.submission_id, self.provider, self.requested_model,
                              self.requested_model, stop, stop, (), (), calls,
                              Usage.zero(), ())
        return PendingResponse(self.provider, native, lambda: turn)


class HumanProvider:
    name = "human"
    models = MODELS

    def preflight(self, models: Iterable[str]) -> None:
        for model in models:
            if model not in self.models:
                raise KeyError(model)

    def open_session(self, model: str, system: str, tools: tuple[ToolSpec, ...],
                     max_tokens: int, context: SessionContext) -> HumanSession:
        return HumanSession(model, system, tools, context)

    def provenance(self, model: str) -> dict:
        return {"name": self.name, "adapter": "interaction-store-v1", "endpoint": "local",
                "model_spec": self.models[model].as_dict()}
