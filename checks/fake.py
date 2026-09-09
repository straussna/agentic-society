"""Provider-neutral scripted sessions for harness checks."""

from __future__ import annotations

from types import SimpleNamespace as NS
import threading

import harness
import providers
from providers import Charge, NormalizedTurn, PendingResponse, Refusal, ToolCall, Usage


def usage(**kw):
    return NS(**{"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation": None, "iterations": None, **kw})


def attempt(model, output_tokens, kind="message", **kw):
    return NS(**{"type": kind, "model": model, "input_tokens": 100,
                 "output_tokens": output_tokens, **kw})


def say(text="done.", u=None, id=None, stop="end_turn", details=None, model=None):
    return {"kind": "say", "text": text, "u": u, "id": id, "stop": stop,
            "details": details, "model": model}


def run(*cmds, u=None, id=None, stop="tool_use", details=None, model=None):
    return {"kind": "agent", "cmds": list(cmds), "u": u, "id": id, "stop": stop,
            "details": details, "model": model}


def use(name, u=None, id=None, stop="tool_use", **args):
    return {"kind": "call", "name": name, "args": args, "u": u, "id": id,
            "stop": stop, "details": None, "model": None}


def refuse(*cmds, category="cyber", u=None, id=None, **detail):
    return run(*cmds, u=u, id=id, stop="refusal",
               details=NS(type="refusal", category=category, explanation="declined", **detail))


def restart(u=None, id=None):
    return run(None, u=u, id=id)


def think(thinking="reasoning.", text="done.", u=None, id=None, stop="end_turn"):
    return {"kind": "think", "thinking": thinking, "text": text, "u": u,
            "id": id, "stop": stop}


class Err(providers.ProviderError):
    def __init__(self, status):
        retryable = status in (408, 409, 429) or status >= 500
        super().__init__(f"status {status}", category="retryable_api" if retryable else "permanent_api",
                         provider="anthropic", status_code=status, native_type="FakeError")


def _usage(raw, provider, model):
    raw = raw or usage()
    read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    uncached = int(getattr(raw, "input_tokens", 0) or 0)
    write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    output = int(getattr(raw, "output_tokens", 0) or 0)
    normalized = Usage(uncached + read + write, uncached, read, write, output, 0)
    spec = providers.model_spec(provider, model)
    write_kind = "cache_write_5m" if provider == "anthropic" else "cache_write"
    charges = (
        Charge("uncached_input", uncached, spec.rate("uncached_input"), uncached * spec.rate("uncached_input")),
        Charge("cache_read", read, spec.rate("cache_read"), read * spec.rate("cache_read")),
        Charge(write_kind, write, spec.rate(write_kind), write * spec.rate(write_kind)),
        Charge("output", output, spec.rate("output"), output * spec.rate("output")),
    )
    return normalized, charges


class FakeSession:
    def __init__(self, steps, provider, model, seen=None, on_request=None):
        self.steps = steps
        self.provider = provider
        self.requested_model = model
        self.seen = seen
        self.on_request = on_request
        self.n = 0

    def request(self, content):
        self.n += 1
        if self.on_request:
            self.on_request(self.n)
        if self.seen is not None:
            self.seen.append({"kind": "request", "provider": self.provider,
                              "model": self.requested_model, "input": content})
        step = self.steps.pop(0) if self.steps else say()
        if isinstance(step, BaseException):
            raise step
        rid = step["id"] or f"msg{self.n}"
        resolved = step.get("model") or f"{self.requested_model}-20990101"
        normalized_usage, charges = _usage(step.get("u"), self.provider, self.requested_model)
        if step["kind"] == "agent":
            calls = tuple(ToolCall(f"t{self.n}_{i}", "bash", {"command": command})
                          for i, command in enumerate(step["cmds"]))
            texts, reasoning = (), ()
        elif step["kind"] == "call":
            calls = (ToolCall(f"t{self.n}_0", step["name"], dict(step["args"])),)
            texts, reasoning = (), ()
        else:
            calls = ()
            texts = (step["text"],)
            reasoning = (step["thinking"],) if step["kind"] == "think" else ()
        detail = step.get("details")
        refusal = None if step["stop"] != "refusal" else Refusal(
            getattr(detail, "type", "refusal"), getattr(detail, "explanation", None),
            getattr(detail, "recommended_model", None), getattr(detail, "__dict__", None))
        if step["stop"] == "refusal" and not calls:
            charges = ()
        turn = NormalizedTurn(rid, self.provider, self.requested_model, resolved, step["stop"],
                              step["stop"], texts, reasoning, calls, normalized_usage,
                              charges, refusal, getattr(detail, "__dict__", None))
        native = {"id": rid, "model": resolved, "stop_reason": step["stop"],
                  "content": [call.as_dict() for call in calls], "usage": normalized_usage.as_dict()}
        return PendingResponse(self.provider, native, lambda: turn)


class FakeRouter:
    def __init__(self, steps=(), seen=None, scripts=None, default=(), on_request=None,
                 on_open=None):
        self.steps = list(steps)
        self.seen = seen
        self.scripts = {name: list(values) for name, values in (scripts or {}).items()}
        self.default = list(default)
        self.on_request = on_request
        self.on_open = on_open

    def preflight(self):
        return None

    def open_session(self, provider, model, system, tools, max_tokens):
        if self.on_open:
            self.on_open()
        if self.seen is not None:
            self.seen.append({"kind": "session", "provider": provider, "model": model,
                              "system": system, "tools": [tool.as_dict() for tool in tools],
                              "max_tokens": max_tokens})
        name = threading.current_thread().name
        steps = self.scripts.get(name)
        if steps is None:
            steps = self.steps if self.steps else list(self.default)
            if self.scripts:
                self.scripts[name] = steps
        return FakeSession(steps, provider, model, self.seen, self.on_request)

    def provenance(self, provider, model):
        return providers.provenance(provider, model)


def fake(*steps, seen=None):
    return FakeRouter(steps, seen)


def per_agent(default=(), on_request=None, on_open=None, **scripts):
    return FakeRouter(scripts=scripts, default=default, on_request=on_request,
                      on_open=on_open)


def stopping_at(turn: int, *steps):
    def stop(n):
        if n == turn:
            harness.STOPPING = True
    return FakeRouter(steps, on_request=stop)


DEFAULT = (run("cat n1"), run("echo hi > state/note.txt", "ls state"), say())
