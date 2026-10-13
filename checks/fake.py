"""Provider-neutral scripted sessions for harness checks."""

from __future__ import annotations

from types import SimpleNamespace as NS
import threading

import harness
import providers
from providers import Charge, NormalizedResponse, RawResponse, Refusal, ToolCall, Usage
from providers.base import classify_error


def usage(**kw):
    return NS(
        **{
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": None,
            "iterations": None,
            **kw,
        }
    )


def attempt(model, output_tokens, kind="message", **kw):
    return NS(
        **{
            "type": kind,
            "model": model,
            "input_tokens": 100,
            "output_tokens": output_tokens,
            **kw,
        }
    )


def say(text="done.", u=None, id=None, stop="end_turn", details=None, model=None):
    return {
        "step_type": "say",
        "text": text,
        "u": u,
        "id": id,
        "stop": stop,
        "details": details,
        "model": model,
    }


def run(*cmds, u=None, id=None, stop="tool_use", details=None, model=None):
    return {
        "step_type": "run",
        "cmds": list(cmds),
        "u": u,
        "id": id,
        "stop": stop,
        "details": details,
        "model": model,
    }


def use(name, u=None, id=None, stop="tool_use", **args):
    return {
        "step_type": "call",
        "name": name,
        "args": args,
        "u": u,
        "id": id,
        "stop": stop,
        "details": None,
        "model": None,
    }


def refuse(*cmds, category="cyber", u=None, id=None, **detail):
    return run(
        *cmds,
        u=u,
        id=id,
        stop="refusal",
        details=NS(type="refusal", category=category, explanation="declined", **detail),
    )


def restart(u=None, id=None):
    return run(None, u=u, id=id)


def think(thinking="reasoning.", text="done.", u=None, id=None, stop="end_turn"):
    return {
        "step_type": "think",
        "thinking": thinking,
        "text": text,
        "u": u,
        "id": id,
        "stop": stop,
    }


class FakeError(Exception):
    """What an SDK raises for an answer with an HTTP status."""

    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class Err(providers.ProviderError):
    """A failed model call with `status`, classified as an adapter classifies one."""

    def __init__(self, status):
        super().__init__(
            f"status {status}",
            category=classify_error(FakeError(status), "anthropic").category,
            provider="anthropic",
            status_code=status,
            native_type="FakeError",
        )


def _usage(raw, provider, model):
    raw = raw or usage()
    read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    uncached = int(getattr(raw, "input_tokens", 0) or 0)
    write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    output = int(getattr(raw, "output_tokens", 0) or 0)
    normalized = Usage(uncached + read + write, uncached, read, write, output, 0)
    spec = providers.model_spec(provider, model)
    write_class = "cache_write_5m" if provider == "anthropic" else "cache_write"
    charges = (
        Charge(
            "uncached_input",
            uncached,
            spec.rate("uncached_input"),
            uncached * spec.rate("uncached_input"),
        ),
        Charge(
            "cache_read", read, spec.rate("cache_read"), read * spec.rate("cache_read")
        ),
        Charge(
            write_class, write, spec.rate(write_class), write * spec.rate(write_class)
        ),
        Charge("output", output, spec.rate("output"), output * spec.rate("output")),
    )
    return normalized, charges


class FakeSession:
    def __init__(self, turns, provider, model, seen=None, on_send=None):
        self.turns = turns
        self.provider = provider
        self.requested_model = model
        self.seen = seen
        self.on_send = on_send
        self.n = 0

    def send(self, content):
        self.n += 1
        if self.on_send:
            self.on_send(self.n)
        if self.seen is not None:
            self.seen.append(
                {
                    "call": "send",
                    "provider": self.provider,
                    "model": self.requested_model,
                    "input": content,
                }
            )
        turn = self.turns.pop(0) if self.turns else say()
        if isinstance(turn, BaseException):
            raise turn
        rid = turn["id"] or f"msg{self.n}"
        resolved = turn.get("model") or f"{self.requested_model}-20990101"
        normalized_usage, charges = _usage(
            turn.get("u"), self.provider, self.requested_model
        )
        if turn["step_type"] == "run":
            calls = tuple(
                ToolCall(f"t{self.n}_{i}", "bash", {"command": command})
                for i, command in enumerate(turn["cmds"])
            )
            texts, reasoning = (), ()
        elif turn["step_type"] == "call":
            calls = (ToolCall(f"t{self.n}_0", turn["name"], dict(turn["args"])),)
            texts, reasoning = (), ()
        else:
            calls = ()
            texts = (turn["text"],)
            reasoning = (turn["thinking"],) if turn["step_type"] == "think" else ()
        detail = turn.get("details")
        refusal = (
            None
            if turn["stop"] != "refusal"
            else Refusal(
                getattr(detail, "type", "refusal"),
                getattr(detail, "explanation", None),
                getattr(detail, "recommended_model", None),
                getattr(detail, "__dict__", None),
            )
        )
        if turn["stop"] == "refusal" and not calls:
            charges = ()
        response = NormalizedResponse(
            rid,
            self.provider,
            self.requested_model,
            resolved,
            turn["stop"],
            turn["stop"],
            texts,
            reasoning,
            calls,
            normalized_usage,
            charges,
            refusal,
            getattr(detail, "__dict__", None),
        )
        native = {
            "id": rid,
            "model": resolved,
            "stop_reason": turn["stop"],
            "content": [call.as_dict() for call in calls],
            "usage": normalized_usage.as_dict(),
        }
        return RawResponse(self.provider, native, lambda: response)


class FakeRouter:
    def __init__(
        self,
        turns=(),
        seen=None,
        scripts=None,
        default=(),
        on_send=None,
        on_open=None,
    ):
        self.turns = list(turns)
        self.seen = seen
        self.scripts = {name: list(values) for name, values in (scripts or {}).items()}
        self.default = list(default)
        self.on_send = on_send
        self.on_open = on_open

    def preflight(self):
        return None

    def open_session(self, provider, model, system, tools, max_tokens, info):
        if self.on_open:
            self.on_open()
        if self.seen is not None:
            self.seen.append(
                {
                    "call": "session",
                    "provider": provider,
                    "model": model,
                    "system": system,
                    "tools": [tool.as_dict() for tool in tools],
                    "max_tokens": max_tokens,
                    "info": {
                        "agent": info.agent,
                        "label": info.label,
                        "episode": info.episode,
                        "interaction_root": info.interaction_root.name,
                        "cancellable": info.cancelled is not None,
                    },
                }
            )
        name = threading.current_thread().name
        turns = self.scripts.get(name)
        if turns is None:
            turns = self.turns if self.turns else list(self.default)
            if self.scripts:
                self.scripts[name] = turns
        return FakeSession(turns, provider, model, self.seen, self.on_send)


def fake(*turns, seen=None):
    return FakeRouter(turns, seen)


def per_agent(default=(), on_send=None, on_open=None, **scripts):
    return FakeRouter(
        scripts=scripts, default=default, on_send=on_send, on_open=on_open
    )


def interrupting_at(at: int, *turns):
    def interrupt(n):
        if n == at:
            harness.INTERRUPTING = True

    return FakeRouter(turns, on_send=interrupt)


DEFAULT = (run("cat n1"), run("echo hi > memory/note.txt", "ls memory"), say())
