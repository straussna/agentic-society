"""The fake API: scripted responses, and a create() that plays them."""

from __future__ import annotations

from types import SimpleNamespace as NS
import threading
import harness


def usage(**kw):
    """A usage object shaped like the API's, with overridable token counts."""
    return NS(**{"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation": None, "iterations": None, **kw})


def attempt(model, output_tokens, kind="message", **kw):
    """One entry of usage.iterations: what a single model's attempt cost.

    The declining attempts of a chain are `message`; the last is
    `fallback_message`. An attempt with no output declined before producing any.
    """
    return NS(**{"type": kind, "model": model, "input_tokens": 100, "output_tokens": output_tokens,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation": None, **kw})


def say(text="done.", u=None, id=None, stop="end_turn", details=None, model=None):
    """A scripted step: reply with text and stop.

    `details` stands in for stop_details, sent only alongside a refusal.
    `model` overrides what the response reports, scripting a fallback-served turn.
    """
    return {"kind": "say", "text": text, "u": u, "id": id, "stop": stop, "details": details,
            "model": model}


def run(*cmds, u=None, id=None, stop="tool_use", details=None, model=None):
    """A scripted step: reply with one bash tool call per command.

    `stop` is the response's stop_reason, so a truncated turn can be scripted.
    """
    return {"kind": "agent", "cmds": list(cmds), "u": u, "id": id, "stop": stop,
            "details": details, "model": model}


def refuse(*cmds, category="cyber", u=None, id=None, **detail):
    """A scripted refusal, in either shape the API sends one.

    With commands it carries the tool calls emitted before the block landed;
    with none its content is empty. `detail` adds fields to stop_details.
    """
    return run(*cmds, u=u, id=id, stop="refusal",
               details=NS(type="refusal", category=category, explanation="declined", **detail))


def restart(u=None, id=None):
    """A scripted step: the {"restart": true} form of the bash tool."""
    return run(None, u=u, id=id)


def think(thinking="reasoning.", text="done.", u=None, id=None, stop="end_turn"):
    """A scripted step: a thinking block plus text, as fable-5 replies."""
    return {"kind": "think", "thinking": thinking, "text": text, "u": u, "id": id, "stop": stop}


class Err(Exception):
    """An API error carrying a status code."""

    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def fake(*steps, seen=None):
    """Build a `create` that plays the given steps, one per call.

    Steps run out into a plain "done." reply. `seen` captures the request params.
    """
    q, n = list(steps), [0]

    def create(**params):
        """Stands in for client.messages.create: plays one step per call.

        Refuses a request without the SDK's conversation keyword, as the SDK would.
        """
        assert "messages" in params, f"no messages= in the request: {sorted(params)}"
        n[0] += 1
        if seen is not None:
            seen.append(params)
        s = q.pop(0) if q else say()
        if isinstance(s, BaseException):
            raise s
        rid, u = s["id"] or f"msg{n[0]}", s["u"] or usage()
        # The real API answers with the dated snapshot the alias resolved to,
        # which is the fallback's name on a turn a fallback served.
        model = s.get("model") or f"{params['model']}-20990101"
        if s["kind"] == "agent":
            return NS(id=rid, model=model, stop_reason=s["stop"], usage=u,
                      stop_details=s.get("details"),
                      content=[NS(type="tool_use", id=f"t{n[0]}_{i}", name="bash", input={"command": c})
                               for i, c in enumerate(s["cmds"])])
        content = [NS(type="text", text=s["text"])]
        if s["kind"] == "think":
            content.insert(0, NS(type="thinking", thinking=s["thinking"]))
        return NS(id=rid, model=model, stop_reason=s["stop"], usage=u, content=content,
                  stop_details=s.get("details"))

    return create


def per_agent(default=(), **scripts):
    """A create that plays one fake() per episode, chosen by the thread's name.

    A simultaneous round names each episode's thread after its agent, so `scripts`
    keyed by agent id give each its own steps; anything else gets `default`.
    """
    fakes = {agent: fake(*steps) for agent, steps in scripts.items()}
    lock = threading.Lock()

    def create(**params):
        name = threading.current_thread().name
        with lock:
            # An episode not scripted by name gets its own copy of `default`,
            # so no two episodes ever draw from one queue.
            mine = fakes.get(name) or fakes.setdefault(name, fake(*default))
        return mine(**params)
    return create


def stopping_at(turn: int, *steps):
    """A create that plays `steps` and raises STOPPING as it serves turn `turn`.

    Stands in for a signal landing mid-call: the flag is read at the top of the
    next turn, so the episode finishes a turn before it can answer.
    """
    inner, n = fake(*steps), [0]

    def create(**params):
        n[0] += 1
        if n[0] == turn:
            harness.STOPPING = True
        return inner(**params)
    return create

DEFAULT = (run("cat n1"), run("echo hi > state/note.txt", "ls state"), say())
