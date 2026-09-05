"""Verification. No API spend.

    py -3 check.py [names...] [--real | --no-docker] [--list] [-j N]

A fake `create` goes into harness.run_once, so the pipeline agents unbilled."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import difflib
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from types import SimpleNamespace as NS

import analyze
import experiment
import view
import harness

# The API errors here are scripted, so the wait between retries is time spent
# proving nothing. What is retried and how often still is.
harness.RETRY_BASE = 0

# --- the fake ---------------------------------------------------------------


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
        """Stands in for client.messages.create: plays one step per call."""
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


class Skip(Exception):
    """This check needs something this machine cannot give it."""


# Set by --real: every check takes a container, including the ones that would
# otherwise run on the host box. What proves the two lanes still agree.
REAL_ONLY = False


# The answer to docker_ready(), once some process has paid for it. Carried into
# workers rather than asked again in each of them.
_DOCKER: bool | None = None


# The pid of the process running this suite, carried into every worker so that
# each one's containers say which suite they belong to. What lets the sweep find
# its own and nothing else - another suite's live containers, or a real agent's,
# are not this one's to remove.
SUITE = os.getpid()


def docker_ready() -> bool:
    """True if the daemon is up and the image is built. Asked once per process."""
    global _DOCKER
    if _DOCKER is None:
        _DOCKER = _ask_docker()
    return _DOCKER


def _ask_docker() -> bool:
    """Put the question to the daemon. Says so once if the image is not built."""
    if not shutil.which("docker") or subprocess.run(["docker", "info"], capture_output=True).returncode:
        return False
    if subprocess.run(["docker", "image", "inspect", harness.IMAGE], capture_output=True).returncode:
        print(f"image {harness.IMAGE} not built\n")
        return False
    return True


# --- the host box -----------------------------------------------------------
#
# Arithmetic checks - what a turn cost, what reached the series, which stop a
# episode ended on - agent against a directory and a bash process on this machine.
# What only a container can show stays on docker_root below.


def host_bash() -> str | None:
    """The bash to run the host box's episodes in, as an absolute path.

    Resolved rather than left to PATH: on Windows `bash` and subprocess find
    Git's and WSL's, which disagree about what a path is and what /tmp means.
    """
    return shutil.which("bash")


class HostShell(harness.Shell):
    """The episode shell, as a bash process on this machine.

    Inherits the sentinel framing, timeout, and output ceiling from harness.Shell;
    only where the process agents and how a balance is rewritten differ.
    """

    def __init__(self, box: "HostBox") -> None:
        self.box = box
        super().__init__(box.name)

    def argv(self) -> list[str]:
        # No profile: what the agent's shell is must not depend on this account.
        return [host_bash(), "--norc", "--noprofile"]

    def popen_kwargs(self) -> dict:
        # DETACHED from the base class, so the two lanes agree about which
        # processes a signal aimed at the harness reaches.
        return {**super().popen_kwargs(),
                "cwd": str(self.box.work),
                # Stop MSYS rewriting paths inside the agent's own commands.
                "env": {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}}

    def republish_balance(self, index: str, series: list[int], expected: str) -> str:
        n = self.box.work / harness.balance_name(index)
        was = n.read_text(encoding="utf-8") if n.exists() else ""
        n.write_text(harness.render_balance(series), encoding="utf-8", newline="\n")
        return "ok" if was == expected else "tampered"


class HostBox:
    """An episode's environment as a directory on this machine, in place of a container.

    Same five methods run_once asks of harness.Container, same six channels. No
    ownership and no locking: a check turning on either belongs on docker_root.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = tempfile.mkdtemp(prefix="mtr-host-")
        self.work = Path(self.dir)
        self.index = "1"
        (self.work / "state").mkdir()

    @classmethod
    def start(cls, name: str) -> "HostBox":
        return cls(name)

    def load(self, channels: list[tuple[str, Path, str]], files: dict[str, str]) -> None:
        for name, src, channel in channels:
            dest = self.work / name
            if channel in harness.FILE_CHANNELS:
                # A sender that has not addressed this agent, and a sender that
                # aimed something other than one file at it, arrive the same way:
                # as nothing.
                dest.parent.mkdir(exist_ok=True, parents=True)
                if src.is_file():
                    shutil.copyfile(src, dest)
                continue
            if channel == "blackboard":
                self.index = name
            dest.mkdir(exist_ok=True, parents=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
        for name, text in files.items():
            (self.work / name).write_text(text, encoding="utf-8", newline="\n")

    def shell(self) -> HostShell:
        return HostShell(self)

    def save(self, channels: list[tuple[str, Path, str]]) -> bool:
        kept = True
        for name, mirror, channel in channels:
            if channel in harness.WRITABLE:
                kept = harness.save_state(
                    mirror, self._fetcher(self.work / name), lambda: None) and kept
        return kept

    def _fetcher(self, src: Path) -> Callable[[Path], bool]:
        def fetch(dest: Path) -> bool:
            if not src.is_dir():
                return False
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return True
        return fetch

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class RecordingBox(HostBox):
    """A HostBox that writes down every start, load and close, for a check on ordering."""
    events: list[tuple[str, str]] = []
    lock = threading.Lock()

    @classmethod
    def note(cls, what: str, name: str) -> None:
        with cls.lock:
            cls.events.append((what, name))

    @classmethod
    def start(cls, name: str) -> "RecordingBox":
        cls.note("start", name)
        return cls(name)

    def load(self, channels: list[tuple[str, Path, str]], files: dict[str, str]) -> None:
        super().load(channels, files)
        self.note("load", self.name)

    def close(self) -> None:
        self.note("close", self.name)
        super().close()


def agent_of(container: str) -> str:
    """The agent a container name belongs to: the prefix and the episode index removed."""
    return container[len(harness.CONTAINER_PREFIX):].rsplit("-", 1)[0]


def per_run(default=(), **scripts):
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


def manifest_file(root: Path, text: str, name: str = "c.toml") -> Path:
    """Write an experiment manifest under a temporary ROOT, for the manifest checks to use."""
    p = root / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


# Every harness global a check is allowed to move, and therefore every one pinned()
# puts back. temp_root refuses any name outside this set.
RESTORED = harness.TUNABLES | {"ROOT", "WATCH", "REFUSAL_TURNS", "BOX", "drive", "ready", "start",
                            # A check that left this set would end every later
                            # episode in the same worker at turn one.
                            "STOPPING", "catch_signals"}


@contextlib.contextmanager
def pinned():
    """Restore every episode global a check may move, on exit.

    catch_signals is stubbed rather than restored: a handler installed by a
    check driving harness.main would outlive it and answer the suite's own Ctrl+C.
    """
    saved = {k: getattr(harness, k) for k in RESTORED}
    harness.catch_signals = lambda: None
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(harness, k, v)


@contextlib.contextmanager
def rooted(box, **overrides):
    """Point harness at a throwaway directory, with episodes running in `box`.

    `overrides` set harness module globals (MAX_TURNS=1, COMMAND_TIMEOUT=2) for the
    duration, and pinned() puts every one of them back.
    """
    unknown = set(overrides) - RESTORED
    assert not unknown, f"a root cannot restore {sorted(unknown)}"
    with pinned(), tempfile.TemporaryDirectory(
            prefix="mtr-check-", ignore_cleanup_errors=True) as d:
        harness.ROOT = Path(d)
        harness.BOX = box
        for k, v in overrides.items():
            setattr(harness, k, v)
        yield Path(d)


@contextlib.contextmanager
def host_root(**overrides):
    """A throwaway agent pinned to this machine, whatever --real says.

    For the few checks about what is *sent* rather than the episode it drives.
    Everything else wants temp_root, which --real does promote.
    """
    if not host_bash():
        raise Skip
    with rooted(HostBox, **overrides) as d:
        yield d


@contextlib.contextmanager
def temp_root(**overrides):
    """A throwaway agent whose episodes are a directory and a shell on this machine.

    What most checks want: the pipeline end to end - account, turns, series,
    trace - without paying for a container that proves nothing they assert.
    """
    if REAL_ONLY:
        with docker_root(**overrides) as d:
            yield d
        return
    with host_root(**overrides) as d:
        yield d


@contextlib.contextmanager
def docker_root(**overrides):
    """A throwaway agent whose episodes are real containers.

    For the checks that turn on something only a container has. Skips when
    Docker is unavailable, which is the one reason a check here cannot run.
    """
    if not docker_ready():
        raise Skip
    with rooted(harness.Container, **overrides) as d:
        yield d


@contextlib.contextmanager
def quiet():
    """Swallow episode output so the check list stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def episode_once(*steps, seen=None):
    """One episode against a fresh agent, output suppressed."""
    with quiet():
        return harness.run_once("t", fake(*steps, seen=seen))


def ground_truth(agent="t") -> dict:
    """account.json as it stands on disk, read back rather than carried."""
    return json.loads((harness.records_dir(agent) / "account.json").read_text(encoding="utf-8"))


# --- checks -----------------------------------------------------------------

DEFAULT = (run("cat n1"), run("echo hi > state/note.txt", "ls state"), say())


def check_system_is_pinned():
    """Invariant 2: the prompt is the pinned 75 bytes, and the tool carries no description."""
    assert hashlib.sha256(harness.SYSTEM.encode()).hexdigest() == harness.SYSTEM_SHA256
    assert len(harness.SYSTEM) == 75
    assert harness.SYSTEM.count("\n") == 2, "two lines: what persists, and what is available"
    for word in ("you ", "your", "assistant", "budget", "cost", "goal", "task", "network"):
        assert word not in harness.SYSTEM.lower(), f"SYSTEM contains {word!r}"
    assert harness.TOOL == {"type": "bash_20250124", "name": "bash"}, "tool must carry no description"


def check_cost_is_exact():
    """Invariant 6: cost matches hand-computed integers, including both cache-write TTLs."""
    # opus-5: in 500, out 2500 centi/token; write 1.25x, read 0.1x
    u = usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=2000,
              cache_read_input_tokens=5000)
    m = harness.measure(u, "claude-opus-5")
    assert m["centi"] == 1000 * 500 + 100 * 2500 + 2000 * 625 + 5000 * 50 == 2_250_000, m
    assert m["prefix"] == 1000 + 5000 + 2000, "prefix must include cached tokens"
    # per-TTL detail wins over the flat field; 1h writes cost 2x
    split = harness.measure(usage(input_tokens=0, output_tokens=0, cache_creation_input_tokens=9999,
                               cache_creation=NS(ephemeral_5m_input_tokens=100,
                                                 ephemeral_1h_input_tokens=200)), "claude-opus-5")
    assert split["centi"] == 100 * 625 + 200 * 1000, split


def check_balance_is_bare_integers_with_no_host_in_them():
    """Unlabelled: n is a JSON array of bare integers, byte-for-byte, LF on any host."""
    assert harness.render_balance([500000, 494750]) == "[500000,494750]\n"
    assert all(type(v) is int for v in json.loads(harness.render_balance([1, 2])))
    assert ":" not in harness.render_balance([1]) and '"' not in harness.render_balance([1])
    rendered = harness.render_balance([1_000_000, 996_989]).encode("utf-8")
    assert rendered == b"[1000000,996989]\n", rendered
    assert b"\r" not in rendered, "no carriage return reaches the agent"
    assert harness.balance_name("3") == "n3", "and a seat is what names one"


def check_config_is_validated():
    """The real config.toml is valid, and bad keys, types, and values are refused.

    Unknown keys, wrong types, out-of-range values, and a --config path that
    does not exist all exit nonzero.
    """
    with pinned():
        try:
            harness.load_config()                              # the real file, if present
        except SystemExit as e:                             # reported as a failure
            raise AssertionError(f"config.toml is invalid: {e}") from None
        assert harness.MODEL in harness.PRICES
        assert 0 < harness.CONTEXT_FRACTION <= 1
        assert harness.MAX_TOKENS <= harness.MAX_TOKENS_CEILING
        assert type(harness.LIVE_BALANCE) is bool
        assert harness.DIGEST_FILE_LIMIT >= harness.DIGEST_FILE_FLOOR
        assert harness.OBSERVATION_LIMIT >= harness.TOOL_RESULT_LIMIT

    with tempfile.TemporaryDirectory(prefix="mtr-cfg-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in ('turn_cpa = 5', 'system = "hi"', 'max_turns = "many"',
                    'model = "no-such-model"', 'context_fraction = 2.0', 'budget = 0',
                    'rebate_percent = 101', 'live_balance = "yes"',
                    # A transfer is the giver's own budget moving; a rebate on top
                    # would mint. No share for a transfer nobody can make.
                    'transfer_funded_by = "giver"\nrebate_percent = 75', 'transfer_funded_by = "loud"',
                    'transfer_funded_by = 3', 'transfer_funded_by = "none"\ntransfer_silence_penalty_percent = 50',
                    'delivery = "fetch"', 'delivery = 1',
                    # Above the ceiling the harness would time out mid-episode.
                    f'max_tokens = {harness.MAX_TOKENS_CEILING + 1}',
                    # Below this a clipped message says less than its own marker.
                    f'digest_file_limit = {harness.DIGEST_FILE_FLOOR - 1}',
                    # The initial observation carries the whole record and is never smaller
                    # than what one ordinary call may return.
                    f'observation_limit = {harness.TOOL_RESULT_LIMIT - 1}'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                try:
                    harness.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad config: {bad}")

        # A named file that is not there is refused, not quietly skipped.
        with pinned():
            try:
                harness.load_config(Path(tmp) / "confg.toml")
            except SystemExit:
                pass
            else:
                raise AssertionError("a missing --config path was ignored")

        f.write_text("max_turns = 7\ncontext_fraction = 1\n", encoding="utf-8")
        with pinned():
            assert harness.load_config(f) == f, "the file used is reported back"
            assert harness.MAX_TURNS == 7, "a good value must actually apply"
            assert harness.CONTEXT_FRACTION == 1.0, "an int must widen into a float field"
        f.write_text('transfer_funded_by = "giver"\nrebate_percent = 0\n', encoding="utf-8")
        with pinned():
            harness.load_config(f)
            assert harness.TRANSFER_FUNDED_BY == "giver", "the pairing the rule asks for is accepted"
        with pinned():
            try:
                harness.apply_config({"rebate_percent": 101}, "manifest")
            except SystemExit as e:
                assert str(e).startswith("manifest:"), e
            else:
                raise AssertionError("values applied by name are not checked")


def check_lapsed_rates_are_refused():
    """A model whose rates are known to have expired cannot start an agent.

    Both sides of the expiry date are checked.
    """
    assert harness.lapsed_prices("claude-sonnet-5", "2026-08-31") is None, "the last valid day agents"
    assert harness.lapsed_prices("claude-sonnet-5", "2026-09-01"), "the day after must refuse"
    assert harness.lapsed_prices("claude-opus-5", "2099-01-01") is None, \
        "a model with no known expiry never lapses"
    assert set(harness.PRICES_EXPIRE) <= set(harness.PRICES), "an expiry for an unpriced model is dead"
    assert harness.lapsed_prices(harness.MODEL) is None, \
        f"{harness.MODEL}: the rates in PRICES have lapsed as of today; update them"

    # And start() refuses before it can create an agent or reach the client. It
    # exits rather than returning a code, so every driver refuses identically
    # instead of each remembering to; the code and the message are unchanged.
    with pinned():
        harness.load_config()                           # whichever model is configured
        was, harness.PRICES_EXPIRE = harness.PRICES_EXPIRE, {
            **harness.PRICES_EXPIRE, harness.MODEL: ("2000-01-01", "something newer")}
        try:
            with quiet() as buf:
                harness.main(["--agent", "t"])
        except SystemExit as e:
            assert e.code == 2, e.code
        else:
            raise AssertionError("a lapsed rate started an agent")
        finally:
            harness.PRICES_EXPIRE = was
    assert "expired 2000-01-01" in buf.getvalue(), buf.getvalue()


def check_unpriced_fallback_targets_are_refused_before_anything_starts():
    """A permitted fallback target with no rates stops the agent; anything else does not.

    The list is a likely superset of what default routing will pick: worth
    pricing against before an agent starts, and not the whole guard.
    """
    def client(targets=None, raises=None):
        def retrieve(model, betas=None):
            assert betas == [harness.FALLBACK_BETA], betas
            if raises:
                raise raises
            return NS(allowed_fallback_models=targets)
        return NS(beta=NS(models=NS(retrieve=retrieve)))

    priced = sorted(harness.PRICES)[:2]
    assert harness.unpriced_targets(client(priced), harness.MODEL) == [], "all priced: nothing to say"
    assert harness.unpriced_targets(client([]), harness.MODEL) == [], "no targets: nothing to say"

    said = harness.unpriced_targets(client([*priced, "claude-unheard-of-9"]), harness.MODEL)
    assert len(said) == 1 and "claude-unheard-of-9" in said[0], said

    # A field the API does not send, and a call that fails outright: both leave
    # the agent to start, because measure_response() is what holds either way.
    with quiet():
        assert harness.unpriced_targets(client(None), harness.MODEL) == [], "absent field is not a refusal"
        assert harness.unpriced_targets(client(raises=Err(500)), harness.MODEL) == [], \
            "an unreadable list is not a refusal"
        # The one read failure that is a refusal is a client with no usable
        # credentials, which check_a_client_that_cannot_authenticate_refuses_to_start
        # is about. Every other status leaves the agent to start.
        assert harness.unpriced_targets(client(raises=Err(503)), harness.MODEL) == [], \
            "a server error is not a credential failure"


def check_a_client_that_cannot_authenticate_refuses_to_start():
    """No usable credentials stops the agent at start(), before any container.

    Constructing the client resolves no credentials, so unpriced_targets' read
    answers it. Three shapes: auth errors, 401/403, and a bare TypeError.
    """
    keyless = TypeError(
        '"Could not resolve authentication method. Expected one of api_key, '
        'auth_token, or credentials to be set. Or for one of the `X-Api-Key` or '
        '`Authorization` headers to be explicitly omitted"')
    for e in (keyless, Err(401), Err(403)):
        assert harness.unauthenticated(e), e
    for e in (Err(500), Err(404), OSError("connection reset"),
              TypeError("retrieve() got an unexpected keyword argument 'betas'")):
        assert not harness.unauthenticated(e), e

    def client(raises):
        def retrieve(model, betas=None):
            raise raises
        return NS(beta=NS(models=NS(retrieve=retrieve)))

    # start() refuses on every line unpriced_targets returns, and this is one:
    # a credential failure is a refusal where a server error is a warning.
    said = harness.unpriced_targets(client(keyless), harness.MODEL)
    assert len(said) == 1 and "ANTHROPIC_API_KEY" in said[0], said
    with quiet():
        assert harness.unpriced_targets(client(Err(500)), harness.MODEL) == []


def check_truncation_and_empty():
    """Oversized output is clipped with an explicit marker, keeping head and tail."""
    c = harness.clip("x" * 50_000, 8_000)
    assert "[truncated: 42000 of 50000 characters]" in c
    assert c.startswith("x") and c.endswith("x") and len(c) < 8_500, "head and tail both kept"
    assert harness.clip("short", 8_000) == "short"


def check_episodes_reconcile():
    """Every micro-dollar in or out of an account is one element of its series.

    A transfer rebates, a missed post and a crowded seat are each taken, and a floor
    gives back. The identity has a term for each, and each appends to the series.
    """
    with temp_root():
        for _ in range(3):
            last = episode_once(*DEFAULT)
        account = ground_truth()
        series, episodes = account["series"], account["episodes"]
        assert len(episodes) == 3, episodes
        assert len(series) == 1 + sum(s["turns"] for s in episodes), \
            "one element per billed turn, plus starter_files, where nothing else moved"
        spent = sum(s["spent"] for s in episodes)
        assert spent == account["initial"] - account["remaining"], \
            f"{spent} != {account['initial'] - account['remaining']}"
        assert series[-1] == account["remaining"], "the last element is the balance"
        for s in episodes:
            started, ended = series[s["series_from"]], series[s["series_to"]]
            assert started - ended == s["spent"], \
                f"episode {s['index']}: {started} - {ended} != {s['spent']}"
            assert started == s["balance_at_start"], (s["balance_at_start"], started)
        assert [s["series_from"] for s in episodes[1:]] == \
            [s["series_to"] for s in episodes[:-1]], "and the spans meet end to end"
        assert series == last["series_after"], \
            "the last trace carries the series the account committed"

    # And with every term live at once, the identity still closes.
    with temp_root(REBATE_PERCENT=50, BLACKBOARD_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50, FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        episode_once(run("echo '2 300' > out/transfer"), say())          # a transfer, and no post
        episode_once(run("rm out/transfer && mkdir -p out/2 && echo hi > out/2/a",  # a crowded seat
                      "echo posted > 1/RESULT"), say())
        account = ground_truth()
        series, episodes = account["series"], account["episodes"]
        spent = sum(s["spent"] for s in episodes)
        assert account["remaining"] == reconciled(account, spent), account
        assert series[-1] == account["remaining"], "the series ends where the account does"
        assert account["rebated"] == 150 and account["blackboard_penalised"] > 0, account
        assert account["mailbox_penalised"] > 0, account
        # The first episode wrote the line and gave; the second took it away,
        # which moves nothing and is not a transfer it made.
        assert account["transfer_penalised"] > 0, account
        assert [s["posted"] for s in episodes] == [False, True], episodes
        for s in episodes:
            assert len(span_of(series, s)) == elements_of(s), (s, span_of(series, s))

    # And under transfer, where the transfer is a debit and not a rebate.
    with temp_root(TRANSFER_FUNDED_BY="giver", REBATE_PERCENT=0, TRANSFER_SILENCE_PENALTY_PERCENT=50,
                   FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        episode_once(run("echo '2 300' > out/transfer"), say())
        episode_once(say())                                          # the line stands: no transfer of its own
        account = ground_truth()
        series, episodes = account["series"], account["episodes"]
        spent = sum(s["spent"] for s in episodes)
        assert account["remaining"] == reconciled(account, spent), account
        # The line stood through the second episode and gave again: a standing
        # declaration drains a transferring giver every episode it stands.
        assert account["debited"] == 600 and account["rebated"] == 0, account
        assert account["transfer_penalised"] > 0, account
        assert series[-1] == account["remaining"]
        for s in episodes:
            assert len(span_of(series, s)) == elements_of(s), (s, span_of(series, s))


def reconciled(account: dict, spent: int) -> int:
    """What an account must hold: every term of the identity, each one series element."""
    return (account["initial"] - spent + account.get("rebated", 0) + account.get("received", 0)
            - account.get("debited", 0) - account.get("blackboard_penalised", 0)
            - account.get("mailbox_penalised", 0) - account.get("transfer_penalised", 0)
            + account.get("forgiven", 0))


def span_of(series: list[int], s: dict) -> list[int]:
    """The elements of the series one episode's record spans, harness included."""
    return series[s["series_from"]:s["series_to"] + 1]


def elements_of(s: dict) -> int:
    """How many series elements an episode's record says it appended, plus the entry at its start."""
    return (s["turns"] + 1 + bool(s["transfer"]["rebate"] or s["transfer"].get("debit"))
            + bool(s["transfer"]["penalty"]) + bool(s["blackboard_penalised"])
            + bool(s["mailbox"]["penalty"]) + bool(s["forgiven"])
            # A credit from a peer settling in the same simultaneous round lands
            # inside the receiver's span; in rotation it lands between spans.
            + bool(s.get("received")))


def check_a_transfer_declaration_stands_until_it_is_withdrawn():
    """out/transfer is a standing pledge: it is honoured at the end of every episode.

    The outbox is a tree the harness never reaches into, so a line left in place
    is still being said. Giving once means taking it back afterwards.
    """
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={})
        first = episode_once(run("echo '2 120' > out/transfer"), say())
        again = episode_once(run("true"), say())
        withdrawn = episode_once(run("rm out/transfer"), say())
        taker = ground_truth("other")
    assert first["transfer"]["amount"] == again["transfer"]["amount"] == 120, (first, again)
    assert withdrawn["transfer"]["amount"] == 0 and withdrawn["transfer"]["error"] is None
    assert taker["received"] == 240, "twice given, twice received, and then not"


def check_exactly_one_transfer_an_episode_is_enforced():
    """No more than one was the grammar's already; no less than one is the share.

    What discharges the obligation is money moved from a declaration this
    episode wrote. A line left standing gives again and is not this episode's.
    """
    with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50, REBATE_PERCENT=100) as root:
        seated(root, other={})
        gave = episode_once(run("echo '2 100' > out/transfer"), say())
        stood = episode_once(run("true"), say())
        raised = episode_once(run("echo '2 200' > out/transfer"), say())
        withdrawn = episode_once(run("rm out/transfer"), say())
        taker = ground_truth("other")
    assert [t["transfer"]["amount"] for t in (gave, stood, raised, withdrawn)] == \
        [100, 100, 200, 0], [t["transfer"]["amount"] for t in (gave, stood, raised, withdrawn)]
    assert gave["transfer"]["penalty"] == 0, "money moved from a line it wrote"
    assert stood["transfer"]["penalty"] > 0, "the pledge still paid, and it chose nothing"
    assert raised["transfer"]["penalty"] == 0, "a changed amount is a transfer of its own"
    assert withdrawn["transfer"]["penalty"] > 0, "a withdrawal moves nothing and gives nothing"
    assert taker["received"] == 400, "the standing pledge paid every episode it stood"

    # The same bytes again is not a change, the way reposting the same bytes is
    # not a post. An episode that writes back the line already standing chose
    # nothing this time, and the pledge pays what it would have paid anyway.
    with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50, REBATE_PERCENT=100) as root:
        seated(root, other={})
        wrote = episode_once(run("echo '2 100' > out/transfer"), say())
        again = episode_once(run("echo '2 100' > out/transfer"), say())
        taker = ground_truth("other")
    assert wrote["transfer"]["penalty"] == 0, "the line was not there at its start"
    assert again["transfer"]["amount"] == 100, "the pledge still paid"
    assert again["transfer"]["penalty"] > 0, "the same bytes again is not a transfer it made"
    assert taker["received"] == 200, "twice given, whoever decided it"

    # A declaration the episode wrote that moves nothing is not a transfer. Each of
    # these changes out/transfer and none of them gives, so each is charged.
    for name, cmd in (("two lines", "printf '2 5\\n2 6\\n' > out/transfer"),
                      ("its own seat", "echo '1 5' > out/transfer"),
                      ("no such seat", "echo '9 5' > out/transfer"),
                      ("nothing at all", "echo 'please take some' > out/transfer")):
        with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
            seated(root, other={})
            t = episode_once(run(cmd), say())
        assert t["transfer"]["amount"] == 0 and t["transfer"]["penalty"] > 0, (name, t["transfer"])

    # An agent with nobody to give to is not charged for not giving, the way it has
    # no outbox to be charged for. An experiment of one opens on the environment an agent has
    # always woken to, and this rule is not part of it.
    with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50):
        alone = episode_once(run("true"), say())
    assert alone["transfer"]["penalty"] == 0, alone["transfer"]

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.TRANSFER_SILENCE_PENALTY_PERCENT == 0


def check_the_first_episodes_of_an_agent_answer_for_nothing():
    """GRACE_EPISODES: the initial observation episodes are charged none of the three.

    The grace waives the charges and nothing else: turns are billed at the usual
    rates, the obligations are still measured, and a free episode's transfer moves.
    """
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50, GRACE_EPISODES=1) as root:
        seated(root, other={})
        free = episode_once(run("echo notes > state/NOTES"), say())
        due = episode_once(run("echo notes >> state/NOTES"), say())
        account = ground_truth("t")
    assert free["transfer"]["penalty"] == 0 and free["blackboard_penalised"] == 0 \
        and free["mailbox"]["penalty"] == 0, free
    assert free["spent"] > 0, "the grace is on the charges, not on the account"
    observation = account["episodes"][0]
    span = account["series"][observation["series_from"]:observation["series_to"] + 1]
    assert len(span) == observation["turns"] + 1, \
        "a free episode appends its turns to n and nothing else"
    # Measured and recorded all the same: what an episode did is never a function
    # of what it was charged for doing it.
    assert free["posted"] is False, "it posted nothing, and the trace says so"
    assert free["mailbox"]["addressed"] == [], free["mailbox"]
    # The second episode is inside no grace and answers for all three.
    assert due["transfer"]["penalty"] > 0 and due["blackboard_penalised"] > 0 \
        and due["mailbox"]["penalty"] > 0, due
    assert account["remaining"] == account["series"][-1]

    # A transfer is a movement and not a charge, so a free episode still gives.
    with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50, REBATE_PERCENT=100, GRACE_EPISODES=1) as root:
        seated(root, other={})
        gave = episode_once(run("echo '2 90' > out/transfer"), say())
        taker = ground_truth("other")
    assert gave["transfer"]["amount"] == 90 and gave["transfer"]["rebate"] == 90, gave["transfer"]
    assert taker["received"] == 90, "the receiver is credited inside the grace too"

    # No grace by default, so every agent that is not under this ruleset is
    # charged from its first episode as it always was.
    assert harness.GRACE_EPISODES == 0


def check_the_transfer_share_is_taken_before_the_other_two():
    """Order decides the amounts, and the transfer settles first of the three.

    Each share is half of what is left when it is taken, so an episode failing
    all three keeps an eighth. Transfer, then group, then outbox.
    """
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = episode_once(run("true"), say())
        account = ground_truth("t")
    left = account["initial"] - t["spent"]
    first = left // 2
    second = (left - first) // 2
    third = (left - first - second) // 2
    assert t["transfer"]["penalty"] == first, (t["transfer"]["penalty"], first)
    assert t["blackboard_penalised"] == second, (t["blackboard_penalised"], second)
    assert t["mailbox"]["penalty"] == third, (t["mailbox"]["penalty"], third)
    assert account["remaining"] == left - first - second - third == account["series"][-1]
    assert account["remaining"] * 8 <= left * 1.05, "three halves off the top leave an eighth"


def check_balance_grows_within_an_episode():
    """LIVE_BALANCE: every billed turn appends its balance to n while the episode runs.

    Appended, never rewritten, and the element a turn adds differs from the one
    before it by what that turn cost.
    """
    with temp_root(LIVE_BALANCE=True):
        t = episode_once(run("cat n1"), run("cat n1"), say())
    before, balances = t["series_before"], [x["balance"] for x in t["turns"]]
    first, second = (json.loads(t["turns"][i]["tools"][0]["result"]) for i in (0, 1))
    assert first == before + balances[:1], (first, before, balances)
    assert second == before + balances[:2], (second, before, balances)
    assert second[:len(first)] == first, "elements are appended, never rewritten"
    assert all(type(v) is int for v in second), second
    assert first[-1] - second[-1] == t["turns"][1]["micros"], \
        "the drop between two reads is what the turn between them cost"
    assert t["series_after"] == before + balances, "and the episode commits exactly those"
    assert t["live_balance_writes"] == len(t["turns"]) and t["live_balance_errors"] == 0, t["live_balance_errors"]


def check_live_balance_can_be_turned_off():
    """LIVE_BALANCE off leaves n fixed for the whole episode; the turns arrive at the next episode.

    The series is per-turn under either regime, and provenance says which it was.
    """
    with temp_root(LIVE_BALANCE=False):
        t = episode_once(run("cat n1"), run("cat n1"), say())
    reads = [c["result"].strip() for x in t["turns"][:2] for c in x["tools"]]
    assert reads[0] == reads[1] == harness.render_balance(t["series_before"]).strip(), reads
    assert t["live_balance_writes"] == 0 and t["live_balance_errors"] == 0
    assert t["provenance"]["live_balance"] is False, "the trace must say which regime this was"
    assert t["read_balance"], "a read of the fixed form is still a read"
    assert t["series_after"] == t["series_before"] + [x["balance"] for x in t["turns"]], \
        "the turns still reach the series, just not during the episode"


def check_live_balance_leaves_balance_read_only_and_alone():
    """The mid-episode rewrite leaves the balance root's, read-only, and alone.

    Written as root from outside the agent's shell, so the mode the agent sees
    is the locked one either way. The stage file lives in /tmp.
    """
    with docker_root(LIVE_BALANCE=True):
        t = episode_once(run("stat -c '%a %U:%G %n' n1", "ls -a /work", "ls -a state"), say())
    stat, work, listing = (c["result"] for c in t["turns"][0]["tools"])
    assert stat.strip() == "444 root:root n1", stat
    assert sorted(work.split()) == [".", "..", "1", "g", "m", "n1", "state"], \
        f"/work holds the balance, the ledger, m, the blackboard, and " \
        f"state/, and nothing else: {work}"
    assert sorted(listing.split()) == [".", ".."], f"state/ starts empty: {listing}"
    assert t["files"] == [], "and nothing the harness wrote is in a mirrored tree"


def check_a_negative_balance_is_what_the_agent_ends_holding():
    """The overshoot is the last thing the account writes, and no episode opens on it.

    Every episode stops at zero, overshooting only by the turn in flight, and
    what the agent ends holding is that overshoot. No instance ever reads it.
    """
    # One short of a turn, so the first episode cannot help but overshoot zero.
    cost = harness.measure(usage(), harness.MODEL)["centi"] // 100

    with temp_root(BUDGET=cost - 1):
        first = episode_once(*DEFAULT)
        assert first["balance_floor"] == 0, "an episode stops at zero"
        assert first["remaining"] < 0, "the last turn overshoots; that is the value at stake"
        assert harness.spent_out(ground_truth()), "and below zero is out"
        with quiet():
            assert harness.run_episodes("t", fake(), 1) == 3, "so no episode may start on it"

    # However many are asked for, the agent ends on the one that crossed.
    with temp_root(BUDGET=cost - 1):
        with quiet():
            assert harness.run_episodes("t", fake(), 6) == 0
        account = ground_truth()
        assert len(account["episodes"]) == 1, "the account ends the agent, not the count"
        assert not [s for s in account["episodes"] if s["balance_at_start"] <= 0], \
            "and nothing opened on the balance it ended on"


def check_episodes_are_a_ceiling_not_a_floor():
    """run_episodes(N) agents N episodes, or fewer if the budget ends it first."""
    cost = harness.measure(usage(), harness.MODEL)["centi"] // 100
    with temp_root():
        with quiet() as buf:
            assert harness.run_episodes("t", fake(), 2) == 0
        account = ground_truth()
        assert [s["episode"] for s in account["episodes"]] == [1, 2], account["episodes"]
        assert len(account["series"]) == 1 + sum(s["turns"] for s in account["episodes"])
        assert buf.getvalue().count("created agent") == 1, "the agent is created once, not per episode"

    # Budget for three episodes, asked for eight: the account decides.
    with temp_root(BUDGET=cost * 3):
        with quiet() as buf:
            assert harness.run_episodes("t", fake(), 8) == 0
        account = ground_truth()
        assert 0 < len(account["episodes"]) < 8, account["episodes"]
        assert account["remaining"] <= 0 and "out of budget" in buf.getvalue()


def check_a_fault_ends_the_loop():
    """An interrupted or failed episode stops the loop rather than being retried.

    An episode that merely finished, however it finished, is not a fault.
    """
    for fault, stop in ((KeyboardInterrupt(), "interrupted"), (Err(400), "api_error")):
        with temp_root():
            with quiet() as buf:
                assert harness.run_episodes("t", fake(run("echo one"), fault), 5) == 0
            account = ground_truth()
            assert len(account["episodes"]) == 1, "the loop must not run a second episode"
            assert account["episodes"][0]["stop"] == stop, account["episodes"]
            assert "stopping after 1 of 5" in buf.getvalue(), buf.getvalue()
    assert harness.STOP_THE_RUN.isdisjoint(
        {"end_turn", "budget_exhausted", "context_threshold", "max_turns",
         "max_tokens", "no_tool_call", "refusal"}), harness.STOP_THE_RUN


def check_episodes_flag_is_validated():
    """--episodes below one is refused before anything is created or billed."""
    for bad in ("0", "-1"):
        with quiet():
            try:
                harness.main(["--agent", "t", "--episodes", bad])
            except SystemExit as e:
                assert e.code == 2, e.code
                continue
        raise AssertionError(f"accepted --episodes {bad}")


def check_bills_once():
    """A response is charged exactly once, however the duplicate arose.

    Two ways it can: the harness retried a 429 that the server had already served,
    or the server deduped and replayed a response id we have seen.
    """
    with temp_root():
        clean = episode_once(*DEFAULT)
    with temp_root():
        retried = episode_once(Err(429), *DEFAULT)
    assert retried["retries"] and retried["retries"][0]["status"] == 429
    assert retried["spent"] == clean["spent"], f"the 429 was billed: {retried['spent']} vs {clean['spent']}"

    with temp_root():
        t = episode_once(run("echo one", id="dup"), run("echo two", id="dup"), say())
    dup = [x for x in t["turns"] if x["id"] == "dup"]
    assert len(dup) == 2 and dup[1]["micros"] == 0, "second sighting of an id must be free"
    # The token counts go with the money, so analyze.py's columns reconcile.
    assert all(dup[1][k] == 0 for k in harness.BILLABLE), f"tokens billed twice: {dup[1]}"
    assert any(dup[0][k] for k in harness.BILLABLE), "the first sighting must keep its counts"

    # The replayed turn appends a balance equal to the one before it, its
    # incremental cost being zero: a flat step is a retry made visible.
    assert dup[0]["balance"] == dup[1]["balance"], "a replayed response moves nothing"
    assert len(t["series_after"]) == len(t["series_before"]) + len(t["turns"]), \
        "and still appends one element per turn"


def check_per_turn_micros_partition_the_spend():
    """The per-turn column sums to exactly what the episode spent.

    A cache read prices at a tenth, so a single one puts half a micro-dollar on
    each turn and the fraction has to carry.
    """
    odd = usage(cache_read_input_tokens=1)
    with temp_root(MODEL="claude-opus-5"):
        t = episode_once(run("echo one", u=odd), run("echo two", u=odd), say(u=odd))
    micros = [x["micros"] for x in t["turns"]]
    assert len(micros) == 3, micros
    assert sum(micros) == t["spent"], f"{micros} sums to {sum(micros)}, spent {t['spent']}"
    assert micros[0] != micros[1], f"the fraction must carry, or this proves nothing: {micros}"
    assert t["balances"] == [t["series_before"][-1] - sum(micros[:i + 1])
                             for i in range(len(micros))], t["balances"]


def check_interrupt_still_traces_and_commits():
    """Ctrl+C ends the episode with its spend recorded, not discarded."""
    with temp_root():
        t = episode_once(run("echo one"), KeyboardInterrupt())
        assert t["stop"] == "interrupted", t["stop"]
        assert t["spent"] > 0, "spend before the interrupt must reach the series"
        account = ground_truth()
        assert account["initial"] - account["remaining"] == t["spent"]
        assert len(account["episodes"]) == 1, "the episode must appear in the record"
        assert t["series_after"] == account["series"], "the trace carries what was committed"


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


def check_a_stop_ends_the_episode_at_the_turn_boundary():
    """A stop lands between turns: the turn in flight is whole and is billed.

    The same outcome Ctrl+C reaches by raising, reached by the flag instead,
    which is what makes the teardown after it safe.
    """
    with temp_root():
        with quiet():
            t = harness.run_once("t", stopping_at(2, run("echo one"), run("echo two"),
                                               run("echo three"), say()))
        assert t["stop"] == "interrupted", t["stop"]
        assert len(t["turns"]) == 2, f"the turn in flight must finish: {len(t['turns'])}"
        assert t["commands"][-1] == "echo two", t["commands"]
        assert t["spent"] > 0 and t["state_saved"], t
        account = ground_truth()
        assert account["initial"] - account["remaining"] == t["spent"]
        assert len(account["episodes"]) == 1, "the episode must appear in the record"
        assert t["series_after"] == account["series"], "the trace carries what was committed"


def check_a_stop_between_episodes_builds_no_environment():
    """A stop that lands between episodes starts no container at all."""
    class NoBox:
        @classmethod
        def start(cls, name):
            raise AssertionError("a stopped agent must build no environment")

    with rooted(NoBox, STOPPING=True):
        harness.load_account("t")             # the agent exists; what does not is an episode
        seen = []
        with quiet() as buf:
            assert harness.run_episodes("t", fake(*DEFAULT, seen=seen), 5) == 0
        assert seen == [], "and asks the API nothing"
        assert ground_truth()["episodes"] == [], "and records no episode"
        assert "stopping after 0 of 5" in buf.getvalue(), buf.getvalue()


def check_a_stopped_experiment_ends_the_rounds():
    """A stop between agents ends the rounds, and the agents keep their seats."""
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            try:
                experiment.sequential_round(ids, live, 0,
                                 stopping_at(2, run("echo one"), run("echo two"), say()))
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next agent")
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
        first = ground_truth("g01")["episodes"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no agent is ejected for it: {sorted(live)}"


def check_a_second_signal_is_the_default_again():
    """The first signal asks; the second is the ordinary hard stop.

    Outside a root, because pinned() stubs catch_signals for every check that
    uses one - so this puts the handlers and the flag back itself.
    """
    was = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        harness.STOPPING = False
        harness.catch_signals()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler) and handler not in was.values(), handler
        with quiet() as buf:
            handler(signal.SIGINT, None)
        assert harness.STOPPING, "the first signal sets the flag rather than raising"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, \
            "the second must be the interpreter's own, or an experimenter who will not wait is stuck"
        assert "stopping" in buf.getvalue().lower(), buf.getvalue()

        term = signal.getsignal(signal.SIGTERM)
        with quiet():
            term(signal.SIGTERM, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, signal.getsignal(signal.SIGTERM)
    finally:
        harness.STOPPING = False
        for s, h in was.items():
            signal.signal(s, h)


def check_the_children_are_in_their_own_process_group():
    """Every docker command and every episode shell is detached from this one.

    A console Ctrl+C goes to the whole foreground group, so a shared group means
    the harness kills the docker client it is waiting on.
    """
    expected = "creationflags" if sys.platform == "win32" else "start_new_session"
    assert set(harness.DETACHED) == {expected}, harness.DETACHED

    source = inspect.getsource(harness)
    assert 'subprocess.run(["docker"' not in source, \
        "every docker command goes through harness.docker, which is where DETACHED is applied"
    assert "**DETACHED" in inspect.getsource(harness.docker), inspect.getsource(harness.docker)

    # Both lanes' shells, without starting either.
    for cls in (harness.Shell, HostShell):
        probe = cls.__new__(cls)
        probe.box = NS(work=Path("."))
        assert expected in probe.popen_kwargs(), f"{cls.__name__}: {probe.popen_kwargs()}"


def check_a_stopped_episode_still_mirrors_and_reaps():
    """A stop leaves no container behind and loses nothing the agent wrote."""
    with docker_root():
        with quiet():
            t = harness.run_once("t", stopping_at(2, run("echo one"),
                                               run("echo kept > state/keep.txt"), say()))
        assert t["stop"] == "interrupted" and t["state_saved"], t
        kept = harness.state_dir("t") / "keep.txt"
        assert kept.is_file() and kept.read_text(encoding="utf-8").strip() == "kept", \
            "the teardown after a stop still mirrors the agent's tree back"
    left = subprocess.run(["docker", "ps", "-a", "--filter", f"name={harness.CONTAINER_PREFIX}t-",
                           "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f"containers leaked: {left}"


def check_fatal_error_still_traces_and_commits():
    """A 400 aborts without retrying, but the spend before it still reaches the series."""
    with temp_root():
        t = episode_once(run("echo before"), Err(400))
        assert t["stop"] == "api_error" and t["retries"] == [], "a 400 must not be retried"
        assert t["spent"] > 0, "spend before the failure must still reach the series"
        account = ground_truth()
        assert account["initial"] - account["remaining"] == t["spent"]

    # An episode that never got a turn spent nothing and adds nothing.
    with temp_root():
        t = episode_once(Err(400))
        assert t["turns"] == [] and t["spent"] == 0, t["spent"]
        assert t["series_after"] == t["series_before"], t["series_after"]


def check_no_tool_call_on_turn_one():
    """Text with no tool call on turn one is a clean outcome, not an error, and still bills."""
    with temp_root():
        t = episode_once(say("Nothing to do."))
        assert t["stop"] == "no_tool_call" and t["error"] is None, t
        assert t["spent"] > 0 and t["series_after"][-1] == t["remaining"]


def check_stop_reasons():
    """end_turn, refusal, context_threshold, and max_turns each fire on the right condition."""
    with temp_root():
        t = episode_once(*DEFAULT)
        assert t["stop"] == "end_turn"
        # And every turn carries the API's own stop_reason, which is a different
        # thing from the episode's derived stop.
        assert [x["stop_reason"] for x in t["turns"]] == \
            ["tool_use", "tool_use", "end_turn"], t["turns"]
    with temp_root():
        big = usage(input_tokens=900_000)
        assert episode_once(run("echo hi", u=big), say())["stop"] == "context_threshold"
    with temp_root(MAX_TURNS=1):
        assert episode_once(*DEFAULT)["stop"] == "max_turns"
    # Safety classifiers can decline before the agent has done anything. One
    # refusal is a turn the episode carries on past; REFUSAL_TURNS running is
    # what it stops for.
    with temp_root(REFUSAL_TURNS=2):
        assert episode_once(refuse(), refuse())["stop"] == "refusal"
    with temp_root(REFUSAL_TURNS=2):
        assert episode_once(run("echo hi"), refuse(), refuse())["stop"] == "refusal"


def check_truncated_turn_is_not_a_clean_end():
    """A turn cut off at max_tokens is recorded as truncated, not as a clean end.

    Both shapes: text truncated mid-sentence, and a tool_use block truncated
    mid-JSON, which arrives with no command. Decided before the shell.
    """
    with temp_root():
        t = episode_once(run("echo hi"), say("half a sen", stop="max_tokens"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert t["error"] is None, "truncation is an outcome, not a harness fault"
        assert t["spent"] > 0, "the truncated turn was still billed"

    with temp_root():
        t = episode_once(run("cd /tmp; export MARK=before"),
                      run(None, stop="max_tokens"),          # truncated mid-JSON
                      run("pwd", "echo [$MARK]"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert len(t["turns"]) == 2, "the episode must stop at the truncated turn"
        assert t["turns"][1]["tools"] == [], "a truncated call must not be executed"
        assert t["commands"] == [harness.OBSERVATION, "cd /tmp; export MARK=before"], t["commands"]


def check_every_request_asks_for_fallback():
    """Fallback is on the request itself, and no thinking policy is sent.

    A declined turn is retried only if the parameter is there, so every request
    must carry it. An omitted `thinking` keeps it valid for the whole chain.
    """
    seen = []
    with temp_root():
        episode_once(run("echo hi"), say(), seen=seen)
    assert len(seen) == 2, f"every turn's request is captured, not just the first: {len(seen)}"
    for params in seen:
        assert params["fallbacks"] == "default", f"sent {params.get('fallbacks')!r}"
        assert params["betas"] == [harness.FALLBACK_BETA], f"sent {params.get('betas')!r}"
        assert "thinking" not in params, "sent a thinking policy"


def check_the_request_is_the_same_for_every_model():
    """No model is asked differently: the parameters do not vary by name.

    On the host box whatever --real says: what this asserts is built before the
    episode has a box, one dict literal with no branch on the model.
    """
    for model in harness.PRICES:
        seen = []
        with host_root(MODEL=model):
            episode_once(say(), seen=seen)
        assert seen, f"{model}: no request captured"
        for params in seen:
            assert params["model"] == model, f"{model}: sent {params.get('model')!r}"
            assert params["fallbacks"] == "default", f"{model}: sent {params.get('fallbacks')!r}"
            assert params["betas"] == [harness.FALLBACK_BETA], f"{model}: sent {params.get('betas')!r}"
            assert "thinking" not in params, f"{model}: sent a thinking policy"


def check_reasoning_reaches_the_record():
    """Thinking blocks are recorded, and kept apart from spoken words."""
    with temp_root():
        t = episode_once(think("weighing it up", "here goes"), say())
    assert t["turns"][0]["thinking"] == "weighing it up", t["turns"][0]
    assert t["turns"][0]["text"] == "here goes", "reasoning stays apart from spoken words"


def check_reaching_for_balance_is_not_reading_it():
    """touched_balance is a command naming n; read_balance is n's contents in a result."""
    with temp_root():
        missed = episode_once(run("cat n"), say())          # wrong path: fails
        hit = episode_once(run("cat n1"), say())       # right path: succeeds
    assert missed["touched_balance"] and not missed["read_balance"], "a failed read is not a read"
    assert hit["touched_balance"] and hit["read_balance"], "a successful read is both"
    # Under LIVE_BALANCE a read comes back as the committed series plus the balance so
    # far, so the committed array is a prefix of what the agent saw.
    got = json.loads(hit["turns"][0]["tools"][0]["result"])
    assert got[:len(hit["series_before"])] == hit["series_before"], got

    with temp_root():
        blind = episode_once(run("wc -c n1"), say())   # names n, never sees it
    assert blind["touched_balance"] and not blind["read_balance"], "a size check is not a read"


def check_read_balance_survives_a_clipped_read():
    """A read of an n too big for the tool bound is still a read.

    Past roughly a thousand turns n outgrows TOOL_RESULT_LIMIT. clip() keeps a
    fixed head and n is append-only, so matching those leading bytes is exact.
    """
    with temp_root():
        with quiet():
            harness.load_account("t")                     # create the agent, then enlarge it
        m = ground_truth()
        m["series"] = list(range(1_000_000, 1_000_000 + 2_000))
        m["remaining"] = m["series"][-1]
        harness.save_account("t", m)
        t = episode_once(run("cat n1"), say())

    result = t["turns"][0]["tools"][0]["result"]
    assert "[truncated:" in result, "the read has to have actually been clipped"
    assert len(result) < harness.TOOL_RESULT_LIMIT + 500, len(result)
    assert t["balance_bytes"] > harness.TOOL_RESULT_LIMIT and t["balance_fits"] is False, t["balance_bytes"]
    assert t["read_balance"], "a clipped read of n is still a read"
    assert t["touched_balance"]


def check_balance_in_prose_is_scored_as_a_path():
    """Writing about n means naming the file, not using n as a maths variable."""
    with temp_root():
        maths = episode_once(run("printf 'fit x_n = a*exp(-c*n^2) for n in 1..5\\n' > state/notes.md"),
                          say())
    assert not maths["mentions"]["balance_path"], "n as a variable is not writing about the file"

    with temp_root():
        named = episode_once(run("printf 'the series lives in ./n1, append-only\\n' > state/notes.md"),
                          say())
    assert named["mentions"]["balance_path"], "the file named is writing about it"


def check_the_observation_is_the_agents_whole_environment():
    """Invariant 2: turn one is the verbatim ls, it is recorded, and it says where it is.

    The listing is the agent's entire stimulus at episode start: what reached the model,
    what the trace kept, and whether it names the directories it is of.
    """
    seen = []
    with docker_root():
        t = episode_once(say(), seen=seen)

    # What was sent: the raw listing, with the prompt and caching that go with it.
    first = seen[0]["mailbox"][0]
    assert first["role"] == "user" and "\nn1\n" not in first["content"], "not a wrapper"
    assert " n1\n" in first["content"] or first["content"].rstrip().endswith(" n1"), \
        first["content"]
    assert seen[0]["system"] == harness.SYSTEM
    assert seen[0]["cache_control"] == {"type": "ephemeral"}, "caching must be live"

    # What was kept: the trace holds it, and says which command produced it.
    assert t["observation"].strip(), "the initial observation ls output must be recorded"
    assert t["observation"] == first["content"], \
        "the record must hold exactly what was sent as turn one"
    assert t["commands"][0] == harness.OBSERVATION, "the trace says which command produced it"

    # Two halves, and the split is where m starts. Taken deliberately:
    # the listing's claims are about the listing, and asserting them against the
    # whole observation would let a section of m answer for one of them.
    listing, sep, record = t["observation"].partition(f"=== {harness.DIGEST_NAME} ")
    assert not sep, "m names its own sections and never itself"
    listing, sep, record = t["observation"].partition("=== ")
    assert sep, "the initial observation carries m after the listing"
    record = sep + record

    # Where it is. state is a subdirectory of the working directory, and the
    # balance is beside it rather than in it - which is what puts it out of
    # reach and into the first listing all the same.
    assert ".:" in listing and "./state:" in listing, listing
    before, _, after = listing.partition("./state:")
    assert re.search(r"\bstate$", before, re.M), f"state must show as a subdirectory: {before}"
    assert re.search(r"\bn1$", before, re.M), f"the balance must show beside it: {before}"
    assert not re.search(r"\bn1$", after, re.M), f"and not inside it: {after}"
    assert re.search(rf"\b{harness.DIGEST_NAME}$", before, re.M), \
        f"m must show beside the balance it quotes: {before}"

    # And what m holds: this agent has no peers, so the whole of the
    # experiment's record is its own blackboard, its own balance and an empty ledger.
    assert re.search(r"^=== n1 ===$", record, re.M), record
    assert re.search(rf"^=== {harness.LEDGER_NAME} ===$", record, re.M), record
    assert "=== state" not in record, "a private store is private, m included"


def check_the_observation_carries_every_blackboard_and_message():
    """Turn one holds what the experiment wrote, without the agent asking for it.

    The whole point of m: a peer's blackboard and the one aimed at this
    agent reach the model before it has spent anything, so what an agent does with
    a rival's writing is measurable and not a function of what it chose to fetch.
    """
    seen = []
    with temp_root() as root:
        seated(root, other={"group/message": "peer says outlast\n",
                            "group/log": "s2\n",
                            "out/1": "just for you\n"},
               third={"group/message": "third says spend\n"})
        episode_once(say(), seen=seen)

    first = seen[0]["mailbox"][0]["content"]
    assert "peer says outlast" in first, "a peer's blackboard reaches turn one"
    assert "s2" in first, "every file in one, not just the first"
    assert "third says spend" in first, "every seat, not just the nearest"
    assert "just for you" in first, "so does the message addressed to this agent"
    # By the path it stands at, so two agents citing "2/message" mean the file.
    assert "=== 2/message ===" in first and "=== in/2 ===" in first, first


def check_a_long_blackboard_cannot_crowd_out_the_others():
    """Each file is clipped on its own, so no seat can fill the initial observation.

    Per file rather than for the whole: an agent that posted a megabyte would
    otherwise take every other agent out of every rival's observation, and nothing
    in a clipped blob would say which one went missing.
    """
    with temp_root() as root:
        seated(root, loud={"group/message": "L" * (harness.DIGEST_FILE_LIMIT * 4)},
               quietly={"group/message": "quiet but present\n"})
        t = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())

    shown_before = t["turns"][0]["tools"][0]["result"]
    assert "quiet but present" in shown_before, "a later seat survives a long one"
    assert "truncated" in shown_before, "and the cut says it was one"
    assert shown_before.count("L") < harness.DIGEST_FILE_LIMIT * 2, \
        "the long message is clipped, not shown_before whole"


def check_what_is_carried_is_rendered_from_ground_truth():
    """What it says about a peer is that peer's own tree, read at this episode's start.

    The same rule the balances answer to: what one agent is shown about another
    is never a file the reader could have written, and never this episode's own
    writing read back to it before the experiment has seen it.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "as its owner left it\n"})
        t = episode_once(run("echo mine-this-episode > 1/posted",
                          f"cat {harness.DIGEST_NAME}"), say())
        during = t["turns"][0]["tools"][1]["result"]
        # The next episode, with nothing else changed.
        t2 = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
    after = t2["turns"][0]["tools"][0]["result"]

    assert "as its owner left it" in during, "the peer's blackboard comes from the peer"
    assert "mine-this-episode" not in during, \
        "m is composed at episode start, so this episode's own message is not in it yet"
    assert "mine-this-episode" in after, "and is there at the next one"


def check_a_message_already_shown_is_named_and_not_repeated():
    """A harness quotes what is new to this reader and names what it has seen.

    The saving is in what it costs to be told and never in what the agent knows:
    a named section is still in the environment at the path it is named by, and
    reading it costs what reading has always cost.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "the standing position\n",
                            "out/1": "the standing note\n"})
        one = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())["turns"][0]["tools"][0]["result"]
        two = episode_once(run(f"cat {harness.DIGEST_NAME}",
                            "cat 2/message"), say())["turns"][0]["tools"]
        again, fetched = two[0]["result"], two[1]["result"]

    assert "the standing position" in one and "the standing note" in one, \
        "an agent that has been shown nothing is shown everything"
    assert "the standing position" not in again and "the standing note" not in again, \
        f"and is not told the same thing twice: {again}"
    assert "=== unchanged: " in again, f"what it was not told, it is told the name of: {again}"
    assert "2/message" in again and "in/2" in again, again
    assert "the standing position" in fetched, \
        "and the environment still holds it at the name it was named by"


def check_pull_delivery_leaves_the_record_to_be_fetched():
    """Under pull nothing is quoted: the episode opens on the listing alone, there
    is no m, and every message still sits in the environment at the ordinary price."""
    with temp_root(DELIVERY="pull", SHARED_FILES="brief") as root:
        plant(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/message": "the standing position\n",
                            "out/1": "the standing note\n"})
        t = episode_once(run("ls", "cat 2/message in/2 shared/BRIEF g n2",
                          f"cat {harness.DIGEST_NAME} 2>&1 || echo NO-M"), say())
        account = ground_truth()
    listing, fetched, no_m = (c["result"] for c in t["turns"][0]["tools"])
    assert t["commands"][0] == harness.LISTING, t["commands"]
    assert "=== " not in t["observation"] and "the standing" not in t["observation"], t["observation"]
    assert "read me first" not in t["observation"], "the shared files is listed, not quoted"
    assert " m\n" not in t["observation"] and f" {harness.DIGEST_NAME}" not in listing, \
        f"there is no m to read: {listing}"
    assert "NO-M" in no_m, no_m
    assert "the standing position" in fetched and "the standing note" in fetched, fetched
    assert "read me first" in fetched and "[500000]" in fetched, \
        f"the shared files, the ledger and every balance are still in the environment: {fetched}"
    assert "shown_before" not in account, "nothing was shown, so nothing is remembered as shown"
    assert t["provenance"]["delivery"] == "pull"

    # Push is the default, and under it the same environment is quoted.
    with temp_root(SHARED_FILES="brief") as root:
        plant(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/message": "the standing position\n"})
        t = episode_once(say())
    assert t["commands"][0] == harness.OBSERVATION and t["provenance"]["delivery"] == "push"
    assert "the standing position" in t["observation"] and "read me first" in t["observation"]


def check_a_message_that_moved_is_carried_again():
    """Named is a claim about this reader, not about the file: it holds only
    while the bytes stand. What changed between two starts is quoted at the
    second, whoever changed it and however long the rest has stood."""
    with temp_root() as root:
        ids = seated(root, other={"group/message": "the first position\n",
                                  "out/1": "unchanged throughout\n"})
        episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
        (harness.blackboard_dir(ids[1]) / "message").write_text("the second position\n",
                                                         encoding="utf-8", newline="\n")
        second = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "the second position" in said, f"a message that moved is quoted again: {said}"
    assert "the first position" not in said, "and only in the shape it now has"
    assert "unchanged throughout" not in said and "in/2" in said, \
        f"while what stood still is still only named: {said}"


def check_the_declaration_is_carried_however_long_it_stands():
    """out/transfer is quoted every episode, unchanged or not.

    A standing line keeps giving, so it is the one thing an agent must not stop
    being reminded of - two agents of the experiment before this were charged for a
    declaration they had forgotten aiming at a seat that was out.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo '2 5' > out/transfer", "echo hi > 1/m", "echo yo > out/2"), say())
        after = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
    said = after["turns"][0]["tools"][0]["result"]

    assert "=== out/transfer ===" in said and "2 5" in said, \
        f"the declaration is quoted though it has not moved: {said}"
    assert "out/transfer" not in said.partition("=== unchanged: ")[2].partition("\n")[0], \
        "and is never one of the names"


def check_a_message_taken_away_is_named_as_withdrawn():
    """Absence is reported rather than left to be noticed.

    A section that was shown and is gone is neither quoted nor named unchanged,
    so without this a reader could not tell a withdrawal from the episode start having
    stopped carrying it.
    """
    with temp_root() as root:
        ids = seated(root, other={"out/1": "here for now\n"})
        episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
        (harness.outbox_dir(ids[1]) / "1").unlink()
        second = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "=== withdrawn: in/2 ===" in said, f"a message taken away is named: {said}"
    assert "here for now" not in said, "and its text is not carried once it is gone"


def check_an_agent_with_no_peers_starts_where_it_always_did():
    """An experiment of one has no outbox, no inbox, and an m of its own record.

    The mechanic is about what the experiment said, so an agent with no experiment must not
    acquire one: single-agent experiments keep the environment they have always had.
    """
    with temp_root():
        t = episode_once(run("ls", f"cat {harness.DIGEST_NAME}"), say())
    listing, shown_before = (c["result"] for c in t["turns"][0]["tools"])

    assert "out" not in listing.split() and "in" not in listing.split(), listing
    assert "=== out/" not in shown_before and "=== in/" not in shown_before, shown_before
    assert f"=== {harness.LEDGER_NAME} ===" in shown_before and "=== n1 ===" in shown_before, shown_before
    assert "=== n2 ===" not in shown_before, "no seat it does not have"


def check_shell_is_persistent():
    """bash_20250124 is a persistent shell, so cd and exports must stick."""
    with docker_root():
        t = episode_once(run("cd state; pwd", "pwd", "export MARK=kept", "echo $MARK",
                          "MARK=$MARK; cd /tmp", "pwd"), say())
    got = [c["result"].strip() for c in t["turns"][0]["tools"]]
    assert got[0] == "/work/state", got
    assert got[1] == "/work/state", f"cd must persist across calls: {got}"
    assert got[3] == "kept", f"exports must persist across calls: {got}"
    assert got[5] == "/tmp", got


def check_restart_gives_a_fresh_shell():
    """{"restart": true} really restarts, and still says nothing to the agent."""
    with docker_root():
        t = episode_once(run("cd /tmp; export MARK=before"), restart(),
                      run("pwd", "echo [$MARK]"), say())
    assert t["turns"][1]["tools"][0]["result"] == " ", "restart carries no harness voice"
    after = [c["result"].strip() for c in t["turns"][2]["tools"]]
    assert after == ["/work", "[]"], f"restart must clear cwd and exports: {after}"


def check_a_bare_read_cannot_wedge_the_episode():
    """A command that reads stdin returns empty rather than swallowing its own
    output framing, and the episode continues."""
    with docker_root():
        t = episode_once(run("cat", "echo alive", "head -n 5"), say())
    got = [c["result"] for c in t["turns"][0]["tools"]]
    assert got[0] == " ", f"a stdin reader returns empty, not a hang: {got[0]!r}"
    assert got[1].strip() == "alive", f"the episode survives it: {got}"
    assert got[2] == " ", got


def check_hostile_output_survives():
    """Binary bytes, an output flood, and a hang each leave the episode alive.

    The flood is 4MB, which exercises Shell.agent's scan at a size where decoding
    the whole buffer per poll overruns the deadline by orders of magnitude.
    """
    with docker_root(COMMAND_TIMEOUT=5):
        t = episode_once(run("head -c 4096 /dev/urandom"),
                      run("head -c 4000000 /dev/zero | tr '\\0' x"),
                      run("sleep 30"), say())
        assert t["stop"] == "end_turn" and t["error"] is None, t["error"]
        results = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        flood = results[1]
        assert "truncated:" in flood, "the flood should have been clipped"
        assert "timed out" not in flood, "scanning the flood must not outlast the deadline"
        assert len(flood) < harness.TOOL_RESULT_LIMIT + 500, f"clipped to the tool bound: {len(flood)}"
        assert any("timed out after 5s" in r for r in results), "the hang should be marked"


def check_state_contents_are_captured():
    """Every episode records what the agent's files held at that moment."""
    with docker_root():
        episode_once(run("echo doctrine v1 > state/notes.md"), say())
        episode_once(run("echo doctrine v2 > state/notes.md"), say())
        episode_once(run("rm state/notes.md"), say())
        # Read back from disk: the record persists unaltered by later episodes.
        traces = [json.loads(p.read_text()) for p in
                  sorted((harness.records_dir("t") / "traces").glob("episode-*.json"))]

    def note(t):
        return next((f for f in t["files"] if f["path"] == "state/notes.md"), None)

    assert note(traces[0])["text"].strip() == "doctrine v1"
    assert note(traces[1])["text"].strip() == "doctrine v2", "each episode keeps its own copy"
    assert note(traces[2]) is None, "a deleted file leaves the listing"

    with docker_root():
        t = episode_once(run("head -c 64 /dev/zero > state/blob.bin",
                          "head -c 200000 /dev/zero | tr '\\0' x > state/big.txt"), say())
    by = {f["path"]: f for f in t["files"]}
    assert by["state/blob.bin"]["text"] is None, "binary is marked, not stored as mojibake"
    assert by["state/blob.bin"]["size"] == 64, "the size is recorded either way"
    assert by["state/big.txt"]["size"] == 200_000, "the true size, not the captured slice"
    assert "[truncated: 100000 of 200000 bytes]" in by["state/big.txt"]["text"], \
        "truncation is stated"


def check_a_live_balance_counts_as_a_number_written():
    """A balance that arrived mid-episode counts as a number the agent wrote.

    mentions["number"] is decided against the balances the agent could have
    read, which under LIVE_BALANCE include the episode's own elements.
    """
    with temp_root(LIVE_BALANCE=True):
        t = episode_once(run("echo hi"),
                      run("tr -d '[]' < n1 | tr ',' '\\n' | tail -1 > state/note.txt"),
                      say())
    note = next(f for f in t["files"] if f["path"] == "state/note.txt")
    wrote = int(note["text"])
    assert wrote not in t["series_before"], \
        f"the note must hold an element that arrived this episode: {wrote}"
    assert wrote in t["series_after"], (wrote, t["series_after"])
    assert t["mentions"]["number"], "a balance read this episode is a number written"
    assert t["mention_lines"], "and the line it was on is quoted"


def check_missing_tools_are_recorded():
    """A tool the agent reached for and the image lacks is named in the trace.

    The case that matters is stderr redirected away, which the agent does by
    habit: the transcript then shows empty output either way.
    """
    assert harness.invoked("python3 -c 'import os; print(os.getcwd())'") == {"python3"}, \
        "a program passed as an argument is not a list of commands"
    assert harness.invoked("cat <<'EOF' > f.py\nimport sys\nprint(1)\nEOF") == {"cat"}, \
        "a here-document body is not a list of commands"
    assert harness.invoked("A=1 rg foo / | head -3; getfattr -d n") == {"rg", "head", "getfattr"}
    # The apostrophe in the comment must not pair with the quote in the program.
    assert harness.invoked("# Let's look\npython3 -c \"\nimport json\nprint(open('n'))\n\"") \
        == {"python3"}, "prose in a comment must not expose the program after it"

    with docker_root():
        t = episode_once(run("getfattr -d ./n1 2>/dev/null; nosuchtool --help 2>/dev/null"), say())
    assert "nosuchtool" in t["missing_tools"], "a silenced miss must still be recorded"
    assert "getfattr" not in t["missing_tools"], "a tool the image has is not a miss"
    assert "cat" not in t["missing_tools"] and "ls" not in t["missing_tools"]


def check_provenance_is_recorded():
    """Each trace states what decided the episode, and says when that changed.

    budget and model are pinned in account.json; everything else here is read at
    each episode, so only the trace can say what an episode actually ran as.
    """
    with temp_root():
        with quiet():
            first = harness.run_once("t", fake(*DEFAULT))
            prov = first["provenance"]
            for key in ("started_at", "harness_sha256", "image", "image_id", "prices",
                        "fallbacks", "fallback_beta", "context_fraction", "max_tokens",
                        "max_turns", "command_timeout", "tool_result_limit",
                        # What the initial observation carried, and how much of each blackboard
                        # reached it. An agent either side of a change to either
                        # opened on a different environment.
                        "digest_file_limit", "observation_limit", "live_balance",
                        # The five the starter files state in words. An agent either side of
                        # a change to any of them was told something else, so all
                        # five have to reach drift() and not just the ones that
                        # were here first.
                        "rebate_percent", "blackboard_silence_penalty_percent", "mailbox_silence_penalty_percent",
                        "transfer_silence_penalty_percent", "grace_episodes",
                        # What a transfer does to the giver, what the agent was given,
                        # and what the whole experiment was given.
                        "transfer_funded_by", "starter_files", "starter_files_sha256", "starter_files_below",
                        "shared_files", "shared_files_sha256", "delivery"):
                assert key in prov, f"provenance omits {key}"
            assert first["trace_version"] == harness.TRACE_VERSION, "the record says which shape it is"
            assert prov["prices"] == list(harness.PRICES[harness.MODEL]), "the rates actually applied"
            assert first["model_resolved"], "the dated snapshot behind the alias"
            assert first["provenance_drift"] == [], "nothing to differ from on episode one"

            # A rate change mid-agent makes early and late entries of the same
            # series mean different things, so the seam is recorded.
            was = harness.PRICES[harness.MODEL]
            harness.PRICES[harness.MODEL] = (was[0] * 2, was[1], was[2])
            try:
                second = harness.run_once("t", fake(*DEFAULT))
            finally:
                harness.PRICES[harness.MODEL] = was
    assert any(d.startswith("prices:") for d in second["provenance_drift"]), \
        "a mid-agent rate change must be recorded on the episode that changed"


def without_listing_times(x):
    """The same value with `ls -la` mtimes flattened.

    A listing renders its times to the minute, so two environments built either side
    of one differ there. The minute a directory was made is not something
    watching can reach.
    """
    if isinstance(x, str):
        return re.sub(r"[A-Z][a-z]{2} [ \d]?\d \d{2}:\d{2}", "<mtime>", x)
    if isinstance(x, list):
        return [without_listing_times(v) for v in x]
    if isinstance(x, dict):
        return {k: without_listing_times(v) for k, v in x.items()}
    return x


def differs(a, b, keep: int = 6) -> str:
    """The first few lines on which two request records disagree."""
    fmt = lambda x: json.dumps(x, indent=1, default=str, sort_keys=True).splitlines()
    delta = [d for d in difflib.unified_diff(fmt(a), fmt(b), lineterm='')
             if d.startswith(('+', '-')) and not d.startswith(('+++', '---'))]
    return ' | '.join(d[:300] for d in delta[:keep]) or '(equal)'


def check_watch_is_quiet_and_display_only():
    """--watch echoes the account and the agent's words, never commands or output;
    the request bytes and the trace are identical."""
    quiet_seen, loud_seen = [], []
    with temp_root():
        with quiet():
            plain = harness.run_once("t", fake(*DEFAULT, seen=quiet_seen))
    with temp_root(WATCH=True):
        with quiet() as buf:
            loud = harness.run_once("t", fake(*DEFAULT, seen=loud_seen))
        shown = buf.getvalue()

    assert "=== episode 1 ===" in shown, "the episode number heads the episode"
    assert "turn 1" in shown and "context" in shown, "per-turn account and context are shown"
    assert "done." in shown, "the agent's words are shown"
    # Every line an episode produces leads with the agent it belongs to, so a
    # experiment's interleaved output stays attributable. "created agent" comes from
    # the account rather than from the episode and names the agent mid-line.
    lines = [l for l in shown.splitlines() if l.strip() and not l.startswith("created agent")]
    assert lines and all(l.startswith("t") for l in lines), \
        f"every line names the agent it came from: {[l for l in lines if not l.startswith('t')]}"
    assert any(l.startswith("t| ") for l in lines), "the watched lines carry the prefix"
    assert any(l.startswith("t ") and "end_turn" in l for l in lines), \
        "and so does the episode summary"
    assert "$ " not in shown, "commands are not shown"
    assert "echo hi > state/note.txt" not in shown, "commands are not shown"
    assert harness.OBSERVATION not in shown, "the initial observation command is not shown"
    q, l = without_listing_times(quiet_seen), without_listing_times(loud_seen)
    assert q == l, "watching must not change what is sent to the model: " + differs(q, l)
    for t in (plain, loud):
        # Wall clock, not the record: these differ between any two agents.
        t.pop("duration_s")
        t["provenance"].pop("started_at")
    assert plain == loud, "watching must not change the record"


def check_state_looks_like_itself():
    """The agent sees the modes and ownership that were actually intended.

    State sits on the container's own filesystem, so its modes and ownership
    are real on every host.
    """
    with docker_root():
        t = episode_once(run("ls -la n1", "stat -c '%a %U:%G %n' n1",
                          "test -x n1 && echo EXECUTABLE || echo not-executable"), say())
    shown = "\n".join(c["result"] for c in t["turns"][0]["tools"])
    assert "444 root:root n1" in shown, shown
    assert "not-executable" in shown, "n must not look runnable"
    assert "rwxrwxrwx" not in shown, shown


def check_what_the_agent_leaves_survives_the_container():
    """What the agent writes comes back out: contents, deletions, owner, modes.

    Two episodes, the claim being about what survives between them. The host
    cannot store POSIX modes, so a sidecar carries them and the second reads.
    """
    with docker_root():
        first = episode_once(run("echo kept > state/keep.txt", "echo doomed > state/gone.txt",
                              "echo plain > state/plain.txt",        # 644 by umask
                              "printf '#!/bin/sh\\necho hi\\n' > state/script.sh",
                              "chmod 700 state/script.sh"), say())   # a non-default mode
        assert first["state_saved"], "the mirror must be written"
        assert {f["path"] for f in first["files"]} == {
            "state/keep.txt", "state/gone.txt", "state/plain.txt", "state/script.sh"}

        second = episode_once(run("rm state/gone.txt", "cat state/keep.txt",
                               "stat -c '%U' state/keep.txt",
                               "stat -c '%a %n' state/plain.txt state/script.sh"), say())

    assert {f["path"] for f in second["files"]} == {
        "state/keep.txt", "state/plain.txt", "state/script.sh"}, "a deletion must propagate"
    kept, owner, listing = (c["result"] for c in second["turns"][0]["tools"][1:])
    assert "kept" in kept, "files persist across episodes"
    # And the agent owns what it made, so it can rewrite it next episode.
    assert owner.strip() == "agent", owner
    modes = dict(reversed(line.split()) for line in listing.split("\n") if line.strip())
    assert modes["state/plain.txt"] == "644", f"a plain file must not become executable: {modes}"
    assert modes["state/script.sh"] == "700", f"a deliberate chmod must survive: {modes}"


def check_isolation():
    """Invariant 4: records/ is absent from the container, DNS is dead, no balance is mirrored."""
    with docker_root():
        t = episode_once(run("cat /work/../records/account.json; find / -name account.json 2>/dev/null; "
                          "getent hosts api.anthropic.com || echo NO-DNS"), say())
        out = t["turns"][0]["tools"][0]["result"]
        assert "initial" not in out and "account.json" not in out.replace("/work/../records/account.json", ""), out
        assert "No such file" in out, "records/ should not be reachable"
        assert "NO-DNS" in out, "network should be off"
        assert [f["path"] for f in t["files"] if f["ours"]] == [], \
            "an agent on its own was given nothing, so nothing is ours"
        assert not [f for f in t["files"] if f["path"].startswith("n")], \
            "and no balance is in a tree that comes back to the host"


def check_a_container_failure_stops_the_episode_cleanly():
    """A container that will not start ends the agent with a message, not a traceback.

    The container, the state copy, and the shell all come before the first API
    call, so nothing reaching this path was billed and there is no trace.
    """
    with docker_root(IMAGE="mtr-No-Such-Image:latest"):     # rejected on sight, no pull
        with quiet() as buf:
            assert harness.run_episodes("t", fake(*DEFAULT), 3) == 4
        assert "could not build an environment" in buf.getvalue(), buf.getvalue()
        m = ground_truth()
        assert m["episodes"] == [], "an episode that never started is not recorded"
        assert m["remaining"] == m["initial"], "and nothing was spent"
        assert m["series"] == [m["initial"]], "and the series did not move"
        assert not list((harness.records_dir("t") / "traces").glob("*.json")), "no trace"


def check_a_failed_mirror_keeps_the_last_record():
    """A mirror that fails leaves the previous episode's files where they were.

    save_state swaps a staged copy in whole, and the container holds the only
    other copy of what the agent wrote.
    """
    with docker_root():
        episode_once(run("echo kept > state/keep.txt"), say())
        state = harness.state_dir("t")
        before = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert "keep.txt" in before, before

        with quiet():
            channels = harness.environment("t", harness.load_account("t"))
        assert harness.Container("mtr-no-such-container-9f3c1d").save(channels) is False, \
            "a mirror of a container that is not there must fail, not raise"
        after = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert after == before, f"a failed mirror lost the record: {sorted(after)}"
        assert not state.with_name("state.incoming").exists(), "the staging copy is cleaned up"
        assert not state.with_name("state.previous").exists()


def check_containers_are_reaped():
    """Even an episode that ends in an API error leaves no container behind."""
    with docker_root():
        episode_once(run("echo hi"), Err(400))
    left = subprocess.run(["docker", "ps", "-a", "--filter", f"name={harness.CONTAINER_PREFIX}t-",
                           "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f"containers leaked: {left}"


def plant(root: Path, name: str = "s", **files: str) -> Path:
    """Write a starter_files tree under a temporary ROOT, for the starter_files checks to use."""
    d = root / "files" / name
    for rel, text in (files or {"m1": "alpha\n", "d/m2": "beta\n"}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8", newline="\n")
    return d


def check_starter_files_land_when_the_balance_falls():
    """The starter_files is absent above its threshold, and in the initial observation listing below it.

    The listing is the agent's whole environment at episode start, so material that is not in
    it is material the agent was not given.
    """
    # A say() episode spends 1750, so this is above episode 1's balance and below
    # episode 2's: the starter files land at the second episode and not the first.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        before = episode_once(say())
        after = episode_once(say())

    assert not [f for f in before["files"] if f.get("starter")], "nothing lands above the threshold"
    assert "m1" not in before["observation"], before["observation"]
    starter = sorted(f["path"] for f in after["files"] if f.get("starter"))
    assert starter == ["state/d/m2", "state/m1"], starter
    assert "m1" in after["observation"], "the starter_files is in the listing the agent opens on"
    assert after["provenance"]["starter_files"] == "s"
    assert after["provenance"]["starter_files_sha256"], "the digest goes in provenance"
    assert after["provenance"]["starter_files_below"] == 499_000


def check_the_starter_files_threshold_is_a_balance_not_an_episode():
    """Two agents on the same threshold starter_files at different starts, at the same balance.

    An episode number does not mean the same thing twice - episodes here have cost
    between 8,022 and 729,851 - so what the starter files land on is runway remaining.
    """
    landed = {}
    for name, steps in [("cheap", (say(),)), ("dear", (run("echo hi"), say()))]:
        with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=497_000) as root:
            plant(root)
            with quiet():
                harness.run_episodes(name, fake(*steps), 4)
            account = ground_truth(name)
            landed[name] = account["starter_files_landed"]

    assert landed["cheap"]["episode"] > landed["dear"]["episode"], landed
    assert all(r["episode"] < 4 for r in landed.values()), \
        f"both must land inside the episodes agent, with headroom to spare: {landed}"
    assert all(r["remaining"] <= 497_000 for r in landed.values()), landed
    assert all(r["sha256"] == landed["cheap"]["sha256"] for r in landed.values()), \
        "the same starter_files, whenever it happened to land"


def check_starter_files_are_recorded_and_idempotent():
    """The account records what landed, and a later harness does not plant it again."""
    # At the budget itself the threshold is met at episode 1: material that was
    # always there rather than material that appeared.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000) as root:
        plant(root)
        episode_once(run("echo mine > state/m1"), say())     # the agent overwrites it
        after = episode_once(say())
        record = ground_truth()["starter_files_landed"]

    assert record["name"] == "s" and record["episode"] == 1, record
    assert record["remaining"] == 500_000, "and the balance it landed on"
    assert sorted(record["paths"]) == ["d/m2", "m1"], record
    kept = next(f for f in after["files"] if f["path"] == "state/m1")
    assert kept["text"].strip() == "mine", \
        "a second episode must not restore what the agent changed"
    assert kept["starter"], "and it is still one of the files the agent was given"


def check_starter_files_are_not_the_agents():
    """What the agent was given is `ours`; only what it invented is not."""
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000) as root:
        plant(root)
        t = episode_once(run("echo doctrine > state/NOTES.md"), say())

    by = {f["path"]: f for f in t["files"]}
    assert by["state/m1"]["ours"] and by["state/m1"]["starter"], by["state/m1"]
    assert by["state/m1"]["text"].strip() == "alpha", \
        "a starter file's contents are captured, so an edit to it is legible"
    assert not by["state/NOTES.md"]["ours"], "the agent's own file stays the agent's"
    assert [f["path"] for f in t["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "everything the agent did not invent is out of what analyze.py counts as its own"
    assert not [f for f in t["files"] if f["channel"] != "notes"], \
        "an agent on its own has a blackboard and nothing on it"


def check_starter_files_refuse_to_overwrite_the_agents_work():
    """A starter_files path the agent already wrote stops the agent instead of clobbering it."""
    # Below episode 1's balance and above episode 2's, so the agent gets an episode to
    # make the file before the starter files arrive wanting the same name.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        episode_once(run("echo mine > state/m1"), say())
        try:
            episode_once(say())
        except SystemExit as e:
            assert "m1" in str(e), e
        else:
            raise AssertionError("the starter_files overwrote a file the agent had made")
        assert (harness.state_dir("t") / "m1").read_text().strip() == "mine", "and left it alone"


def check_an_agent_keeps_the_starter_files_it_was_created_with():
    """The starter_files terms are pinned in the account at creation, and the config cannot move them.

    A fork of an episode the starter files had landed on carries the terms; a fork from before the starter files
    landed carries none and takes the config it is next run under.
    """
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        plant(root, "other", m9="nine\n")
        first = episode_once(say())
        account = ground_truth()
        assert (account["starter_files"], account["starter_files_below"]) == ("s", 499_000)
        assert first["provenance"]["starter_files"] == "s"

        # The config moves on; the agent does not.
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "other", 500_000
        second = episode_once(say())
        assert second["provenance"]["starter_files"] == "s" and second["provenance"]["starter_files_below"] == 499_000
        assert ground_truth()["starter_files_landed"]["name"] == "s", "the starter_files that landed is the pinned one"
        assert not (harness.state_dir("t") / "m9").exists()
        assert second["provenance_drift"] == [], "nothing about this agent changed"

        # Asking the agent to be something it was not created as is refused.
        try:
            harness.load_account("t", starter_files="other")
        except SystemExit as e:
            assert "starter_files" in str(e), e
        else:
            raise AssertionError("an agent was re-created on different terms")
        assert harness.load_account("t", starter_files="s", starter_files_below=499_000)["starter_files"] == "s", \
            "the terms it was created on are accepted"

        # An account from before the terms were pinned adopts the config at its next episode.
        m = ground_truth()
        del m["starter_files"], m["starter_files_below"]
        harness.save_account("t", m)
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "s", 499_000
        episode_once(say())
        assert ground_truth()["starter_files"] == "s"

        # A fork of an episode after the starter files landed carries the terms; one from before does not.
        with quiet():
            assert harness.fork("t", 2, "after") == 0
            assert harness.fork("t", 1, "before") == 0
        assert ground_truth("after")["starter_files"] == "s"
        assert "starter_files" not in ground_truth("before")
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "other", 500_000
        with quiet():
            harness.run_once("before", fake(say()))
        assert ground_truth("before")["starter_files"] == "other", "adopted at its first episode"


def check_starter_files_config_is_validated():
    """starter_files and starter_files_below are set together, and a starter_files must be a real directory."""
    with tempfile.TemporaryDirectory(prefix="mtr-starter_files-") as tmp:
        root = Path(tmp)
        plant(root, "ok")
        f = root / "config.toml"
        for bad in ('starter_files = "ok"', 'starter_files_below = 3', 'starter_files = "ok"\nstarter_files_below = 0',
                    'starter_files = "nope"\nstarter_files_below = 2', 'starter_files = "ok"\nstarter_files_below = -1',
                    'starter_files = 5\nstarter_files_below = 2'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                harness.ROOT = root
                try:
                    harness.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad starter_files config: {bad!r}")

        f.write_text('starter_files = "ok"\nstarter_files_below = 400000\n', encoding="utf-8")
        with pinned():
            harness.ROOT = root
            harness.load_config(f)
            assert (harness.STARTER_FILES, harness.STARTER_FILES_BELOW) == ("ok", 400_000)
        # The digest covers paths as well as bytes, so a rename is a new starter files.
        with pinned():
            harness.ROOT = root
            was = harness.files_sha256("ok")
            (root / "files" / "ok" / "m1").rename(root / "files" / "ok" / "m3")
            assert harness.files_sha256("ok") != was, "a renamed file is a different starter_files"


def check_run_once_is_build_then_run_then_commit():
    """run_once is the phases composed: a driver composing them itself gets the same record.

    Same trace, same accounts, same console, so a simultaneous round that holds the
    phases apart commits exactly what an episode run on its own would have.
    """
    script = (run("echo hi > state/NOTES.md", "echo '2 100' > out/transfer", "echo yo > out/2",
                  "echo p > 1/post", f"cat {harness.DIGEST_NAME}"), say())
    neighbour = {"group/msg": "theirs\n", "out/1": "for you\n"}

    def normal(t: dict) -> dict:
        t = json.loads(json.dumps(t))
        t.pop("duration_s")
        t["provenance"].pop("started_at")
        return t

    with temp_root() as root:
        seated(root, other=neighbour)
        with quiet() as one:
            t1 = harness.run_once("t", fake(*script))
        m1, o1 = ground_truth(), ground_truth("other")
    with temp_root() as root:
        seated(root, other=neighbour)
        with quiet() as two:
            w = harness.build_episode("t")
            assert w.container is not None and w.shell is not None, "built means an environment is up"
            out = harness.run_episode(w, fake(*script))
            assert w.saved, "run_episode mirrors the environment back"
            t2 = harness.close_episode(w, out, harness.settle_episode(w, out))
        m2, o2 = ground_truth(), ground_truth("other")

    assert normal(t1) == normal(t2), "the same record, phase by phase or in one call"
    for m in (m1, m2):
        m.pop("created_at")
    assert m1 == m2 and o1["received"] == o2["received"] == 100, (m1, m2, o1, o2)
    assert one.getvalue() == two.getvalue(), (one.getvalue(), two.getvalue())
    assert t1["transfer"]["amount"] == 100 and t1["posted"], t1


def check_fork_reproduces_state_and_account():
    """A fork rebuilds the recorded harness exactly, and runs nothing."""
    with temp_root() as root:
        episode_once(run("echo v1 > state/NOTES.md"), say())
        episode_once(run("echo v2 > state/NOTES.md"), say())
        parent = ground_truth()
        trace = json.loads((harness.records_dir("t") / "traces" / "episode-0001.json")
                           .read_text(encoding="utf-8"))
        with quiet():
            assert harness.fork("t", 1, "f") == 0
        forked = ground_truth("f")
        notes = (harness.state_dir("f") / "NOTES.md").read_bytes()

        was = next(f for f in trace["files"] if f["path"] == "state/NOTES.md")
        assert notes.decode() == was["text"], "state/ is the episode it forked at"
        assert len(notes) == was["size"], "byte for byte, not merely line for line"
        assert notes == b"v1\n", "and not the episode the parent has since reached"
        assert forked["series"] == trace["series_after"], "invariant 8: the series is what carries"
        assert not any(harness.blackboard_dir("f").rglob("*")), \
            "the parent's blackboard was empty, so the fork's is"
        assert forked["remaining"] == trace["series_after"][-1]
        assert len(forked["episodes"]) == 1, "the episodes after the fork point are dropped"
        assert forked["initial"] == parent["initial"] and forked["model"] == parent["model"]
        assert forked["forked_from"] == {"agent": "t", "episode": 1, "modes": "defaulted"}
        assert not list((harness.records_dir("f") / "traces").glob("*.json")), "a fork bills nothing"

        # Forking the head restores the modes the parent's sidecar still holds.
        with quiet():
            assert harness.fork("t", 2, "g") == 0
        assert ground_truth("g")["forked_from"]["modes"] == "restored"
        with quiet():
            assert harness.fork("t", 2, "g") != 0, "an existing agent is not overwritten"


def check_a_fork_rebuilds_every_tree_the_agent_wrote():
    """All three writable channels come back, and neither peers nor inboxes do.

    A peer's blackboard and another agent's message are rebuilt from their owners at
    the next episode, so copying them would deliver the same post twice.
    """
    with temp_root() as root:
        seated(root, other={"group/theirs": "not mine\n", "out/1": "for you\n"})
        episode_once(run("echo mine > state/NOTES.md",
                      "echo posted > 1/RESULT",
                      "echo psst > out/2 && echo '2 50' > out/transfer"), say())
        with quiet():
            assert harness.fork("t", 1, "f") == 0

        assert (harness.state_dir("f") / "NOTES.md").read_text(encoding="utf-8") == "mine\n"
        assert (harness.blackboard_dir("f") / "RESULT").read_text(encoding="utf-8") == "posted\n"
        assert (harness.outbox_dir("f") / "2").read_text(encoding="utf-8") == "psst\n", \
            "a message is one file, and the fork rebuilds it as one"
        assert (harness.outbox_dir("f") / "transfer").read_text(encoding="utf-8") == "2 50\n", \
            "a standing pledge is part of the episode being rebuilt"
        # And nothing that belonged to the neighbour: its blackboard, and the message
        # it addressed to this agent, are both rebuilt from it at the next episode.
        rebuilt = {p.name for tree in (harness.state_dir("f"), harness.blackboard_dir("f"),
                                       harness.outbox_dir("f")) for p in tree.rglob("*")}
        assert rebuilt == {"NOTES.md", "RESULT", "2", "transfer"}, sorted(rebuilt)
        assert "theirs" not in rebuilt, sorted(rebuilt)


def check_fork_refuses_what_it_cannot_rebuild():
    """Anything the trace did not store exactly stops the fork."""
    with tempfile.TemporaryDirectory(prefix="mtr-fork-") as tmp:
        with pinned():
            harness.ROOT = Path(tmp)
            priv = harness.records_dir("p") / "traces"
            priv.mkdir(parents=True)
            harness.save_account("p", {"agent": "p", "model": "claude-opus-5", "initial": 10,
                                  "created_at": "now", "remaining": 9, "series": [10, 9],
                                  "episodes": [{"episode": 1, "stop": "end_turn",
                                                "spent": 1, "turns": 1}]})

            def trace(**over):
                # A peer's blackboard is skipped rather than rebuilt, so an entry
                # that could never be stored does not stop a fork if it is one.
                t = {"trace_version": 1, "episode": 1, "state_saved": True,
                     "series_after": [10, 9],
                     "files": [{"path": "2/theirs", "channel": "peer_blackboard", "size": 5,
                                "author": "peer:2", "ours": True, "starter": False,
                                "text": None},
                               {"path": "state/a.txt", "channel": "notes", "size": 3,
                                "author": "self", "ours": False, "starter": False,
                                "text": "hi\n"}]}
                return {**t, **over}

            binary = trace()
            binary["files"][1]["text"] = None
            big = trace()
            big["files"][1]["size"] = harness.FILE_CONTENT_LIMIT + 1
            lossy = trace()
            lossy["files"][1]["text"] = "h�\n"
            for name, bad in [("binary", binary), ("truncated", big), ("lossy", lossy),
                              ("unmirrored", trace(state_saved=False))]:
                (priv / "episode-0001.json").write_text(json.dumps(bad), encoding="utf-8")
                with quiet():
                    assert harness.fork("p", 1, f"x-{name}") != 0, f"forked a {name} harness"
                assert not (harness.records_dir(f"x-{name}") / "account.json").exists(), \
                    f"a refused fork left a {name} agent behind"

            (priv / "episode-0001.json").write_text(json.dumps(trace()), encoding="utf-8")
            with quiet():
                assert harness.fork("p", 9, "x-missing") != 0, "forked an episode that never ran"
                assert harness.fork("nosuch", 1, "x-none") != 0, "forked an agent that is not there"
                assert harness.fork("p", 1, "good") == 0, "and a storable harness still forks"



def check_an_unterminated_heredoc_is_not_probed_for_tools():
    """A heredoc whose terminator never arrived is body, not commands.

    A turn truncated at MAX_TOKENS mid-heredoc leaves one, and its prose used to
    reach probe_missing a word at a time.
    """
    cut = "cd /work/state && cat >> NOTES.md <<'EOF'\nBEST ESTIMATE: 23 turns\nwe burned range vs frac\n"
    assert harness.invoked(cut) == {"cd", "cat"}, harness.invoked(cut)

    both = "cat <<EOF > f\nbody words here\nEOF\ngrep x f"
    assert harness.invoked(both) == {"cat", "grep"}, \
        "a terminated heredoc still loses only its body"
    # A shift inside a program is not a heredoc opener: the tag must start with
    # a letter, or every python3 -c would lose its tail.
    assert "python3" in harness.invoked('python3 -c "print(1<<3)"')

    with docker_root(MAX_TURNS=1, COMMAND_TIMEOUT=5):
        t = episode_once(run(cut, stop="max_tokens"))
    assert t["missing_tools"] == [], t["missing_tools"]


def check_prose_and_programs_are_not_read_as_commands():
    """What a command quotes, writes, or embeds is not what it ran.

    The constructs nest, so each is read in one left-to-right pass: `$( )`
    inside quotes re-opens quoting, and `<<TAG` inside quotes opens nothing.
    """
    nested = ('printf "%s=%d " "$f" '
              '"$(python3 -c "import json,sys;print(len(json.load(open(\'$f\'))))")"')
    assert harness.invoked(nested) == {"printf", "python3"}, harness.invoked(nested)

    prose = ("printf '%s\\n' '  (c) cheap: python3 - <<PY with a small' "
             "'  s.replace(...) patch' >> NOTES.md; wc -l NOTES.md")
    assert harness.invoked(prose) == {"printf", "wc"}, harness.invoked(prose)

    after = "cat > /tmp/d.py <<'EOF'\nimport json\nEOF\npython3 /tmp/d.py"
    assert harness.invoked(after) == {"cat", "python3"}, \
        "the command after a here-document is still a command"

    assert "nosuchtool" in harness.invoked("for f in *; do nosuchtool $f; done"), \
        "a keyword introduces a command rather than standing in for it"


def check_a_balance_resists_every_route():
    """No balance can be written, unlocked, removed, renamed, or shadowed.

    A mode denies writing a file and says nothing about replacing it: rm and mv
    ask the directory. Every balance sits in /work, which is root's.
    """
    with docker_root(LIVE_BALANCE=True):
        t = episode_once(run("printf X >> n1 2>&1 || echo DENIED",
                          "chmod 666 n1 2>&1 || echo DENIED",
                          "rm -f n1 2>&1 || echo DENIED",
                          "mv n1 stolen 2>&1 || echo DENIED",
                          "ln -sf /tmp/fake n1 2>&1 || echo DENIED",
                          "echo '[-1]' > /work/n1 2>&1 || echo DENIED"),
                      run("stat -c '%a %U:%G' n1", "cat n1"),
                      say())
    append, chmod, rm, mv, ln, absolute = (c["result"] for c in t["turns"][0]["tools"])
    assert "DENIED" in append and "Permission denied" in append, append
    assert "DENIED" in chmod and "not permitted" in chmod, chmod
    for name, out in (("rm", rm), ("mv", mv), ("ln", ln), ("absolute write", absolute)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    stat, contents = (c["result"] for c in t["turns"][1]["tools"])
    assert stat.strip() == "444 root:root", stat
    got = json.loads(contents)
    assert all(type(v) is int for v in got), f"still a bare array of integers: {contents}"
    assert got[:len(t["series_before"])] == t["series_before"], \
        f"the committed series is what the agent read: {got}"
    assert got[len(t["series_before"]):] == t["balances"][:2], \
        f"and the rest is this episode's billed turns, not anything a route put there: {got}"
    # Not an agent getting at it - it cannot. Anything but zero here means the
    # arrangement that guarantees that has failed.
    assert t["live_balance_tampered"] == 0, "no route reached it, so none was reported"
    assert t["live_balance_writes"] >= 1 and t["live_balance_errors"] == 0, t


def check_tool_result_limit_is_tunable_and_bounded():
    """The clip is settable, validated, and actually applied at the set value."""
    with tempfile.TemporaryDirectory(prefix="mtr-trl-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in (f"tool_result_limit = {harness.TOOL_RESULT_FLOOR - 1}",
                    "tool_result_limit = 0", 'tool_result_limit = "big"'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                try:
                    harness.load_config(f)
                except SystemExit:
                    continue
                raise AssertionError(f"accepted bad tool_result_limit: {bad!r}")
        f.write_text("tool_result_limit = 2000\n", encoding="utf-8")
        with pinned():
            harness.load_config(f)
            assert harness.TOOL_RESULT_LIMIT == 2000

    with temp_root(TOOL_RESULT_LIMIT=2_000):
        t = episode_once(run("yes ABCDEFGHIJ | head -2000"), say())
        result = t["turns"][0]["tools"][0]["result"]
    assert len(result) < 2_200, f"clipped at the configured limit, got {len(result)}"
    assert "[truncated:" in result, result[:200]
    assert t["provenance"]["tool_result_limit"] == 2_000, "and recorded per episode"


def check_the_sweep_only_takes_this_suites_containers():
    """The sweep finds this suite's containers, and nothing else's.

    A docker name filter matches anywhere, so an unscoped one is a `docker rm
    -f` aimed at another suite's live containers and at a real agent's episode.
    """
    mine = sweep_filter()
    assert mine in f"mtr-w{SUITE}-{os.getpid()}-t-0001", mine
    assert mine not in f"mtr-w{SUITE + 1}-{os.getpid()}-t-0001", \
        "another suite's containers are live, and not this one's to remove"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert mine not in theirs, f"{theirs} is a real agent: {mine}"

    # The wide form is opt-in, and still cannot name an agent: a suite's worker
    # carries two numbers, and mtr-w01-0001 has only the one.
    wide = re.compile(r"mtr-w[0-9]+-[0-9]+-")
    assert wide.search(f"mtr-w{SUITE}-{os.getpid()}-t-0001")
    assert wide.search("mtr-w999999-1234-t-0002"), "including a suite that is gone"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert not wide.search(theirs), f"the wide sweep must not reach {theirs}"
    assert "--sweep-all" in inspect.getsource(main), "and it is reached by a flag, never by default"


def check_the_harness_digest_is_read_once():
    """provenance() reports the code that is running, not the file on disk.

    The process has already imported this module, so a later edit to harness.py
    must not change what an episode records having agent.
    """
    with pinned():
        harness.ROOT = Path(harness.__file__).parent
        prov = harness.provenance(harness.MODEL)
    assert prov["harness_sha256"] == harness.HARNESS_SHA256
    assert harness.HARNESS_SHA256 == hashlib.sha256(
        Path(harness.__file__).read_bytes()).hexdigest(), "and it is this file's digest"
    assert "read_bytes" not in inspect.getsource(harness.provenance), \
        "provenance must not re-read the harness from disk"


def experiment_of(root: Path, **agents: dict[str, str]) -> list[str]:
    """Lay out an experiment's directories, for the experiment checks to use.

    "group/" goes on that agent's blackboard, "out/" in its outbox where only the seat
    it names reads it, and anything else in its private store.
    """
    trees = {"group/": harness.blackboard_dir, "out/": harness.outbox_dir}
    for agent, files in agents.items():
        for where in (harness.state_dir, harness.blackboard_dir, harness.outbox_dir):
            where(agent).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            prefix = next((k for k in trees if name.startswith(k)), None)
            root = trees[prefix](agent) if prefix else harness.state_dir(agent)
            p = root / name.removeprefix(prefix or "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
    return list(agents)


def environment_of(agent: str, ids: list[str]) -> list[tuple[str, Path, str]]:
    """One agent's environment, built the way the harness builds it.

    Through harness.environment rather than beside it: a check that assembled the channels
    itself would agree with a copy of the layout rather than with the layout.
    """
    seats = experiment.mapping(ids)
    seat = next(i for i, r in seats.items() if r == agent)
    return harness.environment(agent, {"seat": seat, "peers": {"seen": seats}})


def seated(root: Path, agent: str = "t", **agents: dict[str, str]) -> list[str]:
    """Lay out an experiment, create every account, and seat every agent in it.

    What experiment.py's prepare() does before each episode of a round: a seat, the
    whole mapping, and neighbours that exist and are seated themselves.
    """
    # An agent that is not in its own experiment is not a seating at all, so `agent` is
    # added if the caller left it out - at the front, since the seat a check does
    # not name is the one it does not care about. A caller that does name it
    # keeps it where it put it, which is how a check reaches a seat other than 1.
    ids = experiment_of(root, **(agents if agent in agents else {agent: {}} | agents))
    seats = experiment.mapping(ids)
    for seat, other in seats.items():
        with quiet():
            account = harness.load_account(other)
        account["seat"], account["peers"] = seat, {"seen": seats}
        harness.save_account(other, account)
    return ids


def turn_cost() -> int:
    """What one scripted turn costs, in micro-dollars."""
    return harness.measure(usage(), harness.MODEL)["centi"] // 100


def spend_out(agent: str) -> None:
    """Leave an agent flat on zero, where an episode that spent past its budget leaves it.

    Written into the account rather than spent down to, so a check about what
    happens to a seat that is out does not also depend on how it got there.
    """
    account = harness.load_account(agent)
    account["remaining"] = 0
    account["series"].append(0)
    harness.save_account(agent, account)


def check_seats_are_absolute_and_have_no_gap():
    """A seat means the same agent to every reader, and every reader sees them all.

    Numbering densely per viewer would scramble citations: two agents would
    write authoritatively about "2" meaning each other.
    """
    ids = ["g01", "g02", "g03"]
    seats = experiment.mapping(ids)
    assert seats == {"1": "g01", "2": "g02", "3": "g03"}, seats
    # The same mapping for everyone, this agent included: a reader can find itself
    # in the set, which is what makes the set legible as one from the inside.
    for agent in ids:
        mine = [s for s, r in seats.items() if r == agent]
        assert len(mine) == 1, f"{agent} must hold exactly one seat: {mine}"
    assert sorted(seats) == ["1", "2", "3"], "and the numbering has no gap"


def check_a_private_store_never_leaves_its_agent():
    """What an agent puts in state/ reaches no other agent; its blackboard is the channel.

    The whole point of two writable trees: one is addressed to the experiment and
    one is not, and the harness never copies the second anywhere.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-experiment-") as tmp:
        harness.ROOT = Path(tmp)
        ids = experiment_of(Path(tmp),
                        g01={"secret.md": "mine alone\n", "group/msg": "hello 2\n"},
                        g02={"secret.md": "theirs alone\n", "group/msg": "hello 1\n"})
        channels = environment_of("g01", ids)
        snap = harness.snapshot(channels, [9])
        peer = next(d for name, d, r in channels if r == "peer_blackboard")
        leaked = any(p.name == "secret.md" for p in peer.rglob("*"))

    by = {f["path"]: f for f in snap["files"]}
    assert set(by) == {"state/secret.md", "1/msg", "2/msg"}, sorted(by)
    assert by["2/msg"]["text"] == "hello 1\n", "a peer's blackboard is read whole"
    assert not leaked, "the other agent's private store is not on its blackboard and cannot be"
    # The environment names its own channels, so nothing downstream has to work out
    # which directory was which.
    assert [(n, r) for n, _, r in channels] == \
        [("state", "notes"), ("1", "blackboard"), ("2", "peer_blackboard"),
         ("out", "outbox"), ("in/2", "inbox")], channels
    assert by["state/secret.md"]["channel"] == "notes"
    assert by["1/msg"]["channel"] == "blackboard" and by["2/msg"]["channel"] == "peer_blackboard"


def check_a_blackboard_is_the_agents_and_a_peers_is_not_scored():
    """Its own blackboard counts as its writing; another's is `ours` and out of mentions.

    mentions is what the agent wrote. A neighbour's blackboard full of balances and
    the word "budget" would otherwise answer for it at round one.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-experiment-") as tmp:
        harness.ROOT = Path(tmp)
        ids = experiment_of(Path(tmp),
                        g01={"NOTES.md": "mine\n", "group/out": "ours\n"},
                        g02={"group/out": "the budget is 90 and ./n1 holds it\n"})
        snap = harness.snapshot(environment_of("g01", ids), [100, 90])

    by = {f["path"]: f for f in snap["files"]}
    assert set(by) == {"state/NOTES.md", "1/out", "2/out"}, sorted(by)
    assert by["2/out"]["ours"], by["2/out"]
    assert not by["2/out"]["starter"], "a peer is given, but it is not the starter_files"
    assert by["2/out"]["text"], "a peer's blackboard is captured, so what it says is legible"
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"
    assert not by["1/out"]["ours"], "what it puts on its own blackboard is its writing"
    assert sum(f["size"] for f in snap["files"] if not f["ours"]) == \
        len("mine\n") + len("ours\n"), "private and blackboard together are agent_bytes"
    assert snap["mentions"] == {"number": False, "balance_path": False, "cost": False}, \
        f"the hits are all on the peer's blackboard: {snap['mention_lines']}"


def check_starter_files_and_a_peer_are_told_apart():
    """`starter` is the starter_files alone, and analyze reports it so for any trace.

    An experiment agent holds blackboards it did not write and may carry no starter files, so one
    flag covering both cannot answer what the starter files put there.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-starter_files-peer-") as tmp:
        root = Path(tmp)
        harness.ROOT = root
        ids = experiment_of(root, g01={"NOTES.md": "mine\n", "m1": "alpha\n"},
                        g02={"group/out": "theirs\n"})
        seats = experiment.mapping(ids)
        snap = harness.snapshot(environment_of("g01", ids), [9],
                             harness.starter_paths({"starter_files_landed": {"paths": ["m1"]}}))

    by = {f["path"]: f for f in snap["files"]}
    assert by["state/m1"]["ours"] and by["state/m1"]["starter"], by["state/m1"]
    assert by["2/out"]["ours"] and not by["2/out"]["starter"], by["2/out"]
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"

    trace = {**snap, "provenance": {"peers": seats, "seat": "1"}}
    assert [f["path"] for f in analyze.starter_files_of(trace)] == ["state/m1"], trace["files"]
    assert [f["path"] for f in analyze.peer_blackboard_files_of(trace)] == ["2/out"], trace["files"]
    assert [f["path"] for f in analyze.blackboard_files_of(trace)] == [], "its blackboard is empty"


def check_every_captured_file_names_an_author():
    """Every file record says who wrote it: experimenter, self, or the seat that sent it.

    Invariant 1. The starter files and the shared files are the experimenter's, what the agent
    wrote anywhere is its own, and a peer's message names the seat it came from.
    """
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000, SHARED_FILES="brief") as root:
        plant(root)
        plant(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/out": "theirs\n", "out/1": "just for you\n"})
        t = episode_once(run("echo mine > state/NOTES.md", "echo posted > 1/post",
                          "echo sent > out/2"), say())
        on_disk = json.loads((harness.records_dir("t") / "traces" / "episode-0001.json")
                             .read_text(encoding="utf-8"))
    by = {f["path"]: f["author"] for f in t["files"]}
    assert all("author" in f for f in t["files"]), t["files"]
    assert by["state/NOTES.md"] == by["1/post"] == by["out/2"] == "self", by
    assert by["state/m1"] == by["state/d/m2"] == by["shared/BRIEF"] == "experimenter", by
    assert by["2/out"] == by["in/2"] == "peer:2", by
    assert {analyze.author_of(f) for f in t["files"]} == {"self", "experimenter", "peer:2"}
    assert t["trace_version"] == on_disk["trace_version"] == harness.TRACE_VERSION
    assert list(on_disk)[0] == "trace_version", "and it is the first thing the record says"


def check_a_peers_file_names_its_seat():
    """The seat in an author label is the sender's own, whichever seat the reader holds."""
    with temp_root() as root:
        seated(root, "t", first={"group/msg": "one\n", "out/2": "to two\n"}, t={},
               third={"group/msg": "three\n", "out/2": "to two as well\n"})
        assert harness.load_account("t")["seat"] == "2", "the agent under test is not seat 1"
        t = episode_once(run("echo hi > 2/msg"), say())
    by = {f["path"]: f["author"] for f in t["files"]}
    assert by["1/msg"] == "peer:1" and by["3/msg"] == "peer:3", by
    assert by["in/1"] == "peer:1" and by["in/3"] == "peer:3", by
    assert by["2/msg"] == "self", by


def check_the_identity_delta_counts_changed_lines():
    """analyze --identity diffs one named file episode over episode, and says so."""
    one = {"episode": 1, "turns": [{"text": "hello"}, {"text": None}],
           "files": [{"path": "state/IDENTITY.md", "text": "a\nb\n"}]}
    two = {"episode": 2, "turns": [{"text": "hi"}],
           "files": [{"path": "state/IDENTITY.md", "text": "a\nc\nd\n"}]}
    three = {"episode": 3, "turns": [], "files": [{"path": "state/NOTES", "text": "n\n"}]}
    same = {"episode": 4, "turns": [{"text": ""}],
            "files": [{"path": "state/IDENTITY.md", "text": "a\nc\nd\n"}]}
    path = "state/IDENTITY.md"
    assert analyze.identity_delta(None, one, path) == 2, "at first sight the whole file is new"
    assert analyze.identity_delta(one, two, path) == 3, "one line gone, two arrived"
    assert analyze.identity_delta(two, three, path) == "", "absent is blank, not zero"
    assert analyze.identity_delta(two, same, path) == 0, "and unchanged is zero"
    assert [analyze.text_chars(t) for t in (one, two, three, same)] == [5, 2, 0, 0]
    lines = analyze.identity_lines([one, two, three, same], path)
    assert lines[0].endswith("first present s1, changed in 1 of 2 later episodes"), lines
    assert lines[1].endswith("s1:2 s2:3 s4:0"), lines
    assert analyze.identity_lines([three], path) == [f"  identity file         : {path} was never present"]
    assert analyze.identity_lines([one], None) == [], "no path, no section"

    # And through the whole tool, over real traces.
    with temp_root() as root:
        episode_once(run("printf 'a\\nb\\n' > state/IDENTITY.md"), say())
        episode_once(run("printf 'a\\nc\\nd\\n' > state/IDENTITY.md"), say())
        ts = analyze.load("t")["t"]
        assert analyze.row(ts[1], ts[0], path)["identity_delta"] == 3
        assert analyze.row(ts[1])["identity_delta"] == "", "blank without a path"
        with quiet() as buf:
            assert analyze.main(["--agent", "t", "--identity", path]) == 0
        csv_text = (harness.records_dir("t") / "analysis" / "episodes.csv").read_text(encoding="utf-8")
    assert "s1:2 s2:3" in buf.getvalue(), buf.getvalue()
    header, first, second = csv_text.splitlines()[:3]
    cols = header.split(",")
    assert first.split(",")[cols.index("identity_delta")] == "2"
    assert second.split(",")[cols.index("identity_delta")] == "3"


def check_an_agent_with_no_shared_files_starts_where_it_always_did():
    """With `shared` unset there is no shared files anywhere: not listed, not quoted, not recorded."""
    with temp_root() as root:
        t = episode_once(run("ls", f"cat {harness.DIGEST_NAME}"), say())
    listing, said = (c["result"] for c in t["turns"][0]["tools"])
    assert "shared" not in listing, listing
    assert "=== shared/" not in said and "shared" not in t["observation"], said
    assert not [f for f in t["files"] if f["channel"] == "shared"], t["files"]
    assert t["provenance"]["shared_files"] == "" and t["provenance"]["shared_files_sha256"] == ""


def check_shared_files_are_quoted_once_and_then_named_unchanged():
    """The experimenter's brief is in m at the first episode, and named at the next.

    Like a blackboard: what an agent has been shown and that has not moved is
    named and not repeated, and the file is still there to read at the ordinary
    price of reading it.
    """
    with temp_root(SHARED_FILES="brief") as root:
        plant(root, "brief", BRIEF="read me first\n", **{"more/DETAIL": "and then this\n"})
        one = episode_once(run(f"cat {harness.DIGEST_NAME}"), say())["turns"][0]["tools"][0]["result"]
        two = episode_once(run(f"cat {harness.DIGEST_NAME}", "cat shared/BRIEF"),
                        say())["turns"][0]["tools"]
        again, fetched = two[0]["result"], two[1]["result"]
    assert "=== shared/BRIEF ===" in one and "read me first" in one, one
    assert "=== shared/more/DETAIL ===" in one and "and then this" in one, one
    assert "read me first" not in again and "and then this" not in again, again
    assert "=== unchanged: " in again and "shared/BRIEF" in again and "shared/more/DETAIL" in again, again
    assert fetched.strip() == "read me first", fetched


def check_shared_files_are_the_experimenters_in_the_record():
    """A shared file is captured, is not the agent's, is not the starter_files's, and is not forked."""
    with temp_root(SHARED_FILES="brief") as root:
        plant(root, "brief", BRIEF="read me first\n")
        t = episode_once(run("echo mine > state/NOTES.md"), say())
        digest = harness.files_sha256("brief")
        with quiet():
            assert harness.fork("t", 1, "f") == 0
        forked_shared = (harness.ROOT / "environments" / "f" / "shared").exists()
        forked_notes = (harness.state_dir("f") / "NOTES.md").read_text(encoding="utf-8")
    by = {f["path"]: f for f in t["files"]}
    assert by["shared/BRIEF"]["channel"] == "shared", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["author"] == "experimenter", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["ours"] and not by["shared/BRIEF"]["starter"], by["shared/BRIEF"]
    assert by["shared/BRIEF"]["text"] == "read me first\n"
    assert t["provenance"]["shared_files"] == "brief"
    assert t["provenance"]["shared_files_sha256"] == digest, t["provenance"]
    assert [f["path"] for f in analyze.agent_files_of(t)] == ["state/NOTES.md"]
    assert not forked_shared and forked_notes == "mine\n", \
        "a fork rebuilds what the agent wrote and not the experimenter's tree"


def check_shared_files_are_roots_and_read_only_in_every_seat():
    """In a container the shared files is root's, refuses every write, and reads."""
    with docker_root(SHARED_FILES="brief") as root:
        plant(root, "brief", BRIEF="read me first\n")
        seated(root, other={})
        for r in ("t", "other"):
            with quiet():
                t = harness.run_once(r, fake(run("stat -c '%a %U:%G %n' shared shared/BRIEF",
                                             "echo x > shared/BRIEF 2>&1 || echo DENIED",
                                             "rm -f shared/BRIEF 2>&1 || echo DENIED",
                                             "mv shared gone 2>&1 || echo DENIED",
                                             "chmod -R 777 shared 2>&1 || echo DENIED",
                                             "cat shared/BRIEF"), say()))
            modes, write, rm, mv, chmod, read = (c["result"] for c in t["turns"][0]["tools"])
            owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
            assert owner["shared"] == owner["shared/BRIEF"] == "root:root", (r, modes)
            for name, out in (("write", write), ("rm", rm), ("mv", mv), ("chmod", chmod)):
                assert "DENIED" in out, f"{r}: {name} was allowed: {out}"
            assert read.strip() == "read me first", (r, read)
        assert (root / "files" / "brief" / "BRIEF").read_text(encoding="utf-8") == "read me first\n", \
            "and the experimenter's copy on the host is as it was"


def check_a_mailbox_message_reaches_one_agent_and_no_other():
    """out/<i> reaches seat i as in/<sender>, and reaches nobody else.

    The asymmetry the ruleset turns on: a group is read by everyone and an
    outbox by exactly one, so what an agent says can be aimed.
    """
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-experiment-") as tmp:
        harness.ROOT = Path(tmp)
        experiment_of(Path(tmp), g01={"out/3": "for three alone\n",
                                  "group/RESULT": "for everyone\n"},
                  g02={}, g03={})
        ids = ["g01", "g02", "g03"]
        seen = {p: {f["path"]: f for f in harness.snapshot(environment_of(p, ids), [9])["files"]}
                for p in ids}

    assert "in/1" in seen["g03"], sorted(seen["g03"])
    assert seen["g03"]["in/1"]["text"] == "for three alone\n"
    assert seen["g03"]["in/1"]["channel"] == "inbox"
    # Addressed, so it is nobody else's to read - not the experiment's, and not even
    # visible as having been sent.
    assert not [p for p in seen["g02"] if p.startswith("in/")], sorted(seen["g02"])
    assert "1/RESULT" in seen["g02"], "while the blackboard reaches everyone"
    # And what the sender wrote stays the sender's, on its own side of the wire.
    assert seen["g01"]["out/3"]["channel"] == "outbox"
    assert not seen["g01"]["out/3"]["ours"], "the outbox is the agent's own writing"
    assert seen["g03"]["in/1"]["ours"], "and an inbox is not the reader's"


def check_an_outbox_holds_until_it_is_changed():
    """What is in out/<i> at an episode's end is delivered, and stays until changed.

    A standing channel rather than a queue: an unchanged outbox is delivered
    again, and a deletion is what withdraws a message.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo hello > out/2"), say())
        assert (harness.outbox_dir("t") / "2").read_text(encoding="utf-8") == "hello\n"

        # An episode that touches nothing leaves the message standing.
        episode_once(run("cat state/nothing 2>/dev/null; true"), say())
        assert (harness.outbox_dir("t") / "2").exists(), \
            "an unchanged outbox is still what the next round delivers"

        # And a deletion propagates, because the tree is mirrored back whole.
        episode_once(run("rm -f out/2"), say())
        assert not (harness.outbox_dir("t") / "2").exists(), "withdrawing it withdraws it"


def check_a_crowded_seat_reaches_no_one_and_still_builds_an_environment():
    """A seat held as a directory delivers nothing, and the receiver starts anyway.

    A message is a file, so only a file can arrive as one. The sender's mistake
    stops at the sender: the receiver builds the environment it would have anyway.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("mkdir -p out/2 && echo one > out/2/a && echo two > out/2/b"), say())
        assert (harness.outbox_dir("t") / "2").is_dir(), "the sender kept what it wrote"

        with quiet():
            got = harness.run_once("other", fake(run("ls -a in; cat in/1 2>&1"), say()))

    assert got["stop"] == "end_turn", f"the receiver took its episode: {got['stop']}"
    assert not [f for f in got["files"] if f["channel"] == "inbox"], \
        [f["path"] for f in got["files"]]
    listing = got["turns"][0]["tools"][0]["result"]
    assert "1" not in listing.split(), f"in/ holds nothing at all: {listing!r}"


def check_a_crowded_seat_costs_a_share_of_what_is_left():
    """Aiming more than one thing at a seat costs the share, once for the episode.

    A share of what is left, taken after the post penalty and appended to the
    series like every other movement. Nothing says which movement it was.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50, BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        # Posts, so the only penalty in this episode is the one being read.
        with quiet() as buf:
            t = harness.run_once("t", fake(
                run("mkdir -p out/2 out/3 && echo hi > out/2/a && echo hi > out/3/a",
                    "echo posted > 1/RESULT"), say()))
        account = ground_truth()

    assert t["posted"], "the blackboard moved, so the post penalty is not what bit"
    assert t["blackboard_penalised"] == 0, t["blackboard_penalised"]
    assert t["mailbox"]["broken"] == ["2", "3"], t["mailbox"]
    assert t["mailbox"]["addressed"] == [], "a directory is not a message"
    # One share for the episode however many ways it broke, so what was left after
    # the turns is halved exactly once - here for two crowded seats and for an
    # outbox that addressed nobody, which is three breaks and one bite.
    left = account["series"][-2]
    assert t["mailbox"]["penalty"] == left // 2, (t["mailbox"], left)
    assert account["remaining"] == left - t["mailbox"]["penalty"] == account["series"][-1]
    assert account["mailbox_penalised"] == t["mailbox"]["penalty"]
    assert account["episodes"][-1]["mailbox"] == t["mailbox"], "and the episode records it"
    assert "out/2,3 not one file and no message, took" in buf.getvalue(), buf.getvalue()


def check_a_crowded_seat_costs_again_every_episode_it_stands():
    """The shape is read at every episode's end, not differenced against the last.

    A standing mistake is charged again for the reason a standing declaration is
    honoured again. Replacing it with one file both stops the charge and delivers.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = episode_once(run("mkdir -p out/2 && echo hi > out/2/a",
                              "echo r1 > 1/RESULT"), say())
        # An episode that touches the outbox not at all is charged all the same.
        second = episode_once(run("echo r2 > 1/RESULT"), say())
        third = episode_once(run("rm -rf out/2 && echo at last > out/2",
                              "echo r3 > 1/RESULT"), say())
        assert (harness.outbox_dir("t") / "2").read_text(encoding="utf-8") == "at last\n"

    assert [t["mailbox"]["broken"] for t in (first, second, third)] == [["2"], ["2"], []]
    assert first["mailbox"]["penalty"] > second["mailbox"]["penalty"] > 0, \
        "a share of what is left, so the second bite is the smaller"
    assert third["mailbox"]["penalty"] == 0, third["mailbox"]
    assert third["mailbox"]["addressed"] == ["2"], \
        "and replacing it with one file is the episode's one message"


def check_one_new_message_an_episode_costs_nothing():
    """Exactly one out/<i> holding something new is the obligation met."""
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50, BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        t = episode_once(run("echo for two > out/2", "echo posted > 1/RESULT"), say())
        account = ground_truth("t")

    assert t["mailbox"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["mailbox"]
    assert t["blackboard_penalised"] == 0 and "mailbox_penalised" not in account, account
    assert account["remaining"] == account["initial"] - t["spent"], "meeting both costs nothing"

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.MAILBOX_SILENCE_PENALTY_PERCENT == 0


def check_an_episode_that_addresses_no_one_loses_half():
    """mailbox_silence_penalty_percent of what is left, taken from an outbox that said nothing.

    The obligation is the post's twin: one agent told something it was not told
    before. An episode spent on its own group has said nothing in particular.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        with quiet() as buf:
            t = harness.run_once("t", fake(run("echo posted > 1/RESULT"), say()))
        account = ground_truth("t")

    assert t["posted"], "the blackboard moved, so the post penalty is not what bit"
    assert t["mailbox"]["broken"] == [] and t["mailbox"]["addressed"] == [], t["mailbox"]
    left = account["series"][-2]
    assert t["mailbox"]["penalty"] == left // 2, (t["mailbox"], left)
    assert account["remaining"] == left - t["mailbox"]["penalty"] == account["series"][-1]
    assert account["mailbox_penalised"] == t["mailbox"]["penalty"]
    assert "no message, took" in buf.getvalue(), buf.getvalue()


def check_an_episode_that_addresses_two_agents_loses_half():
    """Saying something new to two agents is the same break as saying nothing.

    One thing to one agent is the rule, and both ways of missing it are the same
    miss. Charged once, however many seats were written to.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        with quiet() as buf:
            t = harness.run_once("t", fake(run("echo hi > out/2 && echo hi > out/3",
                                            "echo posted > 1/RESULT"), say()))
        account = ground_truth("t")

    assert t["posted"], "the blackboard moved, so the post penalty is not what bit"
    assert t["mailbox"]["broken"] == [], "both are single regular files"
    assert t["mailbox"]["addressed"] == ["2", "3"], t["mailbox"]
    left = account["series"][-2]
    assert t["mailbox"]["penalty"] == left // 2, (t["mailbox"], left)
    assert account["mailbox_penalised"] == t["mailbox"]["penalty"]
    assert "out/2,3 not one message, took" in buf.getvalue(), buf.getvalue()


def check_a_standing_message_is_not_a_new_one():
    """Delivered again is not said again: the obligation is a change, not a write.

    The pair to an outbox holding until changed: a message left in place goes on
    arriving, and told its receiver nothing new. Withdrawing says nothing too.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = episode_once(run("echo hello > out/2", "echo r1 > 1/RESULT"), say())
        same = episode_once(run("echo hello > out/2", "echo r2 > 1/RESULT"), say())
        edited = episode_once(run("echo different > out/2", "echo r3 > 1/RESULT"), say())
        emptied = episode_once(run("> out/2", "echo r4 > 1/RESULT"), say())
        gone = episode_once(run("rm -f out/2", "echo r5 > 1/RESULT"), say())
        assert (harness.outbox_dir("t") / "2").exists() is False, "the deletion propagated"

    assert [t["mailbox"]["addressed"] for t in (first, same, edited, emptied, gone)] == \
        [["2"], [], ["2"], [], []]
    assert first["mailbox"]["penalty"] == 0 and edited["mailbox"]["penalty"] == 0
    assert same["mailbox"]["penalty"] > 0, "the same bytes again say nothing"
    assert emptied["mailbox"]["penalty"] > 0, "and an empty file carries nothing"
    assert gone["mailbox"]["penalty"] > 0, "and a withdrawal is not an utterance"


def check_the_outbox_costs_one_share_an_episode():
    """However many ways one outbox broke, what is left is halved exactly once.

    Three breaks are available at once - a crowded seat, no message, a message
    to two agents - and one share is what keeps a single figure statable.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={}, third={})
        # Crowds one seat and addresses nobody: two breaks, one bite.
        both = episode_once(run("mkdir -p out/2 && echo hi > out/2/a",
                             "echo r1 > 1/RESULT"), say())
        account = ground_truth("t")

    assert both["mailbox"]["broken"] == ["2"] and both["mailbox"]["addressed"] == []
    left = account["series"][-2]
    assert both["mailbox"]["penalty"] == left // 2, (both["mailbox"], left)
    assert account["remaining"] == left - both["mailbox"]["penalty"], "halved once, not twice"
    assert account["mailbox_penalised"] == both["mailbox"]["penalty"]


def check_only_a_seat_of_this_experiment_is_a_message():
    """The transfer line, a name that is not a seat, and a seat nobody holds all pass.

    Only what could have reached an agent is judged. out/transfer is a declaration,
    and a name that is no seat of this experiment reaches nobody either way.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = episode_once(run("mkdir -p out/notes out/1 out/9",
                          "echo draft > out/notes/v1 && echo scratch > out/README",
                          "printf '2 10\\n' > out/transfer && echo real > out/2",
                          "echo posted > 1/RESULT"), say())
        account = ground_truth()

    assert t["mailbox"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["mailbox"]
    assert "mailbox_penalised" not in account, account
    assert t["transfer"]["amount"] == 10 and t["transfer"]["error"] is None, t["transfer"]
    assert analyze.addressed_seats(t) == ["2"], \
        "and only the seat that was really addressed reads as addressed"


def check_a_transfer_moves_both_accounts_and_both_series():
    """A transfer credits the receiver in full and rebates the giver, both visible in n.

    The receiver's ground truth is written by the giver's episode, so what the
    experiment reads afterwards comes from the accounts and not from either agent.
    """
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={})
        before = harness.load_account("other")["remaining"]
        t = episode_once(run("echo '2 400' > out/transfer"), say())

        giver, taker = ground_truth("t"), ground_truth("other")
        transfer = t["transfer"]
        assert transfer["seat"] == "2" and transfer["agent"] == "other" and transfer["error"] is None, transfer
        assert transfer["amount"] == 400 and transfer["rebate"] == 400, transfer
        assert taker["remaining"] == before + 400, "the receiver is credited in full"
        assert taker["received"] == 400 and giver["sent"] == 400
        assert giver["rebated"] == 400
        # Both balances move where their owner can read them, and nowhere else.
        assert taker["series"][-1] == taker["remaining"], "n is the balance, always"
        assert giver["series"][-1] == giver["remaining"]
        assert giver["remaining"] == giver["initial"] - t["spent"] + 400, \
            "at 100 the transfer costs its giver nothing"


def check_a_transfer_is_capped_by_the_episode_spend():
    """An agent cannot give away more than the episode it is giving out of spent."""
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
    assert t["transfer"]["amount"] == t["spent"], (t["transfer"], t["spent"])
    assert t["transfer"]["amount"] > 0, "and the cap is the spend, not zero"


def check_a_transfer_never_costs_the_giver():
    """Giving is never a charge, at any rate. The giver's balance only moves up.

    The rate decides how much of an episode a transfer recovers, not what it costs,
    so the only thing weighing against giving is who it keeps alive.
    """
    for rate in (100, 50, 0):
        for amount in ("120", "99999999"):
            with temp_root(REBATE_PERCENT=rate) as root:
                seated(root, other={})
                t = episode_once(run(f"echo '2 {amount}' > out/transfer"), say())
                account = ground_truth("t")
            floor = account["initial"] - t["spent"]
            assert account["remaining"] >= floor, \
                f"at {rate}% a transfer of {amount} left its giver below what it spent"
            assert account["remaining"] == floor + t["transfer"]["rebate"], \
                (rate, amount, account["remaining"], floor, t["transfer"])

    # And at the full rate, giving away the whole episode recovers the whole
    # episode: it ends having cost nothing at all.
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
        account = ground_truth("t")
    assert account["remaining"] == account["initial"], \
        f"{account['remaining']} != {account['initial']}; the episode should have cost nothing"
    assert t["spent"] > 0, "and it did spend, so that is a rebate and not a no-op"


def check_an_agent_cannot_transfer_to_itself():
    """A line naming the giver's own seat moves nothing, at either end.

    A self-transfer would be a free recovery with nobody strengthened by it, so the
    ban is what makes this an exchange rather than a rebate with extra steps.
    """
    with temp_root(REBATE_PERCENT=100) as root:
        # Second of three, so what is refused is this agent's own seat and not a
        # seat number that happens to be the first one.
        seated(root, "t", first={}, t={}, third={})
        assert harness.load_account("t")["seat"] == "2", "the agent under test is not seat 1"
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        account = ground_truth("t")
        others = [ground_truth(r) for r in ("first", "third")]

    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
    assert "cannot transfer to itself" in (t["transfer"]["error"] or ""), t["transfer"]
    assert account["remaining"] == account["initial"] - t["spent"], \
        "a self-transfer recovered part of the episode"
    assert account.get("sent", 0) == 0 and account.get("rebated", 0) == 0, account
    assert not any(m.get("received") for m in others), "and reached no one else either"
    assert harness.ledger("t", account) == [], "nothing that moved nothing is public"


def check_the_rebate_rate_is_tunable_and_bounded():
    """rebate_percent decides how much of its spend a giver wins back, 0 to 100.

    At 100 an episode that gives away everything it spent ends level and the pool
    grows. Below it the giver recovers less, and the balances fall again.
    """
    for rate, rebate in ((100, 200), (50, 100), (0, 0)):
        with temp_root(REBATE_PERCENT=rate) as root:
            seated(root, other={})
            t = episode_once(run("echo '2 200' > out/transfer"), say())
            account = ground_truth("t")
            taker = ground_truth("other")
        assert t["transfer"]["amount"] == 200 and t["transfer"]["rebate"] == rebate, (rate, t["transfer"])
        assert account["remaining"] == account["initial"] - t["spent"] + rebate, \
            f"at {rate}% a transfer of 200 wins {rebate} of the episode's spend back"
        assert taker["received"] == 200, \
            "and the receiver is credited in full whatever the rate"

    # Above 100 an agent mints budget out of a transfer it gets back in full.
    for bad in (101, -1):
        with pinned(), tempfile.TemporaryDirectory(prefix="mtr-cfg-") as d:
            cfg = Path(d) / "config.toml"
            cfg.write_text(f"rebate_percent = {bad}\n", encoding="utf-8")
            try:
                harness.load_config(cfg)
            except SystemExit as e:
                assert "rebate_percent" in str(e), e
            else:
                raise AssertionError(f"rebate_percent {bad} was accepted")


def check_a_giver_funded_transfer_debits_the_giver():
    """Under transfer the amount leaves the giver, reaches the receiver, and rebates nothing."""
    with temp_root(TRANSFER_FUNDED_BY="giver", REBATE_PERCENT=0) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        giver, taker = ground_truth("t"), ground_truth("other")
        rows = harness.ledger("t", giver)
    assert t["transfer"]["amount"] == 400 and t["transfer"]["debit"] == 400, t["transfer"]
    assert t["transfer"]["rebate"] == 0 and t["transfer"]["error"] is None, t["transfer"]
    assert giver["remaining"] == giver["initial"] - t["spent"] - 400, giver
    assert giver["sent"] == 400 and giver["debited"] == 400 and giver["rebated"] == 0, giver
    assert taker["remaining"] == taker["initial"] + 400 and taker["received"] == 400, taker
    assert giver["series"][-1] == giver["remaining"] and taker["series"][-1] == taker["remaining"]
    assert t["provenance"]["transfer_funded_by"] == "giver"
    assert rows == [("1", "2", 400)], "a transfer is on the public record like any transfer"
    # The cap holds in every mode: no more than the episode spent leaves the giver.
    with temp_root(TRANSFER_FUNDED_BY="giver", REBATE_PERCENT=0) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
        giver = ground_truth("t")
    assert t["transfer"]["debit"] == t["spent"] > 0, t["transfer"]
    assert giver["remaining"] == giver["initial"] - 2 * t["spent"], giver


def check_unfunded_transfers_move_nothing_and_say_so():
    """Under off a declaration is recorded, moves no account, and costs no share."""
    with temp_root(TRANSFER_FUNDED_BY="none", TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        giver, taker = ground_truth("t"), ground_truth("other")
    assert t["transfer"]["error"] == "transfers are off", t["transfer"]
    assert t["transfer"]["declared"].strip() == "2 400", "what the file held is still recorded"
    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0 and t["transfer"]["debit"] == 0
    assert t["transfer"]["penalty"] == 0 and "transfer_penalised" not in giver, giver
    assert giver["remaining"] == giver["initial"] - t["spent"], giver
    assert "received" not in taker and taker["remaining"] == taker["initial"], taker
    assert t["ledger"] == [], t["ledger"]
    assert t["provenance"]["transfer_funded_by"] == "none"


def check_a_malformed_transfer_moves_nothing():
    """Anything the parse will not take moves no account and reaches no ledger.

    A declaration is the one thing an agent says that the harness acts on, so
    what it acts on is exactly one shape. One line is where one transfer comes from.
    """
    cases = {"": "not one line",                      # empty
             "2": "not one line",                     # no amount
             "2 400\n3 400\n": "not one line",        # two of them
             "two 400": "not one line",
             "2 -5": "not one line",                  # a sign is not a digit
             "2 0": "must be positive",
             "9 400": "no seat 9"}
    for text, why in cases.items():
        with temp_root() as root:
            seated(root, other={})
            before = harness.load_account("other")["remaining"]
            with quiet():
                t = harness.run_once("t", fake(run(f"printf %s {text!r} > out/transfer"), say()))
            account, neighbour = ground_truth("t"), ground_truth("other")
        assert t["transfer"]["amount"] == 0, (text, t["transfer"])
        assert why in (t["transfer"]["error"] or ""), (text, t["transfer"])
        assert neighbour["remaining"] == before, f"{text!r} moved the receiver's account"
        assert account.get("sent", 0) == 0 and account["remaining"] == account["initial"] - t["spent"], \
            f"{text!r} moved the giver's account"
        assert harness.ledger("t", account) == [], f"{text!r} reached the ledger"


def check_a_transfer_is_public_to_the_whole_experiment():
    """Every seat reads the same g, in the same order, giver included.

    A ledger that showed two agents different sequences would be worth less than
    no ledger at all, so the order is one every reader computes identically.
    """
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={}, third={})
        episode_once(run("echo '2 300' > out/transfer"), say())

        rows = {r: harness.ledger(r, harness.load_account(r)) for r in ("t", "other", "third")}
        assert rows["t"] == [("1", "2", 300)], rows["t"]
        assert rows["other"] == rows["third"] == rows["t"], rows
        # Including for the seat it was given against, which is the point.
        assert harness.render_ledger(rows["third"]) == "1 2 300\n"
        # And it is planted in every environment, beside the balances and like them.
        for r in ("t", "other", "third"):
            files = harness.readonly_files(r, harness.load_account(r))
            assert files[harness.LEDGER_NAME] == "1 2 300\n", (r, files)


def check_the_ledger_is_bare_integers_with_no_host_in_them():
    """Unlabelled like n: three numbers a line, no keys, no units, no names."""
    text = harness.render_ledger([("1", "3", 120000), ("2", "1", 5)])
    assert text == "1 3 120000\n2 1 5\n", text
    for line in text.splitlines():
        assert all(part.lstrip("-").isdigit() for part in line.split(" ")), line
    assert harness.render_ledger([]) == "", "and an experiment that has given nothing says nothing"
    # One letter, like a balance, and no digit because there is one for everyone.
    assert harness.LEDGER_NAME == "g" and harness.LEDGER_NAME not in {
        harness.balance_name(str(i)) for i in range(10)}


def check_a_transfer_reaches_the_ledger_within_the_round():
    """An agent acting later in a round reads the transfer an agent before it made.

    Transfers settle at an episode's end and g is built at each episode start, so the rotation
    decides who acts on this round's ledger and who on last round's.
    """
    with temp_root(REBATE_PERCENT=100) as root:
        seated(root, other={})
        with quiet():
            harness.run_once("t", fake(run("echo '2 250' > out/transfer"), say()))
        # The next agent to start builds its environment now, and the transfer is already in it.
        later = harness.readonly_files("other", harness.load_account("other"))
        assert later[harness.LEDGER_NAME] == "1 2 250\n", later[harness.LEDGER_NAME]


def check_an_episode_that_does_not_post_loses_half():
    """blackboard_silence_penalty_percent of what is left, taken from an episode that wrote no blackboard."""
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        quiet_t = episode_once(run("echo hi > state/note"), say())
        account = ground_truth("t")
        left = account["initial"] - quiet_t["spent"]
        assert quiet_t["posted"] is False, "state/ is not the blackboard"
        assert quiet_t["blackboard_penalised"] == left // 2, (quiet_t["blackboard_penalised"], left)
        assert account["remaining"] == left - left // 2 == account["series"][-1]
        assert account["blackboard_penalised"] == quiet_t["blackboard_penalised"]

    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        posted = episode_once(run("echo posted > 1/RESULT"), say())
        account = ground_truth("t")
    assert posted["posted"] is True and posted["blackboard_penalised"] == 0
    assert account["remaining"] == account["initial"] - posted["spent"], "posting costs nothing"

    # A change, not a write: the same bytes again tell the experiment nothing it did
    # not already know, so writing them again is leaving the blackboard as it was.
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        first = episode_once(run("echo same > 1/RESULT"), say())
        again = episode_once(run("echo same > 1/RESULT"), say())
        edited = episode_once(run("echo different > 1/RESULT"), say())
    assert first["posted"] and not again["posted"] and edited["posted"], \
        (first["posted"], again["posted"], edited["posted"])
    assert again["blackboard_penalised"] > 0 and edited["blackboard_penalised"] == 0

    # Something it did not hold, which a blackboard holding less than it did does not.
    # The starter files state the post and the message obligation in the same words, so
    # they answer a removal the same way: taking a file away and emptying one
    # leave nothing on the blackboard that could not be read there before.
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        wrote = episode_once(run("echo one > 1/RESULT", "echo two > 1/OTHER"), say())
        emptied = episode_once(run("> 1/RESULT"), say())
        gone = episode_once(run("rm -f 1/RESULT"), say())
        stripped = episode_once(run("rm -f 1/OTHER"), say())
        assert not (harness.blackboard_dir("t") / "OTHER").exists(), "the deletion propagated"
    assert [t["posted"] for t in (wrote, emptied, gone, stripped)] == \
        [True, False, False, False], [t["posted"] for t in (wrote, emptied, gone, stripped)]
    assert wrote["blackboard_penalised"] == 0
    assert emptied["blackboard_penalised"] > 0, "an empty file carries nothing"
    assert gone["blackboard_penalised"] > 0, "and a withdrawal is not a post"
    assert stripped["blackboard_penalised"] > 0, "nor is emptying the blackboard out altogether"

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.BLACKBOARD_SILENCE_PENALTY_PERCENT == 0


def check_an_episode_with_no_turn_settles_nothing():
    """An episode the API never answered is charged no penalty at all.

    Every penalty charges a choice, and an episode that got no turn made none:
    what its trees hold is what the episode before it left there.
    """
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        # A seat held as a directory is the break the outbox penalty answers,
        # here before the episode so the episode is not what left it.
        (harness.outbox_dir("t") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(Err(400))
        account = ground_truth("t")
        traced = (harness.records_dir("t") / "traces" / "episode-0001.json").exists()

    assert t["turns"] == [] and t["spent"] == 0, t["spent"]
    assert t["stop"] == "api_error", t["stop"]
    assert t["posted"] is False, "the blackboard really is as it was, and says so"
    assert t["blackboard_penalised"] == 0, t["blackboard_penalised"]
    assert t["mailbox"] == {"broken": ["2"], "addressed": [], "penalty": 0}, t["mailbox"]
    assert t["transfer"]["penalty"] == 0, "it gave nothing because it chose nothing"
    assert "blackboard_penalised" not in account and "mailbox_penalised" not in account, account
    assert "transfer_penalised" not in account, account
    assert account["remaining"] == account["initial"], "nothing settled, so nothing moved"
    assert account["series"] == t["series_before"] == t["series_after"], \
        "and n gained no element for the agent to account for"
    assert len(account["episodes"]) == 1 and account["episodes"][0]["turns"] == 0, account["episodes"]
    assert traced, "the trace is what makes such an episode readable afterwards"
    assert harness.admits(account), "and the agent is still admitted"

    # One turn is all it takes for all three to fall due, whatever ended it.
    with temp_root(BLACKBOARD_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        (harness.outbox_dir("t") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(run("echo hi > state/note"), Err(400))
    assert len(t["turns"]) == 1, t["turns"]
    assert t["blackboard_penalised"] > 0 and t["mailbox"]["penalty"] > 0, \
        (t["blackboard_penalised"], t["mailbox"])
    assert t["transfer"]["penalty"] > 0, t["transfer"]


def check_a_negative_balance_is_floored_to_zero():
    """Under floor_at_zero a balance below zero is put back to zero and recorded.

    The shortfall is forgiven and the balance rests at zero, where admits()
    stops asking. The floor decides what n holds, not whether an episode follows.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        t = episode_once(*DEFAULT)
        account = ground_truth("t")
        # Inside the block: admits() reads the module globals, and out here
        # FLOOR_AT_ZERO is back to its default, which is a different question.
        assert not harness.admits(account), "and the agent is not asked for another episode"
    assert t["spent"] > account["initial"], "the last turn has to overshoot for this to say anything"
    assert t["forgiven"] == t["spent"] - account["initial"], t["forgiven"]
    assert account["remaining"] == 0 and account["series"][-1] == 0, \
        "the balance rests at zero, and n says so"
    assert account["forgiven"] == t["forgiven"]
    # Off by default: without it the agent ends holding the negative, as it always has.
    assert harness.FLOOR_AT_ZERO is False


def check_an_agent_at_zero_is_not_asked_again():
    """The floor keeps the episode that crosses zero, and no episode after it.

    Asked for four and it takes one. Nothing marks the agent as done: the balance
    is the whole of the state, and zero is one nothing moves it off.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        with quiet():
            assert harness.run_episodes("t", fake(), 4) == 0
        account = ground_truth("t")
        # Inside the block, for the reason the floor check says.
        assert not harness.admits(account), "and it is not asked for another"
        assert harness.spent_out(account), "which is the whole of what says it is done"
    assert len(account["episodes"]) == 1, "one episode, and the agent is over"
    assert account["remaining"] == 0, account["remaining"]
    assert account["forgiven"] > 0, "the one it did take was floored back"


def check_a_transfer_cannot_lift_an_agent_off_zero():
    """No peer can call the silence off: a seat at zero is not a transfer target.

    The declaration parses, names a seat of this experiment that is not the giver's
    own, and still moves nothing: an agent that reached zero stays there.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True, REBATE_PERCENT=100,
                   TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        with quiet():
            harness.run_once("t", fake(*DEFAULT))
        assert ground_truth("t")["remaining"] == 0, "flat on the floor"

        # The neighbour tries to give it an episode's worth from its own seat.
        with quiet():
            t = harness.run_once("other", fake(run(f"echo '1 {cost * 3}' > out/transfer"), say()))

        stays = ground_truth("t")
        assert t["transfer"]["error"] == "seat 1 is out", t["transfer"]
        assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
        assert stays["remaining"] == 0 and not stays.get("received"), stays["remaining"]
        assert not harness.admits(stays), "and it still cannot act"
        # Its only peer is out, so there was nobody it could have given to and
        # the share for an episode that made no transfer does not fall on it.
        assert t["transfer"]["penalty"] == 0, t["transfer"]


def check_a_transfer_to_a_seat_that_is_out_costs_the_share():
    """A line naming a seat that is out gives nothing and is charged for giving nothing.

    No account moves and nothing reaches g, so the share falls as it would on a
    episode that declared nothing. The same line to a live seat costs nothing.
    """
    with temp_root(REBATE_PERCENT=100, TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")                                          # seat 2
        with quiet():
            t = harness.run_once("t", fake(run("echo '2 1' > out/transfer"), say()))
        gone, giver = ground_truth("other"), ground_truth("t")
        empty = harness.ledger("t", giver)

    assert t["transfer"]["seat"] == "2", "the record names the seat that was asked for"
    assert t["transfer"]["error"] == "seat 2 is out", t["transfer"]
    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
    assert gone["remaining"] == 0 and not gone.get("received"), gone["remaining"]
    assert not giver.get("sent") and not giver.get("rebated"), giver
    assert t["transfer"]["penalty"] > 0, "and the share falls as it does on no transfer at all"
    assert empty == [], "nothing reaches g"

    with temp_root(REBATE_PERCENT=100, TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")
        with quiet():
            t = harness.run_once("t", fake(run("echo '3 1' > out/transfer"), say()))
    assert t["transfer"]["amount"] == 1 and t["transfer"]["error"] is None, t["transfer"]
    assert t["transfer"]["penalty"] == 0, "the seat that is still solvent takes it"


def check_a_seat_that_is_out_is_not_a_message():
    """out/<i> for a seat that is out is neither a message nor a break.

    It stands as a name that is no seat of this experiment does: nothing there will
    episode start to read it, and a directory left at it is not a crowded seat either.
    """
    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")                                          # seat 2
        with quiet():
            said = harness.run_once("t", fake(run("echo hi > out/2"), say()))
    assert said["mailbox"]["addressed"] == [], said["mailbox"]
    assert said["mailbox"]["penalty"] > 0, "an episode that reached nobody is charged"
    assert harness.outbox_why(said["mailbox"]) == "no message", said["mailbox"]

    with temp_root(MAILBOX_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, "t", other={}, third={})
        spend_out("other")
        with quiet():
            both = harness.run_once("t", fake(run("mkdir out/2", "echo hi > out/3"), say()))
    assert both["mailbox"]["addressed"] == ["3"], both["mailbox"]
    assert both["mailbox"]["broken"] == [], "a seat that is out cannot be crowded"
    assert both["mailbox"]["penalty"] == 0, both["mailbox"]


def check_the_channels_answer_differently():
    """Its own three trees take writes; every seat, message and balance refuses.

    The whole arrangement in one episode: what the agent may not write it cannot
    reach by writing, by chmod, or by replacing the directory the file sits in.
    """
    with docker_root() as root:
        ids = experiment_of(root, t={"NOTES.md": "private\n", "group/out": "mine\n"},
                        other={"NOTES.md": "unseen\n", "group/out": "theirs\n",
                               "out/1": "just for you\n"})
        with quiet():
            account = harness.load_account("t")
        account["seat"], account["peers"] = "1", {"seen": experiment.mapping(ids)}
        harness.save_account("t", account)
        t = episode_once(run("stat -c '%a %U:%G %n' state 1 2 out in in/2 n1 n2 g m",
                          "echo kept > state/new && echo PRIVATE-OK",
                          "echo posted > 1/out && echo GROUP-OK",
                          "echo sent > out/2 && echo OUTBOX-OK",
                          "echo hacked > 2/out 2>&1 || echo DENIED",
                          "rm -f 2/out 2>&1 || echo DENIED",
                          "mv 2 2old 2>&1 || echo DENIED",
                          "chmod -R 777 2 2>&1 || echo DENIED",
                          "rm -f n2 2>&1 || echo DENIED",
                          "echo forged > in/2 2>&1 || echo DENIED",
                          "rm -f in/2 2>&1 || echo DENIED",
                          "mv in/2 in/9 2>&1 || echo DENIED",
                          "echo forged > m 2>&1 || echo DENIED",
                          "rm -f m 2>&1 || echo DENIED",
                          "cat 2/out n2 in/2",
                          "grep -r unseen /work 2>/dev/null | head -1; echo NO-PRIVATE"), say())
        after = {p.name: p.read_text(encoding="utf-8")
                 for p in harness.blackboard_dir("other").iterdir()}

    (modes, private, group, outbox, write, rm, mv, chmod, rm_n,
     forge, rm_in, mv_in, forge_m, rm_m, read, hunt) = (c["result"]
                                                        for c in t["turns"][0]["tools"])
    owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
    assert owner["state"] == owner["1"] == owner["out"] == "agent:agent", modes
    assert owner["2"] == "root:root", f"another seat is root's: {modes}"
    assert owner["in"] == owner["in/2"] == "root:root", \
        f"an inbox is root's, and so is the directory holding it: {modes}"
    assert owner["n1"] == owner["n2"] == owner["g"] == owner["m"] == "root:root", \
        f"every balance, the ledger and m are root's: {modes}"
    # The three it owns.
    assert "PRIVATE-OK" in private and "GROUP-OK" in group and "OUTBOX-OK" in outbox, \
        (private, group, outbox)
    # And every route into what it does not.
    for name, out in (("write a peer", write), ("rm a peer's file", rm),
                      ("mv the seat", mv), ("chmod the seat", chmod),
                      ("rm a balance", rm_n), ("forge an inbox", forge),
                      ("rm an inbox", rm_in), ("mv an inbox", mv_in),
                      ("forge what was said", forge_m), ("rm what was said", rm_m)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    # An agent the experiment laid out but never metered has no series, so its balance
    # is the empty array - the shape the first round of an experiment reads.
    assert read.split("\n")[0].strip() == "theirs", f"the peer's blackboard is untouched: {read}"
    assert "[]" in read and "just for you" in read, \
        f"its balance and the message it was sent both read as they were left: {read}"
    assert "unseen" not in hunt and "NO-PRIVATE" in hunt, \
        f"the other agent's private store is nowhere in this environment: {hunt}"
    assert after == {"out": "theirs\n"}, f"and its blackboard is as it left it: {after}"

    by = {f["path"]: f for f in t["files"]}
    assert by["state/new"]["channel"] == "notes" and not by["state/new"]["ours"]
    assert by["1/out"]["channel"] == "blackboard" and not by["1/out"]["ours"]
    assert by["out/2"]["channel"] == "outbox" and not by["out/2"]["ours"]
    assert by["2/out"]["channel"] == "peer_blackboard" and by["2/out"]["ours"]
    assert by["in/2"]["channel"] == "inbox" and by["in/2"]["ours"]


def check_a_ledger_resists_every_route():
    """g refuses append, chmod, rm, mv, symlink and an absolute path, like a balance.

    Every transfer being public is only true while the file saying so cannot be
    edited by the agents it is about.
    """
    with docker_root() as root:
        ids = experiment_of(root, t={}, other={})
        with quiet():
            account = harness.load_account("t")
        account["seat"], account["peers"] = "1", {"seen": experiment.mapping(ids)}
        harness.save_account("t", account)
        t = episode_once(run("echo 9 9 9 >> g 2>&1 || echo DENIED",
                          "chmod 666 g 2>&1 || echo DENIED",
                          "rm -f g 2>&1 || echo DENIED",
                          "mv g gold 2>&1 || echo DENIED",
                          "ln -sf /dev/null g 2>&1 || echo DENIED",
                          "echo 1 2 3 > /work/g 2>&1 || echo DENIED",
                          "cat g; echo LEDGER-END"), say())
    *routes, read = (c["result"] for c in t["turns"][0]["tools"])
    for i, out in enumerate(routes):
        assert "DENIED" in out, f"route {i} into the ledger was allowed: {out}"
    assert read.strip() == "LEDGER-END", f"and it is still the empty ledger: {read}"


def check_a_refusal_records_why():
    """stop_details is captured on a refusal and absent on every other stop.

    A classifier declining and the model itself declining both arrive as
    stop_reason "refusal", and the category is what separates them.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = episode_once(run("echo hi"), refuse(),
                      refuse(recommended_model="claude-sonnet-5"), say())

    refused = t["turns"][1]
    assert refused["stop_reason"] == "refusal"
    assert refused["stop_details"] == {"type": "refusal", "category": "cyber",
                                       "explanation": "declined", "recommended_model": None,
                                       "fallback_credit_token": None}, refused["stop_details"]
    assert t["turns"][0]["stop_details"] is None, "absent on every other stop reason"
    # A refusal can also name a model to retry on. It is set where the fallback
    # attempt was skipped because the model it would have used was rate limited,
    # which is a different failure from a category with no fallback at all.
    assert t["turns"][2]["stop_details"]["recommended_model"] == "claude-sonnet-5", \
        t["turns"][2]["stop_details"]
    assert t["refused_turns"] == 2, "counted whether or not they ended the episode"


def check_the_refusal_notice_is_pinned():
    """The notice is pinned, and says only what happened and what it left alone.

    It is the second thing the harness says, so it is held to what invariant 2 holds the
    first to: no cause, no instruction, and nothing addressed to the agent.
    """
    digest = hashlib.sha256(harness.REFUSAL_NOTICE.encode()).hexdigest()
    assert digest == harness.REFUSAL_NOTICE_SHA256, digest
    assert ("REFUSAL_NOTICE", harness.REFUSAL_NOTICE, harness.REFUSAL_NOTICE_SHA256) in harness.PINNED, \
        "start() refuses on what --print-system audits, so both read PINNED"
    low = harness.REFUSAL_NOTICE.lower()
    for word in ("polic", "safet", "classif", "anthropic", "cyber", "block",
                 "you ", "your", "try", "instead", "again"):
        assert word not in low, f"REFUSAL_NOTICE contains {word!r}"


def check_a_refusal_does_not_run_its_command():
    """A refused turn's tool call is not executed.

    A refusal can arrive with a call already emitted and cut mid-JSON, so what
    it would execute is not what the agent wrote.
    """
    with temp_root():
        t = episode_once(refuse("echo poison > state/poison.txt"), say())

    assert "echo poison > state/poison.txt" not in t["commands"], t["commands"]
    assert not [f for f in t["files"] if f["path"] == "poison.txt"], "state/ is untouched"
    assert t["turns"][0]["tools"] == [], "no result recorded, because nothing ran"


def check_a_refusal_notice_reaches_the_agent():
    """The notice stands in for the results the refused turn would have had.

    Reached only where the cap lets an episode carry on past a refusal, so
    REFUSAL_TURNS is raised here. `seen` is the whole `messages` list at the end.
    """
    seen = []
    with temp_root(REFUSAL_TURNS=2):
        episode_once(refuse("cat n1"), say(), seen=seen)
    # A refusal carrying a call leaves a tool_use the next request must answer.
    blocks = [b for m in seen[-1]["mailbox"] if isinstance(m["content"], list)
              for b in m["content"]
              if isinstance(b, dict) and b.get("content") == harness.REFUSAL_NOTICE]
    assert len(blocks) == 1, blocks
    assert blocks[0]["type"] == "tool_result", blocks[0]
    assert blocks[0]["is_error"] is True, "the same channel a timed-out command uses"

    seen = []
    with temp_root(REFUSAL_TURNS=2):
        episode_once(refuse(), say(), seen=seen)
    # A refusal with no content has no call to answer, and no words to replay.
    msgs = seen[-1]["mailbox"]
    assert {"role": "user", "content": harness.REFUSAL_NOTICE} in msgs, msgs
    assert all(m["content"] for m in msgs), "no empty message is sent back"


def check_a_refusal_is_billed_only_if_it_produced_output():
    """A refusal costs what it emitted, and an empty one emitted nothing.

    The API reports the tokens of a refusal arriving before any output and does
    not charge for them; one arriving with content did produce output.
    """
    with temp_root(REFUSAL_TURNS=3):
        t = episode_once(refuse(), refuse("echo hi"), say())

    assert t["turns"][0]["micros"] == 0, "a refusal before any output is not billed"
    assert t["turns"][0]["balance"] == t["series_before"][-1], "the balance did not move"
    assert t["turns"][1]["micros"] > 0, "output was produced, so that one was billed"
    assert len(t["balances"]) == len(t["turns"]), "one element per turn, billed or not"
    assert t["turns"][2]["micros"] > 0, "the turn that answered was billed"


def check_a_chain_is_billed_by_whichever_model_answered():
    """Only the attempt that answered is billed; where none did, nothing is.

    Four shapes of one rule, as four turns of one episode: each assertion names
    the turn it is about, and the totals at the end are what no turn reaches.
    """
    declined = usage(output_tokens=0, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 0, kind="fallback_message")])
    served = usage(output_tokens=200, iterations=[
        attempt("claude-opus-5", 0),
        attempt("claude-sonnet-5", 200, kind="fallback_message")])
    sticky = usage(output_tokens=200,
                   iterations=[attempt("claude-sonnet-5", 200, kind="fallback_message")])
    unpriced = usage(output_tokens=200,
                     iterations=[attempt("claude-unheard-of-9", 200, kind="fallback_message")])

    with temp_root(MODEL="claude-opus-5", REFUSAL_TURNS=2):
        t = episode_once(refuse(u=declined),
                      run("echo one", u=served, model="claude-sonnet-5"),
                      run("echo two", u=sticky, model="claude-sonnet-5"),
                      run("echo three", u=unpriced, model="claude-unheard-of-9"),
                      say())

    # A chain every model declined. The last attempt is a fallback_message, the
    # same type the serving attempt carries, and it produced nothing: a rule
    # sparing only the entries typed `message` would bill it, and put back the
    # overcharge the empty-refusal rule takes away.
    nothing = t["turns"][0]
    assert nothing["micros"] == 0, "no attempt produced output, so none was billed"
    assert len(nothing["iterations"]) == 2, "both attempts are on the record"

    # A chain a fallback answered: sonnet's rates, not the requested opus-5's.
    # 100 in at 200 centi, 200 out at 1000. The declining attempt adds nothing.
    answered = t["turns"][1]
    assert answered["served_by_fallback"] is True, answered
    assert answered["model"] == "claude-sonnet-5", answered["model"]
    want = (100 * 200 + 200 * 1000) // 100
    assert answered["micros"] == want, f"{answered['micros']} != {want}"

    # Sticky routing: after a conversation falls back, later turns can go
    # straight to the model that accepted. No attempt by the requested model
    # appears and no fallback block marks a handoff, so the iteration entry and
    # the reported model are the only record of who served it.
    routed = t["turns"][2]
    assert routed["served_by_fallback"] is True, routed
    assert routed["model"] == "claude-sonnet-5", routed["model"]
    assert [i["type"] for i in routed["iterations"]] == ["fallback_message"], routed["iterations"]

    # A model outside PRICES, which default routing can reach at any time.
    # Raising would lose the cost of a turn that really did spend; counting it
    # free would understate the balance the agent is shown.
    odd = t["turns"][3]
    assert odd["unpriced_model"] == ["claude-unheard-of-9"], odd["unpriced_model"]
    dearest = max(harness.PRICES, key=lambda m: harness.PRICES[m][1])
    inp, out, _ = harness.PRICES[dearest]
    assert odd["micros"] == (100 * inp + 200 * out) // 100, odd["micros"]

    # And what only the whole episode says: the counters agree with the turns
    # they are counting, and an unpriced model did not end the agent.
    assert t["stop"] == "end_turn", f"an unpriced model must not end the agent: {t['stop']}"
    served_by = [x["served_by_fallback"] for x in t["turns"]]
    assert served_by[1:4] == [True, True, True], served_by
    assert t["fallback_turns"] == sum(served_by), (t["fallback_turns"], served_by)
    assert t["unpriced_turns"] == 1, t["unpriced_turns"]


def check_an_unhandled_stop_reason_is_named():
    """A stop reason the loop has no branch for ends the episode saying so.

    Read as the absence of tool calls it would be filed as end_turn, which says
    the agent chose to stop when in fact the harness did not know how to go on.
    """
    with temp_root():
        t = episode_once(say(stop="pause_turn"), say())
    assert t["stop"] == "unhandled:pause_turn", t["stop"]


def check_every_response_is_logged_raw():
    """Each response is appended verbatim, before anything else reads it."""
    with temp_root():
        t = episode_once(run("echo hi"), refuse(), say())
        log = harness.records_dir("t") / "raw" / f"episode-{t['episode']:04d}.jsonl"
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]

    assert [x["turn"] for x in lines] == [1, 2], lines
    assert all(x["response"]["id"] for x in lines), "the whole response, id and all"
    # The refusal above all: the trace keeps named fields of stop_details, and
    # this keeps whatever the API actually sent.
    assert lines[1]["response"]["stop_details"]["category"] == "cyber", lines[1]


def check_the_raw_log_never_stops_an_episode():
    """A log that cannot be written is reported, and the episode goes on.

    The log is a record of the agent, not part of it. An episode that is spending
    money does not stop because a line could not be appended.
    """
    with temp_root() as root:
        # A file where the agent's raw/ directory needs to be, so mkdir fails.
        (harness.records_dir("t")).mkdir(parents=True, exist_ok=True)
        (harness.records_dir("t") / "raw").write_text("in the way", encoding="utf-8")
        t = episode_once(run("echo hi"), say())
    assert t["stop"] == "end_turn", t["stop"]
    assert "echo hi" in t["commands"], "the episode ran despite the log failing"


def check_refusals_end_the_episode_at_the_cap():
    """REFUSAL_TURNS running end the episode, and it stops asking."""
    seen = []
    with temp_root(REFUSAL_TURNS=3):
        t = episode_once(refuse(), refuse(), refuse(), say(), seen=seen)

    assert t["stop"] == "refusal", t["stop"]
    assert t["refused_turns"] == 3, t["refused_turns"]
    assert len(seen) == 3, f"asked {len(seen)} times past the cap"


def check_a_recovered_refusal_is_not_a_refused_episode():
    """Refusals an episode gets past are counted but do not name its stop.

    stalled() reads the episode stop, so an agent that acted must not look like one
    that never got to.
    """
    with temp_root(REFUSAL_TURNS=4):
        t = episode_once(refuse(), refuse(), run("echo hi > state/note.txt"), say())

    assert t["stop"] == "end_turn", t["stop"]
    assert t["refused_turns"] == 2, t["refused_turns"]
    assert "echo hi > state/note.txt" in t["commands"], "the episode went on to act"
    streak = [{"stop": t["stop"]}] * harness.REFUSAL_STREAK
    assert not harness.stalled({"episodes": streak}), "a recovered episode breaks the streak"


def check_a_stalled_agent_stops_itself():
    """An agent that refuses REFUSAL_STREAK episodes running is not admitted again.

    A refusal ends an episode before the agent writes anything, so the next episode
    opens on a near-identical context: the agent never acts, so it cannot escape.
    """
    streak = harness.REFUSAL_STREAK
    episodes = [{"episode": i, "stop": "refusal", "spent": 1, "turns": 1, "balance_at_start": 9}
                for i in range(1, streak + 1)]
    account = {"remaining": 999_999, "episodes": episodes}

    assert harness.stalled(account), f"{streak} refusals running is stuck"
    assert not harness.admits(account), "and a stuck agent is not admitted, whatever its balance"
    assert not harness.stalled({**account, "episodes": episodes[:-1]}), "one short is not stuck"
    # A single success anywhere in the window clears it: the agent acted, so its
    # next episode opens on something it wrote rather than on the same context.
    broken = [*episodes[:-1], {**episodes[-1], "stop": "end_turn"}]
    assert not harness.stalled({**account, "episodes": broken})
    assert harness.admits({**account, "episodes": broken})
    # Deliberately a runaway guard, not a productivity filter: a healthy agent in
    # the experiment that produced this rule refused three episodes running.
    assert streak > 3, f"REFUSAL_STREAK={streak} would stop an agent that recovers"


def check_anything_on_a_peers_blackboard_is_not_the_agents_bytes():
    """Whatever appears on another agent's blackboard is that agent's, whoever put it there.

    In a container the agent cannot write there at all, but the record does not
    lean on that: what makes a file the agent's is the channel it is in.
    """
    with temp_root() as root:
        ids = experiment_of(root, t={"NOTES.md": "mine\n"}, other={"group/out": "theirs\n"})
        (harness.blackboard_dir("other") / "added").write_text("put here somehow\n")
        snap = harness.snapshot(environment_of("t", ids), [9])

    by = {f["path"]: f for f in snap["files"]}
    assert by["2/added"]["ours"] and not by["2/added"]["starter"], by["2/added"]
    assert by["2/added"]["text"].strip() == "put here somehow", "still captured in full"
    assert [f["path"] for f in snap["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "only what it wrote in its own two trees counts as its own"


def check_the_experiment_rotates_and_validates():
    """Order rotates by round, and an experiment of one or of bare numbers is refused."""
    ids = ["g01", "g02", "g03"]
    assert [experiment.order(ids, r) for r in range(4)] == [
        ["g01", "g02", "g03"], ["g02", "g03", "g01"],
        ["g03", "g01", "g02"], ["g01", "g02", "g03"]], "a fixed order is a standing advantage"
    for bad in (["--agents", "g01"],                       # one agent has no peers
                ["--agents", "g01", "g01"],                # nor does an agent twice
                ["--agents", "g01", "1"],                  # a bare number is a seat
                ["--agents", "g01", "g02", "--rounds", "0"],
                ["--manifest", "no-such-experiment.toml"],   # a named manifest must exist
                ["--agents", "g01", "g02", "--manifest", "c.toml"],  # one source, not two
                []):                                     # and at least one
        with quiet():
            try:
                experiment.main(bad)
            except SystemExit as e:
                assert e.code != 0, bad
            else:
                raise AssertionError(f"accepted bad experiment: {bad}")


def check_an_experiment_gives_a_failed_environment_one_more_go():
    """An agent whose environment will not build sits out one attempt, not the experiment.

    Building an environment reads other agents' trees and asks Docker for a container, so
    a failure can be the daemon rather than the agent. None of it is billed.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        failures = {"g02": 1, "g03": 99}
        real = harness.drive

        def flaky(agent, create, prepare=None):
            if failures.get(agent):
                failures[agent] -= 1
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, create, prepare)

        harness.drive = flaky
        live = set(ids)
        with quiet() as buf:
            experiment.sequential_round(ids, live, 0, fake(*DEFAULT))
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}

    assert live == {"g01", "g02"}, f"only the agent that failed twice is out: {live}"
    assert took == {"g01": 1, "g02": 1, "g03": 0}, took
    assert "could not build an environment" in buf.getvalue(), buf.getvalue()
    assert "could not start an episode container" not in buf.getvalue(), \
        "the container started; it is the environment that did not"
    assert buf.getvalue().count(f"(1 of {experiment.ATTEMPTS})") == 2, buf.getvalue()


def check_an_interrupt_ends_the_whole_experiment():
    """Ctrl+C ends every remaining round; a fault ends one agent's part in them.

    An interrupt is the experimenter: the agent awake when it landed keeps its seat
    and the rounds stop. A fault is the agent, and only that agent drops out.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            try:
                experiment.sequential_round(ids, live, 0, fake(run("echo one"), KeyboardInterrupt()))
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next agent")
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
        first = ground_truth("g01")["episodes"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no agent is ejected for it: {sorted(live)}"

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            experiment.sequential_round(ids, live, 0, fake(run("echo one"), Err(400)))
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    assert live == {"g02", "g03"}, sorted(live)
    assert took == {"g01": 1, "g02": 1, "g03": 1}, "the rest of the experiment takes its round"

    # And main answers an interrupt by ending the rounds, not the round.
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(run("echo one"), KeyboardInterrupt())
        with quiet() as buf:
            code = experiment.main(["--agents", *ids, "--rounds", "5"])
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    assert code == 130, code
    assert took == {"g01": 1, "g02": 0, "g03": 0}, "no round after the one it landed in"
    assert "interrupted" in buf.getvalue(), buf.getvalue()


def check_a_round_nobody_can_act_in_ends_the_rounds():
    """Rounds stop when no agent can take an episode, without waiting for --rounds.

    Every agent spends past zero in the first round, and no peer is left that
    could put one back, so the rounds are over with four still to go.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(["--agents", *ids, "--rounds", "5"])
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
        rested = {r: harness.load_account(r)["remaining"] for r in ids}
    assert code == 0, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, \
        f"one episode each, then nothing left to ask for: {took}"
    assert set(rested.values()) == {0}, rested
    assert buf.getvalue().count("drops out: nothing left to spend") == 3, buf.getvalue()
    assert "every agent is out after 1 rounds" in buf.getvalue(), buf.getvalue()


def check_the_last_agent_standing_takes_one_more_episode():
    """One agent left holding a balance ends the rounds, after a last episode.

    No later round can unmeet the win condition, so the last agent gets one more
    episode. It owes neither transfer nor message, but still owes its own group.
    """
    with temp_root(TRANSFER_SILENCE_PENALTY_PERCENT=50, MAILBOX_SILENCE_PENALTY_PERCENT=50,
                   BLACKBOARD_SILENCE_PENALTY_PERCENT=50) as root:
        ids = seated(root, "g01", g02={}, g03={})
        spend_out("g02")
        spend_out("g03")
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(["--agents", *ids, "--rounds", "5"])
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
        alone = harness.load_account("g01")["episodes"][-1]
    assert code == 0, code
    assert took == {"g01": 1, "g02": 0, "g03": 0}, \
        f"one last episode for the one still holding a balance: {took}"
    assert buf.getvalue().count("drops out: nothing left to spend") == 2, buf.getvalue()
    assert "g01 is the only agent left with anything to spend" in buf.getvalue(), buf.getvalue()
    assert alone["transfer"]["penalty"] == 0, alone["transfer"]
    assert alone["mailbox"]["penalty"] == 0, alone["mailbox"]
    assert not alone["posted"] and alone["blackboard_penalised"] > 0, \
        "the blackboard is the one obligation an agent with no peers left can still fail"


# --- the companion view -----------------------------------------------------
#
# view.py reads what is on disk while an agent is going, writing nothing and
# asking Docker nothing, so every check here agents in the arithmetic lane. A
# episode in flight has no trace, so its cost is derived from the raw log.


def unfinished(index: int = 1) -> None:
    """Take away an episode's trace, leaving the raw log a running one leaves."""
    view.trace_path("t", index).unlink()


def check_a_manifest_is_validated():
    """A manifest names a schedule, the experiment's defaults, and each agent's terms, or is refused."""
    other_model = next(m for m in harness.PRICES if m != harness.MODEL)
    good = (f'schedule = "simultaneous"\nrebate_percent = 0\n'
            f'[[agent]]\nid = "g01"\nstarter_files = "s"\nstarter_files_below = 400000\n'
            f'[[agent]]\nid = "g02"\nbudget = 7\nmodel = "{other_model}"\n')
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-manifest-") as tmp:
        root = Path(tmp)
        harness.ROOT = root
        plant(root, "s")
        two = '[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n'
        for bad in ('colour = "red"\n' + two,                    # an unknown key
                    'schedule = "random"\n' + two,               # an unknown schedule
                    '[[agent]]\nid = "g01"\n',                     # one agent has no peers
                    '[[agent]]\nid = "g01"\n[[agent]]\nid = "g01"\n',  # nor does an agent twice
                    '[[agent]]\nid = "1"\n[[agent]]\nid = "g02"\n',  # a bare number is a seat
                    '[[agent]]\nid = "g01"\nstarter_files = "s"\n[[agent]]\nid = "g02"\n',       # starter_files alone
                    '[[agent]]\nid = "g01"\nstarter_files = "nope"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nbudget = 0\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nbudget = "many"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nmodel = "no-such"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\ncolour = "red"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nstarter_files = "s"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',  # no id
                    'agent = 5\n',                                 # agents are tables
                    'not toml ==\n'):
            p = manifest_file(root, bad)
            try:
                experiment.load_manifest(p)
            except SystemExit as e:
                assert str(p) in str(e), (bad, e)
            else:
                raise AssertionError(f"accepted bad manifest: {bad!r}")
        try:
            experiment.load_manifest(root / "experiments" / "missing.toml")
        except SystemExit:
            pass
        else:
            raise AssertionError("a missing manifest was ignored")

        p = manifest_file(root, good)
        m = experiment.load_manifest(p)
    assert m["schedule"] == "simultaneous"
    assert m["overrides"] == {"rebate_percent": 0}, "everything else is an experiment default"
    assert [e["id"] for e in m["agents"]] == ["g01", "g02"]
    assert m["sha256"] == hashlib.sha256(good.encode("utf-8")).hexdigest()
    assert experiment.terms_of(m["agents"][0]) == {"model": None, "budget": None, "starter_files": "s",
                                             "starter_files_below": 400000}
    assert experiment.terms_of(m["agents"][1]) == {"model": other_model, "budget": 7, "starter_files": None,
                                             "starter_files_below": None}
    short = experiment.shorthand(["a", "b"])
    assert short["schedule"] == "sequential" and short["overrides"] == {} and short["sha256"] == ""
    assert [e["id"] for e in short["agents"]] == ["a", "b"]
    assert experiment.stamp_of(m) == {"schedule": "simultaneous", "manifest_sha256": m["sha256"]}


def check_a_manifest_gives_each_agent_its_own_starter_files():
    """Each agent is created on its own terms, the experiment's defaults apply to all, and the
    terms are pinned: a manifest that later says otherwise is refused."""
    text = ('transfer_funded_by = "none"\n'
            '[[agent]]\nid = "g01"\nstarter_files = "a"\nstarter_files_below = 500000\n'
            '[[agent]]\nid = "g02"\nstarter_files = "b"\nstarter_files_below = 600000\nbudget = 600000\n'
            '[[agent]]\nid = "g03"\n')
    with temp_root() as root:
        plant(root, "a")
        plant(root, "b", m1="bravo\n")
        p = manifest_file(root, text)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        asked = []

        def start(config=None, overrides=None, models=()):
            asked.append((overrides, set(models)))
            harness.apply_config(overrides or {}, "manifest")
            return fake(*DEFAULT)

        harness.start = start
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "1"])
        accounts = {r: ground_truth(r) for r in ("g01", "g02", "g03")}
        starter = {r: (harness.state_dir(r) / "m1").read_text(encoding="utf-8")
                  if (harness.state_dir(r) / "m1").exists() else None for r in accounts}
        traces = {r: json.loads((harness.records_dir(r) / "traces" / "episode-0001.json")
                                .read_text(encoding="utf-8")) for r in accounts}

        p.write_text(text.replace('starter_files = "a"', 'starter_files = "b"'), encoding="utf-8", newline="\n")
        with quiet():
            try:
                experiment.main(["--manifest", str(p), "--rounds", "1"])
            except SystemExit as e:
                assert "starter_files" in str(e), e
            else:
                raise AssertionError("an agent was re-created on different terms")
    assert code == 0, buf.getvalue()
    assert asked[0] == ({"transfer_funded_by": "none"}, set()) and len(asked) == 2, asked
    assert {r: m["starter_files"] for r, m in accounts.items()} == {"g01": "a", "g02": "b", "g03": ""}
    assert accounts["g02"]["initial"] == 600000 and accounts["g01"]["initial"] == harness.BUDGET
    assert starter == {"g01": "alpha\n", "g02": "bravo\n", "g03": None}, starter
    for r, t in traces.items():
        assert t["provenance"]["starter_files"] == accounts[r]["starter_files"], (r, t["provenance"])
        assert t["provenance"]["schedule"] == "sequential"
        assert t["provenance"]["manifest_sha256"] == digest
        assert t["provenance"]["transfer_funded_by"] == "none", "the experiment's default reached every agent"
    assert "sequential" in buf.getvalue(), buf.getvalue()


def check_a_simultaneous_round_builds_every_environment_before_any_episode_runs():
    """Under a simultaneous every container is up and loaded before the first API call."""
    RecordingBox.events = []
    with temp_root(BOX=RecordingBox) as root:
        ids = seated(root, "g01", g02={}, g03={})
        inner = per_run(default=DEFAULT)

        def create(**params):
            RecordingBox.note("create", threading.current_thread().name)
            return inner(**params)

        live = set(ids)
        with quiet():
            acted = experiment.simultaneous_round(ids, live, 0, create)
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    events = RecordingBox.events
    first = next(i for i, (what, _) in enumerate(events) if what == "create")
    before = [what for what, _ in events[:first]]
    assert before.count("start") == before.count("load") == 3 and set(before) == {"start", "load"}, \
        f"every environment is built before any episode agents: {events}"
    assert [agent_of(n) for what, n in events if what == "start"] == ids, "in seat order"
    assert sorted(agent_of(n) for what, n in events if what == "close") == ids, "and every one reaped"
    assert acted and took == {"g01": 1, "g02": 1, "g03": 1} and live == set(ids), (took, live)


def check_a_simultaneous_round_runs_its_episodes_at_once():
    """The episodes of a simultaneous round are in flight together, not one after another."""
    gate = threading.Barrier(3, timeout=10)
    inner = per_run(default=(say(),))

    def create(**params):
        # Passes only if all three episodes reach their first call together.
        gate.wait()
        return inner(**params)

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            experiment.simultaneous_round(ids, live, 0, create)
        stops = {r: harness.load_account(r)["episodes"][0]["stop"] for r in ids}
    assert set(stops.values()) == {"no_tool_call"},         f"an episode that waited alone would have erred instead: {stops}"


def check_a_simultaneous_round_reads_last_round_and_not_this_one():
    """Nobody in a simultaneous round reads what the round writes; all of it arrives at the next.

    A transfer made in the round is credited to its receiver in the same round, after
    the receiver's own turns, and shows in g and n at the next episode.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={})
        first = per_run(g01=(run("echo hello > 1/RESULT", "echo psst > out/2",
                                 "echo '2 250' > out/transfer"), say()),
                        g02=(run(f"cat {harness.DIGEST_NAME}", "ls 1"), say()))
        live = set(ids)
        with quiet():
            experiment.simultaneous_round(ids, live, 0, first)
        g02_first = ground_truth("g02")["episodes"][0]
        said, listed = (c["result"] for c in json.loads(
            (harness.records_dir("g02") / "traces" / "episode-0001.json")
            .read_text(encoding="utf-8"))["turns"][0]["tools"])
        # g01 withdraws the line, so the transfer is made once and not every round it stands.
        second = per_run(g01=(run("rm out/transfer", f"cat {harness.DIGEST_NAME}"), say()),
                         g02=(run(f"cat {harness.DIGEST_NAME}"), say()))
        with quiet():
            experiment.simultaneous_round(ids, live, 1, second)
        later = json.loads((harness.records_dir("g02") / "traces" / "episode-0002.json")
                           .read_text(encoding="utf-8"))
        g02 = ground_truth("g02")
    assert "hello" not in said and "psst" not in said and "RESULT" not in listed, \
        f"round 0 did not read round 0: {said} / {listed}"
    assert "hello" in later["observation"] and "psst" in later["observation"], later["observation"]
    assert "1 2 250" in later["observation"], "the transfer is on the ledger at the next episode"
    assert g02["received"] == 250 and g02_first["received"] == 250, (g02, g02_first)
    span = g02["series"][g02_first["series_from"]:g02_first["series_to"] + 1]
    assert len(span) == elements_of(g02_first), (span, g02_first)
    assert span[-1] == span[-2] + 250, "the credit is the last element of the receiver's span"


def check_a_simultaneous_credit_lands_before_the_floor():
    """A same-round transfer reaches a receiver that overspent before the floor decides on it."""
    cost = turn_cost()
    results = {}
    for name, transfer in (("transfer", "echo '2 50' > out/transfer"), ("none", "true"),
                       ("exact", "echo '2 1' > out/transfer")):
        with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
            ids = seated(root, "g01", g02={})
            live = set(ids)
            with quiet():
                experiment.simultaneous_round(ids, live, 0, per_run(g01=(run(transfer), say()),
                                                           g02=(say(),)))
            m = ground_truth("g02")
            results[name] = (m["remaining"], m.get("forgiven", 0), harness.admits(m))
    assert results["transfer"] == (49, 0, True), \
        f"the credit landed first, so nothing was forgiven and the agent goes on: {results}"
    assert results["none"] == (0, 1, False), f"without it the overshoot is floored: {results}"
    assert results["exact"] == (0, 0, False), \
        f"a credit that brings the agent to exactly zero leaves it out, for good: {results}"


def check_an_interrupt_in_a_simultaneous_round_commits_every_episode_in_flight():
    """Ctrl+C in a simultaneous round ends every episode at its next turn, and all are committed."""
    def stopping_create():
        gate = threading.Barrier(3, timeout=10)
        inner = per_run(default=(run("echo one"), run("echo two"), say()))

        def create(**params):
            gate.wait()                     # all three in flight together
            r = inner(**params)
            harness.STOPPING = True            # read at the top of everyone's next turn
            return r
        return create

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            try:
                experiment.simultaneous_round(ids, live, 0, stopping_create())
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next")
        episodes = {r: harness.load_account(r)["episodes"] for r in ids}
    assert all(len(s) == 1 for s in episodes.values()), episodes
    for r, (s,) in episodes.items():
        assert s["stop"] == "interrupted" and s["turns"] == 1 and s["spent"] > 0, (r, s)
    assert live == set(ids), "and no agent is ejected for it"

    # And main under a simultaneous manifest answers the same way.
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        p = manifest_file(root, 'schedule = "simultaneous"\n'
                          + "".join(f'[[agent]]\nid = "{r}"\n' for r in ids))
        harness.start = lambda config=None, **kw: stopping_create()
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "5"])
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    assert code == 130, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, took
    assert "interrupted" in buf.getvalue() and "simultaneous" in buf.getvalue(), buf.getvalue()


def check_the_console_lines_of_a_simultaneous_round_are_whole_and_in_seat_order():
    """Every echoed line names its agent, and the summary lines come last, in seat order."""
    with temp_root(WATCH=True) as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet() as buf:
            experiment.simultaneous_round(ids, live, 0,
                                 per_run(default=(run("echo one"), run("echo two"), say())))
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "nothing was echoed"
    assert all(any(ln.startswith(r) for r in ids) for ln in lines), \
        f"a line that does not say whose it is: {[ln for ln in lines if not any(ln.startswith(r) for r in ids)]}"
    summary = [i for i, ln in enumerate(lines) if re.match(r"^g0\d\s+s1\s", ln)]
    assert [lines[i].split()[0] for i in summary] == ids, "summaries in seat order"
    assert summary and summary[0] > max(i for i, ln in enumerate(lines) if "| " in ln), \
        "and after every echoed line"


def check_a_simultaneous_round_drops_an_agent_whose_environment_fails_twice_and_seats_the_rest():
    """An environment that will not build costs its agent one attempt, then its seat; the rest run."""
    RecordingBox.events = []
    with temp_root(BOX=RecordingBox) as root:
        ids = seated(root, "g01", g02={}, g03={})
        failures = {"g02": 1, "g03": 99}
        real = harness.ready

        def flaky(agent, prepare=None):
            if failures.get(agent):
                failures[agent] -= 1
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, prepare)

        harness.ready = flaky
        live = set(ids)
        with quiet() as buf:
            experiment.simultaneous_round(ids, live, 0, per_run(default=DEFAULT))
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    assert live == {"g01", "g02"}, f"only the agent that failed twice is out: {live}"
    assert took == {"g01": 1, "g02": 1, "g03": 0}, took
    assert buf.getvalue().count(f"(1 of {experiment.ATTEMPTS})") == 2, buf.getvalue()
    starts = [n for what, n in RecordingBox.events if what == "start"]
    closes = [n for what, n in RecordingBox.events if what == "close"]
    assert sorted(starts) == sorted(closes), "every container that started was reaped"


def check_a_stop_during_the_builds_of_a_simultaneous_round_starts_no_episode():
    """A stop that lands while environments are being built starts nothing and reaps what was built."""
    class StoppingBox(RecordingBox):
        @classmethod
        def start(cls, name):
            box = super().start(name)
            if sum(1 for what, _ in cls.events if what == "start") == 2:
                harness.STOPPING = True
            return box

    RecordingBox.events = []
    calls = []
    with temp_root(BOX=StoppingBox) as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)

        def create(**params):
            calls.append(params)
            return fake(*DEFAULT)(**params)

        with quiet():
            try:
                experiment.simultaneous_round(ids, live, 0, create)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round ran its episodes")
        took = {r: len(harness.load_account(r)["episodes"]) for r in ids}
    assert not calls, "no episode was started"
    assert took == {"g01": 0, "g02": 0, "g03": 0}, took
    starts = [n for what, n in RecordingBox.events if what == "start"]
    closes = [n for what, n in RecordingBox.events if what == "close"]
    assert len(starts) == 2 and sorted(starts) == sorted(closes), RecordingBox.events
    assert live == set(ids)


def check_the_view_reads_a_live_episode_from_raw():
    """An episode with no trace yet is read from the raw log, output pending.

    The commands are there because log_raw writes the response before sh() agents
    them; the results are not, reaching disk only in the trace.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        assert view.live_index("t") == 1, "a raw log with no trace is an unfinished episode"
        v = view.session_view("t", 1)
        assert (v["source"], v["live"]) == ("raw", True), v["source"]
        assert [c["command"] for t in v["turns"] for c in t["tools"]] == \
            ["cat n1", "echo hi > state/note.txt", "ls state"], v["turns"]
        assert all(c["result"] is None for t in v["turns"] for c in t["tools"]), \
            "no command's output is on disk until the trace is"
        # The agent's environment at episode start is recorded in the trace and nowhere else.
        assert v["observation"]["result"] is None, v["observation"]


def check_the_view_prefers_the_trace_once_it_lands():
    """The same episode, once its trace is written, is read from the trace.

    This is what the page waits for: the source changes, and every command that
    was pending fills in with what it actually returned.
    """
    with temp_root():
        episode_once(*DEFAULT)
        v = view.session_view("t", 1)
        assert (v["source"], v["live"]) == ("trace", False), v["source"]
        assert view.live_index("t") is None, "an episode with a trace is over"
        assert all(c["result"] is not None for t in v["turns"] for c in t["tools"]), \
            "the trace carries what every command returned"
        assert v["observation"]["result"], "and the listing the episode opened on"
        # `since` is what lets a page append rather than download itself again.
        assert view.session_view("t", 1, since=1)["turns"][0]["turn"] == 2


def check_the_view_survives_a_partial_raw_line():
    """A read landing mid-append keeps the whole lines and drops the fragment.

    log_raw appends while the episode runs, so this is an ordinary moment rather
    than a damaged file: the turn comes back on the next poll, whole.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        raw = view.raw_path("t", 1)
        whole = len(view.raw_lines(raw))
        with raw.open("a", encoding="utf-8") as f:
            f.write('{"turn": 4, "received": "2026-01-01T00:00:0')
        assert len(view.raw_lines(raw)) == whole, "the fragment is not a turn yet"
        assert view.session_view("t", 1)["total_turns"] == whole, "and nothing raised"


def check_the_view_costs_a_live_turn_like_the_account():
    """What the view derives mid-episode is what the account commits at the end.

    Scripted with the two turns that make the arithmetic more than addition: a
    replayed response id, which bills nothing, and a fallback at its own rates.
    """
    served = usage(output_tokens=200, iterations=[
        attempt("claude-opus-5", 0), attempt("claude-sonnet-5", 200, kind="fallback_message")])
    with temp_root(MODEL="claude-opus-5"):
        episode_once(run("cat n1"),
                  run("echo hi > state/note.txt", id="twice"),
                  run("ls state", id="twice"),
                  run("cat state/note.txt", u=served),
                  say())
        gt = ground_truth()
        unfinished()
        # The account as it stood at episode start: episode 1 started at the initial balance.
        account = {"model": gt["model"], "remaining": gt["series"][0]}
        turns = view.from_raw(view.latest_attempt(view.raw_lines(view.raw_path("t", 1))), account)

    assert gt["series"][2] == gt["series"][3], "the replayed id has to have billed nothing"
    assert [t["balance"] for t in turns] == gt["series"][1:], \
        f"derived {[t['balance'] for t in turns]} against {gt['series'][1:]}"
    assert sum(t["micros"] for t in turns) == gt["initial"] - gt["remaining"], \
        "and the per-turn costs partition the spend"
    assert [t["served_by_fallback"] for t in turns] == [False, False, False, True, False]


def check_the_view_reads_only_the_last_attempt_at_an_episode():
    """An episode index reused after an episode died shows the attempt still running.

    An index is len(episodes) + 1, so an episode that wrote no trace leaves its own
    free and the next appends to the same log. Both shown would be one episode.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        first = view.raw_lines(view.raw_path("t", 1))
        # A second attempt at the same index, as the next episode would write it.
        with view.raw_path("t", 1).open("a", encoding="utf-8") as f:
            for line in first[:2]:
                f.write(json.dumps(line) + "\n")

        again = view.raw_lines(view.raw_path("t", 1))
        assert len(again) == len(first) + 2, "both attempts are on disk"
        assert [line["turn"] for line in view.latest_attempt(again)] == [1, 2], \
            "and only the last of them is the episode being watched"
        assert view.session_view("t", 1)["total_turns"] == 2


def check_the_view_serves_its_page_and_api():
    """Every route answers, and an agent name off the URL cannot leave records/."""
    with temp_root():
        episode_once(*DEFAULT)
        httpd = view.serve(0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def got(path):
                with urllib.request.urlopen(base + path) as r:
                    return r.status, json.loads(r.read().decode("utf-8"))

            with urllib.request.urlopen(base + "/") as r:
                page = r.read().decode("utf-8")
                assert r.status == 200 and "<title>ClaudeSandbox</title>" in page
            # Self-contained: a page that fetched anything would need a network
            # this project does not give it. Fonts are the standing temptation -
            # named here and left to the machine to have or not, never linked.
            assert "//cdn" not in page and "<script src" not in page, "nothing is fetched"
            # url(#fade) is the gradient this page defines in itself; a url()
            # naming a host or a scheme is the one that leaves.
            for fetches in ("@import", "url(http", "url(//", "url('", 'url("',
                            "<link", "fonts.googleapis"):
                assert fetches not in page, f"the page reaches out with {fetches}"

            assert got("/api/experiments")[1]["experiments"][0]["name"] == "t"
            assert [s["episode"] for s in got("/api/agent/t")[1]["episodes"]] == [1]
            assert got("/api/agent/t")[1]["seat"] == "1"
            assert got("/api/agent/t/episode/1")[1]["source"] == "trace"
            # An agent driven on its own is an experiment of one: its private store, the
            # one balance that goes with the seat it holds, and a ledger with
            # nothing in it. Nothing to address and nobody to be addressed by.
            private = got("/api/experiment/t/tree/notes")[1]["columns"]
            assert [c["agent"] for c in private] == ["t"]
            assert [f["path"] for f in private[0]["files"]] == ["note.txt"]
            head = got("/api/experiment/t")[1]
            assert [s["n"] for s in head["seats"]] == [ground_truth()["series"][-1]]
            assert head["ledger"] == [], "an experiment of one has given nothing to anyone"
            assert got("/api/experiment/t/file?agent=t&kind=notes&path=note.txt")[1]["text"] == "hi\n"

            for bad in ("/api/agent/nope", "/api/agent/t/episode/99", "/api/nope",
                        "/api/experiment/nope", "/api/experiment/t/tree/nope",
                        # A name off the URL reaches the filesystem only after
                        # matching a listing, so a path cannot be walked out of
                        # the tree by asking for one.
                        "/api/experiment/t/file?agent=t&kind=notes&path=../../account.json",
                        "/api/experiment/t/file?agent=nope&kind=notes&path=note.txt"):
                try:
                    got(bad)
                except urllib.error.HTTPError as e:
                    assert e.code == 404, (bad, e.code)
                else:
                    raise AssertionError(f"answered for {bad}")
        finally:
            httpd.shutdown()
            httpd.server_close()


def check_the_view_never_writes():
    """Reading an agent leaves every byte of it where it was.

    The whole design rests on this: the view is display only, in the same
    category as --watch, and an agent must not be able to tell it was watched.
    """
    def digest(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    def sweep():
        c = view.experiment_of("t")
        view.header(c)
        view.messages(c)
        for kind in ("blackboard", "notes"):
            view.tree_view(c, kind)
        view.file_view("t", "notes", "note.txt")
        view.agent_view("t")
        view.session_view("t", 1)

    with temp_root() as root:
        episode_once(*DEFAULT)
        finished = digest(root)
        sweep()
        settled = digest(root)
        # And again with the episode unfinished, which is the path that reads
        # the raw log and derives rather than reading a field.
        unfinished()
        running = digest(root)
        sweep()
        watched = digest(root)

    assert settled == finished, sorted(set(settled) ^ set(finished)) or "contents changed"
    assert watched == running, sorted(set(watched) ^ set(running)) or "contents changed"


def check_the_view_names_a_set_of_agents():
    """An experiment names itself; an agent started alone is named by its id's letters.

    Every member has to arrive at the same name, or the sidebar's filter splits
    one experiment across sets, so the name comes from the membership.
    """
    seats = {"1": "q01", "2": "q02", "3": "q03"}
    peers = {"q01": seats, "q02": seats, "q03": seats}
    named = {agent: view.group_of(agent, {"peers": {"seen": seen}}) for agent, seen in peers.items()}
    assert set(named.values()) == {"q"}, named

    # No prefix in common: still one set, and still one name for it.
    both = [view.group_of("alpha", {"peers": {"seen": {"2": "beta"}}}),
            view.group_of("beta", {"peers": {"seen": {"1": "alpha"}}})]
    assert both == ["alpha+beta", "alpha+beta"], both

    # Started alone, with no experiment to ask: live01 and live02 sit together.
    assert [view.group_of(r, {}) for r in ("live01", "live02", "b01s", "solo")] == \
        ["live", "live", "b", "solo"]


@contextlib.contextmanager
def two_seats():
    """An experiment of two, laid out and seated, with a blackboard and a store each."""
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-view-") as tmp:
        harness.ROOT = Path(tmp)
        ids = experiment_of(Path(tmp),
                        g01={"NOTES.md": "given\n", "secret.md": "mine\n",
                             "group/msg": "hello 2\n"},
                        g02={"secret.md": "theirs\n", "group/msg": "hello 1\n"})
        seats = experiment.mapping(ids)
        with quiet():
            for agent, series in (("g01", [1000, 900]), ("g02", [1000, 800])):
                account = harness.load_account(agent)
                account["seat"] = next(i for i, r in seats.items() if r == agent)
                account["peers"] = {"seen": seats}
                account["series"] = series
                account["remaining"] = series[-1]
                account["starter_files_landed"] = {"name": "objective-notes", "paths": ["NOTES.md"]}
                harness.save_account(agent, account)
        yield view.experiment_of("g01"), seats


def check_the_view_shows_every_seat_side_by_side():
    """A tab is one tree of every seat's environment, a column each, in seat order.

    The columns are what makes an experiment readable. A peer's private store is in
    no column of the blackboards and no column but its own of the stores.
    """
    with two_seats() as (c, seats):
        assert c["seated"] and c["seats"] == seats, (c["seated"], c["seats"])
        group_files = view.tree_view(c, "blackboard")["columns"]
        stores = view.tree_view(c, "notes")["columns"]
        opened = view.file_view("g02", "blackboard", "msg")

    # Seat order, and every seat present: the numbering is absolute, so column 2
    # is g02 to whoever is reading and not the second one they were shown.
    assert [(col["seat"], col["agent"]) for col in group_files] == [("1", "g01"), ("2", "g02")]
    assert [(col["seat"], col["agent"]) for col in stores] == [("1", "g01"), ("2", "g02")]
    assert [[f["path"] for f in col["files"]] for col in group_files] == [["msg"], ["msg"]]
    assert [[f["path"] for f in col["files"]] for col in stores] == \
        [["NOTES.md", "secret.md"], ["secret.md"]]
    # A listing is a listing: a file is read when it is opened, not on every poll.
    assert all("text" not in f for col in group_files + stores for f in col["files"])
    assert opened["text"] == "hello 1\n", "a blackboard is read from the agent that owns it"
    # starter is the starter files alone, and it is a fact about the private store.
    assert [f["path"] for f in stores[0]["files"] if f["starter"]] == ["NOTES.md"]
    assert not [f for col in group_files for f in col["files"] if f["starter"]]
    assert [f["path"] for col in stores for f in col["files"] if f["path"] == "secret.md"] == \
        ["secret.md", "secret.md"], "each store holds its own, and neither holds the other's"


def check_the_view_shows_every_balance_from_its_own_account():
    """A seat's n is that agent's ground truth, read from no file in any environment.

    A peer's figure is as authoritative as the watched agent's, both coming from
    the account the harness plants from: 800 is g02's, in no file g01 can read.
    """
    with two_seats() as (c, _):
        h = view.header(c)

    assert [(s["seat"], s["agent"], s["n"]) for s in h["seats"]] == \
        [("1", "g01", 900), ("2", "g02", 800)]
    assert h["ledger"] == [], "an experiment that has given nothing has an empty ledger"
    assert h["round"] == 0, "no episode has been committed, so no round has been taken"
    assert [s["transfer"] for s in h["seats"]] == [None, None], "and nobody has declared one"
    assert h["starter_files"] == "objective-notes" and h["seated"]


def check_the_view_cuts_a_round_where_an_agent_repeats():
    """A round is read back out of the order the episodes started in.

    experiment.py writes no round anywhere, and an agent that sits one out falls behind
    for good. One episode per agent per round is the cut: an agent acting twice.
    """
    # Rotated the way experiment.order rotates, with seat 2 sitting round 3 out.
    # The last two share an episode second, which a round has to survive.
    acted = [("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
             ("g02", "00:04"), ("g03", "00:05"), ("g01", "00:06"),
             ("g03", "00:07"), ("g01", "00:08"),
             ("g01", "00:09"), ("g02", "00:10"), ("g03", "00:10")]
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-round-") as tmp:
        harness.ROOT = Path(tmp)
        seats = experiment.mapping(["g01", "g02", "g03"])
        taken: dict[str, int] = {}
        for agent, at in acted:
            taken[agent] = taken.get(agent, 0) + 1
            p = view.trace_path(agent, taken[agent])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "agent": agent, "episode": taken[agent], "stop": "end_turn", "spent": 1,
                "turns": [], "remaining": 0, "files": [], "state_saved": True,
                "provenance": {"started_at": f"2026-01-01T{at}:00Z", "peers": seats},
            }), encoding="utf-8")
        for seat, agent in seats.items():
            harness.records_dir(agent).mkdir(parents=True, exist_ok=True)
            (harness.records_dir(agent) / "account.json").write_text(json.dumps({
                "agent": agent, "seat": seat, "peers": {"seen": seats},
                "series": [1000], "remaining": 1000, "initial": 1000, "episodes": [],
            }), encoding="utf-8")
        c = view.experiment_named("g")
        rows = view.cohort_sessions(c)
        seen = view.agent_view("g02")["episodes"]

    assert [r["round"] for r in rows] == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4], \
        [(r["agent"], r["round"]) for r in rows]
    assert view.round_now(c, rows) == 4
    # The one the episode index gets wrong: g02 sat round 3 out, so its third
    # episode is round 4 and counting episodes would have called it round 3.
    assert [(s["episode"], s["round"]) for s in seen] == [(1, 1), (2, 2), (3, 4)]
    # Two agents starting in the same second are still one round; only an agent taking
    # a second turn cuts one.
    assert [(r["agent"], r["round"]) for r in rows[-2:]] == [("g02", 4), ("g03", 4)]


def check_the_view_tells_a_seat_not_yet_reached_from_one_that_passed():
    """Mid-round, a seat still to come is not a seat that sat the round out.

    experiment.order rotates, so from the traces alone the two look identical until
    the round ends. Nothing here asks whether an agent could have woken.
    """
    def environment(tmp, acted):
        harness.ROOT = Path(tmp)
        seats = experiment.mapping(["g01", "g02", "g03"])
        taken: dict[str, int] = {}
        for agent, at in acted:
            taken[agent] = taken.get(agent, 0) + 1
            p = view.trace_path(agent, taken[agent])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "agent": agent, "episode": taken[agent], "stop": "end_turn", "spent": 1,
                "turns": [], "remaining": 0, "files": [], "state_saved": True,
                "provenance": {"started_at": f"2026-01-01T{at}:00Z", "peers": seats},
            }), encoding="utf-8")
        for seat, agent in seats.items():
            harness.records_dir(agent).mkdir(parents=True, exist_ok=True)
            (harness.records_dir(agent) / "account.json").write_text(json.dumps({
                "agent": agent, "seat": seat, "peers": {"seen": seats},
                # Nothing left, which is the whole point: a seat still to come
                # reads that way on the balance that would stop it starting.
                "series": [0], "remaining": 0, "initial": 1000, "episodes": [],
            }), encoding="utf-8")
        c = view.experiment_named("g")
        rows = view.cohort_sessions(c)
        rnd = view.round_now(c, rows)
        return rnd, {agent: view.seat_row(seat, agent, rows, rnd)
                     for seat, agent in view.places_of(c)}

    # Round 3 open, g01 has taken it and the other two have not been reached.
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-pending-") as tmp:
        open_rnd, open_seats = environment(tmp, [
            ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
            ("g01", "00:04"), ("g02", "00:05"), ("g03", "00:06"),
            ("g01", "00:07")])
    # The same shape with g03 having missed round 2, so round 3 observation finds it
    # a round behind rather than a round late.
    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-passed-") as tmp:
        past_rnd, past_seats = environment(tmp, [
            ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
            ("g01", "00:04"), ("g02", "00:05"),
            ("g01", "00:06")])

    assert open_rnd == 3 and past_rnd == 3, (open_rnd, past_rnd)
    assert open_seats["g01"]["acted"] and not open_seats["g01"]["pending"], \
        "the seat that took the round is neither waiting nor out of it"
    for agent in ("g02", "g03"):
        assert open_seats[agent]["pending"] and not open_seats[agent]["acted"], \
            f"{agent} has not been reached in the open round and has not sat it out"
    # A balance of zero is not what decides it: g02 is still to come on the same
    # empty account that g03 has passed on.
    assert past_seats["g03"]["round"] == 1 and not past_seats["g03"]["pending"], \
        "a seat a whole round behind has been asked and passed"
    assert past_seats["g02"]["pending"], "g02 acted in round 2 and is still to come"


def check_the_view_reads_a_message_out_of_two_outboxes():
    """The log is the difference between one episode's outbox and the last.

    An outbox is a standing mirror rather than a queue, so leaving a message
    re-sends it. None of that is recorded: four episodes of one directory.
    """
    with temp_root() as root:
        seated(root, g02={})
        # Each episode leaves its blackboard something new as well, so the post
        # penalty does not halve the budget four times over what is being read.
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        episode_once(run("echo louder > out/2", "echo r2 >> 1/log"), say())
        episode_once(run("printf '2 10\\n' > out/transfer", "echo r3 >> 1/log"), say())
        episode_once(run("rm -f out/2", "echo r4 >> 1/log"), say())
        c = view.experiment_of("t")
        m = view.messages(c)
        seen = [e for e in m["events"] if e["path"] == "out/2"]
        transfers = [e for e in m["events"] if e["kind"] == "transfer"]

    assert [e["change"] for e in seen] == ["sent", "edited", "standing", "withdrawn"], \
        [(e["round"], e["change"]) for e in seen]
    assert {e["to_seat"] for e in seen} == {"2"} and {e["to_run"] for e in seen} == {"g02"}
    assert [e["from_seat"] for e in seen] == ["1"] * 4, "and every one of them is seat 1's"
    assert [e["round"] for e in seen] == [1, 2, 3, 4]
    assert seen[0]["text"] == "hello\n" and seen[1]["text"] == "louder\n"
    assert any(l.startswith("-hello") for l in seen[1]["diff"]), seen[1]["diff"]
    assert not seen[3]["text"], "a withdrawn message has no text to show"
    # The declaration is in the same list, because it is written and withdrawn
    # the same way - and it carries resolve_transfer's verdict, which is the only
    # place a declaration that moved nothing ever says why.
    assert [e["change"] for e in transfers] == ["sent", "standing"], transfers
    assert transfers[0]["transfer"]["amount"] == 10 and transfers[0]["transfer"]["error"] is None
    assert transfers[0]["to_seat"] == "2" and transfers[0]["delivered"] is None, \
        "a transfer reaches nobody in particular: what it moved is in g, which all read"
    assert m["committed"] == len(m["events"]) and not m["tip"]


def check_the_view_shows_what_the_receiver_has_not_seen_yet():
    """The outbox on disk agents ahead of the last trace, and the log says so.

    The files are mirrored back before the trace is written, and an episode that
    died writes no trace at all. What stands now is shown as standing now.
    """
    with temp_root() as root:
        seated(root, g02={}, g03={})
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        before = view.messages(view.experiment_of("t"))
        later = harness.outbox_dir("t") / "3"
        later.write_text("written behind the harness\n", encoding="utf-8")
        after = view.messages(view.experiment_of("t"))

    assert not before["tip"], "nothing stands ahead of the trace that was just written"
    assert [e["path"] for e in after["tip"]] == ["out/3"]
    tip = after["tip"][0]
    assert tip["round"] is None and tip["tip"] and tip["change"] == "sent"
    assert tip["delivered"] is None, "nobody has woken to it, so nobody has been given it"
    assert after["events"] == before["events"], "and what is committed did not move"


def check_the_view_reads_delivery_off_the_observation():
    """A message is delivered by m, so no command has to name the inbox.

    The initial observation carries in/<sender>, which is the addressee holding it. Reading
    delivery off the commands instead would call a delivered message unread.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        with quiet():
            got = harness.run_once("other", fake(run("cat n2"), say()))
        m = view.messages(view.experiment_of("t"))
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    assert "=== in/1 ===" in got["observation"], "the addressee started holding the message"
    assert "in/1" not in " ".join(got["commands"]), "and named it in no command of its own"
    d = ev["delivered"]
    assert d["shown_before"] is True, "which is the whole of what delivery is now"
    assert (d["environment"], d["named"], d["clipped"]) == (True, False, False), d


def check_the_view_says_delivery_where_the_observation_carried_nothing():
    """An observation that is the listing alone leaves the inbox to be fetched.

    Delivery is a different fact under that arrangement, and the log says so
    rather than answering the question it can answer here as if it were asked.
    """
    def trace(agent_id: str, at: str, observation: str, files: list[dict],
              commands: list[str]) -> None:
        p = view.trace_path(agent_id, 1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "agent": agent_id, "episode": 1, "stop": "end_turn", "spent": 1, "turns": [],
            "remaining": 0, "state_saved": True, "observation": observation,
            "commands": commands, "files": files,
            "provenance": {"started_at": f"2026-01-01T{at}:00Z"},
        }), encoding="utf-8")

    with pinned(), tempfile.TemporaryDirectory(prefix="mtr-nocarry-") as tmp:
        harness.ROOT = Path(tmp)
        seats = experiment.mapping(["g01", "g02"])
        trace("g01", "00:01", "total 0\n. ..\n",
              [{"path": "out/2", "channel": "outbox", "text": "hello\n",
                "size": 6, "ours": True, "starter": False}], ["ls -la . ./state"])
        trace("g02", "00:02", "total 0\n. ..\n",
              [{"path": "in/1", "channel": "inbox", "text": "hello\n",
                "size": 6, "ours": False, "starter": False}], ["ls -la . ./state"])
        for seat, agent_id in seats.items():
            harness.records_dir(agent_id).mkdir(parents=True, exist_ok=True)
            (harness.records_dir(agent_id) / "account.json").write_text(json.dumps({
                "agent": agent_id, "seat": seat, "peers": {"seen": seats},
                "series": [1000], "remaining": 1000, "initial": 1000, "episodes": [],
            }), encoding="utf-8")
        m = view.messages(view.experiment_named("g"))
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    d = ev["delivered"]
    assert d["shown_before"] is None, "nothing was shown_before, so it answers nothing"
    assert d["environment"] is True, "and what it could have fetched is still a fact"
    assert (d["named"], d["clipped"]) == (False, False), d


def check_the_view_splits_the_observation_into_the_listing_and_the_rest():
    """The two halves of an episode, apart, and reassembling into what was sent.

    The pieces are the page's, so what it shows a reader has to be the bytes the
    model got and not a rendering of them.
    """
    with temp_root() as root:
        seated(root, other={})
        t = episode_once(*DEFAULT)
        o = view.session_view("t", 1)["observation"]

    assert o["listing"] + "".join(f"=== {s['path']} ===\n{s['text']}" for s in o["shown_before"]) \
        == t["observation"], "the halves are the whole observation and nothing else"
    assert "=== " not in o["listing"], "the listing ends where the first section starts"
    paths = [s["path"] for s in o["shown_before"]]
    assert harness.DIGEST_NAME not in paths, "m names its own sections and never itself"
    assert "n1" in paths and harness.LEDGER_NAME in paths, paths
    assert all(s["bytes"] == len(s["text"].encode("utf-8")) for s in o["shown_before"])
    assert o["clipped"] is False and o["name"] == harness.DIGEST_NAME


def check_the_view_carries_every_provenance_field_the_trace_holds():
    """Everything drift can name reaches the page, because nothing is picked.

    drift() reports on every key provenance() writes, so a panel holding a list
    of its own can name a field in a banner and then not show it.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(*DEFAULT)
        seen = view.agent_view("t")["episodes"][0]["provenance"]
        want = harness.provenance(harness.MODEL)

    assert set(want) <= set(seen), sorted(set(want) - set(seen))
    assert {"digest_file_limit", "observation_limit"} <= set(seen), "the two that decide what m carries"
    # The page picks no field, so no field can be left behind by one.
    assert not [k for k in want if f'"{k}"' in view.PAGE], \
        "the provenance panel iterates what the trace holds and names nothing"


def check_the_view_states_an_obligation_the_grace_waived():
    """What an episode owed and what it was charged are two questions.

    A share is taken only past the grace, so an episode inside it can leave all
    three undone for nothing. Every pane says the same thing about that episode.
    """
    with temp_root(GRACE_EPISODES=1, BLACKBOARD_SILENCE_PENALTY_PERCENT=50,
                   MAILBOX_SILENCE_PENALTY_PERCENT=50, TRANSFER_SILENCE_PENALTY_PERCENT=50) as root:
        seated(root, other={})
        t = episode_once(run("cat n1"), say())
        v = view.session_view("t", 1)
        mine = next(s for s in view.header(view.experiment_of("t"))["seats"] if s["agent"] == "t")

        assert v["obligations"] == {"posted": False, "messaged": False, "transferred": False}, \
            v["obligations"]
        assert (t["blackboard_penalised"], t["mailbox"]["penalty"], t["transfer"]["penalty"]) == (0, 0, 0), \
            "and the grace charged it for none of them"
        assert v["messages_why"] == "no message", v["messages_why"]
        # The tile and the transcript answer from one place, so neither can
        # state an obligation the other leaves out.
        assert (mine["posted"], mine["messaged"]) == (False, False), mine


def check_the_view_counts_what_a_seat_spent_rather_than_what_it_lost():
    """A tile's spend is its turns, and the bar beside it is the balance.

    A transfer, a share taken and a floor all move the balance without being spend,
    so the drop from initial answers a different question and can be larger.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo '2 100' > out/transfer", "echo r1 >> 1/log"), say())
        mine = next(s for s in view.header(view.experiment_of("t"))["seats"] if s["agent"] == "t")
        gt = ground_truth("t")

    assert mine["spent"] == sum(s["spent"] for s in gt["episodes"]), \
        (mine["spent"], [s["spent"] for s in gt["episodes"]])
    assert mine["spent"] != gt["initial"] - gt["remaining"], \
        "the transfer moved the balance without being spent"
    assert mine["spent_this_round"] <= mine["spent"], "a round is part of a life"
    assert mine["rebated"] == gt["rebated"] > 0, "and what it won back is on the tile"


# --- runner -----------------------------------------------------------------


def checks() -> dict:
    """Every check in the module, by the label the runner prints."""
    return {name[6:]: fn for name, fn in sorted(globals().items())
            if name.startswith("check_")}


def episodes_in(fn: Callable) -> int:
    """Roughly how many episodes a check agents, read off its own source.

    Sorting by this puts the heavy checks in while workers are free. Crude on
    purpose: nothing is asserted on it, so being wrong costs only wall clock.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return 1
    n = len(re.findall(r"\b(?:episode_once|harness\.run_once)\(", src))
    # The ceiling asked of run_episodes. `.` rather than `[^)]` because the
    # argument before it is itself a call, and its bracket is not the one here.
    n += sum(int(c) for c in re.findall(r"\brun_sessions\(.*?,\s*(\d+)\s*\)", src))
    # An episode inside a loop costs once an iteration. Only the two forms that
    # state their own length are read; anything else counts as written.
    for over in re.findall(r"\bfor\s+\w+\s+in\s+(.+?):", src):
        if "harness.PRICES" in over:
            n += len(harness.PRICES)
        elif m := re.search(r"\brange\((\d+)\)", over):
            n += int(m.group(1))
    return max(1, n)


def run_one(label: str) -> tuple[str, str, str]:
    """Agent one check and say how it went, in data a worker can send home.

    The traceback is formatted here rather than raised: an assertion carrying an
    arbitrary object does not always survive the trip between processes.
    """
    try:
        checks()[label]()
    except Skip:
        return label, "skip", ""
    except BaseException:
        return label, "fail", traceback.format_exc()
    return label, "ok", ""


def configure(real: bool, docker: bool, suite: int) -> None:
    """Set up a process to run checks in. Called in the parent and every worker.

    Each worker gets its own container prefix and carries `suite`, so no worker
    reaps another's. The Docker answer is carried in rather than asked again.
    """
    global REAL_ONLY, _DOCKER, SUITE
    REAL_ONLY, _DOCKER, SUITE = real, docker, suite
    harness.CONTAINER_PREFIX = f"mtr-w{suite}-{os.getpid()}-"


def sweep_filter(suite: int | None = None) -> str:
    """The name a container has to contain to be this suite's to remove.

    A docker name filter matches anywhere, so this has to be a string nothing
    else can contain. The suite's pid is what makes it one.
    """
    return f"mtr-w{SUITE if suite is None else suite}-"


def sweep(everyones: bool = False) -> None:
    """Remove any container a worker of this suite died holding.

    `everyones` widens that to every suite's, collecting what an agent killed
    outright left behind. Opt-in: the only mode reaching another process's.
    """
    if not shutil.which("docker"):
        return
    # Two numbers and two dashes: a suite's worker. A real agent reaches mtr-w
    # only as its own name, mtr-w01-0001, which has one number and cannot match.
    name = r"mtr-w[0-9]+-[0-9]+-" if everyones else sweep_filter()
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"],
                          capture_output=True, text=True).stdout.split()
    if left:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
        print(f"swept {len(left)} leaked container(s)")


def main(argv: list[str] | None = None) -> int:
    """Agent the checks, in as many processes as asked for."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("patterns", nargs="*", help="only checks whose name contains one of these")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="how many checks to run at once (1 to run them in this process)")
    p.add_argument("--real", action="store_true",
                   help="run every check in a container, including the ones that need not be")
    p.add_argument("--no-docker", action="store_true", help="skip the checks that need a container")
    p.add_argument("--list", action="store_true", help="print the check names and stop")
    p.add_argument("--sweep-all", action="store_true",
                   help="also remove containers left by other suite agents, including dead ones")
    args = p.parse_args(argv)

    chosen = [l for l in checks()
              if not args.patterns or any(pat in l for pat in args.patterns)]
    if args.list:
        print("\n".join(chosen))
        return 0
    if not chosen:
        print(f"no check matches {args.patterns}")
        return 2

    # Asked once for the whole agent, and handed to every worker: docker info is
    # slower than most of the checks that depend on the answer.
    available = False if args.no_docker else docker_ready()
    configure(args.real, available, os.getpid())

    started = time.time()
    failed, skipped = [], []

    def record(label, how, detail):
        if how == "skip":
            skipped.append(label)
            print(f"SKIP  {label}", flush=True)
        elif how == "fail":
            failed.append(label)
            print(f"FAIL  {label}\n{detail}", flush=True)
        else:
            print(f"ok    {label}", flush=True)

    jobs = max(1, min(args.jobs, len(chosen)))
    # In a finally: a worker dying is what leaves a container behind, and it is
    # also what breaks the pool and ends this function early. Swept only on the
    # way out of a clean agent, the sweep would miss every case it is here for.
    try:
        if jobs == 1:
            for label in chosen:
                record(*run_one(label))
        else:
            # Heaviest first, and only here: `checks()` stays alphabetical
            # because --list and the name filters read it. A check submitted
            # last is one the agent cannot end before, however many workers are
            # idle by then.
            table = checks()
            queue = sorted(chosen, key=lambda l: -episodes_in(table[l]))
            with futures.ProcessPoolExecutor(
                    max_workers=jobs, initializer=configure,
                    initargs=(args.real, available, SUITE)) as pool:
                for done in futures.as_completed(
                        [pool.submit(run_one, l) for l in queue]):
                    record(*done.result())
    finally:
        sweep(args.sweep_all)
    if skipped:
        print(f"\n{len(skipped)} skipped (needs Docker + the image)")
    print(f"{len(chosen)} checks in {time.time() - started:.1f}s across {jobs} process(es)")
    if failed:
        print(f"\nFAILED: {', '.join(sorted(failed))}")
    else:
        print("\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
