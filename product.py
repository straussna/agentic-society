"""Durable experiment-level progress, cost, lineage, and outcome records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable


VERSION = 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
PHASES = ("preparing_round", "waiting_autonomous", "waiting_player", "resolving_actions",
          "settling_round", "round_completed", "completed", "interrupted", "cost_ceiling")
_LOCK = threading.Lock()

# replace's retry policy for a rename that finds the target open.
RENAME_ATTEMPTS = 5
RENAME_WAIT_S = 0.05


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def root(runtime_root: Path) -> Path:
    return Path(runtime_root) / "experiment_records"


def directory(runtime_root: Path, experiment_id: str) -> Path:
    if not IDENTIFIER.fullmatch(experiment_id) or experiment_id in (".", ".."):
        raise ValueError(f"invalid experiment_id {experiment_id!r}")
    return root(runtime_root) / experiment_id


def displace(runtime_root: Path, experiment_id: str, bundle: Path | None = None) -> Path | None:
    """Preserve an existing experiment record before a fresh run reuses its identity."""
    source = directory(runtime_root, experiment_id)
    if not source.exists():
        return None
    if bundle is None:
        bundle = Path(runtime_root) / "displaced" / f"product-{time.time_ns()}"
    destination = bundle / "experiment_records" / experiment_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def replace(source: Path, destination: Path) -> None:
    """os.replace, retried a few times on Windows, where a reader holding `destination`
    open makes the rename fail with PermissionError for as long as the read takes."""
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(RENAME_WAIT_S)


def atomic(path: Path, value: dict[str, Any]) -> None:
    """Write `value` as compact JSON: a synced temporary file beside `path`, renamed over it.

    A reader sees the old file or the new one whole, and a write that fails leaves
    neither a partial file nor its temporary behind.
    """
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def append(path: Path, values: Iterable[dict[str, Any]]) -> None:
    """Append each of `values` to `path` as one line of compact JSON, synced before returning.

    A crash can cut the last line short. The next append starts a line of its own, so
    the cut costs that one line and no other.
    """
    data = b"".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    + b"\n" for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if end := handle.seek(0, os.SEEK_END):
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                data = b"\n" + data
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read(path: Path) -> dict[str, Any] | None:
    """The JSON object in `path`, or None when there is no file there.

    A file that is there and does not read, does not parse, or holds something other
    than an object raises ValueError naming it, so it is never taken for an absent one.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} is not a readable JSON object: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a readable JSON object: it holds "
                         f"{type(value).__name__}")
    return value


def logged(path: Path) -> list[dict[str, Any]]:
    """Every object append() wrote to `path`, in order; none when there is no file.

    A line a crash cut short does not parse and is skipped. A file that is there and
    does not read raises ValueError naming it.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as e:
        raise ValueError(f"{path} does not read: {e}") from e
    found = []
    for line in data.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def progress(runtime_root: Path, experiment_id: str, phase: str, round_number: int,
             detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record that the experiment has reached `phase`, and return the event.

    The event is appended to progress.jsonl, and progress.json is rewritten to hold the
    header and this event as the latest. A progress.json that does not read stops the
    write, and is left as it is.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown progress phase {phase!r}")
    base = directory(runtime_root, experiment_id)
    with _LOCK:
        event = {"at": now(), "phase": phase, "round": round_number, **(detail or {})}
        previous = read(base / "progress.json") or {}
        log = base / "progress.jsonl"
        # A progress.json that holds its own events list starts the log with them.
        carried = [] if log.exists() else previous.get("events", [])
        append(log, [*carried, event])
        atomic(base / "progress.json",
               {"version": VERSION, "experiment_id": experiment_id, "latest": event})
    return event


def cost(agents: Iterable[str], load_account, policy: dict[str, Any],
         interactive: Callable[[str | None], bool]) -> dict[str, Any]:
    spent = 0
    tiers = {}
    for agent in agents:
        account = load_account(agent)
        if not interactive(account.get("provider")):
            spent += sum(int(episode.get("spent", 0)) for episode in account.get("episodes", []))
        tiers[agent] = (account.get("product") or {}).get("quality_tier", "standard")
    maximum = policy.get("maximum")
    reserve = int(policy.get("reserved_completion", 0))
    warning = policy.get("warning")
    return {"autonomous_spend": spent, "maximum": maximum,
            "reserved_completion": reserve, "warning": warning,
            "warning_reached": warning is not None and spent >= warning,
            "ceiling_reached": maximum is not None and spent + reserve >= maximum,
            "policy": policy.get("ceiling_policy", "stop"), "quality_tiers": tiers}


def trace_evidence(agent: str, episode: int, path: Path) -> dict[str, Any] | None:
    """An agent's episode and the digest of its trace at `path`; None when that does
    not read."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {"agent": agent, "episode": episode, "trace_sha256": hashlib.sha256(data).hexdigest()}


def outcome(runtime_root: Path, experiment_id: str, agents: list[str], labels: dict[str, str],
            live: set[str], reason: str, load_account: Callable[[str], dict[str, Any]],
            trace_path: Callable[[str, int], Path],
            reveal: dict[str, list[str]]) -> dict[str, Any]:
    """Write outcome.json: who won or survived, the elimination order, the balances, and
    the digest of each agent's last trace, which `trace_path` locates."""
    label_by_agent = {agent: labels[str(index)] for index, agent in enumerate(agents, 1)}
    accounts = {agent: load_account(agent) for agent in agents}
    last = {agent: len(account.get("episodes", [])) for agent, account in accounts.items()}
    eliminated = sorted((account["eliminated"]["round"], label_by_agent[agent],
                         account["eliminated"].get("reason", ""))
                        for agent, account in accounts.items() if account.get("eliminated"))
    survivors = [label_by_agent[agent] for agent in agents if agent in live]
    winners = survivors if reason in ("one_remains", "final_tie") else []
    evidence = [item for agent, episode in last.items()
                if (item := trace_evidence(agent, episode, trace_path(agent, episode)))]
    record = {"version": VERSION, "experiment_id": experiment_id, "recorded_at": now(),
              "termination_reason": reason, "winners": winners, "survivors": survivors,
              "draw": reason == "final_tie", "elimination_order": [
                  {"round": rnd, "seat": seat, "reason": why} for rnd, seat, why in eliminated],
              "scores": {label_by_agent[a]: accounts[a].get("remaining", 0) for a in agents},
              "resources": {label_by_agent[a]: {"micro_dollars": accounts[a].get("remaining", 0)}
                            for a in agents},
              "evidence": evidence, "reveal": reveal}
    atomic(directory(runtime_root, experiment_id) / "outcome.json", record)
    return record


def revealed(record: dict[str, Any], phase: str, experimenter: bool = False) -> dict[str, Any]:
    """Project an outcome through its declared audience and lifecycle policy."""
    if experimenter:
        return dict(record)
    if phase not in ("during_play", "at_elimination", "at_completion"):
        raise ValueError(f"unknown reveal phase {phase!r}")
    policy = record.get("reveal") or {}
    fields = set(policy.get("during_play", []))
    if phase != "during_play":
        fields.update(policy.get(phase, []))
    public = {key: record[key] for key in fields if key in record}
    return {"version": record.get("version"), "experiment_id": record.get("experiment_id"),
            **public}


def recorded_progress(base: Path) -> dict[str, Any] | None:
    """progress.json with its history as `events`: the log's, or the list progress.json
    holds itself when there is no log. None when there is no progress.json."""
    record = read(base / "progress.json")
    if record is None:
        return None
    log = base / "progress.jsonl"
    return {**record, "events": logged(log) if log.exists() else record.get("events", [])}


def records(runtime_root: Path, experiment_id: str) -> dict[str, Any]:
    """An experiment's progress, outcome and lineage for display, each None when it is
    absent or does not read."""
    try:
        base = directory(runtime_root, experiment_id)
    except ValueError:
        return {"progress": None, "outcome": None, "lineage": None}

    def shown(record: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
        try:
            return record()
        except ValueError:
            return None

    return {"progress": shown(lambda: recorded_progress(base)),
            "outcome": shown(lambda: read(base / "outcome.json")),
            "lineage": shown(lambda: read(base / "lineage.json"))}
