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
from typing import Any, Iterable


VERSION = 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
PHASES = ("preparing_round", "waiting_autonomous", "waiting_player", "resolving_actions",
          "settling_round", "round_completed", "completed", "interrupted", "cost_ceiling")
_LOCK = threading.Lock()


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


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def progress(runtime_root: Path, experiment_id: str, phase: str, round_number: int,
             detail: dict[str, Any] | None = None) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown progress phase {phase!r}")
    path = directory(runtime_root, experiment_id) / "progress.json"
    with _LOCK:
        previous = read(path) or {"version": VERSION, "experiment_id": experiment_id, "events": []}
        event = {"at": now(), "phase": phase, "round": round_number, **(detail or {})}
        record = {**previous, "latest": event, "events": [*previous.get("events", []), event]}
        atomic(path, record)
    return record


def cost(agents: Iterable[str], load_account, policy: dict[str, Any]) -> dict[str, Any]:
    spent = 0
    tiers = {}
    for agent in agents:
        account = load_account(agent)
        if account.get("provider") != "human":
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


def trace_evidence(runtime_root: Path, agent: str, episode: int) -> dict[str, Any] | None:
    path = Path(runtime_root) / "records" / agent / "traces" / f"episode-{episode:04d}.json"
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {"agent": agent, "episode": episode, "trace_sha256": hashlib.sha256(data).hexdigest()}


def outcome(runtime_root: Path, experiment_id: str, agents: list[str], labels: dict[str, str],
            live: set[str], reason: str, load_account, reveal: dict[str, list[str]]) -> dict[str, Any]:
    label_by_agent = {agent: labels[str(index)] for index, agent in enumerate(agents, 1)}
    accounts = {agent: load_account(agent) for agent in agents}
    eliminated = sorted((account["eliminated"]["round"], label_by_agent[agent],
                         account["eliminated"].get("reason", ""))
                        for agent, account in accounts.items() if account.get("eliminated"))
    survivors = [label_by_agent[agent] for agent in agents if agent in live]
    winners = survivors if reason in ("one_remains", "final_tie") else []
    evidence = [item for agent, account in accounts.items()
                if (item := trace_evidence(runtime_root, agent, len(account.get("episodes", []))))]
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


def records(runtime_root: Path, experiment_id: str) -> dict[str, Any]:
    try:
        base = directory(runtime_root, experiment_id)
    except ValueError:
        return {"progress": None, "outcome": None, "lineage": None}
    return {"progress": read(base / "progress.json"), "outcome": read(base / "outcome.json"),
            "lineage": read(base / "lineage.json")}
