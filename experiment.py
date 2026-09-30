"""Several agents advancing together, each reading the others' blackboards.

    py -3 experiment.py sandbox -r 20

One round is one episode for each agent; a seat names a blackboard and a balance.
Every run names its manifest: the schedule, the environment, and each agent's terms.
config.toml holds only what is true of every run whatever the experiment."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar

import harness
import product
import providers
from harness import check_keys
from providers import ProviderRouter

T = TypeVar("T")

# A seat is named by its number, and a label defaults to the seat number, so a
# bare number is what a seat is called in every path and file the agents see.
# An agent id must be something else.
BARE_NUMBER = re.compile(r"^\d+$")

# Goes at building an environment before the agent drops out. An environment is
# built from containers and copies of other agents' trees, so a failure can be
# the daemon and not the agent. None of it is billed, so another go spends only time.
ATTEMPTS = 2

# How a round is driven. "sequential": one episode at a time in fixed seat order,
# each episode reading what the ones before it in the round
# wrote. "simultaneous": every environment built before any episode runs, the
# episodes run at once, and the results settle in seat order, so nobody reads
# this round's writes and a transfer made in one round is seen at the next.
SCHEDULES = ("sequential", "simultaneous")
EXPERIMENT_KEYS = {"experiment_id", "schedule", "stop_when_one_remains",
                   "stop_when_two_remain_after_tie", "provider", "model", "agent", "channel",
                   "harness_files", "tool", "cost", "reveal"}

# What a manifest may say about one agent. Everything else an agent is comes from
# the experiment's defaults and config.toml.
AGENT_KEYS = {"id", "label", "seats", "starter_files", "starter_files_below", "budget",
              "provider", "model", "system_prompt", "memory_from", "quality_tier"}

AGENT_TYPES = (("id", str), ("label", str), ("seats", int), ("starter_files", str),
               ("starter_files_below", int), ("budget", int), ("provider", str), ("model", str),
               ("system_prompt", str), ("memory_from", dict), ("quality_tier", str))

MEMORY_FROM_KEYS = {"agent", "episode"}
MEMORY_FROM_TYPES = (("agent", str), ("episode", int))

# What an agent may be called to its peers: one path segment, since it lands in
# paths and file names. The default is the seat number.
LABEL = re.compile(r"^[A-Za-z0-9._-]+$")
COST_KEYS = {"maximum", "reserved_completion", "warning", "ceiling_policy"}
REVEAL_KEYS = {"during_play", "at_elimination", "at_completion", "experimenter_only"}
OUTCOME_FIELDS = {"winners", "survivors", "draw", "elimination_order", "scores", "resources",
                  "termination_reason", "evidence"}


class Manifest(TypedDict):
    """An experiment manifest as load_manifest reads it.

    `overrides` holds every top-level key that is not the experiment's own: the
    defaults for every seat, applied by start(). `agents` holds one [[agent]] table
    per seat, in seat order, and `labels` maps each seat to what its agent is called.
    `channels`, `harness_files` and `tools` are the tables as declared, None where
    the manifest declares none. `sha256` is the digest of the file's bytes.
    """
    schedule: str
    stop_when_one_remains: bool
    stop_when_two_remain_after_tie: bool
    experiment_id: str
    cost: dict[str, Any]
    reveal: dict[str, list[str]]
    overrides: dict[str, Any]
    agents: list[dict[str, Any]]
    labels: dict[str, str]
    channels: list[dict[str, Any]] | None
    harness_files: dict[str, Any] | None
    tools: list[dict[str, Any]] | None
    sha256: str


# What a round tells the experiment's progress record: the phase it has reached, the
# round's number, and the detail of the event. product.progress bound to one record.
Progress = Callable[[str, int, dict[str, Any] | None], object]

# How a round is driven, sequential_round or simultaneous_round. Called with the agents
# in seat order, those still at the table, the round's index, the router, the stamp,
# the labels and the progress record; returns whether any agent took an episode.
Round = Callable[[list[str], set[str], int, ProviderRouter, dict[str, Any] | None,
                  dict[str, str] | None, Progress], bool]


def seats_of(agents: list[str]) -> dict[str, str]:
    """Seat -> agent, for the whole experiment.

    Absolute and complete: a seat means the same agent to every reader, so a note
    citing one resolves the same way for all of them.
    """
    return {str(i): agent for i, agent in enumerate(agents, 1)}


# --- the manifest -------------------------------------------------------------


def shorthand(ids: list[str], schedule: str = "sequential") -> Manifest:
    """A bare manifest for these agents, which is what a round stamps when driven directly."""
    return {"schedule": schedule, "stop_when_one_remains": False,
            "stop_when_two_remain_after_tie": False,
            "experiment_id": "-".join(ids), "cost": {}, "reveal": {},
            "overrides": {}, "agents": [{"id": i, "provider": "anthropic",
                                           "model": "claude-sonnet-5"} for i in ids],
            "labels": {str(n): str(n) for n in range(1, len(ids) + 1)},
            "channels": None, "harness_files": None, "tools": None, "sha256": ""}


def check_ids(ids: list[str], where: str) -> None:
    """Refuse an experiment with no agents, a repeated id, or an id that is a bare number.

    One seat is an experiment: harness.py runs a single agent under the manifest that
    declares its situation, and a seat with no peers reaches nobody.
    """
    if not ids:
        raise SystemExit(f"{where}: an experiment needs at least one [[agent]]")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{where}: an agent is listed twice: {ids}")
    if bad := [i for i in ids if not i or BARE_NUMBER.match(i)]:
        raise SystemExit(f"{where}: agent ids must not be bare numbers, which is what seats are "
                         f"called: {bad}")


def refuser(path: Path) -> Callable[[str], None]:
    """How this manifest refuses: every message led by the file it came from."""
    def refuse(why: str) -> None:
        raise SystemExit(f"{path}: {why}")
    return refuse


def check_agent(path: Path, entry: dict) -> None:
    """Refuse an [[agent]] table with an unknown key, a wrong type, or terms that do not go together."""
    refuse = refuser(path)
    check_keys(refuse, "", entry, AGENT_KEYS, AGENT_TYPES)
    agent = entry.get("id")
    if not isinstance(agent, str) or not agent:
        refuse("every agent needs an id")
    label = entry.get("label")
    if label is not None and (not LABEL.match(label) or label in (".", "..")):
        refuse(f"{agent}: label {label!r} must be letters, digits, '.', '_' or '-'")
    if entry.get("seats", 1) < 1:
        refuse(f"{agent}: seats must be positive, got {entry['seats']}")
    if "memory_from" in entry:
        memory_from = entry["memory_from"]
        check_keys(refuse, f"{agent}: memory_from: ", memory_from,
                   MEMORY_FROM_KEYS, MEMORY_FROM_TYPES)
        source = memory_from.get("agent")
        episode = memory_from.get("episode")
        if not source:
            refuse(f"{agent}: memory_from: agent must name an existing agent")
        if not isinstance(episode, int) or isinstance(episode, bool) or episode < 1:
            refuse(f"{agent}: memory_from: episode must be a positive integer, got {episode!r}")
    harness.validate_terms(str(path), who=agent, provider=entry.get("provider"),
                           model=entry.get("model"), budget=entry.get("budget"),
                           starter_files=entry.get("starter_files"),
                           starter_files_below=entry.get("starter_files_below"))


def expand_agents(definitions: list[dict]) -> list[dict]:
    """Expand each grouped definition into the concrete agents that occupy its seats."""
    agents: list[dict] = []
    for definition in definitions:
        if "seats" not in definition:
            agents.append(dict(definition))
            continue
        count = definition["seats"]
        width = max(2, len(str(count)))
        for ordinal in range(1, count + 1):
            entry = {key: value for key, value in definition.items() if key != "seats"}
            suffix = str(ordinal).zfill(width)
            entry["id"] += suffix
            if "label" in entry:
                entry["label"] += suffix
            agents.append(entry)
    return agents


def manifest_path(named: str) -> Path:
    """Resolve a manifest: a bare name sits under experiments/, anything else is the path given.

    A name with a suffix or a directory in it is a path and is used as written, so a
    manifest anywhere on disk stays reachable. A bare name is looked for in
    experiments/ and then experiments/examples/, and when it is in neither the
    experiments/ candidate is returned for load_manifest to refuse by name.
    """
    given = Path(named)
    if given.suffix or len(given.parts) > 1:
        return given
    here = Path(__file__).resolve().parent / "experiments"
    candidates = (here / f"{named}.toml", here / "examples" / f"{named}.toml")
    return next((c for c in candidates if c.exists()), candidates[0])


def load_manifest(path: Path) -> Manifest:
    """Read an experiment manifest: the schedule, the experiment's defaults, and each agent's terms.

    Returns a Manifest. Unknown keys, wrong types, and terms that do not go together
    exit with a message naming the file, the way load_config does. The overrides' own
    types and ranges are checked when start() applies them, so one set of rules
    holds for both.

    Every manifest declares a system_prompt, at the top level or on each agent. The
    empty string is a declaration and says nothing; an omitted key is refused, so
    what an agent is told is always something the experiment wrote down.
    """
    if not path.exists():
        raise SystemExit(f"{path}: no such manifest")
    data = path.read_bytes()
    try:
        top = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"{path}: not a manifest: {e}") from None

    allowed = EXPERIMENT_KEYS | {t.lower() for t in harness.TREATMENT}

    check_keys(refuser(path), "", top, allowed,
               (("stop_when_one_remains", bool), ("stop_when_two_remain_after_tie", bool)),
               retired=harness.RETIRED,
               elsewhere={t.lower(): harness.NOT_MANIFEST for t in harness.PROCESS},
               expected="schedule, [[agent]] tables, [[channel]] tables, [[tool]] tables, "
                        "[harness_files], and any experiment setting")
    schedule = top.get("schedule", "sequential")
    if schedule not in SCHEDULES:
        raise SystemExit(f"{path}: schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    experiment_id = top.get("experiment_id", path.stem)
    if not isinstance(experiment_id, str) or not product.IDENTIFIER.fullmatch(experiment_id) or \
            experiment_id in (".", ".."):
        raise SystemExit(f"{path}: experiment_id must be letters, digits, '.', '_' or '-'")
    cost = top.get("cost", {})
    if not isinstance(cost, dict):
        raise SystemExit(f"{path}: cost must be a table")
    check_keys(refuser(path), "cost: ", cost, COST_KEYS,
               (("maximum", int), ("reserved_completion", int), ("warning", int),
                ("ceiling_policy", str)))
    if cost.get("maximum", 1) <= 0 or cost.get("reserved_completion", 0) < 0 or \
            cost.get("warning", 0) < 0:
        raise SystemExit(f"{path}: cost amounts must be non-negative and maximum must be positive")
    if cost.get("ceiling_policy", "stop") != "stop":
        raise SystemExit(f"{path}: cost.ceiling_policy must be 'stop'")
    if cost.get("maximum") is not None and (cost.get("reserved_completion", 0) >= cost["maximum"] or
            cost.get("warning", 0) > cost["maximum"]):
        raise SystemExit(f"{path}: cost reserve and warning must fit below maximum")
    reveal = top.get("reveal", {})
    if not isinstance(reveal, dict):
        raise SystemExit(f"{path}: reveal must be a table")
    check_keys(refuser(path), "reveal: ", reveal, REVEAL_KEYS)
    for phase, fields in reveal.items():
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            raise SystemExit(f"{path}: reveal.{phase} must be a list of outcome field names")
        if unknown := set(fields) - OUTCOME_FIELDS:
            raise SystemExit(f"{path}: reveal.{phase} names unknown outcome fields {sorted(unknown)}")
    definitions = top.get("agent")
    if not isinstance(definitions, list) or not all(isinstance(r, dict) for r in definitions):
        raise SystemExit(f"{path}: agents are [[agent]] tables, each with an id")
    for terms in [top, *definitions]:
        starter = terms.get("starter_files")
        if isinstance(starter, str) and starter.startswith(("./", "../", ".\\", "..\\")):
            terms["starter_files"] = str((path.parent / starter).resolve())
    default_provider, default_model = top.get("provider"), top.get("model")
    if (default_provider is None) != (default_model is None):
        raise SystemExit(f"{path}: top-level provider and model must be set together")
    for entry in definitions:
        if "provider" not in entry and default_provider is not None:
            entry["provider"] = default_provider
        if "model" not in entry and default_model is not None:
            entry["model"] = default_model
        if "provider" not in entry or "model" not in entry:
            raise SystemExit(f"{path}: {entry.get('id', 'agent')}: provider and model must resolve "
                             "from top-level defaults or the [[agent]] table")
        check_agent(path, entry)
    agents = expand_agents(definitions)
    ids = [entry["id"] for entry in agents]
    check_ids(ids, str(path))
    if inherited_inside := [entry["id"] for entry in agents
                            if (entry.get("memory_from") or {}).get("agent") in ids]:
        raise SystemExit(f"{path}: memory_from must name an agent outside this experiment; "
                         f"starting fresh would displace the source for {inherited_inside[0]!r}")

    labels: dict[str, str] = {}
    for seat, entry in enumerate(agents, 1):
        label = entry.get("label", str(seat))
        if not isinstance(label, str) or not LABEL.match(label) or label in (".", ".."):
            raise SystemExit(f"{path}: {entry['id']}: label {label!r} must be letters, digits, "
                             f"'.', '_' or '-', and is what names the agent in paths")
        if label in labels.values():
            other = next(a["id"] for a, l in zip(agents, labels.values()) if l == label)
            raise SystemExit(f"{path}: label {label!r} is held by {other!r} and {entry['id']!r}")
        labels[str(seat)] = label

    tables, harness_files, tool_tables = (top.get("channel"), top.get("harness_files"),
                                          top.get("tool"))
    if tables is not None or harness_files is not None or tool_tables is not None:
        # A tool is held against the table this manifest declares and not against
        # the one in force, so it is refused here for the reason start() would
        # refuse it later.
        chans = harness.validate_channels(tables, harness_files, str(path),
                                          tuple(labels.values()))[0]
        if tool_tables is not None:
            harness.validate_tools(tool_tables, chans, str(path))

    # Last, so a manifest with a structural fault is refused for that fault first.
    if "system_prompt" not in top and not all("system_prompt" in e for e in definitions):
        raise SystemExit(f"{path}: declare system_prompt, at the top level or on every "
                         f"[[agent]]. What an agent is told is the experiment's and reaches "
                         f'every request, so it is stated and never inherited; system_prompt = ""'
                         f" is the declaration that says nothing")

    overrides = {k: v for k, v in top.items()
                 if k not in EXPERIMENT_KEYS}
    return {"schedule": schedule,
            "stop_when_one_remains": top.get("stop_when_one_remains", False),
            "stop_when_two_remain_after_tie": top.get("stop_when_two_remain_after_tie", False),
            "experiment_id": experiment_id, "cost": cost, "reveal": reveal,
            "overrides": overrides, "agents": agents, "labels": labels,
            "channels": tables, "harness_files": harness_files, "tools": tool_tables,
            "sha256": hashlib.sha256(data).hexdigest()}


def terms_of(entry: dict) -> dict[str, Any]:
    """One agent's pinned settings as load_account's keywords, None where the manifest is silent."""
    return {k: entry.get(k) for k in ("provider", "model", "budget", "starter_files", "starter_files_below",
                                      "system_prompt")}


def stamp_of(manifest: Manifest) -> dict[str, Any]:
    """What every episode records about how the experiment was driven."""
    return {"schedule": manifest["schedule"],
            "experiment_id": manifest["experiment_id"], "cost": manifest["cost"],
            "stop_when_one_remains": manifest["stop_when_one_remains"],
            "stop_when_two_remain_after_tie": manifest["stop_when_two_remain_after_tie"],
            "manifest_sha256": manifest["sha256"]}


def inherit_memory(entry: dict, account: dict) -> None:
    """Give a fresh agent the exact private memories recorded by another agent's episode.

    Only files owned by write_memory tools cross the boundary. The new agent keeps its
    own account, model, budget, peers and every episode-scoped channel.
    """
    agent = entry["id"]
    declared = entry.get("memory_from")
    recorded = account.get("memory_from")
    if recorded != declared:
        if recorded is not None or account.get("episodes"):
            raise SystemExit(f"agent {agent} was created with memory_from={recorded!r}, and is "
                             f"now asked to run with {declared!r}; start a fresh agent")
        account["memory_from"] = dict(declared) if declared else None
        harness.save_account(agent, account)
    if not declared or account.get("memory_inherited"):
        return

    source, episode = declared["agent"], declared["episode"]
    source_path = harness.trace_path(source, episode)
    if not source_path.exists():
        raise SystemExit(f"agent {agent}: memory_from names {source!r} episode {episode}, but "
                         f"{source_path} does not exist")
    trace = json.loads(source_path.read_text(encoding="utf-8"))
    if trace.get("trace_version") != harness.TRACE_VERSION:
        raise SystemExit(f"agent {agent}: memory_from source {source!r} episode {episode} has "
                         f"trace version {trace.get('trace_version')!r}, expected {harness.TRACE_VERSION}")
    if not trace.get("state_saved"):
        raise SystemExit(f"agent {agent}: memory_from source {source!r} episode {episode} did not "
                         "save its state")

    source_tools = {tool.name: tool for tool in harness.tools_from(
        (trace.get("provenance") or {}).get("tools")) if tool.kind == "write_memory"}
    target_tools = {tool.name: tool for tool in harness.tools() if tool.kind == "write_memory"}
    if not source_tools or source_tools.keys() != target_tools.keys():
        raise SystemExit(f"agent {agent}: memory_from requires matching write_memory tools; "
                         f"source has {sorted(source_tools)}, target has {sorted(target_tools)}")

    source_channels = harness.table_of(trace)
    inherited = []
    destinations: set[Path] = set()
    for name, target_tool in target_tools.items():
        source_tool = source_tools[name]
        source_channel = harness.channel(source_tool.channel, source_channels)
        matches = [record for record in trace.get("files", [])
                   if record.get("role", "own") == "own"
                   and record.get("channel") == source_channel.name
                   and record.get("path", "").endswith("/memory.md")]
        if len(matches) > 1:
            raise SystemExit(f"agent {agent}: memory_from source {source!r} episode {episode} "
                             f"contains several memories for tool {name!r}")
        destination = harness.mirror(agent, target_tool.channel) / "memory.md"
        if destination in destinations:
            continue
        destinations.add(destination)
        item = {"tool": name, "channel": target_tool.channel, "present": bool(matches)}
        if matches:
            record = matches[0]
            text = record.get("text")
            data = text.encode("utf-8") if isinstance(text, str) else b""
            if text is None or record.get("size") != len(data) or "\ufffd" in text:
                raise SystemExit(f"agent {agent}: memory_from source {source!r} episode {episode} "
                                 f"does not contain an exact text copy for tool {name!r}")
            if destination.exists() and destination.read_bytes() != data:
                raise SystemExit(f"agent {agent}: inherited memory would overwrite "
                                 f"{target_tool.channel}/memory.md")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            item.update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        inherited.append(item)

    account["memory_inherited"] = {"agent": source, "episode": episode,
                                   "memories": inherited}
    harness.save_account(agent, account)
    total = sum(item.get("bytes", 0) for item in inherited)
    print(f"{agent}: inherited {total} bytes of private memory from "
          f"{source} episode {episode}")


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return str(value)


def branch_manifest(manifest: Manifest, entries: list[dict], experiment_id: str) -> str:
    """Serialize the concrete branch manifest accepted by load_manifest()."""
    lines = [f"experiment_id = {toml_value(experiment_id)}",
             f"schedule = {toml_value(manifest['schedule'])}",
             f"stop_when_one_remains = {toml_value(manifest['stop_when_one_remains'])}",
             f"stop_when_two_remain_after_tie = {toml_value(manifest['stop_when_two_remain_after_tie'])}"]
    for key, value in manifest["overrides"].items():
        if key not in ("system_prompt", "starter_files", "starter_files_below", "budget"):
            lines.append(f"{key} = {toml_value(value)}")
    if manifest["cost"]:
        lines.append("\n[cost]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in manifest["cost"].items())
    if manifest["reveal"]:
        lines.append("\n[reveal]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in manifest["reveal"].items())
    if manifest["harness_files"]:
        lines.append("\n[harness_files]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in manifest["harness_files"].items())
    for entry in entries:
        lines.append("\n[[agent]]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in entry.items()
                     if value is not None)
    for channel in manifest["channels"] or []:
        lines.append("\n[[channel]]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in channel.items())
    for tool in manifest["tools"] or []:
        lines.append("\n[[tool]]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in tool.items())
    return "\n".join(lines) + "\n"


def branch_totals(account: dict, index: int) -> dict[str, Any]:
    """Cumulative settlement fields as they stood at one completed episode."""
    totals: dict[str, Any] = {"sent": 0, "received": 0, "rebated": 0, "debited": 0,
                              "forgiven": 0, "penalised": {}}
    for episode in account.get("episodes", [])[:index]:
        transfer = episode.get("transfer") or {}
        totals["sent"] += int(transfer.get("amount", 0))
        totals["rebated"] += int(transfer.get("rebate", 0))
        totals["debited"] += int(transfer.get("debit", 0))
        totals["received"] += int(episode.get("received", 0))
        totals["forgiven"] += int(episode.get("forgiven", 0))
        for name, record in (episode.get("channels") or {}).items():
            penalty = int(record.get("penalty", 0))
            if penalty:
                totals["penalised"][name] = totals["penalised"].get(name, 0) + penalty
    return totals


def branch_experiment(source: Path, at_round: int, experiment_id: str, takeover_seat: str,
                      output: Path | None = None) -> Path:
    """Fork every seat at a completed round and make one new seat interactive."""
    manifest = load_manifest(source)
    if at_round < 1:
        raise SystemExit("--at-round must be positive")
    if takeover_seat not in manifest["labels"]:
        raise SystemExit(f"no seat {takeover_seat!r} in {source}")
    output = output or Path(__file__).with_name("experiments") / "branches" / f"{experiment_id}.toml"
    if output.exists():
        raise SystemExit(f"branch manifest {output} already exists")
    if product.directory(harness.ROOT, experiment_id).exists():
        raise SystemExit(f"experiment record {experiment_id!r} already exists")
    new_ids = {seat: f"{experiment_id}-{int(seat):02d}" for seat in manifest["labels"]}
    check_ids(list(new_ids.values()), "branch")
    sources = []
    for seat, entry in zip(manifest["labels"], manifest["agents"]):
        account_path = harness.records_dir(entry["id"]) / "account.json"
        try:
            account = json.loads(account_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SystemExit(f"{entry['id']} has no readable source account") from None
        index = min(at_round, len(account.get("episodes", [])))
        if index < at_round and harness.why_out(account) is None:
            raise SystemExit(f"{entry['id']} did not reach completed round {at_round}")
        if index < 1 or not harness.trace_path(entry["id"], index).exists():
            raise SystemExit(f"{entry['id']} has no completed state at round {at_round}")
        if harness.records_dir(new_ids[seat]).exists() or \
                (harness.ROOT / "environments" / new_ids[seat]).exists():
            raise SystemExit(f"branch target {new_ids[seat]!r} already exists")
        trace = json.loads(harness.trace_path(entry["id"], index).read_text(encoding="utf-8"))
        if harness.rebuildable(trace, entry["id"], index) is None:
            raise SystemExit(f"{entry['id']} cannot be rebuilt at round {at_round}")
        sources.append((seat, entry, account, index))
    entries, evidence = [], []
    for seat, entry, parent, index in sources:
        new = new_ids[seat]
        if harness.fork(entry["id"], index, new):
            raise SystemExit(f"could not fork {entry['id']}")
        account = harness.load_account(new)
        account.update(branch_totals(parent, index))
        for key in ("last_election", "eliminated"):
            value = parent.get(key)
            if value and value.get("round", at_round + 1) <= at_round:
                account[key] = value
        account["branch"] = {"experiment_id": experiment_id, "source_experiment_id":
                             manifest["experiment_id"], "source_agent": entry["id"],
                             "source_episode": index, "source_round": at_round}
        if seat == takeover_seat:
            account["provider"], account["model"] = "human", "interactive"
        quality_tier = ("interactive" if seat == takeover_seat
                        else entry.get("quality_tier", "standard"))
        account["product"] = {"quality_tier": quality_tier}
        harness.save_account(new, account)
        generated = {"id": new, "label": manifest["labels"][seat],
                     "provider": account["provider"], "model": account["model"],
                     "budget": account["initial"],
                     "system_prompt": account.get("system_prompt", ""),
                     "quality_tier": (account.get("product") or {}).get("quality_tier", "standard")}
        for key in ("starter_files", "starter_files_below"):
            if key in account:
                generated[key] = account[key]
        entries.append(generated)
        trace_path = harness.trace_path(entry["id"], index)
        evidence.append({"seat": seat, "source_agent": entry["id"], "source_episode": index,
                         "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                         "new_agent": new, "provider": account["provider"], "model": account["model"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(branch_manifest(manifest, entries, experiment_id), encoding="utf-8", newline="\n")
    lineage = {"version": 1, "experiment_id": experiment_id,
               "source_experiment_id": manifest["experiment_id"], "source_manifest": str(source),
               "source_manifest_sha256": manifest["sha256"], "source_round": at_round,
               "takeover_seat": takeover_seat, "created_agents": new_ids, "traces": evidence,
               "branch_manifest": str(output)}
    product.atomic(product.directory(harness.ROOT, experiment_id) / "lineage.json", lineage)
    return output


# --- rounds -------------------------------------------------------------------


def preparer(agent: str, seats: dict[str, str], stamp: dict[str, Any],
             labels: dict[str, str] | None = None) -> Callable[[dict], None]:
    """What an agent's account is told before each episode: where it sits, what it and
    every other agent is called, and how the experiment is being driven. Nothing is
    copied into anything the agent can write, so there is nothing to revert afterwards."""
    named = {seat: (labels or {}).get(seat, seat) for seat in seats}
    def prepare(account: dict) -> None:
        seat = next(s for s, a in seats.items() if a == agent)
        account["seat"] = seat
        account["label"] = named[seat]
        account["peers"] = {"seen": seats, "labels": named,
                            "presentation": list(seats)}
        account["experiment"] = {**stamp,
                                 "memory_from": account.get("memory_inherited")}
    return prepare


def preparers(agents: list[str], stamp: dict[str, Any] | None, labels: dict[str, str] | None,
              schedule: str = "sequential") -> Callable[[str], Callable[[dict], None]]:
    """How each agent's account is stamped this round: its seat, every label, and the schedule.

    Both schedules prepare an agent the same way, and a round driven straight from
    a list of ids stands on the shorthand manifest those ids mean.
    """
    seats = seats_of(agents)
    stamp = stamp or stamp_of(shorthand(agents, schedule))
    return lambda agent: preparer(agent, seats, stamp, labels)


def attempted(agent: str, live: set[str], build: Callable[[], T]) -> T | None:
    """Call `build` up to ATTEMPTS times against environment failures, or drop the agent.

    Building an environment precedes the first API call, so nothing here is
    billed. The container name is the agent and episode index, so the next
    attempt reaps the last one's leavings by name.
    """
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return build()
        except harness.BUILD_FAILURES as e:
            print(f"{agent}: could not build an environment for this episode "
                  f"({attempt} of {ATTEMPTS}): {harness.failure(e)}", file=sys.stderr)
            if attempt == ATTEMPTS:
                print(f"{agent}: dropping out, it has no environment to run in", file=sys.stderr)
                live.discard(agent)
                return None
    return None


def drop_out(agent: str, live: set[str]) -> None:
    """Take an agent whose account admits no episode off the table, saying why."""
    print(f"{agent:<6} drops out: "
          f"{harness.why_out(harness.load_account(agent)) or 'no episode could start on it'}")
    live.discard(agent)


def unrecorded(phase: str, round_number: int, detail: dict[str, Any] | None) -> None:
    """The progress of a round driven straight from a list of ids, which has no
    experiment record to write to."""


def tell(progress: Progress, held: list[Exception], phase: str, round_number: int,
         agents: list[str]) -> None:
    """Write a phase reached between billed episodes and their commit.

    A raise there would leave billed spend uncommitted, so a write that fails is kept
    in `held`, for the round to raise once its episodes are committed.
    """
    try:
        progress(phase, round_number, {"agents": agents})
    except Exception as e:
        held.append(e)


def sequential_round(agents: list[str], live: set[str], rnd: int, router: ProviderRouter,
                     stamp: dict[str, Any] | None = None, labels: dict[str, str] | None = None,
                     progress: Progress = unrecorded) -> bool:
    """One episode for each agent still in the experiment, in seat order.

    Each agent's phases go to `progress` as it reaches them: the round waits on its
    episode, resolves its actions, and settles it. Returns whether any of them took
    one; a round where none did moved nothing.
    """
    prepare = preparers(agents, stamp, labels)
    acted = False
    for agent in agents:
        if harness.STOPPING:
            # A stop that landed while no episode was in flight ends the rounds the
            # way one that landed inside an episode does: main() answers it, and
            # no environment is built for an agent that will not run.
            raise KeyboardInterrupt
        if agent not in live:
            continue
        player = providers.is_interactive(harness.account_on_disk(agent).get("provider"))
        progress("waiting_player" if player else "waiting_autonomous", rnd + 1, {"agents": [agent]})
        ep = attempted(agent, live, lambda: harness.ready(agent, prepare(agent)))
        if agent not in live:
            continue
        if ep is None:
            drop_out(agent, live)
            continue
        out = harness.run_episode(ep, router)
        held: list[Exception] = []
        tell(progress, held, "resolving_actions", rnd + 1, [agent])
        settled = harness.settle_episode(ep, out)
        tell(progress, held, "settling_round", rnd + 1, [agent])
        trace = harness.close_episode(ep, out, settled)
        if held:
            raise held[0]
        acted = True
        if trace["stop"] in harness.STOPS_THE_EXPERIMENT:
            # Ctrl+C is the experimenter: the episode it landed in is committed and
            # traced above, and main() ends the rounds.
            raise KeyboardInterrupt
        if trace["stop"] in harness.STOPS_THE_AGENT:
            print(f"{agent}: dropping out, episode {trace['episode']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(agent)
    return acted


def build_all(agents: list[str], live: set[str],
              prepare: Callable[[str], Callable[[dict], None]]) -> dict[str, harness.Episode]:
    """Every live agent's environment, built in seat order before any episode runs.

    An agent whose environment fails ATTEMPTS times drops out; the rest are built.
    Anything that ends the builds early, a stop included, abandons every
    environment built so far, since none has billed.
    """
    built: dict[str, harness.Episode] = {}
    try:
        for agent in agents:
            if harness.STOPPING:
                raise KeyboardInterrupt
            if agent not in live:
                continue
            ep = attempted(agent, live, lambda: harness.ready(agent, prepare(agent)))
            if ep is None:
                if agent in live:
                    drop_out(agent, live)
                continue
            built[agent] = ep
    except BaseException:
        for ep in built.values():
            ep.abandon()
        raise
    return built


def simultaneous_round(agents: list[str], live: set[str], rnd: int, router: ProviderRouter,
                       stamp: dict[str, Any] | None = None, labels: dict[str, str] | None = None,
                       progress: Progress = unrecorded) -> bool:
    """One episode for each agent still in the experiment, all at once.

    Every environment is built first, so no episode reads this round's writes. The
    episodes run in threads. Then each settles in seat order, the transfers they
    made are credited, and each closes in seat order: a credit lands after the
    receiver's own turns and before its floor, so an agent is out on the round's
    net and never lifted back by a transfer that had already arrived. Each phase
    goes to `progress` as the round reaches it, and so does each autonomous episode
    as it ends. Returns whether any agent took an episode.
    """
    built = build_all(agents, live, preparers(agents, stamp, labels, "simultaneous"))
    if harness.STOPPING:
        for ep in built.values():
            ep.abandon()
        raise KeyboardInterrupt
    if not built:
        return False

    interactive = {agent: providers.is_interactive(ep.account.get("provider"))
                   for agent, ep in built.items()}
    outs: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}
    def go(agent: str, ep: harness.Episode) -> None:
        try:
            outs[agent] = harness.run_episode(ep, router)
            if not interactive[agent]:
                running = [name for name in built if name not in outs and name not in errors]
                phase = ("waiting_player" if any(interactive[name] for name in running) else
                         "waiting_autonomous" if running else "resolving_actions")
                progress(phase, rnd + 1, {"completed_autonomous": agent})
        except BaseException as e:      # run_episode has saved and reaped on its way out
            errors[agent] = e

    threads = [threading.Thread(target=go, args=(agent, ep), name=agent, daemon=True)
               for agent, ep in built.items()]
    autonomous = [agent for agent in built if not interactive[agent]]
    players = [agent for agent in built if interactive[agent]]
    try:
        if autonomous:
            progress("waiting_autonomous", rnd + 1, {"agents": autonomous})
        if players:
            progress("waiting_player", rnd + 1, {"agents": players})
        for t in threads:
            t.start()
    except BaseException:
        for ep in built.values():
            ep.abandon()
        raise
    # A timed join, so a signal reaches the handler while the episodes run: the
    # flag it sets is read at the top of every episode's next turn.
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(0.2)

    # Each agent settles and closes on its own, so one agent's failure to commit
    # does not discard the others' billed spend; the first failure is raised after
    # every agent has had its turn.
    pending: list[tuple[str, int]] = []

    def defer(receiver: str, amount: int) -> None:
        pending.append((receiver, amount))

    held: list[Exception] = []
    tell(progress, held, "resolving_actions", rnd + 1, [agent for agent in built if agent in outs])
    settled: dict[str, dict] = {}
    for agent, ep in built.items():
        if agent not in outs:
            continue
        try:
            settled[agent] = harness.settle_episode(ep, outs[agent], defer)
        except BaseException as e:
            errors.setdefault(agent, e)
    for receiver, amount in pending:
        if receiver in settled:
            harness.credit_episode(built[receiver], amount)
        else:
            harness.credit_on_disk(receiver, amount)
    tell(progress, held, "settling_round", rnd + 1, list(settled))
    traces: dict[str, dict] = {}
    for agent, ep in built.items():
        if agent not in settled:
            continue
        try:
            traces[agent] = harness.close_episode(ep, outs[agent], settled[agent])
        except BaseException as e:
            errors.setdefault(agent, e)
    if errors:
        raise next(iter(errors.values()))
    if held:
        raise held[0]

    for agent, trace in traces.items():
        if trace["stop"] in harness.STOPS_THE_AGENT and trace["stop"] not in harness.STOPS_THE_EXPERIMENT:
            print(f"{agent}: dropping out, episode {trace['episode']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(agent)
    if any(trace["stop"] in harness.STOPS_THE_EXPERIMENT for trace in traces.values()):
        # Every episode in flight ended at its next turn and is committed and
        # traced above; what does not happen is the next round.
        raise KeyboardInterrupt
    return True


def resolve_vote(agents: list[str], live: set[str], labels: dict[str, str],
                 vote: dict | None) -> int | None:
    """Resolve one completed voting round and remove every eliminated agent.

    The vote tool's cadence defines the cycle. A ballot is the tool's private,
    episode-scoped file in the voter's own channel; its final contents are the final
    tool call. The result contains aggregate totals, never voter-to-target mappings.
    """
    if vote is None or not live:
        return None
    counts = {agent: len(harness.load_account(agent)["episodes"]) for agent in live}
    rounds = set(counts.values())
    every = vote["every"]
    if len(rounds) != 1 or not (round_number := next(iter(rounds))) or round_number % every:
        return None

    electorate = [agent for agent in agents if agent in live]
    agent_by_label = {labels[str(i)]: agent for i, agent in enumerate(agents, 1)}
    ballots: dict[str, str] = {}
    for agent in electorate:
        path = harness.mirror(agent, vote["channel"]) / "vote"
        try:
            target = path.read_text(encoding="utf-8").strip()
        except OSError:
            target = ""
        path.unlink(missing_ok=True)
        if target in agent_by_label and agent_by_label[target] != agent:
            ballots[agent] = agent_by_label[target]

    label_by_agent = {agent: labels[str(i)] for i, agent in enumerate(agents, 1)}
    tally = {agent: 0 for agent in electorate}
    for target in ballots.values():
        if target in tally:
            tally[target] += 1
    abstainers = set(electorate) - set(ballots)
    leaders: set[str] = set()
    if ballots:
        most = max(tally.values())
        leaders = {agent for agent, total in tally.items() if total == most}
        if len(leaders) != 1:
            print(f"election after round {round_number}: top vote tied at {most}; "
                  "no agent is eliminated by the vote")
            leaders.clear()

    voted_out = leaders - abstainers
    eliminated = abstainers | voted_out
    result = {
        "round": round_number,
        "tally": {label_by_agent[agent]: tally[agent] for agent in electorate},
        "abstainers": [label_by_agent[agent] for agent in electorate if agent in abstainers],
        "voted_out": next((label_by_agent[agent] for agent in electorate
                           if agent in voted_out), ""),
        "top_votes": max(tally.values(), default=0),
        "top_tied": bool(ballots and not leaders),
        "remaining": [label_by_agent[agent] for agent in electorate if agent not in eliminated],
    }
    for agent in electorate:
        account = harness.load_account(agent)
        account["last_election"] = result
        if agent in eliminated:
            total = tally[agent]
            reason = (f"did not vote in round {round_number}" if agent in abstainers else
                      f"received the most votes ({total}) in round {round_number}")
            account["eliminated"] = {"round": round_number, "reason": reason, "votes": total}
        harness.save_account(agent, account)
        if agent in eliminated:
            live.discard(agent)
            print(f"{agent}: eliminated after round {round_number}: {reason}")
    if not ballots:
        print(f"election after round {round_number}: no ballots were cast")
    return round_number


def play_round(a_round: Round, agents: list[str], live: set[str], rnd: int, router: ProviderRouter,
               stamp: dict[str, Any], labels: dict[str, str], stop_when_one_remains: bool,
               vote: dict | None = None, stop_when_two_remain_after_tie: bool = False,
               progress: Progress = unrecorded) -> str | None:
    """One round: drop the agents that cannot act, name the round, run it, and say
    why the rounds end, or None while they go on.

    The reason is the outcome's termination_reason: "all_eliminated" when every agent
    is out, "one_remains" when the manifest stops after one agent remains, and
    "final_tie" when it stops at a tied vote between the last two.
    """
    # Asked before the round, so the header names who will act. Between here and
    # an agent's own turn its balance can only move up, a peer's transfer being
    # the only thing that reaches it, so this is the answer its episode would give.
    for agent in agents:
        if agent in live and harness.why_out(harness.load_account(agent)):
            drop_out(agent, live)
    if not live:
        print(f"every agent is out after {rnd} rounds")
        return "all_eliminated"
    if stop_when_one_remains and len(live) == 1:
        print(f"{next(iter(live))} is the only agent left with anything to spend; the competition ends")
        return "one_remains"
    # The seat order decides who acts on this round's information and who on last
    # round's, so it is on screen beside the round number. Under a simultaneous
    # round nobody acts on this round's.
    acting = [agent for agent in agents if agent in live]
    print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")

    if not a_round(agents, live, rnd, router, stamp, labels, progress):
        # Every agent at the table either takes an episode or leaves it, so a round
        # nobody took one in has put every agent out.
        print(f"no agent could take an episode in round {rnd + 1}; "
              f"the rounds end here with {len(live)} agents at the table")
        return "all_eliminated"
    election = resolve_vote(agents, live, labels, vote)
    if election is not None and not live:
        print(f"every agent is out after {election} rounds")
        return "all_eliminated"
    if election is not None and stop_when_one_remains and len(live) == 1:
        print(f"{next(iter(live))} is the only agent left; the competition ends")
        return "one_remains"
    if election is not None and stop_when_two_remain_after_tie and len(live) == 2:
        result = harness.load_account(next(iter(live))).get("last_election", {})
        if result.get("top_tied"):
            survivors = [agent for agent in agents if agent in live]
            print(f"the final vote tied between {' and '.join(survivors)}; "
                  "both survive and the competition ends")
            return "final_tie"
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI. Verifies the shipped digests and the endpoint, then runs the rounds."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", metavar="NAME",
                    help="an experiment manifest: the schedule, the experiment's defaults, and each "
                         "agent's own terms. Every run names the experiment it is part of. A bare "
                         "name is looked for under experiments/ and experiments/examples/; anything "
                         "with a suffix or a directory in it is taken as the path it is")
    ap.add_argument("-m", "--manifest", dest="named", metavar="NAME",
                    help="the same manifest, given as a flag")
    ap.add_argument("-r", "--rounds", type=int, default=1, metavar="N",
                    help="up to N episodes for each agent, stopping early as budgets end")
    ap.add_argument("--provider", metavar="PROVIDER",
                    help="use PROVIDER for every seat together with --model")
    ap.add_argument("--model", metavar="MODEL",
                    help="use MODEL for every seat together with --provider; under --resume, both "
                         "must match existing agent accounts")
    ap.add_argument("--resume", action="store_true",
                    help="continue existing compatible agent accounts; without this flag, "
                         "previous state is preserved under displaced/ and a fresh run starts")
    ap.add_argument("-c", "--config", type=Path, help="default: config.toml beside harness.py")
    ap.add_argument("--branch-from", type=Path, metavar="MANIFEST",
                    help="create an immutable experiment branch from a completed round")
    ap.add_argument("--at-round", type=int, metavar="N", help="completed source round to branch")
    ap.add_argument("--branch-id", help="new experiment identity and agent-id prefix")
    ap.add_argument("--takeover-seat", metavar="SEAT",
                    help="seat whose new agent uses human/interactive")
    ap.add_argument("--output", type=Path, help="generated branch manifest path")
    a = ap.parse_args(argv)

    if a.branch_from is not None:
        if a.at_round is None or not a.branch_id or not a.takeover_seat:
            ap.error("--branch-from requires --at-round, --branch-id and --takeover-seat")
        output = branch_experiment(a.branch_from, a.at_round, a.branch_id,
                                   a.takeover_seat, a.output)
        print(f"branch manifest: {output}")
        return 0
    if a.rounds < 1:
        ap.error("--rounds must be at least 1")
    named = a.named or a.name
    if named is None:
        ap.error("name a manifest: a name under experiments/, or a path to one")
    manifest = load_manifest(manifest_path(named))
    if (a.provider is None) != (a.model is None):
        ap.error("--provider and --model must be supplied together")
    if a.model is not None:
        harness.validate_terms("command line", provider=a.provider, model=a.model, budget=None,
                               starter_files=None, starter_files_below=None)
        for entry in manifest["agents"]:
            entry["provider"] = a.provider
            entry["model"] = a.model

    router = harness.start(a.config, overrides=manifest["overrides"],
                           requirements={(e["provider"], e["model"]) for e in manifest["agents"]},
                           channel_tables=manifest["channels"], harness_files=manifest["harness_files"],
                           labels=tuple(manifest["labels"].values()),
                           tool_tables=manifest["tools"])
    harness.catch_signals()
    agents = [e["id"] for e in manifest["agents"]]
    live = set(agents)
    stamp = stamp_of(manifest)
    experiment_id = manifest["experiment_id"]
    vote = next((tool for tool in manifest["tools"] or [] if tool["kind"] == "vote"), None)
    a_round = simultaneous_round if manifest["schedule"] == "simultaneous" else sequential_round
    if not a.resume:
        bundle = harness.displace_agents(agents)
        product.displace(harness.ROOT, experiment_id, bundle)
    # Every agent is created before the first round, so the first to act finds its
    # peers' blackboards in place. Each is created on its own terms, and one that
    # exists must have been created on the same.
    for entry in manifest["agents"]:
        account = harness.load_account(entry["id"], **terms_of(entry))
        account["product"] = {"quality_tier": entry.get(
            "quality_tier", "interactive" if providers.is_interactive(entry["provider"]) else "standard")}
        harness.save_account(entry["id"], account)
        inherit_memory(entry, account)
    print(f"experiment: {', '.join(agents)}  ({len(agents)} agents, up to {a.rounds} rounds, "
          f"{manifest['schedule']})")
    progress = functools.partial(product.progress, harness.ROOT, experiment_id)
    reason = "round_limit"
    code = 0
    try:
        for _ in range(a.rounds):
            round_number = max((len(harness.load_account(agent).get("episodes", []))
                                for agent in agents), default=0) + 1
            rnd = round_number - 1
            cost = product.cost(agents, harness.load_account, manifest["cost"],
                                providers.is_interactive)
            progress("preparing_round", round_number,
                     {"cost": cost, "agents": agents, "schedule": manifest["schedule"]})
            if cost["ceiling_reached"]:
                reason = "cost_ceiling"
                break
            ended = play_round(a_round, agents, live, rnd, router, stamp, manifest["labels"],
                               manifest["stop_when_one_remains"], vote,
                               manifest["stop_when_two_remain_after_tie"], progress)
            progress("round_completed", round_number,
                     {"active": [agent for agent in agents if agent in live]})
            if ended is not None:
                reason = ended
                break
    except KeyboardInterrupt:
        # Every agent still at the table keeps its account, its traces and its seat,
        # so the experiment can be started again from where it stopped. What ends is
        # the rounds.
        print(f"interrupted; the rounds end here with {len(live)} agents at the table",
              file=sys.stderr)
        reason, code = "interrupted", 130
    progress(reason if reason in ("interrupted", "cost_ceiling") else "completed",
             max((len(harness.load_account(agent).get("episodes", [])) for agent in agents),
                 default=0), {"termination_reason": reason})
    product.outcome(harness.ROOT, experiment_id, agents, manifest["labels"], live, reason,
                    harness.load_account, harness.trace_path, manifest["reveal"])
    return code


if __name__ == "__main__":
    sys.exit(main())
