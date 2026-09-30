"""Watch an experiment and answer interactive seats: py -3 view.py [--experiment h | --agent h02]

Serves a page on 127.0.0.1 showing one experiment's activity, every directory
channel, and one transcript at a time, above every seat's balance, what it has
spent, and any transfer ledger. Every GET route only reads. The one POST route,
held to the page's origin and token, writes an interactive seat's submission into
the interactive provider's coordination store under interactions/, and nothing
else."""

from __future__ import annotations

import argparse
import copy
import http.server
import json
import re
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, NamedTuple

import analyze
import experiment
import harness
import product
import providers
from interaction import (InteractionConflict, InteractionError, InteractionStore,
                         InvalidSubmission, StaleRequest, UnreadableRecord)

PORT = 8765
MAX_INTERACTION_BODY = 64 * 1024

# What the page polls at, in milliseconds. Fast enough that a turn appears while
# the turn after it is still being thought about, slow enough that an experiment's
# traces are read once a second and a half and not continuously.
POLL_MS = 1500

# An episode quiet for longer than this is not being waited on, it is over: no
# trace will follow it, and the next episode will take its index back. A turn is an
# API call plus the commands it runs, so the threshold clears a slow one.
STALE_AFTER = 180

# Points kept in a header sparkline. A long-lived agent's series runs to thousands
# of elements and the strip is 240 pixels wide, so the rest is bytes on the wire
# for pixels that do not exist, once per seat, on every poll.
SPARK_POINTS = 240

# The leading letters of an agent id, which is what names a set of them when
# nothing better is on disk: comp01..comp05 are the comp agents.
AGENT_PREFIX = re.compile(r"^[^\d]*")

# Stands for a path an outbox did not hold, which is not the same as a path it
# held with no text: a binary file reads as None and is still there.
ABSENT = object()

# The page, served as it stands on disk beside this file.
PAGE = Path(__file__).with_name("view.html").read_text(encoding="utf-8")


# manifest digest -> (manifest, path). The account carries the digest rather than
# a path, so shipped manifests can describe a first episode before its trace lands.
_MANIFESTS: dict[str, tuple[dict, Path]] | None = None
_MANIFEST_STATE: tuple[tuple[str, int, int], ...] | None = None

# (the state of disk it was read from, the grouping). Guarded because the server is
# threaded and every open page asks for it on every poll.
_EXPERIMENTS: tuple[tuple, list[dict]] | None = None
_EXPERIMENTS_LOCK = threading.Lock()


# --- reading what is on disk ------------------------------------------------


def read_json(path: Path) -> dict | None:
    """One JSON object, or None if the file is not readable or holds something else.

    save_account commits with os.replace, which on Windows surfaces to a reader as
    a PermissionError, so a poll landing on a commit is retried once.
    """
    for attempt in (1, 2):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except FileNotFoundError:
            return None
        except (PermissionError, OSError, ValueError):
            if attempt == 2:
                return None
            time.sleep(0.05)
    return None


def stat_key(path: Path) -> tuple[int, int] | None:
    """A file's mtime and size, which is what a cache here keys on; None where it is gone."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


# path -> ((mtime, size), parsed). Guarded because the server is threaded and
# two polls can want the same trace at once.
_TRACES: dict[Path, tuple[tuple[int, int], dict]] = {}
_TRACE_LOCK = threading.Lock()


def load_trace(path: Path) -> dict | None:
    """One trace, parsed once.

    A trace is written whole when its episode ends and never touched again, so
    it is cached against the mtime and size that identify it.
    """
    key = stat_key(path)
    if key is None:
        return None
    with _TRACE_LOCK:
        hit = _TRACES.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    trace = read_json(path)
    if trace is None:
        return None
    with _TRACE_LOCK:
        _TRACES[path] = (key, trace)
    return trace


def agent_names() -> list[str]:
    """Every agent with an account, in name order.

    The account is what makes a directory an agent: records/analysis/ is where
    analyze.py writes when it was given no agent id, and it has none.
    """
    return [d.name for d in sorted(harness.records_root().glob("*"))
            if harness.account_path(d.name).exists()]


def account_of(agent: str) -> dict:
    """One agent's account, empty where it has none this poll."""
    return read_json(harness.account_path(agent)) or {}


def manifests() -> dict[str, tuple[dict, Path]]:
    """Shipped manifests by digest, reparsed when one changes on disk."""
    global _MANIFESTS, _MANIFEST_STATE
    root = Path(__file__).with_name("experiments")
    paths = sorted(root.rglob("*.toml"))
    state = tuple((str(path), stat.st_mtime_ns, stat.st_size)
                  for path in paths if (stat := path.stat()))
    if _MANIFESTS is None or state != _MANIFEST_STATE:
        found = {}
        for path in paths:
            try:
                manifest = experiment.load_manifest(path)
            except SystemExit:
                continue
            found[manifest["sha256"]] = (manifest, path)
        _MANIFESTS, _MANIFEST_STATE = found, state
    return _MANIFESTS


def manifest_of(agent: str) -> tuple[dict, Path] | None:
    """The shipped manifest stamped on an agent's account, where it is available."""
    digest = (account_of(agent).get("experiment") or {}).get("manifest_sha256")
    return manifests().get(digest) if digest else None


def manifest_ahead_of(last: dict | None, agent: str) -> tuple[dict, Path] | None:
    """The stamped manifest when it is newer than the agent's last trace."""
    found = manifest_of(agent)
    if found is None:
        return None
    digest = last["provenance"].get("manifest_sha256") if last else None
    return found if digest != found[0]["sha256"] else None


def traces_of(agent: str) -> list[dict]:
    """Every finished episode of an agent, in order."""
    return [t for t in (load_trace(p) for p in harness.trace_paths(agent)) if t is not None]


def latest_trace(agent: str) -> dict | None:
    """The agent's last committed episode, or None before it has one."""
    paths = harness.trace_paths(agent)
    return load_trace(paths[-1]) if paths else None


def live_index(agent: str) -> int | None:
    """The episode with no trace yet, or None if the agent is between starts.

    Unfinished is exactly a raw log with no trace beside it, which is not the
    same as running: how long since the log grew is what live_age reports.
    """
    raws = harness.raw_paths(agent)
    if not raws:
        return None
    index = harness.episode_number(raws[-1])
    return None if harness.trace_path(agent, index).exists() else index


def live_age(agent: str, index: int | None) -> float | None:
    """Seconds since the episode's raw log last grew, or None if there is none.

    A turn takes as long as the API call plus the commands it runs, so a live
    episode is quiet for stretches; a dead one is quiet for good.
    """
    if index is None:
        return None
    try:
        return max(0.0, time.time() - harness.raw_path(agent, index).stat().st_mtime)
    except OSError:
        return None


def acting(agent: str, index: int | None) -> bool:
    """Whether the agent is moving, not merely holding an unfinished index."""
    return index is not None and (live_age(agent, index) or 0) < STALE_AFTER


def latest_attempt(lines: list[dict]) -> list[dict]:
    """The last attempt at an episode, out of a log that may hold more than one.

    An episode that died without writing a trace leaves its index free for the next
    episode, which appends to the same log. Turn numbers restarting at 1 is the seam.
    """
    starts = [i for i, line in enumerate(lines)
              if line.get("turn") == 1 and line.get("kind", "native_response") == "native_response"]
    return lines[starts[-1]:] if starts else lines


def raw_lines(path: Path) -> list[dict]:
    """Every whole response in an episode's raw log.

    log_raw appends while the episode runs, so a trailing fragment is dropped
    and the next poll picks it up whole.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
    return out


def read_file(p: Path) -> tuple[int, str | None] | None:
    """One file's size and its text, or None if it is unreadable.

    harness.bounded_read is the one reader: the same bound and the same
    truncation marker the trace's own capture uses. Text is None for a binary.
    """
    got = harness.bounded_read(p)
    if got is None:
        return None
    size, text, binary = got
    return size, None if binary else text


# --- the initial observation ------------------------------------------------


def observation_split(text: str) -> tuple[str, list[dict]]:
    """The listing an episode opened on, and the digest's quoted files after it.

    The listing is everything before the first header line, and a quoted file's
    body runs to the next header. A header naming files the digest did not quote
    (unchanged, withdrawn) ends the body before it and is no section itself. An
    observation with no header is all listing.
    """
    marks = list(harness.SECTION.finditer(text))
    if not marks:
        return text, []
    out = []
    for i, m in enumerate(marks):
        if harness.NAMED.match(m.group(0)):
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() + 1:end]
        out.append({"path": m.group("path"), "text": body,
                    "bytes": len(body.encode("utf-8"))})
    return text[:marks[0].start()], out


def observation_clipped(t: dict) -> bool:
    """Whether the initial observation ran past the ceiling that episode served it under.

    harness.clip keeps a head and a tail with a marker between, so an observation over
    its own limit is one that lost a middle. The limit is the agent's, read off the
    episode's provenance and not this process's config.
    """
    limit = t["provenance"].get("observation_limit")
    return bool(limit) and len(t.get("observation") or "") > limit


def message_paths(t: dict) -> set[str] | None:
    """Every item name the digest quoted, or None where it carried none."""
    _, sections = observation_split(t.get("observation") or "")
    return {s["path"] for s in sections} if sections else None


def digest_name(agent: str, t: dict, path: str) -> str:
    """An internal path as the episode's channel presents it to that agent."""
    account = account_of(agent)
    instances = harness.environment(agent, account, harness.table_of(t))
    return harness.digest_name(path, instances, harness.experimenter_digest_paths(account, instances))


def inbox_prefix(mail: harness.Channel | None) -> str | None:
    """The prefix used for inbox items in the opening digest."""
    if mail is None:
        return None
    return "Letter from " if mail.agent_view == "letters" else f"{mail.inbox}/"


def thin(series: list[int], points: int = SPARK_POINTS) -> list[int]:
    """A long series sampled down, keeping the first element and the last."""
    if len(series) <= points:
        return list(series)
    step = (len(series) - 1) / (points - 1)
    return [series[round(i * step)] for i in range(points - 1)] + [series[-1]]


# --- one turn, from either source -------------------------------------------


def from_trace(t: dict) -> list[dict]:
    """A finished episode's turns, with the command output they returned."""
    return [{
        "turn": turn.get("turn"),
        "micros": turn.get("micros"),
        "prefix": (turn.get("usage") or {}).get("prefix_tokens"),
        "balance": turn.get("balance"),
        "stop_reason": turn.get("stop_reason"),
        "refusal": turn.get("refusal"),
        "provider": turn.get("provider"),
        "model": turn.get("resolved_model"),
        "text": turn.get("text") or "",
        "thinking": turn.get("thinking") or "",
        "tools": [{"result": tool.get("result"),
                   "shell": analyze.is_shell(tool.get("tool")),
                   "call": analyze.call_shown(tool.get("tool"), tool.get("command"),
                                              tool.get("input"))}
                  for tool in turn.get("tools") or []],
        "tokens": turn.get("usage") or {},
    } for turn in t.get("turns") or []]


def from_raw(lines: list[dict], account: dict) -> list[dict]:
    """A running episode's turns from canonical events written after native responses.

    Billed by the rule harness.bill_once applies, restated over the logged dict: a
    response id's charges count once and a replay of it is zeroed, so each balance
    is the one the account commits. Command results are None until the trace lands.
    """
    remaining = account["remaining"]
    centi, seen, out = 0, set(), []
    for line in lines:
        if line.get("kind") != "normalized_response":
            continue
        data = line.get("response") or {}
        rid = data.get("id") or f"anon-{line.get('turn')}"
        duplicate = rid in seen
        seen.add(rid)
        u = {} if duplicate else data.get("usage") or {}
        turn_centi = 0 if duplicate else sum(c.get("centi_micros", 0) for c in data.get("charges") or [])
        previous = remaining - centi // 100
        centi += turn_centi
        balance = remaining - centi // 100
        calls = data.get("tool_calls") or []
        out.append({
            "turn": line.get("turn"),
            "received": line.get("received"),
            "micros": previous - balance,
            "prefix": u.get("prefix_tokens", 0),
            "balance": balance,
            "stop_reason": data.get("stop_reason"),
            "refusal": data.get("refusal"),
            "provider": data.get("provider"),
            "model": data.get("resolved_model"),
            "text": "\n".join(data.get("text") or []),
            "thinking": "\n".join(data.get("reasoning") or []),
            "tools": [{"result": None,
                       "shell": analyze.is_shell(call.get("name")),
                       "call": analyze.call_shown(call.get("name"),
                                                  (call.get("input") or {}).get("command"),
                                                  call.get("input"))}
                      for call in calls],
            "tokens": u,
        })
    return out


def live_turns(agent: str, index: int, account: dict) -> list[dict]:
    """The turns of an unfinished episode, read off its raw log."""
    return from_raw(latest_attempt(raw_lines(harness.raw_path(agent, index))), account)


def live_state(agent: str, index: int | None, account: dict) -> dict:
    """What an unfinished episode has done so far, derived from its raw log.

    `turns` are its turns, `spent` what they have cost, `balance` what the last
    of them left and None before any is on disk, and `remaining` that balance or
    the account's where there is none yet. Empty for an agent with no episode in
    flight (`index` None).
    """
    turns = live_turns(agent, index, account) if index is not None else []
    balance = turns[-1]["balance"] if turns else None
    return {"turns": turns, "balance": balance,
            "spent": account.get("remaining", 0) - balance if turns else 0,
            "remaining": balance if turns else account.get("remaining")}


# --- who is at the table ----------------------------------------------------


def group_of(agent: str, account: dict) -> str:
    """The set of agents this one belongs to, as one name.

    An experiment knows its own membership, so that is used where it exists; the
    leading letters of the id cover agents started one at a time.
    """
    peers = (account.get("peers") or {}).get("seen") or {}
    if not peers:
        return AGENT_PREFIX.match(agent).group(0) or agent
    members = sorted({agent, *peers.values()})
    head = AGENT_PREFIX.match(members[0]).group(0)
    return head if head and all(m.startswith(head) for m in members) else "+".join(members)


def seating_key(agent: str, account: dict) -> tuple[str, ...] | None:
    """The experiment an agent is seated in, as its members in seat order.

    An agent is seated when the mapping it carries puts it in its own seat; one
    that leaves the agent out names no blackboard as its own, and None says so.
    """
    s = harness.seating_of(agent, account)
    if s.seen.get(s.seat) != agent:
        return None
    return tuple(s.seen.values())


def manifest_environment(manifest: dict,
                         path: Path) -> tuple[list[harness.Channel], dict[str, str]]:
    """A shipped manifest's channel table and harness file names, validated together."""
    return harness.validate_channels(manifest["channels"], manifest["harness_files"],
                                     str(path), tuple(manifest["labels"].values()))


def agent_environment(last: dict | None,
                      agent: str) -> tuple[list[harness.Channel], dict[str, str]]:
    """The channel table and harness file names an agent runs under.

    `last` is its latest trace. The stamped manifest's where the manifest is newer
    than `last` or there is no trace yet, `last`'s otherwise, and the process's
    where there is neither.
    """
    found = manifest_ahead_of(last, agent)
    if found:
        return manifest_environment(*found)
    if last:
        return harness.table_of(last), analyze.harness_files_of(last)
    return harness.channels(), dict(harness.HARNESS_FILES)


def experiment_table(exp: dict) -> list[harness.Channel]:
    """The experiment's channel table, as experiments() recorded it."""
    return harness.channels_from(exp["channels"])


def seat_of_label(exp: dict, label: str | None) -> str | None:
    """The seat an agent-facing label names, or None for a label no seat has."""
    return next((s for s, l in exp["labels"].items() if l == label), None)


def named_group(members: list[str]) -> str:
    """One stable short name for a complete manifest membership."""
    seen = {str(i): agent for i, agent in enumerate(members, 1)}
    return group_of(members[0], {"peers": {"seen": seen}})


def anchored_groups(accounts: dict[str, dict],
                    catalog: dict[str, tuple[dict, Path]] | None = None
                    ) -> tuple[list[dict], set[str]]:
    """Experiments identified by a manifest stamp already written to one account.

    A simultaneous round prepares every environment before any episode starts. During
    that preparation some accounts carry the complete seating while later accounts are
    still fresh singletons. The stamped manifest is the common declaration and groups
    every account it names without waiting for all account writes to finish.
    """
    out, claimed = [], set()
    catalog = manifests() if catalog is None else catalog
    for digest, (manifest, path) in catalog.items():
        declared = [entry["id"] for entry in manifest["agents"]]
        if not any(((accounts.get(agent, {}).get("experiment") or {}).get("manifest_sha256")
                    == digest) for agent in declared):
            continue
        members = [agent for agent in declared if agent in accounts and
                   ((accounts[agent].get("experiment") or {}).get("manifest_sha256")
                    in (None, digest))]
        if not members:
            continue
        seats = {seat: agent for seat, agent in experiment.seats_of(declared).items()
                 if agent in members}
        labels = {seat: manifest["labels"][seat] for seat in seats}
        table, _ = manifest_environment(manifest, path)
        out.append({
            "name": named_group(declared), "seated": True, "seats": seats,
            "experiment_id": manifest.get("experiment_id", path.stem),
            "members": members, "posts": bool(harness.mailbox_channel(table)
                                                or harness.schema_channel(table)),
            "channels": [ch.as_table() for ch in table], "labels": labels,
            "tools": manifest["tools"] or [],
        })
        claimed.update(members)
    return out, claimed


def grouping_state() -> tuple:
    """What the experiment grouping is read from, as it stands: every account, each
    agent's latest trace, which is the one the grouping reads, and the shipped
    manifests. Equal states group alike."""
    # Rereads the manifests where they moved, which _MANIFEST_STATE then records.
    manifests()
    agents = []
    for agent in agent_names():
        paths = harness.trace_paths(agent)
        latest = (paths[-1].name, stat_key(paths[-1])) if paths else None
        agents.append((agent, stat_key(harness.account_path(agent)), latest))
    return str(harness.records_root()), _MANIFEST_STATE, tuple(agents)


def experiments() -> list[dict]:
    """Every set of agents on disk, the seated ones first, each with how many of its
    members are acting.

    The grouping is read again only when what it is read from has moved. Acting is
    a question of the clock as well, so it is asked on every call.
    """
    global _EXPERIMENTS
    state = grouping_state()
    with _EXPERIMENTS_LOCK:
        if _EXPERIMENTS is None or _EXPERIMENTS[0] != state:
            _EXPERIMENTS = (state, read_experiments())
        out = copy.deepcopy(_EXPERIMENTS[1])
    for exp in out:
        exp["running"] = sum(acting(agent, live_index(agent)) for agent in exp["members"])
    return out


def read_experiments() -> list[dict]:
    """Every set of agents on disk, the seated ones first, read afresh.

    Agents sharing a seating are one experiment, named by group_of. One whose mapping
    does not seat it is grouped by its id's letters and marked unseated.
    """
    accounts = {agent: account for agent in agent_names()
                if (account := read_json(harness.account_path(agent))) is not None}
    anchored, claimed = anchored_groups(accounts)
    groups: dict[tuple, dict] = {}
    for agent, account in accounts.items():
        if agent in claimed:
            continue
        key = seating_key(agent, account)
        seen = harness.seating_of(agent, account).seen
        ident = key or ("unseated", group_of(agent, account))
        exp: dict[str, Any] | None = groups.get(ident)
        if exp is None:
            exp = groups[ident] = {
                "name": group_of(agent, account), "seated": key is not None,
                "experiment_id": (account.get("experiment") or {}).get("experiment_id")
                                 or group_of(agent, account),
                "seats": dict(seen) if key else {}, "members": [],
                "posts": False,
            }
        exp["members"].append(agent)

    out = sorted([*anchored, *groups.values()],
                 key=lambda exp: (not exp["seated"], exp["name"]))
    # Two sets can arrive at one name: a seated experiment and a leftover agent whose
    # id starts with the same letters. The seated one is sorted first and keeps
    # the short name, so what the other is called says what it is.
    taken: set[str] = set()
    for exp in out:
        exp["members"].sort()
        if exp["name"] in taken:
            exp["name"] = "+".join(exp["members"])
        taken.add(exp["name"])
        # The table and the labels the members ran under, from the records of the
        # first member that has any: every member of one experiment ran under the
        # same ones.
        if "channels" not in exp:
            first = next((a for a in exp["members"] if harness.trace_paths(a)), exp["members"][0])
            last = latest_trace(first)
            table, _ = agent_environment(last, first)
            account = account_of(first)
            exp["channels"] = [ch.as_table() for ch in table]
            exp["labels"] = dict(harness.seating_of(first, account).labels) if exp["seated"] else {}
            mail = harness.mailbox_channel(table)
            schema = harness.schema_channel(table)
            exp["posts"] = bool(mail or schema)
            found = manifest_of(first)
            exp["tools"] = ((found[0]["tools"] or []) if found else
                            (last["provenance"].get("tools") if last else []) or [])
    return out


def experiment_named(name: str) -> dict | None:
    """The experiment by this name, or None."""
    return next((exp for exp in experiments() if exp["name"] == name), None)


def experiment_of(agent: str) -> dict | None:
    """The experiment this agent sits in."""
    return next((exp for exp in experiments() if agent in exp["members"]), None)


def places_of(exp: dict) -> list[tuple[str | None, str]]:
    """Every agent of the experiment in the order the tabs show it.

    By seat where there are seats, which is the order the agents themselves see
    each other in, and by name where there are none.
    """
    if exp["seated"]:
        return list(exp["seats"].items())
    return [(None, agent) for agent in exp["members"]]


# --- the round --------------------------------------------------------------


def started_at(t: dict) -> str:
    """When the episode started, from its provenance."""
    return t["provenance"]["started_at"]


def experiment_episodes(exp: dict) -> list[dict]:
    """Every committed episode of every member, in the order they started.

    One episode per agent per round is what sequential_round holds to, so the round is
    read out of start order, cut where an agent would take a second turn.
    """
    rows = sorted(({"seat": seat, "agent": agent, "episode": t["episode"], "at": started_at(t),
                    "trace": t}
                   for seat, agent in places_of(exp) for t in traces_of(agent)),
                  key=lambda s: (s["at"], s["agent"], s["episode"]))
    rnd, acted = 0, set()
    for s in rows:
        if not rnd or s["agent"] in acted:
            rnd, acted = rnd + 1, set()
        acted.add(s["agent"])
        s["round"] = rnd
        s["live"] = False
    return rows


def live_rows(exp: dict, rows: list[dict]) -> list[dict]:
    """The episodes in flight, each in the round it belongs to.

    An agent with a raw log and no trace is taking its turn now, which is the round
    after the last one it acted in.
    """
    out = []
    for seat, agent in places_of(exp):
        live = live_index(agent)
        if live is None:
            continue
        mine = [r["round"] for r in rows if r["agent"] == agent]
        out.append({"seat": seat, "agent": agent, "episode": live, "at": None, "trace": None,
                    "round": (mine[-1] if mine else 0) + 1, "live": True})
    return out


def round_now(exp: dict, rows: list[dict]) -> int:
    """The round the experiment is in, counting one in flight."""
    return max([r["round"] for r in rows + live_rows(exp, rows)] or [0])


# --- what every seat is holding ---------------------------------------------


def latest_transfer(latest: dict, table: list[harness.Channel]) -> dict | None:
    """The transfer submitted by the latest episode and what settlement made of it."""
    if harness.schema_channel(table) is None:
        return None
    resolved = latest.get("transfer") or {}
    if not resolved.get("declared"):
        return None
    return {
        "declared": resolved.get("declared"),
        "submitted": True,
        "seat": resolved.get("seat"), "label": resolved.get("label"), "agent": resolved.get("agent"),
        "amount": resolved.get("amount") or 0, "rebate": resolved.get("rebate") or 0,
        "error": resolved.get("error"),
    }


def obligations(t: dict | None) -> dict[str, bool | None]:
    """What each obligation this episode owed came to, by channel name, in table order.

    One entry a channel and not three by kind: the harness settles every obligated
    channel apart, so a table declaring two blackboards owes two things and says
    so here. What was met and what was charged are two questions: a share is taken
    only from an episode the API answered, past the grace, at a rate above zero,
    so an episode can leave every obligation undone and be charged for none. Empty
    with no trace. Every pane that states an obligation states it from here, so no
    two of them can answer differently.
    """
    if t is None:
        return {}
    return {ch.name: analyze.met_of(ch, rec) for ch, rec in analyze.settled_channels(t)}


def unmet(t: dict | None) -> list[dict]:
    """Each obligation this episode left undone, in table order, and what it cost.

    Said in the words the harness's own console line uses, so the page and the
    console cannot describe one episode differently. `penalty` is what was taken,
    which is zero inside the grace and at a rate of zero.
    """
    out = []
    for ch, rec in analyze.settled_channels(t or {}):
        if analyze.met_of(ch, rec) is not False:
            continue
        if ch.schema:
            why = rec.get("error") or "no transfer of its own"
        elif ch.shape == "mailbox":
            why = harness.outbox_why(rec, ch)
        else:
            why = "no post"
        out.append({"channel": ch.name, "why": why, "penalty": rec.get("penalty") or 0})
    return out


def seat_row(seat: str | None, agent: str, rows: list[dict], rnd: int) -> dict:
    """One seat's tile: what it holds, what it is doing, and what it has moved."""
    account = account_of(agent)
    ts = traces_of(agent)
    last = ts[-1] if ts else None
    episodes = account.get("episodes") or []
    latest = episodes[-1] if episodes else {}
    live = live_index(agent)
    try:
        interaction = InteractionStore(harness.interactions_root()).current(agent)
    except InteractionError:
        # Left out, as /api/interaction leaves it out: the seat's own route reports the
        # file, and the seat's next request is written over it.
        interaction = None
    going = live_state(agent, live, account)
    table, _ = agent_environment(last, agent)
    mine = [r for r in rows if r["agent"] == agent]
    out_reason = harness.why_out(account) if account else None
    # Not having acted in the round yet is two things, and the round has to be
    # over to tell them apart: for most of a sequential round some seats have
    # simply not been reached.
    pending = (live is None and not account.get("eliminated")
               and latest.get("stop") not in harness.STOPS_THE_AGENT
               and (mine[-1]["round"] if mine else 0) == rnd - 1)
    return {
        "seat": seat, "agent": agent, "label": account.get("label") or seat,
        "provider": account.get("provider"), "model": account.get("model"),
        "starter_files": (account.get("starter_files_landed") or {}).get("name")
                         or account.get("starter_files") or "",
        "n": account.get("remaining"), "initial": account.get("initial"),
        "series": thin(account.get("series") or []),
        # Derived from the raw log until the trace lands, which is what the
        # header labels it as: the arithmetic is the account's, the commit is not.
        "live": live, "live_age": live_age(agent, live),
        "pending_human": interaction.as_dict() if interaction else None,
        "live_turns": len(going["turns"]),
        "live_balance": going["balance"],
        "committed": len(episodes),
        "round": mine[-1]["round"] if mine else 0,
        "acted": bool(mine and mine[-1]["round"] == rnd) or live is not None,
        "pending": pending,
        "preparing": bool(account and not episodes and live is None and
                          (account.get("experiment") or {}).get("manifest_sha256")),
        "out_reason": out_reason, "eliminated": account.get("eliminated"),
        "last_election": account.get("last_election"),
        # What its turns cost, summed from the episodes that ran them and the one
        # in flight. A transfer, a share taken and a floor all move the balance
        # without being spend, so the drop from initial is a different number,
        # which the bar above draws.
        "spent": sum(s["spent"] for s in episodes) + going["spent"],
        "spent_this_round": sum(r["trace"]["spent"] for r in mine if r["round"] == rnd)
                            + going["spent"],
        "stop": last["stop"] if last else None,
        "halted": bool(last and last["stop"] in harness.STOPS_THE_AGENT),
        # What the last committed episode owed and left undone, one entry a
        # channel, each in the words the console line uses.
        "unmet": unmet(last),
        "refused": sum(len(analyze.refused_turns_of(t)) for t in ts),
        "drift": (last or {}).get("provenance_drift") or [],
        # Everything that moved the balance without being a turn. Read off the
        # account, not summed from the traces: these are cumulative there, and
        # an agent can be credited between its own starts.
        "sent": account.get("sent", 0), "received": account.get("received", 0),
        "rebated": account.get("rebated", 0),
        # What each channel's silence has cost, by channel name.
        "penalised": account.get("penalised") or {},
        "forgiven": account.get("forgiven", 0),
        "transfer": latest_transfer(latest, table),
    }


def header(exp: dict) -> dict:
    """What every seat is holding, and the ledger they all read.

    Each balance comes from its agent's own account, the source the harness
    renders the balance files from. The ledger is harness.ledger for any one
    member; every reader computes it alike.
    """
    rows = experiment_episodes(exp)
    rnd = round_now(exp, rows)
    first = exp["members"][0]
    account = account_of(first)
    last = rows[-1]["trace"] if rows else None
    table, hf = agent_environment(last, first)
    schema = harness.schema_channel(table)
    stamp = account.get("experiment") or {}
    seats = [seat_row(seat, agent, rows, rnd) for seat, agent in places_of(exp)]
    vote = next((tool for tool in exp.get("tools", []) if tool.get("kind") == "vote"), None)
    elections = {}
    for seat in seats:
        election = seat.get("last_election") or {}
        if isinstance(election.get("round"), int):
            elections[election["round"]] = election
    every = vote.get("every") if vote else None
    phase = ("vote complete" if rnd in elections else
             "vote" if every and rnd and rnd % every == 0 else "discussion")
    experiment_id = exp.get("experiment_id") or stamp.get("experiment_id") or exp["name"]
    records = product.records(harness.ROOT, experiment_id)
    cost = product.cost(exp["members"], account_of, stamp.get("cost") or {}, providers.is_interactive)
    return {
        "experiment": exp["name"], "experiment_id": experiment_id,
        "seated": exp["seated"], "posts": exp["posts"],
        "members": exp["members"], "tabs": tabs(exp), "labels": exp["labels"],
        "balance": hf["balance"],
        "ledger_name": schema.ledger if schema else "",
        "seats": seats,
        "ledger": [list(g) for g in harness.ledger(first, account)] if exp["seated"] else [],
        "round": rnd,
        "schedule": (last["provenance"].get("schedule") if last else None)
                    or stamp.get("schedule") or "",
        "stop_when_one_remains": stamp.get("stop_when_one_remains", False),
        "active": sum(seat["out_reason"] is None for seat in seats),
        "vote_every": every,
        "phase": phase,
        "cycle": ((rnd - 1) // every + 1) if every and rnd else None,
        "elections": [elections[r] for r in sorted(elections)],
        "progress": records["progress"], "outcome": records["outcome"],
        "lineage": records["lineage"], "cost": cost,
    }


# --- the message log --------------------------------------------------------


class Mailroom(NamedTuple):
    """The experiment's mailbox and schema channel, read once for a whole log."""
    exp: dict
    mail: harness.Channel | None
    schema: harness.Channel | None


def mailroom(exp: dict) -> Mailroom:
    table = experiment_table(exp)
    return Mailroom(exp, harness.mailbox_channel(table), harness.schema_channel(table))


def outbox_of(t: dict) -> dict[str, str | None]:
    """What the agent was addressing or declaring when the episode ended, by path.

    snapshot runs after the writable trees are mirrored back, so a trace holds
    the outbox its episode left, not the one it opened on.
    """
    return {f["path"]: f["text"] for f in analyze.outbox_files(t) + analyze.schema_files(t)}


def outbox_now(agent: str, room: Mailroom) -> dict[str, str | None]:
    """The host mirror of the outbox, which is the latest episode's delivery.

    Ahead of the last trace between an episode's files being mirrored back and
    its trace being written, and permanently for an episode that wrote none.
    A receipt the harness planted there is its own and left out. A schema file is
    included even where the experiment declares no mailbox.
    """
    out = {}
    if room.mail:
        root = harness.mirror(agent, room.mail.name)
        for p in sorted(root.rglob("*")) if root.is_dir() else []:
            if not p.is_file():
                continue
            path = f"{room.mail.outbox}/{p.relative_to(root).as_posix()}"
            if room.schema and path == room.schema.receipt:
                continue
            got = read_file(p)
            if got is not None:
                out[path] = got[1]
    if room.schema:
        account = account_of(agent)
        inst = next((i for i in harness.environment(agent, account, experiment_table(room.exp))
                     if i.writable and i.channel.name == room.schema.name), None)
        if inst and room.schema.shape == "mailbox":
            root = inst.host
            for p in sorted(root.iterdir()) if root.is_dir() else []:
                got = read_file(p) if p.is_file() else None
                if got is not None:
                    out[f"{room.schema.outbox}/{p.name}"] = got[1]
        else:
            got = read_file(inst.host) if inst else None
            if got is not None:
                out[room.schema.path] = got[1]
    return out


def addressed_to(path: str, room: Mailroom) -> tuple[str | None, str | None]:
    """The seat and label a path in an outbox reaches; absent for a parsed file.

    <outbox>/<label> arrives at that label's seat as <inbox>/<this agent's label>
    and nowhere else. A parsed file reaches no one; what it moves shows up in the
    ledger. A transfer-mailbox slot reaches its addressee as well as settlement.
    """
    if room.schema:
        if room.schema.shape == "file" and path == room.schema.path:
            return None, None
        if room.schema.shape == "mailbox" and path.startswith(room.schema.outbox + "/"):
            label = path[len(room.schema.outbox) + 1:]
            return seat_of_label(room.exp, label), label
    if room.mail and path.startswith(room.mail.outbox + "/"):
        label = path[len(room.mail.outbox) + 1:]
        return seat_of_label(room.exp, label), label
    return None, None


def change_of(before: Any, after: Any) -> str:
    """What one uncommitted path did relative to the latest trace."""
    if before is ABSENT:
        return "sent"
    if after is ABSENT:
        return "withdrawn"
    return "edited" if before != after else "standing"


def transfer_path(room: Mailroom, path: str) -> bool:
    """Whether a path is the current episode's transfer declaration."""
    return bool(room.schema and
                (path == room.schema.path or
                 (room.schema.shape == "mailbox" and
                  path.startswith(room.schema.outbox + "/"))))


def message_event(room: Mailroom, row: dict, path: str, before: Any, after: Any,
                  tip: bool = False) -> dict:
    """One movement of one path in one outbox."""
    exp = room.exp
    change = change_of(before, after)
    text = None if after is ABSENT else after
    seat, label = addressed_to(path, room)
    from_seat = row["seat"]
    ev = {
        "round": None if tip else row["round"], "at": row["at"], "episode": row["episode"],
        "from_seat": from_seat, "from_label": exp["labels"].get(from_seat or "", from_seat),
        "from_agent": row["agent"],
        "to_seat": seat, "to_label": label, "to_agent": exp["seats"].get(seat) if seat else None,
        "path": path, "kind": "transfer" if transfer_path(room, path) else "message",
        "change": change,
        "size": len(text.encode("utf-8")) if text else 0,
        "text": text, "binary": after is not ABSENT and after is None,
        "diff": [], "transfer": None, "delivered": None, "tip": tip,
    }
    if change == "edited" and isinstance(before, str) and isinstance(after, str):
        ev["diff"] = analyze.state_changes({path: before}, {path: after})
    if ev["kind"] == "transfer":
        resolved = (row["trace"] or {}).get("transfer") or {}
        raw = ((f"{ev['to_label']} {(text or '').strip()}")
               if room.schema and room.schema.shape == "mailbox" else (text or "").strip())
        line = harness.TRANSFER_LINE.match(raw)
        ev["transfer"] = resolved
        ev["to_label"] = resolved.get("label") or (line.group("label") if line else None)
        ev["to_seat"] = resolved.get("seat") or seat_of_label(exp, ev["to_label"])
        ev["to_agent"] = resolved.get("agent") or exp["seats"].get(ev["to_seat"] or "")
    return ev


def delivery_of(ev: dict, rows: list[dict], carried_paths: dict[tuple, set[str] | None],
                room: Mailroom) -> dict | None:
    """The addressee's next episode after the message was written, and what it held.

    Delivery is the addressee's first episode to start after this one. `shown_before` is
    the inbox arriving in that episode's observation, and is None where the initial
    observation carried nothing at all, an arrangement where the inbox was there to
    be fetched and nothing was handed over. `environment` is the inbox being in the
    environment either way, `named` is a command of that episode naming it, and
    `clipped` says the initial observation ran past its ceiling, which is how a
    section goes missing.
    """
    if ev["tip"] or ev["kind"] == "transfer" or not ev["to_agent"] or room.mail is None:
        return None
    nxt = next((r for r in rows if r["agent"] == ev["to_agent"] and r["at"] > ev["at"]), None)
    if nxt is None:
        return None
    box = f"{room.mail.inbox}/{ev['from_label']}"
    shown_box = digest_name(nxt["agent"], nxt["trace"], box)
    paths = carried_paths.get((nxt["agent"], nxt["episode"]))
    return {"round": nxt["round"], "episode": nxt["episode"], "box": box,
            "shown_as": shown_box,
            "shown_before": None if paths is None else shown_box in paths,
            "environment": any(f["path"] == box for f in analyze.inbox_files(nxt["trace"])),
            "named": any(analyze.names(s, box) for s in analyze.reached(nxt["trace"])),
            "clipped": observation_clipped(nxt["trace"])}


def messages(exp: dict, since: int = 0) -> dict:
    """Every mailbox and schema-channel event, in round order.

    Under episode delivery a message lasts one episode, so what a committed outbox
    holds was sent afresh by that episode. A trace naming no message_delivery was
    written while an outbox stood until changed, so its messages are diffed against
    the sender's episode before: standing where unchanged, withdrawn where gone.
    A transfer declaration is every episode's own under either.
    """
    rows = experiment_episodes(exp)
    room = mailroom(exp)
    events, tips = [], []
    for seat, agent in places_of(exp):
        prev: dict[str, Any] = {}
        last = None
        for row in [r for r in rows if r["agent"] == agent]:
            # An episode whose files were never mirrored back carries the
            # previous episode's, so it says nothing about what moved.
            if not row["trace"].get("state_saved"):
                continue
            now = outbox_of(row["trace"])
            episodic = row["trace"]["provenance"].get("message_delivery") == "episode"
            for path in sorted(now if episodic else set(prev) | set(now)):
                if episodic or transfer_path(room, path):
                    if path in now:
                        events.append(message_event(room, row, path, ABSENT, now[path]))
                else:
                    events.append(message_event(room, row, path,
                                                prev.get(path, ABSENT), now.get(path, ABSENT)))
            prev, last = now, row
        head = last or {"round": None, "at": None, "episode": None, "seat": seat,
                        "agent": agent, "trace": None}
        tip = outbox_now(agent, room)
        for path in sorted(set(prev) | set(tip)):
            if transfer_path(room, path):
                continue
            before, after = prev.get(path, ABSENT), tip.get(path, ABSENT)
            if change_of(before, after) != "standing":
                tips.append(message_event(room, head, path, before, after, tip=True))

    events.sort(key=lambda e: (e["round"], e["at"], e["from_seat"] or "", e["path"]))
    # Every event is resolved against every episode on every poll, and an observation
    # is the largest thing a trace holds, so each is parsed once for the lot.
    carried_paths = {(r["agent"], r["episode"]): message_paths(r["trace"]) for r in rows}
    for ev in events:
        ev["delivered"] = delivery_of(ev, rows, carried_paths, room)
    return {"experiment": exp["name"], "posts": exp["posts"], "seats": len(places_of(exp)),
            "committed": len(events), "events": events[since:], "tip": tips}


def player_history(exp: dict, agent: str) -> dict:
    """Public posts and the player's private conversations, including accepted sends."""
    rows = experiment_episodes(exp)
    table = experiment_table(exp)
    post_tool = next((tool for tool in exp.get("tools") or []
                      if tool.get("kind") == "post_public"), None)
    post_channel = next((ch for ch in table
                         if post_tool and ch.name == post_tool.get("channel")), None)
    public = []
    if post_channel:
        for row in rows:
            if not row["trace"].get("state_saved"):
                continue
            seat = row["seat"]
            account = account_of(row["agent"])
            label = exp.get("labels", {}).get(seat) or account.get("label") or seat or row["agent"]
            prefix = post_channel.path_for(label).strip("/")
            for rec in row["trace"].get("files") or []:
                if rec.get("channel") != post_channel.name or rec.get("role") != "own":
                    continue
                path = str(rec.get("path") or "").strip("/")
                if prefix and not path.startswith(prefix + "/"):
                    continue
                text = rec.get("text")
                if isinstance(text, str) and text:
                    public.append({"episode": row["round"], "agent_episode": row["episode"],
                                   "from_agent": row["agent"], "from_label": label,
                                   "text": text})
    private = [event for event in messages(exp)["events"]
               if event["kind"] == "message"
               and event["change"] not in ("withdrawn", "standing")
               and agent in (event["from_agent"], event["to_agent"])
               and isinstance(event.get("text"), str) and event["text"]]

    player_seat = next((seat for seat, member in places_of(exp) if member == agent), None)
    tool_kinds = {tool.get("name"): tool.get("kind") for tool in exp.get("tools") or []}
    public_keys = {(event["episode"], event["from_agent"], event["text"])
                   for event in public}
    private_keys = {(event["round"], event["from_agent"], event["to_label"], event["text"])
                    for event in private}
    for request, submission in InteractionStore(harness.interactions_root()).history(agent):
        episode = request.episode
        for call in submission.tool_calls:
            text = call.input.get("body")
            if not isinstance(text, str) or not text:
                continue
            kind = tool_kinds.get(call.name)
            if kind == "post_public":
                key = (episode, agent, text)
                if key not in public_keys:
                    public.append({"episode": episode, "agent_episode": episode,
                                   "from_agent": agent, "from_label": request.label,
                                   "text": text, "accepted": True})
                    public_keys.add(key)
            elif kind == "send_message_to":
                label = call.input.get("to")
                seat = seat_of_label(exp, label)
                key = (episode, agent, label, text)
                if isinstance(label, str) and key not in private_keys:
                    private.append({"round": episode, "episode": episode,
                                    "from_seat": player_seat,
                                    "from_label": request.label,
                                    "from_agent": agent, "to_seat": seat, "to_label": label,
                                    "to_agent": exp["seats"].get(seat) if seat else None,
                                    "text": text, "accepted": True})
                    private_keys.add(key)
    public.sort(key=lambda event: (event["episode"], event.get("agent_episode", 0),
                                   event["from_agent"], event["text"]))
    private.sort(key=lambda event: (event["round"], event["episode"],
                                    event["from_agent"], event.get("to_label") or ""))
    return {"experiment": exp["name"], "agent": agent, "public": public, "private": private}


# --- directory channels ---------------------------------------------------------


def trees(exp: dict) -> dict[str, harness.Channel]:
    """Every directory channel in the environment, by name: one tab each."""
    return {ch.name: ch for ch in experiment_table(exp) if ch.shape == "directory"}


def what_of(ch: harness.Channel) -> str:
    """Who reads a tree, for the line above its columns."""
    if ch.writer == "experimenter":
        return "provided by the experimenter; every agent reads the same files"
    return "every agent reads this one" if ch.readers == "all" else "no other agent ever reads this one"


def tabs(exp: dict) -> list[dict]:
    """The page's tabs: activity, directory declaration order, and transcripts."""
    out = []
    table = experiment_table(exp)
    mail, schema = harness.mailbox_channel(table), harness.schema_channel(table)
    if mail or schema:
        activity = " + ".join(ch.name for ch in (mail, schema) if ch)
        out.append({"key": "mailbox", "label": activity})
    out += [{"key": name, "label": name} for name in trees(exp)]
    return out + [{"key": "agent", "label": "transcripts"}]


def listing(root: Path, channel: str, given: set[str]) -> list[dict]:
    """Every file under one mirrored tree, with what the modes sidecar says.

    Records carry a stamp of mtime and size, not contents: a column per seat
    re-read every poll is a listing, and a file is read when it is opened.
    `given` is the paths the starter files put there.
    """
    modes = harness.read_modes(harness.modes_file(root))
    out = []
    for p in sorted(root.rglob("*")) if root.is_dir() else []:
        if not p.is_file():
            continue
        inner = p.relative_to(root).as_posix()
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"path": inner, "channel": channel, "size": st.st_size,
                    "mode": modes.get(inner),
                    "starter": inner in given,
                    "stamp": [st.st_mtime_ns, st.st_size]})
    return out


def given_in(ch: harness.Channel, account: dict) -> set[str]:
    """The paths the starter files put in this tree: only a private store holds any."""
    return harness.starter_paths(account) if ch.readers == "self" else set()


def ephemeral_tool(exp: dict, kind: str) -> dict | None:
    """The tool whose channel is cleared around each episode, if this is one."""
    return next((tool for tool in exp.get("tools") or []
                 if tool.get("channel") == kind
                 and tool.get("kind") in ("post_public", "vote")), None)


def completed_rounds(exp: dict, rows: list[dict]) -> list[int]:
    """Rounds whose last eligible episode has committed."""
    rounds = sorted({row["round"] for row in rows})
    if not rounds:
        return []
    latest = rounds[-1]
    if live_rows(exp, rows):
        return rounds[:-1]
    active = {agent for _, agent in places_of(exp) if harness.why_out(account_of(agent)) is None}
    acted = {row["agent"] for row in rows if row["round"] == latest}
    return rounds if not active or active <= acted else rounds[:-1]


def channel_history(exp: dict, kind: str, rows: list[dict] | None = None) -> dict | None:
    """The committed round snapshots an episode-scoped channel can show."""
    tool = ephemeral_tool(exp, kind)
    if tool is None:
        return None
    every = int(tool.get("every") or 1) if tool.get("kind") == "vote" else 1
    available = [rnd for rnd in completed_rounds(exp, rows or experiment_episodes(exp))
                 if rnd % every == 0]
    return {"kind": "vote" if tool.get("kind") == "vote" else "round",
            "every": every, "rounds": available}


def historical_listing(row: dict | None, ch: harness.Channel, kind: str,
                       label: str) -> list[dict]:
    """The files this seat owned when its episode in a round committed."""
    if row is None or not row["trace"].get("state_saved"):
        return []
    prefix = ch.path_for(label).strip("/")
    out = []
    for rec in row["trace"].get("files") or []:
        if rec.get("channel") != kind or rec.get("role") != "own":
            continue
        path = str(rec.get("path") or "").strip("/")
        if prefix:
            if not path.startswith(prefix + "/"):
                continue
            path = path[len(prefix) + 1:]
        if not path:
            continue
        out.append({"path": path, "channel": kind, "size": rec.get("size") or 0,
                    "mode": None, "starter": bool(rec.get("starter")),
                    "stamp": ["round", row["round"], row["episode"], rec.get("size") or 0]})
    return sorted(out, key=lambda rec: rec["path"])


def tree_view(exp: dict, kind: str, round_at: int | None = None) -> dict | None:
    """One tree of every seat's environment, a column each.

    save_state runs when an episode ends, so each column is current as of that
    agent's last committed episode and two columns can be stamped differently.
    """
    ch = trees(exp)[kind]
    rows = experiment_episodes(exp)
    history = channel_history(exp, kind, rows)
    if round_at is not None:
        if history is None or round_at not in history["rounds"]:
            return None
        columns = []
        for seat, agent in places_of(exp):
            account = account_of(agent)
            row = next((item for item in rows
                        if item["agent"] == agent and item["round"] == round_at), None)
            label = exp.get("labels", {}).get(seat) or account.get("label") or seat or agent
            columns.append({
                "seat": seat, "agent": agent, "label": account.get("label") or seat,
                "committed": row["episode"] if row else None, "live": None, "live_age": None,
                "saved": bool(row and row["trace"].get("state_saved")),
                "files": historical_listing(row, ch, kind, label),
            })
        return {"experiment": exp["name"], "kind": kind, "what": what_of(ch),
                "static": False, "history": {**history, "selected": round_at},
                "columns": columns}
    if ch.writer == "experimenter":
        files = listing(harness.files_dir(ch.source), kind, set())
        return {"experiment": exp["name"], "kind": kind, "what": what_of(ch),
                "static": True,
                "columns": [{"seat": None, "agent": "experimenter", "label": "experimenter",
                             "committed": None, "live": None, "live_age": None,
                             "files": files}]}
    columns = []
    for seat, agent in places_of(exp):
        account = account_of(agent)
        live = live_index(agent)
        columns.append({
            "seat": seat, "agent": agent, "label": account.get("label") or seat,
            "committed": len(account.get("episodes") or []),
            "live": live, "live_age": live_age(agent, live),
            "files": listing(harness.mirror(agent, kind), kind, given_in(ch, account)),
        })
    return {"experiment": exp["name"], "kind": kind, "what": what_of(ch),
            "static": False, "history": ({**history, "selected": None} if history else None),
            "columns": columns}


def file_view(exp: dict, agent: str, kind: str, inner: str,
              round_at: int | None = None) -> dict | None:
    """One file of one tree of one member of `exp`, found in a listing and never
    joined onto a root.

    The name off the URL is compared for equality against paths rglob produced
    under the tree, so no request can walk out of it by asking.
    """
    ch = trees(exp).get(kind)
    source = bool(ch and ch.writer == "experimenter")
    if (ch is None or (source and agent != "experimenter")
            or (not source and agent not in exp["members"])):
        return None
    if round_at is not None:
        rows = experiment_episodes(exp)
        history = channel_history(exp, kind, rows)
        if history is None or round_at not in history["rounds"]:
            return None
        row = next((item for item in rows
                    if item["agent"] == agent and item["round"] == round_at), None)
        seat = next((place for place, member in places_of(exp) if member == agent), None)
        account = account_of(agent)
        label = exp.get("labels", {}).get(seat) or account.get("label") or seat or agent
        rec = next((item for item in historical_listing(row, ch, kind, label)
                    if item["path"] == inner), None)
        if rec is None or row is None:
            return None
        prefix = ch.path_for(label).strip("/")
        full = f"{prefix}/{inner}" if prefix else inner
        source_rec = next((item for item in row["trace"].get("files") or []
                           if item.get("channel") == kind and item.get("role") == "own"
                           and str(item.get("path") or "").strip("/") == full), None)
        if source_rec is None:
            return None
        return {**rec, "agent": agent, "kind": kind,
                "size": source_rec.get("size") or 0, "text": source_rec.get("text")}
    root = harness.files_dir(ch.source) if source else harness.mirror(agent, kind)
    given = set() if source else given_in(ch, account_of(agent))
    rec = next((f for f in listing(root, kind, given) if f["path"] == inner), None)
    if rec is None:
        return None
    got = read_file(root / inner)
    if got is None:
        return None
    return {**rec, "agent": agent, "kind": kind, "size": got[0], "text": got[1]}


# --- one agent's transcript -------------------------------------------------


def agent_view(agent: str, exp: dict | None) -> dict:
    """One agent's episodes, each in the round it acted in. `exp` is the experiment
    the agent sits in, or None for one seated in no experiment."""
    account = account_of(agent)
    rows = experiment_episodes(exp) if exp else []
    rnd = {r["episode"]: r["round"] for r in rows if r["agent"] == agent}
    ts = traces_of(agent)
    live = live_index(agent)
    episodes = [{
        "episode": t["episode"], "round": rnd.get(t["episode"]),
        "stop": t["stop"], "spent": t["spent"], "turns": len(t["turns"]),
        "remaining": t["remaining"], "duration_s": t.get("duration_s"),
        "refused": len(analyze.refused_turns_of(t)),
        "transfer": t.get("transfer") or {}, "channels": analyze.channel_records(t),
        "forgiven": t.get("forgiven") or 0,
        "provenance": t["provenance"],
        "drift": t.get("provenance_drift") or [],
        "halted": t["stop"] in harness.STOPS_THE_AGENT,
        "live": False,
    } for t in ts]
    if live is not None:
        going = live_state(agent, live, account)
        episodes.append({
            "episode": live, "round": max(rnd.values(), default=0) + 1,
            "stop": None, "spent": going["spent"], "turns": len(going["turns"]),
            "remaining": going["remaining"], "duration_s": None,
            "refused": len([t for t in going["turns"] if t.get("refusal")]),
            "transfer": {}, "channels": {}, "forgiven": 0,
            "provenance": {}, "drift": [], "halted": False, "live": True,
        })
    seating = harness.seating_of(agent, account)
    return {
        "agent": agent, "experiment": exp["name"] if exp else "",
        "provider": account.get("provider"), "model": account.get("model"),
        "initial": account.get("initial"), "remaining": account.get("remaining"),
        "episodes": episodes,
        "live": live, "live_age": live_age(agent, live),
        # The mapping has no gap and holds every seat, this agent's among them, so
        # which one is its own has to be said: absence cannot say it.
        "seat": seating.seat, "peers": seating.seen,
        "starter_files": account.get("starter_files_landed") or {},
        "out_reason": harness.why_out(account) if account else None,
        "eliminated": account.get("eliminated"),
        "last_election": account.get("last_election"),
    }


def episode_view(agent: str, index: int, since: int = 0) -> dict | None:
    """One episode's transcript, from the trace or the raw log; None for neither.

    `since` is the last turn the page holds, so an episode in flight appends.
    `source` changing from raw to trace tells the page to ask again from zero.
    """
    if not harness.trace_path(agent, index).exists() and not harness.raw_path(agent, index).exists():
        return None
    trace = load_trace(harness.trace_path(agent, index))
    out = raw_view(agent, index, since) if trace is None else traced_view(agent, index, trace, since)
    turns = out["turns"]
    out["turns"] = [t for t in turns if (t["turn"] or 0) > since]
    out["total_turns"] = len(turns)
    return out


def traced_view(agent: str, index: int, trace: dict, since: int) -> dict:
    """A finished episode, read off its trace. The observation and the diffs travel
    only with the first request (`since` 0); the page holds them from there."""
    table = harness.table_of(trace)
    mail = harness.mailbox_channel(table)
    out = {
        "source": "trace", "live": False, "age": None, "episode": index,
        "stop": trace["stop"], "spent": trace["spent"], "remaining": trace["remaining"],
        "duration_s": trace.get("duration_s"), "error": trace.get("error"),
        "series_before": trace.get("series_before") or [],
        "series_after": trace.get("series_after") or [],
        "missing_tools": trace.get("missing_tools") or [],
        "balance_fits": trace.get("balance_fits"), "read_balance": trace.get("read_balance"),
        "transfer": trace.get("transfer") or {}, "channels": analyze.channel_records(trace),
        "forgiven": trace.get("forgiven") or 0,
        "obligations": obligations(trace), "unmet": unmet(trace),
        "turns": from_trace(trace),
    }
    if since == 0:
        listing, sections = observation_split(trace.get("observation") or "")
        out["observation"] = {
            "command": trace["commands"][0],
            "shell": trace["provenance"].get("shell_tool", True),
            "result": trace.get("observation") or "",
            # `listing` and `shown_before` are `result` split where the digest's
            # record begins; `name` is the digest file the record came from.
            "name": analyze.harness_files_of(trace)["digest"],
            "inbox": mail.inbox if mail else None,
            "inbox_prefix": inbox_prefix(mail),
            "listing": listing, "shown_before": sections,
            "clipped": observation_clipped(trace),
        }
        out["changes"] = episode_changes(agent, index)
    return out


def raw_view(agent: str, index: int, since: int) -> dict:
    """An episode in flight, derived from its raw log.

    The table, the harness file names and the delivery are the agent's latest
    trace's, and the ones in force before it has a trace. The environment at
    episode start is recorded in the trace and nowhere else, so while the
    episode runs it is pending like any other command's output.
    """
    account = account_of(agent)
    going = live_state(agent, index, account)
    last = latest_trace(agent)
    table, hf = agent_environment(last, agent)
    found = manifest_ahead_of(last, agent)
    delivery = (found[0]["overrides"].get("delivery", harness.DELIVERY) if found else
                (last["provenance"]["delivery"] if last else harness.DELIVERY))
    shell = (any(t.get("kind") == "bash" for t in found[0]["tools"]) if found else
             (last["provenance"].get("shell_tool", True) if last else harness.SHELL_TOOL))
    mail = harness.mailbox_channel(table)
    out = {
        "source": "raw", "live": True, "age": live_age(agent, index), "episode": index,
        "stop": None, "spent": going["spent"], "remaining": going["remaining"],
        "duration_s": None, "error": None,
        "series_before": account.get("series") or [], "series_after": [],
        "missing_tools": [], "balance_fits": None, "read_balance": None,
        "transfer": {}, "channels": {}, "forgiven": 0,
        "obligations": obligations(None), "unmet": unmet(None),
        "turns": going["turns"],
    }
    if since == 0:
        out["observation"] = {"command": harness.observation(table, hf["digest"], delivery, shell),
                              "shell": shell,
                              "result": None, "name": hf["digest"],
                              "inbox": mail.inbox if mail else None,
                              "inbox_prefix": inbox_prefix(mail), "listing": None,
                              "shown_before": [], "clipped": False}
    return out


def episode_changes(agent: str, index: int) -> list[dict]:
    """An episode's diffs against the episode before it, one block per channel the
    agent writes, in table order.

    Each block says who can see it: what the agent kept to itself, what it put
    where every other agent reads it, what it addressed to one of them, and what
    it declared to the harness.
    """
    def files(t: dict | None, name: str) -> dict[str, str]:
        return {f["path"]: f["text"] for f in (t or {}).get("files") or []
                if f["channel"] == name and f["role"] == "own" and f["text"] is not None}

    this = load_trace(harness.trace_path(agent, index))
    if this is None:
        return []
    before = load_trace(harness.trace_path(agent, index - 1))
    label = analyze.label_of(this)
    out = []
    for ch in harness.table_of(this):
        if ch.writer != "self":
            continue
        if ch.shape == "mailbox":
            what = f"{ch.outbox}/ · one file each, one agent reads it"
        elif ch.shape == "file":
            what = f"{ch.path} · the harness parses it"
        elif ch.readers == "self":
            what = f"{ch.path}/ · nobody else reads this"
        else:
            what = f"{ch.path_for(label)}/ · every agent reads this"
        out.append({"channel": ch.name, "what": what,
                    "lines": analyze.state_changes(files(before, ch.name), files(this, ch.name))})
    return out


# --- the server -------------------------------------------------------------


class View(http.server.BaseHTTPRequestHandler):
    """Observational GET routes plus a narrow interactive-provider submission route."""

    server_version = "view.py"

    def do_GET(self) -> None:                    # noqa: N802 - BaseHTTPRequestHandler's name
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        query = urllib.parse.parse_qs(url.query)
        try:
            self.route(parts, query)
        except ConnectionError:                  # the page navigated away mid-answer
            pass
        except Exception as e:                   # noqa: BLE001 - a viewer never takes the page down
            self.send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    def do_POST(self) -> None:                   # noqa: N802 - BaseHTTPRequestHandler's name
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        if len(parts) != 4 or parts[:2] != ["api", "interaction"]:
            return self.send_json({"error": "no such route"}, status=404)
        if parts[2] not in agent_names():
            return self.send_json({"error": f"no agent {parts[2]}"}, status=404)
        server = self.view_server
        origin = self.headers.get("Origin")
        if origin is None or origin != server.origin:
            return self.send_json({"error": "origin refused"}, status=403)
        token = self.headers.get("X-Interaction-Token")
        if token is None or token != server.control_token:
            return self.send_json({"error": "control token required"}, status=403)
        if self.headers.get_content_type() != "application/json":
            return self.send_json({"error": "Content-Type must be application/json"}, status=415)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_INTERACTION_BODY:
            return self.send_json({"error": "request body size refused"}, status=413)
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("body is not an object")
            result = self.interactions().submit(parts[2], parts[3], payload)
        except (InvalidSubmission, ValueError, json.JSONDecodeError) as error:
            return self.send_json({"error": str(error)}, status=400)
        except StaleRequest as error:
            return self.send_json({"error": str(error)}, status=409)
        except InteractionConflict as error:
            return self.send_json({"error": str(error)}, status=409)
        except UnreadableRecord as error:
            return self.send_json({"error": str(error)}, status=500)
        except InteractionError as error:
            return self.send_json({"error": str(error)}, status=404)
        return self.send_json({"submission": result.as_dict()}, status=201)

    def route(self, parts: list[str], query: dict[str, list[str]]) -> None:
        """One request. `parts` is the path split on slashes, already unquoted.

        A name off the URL reaches the filesystem only after matching one
        already there, so no path can be walked out of records/ or environments/.
        """
        if not parts:
            return self.send_page()
        if parts == ["api", "experiments"]:
            return self.send_json({"experiments": experiments(), "focus": self.view_server.focus,
                                   "poll": POLL_MS, "stale": STALE_AFTER, "root": str(harness.ROOT)})
        if parts == ["api", "interaction"]:
            pending = self.interactions().pending()
            return self.send_json({"requests": [request.as_dict() for request in pending]})
        if len(parts) == 3 and parts[:2] == ["api", "interaction"]:
            if parts[2] not in agent_names():
                return self.send_json({"error": f"no agent {parts[2]}"}, status=404)
            request = self.interactions().current(parts[2])
            if request is None:
                return self.send_json({"request": None})
            return self.send_json({"request": request.as_dict()})
        if len(parts) >= 3 and parts[:2] == ["api", "experiment"]:
            exp = experiment_named(parts[2])
            if exp is None:
                return self.send_json({"error": f"no experiment {parts[2]}"}, status=404)
            rest = parts[3:]
            if not rest:
                return self.send_json(header(exp))
            if rest == ["messages"]:
                since = query.get("since", ["0"])[0]
                return self.send_json(messages(exp, int(since) if since.isdigit() else 0))
            if rest == ["player-history"]:
                agent = query.get("agent", [""])[0]
                if agent not in exp["members"]:
                    return self.send_json({"error": f"no agent {agent} in experiment"}, status=404)
                return self.send_json(player_history(exp, agent))
            if len(rest) == 2 and rest[0] == "tree" and rest[1] in trees(exp):
                selected = query.get("round", [""])[0]
                if selected and not selected.isdigit():
                    return self.send_json({"error": f"invalid round {selected}"}, status=404)
                got = tree_view(exp, rest[1], int(selected) if selected else None)
                if got is None:
                    return self.send_json({"error": f"no round {selected} snapshot for {rest[1]}"},
                                          status=404)
                return self.send_json(got)
            if rest == ["file"]:
                agent = query.get("agent", [""])[0]
                kind = query.get("channel", [""])[0]
                inner = query.get("path", [""])[0]
                selected = query.get("round", [""])[0]
                if selected and not selected.isdigit():
                    return self.send_json({"error": f"invalid round {selected}"}, status=404)
                got = file_view(exp, agent, kind, inner, int(selected) if selected else None)
                if got is None:
                    return self.send_json({"error": f"no {kind} file {inner} in {agent}"}, status=404)
                return self.send_json(got)
        if len(parts) >= 3 and parts[:2] == ["api", "agent"]:
            agent = parts[2]
            if agent not in agent_names():
                return self.send_json({"error": f"no agent {agent}"}, status=404)
            rest = parts[3:]
            if not rest:
                return self.send_json(agent_view(agent, experiment_of(agent)))
            if len(rest) == 2 and rest[0] == "episode" and rest[1].isdigit():
                since = query.get("since", ["0"])[0]
                view = episode_view(agent, int(rest[1]), int(since) if since.isdigit() else 0)
                if view is None:
                    return self.send_json({"error": f"no episode {rest[1]} in {agent}"}, status=404)
                return self.send_json(view)
        return self.send_json({"error": "no such route"}, status=404)

    def send_page(self) -> None:
        body = PAGE.replace("__INTERACTION_CONTROL_TOKEN__",
                            self.view_server.control_token).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The page is read once at import and served uncached.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Polls of the same URL are answered from disk, never from the
        # browser's cache.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet. A line per poll is a line every second and a half, forever."""

    @property
    def view_server(self) -> ViewServer:
        """The server answering, with the focus, token and origin serve() gave it."""
        if not isinstance(self.server, ViewServer):
            raise TypeError(f"View is served by a ViewServer, not {type(self.server).__name__}")
        return self.server

    def interactions(self) -> InteractionStore:
        """The store the human provider publishes to, under the ROOT this request reads."""
        return InteractionStore(harness.interactions_root())


class ViewServer(http.server.ThreadingHTTPServer):
    """Loopback only. An interactive POST must carry `control_token`, which the served page
    embeds, and come from `origin`, this server's own."""

    focus: str | None
    control_token: str
    origin: str

    def __init__(self, port: int, focus: str | None):
        super().__init__(("127.0.0.1", port), View)
        self.focus = focus
        self.control_token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_address[1]}"


def serve(port: int = PORT, focus: str | None = None) -> ViewServer:
    """A server bound and ready, which the caller starts.

    Returned, not started, so a check can drive the real handler in-process.
    """
    return ViewServer(port, focus)


# --- cli --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI. Serves the page until interrupted."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT, help=f"default {PORT}; 0 picks a free one")
    ap.add_argument("--experiment", help="open on this set of agents")
    ap.add_argument("--agent", help="open on the experiment this agent sits in")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    a = ap.parse_args(argv)

    sets = experiments()
    focus = a.experiment
    if a.agent:
        if a.agent not in agent_names():
            ap.error(f"no agent {a.agent} under {harness.records_root()}")
        held = experiment_of(a.agent)
        focus = held["name"] if held else None
    if focus and not any(exp["name"] == focus for exp in sets):
        ap.error(f"no experiment {focus}; there is {', '.join(exp['name'] for exp in sets) or 'nothing'}")

    try:
        httpd = serve(a.port, focus)
    except OSError as e:
        print(f"port {a.port}: {e}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    seated = sum(exp["seated"] for exp in sets)
    print(f"{url}  ({len(sets)} sets, {seated} seated, under {harness.records_root()})")
    print("observer views are read-only; the player panel submits only to interactive seats")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
