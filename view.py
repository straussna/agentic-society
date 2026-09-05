"""Watch an experiment while it runs: py -3 view.py [--experiment h | --agent h02]

Serves a read-only page on 127.0.0.1 showing one experiment four ways - messages,
blackboards, private stores, and one transcript at a time - above every seat's n,
what it has spent, and the ledger g. Nothing it shows reaches the agent."""

from __future__ import annotations

import argparse
import http.server
import json
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import analyze
import harness

PORT = 8765

# What the page polls at, in milliseconds. Fast enough that a turn appears while
# the turn after it is still being thought about, slow enough that an experiment's
# traces are read once a second and a half rather than continuously.
POLL_MS = 1500

# An episode quiet for longer than this is not being waited on, it is over: no
# trace will follow it, and the next episode will take its index back. A turn is an
# API call plus the commands it runs, so the threshold clears a slow one.
STALE_AFTER = 180

# Points kept in a header sparkline. A long agent's series agents to thousands of
# elements and the strip is 240 pixels wide, so the rest is bytes on the wire for
# pixels that do not exist - once per seat, on every poll.
SPARK_POINTS = 240

# Bytes of a file read for the page. the harness's own snapshot bounds itself the same
# way, for the same reason: the agent can write anything.
FILE_LIMIT = 100_000

PENDING = "output arrives when the episode ends"

# The leading letters of an agent id, which is what names a set of them when
# nothing better is on disk: c01..c05 are the c agents.
AGENT_PREFIX = re.compile(r"^[^\d]*")

# The one path in an outbox that is not addressed to anybody.

# Stands for a path an outbox did not hold, which is not the same as a path it
# held with no text: a binary file reads as None and is still there.
ABSENT = object()


# --- reading what is on disk ------------------------------------------------


def read_json(path: Path) -> dict | None:
    """One JSON file, or None if it is not readable.

    save_account commits with os.replace, which on Windows surfaces to a reader as
    a PermissionError, so a poll landing on a commit is retried once.
    """
    for attempt in (1, 2):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, OSError, ValueError):
            if attempt == 2:
                return None
            time.sleep(0.05)
    return None


# path -> ((mtime, size), parsed). Guarded because the server is threaded and
# two polls can want the same trace at once.
_TRACES: dict[Path, tuple[tuple[int, int], dict]] = {}
_TRACE_LOCK = threading.Lock()


def load_trace(path: Path) -> dict | None:
    """One trace, parsed once.

    A trace is written whole when its episode ends and never touched again, so
    it is cached against the mtime and size that identify it.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
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


def agent_names(only: str | None = None) -> list[str]:
    """Every agent with an account, in name order.

    The account is what makes a directory an agent: records/analysis/ is where
    analyze.py writes when it was given no agent id, and it has none.
    """
    return [d.name for d in sorted((harness.ROOT / "records").glob("*"))
            if (d / "account.json").exists() and (only is None or d.name == only)]


def session_number(path: Path) -> int:
    """The index in an episode-NNNN file name."""
    return int(path.stem.rsplit("-", 1)[1])


def trace_path(agent: str, index: int) -> Path:
    return harness.records_dir(agent) / "traces" / f"episode-{index:04d}.json"


def raw_path(agent: str, index: int) -> Path:
    return harness.records_dir(agent) / "raw" / f"episode-{index:04d}.jsonl"


def trace_paths(agent: str) -> list[Path]:
    return sorted((harness.records_dir(agent) / "traces").glob("episode-*.json"))


def traces_of(agent: str) -> list[dict]:
    """Every finished episode of an agent, in order."""
    return [t for t in (load_trace(p) for p in trace_paths(agent)) if t is not None]


def live_index(agent: str) -> int | None:
    """The episode with no trace yet, or None if the agent is between starts.

    Unfinished is exactly a raw log with no trace beside it, which is not the
    same as running: how long since the log grew is what live_age reports.
    """
    raws = sorted((harness.records_dir(agent) / "raw").glob("episode-*.jsonl"))
    if not raws:
        return None
    index = session_number(raws[-1])
    return None if trace_path(agent, index).exists() else index


def live_age(agent: str, index: int) -> float | None:
    """Seconds since the episode's raw log last grew, or None if there is none.

    A turn takes as long as the API call plus the commands it runs, so a live
    episode is quiet for stretches; a dead one is quiet for good.
    """
    try:
        return max(0.0, time.time() - raw_path(agent, index).stat().st_mtime)
    except OSError:
        return None


def acting(agent: str, index: int | None) -> bool:
    """Whether the agent is moving rather than merely holding an unfinished index."""
    return index is not None and (live_age(agent, index) or 0) < STALE_AFTER


def latest_attempt(lines: list[dict]) -> list[dict]:
    """The last agent at an episode, out of a log that may hold more than one.

    An episode that died without writing a trace leaves its index free for the next
    episode, which appends to the same log. Turn numbers restarting at 1 is the seam.
    """
    starts = [i for i, line in enumerate(lines) if line.get("turn") == 1]
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


def read_modes(path: Path) -> dict[str, str]:
    """The modes sidecar as path -> mode, or empty if there is none."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        mode, _, rel = line.partition(" ")
        if rel:
            out[rel] = mode
    return out


def read_file(p: Path) -> tuple[int, str | None] | None:
    """One file's size and its text, bounded and marked where it is cut short.

    FILE_LIMIT bytes, which is what the harness's own snapshot bounds itself to and for
    the same reason. A NUL marks it binary, and text is None for one.
    """
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            data = f.read(FILE_LIMIT)
    except OSError:
        return None
    if b"\x00" in data:
        return size, None
    text = data.decode("utf-8", "replace")
    if size > len(data):
        text += f"\n[truncated: {size - len(data)} of {size} bytes]\n"
    return size, text


# --- the initial observation ------------------------------------------------------------


# harness.digest_for writes one `=== <path> ===` line per file it carries, and an observation
# is the episode listing followed by all of them. Anchored to whole lines, so the
# same shape inside a message's body is body and does not move the split.
SECTION = re.compile(r"^=== (?P<path>.+) ===$", re.M)


def observation_split(text: str) -> tuple[str, list[dict]]:
    """The listing an episode opened on, and the sections of m after it.

    The pieces concatenate back to what was sent: a section's body agents to the
    next header, and the listing is everything before the first one. An observation
    that carries no m is all listing.
    """
    marks = list(SECTION.finditer(text))
    if not marks:
        return text, []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() + 1:end]
        out.append({"path": m.group("path"), "text": body,
                    "bytes": len(body.encode("utf-8"))})
    return text[:marks[0].start()], out


def observation_clipped(t: dict) -> bool:
    """Whether the initial observation ran past the ceiling that episode served it under.

    harness.clip keeps a head and a tail with a marker between, so an observation over
    its own limit is one that lost a middle. The limit is the agent's, read off the
    episode's provenance rather than this process's config.
    """
    limit = (t.get("provenance") or {}).get("observation_limit")
    return bool(limit) and len(t.get("observation") or "") > limit


def message_paths(t: dict) -> set[str] | None:
    """Every path m named, or None where the initial observation carried none."""
    _, sections = observation_split(t.get("observation") or "")
    return {s["path"] for s in sections} if sections else None


def thin(series: list[int], points: int = SPARK_POINTS) -> list[int]:
    """A long series sampled down, keeping the first element and the last."""
    if len(series) <= points:
        return list(series)
    step = (len(series) - 1) / (points - 1)
    return [series[round(i * step)] for i in range(points - 1)] + [series[-1]]


# --- one turn, from either source -------------------------------------------


def namespace(x: Any) -> Any:
    """A parsed JSON value as something harness's own functions can read.

    measure_response and its neighbours reach into a response with getattr, so
    the live view prices a turn with the harness's arithmetic, not a copy of it.
    """
    if isinstance(x, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in x.items()})
    if isinstance(x, list):
        return [namespace(v) for v in x]
    return x


def from_trace(t: dict) -> list[dict]:
    """A finished episode's turns, with the command output they returned."""
    return [{
        "turn": turn.get("turn"),
        "micros": turn.get("micros"),
        "prefix": turn.get("prefix"),
        "balance": turn.get("balance"),
        "stop_reason": turn.get("stop_reason"),
        "stop_details": turn.get("stop_details"),
        "model": turn.get("model"),
        "served_by_fallback": bool(turn.get("served_by_fallback")),
        "unpriced_model": turn.get("unpriced_model"),
        "text": turn.get("text") or "",
        "thinking": turn.get("thinking") or "",
        "tools": [{"command": c.get("command"), "result": c.get("result")}
                  for c in turn.get("tools") or []],
        "tokens": {k: turn.get(k, 0) for k in analyze.TOKEN_KEYS},
    } for turn in t.get("turns") or []]


def from_raw(lines: list[dict], account: dict) -> list[dict]:
    """A running episode's turns, priced the way the harness prices them.

    Each response goes back through harness.measure_response: an id is billed once,
    a replay is zeroed. Command results are None until the trace lands.
    """
    model, remaining = account["model"], account["remaining"]
    centi, seen, out = 0, set(), []
    for line in lines:
        r = namespace(line.get("response") or {})
        rid = getattr(r, "id", None) or f"anon-{line.get('turn')}"
        u = harness.measure_response(r, model)
        previous = remaining - centi // 100
        if rid not in seen:
            seen.add(rid)
            centi += u["centi"]
        else:
            u = {**u, **dict.fromkeys(harness.BILLABLE, 0)}
        balance = remaining - centi // 100
        content = list(getattr(r, "content", None) or [])
        out.append({
            "turn": line.get("turn"),
            "received": line.get("received"),
            "micros": previous - balance,
            "prefix": u["prefix"],
            "balance": balance,
            "stop_reason": getattr(r, "stop_reason", None),
            "stop_details": harness.refusal_detail(r),
            "model": getattr(r, "model", None),
            "served_by_fallback": harness.served_by_fallback(r),
            "unpriced_model": u["unpriced"] or None,
            "text": harness.blocks(content, "text", "text"),
            "thinking": harness.blocks(content, "thinking", "thinking"),
            "tools": [{"command": command_of(b), "result": None}
                      for b in content if getattr(b, "type", "") == "tool_use"],
            "tokens": {k: u.get(k, 0) for k in analyze.TOKEN_KEYS},
        })
    return out


def command_of(block: Any) -> str | None:
    """The command a tool_use block asked for, or None for a bare restart."""
    return getattr(getattr(block, "input", None), "command", None)


def live_turns(agent: str, index: int, account: dict) -> list[dict]:
    """The turns of an unfinished episode, read off its raw log."""
    return from_raw(latest_attempt(raw_lines(raw_path(agent, index))), account)


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
    seat, seen = harness.seating(agent, account)
    if seen.get(seat) != agent:
        return None
    return tuple(seen[s] for s in sorted(seen, key=int))


def table_for(agent: str) -> list[harness.Channel]:
    """The channel table the agent last ran under, from its latest trace's provenance;
    the code default before it has one."""
    paths = trace_paths(agent)
    t = load_trace(paths[-1]) if paths else None
    return harness.channels_from(((t or {}).get("provenance") or {}).get("channels"))


def table_of(c: dict) -> list[harness.Channel]:
    """The experiment's channel table, as experiments() recorded it."""
    return harness.channels_from(c.get("channels"))


def seat_of_label(c: dict, label: str | None) -> str | None:
    """The seat an agent-facing label names, or None for a label no seat has."""
    return next((s for s, l in (c.get("labels") or {}).items() if l == label), None)


def experiments() -> list[dict]:
    """Every set of agents on disk, the seated ones first.

    Agents sharing a seating are one experiment, named by group_of. One whose mapping
    does not seat it is grouped by its id's letters and marked unseated.
    """
    groups: dict[tuple, dict] = {}
    for agent in agent_names():
        account = read_json(harness.records_dir(agent) / "account.json")
        if account is None:
            continue
        key = seating_key(agent, account)
        _, seen = harness.seating(agent, account)
        ident = key or ("unseated", group_of(agent, account))
        c = groups.get(ident)
        if c is None:
            c = groups[ident] = {
                "name": group_of(agent, account), "seated": key is not None,
                "seats": dict(seen) if key else {}, "members": [],
                "posts": False, "running": 0, "episodes": 0,
            }
        c["members"].append(agent)
        c["episodes"] += len(account.get("episodes") or [])
        c["running"] += acting(agent, live_index(agent))

    out = sorted(groups.values(), key=lambda c: (not c["seated"], c["name"]))
    # Two sets can arrive at one name - a seated experiment and a leftover agent whose
    # id starts with the same letters. The seated one is sorted first and keeps
    # the short name, so what the other is called says what it is.
    taken: set[str] = set()
    for c in out:
        c["members"].sort()
        if c["name"] in taken:
            c["name"] = "+".join(c["members"])
        taken.add(c["name"])
        # The table and the labels the members ran under, from the records of the
        # first member that has any: every member of one experiment ran under the
        # same ones.
        first = next((a for a in c["members"] if trace_paths(a)), c["members"][0])
        table = table_for(first)
        account = read_json(harness.records_dir(first) / "account.json") or {}
        c["channels"] = [ch.as_table() for ch in table]
        c["labels"] = ((account.get("peers") or {}).get("labels")) or {s: s for s in c["seats"]}
        mail = harness.mailbox_channel(table)
        c["posts"] = bool(mail) and any(harness.mirror(a, mail.name).is_dir() for a in c["members"])
    return out


def experiment_named(name: str) -> dict | None:
    return next((c for c in experiments() if c["name"] == name), None)


def experiment_of(agent: str) -> dict | None:
    """The experiment this agent sits in."""
    return next((c for c in experiments() if agent in c["members"]), None)


def places_of(c: dict) -> list[tuple[str | None, str]]:
    """Every agent of the experiment in the order the tabs show it.

    By seat where there are seats, which is the order the agents themselves see
    each other in, and by name where there are none.
    """
    if c["seated"]:
        return list(c["seats"].items())
    return [(None, agent) for agent in c["members"]]


def seats_by_run(c: dict) -> dict[str, str | None]:
    return {agent: seat for seat, agent in places_of(c)}


# --- the round --------------------------------------------------------------


def started_at(t: dict) -> str:
    """When the episode started, from its provenance."""
    return ((t.get("provenance") or {}).get("started_at")) or ""


def cohort_sessions(c: dict) -> list[dict]:
    """Every committed episode of every member, in the order they started.

    One episode per agent per round is what sequential_round holds to, so the round is
    read out of start order, cut where an agent would take a second turn.
    """
    rows = sorted(({"agent": agent, "episode": t["episode"], "at": started_at(t), "trace": t}
                   for _, agent in places_of(c) for t in traces_of(agent)),
                  key=lambda s: (s["at"], s["agent"], s["episode"]))
    rnd, acted = 0, set()
    for s in rows:
        if not rnd or s["agent"] in acted:
            rnd, acted = rnd + 1, set()
        acted.add(s["agent"])
        s["round"] = rnd
        s["live"] = False
    return rows


def live_rows(c: dict, rows: list[dict]) -> list[dict]:
    """The episodes in flight, each in the round it belongs to.

    An agent with a raw log and no trace is taking its turn now, which is the round
    after the last one it acted in.
    """
    out = []
    for _, agent in places_of(c):
        live = live_index(agent)
        if live is None:
            continue
        mine = [r["round"] for r in rows if r["agent"] == agent]
        out.append({"agent": agent, "episode": live, "at": None, "trace": None,
                    "round": (mine[-1] if mine else 0) + 1, "live": True})
    return out


def round_now(c: dict, rows: list[dict]) -> int:
    """The round the experiment is in, counting one in flight."""
    return max([r["round"] for r in rows + live_rows(c, rows)] or [0])


# --- what every seat is holding ---------------------------------------------


def standing_gift(agent: str, latest: dict, table: list[harness.Channel]) -> dict | None:
    """The declaration sitting in the parsed channel, and what the last episode made of it.

    A declaration re-applies every episode it is left in place. resolve_transfer's
    reason is the only statement anywhere of why one moved nothing. None where the
    table has no parsed channel.
    """
    schema = harness.schema_channel(table)
    if schema is None:
        return None
    account = read_json(harness.records_dir(agent) / "account.json") or {}
    inst = next((i for i in harness.environment(agent, account, table)
                 if i.writable and i.channel is schema), None)
    got = read_file(inst.host) if inst else None
    declared = got[1] if got else None
    resolved = latest.get("transfer") or {}
    if declared is None and not resolved.get("declared"):
        return None
    return {
        "declared": declared if declared is not None else resolved.get("declared"),
        "standing": declared is not None,
        "seat": resolved.get("seat"), "label": resolved.get("label"), "agent": resolved.get("agent"),
        "amount": resolved.get("amount") or 0, "rebate": resolved.get("rebate") or 0,
        "error": resolved.get("error"),
    }


def obligations(t: dict) -> dict:
    """The three things an episode owes, as its own record has them.

    What was met and what was charged are two questions: a share is taken only
    from an episode the API answered, past the grace, at a rate above zero, so a
    episode can leave all three undone and be charged for none of them. None
    where the record is silent. Every pane that states an obligation states it
    from here, so no two of them can answer differently.
    """
    transfer, board, msgs = t.get("transfer") or {}, analyze.board_of(t), analyze.mailbox_of(t)
    return {
        "posted": board.get("posted") if board else None,
        # resolve_mailbox' own rule: none and two break it as a crowded seat does.
        "messaged": None if not msgs
                    else not (msgs.get("broken") or len(msgs.get("addressed") or []) != 1),
        # A declaration left standing moves nothing a second time, so what counts
        # is money moved this episode and no share taken for having moved none.
        "transferred": None if "transfer" not in t
                  else bool(transfer.get("amount")) and not transfer.get("penalty"),
    }


def seat_row(seat: str | None, agent: str, rows: list[dict], rnd: int) -> dict:
    """One seat's tile: what it holds, what it is doing, and what it has moved."""
    account = read_json(harness.records_dir(agent) / "account.json") or {}
    ts = traces_of(agent)
    last = ts[-1] if ts else None
    episodes = account.get("episodes") or []
    latest = episodes[-1] if episodes else {}
    live = live_index(agent)
    turns = live_turns(agent, live, account) if live is not None else []
    mine = [r for r in rows if r["agent"] == agent]
    # Not having acted in the round yet is two things, and the round has to be
    # over to tell them apart: the order rotates, so for most of a round some
    # seats have simply not been reached. Nothing is asked about budget here -
    # admits() reads config only start() loads, so from here it would answer for
    # the defaults. The balance is on the tile beside this.
    pending = live is None and (mine[-1]["round"] if mine else 0) == rnd - 1
    # What the episode in flight has cost so far, which no episode record holds
    # yet. It belongs to this round and to the agent's whole life alike.
    live_spend = account.get("remaining", 0) - turns[-1]["balance"] if turns else 0
    met = obligations(latest)
    table = harness.channels_from(analyze.table_of(last)) if last else harness.channels()
    return {
        "seat": seat, "agent": agent, "label": account.get("label") or seat,
        "n": account.get("remaining"), "initial": account.get("initial"),
        "series": thin(account.get("series") or []),
        # Derived from the raw log until the trace lands, which is what the
        # header labels it as: the arithmetic is the account's, the commit is not.
        "live": live, "live_age": live_age(agent, live) if live is not None else None,
        "live_turns": len(turns),
        "live_balance": turns[-1]["balance"] if turns else None,
        "committed": len(episodes),
        "round": mine[-1]["round"] if mine else 0,
        "acted": bool(mine and mine[-1]["round"] == rnd) or live is not None,
        "pending": pending,
        # What its turns cost, summed from the episodes that ran them. A transfer, a
        # share taken and a floor all move the balance without being spend, so
        # the drop from initial is a different number - the bar above draws it.
        "spent": sum(s["spent"] for s in episodes) + live_spend,
        "spent_this_round": sum(r["trace"]["spent"] for r in mine if r["round"] == rnd)
                            + live_spend,
        "stop": last["stop"] if last else None,
        "halted": bool(last and last["stop"] in harness.STOP_THE_RUN),
        # What the last committed episode owed and met. The chip below says what
        # its transfer did, so the third is not repeated here.
        "posted": met["posted"], "messaged": met["messaged"],
        "refused": sum(len(analyze.refused_turns_of(t)) for t in ts),
        "fallback": sum(len(analyze.fallback_turns_of(t)) for t in ts),
        "drift": (last or {}).get("provenance_drift") or [],
        # Everything that moved the balance without being a turn. Read off the
        # account rather than summed from the traces, because these are cumulative
        # there and an agent can be credited between its own starts.
        "sent": account.get("sent", 0), "received": account.get("received", 0),
        "rebated": account.get("rebated", 0),
        # What each channel's silence has cost, by channel name.
        "penalised": account.get("penalised") or {},
        "forgiven": account.get("forgiven", 0),
        "transfer": standing_gift(agent, latest, table),
    }


def header(c: dict) -> dict:
    """What every seat is holding, and the ledger they all read.

    n comes from each agent's own account, the same source plant_readonly renders
    from. g is harness.ledger for any one member; every reader computes it alike.
    """
    rows = cohort_sessions(c)
    rnd = round_now(c, rows)
    first = c["members"][0]
    account = read_json(harness.records_dir(first) / "account.json") or {}
    hf = analyze.harness_files_of(rows[-1]["trace"]) if rows else dict(harness.HARNESS_FILES)
    return {
        "experiment": c["name"], "seated": c["seated"], "posts": c["posts"],
        "members": c["members"], "tabs": tabs(c), "labels": c.get("labels") or {},
        "balance": hf.get("balance", ""),
        "seats": [seat_row(seat, agent, rows, rnd) for seat, agent in places_of(c)],
        "ledger": [list(g) for g in harness.ledger(first, account)] if c["seated"] else [],
        "round": rnd,
        "model": account.get("model"),
        "starter_files": (account.get("starter_files_landed") or {}).get("name") or "",
        "poll": POLL_MS, "stale": STALE_AFTER, "root": str(harness.ROOT),
    }


# --- the message log --------------------------------------------------------


def outbox_of(t: dict) -> dict[str, str | None]:
    """What the agent was sending when the episode ended, by path.

    snapshot agents after the writable trees are mirrored back, so a trace holds
    the outbox its episode left rather than the one it opened on.
    """
    return {f["path"]: f["text"] for f in analyze.outbox_files(t) + analyze.schema_files(t)}


def outbox_now(agent: str, c: dict) -> dict[str, str | None]:
    """The host mirror of the outbox, which is what stands right now.

    Ahead of the last trace between an episode's files being mirrored back and
    its trace being written, and permanently for an episode that wrote none.
    Empty where the table has no mailbox. A receipt the harness planted there
    is its own and left out.
    """
    table = table_of(c)
    mail, schema = harness.mailbox_channel(table), harness.schema_channel(table)
    if mail is None:
        return {}
    root = harness.mirror(agent, mail.name)
    out = {}
    for p in sorted(root.rglob("*")) if root.is_dir() else []:
        if not p.is_file():
            continue
        path = f"{mail.outbox}/{p.relative_to(root).as_posix()}"
        if schema and path == schema.receipt:
            continue
        got = read_file(p)
        if got is not None:
            out[path] = got[1]
    return out


def addressed_to(path: str, c: dict) -> tuple[str | None, str | None]:
    """The seat and label a path in an outbox reaches; (None, None) for the declaration.

    <outbox>/<label> arrives at that label's seat as <inbox>/<this agent's label>
    and nowhere else. The parsed file reaches no one; what it moves shows up in
    the ledger.
    """
    table = table_of(c)
    mail, schema = harness.mailbox_channel(table), harness.schema_channel(table)
    if schema and path == schema.path:
        return None, None
    if mail and path.startswith(mail.outbox + "/"):
        label = path[len(mail.outbox) + 1:]
        return seat_of_label(c, label), label
    return None, None


def change_of(before: Any, after: Any) -> str:
    """What one path did between two of a sender's episodes."""
    if before is ABSENT:
        return "sent"
    if after is ABSENT:
        return "withdrawn"
    return "edited" if before != after else "standing"


def message_event(c: dict, by_run: dict, row: dict, path: str,
                  before: Any, after: Any, tip: bool = False) -> dict:
    """One movement of one path in one outbox."""
    change = change_of(before, after)
    text = None if after is ABSENT else after
    seat, label = addressed_to(path, c)
    schema = harness.schema_channel(table_of(c))
    from_seat = by_run.get(row["agent"])
    labels = c.get("labels") or {}
    ev = {
        "round": None if tip else row["round"], "at": row["at"], "episode": row["episode"],
        "from_seat": from_seat, "from_label": labels.get(from_seat or "", from_seat),
        "from_run": row["agent"],
        "to_seat": seat, "to_label": label, "to_run": c["seats"].get(seat) if seat else None,
        "path": path, "kind": "transfer" if schema and path == schema.path else "message",
        "change": change,
        "size": len(text.encode("utf-8")) if text else 0,
        "text": text, "binary": after is not ABSENT and after is None,
        "diff": [], "transfer": None, "delivered": None, "tip": tip,
    }
    if change == "edited" and isinstance(before, str) and isinstance(after, str):
        ev["diff"] = analyze.state_changes({path: before}, {path: after})
    if ev["kind"] == "transfer":
        resolved = (row["trace"] or {}).get("transfer") or {}
        line = harness.TRANSFER_LINE.match((text or "").strip())
        ev["transfer"] = resolved
        ev["to_label"] = resolved.get("label") or (line.group("label") if line else None)
        ev["to_seat"] = resolved.get("seat") or seat_of_label(c, ev["to_label"])
        ev["to_run"] = resolved.get("agent") or c["seats"].get(ev["to_seat"] or "")
    return ev


def delivery_of(ev: dict, rows: list[dict], carried_paths: dict[tuple, set[str] | None],
                c: dict) -> dict | None:
    """The addressee's next episode after the message was written, and what it held.

    Delivery is the addressee's first episode to start after this one. `shown_before` is
    the inbox arriving in that episode's observation, and is None where the initial observation
    carried nothing at all - an arrangement where the inbox was there to be
    fetched and nothing was handed over. `environment` is the inbox being in the environment
    either way, `named` is a command of that episode naming it, and `clipped`
    says the initial observation ran past its ceiling, which is how a section goes missing.
    """
    if ev["tip"] or ev["kind"] == "transfer" or not ev["to_run"]:
        return None
    nxt = next((r for r in rows if r["agent"] == ev["to_run"] and r["at"] > ev["at"]), None)
    if nxt is None:
        return None
    mail = harness.mailbox_channel(table_of(c))
    box = f"{mail.inbox if mail else 'in'}/{ev['from_label']}"
    paths = carried_paths.get((nxt["agent"], nxt["episode"]))
    return {"round": nxt["round"], "episode": nxt["episode"], "box": box,
            "shown_before": None if paths is None else box in paths,
            "environment": any(f["path"] == box for f in analyze.inbox_files(nxt["trace"])),
            "named": any(box in cmd for cmd in nxt["trace"].get("commands") or []),
            "clipped": observation_clipped(nxt["trace"])}


def messages(c: dict, since: int = 0) -> dict:
    """Every event in the mailbox channel, in round order.

    An outbox is a standing mirror, so the log is the difference between
    successive outboxes, per sender; the parsed file is in it. `since` counts events.
    """
    rows = cohort_sessions(c)
    by_run = seats_by_run(c)
    events, tips = [], []
    for _, agent in places_of(c):
        prev: dict[str, Any] = {}
        last = None
        for row in [r for r in rows if r["agent"] == agent]:
            # An episode whose files were never mirrored back carries the
            # previous episode's, so it says nothing about what moved.
            if not row["trace"].get("state_saved"):
                continue
            now = outbox_of(row["trace"])
            for path in sorted(set(prev) | set(now)):
                events.append(message_event(c, by_run, row, path,
                                            prev.get(path, ABSENT), now.get(path, ABSENT)))
            prev, last = now, row
        head = last or {"round": None, "at": None, "episode": None, "agent": agent, "trace": None}
        tip = outbox_now(agent, c)
        for path in sorted(set(prev) | set(tip)):
            before, after = prev.get(path, ABSENT), tip.get(path, ABSENT)
            if change_of(before, after) != "standing":
                tips.append(message_event(c, by_run, head, path, before, after, tip=True))

    events.sort(key=lambda e: (e["round"], e["at"], e["from_seat"] or "", e["path"]))
    # Every event is resolved against every episode on every poll, and an observation
    # is the largest thing a trace holds, so each is parsed once for the lot.
    carried_paths = {(r["agent"], r["episode"]): message_paths(r["trace"]) for r in rows}
    for ev in events:
        ev["delivered"] = delivery_of(ev, rows, carried_paths, c)
    return {"experiment": c["name"], "posts": c["posts"], "seats": len(places_of(c)),
            "committed": len(events), "events": events[since:], "tip": tips}


# --- the blackboards and the private stores --------------------------------------


def trees(c: dict) -> dict[str, harness.Channel]:
    """Every directory channel the agents write, by name: one tab each. The
    mailbox is the other thing they write, and the messages tab is what that is for."""
    return {ch.name: ch for ch in table_of(c) if ch.writer == "self" and ch.shape == "directory"}


def what_of(ch: harness.Channel) -> str:
    """Who reads a tree, for the line above its columns."""
    return "every agent reads this one" if ch.readers == "all" else "no other agent ever reads this one"


def tabs(c: dict) -> list[dict]:
    """The page's tabs, in table order: the mailbox, each tree, and the transcripts."""
    out = []
    if mail := harness.mailbox_channel(table_of(c)):
        out.append({"key": "mailbox", "label": mail.name})
    out += [{"key": name, "label": name} for name in trees(c)]
    return out + [{"key": "agent", "label": "transcripts"}]


def listing(root: Path, channel: str, given: set[str], store: bool = False) -> list[dict]:
    """Every file under one mirrored tree, with what the modes sidecar says.

    Records carry a stamp of mtime and size rather than contents: a column per
    seat re-read every poll is a listing, and a file is read when it is opened.
    """
    modes = read_modes(harness.modes_file(root))
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
                    "starter": store and inner in given,
                    "stamp": [st.st_mtime_ns, st.st_size]})
    return out


def tree_view(c: dict, kind: str) -> dict:
    """One tree of every seat's environment, a column each.

    save_state agents when an episode ends, so each column is current as of that
    agent's last committed episode and two columns can be stamped differently.
    """
    ch = trees(c)[kind]
    columns = []
    for seat, agent in places_of(c):
        account = read_json(harness.records_dir(agent) / "account.json") or {}
        live = live_index(agent)
        columns.append({
            "seat": seat, "agent": agent, "label": account.get("label") or seat,
            "committed": len(account.get("episodes") or []),
            "live": live, "live_age": live_age(agent, live) if live is not None else None,
            "files": listing(harness.mirror(agent, kind), kind, harness.starter_paths(account),
                             ch.readers == "self"),
        })
    return {"experiment": c["name"], "kind": kind, "what": what_of(ch), "columns": columns}


def file_view(agent: str, kind: str, inner: str) -> dict | None:
    """One file of one tree, found in a listing rather than joined onto a root.

    The name off the URL is compared for equality against paths rglob produced
    under the tree, so no request can walk out of it by asking.
    """
    c = experiment_of(agent)
    ch = trees(c).get(kind) if c else None
    if ch is None:
        return None
    root = harness.mirror(agent, kind)
    account = read_json(harness.records_dir(agent) / "account.json") or {}
    rec = next((f for f in listing(root, kind, harness.starter_paths(account), ch.readers == "self")
                if f["path"] == inner), None)
    if rec is None:
        return None
    got = read_file(root / inner)
    if got is None:
        return None
    return {**rec, "agent": agent, "kind": kind, "size": got[0], "text": got[1]}


# --- one agent's transcript -------------------------------------------------


def agent_view(agent: str) -> dict:
    """One agent's episodes, each in the round it acted in."""
    account = read_json(harness.records_dir(agent) / "account.json") or {}
    c = experiment_of(agent)
    rows = cohort_sessions(c) if c else []
    rnd = {r["episode"]: r["round"] for r in rows if r["agent"] == agent}
    ts = traces_of(agent)
    live = live_index(agent)
    episodes = [{
        "episode": t["episode"], "round": rnd.get(t["episode"]),
        "stop": t["stop"], "spent": t["spent"], "turns": len(t["turns"]),
        "remaining": t["remaining"], "duration_s": t.get("duration_s"),
        "refused": len(analyze.refused_turns_of(t)),
        "fallback": len(analyze.fallback_turns_of(t)),
        "transfer": t.get("transfer") or {}, "channels": t.get("channels") or {},
        "forgiven": t.get("forgiven") or 0,
        "provenance": t.get("provenance") or {},
        "drift": t.get("provenance_drift") or [],
        "live": False,
    } for t in ts]
    if live is not None:
        turns = live_turns(agent, live, account)
        episodes.append({
            "episode": live, "round": max(rnd.values(), default=0) + 1,
            "stop": None,
            "spent": account.get("remaining", 0) - turns[-1]["balance"] if turns else 0,
            "turns": len(turns),
            "remaining": turns[-1]["balance"] if turns else account.get("remaining"),
            "duration_s": None,
            "refused": len([t for t in turns if t["stop_details"]]),
            "fallback": len([t for t in turns if t["served_by_fallback"]]),
            "transfer": {}, "channels": {}, "forgiven": 0,
            "provenance": {}, "drift": [], "live": True,
        })
    seat, seen = harness.seating(agent, account)
    return {
        "agent": agent, "experiment": c["name"] if c else "",
        "model": account.get("model"),
        "initial": account.get("initial"), "remaining": account.get("remaining"),
        "episodes": episodes,
        "live": live, "live_age": live_age(agent, live) if live is not None else None,
        # The mapping has no gap and holds every seat, this agent's among them, so
        # which one is its own has to be said rather than inferred from absence.
        "seat": seat, "peers": seen,
        "starter_files": account.get("starter_files_landed") or {},
    }


def session_view(agent: str, index: int, since: int = 0) -> dict | None:
    """One episode's transcript, from the trace or the raw log; None for neither.

    `since` is the last turn the page holds, so an episode in flight appends.
    `source` changing from raw to trace tells the page to ask again from zero.
    """
    if not trace_path(agent, index).exists() and not raw_path(agent, index).exists():
        return None
    trace = load_trace(trace_path(agent, index))
    if trace is not None:
        turns = from_trace(trace)
        table = harness.channels_from(analyze.table_of(trace))
        mail, mail_rec = harness.mailbox_channel(table), analyze.mailbox_of(trace)
        out = {
            "source": "trace", "live": False, "age": None, "episode": index,
            "stop": trace["stop"], "spent": trace["spent"], "remaining": trace["remaining"],
            "duration_s": trace.get("duration_s"), "error": trace.get("error"),
            "series_before": trace.get("series_before") or [],
            "series_after": trace.get("series_after") or [],
            "missing_tools": trace.get("missing_tools") or [],
            "balance_fits": trace.get("balance_fits"), "read_balance": trace.get("read_balance"),
            "transfer": trace.get("transfer") or {}, "channels": trace.get("channels") or {},
            "board": analyze.board_of(trace), "mailbox": mail_rec,
            "forgiven": trace.get("forgiven") or 0,
            "obligations": obligations(trace),
            "messages_why": harness.outbox_why(mail_rec, mail) if mail_rec else "",
        }
        if since == 0:
            listing, sections = observation_split(trace.get("observation") or "")
            out["observation"] = {
                "command": trace["commands"][0] if trace.get("commands") else harness.observation(),
                "result": trace.get("observation") or "",
                # The listing and the record it opened on, apart. The two
                # concatenate back to result, which is what reached the model.
                "name": analyze.harness_files_of(trace).get("digest", ""),
                "inbox": mail.inbox if mail else None,
                "listing": listing, "shown_before": sections,
                "clipped": observation_clipped(trace),
            }
            out["changes"] = session_changes(agent, index)
    else:
        account = read_json(harness.records_dir(agent) / "account.json") or {}
        turns = live_turns(agent, index, account)
        out = {
            "source": "raw", "live": True, "age": live_age(agent, index), "episode": index,
            "stop": None, "spent": (account.get("remaining", 0) - turns[-1]["balance"]) if turns else 0,
            "remaining": turns[-1]["balance"] if turns else account.get("remaining"),
            "duration_s": None, "error": None,
            "series_before": account.get("series") or [], "series_after": [],
            "missing_tools": [], "balance_fits": None, "read_balance": None,
            "transfer": {}, "channels": {}, "board": {}, "mailbox": {}, "forgiven": 0,
            "obligations": obligations({}), "messages_why": "",
        }
        if since == 0:
            # The agent's environment at episode start is recorded in the trace and nowhere
            # else, so while the episode runs it is pending like any other
            # command's output.
            mail = harness.mailbox_channel(harness.channels())
            out["observation"] = {"command": harness.observation(), "result": None,
                              "name": harness.HARNESS_FILES["digest"],
                              "inbox": mail.inbox if mail else None, "listing": None,
                              "shown_before": [], "clipped": False}
    out["turns"] = [t for t in turns if (t["turn"] or 0) > since]
    out["total_turns"] = len(turns)
    return out


def session_changes(agent: str, index: int) -> list[dict]:
    """An episode's diffs against the episode before it, one block per channel the
    agent writes, in table order.

    Each block says who can see it: what the agent kept to itself, what it put
    where every other agent reads it, what it addressed to one of them, and what
    it declared to the harness.
    """
    def files(t: dict | None, name: str) -> dict[str, str]:
        return {f["path"]: f["text"] for f in (t or {}).get("files") or []
                if analyze.channel_of(f) == name and analyze.role_of(f) == "own"
                and f.get("text") is not None}

    this = load_trace(trace_path(agent, index))
    if this is None:
        return []
    before = load_trace(trace_path(agent, index - 1))
    label = analyze.label_of(this)
    out = []
    for ch in harness.channels_from(analyze.table_of(this)):
        if ch.writer != "self":
            continue
        if ch.shape == "mailbox":
            what = f"{ch.outbox}/ \u00b7 one file each, one agent reads it"
        elif ch.shape == "file":
            what = f"{ch.path} \u00b7 the harness parses it"
        elif ch.readers == "self":
            what = f"{ch.path}/ \u00b7 nobody else reads this"
        else:
            what = f"{ch.path_for(label)}/ \u00b7 every agent reads this"
        out.append({"channel": ch.name, "what": what,
                    "lines": analyze.state_changes(files(before, ch.name), files(this, ch.name))})
    return out


# --- the page ---------------------------------------------------------------


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClaudeSandbox</title>
<style>
/* Dark only, and low contrast on purpose: this is a page left open beside an agent
   for hours. Fira Code is the numbers and everything the agent wrote - a balance
   has to line up column-wise against the one above it - and Fira Sans is the
   chrome around them. Both are named first and degrade to whatever the machine
   has; nothing is fetched, because a page that fetched a font would be a page
   that needs a network. */
:root {
  --bg:#0e0f10; --sunk:#0a0b0b; --panel:#141618; --raise:#191c1e;
  --line:#232628; --line2:#2d3134;
  --ink:#d6d3cd; --dim:#8a8e91; --faint:#5b6063;
  --accent:#7f9bb0; --live:#8fa87d; --warn:#c2a06b; --bad:#bd8078; --starter:#9c8bab;
  --sans:"Fira Sans","Fira Sans Condensed",Inter,"Segoe UI Variable Text","Segoe UI",
         system-ui,sans-serif;
  --mono:"Fira Code","Cascadia Mono",Consolas,ui-monospace,monospace;
  --r:10px;
}
* { box-sizing:border-box; }
html { color-scheme:dark; height:100%; }
body { margin:0; height:100%; background:var(--bg); color:var(--ink);
       font:400 14px/1.5 var(--sans); -webkit-font-smoothing:antialiased; }
::selection { background:rgba(127,155,176,.28); }
::-webkit-scrollbar { width:11px; height:11px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--line2); border-radius:7px;
                            border:3px solid transparent; background-clip:content-box; }
::-webkit-scrollbar-thumb:hover { background:var(--faint); background-clip:content-box; }

h2 { margin:16px 0 8px; font:600 11.5px/1 var(--sans); letter-spacing:.16em;
     text-transform:uppercase; color:var(--faint); }
.num { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.note { color:var(--faint); font-size:12.5px; margin:0; }
.empty { color:var(--faint); padding:28px 0; text-align:center; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
         padding:12px 14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(205px,1fr)); gap:16px 22px; }
.kv .k { color:var(--faint); font:500 11.5px/1 var(--sans); letter-spacing:.09em;
         text-transform:uppercase; }
.kv .v { font:400 13px/1.5 var(--mono); color:var(--dim); overflow-wrap:anywhere; margin-top:5px; }
.dot { width:6px; height:6px; border-radius:50%; display:inline-block; background:currentColor;
       animation:pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 50% { opacity:.2; } }
.tag { font:500 11.5px/1.5 var(--sans); letter-spacing:.03em; padding:2px 9px; border-radius:999px;
       border:1px solid var(--line2); color:var(--dim); display:inline-flex; gap:5px;
       align-items:center; white-space:nowrap; }
.tag.live { color:var(--live); border-color:rgba(143,168,125,.38); background:rgba(143,168,125,.08); }
.tag.bad  { color:var(--bad);  border-color:rgba(189,128,120,.38); background:rgba(189,128,120,.08); }
.tag.warn { color:var(--warn); border-color:rgba(194,160,107,.38); background:rgba(194,160,107,.08); }
.tag.starter { color:var(--starter); border-color:rgba(156,139,171,.38); background:rgba(156,139,171,.08); }
/* The channels the agent writes for someone else to read. Its peers, its inboxes,
   the balances and the ledger keep the plain tag: they are the environment, not this
   agent's doing. */
.tag.edit { color:var(--accent); border-color:rgba(127,155,176,.38); background:rgba(127,155,176,.08); }
.bar { height:3px; background:var(--line); border-radius:2px; margin:7px 0 6px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--accent); transition:width .3s ease; }
.bar.over i { background:var(--bad); }

/* --- the header, which every tab is read under --- */
/* The page is the window: a column of a fixed header over one channel that takes
   what is left. Every pane sizes itself against that channel rather than against
   the viewport, so the only scrollbar on screen belongs to whatever is being
   read. */
#page { height:100dvh; display:flex; flex-direction:column; padding:0 20px; }
/* Unfolded, the header is as tall as the round it is describing: a long ledger
   or a seat carrying every penalty at once would otherwise leave the transcript
   a few lines. It is capped at half the window and scrolls inside that, so what
   is being read always has the other half. */
#top { flex:0 0 auto; background:var(--bg); border-bottom:1px solid var(--line);
       padding:12px 0 0; max-height:48dvh; overflow-y:auto; overflow-x:hidden; }
#top.compact { max-height:none; overflow:visible; }
.who { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
.who b { font:600 15px/1 var(--mono); letter-spacing:.01em; }
.who .sub { color:var(--faint); font-size:12.5px; }
.picks { display:flex; gap:6px; flex-wrap:wrap; }
.filt { padding:4px 10px; border:1px solid var(--line2); border-radius:999px; cursor:pointer;
        background:transparent; color:var(--dim); font:500 12px/1.4 var(--sans);
        letter-spacing:.02em; transition:color .12s ease, border-color .12s ease,
        background .12s ease; }
.filt:hover { color:var(--ink); border-color:var(--faint); }
.filt.on { color:var(--ink); border-color:var(--accent); background:rgba(127,155,176,.14); }
.filt em { font-style:normal; color:var(--faint); margin-left:5px; }

.fold { margin-left:auto; padding:3px 10px; border:1px solid var(--line2); border-radius:999px;
        cursor:pointer; background:transparent; color:var(--faint);
        font:500 11.5px/1.5 var(--sans); letter-spacing:.02em; }
.fold:hover { color:var(--ink); border-color:var(--faint); }

.board { display:grid; grid-template-columns:1fr 230px; gap:16px; align-items:start; }
.seats { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1px;
         background:var(--line); border:1px solid var(--line); border-radius:var(--r);
         overflow:hidden; }
.tile { background:var(--panel); padding:8px 11px 9px; }
/* Named for the seat rather than for the round, because `out` is already the
   pre a command's output is read in and a tile is not one. */
.tile.idle { background:var(--sunk); }
.tile .k { display:flex; justify-content:space-between; align-items:baseline; gap:8px;
           color:var(--faint); font:500 11px/1 var(--sans); letter-spacing:.11em;
           text-transform:uppercase; }
/* An agent id is a name rather than a label, so it keeps the letters it was given. */
.tile .k b { color:var(--dim); font:600 11px/1 var(--mono); letter-spacing:.02em;
             text-transform:none; }
.tile .v { margin-top:5px; font:400 19px/1 var(--mono); font-variant-numeric:tabular-nums; }
.tile .v.neg { color:var(--bad); }
/* One wrapping row, so a seat carrying a transfer and a halt is no taller than a
   seat carrying neither. */
.tile .m { color:var(--faint); font-size:12px; display:flex; gap:5px 9px; flex-wrap:wrap;
           align-items:center; margin-top:5px; }
.tile .m i { font-style:normal; color:var(--line2); }

/* Collapsed, the header is the balances and the tabs: the rows under each
   number are the round's detail, and a reader watching a transcript is not
   reading them. */
#top.compact .m, #top.compact #gwrap, #top.compact .bar { display:none; }
#top.compact .board { grid-template-columns:1fr; }
#top.compact .tile { padding:7px 11px 8px; }
#top.compact .tile .v { margin-top:4px; font-size:15px; }
#top.compact .who { margin-bottom:8px; }

.gbox { border:1px solid var(--line); border-radius:var(--r); background:var(--panel);
        padding:10px 12px; }
/* An experiment that has been trading for a while has a ledger longer than the
   seats beside it; it scrolls rather than setting how tall the header is. */
.gbox .led { max-height:150px; overflow-y:auto; margin-top:6px; }
.gbox td { padding:3px 12px 3px 0; border:0; }
#tabs { display:flex; gap:2px; margin-top:10px; }
.tab { padding:8px 15px 9px; border:0; border-bottom:2px solid transparent; cursor:pointer;
       background:transparent; color:var(--faint); font:500 13.5px/1 var(--sans);
       letter-spacing:.02em; }
.tab:hover { color:var(--ink); }
.tab.on { color:var(--ink); border-bottom-color:var(--accent); }
.tab em { font-style:normal; color:var(--faint); margin-left:7px; font-size:12px; }
/* What is left of the window after the header. Nothing inside it may grow the
   page: a pane that wants more room scrolls itself. */
#body { flex:1 1 auto; min-height:0; overflow:hidden; padding-top:8px;
        display:flex; flex-direction:column; }
#body > * { min-height:0; }
/* A tab that fills the channel: its own column, with one child taking the slack. */
.fill { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }

/* --- the message log, read as the chat it is --- */
.round { display:flex; align-items:center; gap:12px; margin:14px 0 8px; color:var(--faint);
         font:500 11.5px/1 var(--sans); letter-spacing:.16em; text-transform:uppercase; }
.round i { flex:1; height:1px; background:var(--line); }
.chat { flex:1 1 auto; min-height:0; display:grid; grid-template-columns:262px minmax(0,1fr);
        gap:12px; align-items:stretch; }
.chat.narrow { grid-template-columns:44px minmax(0,1fr); }
.chat.narrow .pair .p, .chat.narrow .pair .t em { display:none; }
.chat.narrow .pair { padding:9px 6px; text-align:center; }
.chat.narrow .pair .t { justify-content:center; }
.pairs { border:1px solid var(--line); border-radius:var(--r); background:var(--panel);
         min-height:0; overflow-y:auto; }
.pair { display:block; width:100%; text-align:left; padding:9px 13px; cursor:pointer;
        background:transparent; border:0; border-bottom:1px solid var(--line);
        font:400 14px/1.5 var(--sans); transition:background .12s ease; }
.pair:last-child { border-bottom:0; }
.pair:hover { background:var(--raise); }
.pair.on { background:var(--raise); box-shadow:inset 2px 0 0 var(--accent); }
.pair.mute { opacity:.5; }
.pair .t { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
.pair .t .nm { font:400 13.5px/1.3 var(--mono); color:var(--ink); }
.pair .t em { font-style:normal; font:400 11.5px/1.3 var(--sans); color:var(--faint); }
.pair .p { margin-top:4px; font-size:12.5px; color:var(--faint);
           overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

.thread { border:1px solid var(--line); border-radius:var(--r); background:var(--panel);
          min-width:0; min-height:0; display:flex; flex-direction:column; }
.thead { padding:9px 14px; border-bottom:1px solid var(--line);
         display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
.thead .nm { font:400 14px/1.4 var(--mono); }
#thread { flex:1 1 auto; min-height:0; overflow-y:auto; overflow-anchor:none;
          padding:2px 14px 12px; }
#thread .round:first-child { margin-top:12px; }
/* A log, not a chat: a message takes the width it is given, and which way it
   went is said in its header rather than by which side it sits on. */
.msg { margin-top:8px; }
.msg .mh { display:flex; gap:9px; align-items:baseline; flex-wrap:wrap; margin-bottom:3px;
           color:var(--faint); font-size:12px; }
.msg .mh b { font:400 13px/1.4 var(--mono); font-weight:400; }
.msg .bub { border:1px solid var(--line2); border-radius:8px;
            background:var(--raise); padding:7px 11px; }
.msg.r .bub { border-left:2px solid var(--line2); }
.msg.tip .bub { border-style:dashed; background:var(--sunk); }
.msg.same .bub { opacity:.62; }
.msg .bub pre { margin:0; white-space:pre-wrap; color:var(--ink);
                font:400 13px/1.5 var(--mono); max-height:26em; overflow:auto; }
.sec { border-top:1px solid var(--line); }
.sec summary { cursor:pointer; padding:5px 0; color:var(--faint); font-size:12px;
  font-family:var(--mono); }
.sec summary i { color:var(--ink); font-style:normal; }
.sec[open] summary { color:var(--ink); }
.sec pre.out { margin-bottom:7px; }
.msg .bub details { margin-top:7px; }
.msg .bub summary { cursor:pointer; color:var(--faint); font-size:12px; }
.msg .bub details pre { margin-top:6px; padding:7px 10px; background:var(--sunk);
                        border-radius:8px; color:var(--dim); }
.msg .ft { margin-top:3px; color:var(--faint); font-size:12px; }
.sys { margin-top:10px; text-align:center; color:var(--faint); font-size:12.5px; }
.sys b { font:400 12.5px/1.5 var(--mono); font-weight:400; }

/* --- the blackboards and the private stores, side by side --- */
/* The listing is an index and stays a column a seat; what a file says is read
   below it, across the whole window, in as many columns as there are files
   open. A file's contents are the reason the tab exists, so they get the room. */
.trees { flex:1 1 auto; min-height:0; overflow-y:auto; padding-bottom:8px; }
/* Both grids wrap rather than run off the side: a listing too narrow to read is
   no better than one that is not on screen. */
.cols { display:grid; gap:12px; align-items:start;
        grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr)); }
.col { border:1px solid var(--line); border-radius:var(--r); background:var(--panel);
       min-width:0; }
.col .h { padding:9px 13px; border-bottom:1px solid var(--line);
          display:flex; gap:9px; align-items:baseline; flex-wrap:wrap; }
.col .h b { font:600 14px/1 var(--mono); }
.col .h span { color:var(--faint); font-size:12.5px; }
.col .b { padding:2px 13px 8px; }
.bodies { display:grid; gap:12px; margin-top:12px; align-items:start;
          grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); }
.bodies .col .b { padding:10px 13px 12px; }

table { border-collapse:collapse; width:100%; font:400 13px/1 var(--mono); }
td, th { text-align:left; padding:6px 12px 6px 0; border-bottom:1px solid var(--line); }
tr:last-child td { border-bottom:0; }
th { color:var(--faint); font:500 11.5px/1 var(--sans); letter-spacing:.11em; text-transform:uppercase; }
tr.file { cursor:pointer; transition:background .12s ease; }
tr.file:hover td { background:var(--raise); }
tr.file.on td { background:var(--raise); }
td.sz { color:var(--faint); font-variant-numeric:tabular-nums; text-align:right;
        width:1%; white-space:nowrap; }
td.md, td.kd { color:var(--faint); width:1%; white-space:nowrap; }

/* --- the transcript --- */
.strip { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.strip .lbl { color:var(--faint); font:600 11.5px/1 var(--sans); letter-spacing:.16em;
              text-transform:uppercase; width:56px; flex:0 0 auto; }
.chip { padding:5px 10px; border:1px solid var(--line2); border-radius:7px; cursor:pointer;
        background:var(--panel); color:var(--dim); font:400 13px/1 var(--mono);
        transition:color .12s ease, border-color .12s ease, background .12s ease; }
.chip:hover { color:var(--ink); border-color:var(--faint); }
.chip.on { color:var(--ink); border-color:var(--accent); background:rgba(127,155,176,.12); }
.chip.live { color:var(--live); border-color:rgba(143,168,125,.45); }
.chip.halt { color:var(--warn); border-color:rgba(194,160,107,.45); }
.chip.off { opacity:.4; cursor:default; }
.txbar { display:flex; justify-content:space-between; align-items:center; gap:16px;
         margin-bottom:8px; }
#tx { flex:1 1 auto; min-height:0; background:var(--panel); border:1px solid var(--line);
      border-radius:var(--r); overflow-y:auto; overflow-anchor:none; }
.turn { padding:11px 15px; border-top:1px solid var(--line); }
.turn:first-child { border-top:0; }
.th { color:var(--faint); font-size:12.5px; display:flex; gap:10px; flex-wrap:wrap;
      align-items:center; margin-bottom:7px; }
.th .num { font-size:12.5px; color:var(--dim); }
/* Prose is read a line at a time, so it keeps a measure however wide the window
   is. Everything the agent ran, wrote, or was shown takes the whole of it. */
.say { white-space:pre-wrap; margin:0; color:var(--ink); max-width:110ch; }
.think { white-space:pre-wrap; margin:0 0 9px; color:var(--faint); font-style:italic;
         font-size:13.5px; border-left:1px solid var(--line2); padding-left:11px;
         max-width:110ch; }
.cmd { margin:9px 0 0; color:var(--accent); white-space:pre-wrap; font:400 13px/1.55 var(--mono); }
.cmd::before { content:"$ "; color:var(--faint); }
.out { margin:5px 0 0; padding:8px 11px; white-space:pre-wrap; color:var(--dim);
       font:400 13px/1.5 var(--mono); background:var(--sunk); border-radius:8px;
       max-height:26em; overflow:auto; }
.pend { margin:5px 0 0; padding:8px 11px; color:var(--warn); font-size:13.5px;
        background:rgba(194,160,107,.07); border-radius:8px; }
.diff .a { color:var(--live); } .diff .d { color:var(--bad); } .diff .h { color:var(--faint); }
.tail { color:var(--faint); font-size:12.5px; cursor:pointer; user-select:none;
        display:inline-flex; gap:7px; align-items:center; }
.tail input { accent-color:var(--accent); margin:0; }
/* Sits under the transcript when it is following, so the button is a way back
   rather than a fight with the scroll position. */
.jump { position:absolute; right:20px; bottom:14px; padding:5px 12px; border-radius:999px;
        border:1px solid var(--line2); background:var(--raise); color:var(--dim);
        font:500 12px/1.5 var(--sans); cursor:pointer; }
.jump:hover { color:var(--ink); border-color:var(--faint); }
.txwrap { position:relative; flex:1 1 auto; min-height:0; display:flex;
          flex-direction:column; }
</style>
<div id="page">
  <header id="top">
    <div class="who"><b>ClaudeSandbox</b><span class="sub" id="whosub"></span>
      <div class="picks" id="picks"></div>
      <button class="fold" id="fold"></button></div>
    <div class="blackboard"><div class="seats" id="seats"></div>
      <div id="gwrap"></div></div>
    <nav id="tabs"></nav>
  </header>
  <main id="body"><div class="empty">loading&hellip;</div></main>
</div>
<script>
const $ = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d; };
const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const num = (n) => n == null ? "\\u2013" : Number(n).toLocaleString();
const get = (u) => fetch(u).then(r => r.ok ? r.json() : Promise.reject(r.status));
// One provenance answer as a line. Rates and fallbacks are lists and an experiment's
// seating is a mapping, and both read as what they are rather than as JSON.
const provValue = (v) => Array.isArray(v) ? v.join(", ")
  : v && typeof v === "object"
    ? (Object.entries(v).map(([k, r]) => `${k} ${r}`).join(" \\u00b7 ") || "none")
  : String(v);

// What an obligation cost, which is a separate question from whether it was met.
const charged = (n) => n ? ` \\u00b7 penalised ${num(n)}`
  : ` \\u00b7 <span title="a share is taken only from an episode the API answered,
     past the grace, at a rate above zero">charged nothing</span>`;

// What the page is folded down to is the reader's, not the agent's, so it is kept
// where a reload can find it again.
const held = (k, dflt) => { try { const v = localStorage.getItem(k);
  return v == null ? dflt : v === "1"; } catch (e) { return dflt; } };
const hold = (k, v) => { try { localStorage.setItem(k, v ? "1" : "0"); } catch (e) {} };

// --- where the reader left each box --------------------------------------

// How far a box is from its end, in pixels.
function behind(b) { return b.scrollHeight - b.scrollTop - b.clientHeight; }

// At the end means there is something to scroll and the reader is against the
// bottom of it. A box with nothing to scroll is not at its end, so filling one
// carries the reader nowhere.
const NEAR = 40;
const atEnd = (b) => b.scrollHeight > b.clientHeight && behind(b) < NEAR;

// Every box the page can be scrolled in. A pane is drawn by replacing its
// nodes, and the new nodes carry none of the scroll the old ones had.
const BOXES = "#tx, #thread, .pairs, .trees, .led, .out, .bub pre";
const boxesIn = (el) =>
  (el.matches(BOXES) ? [el] : []).concat([...el.querySelectorAll(BOXES)]);

// Everything #body holds is drawn against a signature kept on #body itself, so
// whatever replaces the pane drops the signature with it. A renderer comparing
// against a signature whose markup is gone takes the pane for already drawn.
function blank(el, html) { el.dataset.sig = ""; el.innerHTML = html; return el; }

// Write `html` into `el`, leaving every box inside it where its reader had it:
// one against the end stays against the end as content arrives, and every other
// one keeps the offset it was read at. Boxes are matched across the write by
// the order they are written in, so a pane that grows at its end hands each box
// back its own place.
function redraw(el, html) {
  const was = boxesIn(el).map(b => [b.scrollTop, b.scrollLeft, atEnd(b)]);
  el.innerHTML = html;
  boxesIn(el).forEach((b, i) => {
    const had = was[i];
    if (!had) return;
    b.scrollTop = had[2] ? b.scrollHeight : had[0];
    b.scrollLeft = had[1];
  });
}

const S = { experiments: [], experiment: null, head: null, tab: "mailbox",
            // The message log is append-only, so what is held is extended
            // rather than re-fetched; the tip is whatever is ahead of it.
            msgs: [], tips: [], msgn: 0, standing: false, pair: null,
            tree: { blackboard: null, notes: null }, open: {}, body: {},
            // One agent's transcript, and how many of its turns are drawn.
            agent: null, detail: null, episode: null, view: null,
            since: 0, source: null, drawn: 0, tail: true,
            // Where the transcript is being read, held here rather than in the
            // pane, because the pane is replaced under it.
            txtop: 0, txend: false,
            // How much of the header, and of the thread list, is shown.
            compact: held("compact", true), rail: held("rail", false),
            poll: 1500, stale: 180 };

const PENDING = "output arrives when the episode ends";

const ago = (s) => s == null ? "" : s < 90 ? `${Math.round(s)}s ago`
  : s < 5400 ? `${Math.round(s / 60)}m ago` : `${(s / 3600).toFixed(1)}h ago`;
const running = (d) => d && d.live != null && d.live_age != null && d.live_age < S.stale;
const seatName = (n) => {
  const s = (S.head ? S.head.seats : []).find(x => String(x.seat) === String(n) || x.label === n);
  return s ? s.agent : null;
};
// What the other agents call a seat: its balance file, which is the balance
// file's name and then the seat's label.
const nameOf = (seat) => {
  const h = S.head || {};
  return (h.balance == null ? "n" : h.balance) + ((h.labels || {})[seat] || seat);
};

// --- little svg ----------------------------------------------------------

// Folded rather than spread: a long agent's series agents to thousands of elements,
// and Math.max(...s) on one of those is an argument list, not a loop. The floor
// is held at zero so a balance that never moves still has a scale, and one that
// went negative is drawn below the line rather than filling the box.
const hiOf = (s) => s.reduce((a, b) => b > a ? b : a, -Infinity);
const loOf = (s) => s.reduce((a, b) => b < a ? b : a, 0);

// `lo`, `hi` and `n` are given when several series share one picture: without
// them each would be drawn against its own scale and its own length, and two
// seats that spent differently would look alike.
function path(series, w, h, pad, lo, hi, n) {
  if (series.length < 2) return "";
  if (lo == null) { lo = loOf(series); hi = hiOf(series); }
  const span = (hi - lo) || 1, den = ((n || series.length) - 1) || 1;
  return series.map((v, i) => {
    const x = pad + i * (w - 2 * pad) / den;
    const y = pad + (h - 2 * pad) * (1 - (v - lo) / span);
    return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
  }).join(" ");
}

const INK = ["var(--accent)", "var(--live)", "var(--warn)", "var(--starter)", "var(--bad)"];
const inkOf = (i) => INK[i % INK.length];

// Every seat on one scale, which is the comparison a chart of one agent cannot
// make: who is ahead, and where the lines crossed.
function overlay(seats) {
  const all = seats.flatMap(s => s.series || []);
  if (all.length < 2) return `<div class="note">no billed turns yet</div>`;
  const lo = loOf(all), hi = hiOf(all);
  const n = seats.reduce((a, s) => Math.max(a, (s.series || []).length), 0);
  const lines = seats.map((s, i) => {
    const d = path(s.series || [], 240, 54, 3, lo, hi, n);
    return d ? `<path d="${d}" fill="none" stroke="${inkOf(i)}" stroke-width="1.25"
      opacity=".85" vector-effect="non-scaling-stroke"/>` : "";
  }).join("");
  const zero = lo < 0 ? `<line x1="0" x2="240" y1="${(3 + 48 * (1 - (0 - lo) / ((hi - lo) || 1))).toFixed(1)}"
     y2="${(3 + 48 * (1 - (0 - lo) / ((hi - lo) || 1))).toFixed(1)}" stroke="var(--bad)"
     stroke-width="1" opacity=".5"/>` : "";
  return `<svg width="100%" height="54" viewBox="0 0 240 54" preserveAspectRatio="none"
    >${zero}${lines}</svg>`;
}

// --- the header ----------------------------------------------------------

function renderPicks() {
  const el = document.getElementById("picks");
  // One set is not a choice between sets.
  if (S.experiments.length < 2) { el.innerHTML = ""; return; }
  const why = (c) => c.seated ? `${c.members.length} seated` : "no seating: no group or mailbox messages";
  el.innerHTML = S.experiments.map(c => `<button class="filt ${c.name === S.experiment ? "on" : ""}"
      data-c="${esc(c.name)}" title="${esc(why(c))}">${esc(c.name)}<em>${c.members.length}</em></button>`).join("");
  el.querySelectorAll(".filt").forEach(b => b.onclick = () => pickCohort(b.dataset.c));
}

function pickCohort(name) {
  if (name === S.experiment) return;
  S.experiment = name;
  S.msgs = []; S.tips = []; S.msgn = 0; S.pair = null;
  S.tree = {}; S.open = {}; S.body = {};
  S.agent = null; S.detail = null; S.episode = null; S.view = null; S.drawn = 0;
  blank(document.getElementById("body"), `<div class="empty">loading&hellip;</div>`);
  renderPicks();
  refresh();
}

function tileHtml(s, i) {
  const n = s.live_balance != null ? s.live_balance : s.n;
  const left = s.initial ? Math.max(0, Math.min(1, n / s.initial)) : 0;
  const g = s.transfer;
  const taken = Object.entries(s.penalised || {}).filter(([, v]) => v);
  // A declaration that moved nothing goes on moving nothing every episode it is
  // left in place, and the only statement of why is here.
  const transfer = g && g.standing && !g.amount
    ? `<span class="tag warn" title="${esc(g.error || "")}">transfer declared, moved nothing</span>`
    : g && g.amount ? `<span class="tag">transfer ${num(g.amount)} \u2192 ${esc(g.label || g.seat)}</span>` : "";
  const state = running(s)
      ? `<span class="tag live"><i class="dot"></i>s${s.live} \\u00b7 ${s.live_turns}t \\u00b7 derived</span>`
    : s.live != null
      ? `<span class="tag warn" title="no trace was ever written for it">s${s.live} unfinished \\u00b7 ${ago(s.live_age)}</span>`
    : s.pending
      ? `<span class="tag" title="the round is still going and it has not acted in it; the order rotates, so a seat it has not reached yet looks no different from one it has">not yet this round</span>`
    : !s.acted
      ? `<span class="tag" title="nothing left to spend, or it refused its last episodes running; either way it is off the table">out of the agent</span>`
      : `<span>s${s.committed} committed</span>`;
  return `<div class="tile ${s.acted || s.pending ? "" : "idle"}">
    <div class="k"><span style="color:${inkOf(i)}">${s.seat == null ? "agent" : "n" + esc(s.seat)}</span>
      <b>${esc(s.agent)}</b></div>
    <div class="v ${n < 0 ? "neg" : ""}">${num(n)}</div>
    <div class="bar ${n < 0 ? "over" : ""}"><i style="width:${(left * 100).toFixed(1)}%"></i></div>
    <div class="m">${state}<i>|</i>
      <span title="spent this round">round ${num(s.spent_this_round)}</span>
      <span title="spent in all, summed from its episodes; the bar above is how
        far the balance itself has fallen, which transfers and shares also move"
        >of ${num(s.spent)}</span>
      ${s.posted === false ? `<span class="tag bad" title="its blackboard held nothing new">did not post</span>` : ""}
      ${s.messaged === false ? `<span class="tag bad" title="its outbox did not say one new thing to one agent">said nothing new</span>` : ""}
      ${(s.sent || s.received || s.rebated || taken.length || s.forgiven || transfer) ? `<i>|</i>
        ${s.sent ? `<span title="given away">\\u2192 ${num(s.sent)}</span>` : ""}
        ${s.received ? `<span title="given to it">\\u2190 ${num(s.received)}</span>` : ""}
        ${s.rebated ? `<span title="won back for what it gave">\\u21ba ${num(s.rebated)}</span>` : ""}
        ${taken.map(([name, n]) => `<span title="taken for silence on ${esc(name)}">\\u2212 ${num(n)} ${esc(name)}</span>`).join(" ")}
        ${s.forgiven ? `<span title="floored back to zero; its environment never says so"
          >floored ${num(s.forgiven)}</span>` : ""}${transfer}` : ""}
      ${(s.halted || s.drift.length) ? `<i>|</i>
        ${s.halted ? `<span class="tag bad">${esc(s.stop)}</span>` : ""}
        ${s.drift.length ? `<span class="tag warn" title="${esc(s.drift.join("; "))}">drift</span>` : ""}` : ""}</div>
  </div>`;
}

function renderFold() {
  document.getElementById("top").classList.toggle("compact", S.compact);
  document.getElementById("fold").textContent =
    S.compact ? "\\u25be the round" : "\\u25b4 the round";
}

function renderHeader() {
  const h = S.head;
  if (!h) return;
  renderFold();
  document.getElementById("whosub").innerHTML =
    `${h.seats.length} ${h.seated ? "seats" : "agents"} \\u00b7 round ${h.round}` +
    (h.model ? ` \\u00b7 ${esc(h.model)}` : "") +
    (h.starter_files ? ` \\u00b7 <span class="tag starter" style="vertical-align:middle">${esc(h.starter_files)}</span>` : "") +
    (h.seated ? "" : ` \\u00b7 no seating: these agents have no group or mailbox messages`);
  document.getElementById("seats").innerHTML = h.seats.map(tileHtml).join("");
  // g reads the same three bare numbers for everyone, the agent it was aimed
  // against included. The gloss is the page's, not the environment's.
  const rows = h.ledger.map(([giver, taker, amount]) => `<tr>
      <td class="num">${esc(giver)} ${esc(taker)} ${num(amount)}</td>
      <td class="note">${esc(seatName(giver) || "")} \\u2192 ${esc(seatName(taker) || "")}</td></tr>`).join("");
  redraw(document.getElementById("gwrap"), !h.seated ? "" : `<div class="gbox">
    <div class="kv"><div class="k">g \\u00b7 the transfer ledger</div></div>
    ${rows ? `<div class="led"><table>${rows}</table></div>`
           : `<div class="note" style="margin-top:8px">no transfer has moved</div>`}
    <div style="margin-top:10px">${overlay(h.seats)}</div></div>`);
  const tabs = h.tabs || [];
  if (!tabs.some(t => t.key === S.tab)) S.tab = (tabs[0] || {}).key;
  document.getElementById("tabs").innerHTML = tabs.map(({key, label}) => {
    const off = key === "mailbox" && (!h.posts || h.seats.length < 2);
    return `<button class="tab ${S.tab === key ? "on" : ""}" data-t="${key}">${label}${
      key === "mailbox" && S.msgn ? `<em>${S.msgn}</em>` : ""}${off ? `<em>\\u2013</em>` : ""}</button>`;
  }).join("");
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
    if (b.dataset.t === S.tab) return;
    S.tab = b.dataset.t;
    renderHeader();
    blank(document.getElementById("body"), `<div class="empty">loading&hellip;</div>`);
    renderTab();
  });
}

// --- the message log -----------------------------------------------------

// A thread is the pair of seats a channel agents between: out/1 written by seat 0
// and out/0 written by seat 1 are the two directions of one conversation, so
// events are keyed on the pair rather than on the sender.
const pairKey = (a, b) => [a, b].sort((x, y) => Number(x) - Number(y)).join("\\u00b7");

const seatInk = (seat) => {
  const i = (S.head ? S.head.seats : []).findIndex(s => String(s.seat) === String(seat));
  return i < 0 ? "var(--ink)" : inkOf(i);
};

const seatTag = (seat) => `<b style="color:${seatInk(seat)}">${esc(nameOf(seat))}</b>`;

// Every pair of the experiment, whether or not anything has passed between them: two
// agents that have never addressed each other are a fact about the round, and a
// thread that only appeared once it had traffic could not be looked for.
function threadsOf() {
  const seats = (S.head ? S.head.seats : []).filter(s => s.seat != null);
  const by = new Map();
  for (let i = 0; i < seats.length; i++) {
    for (let j = i + 1; j < seats.length; j++) {
      by.set(pairKey(seats[i].seat, seats[j].seat),
             { key: pairKey(seats[i].seat, seats[j].seat), a: seats[i], b: seats[j],
               events: [], tips: [] });
    }
  }
  // A path naming what is not a seat of this experiment reaches nobody, and so does
  // a declaration that resolved to no one. Neither sits between two agents, and
  // both are still something a sender wrote.
  const loose = { key: "\\u2205", a: null, b: null, events: [], tips: [] };
  const bin = (e) => (e.to_run == null ? null : by.get(pairKey(e.from_seat, e.to_seat))) || loose;
  for (const e of S.msgs) bin(e).events.push(e);
  for (const e of S.tips) bin(e).tips.push(e);
  const out = [...by.values()];
  if (loose.events.length || loose.tips.length) out.push(loose);
  return out;
}

// An outbox left alone is delivered again every round, so the repeats are the
// bulk of the log and say nothing new. They are off by default and counted.
const saidIn = (t) => t.events.filter(e => S.standing || e.change !== "standing");

function previewOf(t) {
  const all = saidIn(t).concat(t.tips);
  const e = all[all.length - 1];
  if (!e) return "nothing addressed";
  if (e.kind === "transfer") return `transfer \u00b7 ${(e.transfer || {}).amount
    ? num(e.transfer.amount) : "moved nothing"}`;
  if (e.change === "withdrawn") return `${nameOf(e.from_seat)} took back ${e.path}`;
  return `${nameOf(e.from_seat)}: ${(e.text || "").trim().split("\n")[0] || "(empty)"}`;
}

const whenOf = (e) => e.tip ? "not traced yet" : `round ${e.round}`;

// What the addressee started holding. m is the authority wherever the
// observation carried one: the inbox arrives in it, so an episode that named no
// command about the inbox has the message all the same. Where none was carried
// there is only what stood in the environment for that episode to go and fetch.
// Naming the inbox in a command is a second thing either way - a turn spent on
// a read, beside a delivery that cost nothing.
function gotHtml(d, from) {
  const box = esc(d.box || `in/${from}`);
  const again = d.named ? ` \\u00b7 <span class="tag">read ${box}${
    d.shown_before === true ? " again" : ""}</span>` : "";
  if (d.shown_before === true) {
    return `<span style="color:var(--live)">in its observation</span>${again}`;
  }
  if (d.shown_before === false) {
    return d.environment
      ? `<span style="color:var(--warn)" title="${d.clipped
          ? "the initial observation ran past its limit, and what clip takes is the middle"
          : "it stood in the environment and the initial observation did not carry it"}"
         >in its environment, not in its observation</span>${again}`
      : `never reached it${again}`;
  }
  return (d.environment ? `in its environment \\u00b7 <span title="the initial observation was the listing alone,
    so the inbox was there to be fetched">nothing was carried</span>`
                  : `never reached it`) + again;
}

function msgHtml(e, t) {
  const g = e.transfer || {};
  if (e.kind === "transfer") {
    const moved = g.amount ? `moved ${num(g.amount)} to ${esc(nameOf(g.seat || e.to_seat))}`
      : `<span style="color:var(--warn)" title="${esc(g.error || "")}">moved nothing</span>`;
    return `<div class="sys">${seatTag(e.from_seat)} ${e.change === "withdrawn"
      ? "withdrew its transfer declaration" : `declared a transfer \\u00b7 ${moved}`}
      \\u00b7 ${whenOf(e)}</div>`;
  }
  if (e.change === "withdrawn") {
    return `<div class="sys">${seatTag(e.from_seat)} took back ${esc(e.path)}, and nothing
      stands there now \\u00b7 ${whenOf(e)}</div>`;
  }
  const tag = e.change === "edited"
      ? `<span class="tag edit">edited</span>`
    : e.change === "standing"
      ? `<span class="tag" title="left in place, so it is delivered again">said again</span>` : "";
  const diff = e.diff.length ? `<details><summary>what changed</summary><pre class="diff">${
    e.diff.map(l => `<span class="${l[0] === "+" ? "a" : l[0] === "-" ? "d" : "h"}"
      >${esc(l)}</span>`).join("\\n")}</pre></details>` : "";
  const body = e.binary ? `<pre class="note">not text</pre>`
    : e.text ? `<pre>${esc(e.text)}</pre>` : `<pre class="note">empty</pre>`;
  const ft = e.delivered
    ? `\\u2192 ${esc(e.to_run)} s${e.delivered.episode} (round ${
        e.delivered.round}) \\u00b7 ${gotHtml(e.delivered, e.from_seat)}`
    : e.tip ? "standing now; nobody has woken to it yet"
    : "not delivered yet";
  return `<div class="msg ${!t.a || String(e.from_seat) === String(t.a.seat) ? "" : "r"}
      ${e.tip ? "tip" : ""} ${e.change === "standing" ? "same" : ""}">
    <div class="mh">${seatTag(e.from_seat)}
      <span>${esc(e.from_run)} s${e.episode == null ? "?" : e.episode}</span>
      <span class="num">${esc(e.path)}${e.size ? ` \\u00b7 ${num(e.size)} B` : ""}</span>${tag}</div>
    <div class="bub">${body}${diff}</div>
    <div class="ft">${ft}</div></div>`;
}

function threadHtml(t) {
  let out = "", seen = null;
  for (const e of saidIn(t)) {
    if (e.round !== seen) { seen = e.round; out += `<div class="round">round ${e.round}<i></i></div>`; }
    out += msgHtml(e, t);
  }
  if (!out) {
    out = `<div class="empty">${t.a ? `n${esc(t.a.seat)} and n${esc(t.b.seat)} have addressed
      nothing to each other` : "nothing has reached nobody"}</div>`;
  }
  if (t.tips.length) {
    out += `<div class="round">ahead of the last trace<i></i></div>`
         + t.tips.map(e => msgHtml(e, t)).join("");
  }
  return out;
}

function renderMessages() {
  const el = document.getElementById("body"), h = S.head;
  if (h && h.seats.length < 2) {
    blank(el, `<div class="empty">an experiment of one: there is nobody to address</div>`);
    return;
  }
  if (h && !h.posts) {
    blank(el, `<div class="empty">no outbox in this experiment's environment</div>`);
    return;
  }
  const ts = threadsOf();
  if (!ts.length) { blank(el, `<div class="empty">no seats to address</div>`); return; }
  // The busiest thread, so a page opened on an experiment mid-round lands on one
  // that has something in it rather than on whichever pair sorts first.
  if (!ts.some(t => t.key === S.pair)) {
    S.pair = ts.reduce((a, b) => saidIn(b).length > saidIn(a).length ? b : a).key;
  }
  const t = ts.find(x => x.key === S.pair);
  const sig = JSON.stringify([S.pair, S.standing, S.rail, ts.map(x => [x.key, x.events.length]),
    S.tips.map(e => [e.from_seat, e.path, e.change, e.size])]);
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  const side = ts.map(x => {
    const n = saidIn(x).length + x.tips.length;
    return `<button class="pair ${x.key === S.pair ? "on" : ""} ${n ? "" : "mute"}"
        data-k="${esc(x.key)}"><div class="t"><span class="nm">${
        x.a ? `${seatTag(x.a.seat)} \\u21c4 ${seatTag(x.b.seat)}` : "addressed to nobody"}</span>
        <em>${n || "\\u2013"}</em></div>
      <div class="p">${esc(previewOf(x))}</div></button>`;
  }).join("");
  // The toggle is the experiment's, not this thread's, so it is counted over the
  // whole log: switching threads must not move the number it offers.
  const same = S.msgs.filter(e => e.change === "standing").length;
  const head = `<div class="thead">
    <button class="fold" id="rail" style="margin-left:0" title="the other threads"
      >${S.rail ? "\\u203a" : "\\u2039"}</button>
    <span class="nm">${
    t.a ? `${seatTag(t.a.seat)} \\u21c4 ${seatTag(t.b.seat)}` : "addressed to nobody"}</span>
    <span class="note">${t.a ? `${esc(t.a.agent)} \\u00b7 ${esc(t.b.agent)}`
      : "a name that is no seat of this experiment"}</span>
    ${same ? `<label class="tail" style="margin-left:auto"><input type="checkbox" id="stand"
      ${S.standing ? "checked" : ""}>show the ${same} round${same === 1 ? "" : "s"} an outbox
      was left alone</label>` : ""}
    </div>`;
  redraw(el, `<div class="fill"><div class="note" style="margin:2px 0 8px">one file a seat
      on the out/&lt;i&gt; channel \\u00b7 an outbox left alone is delivered again every round</div>
    <div class="chat ${S.rail ? "narrow" : ""}"><div class="pairs">${side}</div>
      <div class="thread">${head}<div id="thread">${threadHtml(t)}</div></div></div></div>`);
  el.querySelectorAll(".pair").forEach(b => b.onclick = () => {
    if (b.dataset.k === S.pair) return;
    S.pair = b.dataset.k; el.dataset.sig = ""; renderMessages();
    // A thread is opened at its first round, not at wherever the last one was
    // being read: the offset held across the redraw belongs to the other thread.
    const now = document.getElementById("thread");
    if (now) now.scrollTop = 0;
  });
  const stand = document.getElementById("stand");
  if (stand) stand.onchange = e => { S.standing = e.target.checked; el.dataset.sig = ""; renderMessages(); };
  document.getElementById("rail").onclick = () => {
    S.rail = !S.rail; hold("rail", S.rail); el.dataset.sig = ""; renderMessages();
  };
}

// --- the blackboards and the private stores -----------------------------------

function keyOf(kind, agent) { return kind + ":" + agent; }

function renderTree(kind) {
  const d = S.tree[kind], el = document.getElementById("body");
  if (!d) return;
  const sig = JSON.stringify([kind, d.columns.map(c =>
    [c.agent, c.committed, c.live, c.files.map(f => [f.path, f.stamp, f.mode])]),
    Object.entries(S.open), Object.entries(S.body).map(([k, v]) => [k, v && v.stamp])]);
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  const what = d.what || "";
  const stamp = (c) => `${c.seat == null ? "" : esc(c.agent) + " \\u00b7 "}as of s${c.committed}${
    c.live == null ? "" : running(c)
      ? " \\u00b7 an episode is running" : " \\u00b7 an episode never finished"}`;
  const index = d.columns.map(c => {
    const open = S.open[keyOf(kind, c.agent)];
    const rows = c.files.map(f => `<tr class="file ${f.path === open ? "on" : ""}"
        data-agent="${esc(c.agent)}" data-p="${esc(f.path)}">
        <td>${esc(f.path)}</td><td class="sz">${num(f.size)} B</td>
        <td class="md">${esc(f.mode || "")}</td>
        <td class="kd">${f.starter ? `<span class="tag starter">starter files</span>` : ""}</td></tr>`).join("");
    return `<div class="col"><div class="h">
        <b>${c.seat == null ? esc(c.agent) : esc(c.seat) + "/"}</b>
        <span>${stamp(c)}</span></div>
      <div class="b">${rows ? `<table>${rows}</table>`
        : `<div class="note" style="padding:10px 0">empty</div>`}</div></div>`;
  }).join("");
  // Only the columns with something open, so one file gets the window and two
  // get half of it each. A listing is narrow; what a file says is not.
  const shown = d.columns.filter(c => S.open[keyOf(kind, c.agent)] != null);
  const bodies = shown.map(c => {
    const open = S.open[keyOf(kind, c.agent)];
    const got = S.body[keyOf(kind, c.agent) + ":" + open];
    return `<div class="col"><div class="h">
        <b>${c.seat == null ? esc(c.agent) : esc(c.seat) + "/"}${esc(open)}</b>
        <span>${stamp(c)}</span></div>
      <div class="b">${got === undefined ? `<div class="note">reading&hellip;</div>`
        : got == null ? `<div class="note">gone</div>`
        : got.text == null ? `<div class="note">binary</div>`
        : `<pre class="out">${esc(got.text)}</pre>`}</div></div>`;
  }).join("");
  redraw(el, `<div class="fill"><div class="note" style="margin:2px 0 8px">${what} \\u00b7
      mirrored back when an episode ends, so each column is as of that agent's own last
      committed one</div>
    <div class="trees">
      <div class="cols">${index}</div>
      ${shown.length ? `<div class="bodies">${bodies}</div>` : ""}
    </div></div>`);
  el.querySelectorAll("tr.file").forEach(tr => tr.onclick = () => {
    const k = keyOf(kind, tr.dataset.agent);
    S.open[k] = S.open[k] === tr.dataset.p ? undefined : tr.dataset.p;
    if (S.open[k] === undefined) delete S.open[k];
    el.dataset.sig = "";
    pullFiles(kind).then(() => renderTree(kind));
  });
}

// A file is read when it is opened, and again when its stamp moves. The listing
// carries the stamp, so a column of files nobody has opened costs nothing.
function pullFiles(kind) {
  const d = S.tree[kind];
  if (!d) return Promise.resolve();
  return Promise.all(d.columns.map(c => {
    const inner = S.open[keyOf(kind, c.agent)];
    if (inner == null) return null;
    const f = c.files.find(x => x.path === inner);
    const key = keyOf(kind, c.agent) + ":" + inner;
    if (!f) { delete S.body[key]; return null; }
    const held = S.body[key];
    if (held && JSON.stringify(held.stamp) === JSON.stringify(f.stamp)) return null;
    const q = `agent=${encodeURIComponent(c.agent)}&channel=${encodeURIComponent(kind)}&path=${encodeURIComponent(inner)}`;
    return get(`/api/experiment/${S.experiment}/file?${q}`)
      .then(got => { S.body[key] = got; }).catch(() => { delete S.body[key]; });
  }).filter(Boolean));
}

// --- one agent's transcript ----------------------------------------------

function openRun(agent) {
  if (S.agent !== agent) {
    S.agent = agent; S.episode = null; S.view = null; S.since = 0; S.source = null; S.tail = true;
    S.txtop = 0; S.txend = false;
  }
  return get(`/api/agent/${agent}`).then(d => {
    S.detail = d;
    if (S.episode == null) {
      S.episode = d.live != null ? d.live
        : (d.episodes.length ? d.episodes[d.episodes.length - 1].episode : null);
    }
    drawTranscript();
    return S.episode == null ? null : pullTurns(true);
  }).catch(() => {});
}

function openSession(n) {
  S.episode = n; S.since = 0; S.source = null; S.view = null; S.tail = true;
  S.txtop = 0; S.txend = false;
  drawTranscript();
  return pullTurns(true);
}

function pullTurns(reset) {
  if (S.agent == null || S.episode == null) return Promise.resolve();
  const agent = S.agent, sess = S.episode, since = reset ? 0 : S.since;
  return get(`/api/agent/${agent}/episode/${sess}?since=${since}`).then(d => {
    if (S.agent !== agent || S.episode !== sess) return;
    // The episode finished between polls: what was pending has landed, so it is
    // asked for again from the start rather than appended to.
    if (!reset && S.source === "raw" && d.source === "trace") return pullTurns(true);
    const fresh = reset || !S.view;
    if (fresh) { S.view = d; } else { S.view.turns = S.view.turns.concat(d.turns);
      Object.assign(S.view, { ...d, turns: S.view.turns }); }
    S.source = d.source;
    S.since = S.view.turns.reduce((m, t) => Math.max(m, t.turn || 0), 0);
    renderTranscript(fresh);
  }).catch(() => {});
}

function drawTranscript() {
  const d = S.detail, el = document.getElementById("body");
  if (!d) return;
  const seats = (S.head ? S.head.seats : []).map(s =>
    `<button class="chip ${s.agent === S.agent ? "on" : ""} ${running(s) ? "live" : ""}"
       data-agent="${esc(s.agent)}">${s.seat == null ? "" : esc(s.seat) + " \\u00b7 "}${esc(s.agent)}</button>`).join("");
  // A round the agent has no episode in is one it was already out of, and there
  // is nothing to open: the chip says so rather than disappearing and closing
  // the gap.
  const last = d.episodes.reduce((a, s) => Math.max(a, s.round || 0), 0);
  const byRound = {};
  d.episodes.forEach(s => { if (s.round) byRound[s.round] = s; });
  const rounds = Array.from({ length: last }, (_, i) => i + 1).map(r => {
    const s = byRound[r];
    if (!s) return `<button class="chip off" title="it was out of the agent by this round">r${r}</button>`;
    return `<button class="chip ${s.episode === S.episode ? "on" : ""}
       ${s.live && running(d) ? "live" : ""} ${s.live && !running(d) ? "halt" : ""}
       ${["interrupted","api_error","harness_error"].includes(s.stop) ? "halt" : ""}"
       data-s="${s.episode}" title="s${s.episode} \\u00b7 ${esc(s.stop || (running(d) ? "running" : "unfinished"))
         } \\u00b7 ${s.turns} turns \\u00b7 spent ${num(s.spent)}">r${r}</button>`;
  }).join("");
  const here = d.episodes.find(s => s.episode === S.episode) || {};
  el.dataset.sig = "";
  // Provenance is config, and config is what an episode was. Drift is the
  // exception, so it is what stays on the page; the rest is put one click away
  // rather than between the reader and the transcript. Every key the trace
  // holds is shown, in the order provenance() writes them, because drift
  // reports on all of them and a banner may not name a field the panel hides.
  const drifted = here.drift && here.drift.length;
  // "<key>: <was> -> <now>", so the key is what stands before the first colon.
  const moved = new Set((here.drift || []).map(s => s.split(":")[0]));
  el.innerHTML = `<div class="fill">
    <div class="strip" style="margin-bottom:6px"><span class="lbl">agent</span>${seats}</div>
    <div class="strip"><span class="lbl">round</span>${
      rounds || `<span class="note">none yet</span>`}</div>
    <div class="txbar" style="margin-top:10px">
      <div id="txhead" class="note"></div>
      <label class="tail"><input type="checkbox" id="tailbox" ${S.tail ? "checked" : ""}>follow</label>
    </div>
    <div class="txwrap"><div id="tx"><div class="empty">loading&hellip;</div></div>
      <button class="jump" id="jump" style="display:none">jump to latest \\u2193</button></div>
    ${drifted ? `<div class="note" style="margin-top:8px;color:var(--warn)"
      >provenance drifted mid-agent: ${esc(here.drift.join("; "))}</div>` : ""}
    <details style="margin-top:8px"><summary class="note" style="cursor:pointer"
      >provenance \\u00b7 episode ${S.episode}</summary>
      <div class="panel" style="margin-top:8px">${
      Object.keys(here.provenance || {}).length
        ? `<div class="grid">${Object.entries(here.provenance).map(([k, v]) =>
            `<div class="kv"><div class="k" ${moved.has(k)
              ? `style="color:var(--warn)"` : ""}>${esc(k.replace(/_/g, " "))}</div>
             <div class="v">${esc(provValue(v))}</div></div>`).join("")}</div>`
        : `<div class="note">no trace yet</div>`}</div>
    </details></div>`;
  el.querySelectorAll(".chip[data-agent]").forEach(b => b.onclick = () => openRun(b.dataset.agent));
  el.querySelectorAll(".chip[data-s]").forEach(b => b.onclick = () => openSession(Number(b.dataset.s)));
  const tx = document.getElementById("tx"), toEnd = () => {
    tx.scrollTop = tx.scrollHeight;
    updateJump();
  };
  document.getElementById("tailbox").onchange = e => {
    S.tail = e.target.checked;
    if (S.tail) toEnd();
  };
  // The pane is thrown away and rebuilt whenever an episode starts or ends, so
  // where it is being read is recorded as it is scrolled rather than read back
  // off a node that may no longer be there.
  tx.onscroll = () => { S.txtop = tx.scrollTop; S.txend = atEnd(tx); updateJump(); };
  document.getElementById("jump").onclick = toEnd;
  // The pane these counted is gone, rebuilt empty by the line above.
  S.drawn = 0;
  if (S.view) renderTranscript(true);
}

function turnHtml(t, pend) {
  const bits = [`turn ${t.turn}`, `${num(t.balance)} left`];
  if (t.micros) bits.push(`\\u2212${num(t.micros)}`);
  if (t.prefix) bits.push(`ctx ${num(t.prefix)}`);
  const tags = [
    t.stop_reason ? `<span class="tag">${esc(t.stop_reason)}</span>` : "",
    t.served_by_fallback ? `<span class="tag warn">fallback \\u00b7 ${esc(t.model)}</span>` : "",
    t.stop_details ? `<span class="tag bad">refused${t.stop_details.category ? " \\u00b7 " + esc(t.stop_details.category) : ""}</span>` : "",
    (t.unpriced_model || []).length ? `<span class="tag warn">unpriced ${esc((t.unpriced_model || []).join(", "))}</span>` : "",
  ].join(" ");
  const tools = (t.tools || []).map(c =>
    `<div class="cmd">${esc(c.command == null ? "(restart)" : c.command)}</div>` +
    (c.result == null ? `<div class="pend">\\u23f3 ${pend}</div>`
                      : `<pre class="out">${esc(c.result)}</pre>`)).join("");
  return `<div class="turn"><div class="th"><span class="num">${bits.join(" \\u00b7 ")}</span>${tags}</div>
    ${t.thinking ? `<div class="think">${esc(t.thinking)}</div>` : ""}
    ${t.text ? `<div class="say">${esc(t.text)}</div>` : ""}
    ${t.stop_details && t.stop_details.explanation ? `<div class="note">${esc(t.stop_details.explanation)}</div>` : ""}
    ${tools}</div>`;
}

function updateJump() {
  const tx = document.getElementById("tx"), b = document.getElementById("jump");
  if (tx && b) b.style.display = behind(tx) > 60 ? "block" : "none";
}

// One block per tree the episode changed, because what matters about a change
// here is who can see it.
function diffHtml(changes) {
  return (changes || []).filter(c => c.lines.length).map(c =>
    `<div class="turn"><div class="th">${esc(c.what)}</div><pre class="out diff">${
      c.lines.map(l => `<span class="${l[0] === "+" ? "a" : l[0] === "-" ? "d" : "h"}">${esc(l)}</span>`)
        .join("\\n")}</pre></div>`).join("");
}

// The record the episode started holding, a file at a time. It reached the model as
// one command's stdout, but every section of it is a file some other agent wrote
// or the harness rendered, and a reader wants one of them rather than the blob.
// The inboxes are open because they are what a round turns on; the rest is a
// click. An episode whose observation carried no record renders no block at all.
function carriedHtml(o) {
  if (!o.shown_before || !o.shown_before.length) return "";
  const bytes = o.shown_before.reduce((n, s) => n + s.bytes, 0);
  const files = o.shown_before.map(s =>
    `<details class="sec" ${o.inbox && s.path.startsWith(o.inbox + "/") ? "open" : ""}>
       <summary>=== ${esc(s.path)} === <i>${num(s.bytes)} B</i></summary>
       <pre class="out">${esc(s.text)}</pre></details>`).join("");
  return `<div class="turn"><div class="th">${esc(o.name)} \\u00b7 the record it started holding
    <span class="num">${o.shown_before.length} files \\u00b7 ${num(bytes)} B</span>${o.clipped
      ? `<span class="tag warn" title="the initial observation ran past its limit, and what clip
          takes is the middle">clipped</span>` : ""}</div>${files}</div>`;
}

// `fresh` rebuilds; without it only the turns that arrived since the last draw
// are appended, so the nodes the reader is scrolled in are the nodes that stay.
function renderTranscript(fresh) {
  const v = S.view, tx = document.getElementById("tx");
  if (!v || !tx) return;
  // Sent whole with the first turn of an episode and held from there, so an
  // append carries none of it and leaves what is on the page alone.
  const o = v.observation || {};
  const dead = v.live && v.age != null && v.age >= S.stale;
  // A command whose episode is over has no output coming: the trace that would
  // have carried it was never written.
  const pend = dead ? "no trace was written; this output is lost" : PENDING;
  const state = !v.live ? esc(v.stop)
    : dead ? `<b style="color:var(--warn)">unfinished</b> \\u00b7 last turn ${ago(v.age)} \\u00b7 derived cost`
           : "<b style='color:var(--live)'>running</b> \\u00b7 derived cost";
  const ob = v.obligations || {};
  document.getElementById("txhead").innerHTML =
    `episode ${v.episode} \\u00b7 ${state} \\u00b7 ${v.total_turns} turns \\u00b7 spent ${num(v.spent)}` +
    (ob.posted === false ? ` \u00b7 <span style="color:var(--bad)">did not post</span>${
      charged((v.board || {}).penalty)}` : "") +
    (ob.messaged === false ? ` \u00b7 <span style="color:var(--bad)">${
      esc(v.messages_why)}</span>${charged((v.mailbox || {}).penalty)}` : "") +
    (ob.transferred === false ? ` \\u00b7 <span style="color:var(--bad)"
      title="${esc((v.transfer || {}).error || "nothing it declared moved anything")}"
      >no transfer of its own</span>${charged((v.transfer || {}).penalty)}` : "") +
    (v.forgiven ? ` \\u00b7 floored ${num(v.forgiven)}` : "") +
    (v.error ? ` \\u00b7 <span style="color:var(--bad)">${esc(v.error)}</span>` : "") +
    ((v.missing_tools || []).length ? ` \\u00b7 reached for and absent: ${esc(v.missing_tools.join(", "))}` : "");
  // Measured before anything is written, or the answer is about the page the
  // reader has not seen yet.
  const wasAtEnd = atEnd(tx);

  if (fresh) {
    redraw(tx,
      `<div class="turn"><div class="th">start \\u00b7 n at start ${
        v.series_before.length ? num(v.series_before[v.series_before.length - 1]) : "\\u2013"}</div>
        <div class="cmd">${esc(o.command)}</div>` +
        (o.listing == null ? `<div class="pend">\\u23f3 ${pend}</div>`
                           : `<pre class="out">${esc(o.listing)}</pre>`) + `</div>` +
      carriedHtml(o) +
      v.turns.map(t => turnHtml(t, pend)).join("") +
      diffHtml(v.changes));
    S.drawn = v.turns.length;
    // An episode opens where it starts and stays there. Only a reader who had
    // scrolled to the end, and is following, is carried to the new one - and
    // that is asked of the state, because the pane may be a new one that never
    // held the position it is being given back.
    tx.scrollTop = S.tail && S.txend ? tx.scrollHeight : S.txtop;
  } else {
    const extra = v.turns.slice(S.drawn);
    if (extra.length) {
      tx.insertAdjacentHTML("beforeend", extra.map(t => turnHtml(t, pend)).join(""));
      S.drawn = v.turns.length;
      // Follow only a reader who was already at the end. Anyone who has scrolled
      // up is reading, and the new turn waits behind the jump button.
      if (S.tail && wasAtEnd) tx.scrollTop = tx.scrollHeight;
    }
  }
  updateJump();
}

// --- the loop ------------------------------------------------------------

function renderTab() {
  if (S.tab === "mailbox") {
    return get(`/api/experiment/${S.experiment}/messages?since=${S.msgn}`).then(d => {
      S.msgs = S.msgs.concat(d.events); S.msgn = d.committed; S.tips = d.tip;
      renderMessages();
    }).catch(() => {});
  }
  if (S.tab !== "agent") {
    return get(`/api/experiment/${S.experiment}/tree/${encodeURIComponent(S.tab)}`).then(d => {
      S.tree[S.tab] = d;
      return pullFiles(S.tab).then(() => renderTree(S.tab));
    }).catch(() => {});
  }
  const seats = S.head ? S.head.seats : [];
  if (!seats.length) return Promise.resolve();
  if (S.agent == null || !seats.some(s => s.agent === S.agent)) {
    return openRun(((seats.find(running) || seats[0]) || {}).agent);
  }
  // An episode started or ended since the last poll: the round chips and the
  // episode stamp are both out of date, so the agent is re-read rather than
  // patched. The pane is another tab's until this one has drawn it, and an agent
  // whose detail never arrived has no scaffold either; a turn can only be
  // appended to a transcript that is on the page.
  const me = seats.find(s => s.agent === S.agent);
  if (!S.detail || (me && me.live !== S.detail.live)
      || !document.getElementById("tx")) return openRun(S.agent);
  return pullTurns(false);
}

function refresh() {
  return get(`/api/experiment/${S.experiment}`).then(h => {
    S.head = h; S.poll = h.poll; S.stale = h.stale;
    renderHeader();
    return renderTab();
  }).catch(() => {});
}

function poll() {
  get("/api/experiments").then(d => {
    S.experiments = d.experiments; S.poll = d.poll; S.stale = d.stale;
    if (!S.experiments.length) {
      blank(document.getElementById("body"),
        `<div class="empty">no agents under ${esc(d.root)}/records</div>`);
      return;
    }
    if (S.experiment == null || !S.experiments.some(c => c.name === S.experiment)) {
      // An experiment actually moving beats one merely on disk: a set that stopped
      // months ago should not be what the page opens on.
      const first = S.experiments.find(c => c.name === d.focus)
        || S.experiments.find(c => c.running) || S.experiments[0];
      S.experiment = first.name;
    }
    renderPicks();
    return refresh();
  }).catch(() => {}).then(() => setTimeout(poll, S.poll));
}

document.getElementById("fold").onclick = () => {
  S.compact = !S.compact; hold("compact", S.compact); renderFold();
};
renderFold();
poll();
</script>
"""


# --- the server -------------------------------------------------------------


class View(http.server.BaseHTTPRequestHandler):
    """Read-only. Every route is a GET, and nothing here opens a file to write."""

    server_version = "view.py"

    def do_GET(self) -> None:                    # noqa: N802 - BaseHTTPRequestHandler's name
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        query = urllib.parse.parse_qs(url.query)
        try:
            self.route(parts, query)
        except BrokenPipeError:                  # the page navigated away mid-answer
            pass
        except Exception as e:                   # noqa: BLE001 - a viewer never takes the page down
            self.send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    def route(self, parts: list[str], query: dict[str, list[str]]) -> None:
        """One request. `parts` is the path split on slashes, already unquoted.

        A name off the URL reaches the filesystem only after matching one
        already there, so no path can be walked out of records/ or environments/.
        """
        if not parts:
            return self.send_page()
        if parts == ["api", "experiments"]:
            return self.send_json({"experiments": experiments(), "focus": getattr(self.server, "focus", None),
                                   "poll": POLL_MS, "stale": STALE_AFTER, "root": str(harness.ROOT)})
        if len(parts) >= 3 and parts[:2] == ["api", "experiment"]:
            c = experiment_named(parts[2])
            if c is None:
                return self.send_json({"error": f"no experiment {parts[2]}"}, status=404)
            rest = parts[3:]
            if not rest:
                return self.send_json(header(c))
            if rest == ["mailbox"]:
                since = query.get("since", ["0"])[0]
                return self.send_json(messages(c, int(since) if since.isdigit() else 0))
            if len(rest) == 2 and rest[0] == "tree" and rest[1] in trees(c):
                return self.send_json(tree_view(c, rest[1]))
            if rest == ["file"]:
                agent = query.get("agent", [""])[0]
                kind = query.get("channel", [""])[0]
                inner = query.get("path", [""])[0]
                if agent not in c["members"] or kind not in trees(c):
                    return self.send_json({"error": "no such file"}, status=404)
                got = file_view(agent, kind, inner)
                if got is None:
                    return self.send_json({"error": f"no {kind} file {inner} in {agent}"}, status=404)
                return self.send_json(got)
        if len(parts) >= 3 and parts[:2] == ["api", "agent"]:
            agent = parts[2]
            if agent not in agent_names():
                return self.send_json({"error": f"no agent {agent}"}, status=404)
            rest = parts[3:]
            if not rest:
                return self.send_json(agent_view(agent))
            if len(rest) == 2 and rest[0] == "episode" and rest[1].isdigit():
                since = query.get("since", ["0"])[0]
                view = session_view(agent, int(rest[1]), int(since) if since.isdigit() else 0)
                if view is None:
                    return self.send_json({"error": f"no episode {rest[1]} in {agent}"}, status=404)
                return self.send_json(view)
        return self.send_json({"error": "no such route"}, status=404)

    def send_page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The page is a constant in a file that gets edited. Cached, a restarted
        # server keeps serving the browser the version it had before.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Polls of the same URL must not be answered from the browser's cache,
        # or a running episode stops moving on screen while it moves on disk.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet. A line per poll is a line every second and a half, forever."""


def serve(port: int = PORT, focus: str | None = None) -> http.server.ThreadingHTTPServer:
    """A server bound and ready, which the caller starts.

    Bound to loopback and nothing else: there is no authentication here.
    Returned rather than agent, so a check can drive the real handler in-process.
    """
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), View)
    httpd.focus = focus
    return httpd


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
            ap.error(f"no agent {a.agent} under {harness.ROOT / 'records'}")
        held = experiment_of(a.agent)
        focus = held["name"] if held else None
    if focus and not any(c["name"] == focus for c in sets):
        ap.error(f"no experiment {focus}; there is {', '.join(c['name'] for c in sets) or 'nothing'}")

    try:
        httpd = serve(a.port, focus)
    except OSError as e:
        print(f"port {a.port}: {e}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    seated = sum(c["seated"] for c in sets)
    print(f"{url}  ({len(sets)} sets, {seated} seated, under {harness.ROOT / 'records'})")
    print("read-only: nothing here is written, and nothing here reaches the agent")
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
