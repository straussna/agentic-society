"""Several agents advancing together, each reading the others' blackboards.

    py -3 experiment.py --agents g01 g02 g03 --rounds 20
    py -3 experiment.py --manifest experiments/persona.toml --rounds 20

One round is one episode for each agent; a seat names a blackboard and a
balance. A manifest gives each agent its own terms and the experiment its schedule."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable

import harness

# A seat is named by its number, which is what makes a bare number unavailable
# as an agent id: the two share a namespace in the environment an episode sees.
# A label defaults to the seat number, so a bare number is what a seat is called
# in every path and file the agents see; an agent id must be something else.
BARE_NUMBER = re.compile(r"^\d+$")

# Goes at building an environment before the agent drops out. An environment is built from the
# containers and the copies of several other agents' trees, so a failure can be
# the daemon rather than the agent, and dropping on the first one costs the experiment
# an agent for the rest of the experiment. None of it is billed, so the only
# thing another go spends is the time.
ATTEMPTS = 2

# How a round is driven. "sequential": one episode at a time, the starting agent
# moving each round, each episode reading what the ones before it in the round
# wrote. "simultaneous": every environment built before any episode agents, the episodes agent
# at once, and the results settle in seat order, so nobody reads this round's
# writes and a transfer made in one round is seen at the next.
SCHEDULES = ("sequential", "simultaneous")

# What a manifest may say about one agent. Everything else an agent is comes from
# the experiment's defaults and config.toml.
AGENT_KEYS = {"id", "label", "starter_files", "starter_files_below", "budget", "model"}

# What an agent may be called to its peers: one path segment, since it lands in
# paths and file names. The default is the seat number.
LABEL = re.compile(r"^[A-Za-z0-9._-]+$")

# The failures that mean an environment could not be built, none of them billed.
BUILD_FAILURES = (subprocess.CalledProcessError, OSError, harness.EnvironmentBuildError)


def mapping(agents: list[str]) -> dict[str, str]:
    """Seat -> agent, for the whole experiment.

    Absolute and complete: a seat means the same agent to every reader, so a note
    citing one resolves the same way for all of them.
    """
    return {str(i): agent for i, agent in enumerate(agents, 1)}


def order(agents: list[str], rnd: int) -> list[str]:
    """The agents, rotated by the round, so no seat has a standing advantage."""
    i = rnd % len(agents)
    return agents[i:] + agents[:i]


def why_out(agent: str) -> str | None:
    """Why the agent can take no further episode, or None where it can take one.

    Both reasons are final: an agent at zero or less is not a transfer target, so no
    peer can fund it back to the table.
    """
    account = harness.load_account(agent)
    if harness.stalled(account):
        return f"refused its last {harness.REFUSAL_STREAK} episodes running"
    if harness.spent_out(account):
        return "nothing left to spend"
    return None


# --- the manifest -------------------------------------------------------------


def shorthand(ids: list[str]) -> dict:
    """The manifest `--agents` stands for: these agents, on config.toml, in rotation."""
    return {"schedule": "sequential", "overrides": {}, "agents": [{"id": i} for i in ids],
            "labels": {str(n): str(n) for n in range(1, len(ids) + 1)},
            "channels": None, "harness_files": None, "sha256": ""}


def load_manifest(path: Path) -> dict:
    """Read an experiment manifest: the schedule, the experiment's defaults, and each agent's terms.

    Returns {"schedule", "overrides", "agents", "sha256"}. Unknown keys, wrong
    types, and terms that do not go together exit with a message naming the
    file, the way load_config does. The overrides' own types and ranges are
    checked when start() applies them, so one set of rules holds for both.
    """
    if not path.exists():
        raise SystemExit(f"{path}: no such manifest")
    data = path.read_bytes()
    try:
        top = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"{path}: not a manifest: {e}") from None

    allowed = {"schedule", "agent", "channel", "harness_files"} | {t.lower() for t in harness.TUNABLES}
    if unknown := sorted(set(top) - allowed):
        if unknown[0] in harness.RETIRED:
            raise SystemExit(f"{path}: unknown key {unknown[0]!r}; it is now {harness.RETIRED[unknown[0]]}")
        raise SystemExit(f"{path}: unknown key {unknown[0]!r}; expected schedule, [[agent]] tables, "
                         f"[[channel]] tables, [harness_files], and any config.toml key")
    tables, harness_files = top.get("channel"), top.get("harness_files")
    if tables is not None and not (isinstance(tables, list) and all(isinstance(x, dict) for x in tables)):
        raise SystemExit(f"{path}: channels are [[channel]] tables")
    if harness_files is not None and not isinstance(harness_files, dict):
        raise SystemExit(f"{path}: [harness_files] is a table")
    schedule = top.get("schedule", "sequential")
    if schedule not in SCHEDULES:
        raise SystemExit(f"{path}: schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    agents = top.get("agent")
    if not isinstance(agents, list) or not all(isinstance(r, dict) for r in agents):
        raise SystemExit(f"{path}: agents are [[agent]] tables, each with an id")
    if len(agents) < 2:
        raise SystemExit(f"{path}: an experiment needs two or more agents; one agent has no peers")

    ids = []
    for entry in agents:
        if unknown := sorted(set(entry) - AGENT_KEYS):
            raise SystemExit(f"{path}: unknown key {unknown[0]!r} in an agent; "
                             f"expected {sorted(AGENT_KEYS)}")
        agent = entry.get("id")
        if not isinstance(agent, str) or not agent:
            raise SystemExit(f"{path}: every agent needs an id")
        if BARE_NUMBER.match(agent):
            raise SystemExit(f"{path}: agent id {agent!r} is a bare number, which is what seats "
                             f"are called")
        if agent in ids:
            raise SystemExit(f"{path}: agent {agent!r} is listed twice")
        ids.append(agent)
        for key, kind in (("starter_files", str), ("starter_files_below", int), ("budget", int), ("model", str)):
            if key in entry and (type(entry[key]) is not kind):
                raise SystemExit(f"{path}: {agent}: {key} must be {kind.__name__}, "
                                 f"got {type(entry[key]).__name__}")
        if ("starter_files" in entry) != ("starter_files_below" in entry) or \
                bool(entry.get("starter_files")) != bool(entry.get("starter_files_below")):
            raise SystemExit(f"{path}: {agent}: starter_files and starter_files_below are set together or not at "
                             f"all; got starter_files={entry.get('starter_files')!r}, "
                             f"starter_files_below={entry.get('starter_files_below')!r}")
        if entry.get("starter_files_below", 0) < 0:
            raise SystemExit(f"{path}: {agent}: starter_files_below must be zero or positive")
        if entry.get("starter_files") and not harness.files_dir(entry["starter_files"]).is_dir():
            raise SystemExit(f"{path}: {agent}: starter_files {entry['starter_files']!r} is not a directory under "
                             f"{harness.ROOT / 'starter_files'}")
        if "budget" in entry and entry["budget"] <= 0:
            raise SystemExit(f"{path}: {agent}: budget must be positive")
        if "model" in entry and entry["model"] not in harness.PRICES:
            raise SystemExit(f"{path}: {agent}: model {entry['model']!r} has no rates; add it to "
                             f"PRICES in harness.py")

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

    if tables is not None or harness_files is not None:
        harness.validate_channels(tables, harness_files, str(path), tuple(labels.values()))

    overrides = {k: v for k, v in top.items()
                 if k not in ("schedule", "agent", "channel", "harness_files")}
    return {"schedule": schedule, "overrides": overrides, "agents": agents, "labels": labels,
            "channels": tables, "harness_files": harness_files,
            "sha256": hashlib.sha256(data).hexdigest()}


def terms_of(entry: dict) -> dict[str, Any]:
    """One agent's creation terms as load_account's keywords, None where the manifest is silent."""
    return {k: entry.get(k) for k in ("model", "budget", "starter_files", "starter_files_below")}


def stamp_of(manifest: dict) -> dict[str, str]:
    """What every episode of the experiment records about how it was driven."""
    return {"schedule": manifest["schedule"], "manifest_sha256": manifest["sha256"]}


# --- rounds -------------------------------------------------------------------


def seat_of(seats: dict[str, str], agent: str) -> str:
    return next(i for i, r in seats.items() if r == agent)


def labels_for(seats: dict[str, str], labels: dict[str, str] | None) -> dict[str, str]:
    """Seat -> label for every seat: the manifest's where it gave one, the seat number otherwise."""
    return {seat: (labels or {}).get(seat, seat) for seat in seats}


def preparer(agent: str, seats: dict[str, str], stamp: dict[str, str],
             labels: dict[str, str] | None = None) -> Callable[[dict], None]:
    """What an agent's account is told before each episode: where it sits, what it and
    every other agent is called, and how the experiment is being driven. Nothing is
    copied into anything the agent can write, so there is nothing to revert afterwards."""
    named = labels_for(seats, labels)

    def prepare(account: dict) -> None:
        seat = seat_of(seats, agent)
        account["seat"] = seat
        account["label"] = named[seat]
        account["peers"] = {"seen": seats, "labels": named}
        account["experiment"] = dict(stamp)
    return prepare


def sequential_round(agents: list[str], live: set[str], rnd: int, create,
              stamp: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> bool:
    """One episode for each agent still in the experiment, in rotated order.

    Returns whether any of them took one; a round where none did moved nothing.
    """
    stamp = stamp or stamp_of(shorthand(agents))
    seats = mapping(agents)
    acted = False
    for agent in order(agents, rnd):
        if harness.STOPPING:
            # A stop that landed while no episode was in flight. Same end as one
            # that landed inside an episode, and for the same reason: main() ends
            # the rounds, and no environment is built for an agent that will not harness.
            raise KeyboardInterrupt
        if agent not in live:
            continue

        prepare = preparer(agent, seats, stamp, labels)
        trace, before = None, len(harness.load_account(agent)["episodes"])
        for attempt in range(1, ATTEMPTS + 1):
            try:
                trace = harness.drive(agent, create, prepare)
                break
            except BUILD_FAILURES as e:
                # Building the environment precedes the first API call, so nothing
                # here was billed. The container name is the agent and episode
                # index, so the next attempt reaps the last one's leavings by
                # name. The episode count is what says the attempt was free.
                print(f"{agent}: could not build an environment for this episode "
                      f"({attempt} of {ATTEMPTS}): {type(e).__name__}: {e}", file=sys.stderr)
                if attempt == ATTEMPTS or len(harness.load_account(agent)["episodes"]) != before:
                    print(f"{agent}: dropping out, it has no environment to harness to", file=sys.stderr)
                    live.discard(agent)
                    break
        if trace is None and agent not in live:
            continue

        if trace is None:
            # main() asks the same question of everyone before the round, so a
            # agent reaching here was asked by a caller that did not. Either way it
            # is off the table: neither reason it could not act goes away.
            print(f"{agent:<6} drops out: {why_out(agent) or 'no episode could start on it'}")
            live.discard(agent)
            continue
        acted = True
        if trace["stop"] in harness.STOP_EVERYTHING:
            # Ctrl+C reaches the episode because the experiment is in the foreground
            # of the shell it was pressed in, so what it asks to stop is the
            # experiment rather than the agent that happened to be awake. The
            # episode it landed in is committed and traced before this; what
            # does not happen is the next one. main() ends the rounds on it.
            raise KeyboardInterrupt
        if trace["stop"] in harness.STOP_THE_RUN:
            print(f"{agent}: dropping out, episode {trace['episode']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(agent)
    return acted


def build_all(agents: list[str], live: set[str], seats: dict[str, str],
              stamp: dict[str, str], labels: dict[str, str] | None = None) -> dict[str, harness.Episode]:
    """Every live agent's environment, built in seat order before any episode agents.

    An agent whose environment fails ATTEMPTS times drops out; the rest are built. A stop
    landing here abandons every environment built so far, since none has billed.
    """
    starts: dict[str, harness.Episode] = {}
    for agent in agents:
        if harness.STOPPING:
            for w in starts.values():
                w.abandon()
            raise KeyboardInterrupt
        if agent not in live:
            continue
        built, before = None, len(harness.load_account(agent)["episodes"])
        for attempt in range(1, ATTEMPTS + 1):
            try:
                built = harness.ready(agent, preparer(agent, seats, stamp, labels))
                break
            except BUILD_FAILURES as e:
                print(f"{agent}: could not build an environment for this episode "
                      f"({attempt} of {ATTEMPTS}): {type(e).__name__}: {e}", file=sys.stderr)
                if attempt == ATTEMPTS or len(harness.load_account(agent)["episodes"]) != before:
                    print(f"{agent}: dropping out, it has no environment to harness to", file=sys.stderr)
                    live.discard(agent)
                    break
        if built is None:
            if agent in live:
                print(f"{agent:<6} drops out: {why_out(agent) or 'no episode could start on it'}")
                live.discard(agent)
            continue
        starts[agent] = built
    return starts


def simultaneous_round(agents: list[str], live: set[str], rnd: int, create,
                  stamp: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> bool:
    """One episode for each agent still in the experiment, all at once.

    Every environment is built first, so no episode reads this round's writes. The
    episodes agent in threads. Then each settles in seat order, the transfers they
    made are credited, and each closes in seat order: a credit lands after the
    receiver's own turns and before its floor, so an agent is out on the round's
    net and never lifted back by a transfer that had already arrived. Returns
    whether any agent took an episode.
    """
    stamp = stamp or {"schedule": "simultaneous", "manifest_sha256": ""}
    seats = mapping(agents)
    starts = build_all(agents, live, seats, stamp, labels)
    if harness.STOPPING:
        for w in starts.values():
            w.abandon()
        raise KeyboardInterrupt
    if not starts:
        return False

    outs: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def go(agent: str, w: harness.Episode) -> None:
        try:
            outs[agent] = harness.run_episode(w, create)
        except BaseException as e:      # run_episode has saved and reaped on its way out
            errors[agent] = e

    threads = [threading.Thread(target=go, args=(agent, w), name=agent, daemon=True)
               for agent, w in starts.items()]
    for t in threads:
        t.start()
    # A timed join, so a signal reaches the handler while the episodes agent: the
    # flag it sets is read at the top of every episode's next turn.
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(0.2)

    pending: list[tuple[str, int]] = []
    settled = {agent: harness.settle_episode(w, outs[agent], lambda r, a: pending.append((r, a)))
               for agent, w in starts.items() if agent in outs}
    for agent, amount in pending:
        if agent in settled:
            harness.credit_meter(starts[agent].account, amount)
            starts[agent].credited += amount
        else:
            harness.credit_on_disk(agent, amount)
    traces = {agent: harness.close_episode(w, outs[agent], settled[agent])
              for agent, w in starts.items() if agent in settled}
    if errors:
        raise next(iter(errors.values()))

    for agent, trace in traces.items():
        if trace["stop"] in harness.STOP_THE_RUN and trace["stop"] not in harness.STOP_EVERYTHING:
            print(f"{agent}: dropping out, episode {trace['episode']} ended {trace['stop']}",
                  file=sys.stderr)
            live.discard(agent)
    if any(trace["stop"] in harness.STOP_EVERYTHING for trace in traces.values()):
        # Every episode in flight ended at its next turn and is committed and
        # traced above; what does not happen is the next round.
        raise KeyboardInterrupt
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI. Verifies the prompt digest and the endpoint, then runs the rounds."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--agents", nargs="+", metavar="ID",
                     help="two or more agent ids, each on config.toml, in rotation")
    src.add_argument("--manifest", type=Path, metavar="PATH",
                     help="an experiment manifest: the schedule, the experiment's defaults, and each "
                          "agent's own terms")
    ap.add_argument("--rounds", type=int, default=1, metavar="N",
                    help="up to N episodes for each agent, stopping early as budgets end")
    ap.add_argument("--config", type=Path, help="default: config.toml beside harness.py")
    a = ap.parse_args(argv)

    if a.rounds < 1:
        ap.error("--rounds must be at least 1")
    if a.agents is not None:
        if len(a.agents) < 2 or len(set(a.agents)) != len(a.agents):
            ap.error("--agents needs two or more distinct ids; one agent has no peers")
        if bad := [r for r in a.agents if not r or BARE_NUMBER.match(r)]:
            ap.error(f"agent ids must not be bare numbers, which is what seats are called: {bad}")
        manifest = shorthand(a.agents)
    else:
        manifest = load_manifest(a.manifest)

    create = harness.start(a.config, overrides=manifest["overrides"],
                           models={e["model"] for e in manifest["agents"] if e.get("model")},
                           channel_tables=manifest["channels"], harness_files=manifest["harness_files"],
                           labels=tuple(manifest["labels"].values()))
    harness.catch_signals()
    agents = [e["id"] for e in manifest["agents"]]
    live = set(agents)
    stamp = stamp_of(manifest)
    a_round = simultaneous_round if manifest["schedule"] == "simultaneous" else sequential_round
    # Create them all before the first round. An agent's directories do not exist
    # until it is created, and without this the agent that goes first would find
    # its neighbours' blackboards missing - in the episode where every baseline
    # far has formed the doctrine it then keeps. Each is created on its own
    # terms, and an agent that exists already must have been created on the same.
    for entry in manifest["agents"]:
        harness.load_account(entry["id"], **terms_of(entry))
    print(f"experiment: {', '.join(agents)}  ({len(agents)} agents, up to {a.rounds} rounds, "
          f"{manifest['schedule']})")
    try:
        for rnd in range(a.rounds):
            # Asked before the round rather than discovered in the middle of one,
            # so the header names who will act. Between here and an agent's own turn
            # its balance can only move up, a peer's transfer being the only thing
            # that reaches it, so this is the answer its episode would give too.
            for agent in order(agents, rnd):
                if agent in live and (why := why_out(agent)):
                    print(f"{agent:<6} drops out: {why}")
                    live.discard(agent)
            if not live:
                print(f"every agent is out after {rnd} rounds")
                break
            # The order is what decides who acts on this round's information and
            # who acts on last round's, so it belongs on screen beside the round
            # number. Under a simultaneous nobody acts on this round's, and the seats
            # are named in their own order.
            ordered = agents if a_round is simultaneous_round else order(agents, rnd)
            acting = [r for r in ordered if r in live]
            print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")
            if len(live) == 1:
                # Every other seat is out, so the experiment is decided and no later
                # round can decide it differently. The one agent still holding a
                # balance takes a last episode - owing no transfer and no message,
                # there being nobody left to make either to - and the rounds end
                # on it rather than running it down alone.
                a_round(agents, live, rnd, create, stamp, manifest["labels"])
                print(f"{acting[0]} is the only agent left with anything to spend; "
                      f"the rounds end here")
                break
            if not a_round(agents, live, rnd, create, stamp, manifest["labels"]):
                # Nobody seated could be given an environment to start in, and a round
                # that moved nothing would be asked the same question again.
                print(f"no agent could take an episode in round {rnd + 1}; "
                      f"the rounds end here with {len(live)} agents at the table")
                break
    except KeyboardInterrupt:
        # Every agent still at the table keeps its account, its traces and its seat,
        # so the experiment can be started again from where it stopped. What ends is
        # the rounds.
        print(f"interrupted; the rounds end here with {len(live)} agents at the table",
              file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
