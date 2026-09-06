"""Several agents advancing together, each reading the others' blackboards.

    py -3 experiment.py --agents g01 g02 g03 --rounds 20
    py -3 experiment.py --manifest experiments/example.toml --rounds 20

One round is one episode for each agent; a seat names a blackboard and a
balance. A manifest gives each agent its own terms and the experiment its schedule."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any, Callable, TypeVar

import harness
from harness import check_keys

T = TypeVar("T")

# A seat is named by its number, and a label defaults to the seat number, so a
# bare number is what a seat is called in every path and file the agents see.
# An agent id must be something else.
BARE_NUMBER = re.compile(r"^\d+$")

# Goes at building an environment before the agent drops out. An environment is
# built from containers and copies of other agents' trees, so a failure can be
# the daemon and not the agent. None of it is billed, so another go spends only time.
ATTEMPTS = 2

# How a round is driven. "sequential": one episode at a time, the starting agent
# moving each round, each episode reading what the ones before it in the round
# wrote. "simultaneous": every environment built before any episode runs, the
# episodes run at once, and the results settle in seat order, so nobody reads
# this round's writes and a transfer made in one round is seen at the next.
SCHEDULES = ("sequential", "simultaneous")

# What a manifest may say about one agent. Everything else an agent is comes from
# the experiment's defaults and config.toml.
AGENT_KEYS = {"id", "label", "starter_files", "starter_files_below", "budget", "model"}

AGENT_TYPES = (("id", str), ("label", str), ("starter_files", str),
               ("starter_files_below", int), ("budget", int), ("model", str))

# What an agent may be called to its peers: one path segment, since it lands in
# paths and file names. The default is the seat number.
LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


def seats_of(agents: list[str]) -> dict[str, str]:
    """Seat -> agent, for the whole experiment.

    Absolute and complete: a seat means the same agent to every reader, so a note
    citing one resolves the same way for all of them.
    """
    return {str(i): agent for i, agent in enumerate(agents, 1)}


def order(agents: list[str], rnd: int) -> list[str]:
    """The agents, rotated by the round, so no seat has a standing advantage."""
    i = rnd % len(agents)
    return agents[i:] + agents[:i]


# --- the manifest -------------------------------------------------------------


def shorthand(ids: list[str], schedule: str = "sequential") -> dict:
    """The manifest `--agents` stands for: these agents, on config.toml, in rotation."""
    return {"schedule": schedule, "overrides": {}, "agents": [{"id": i} for i in ids],
            "labels": {str(n): str(n) for n in range(1, len(ids) + 1)},
            "channels": None, "harness_files": None, "sha256": ""}


def check_ids(ids: list[str], where: str) -> None:
    """Refuse an experiment of fewer than two agents, a repeated id, or an id that is a bare number."""
    if len(ids) < 2:
        raise SystemExit(f"{where}: an experiment needs two or more agents; one agent has no peers")
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
    harness.validate_terms(str(path), who=agent, model=entry.get("model"), budget=entry.get("budget"),
                           starter_files=entry.get("starter_files"),
                           starter_files_below=entry.get("starter_files_below"))


def load_manifest(path: Path) -> dict:
    """Read an experiment manifest: the schedule, the experiment's defaults, and each agent's terms.

    Returns {"schedule", "overrides", "agents", "labels", "channels", "harness_files",
    "sha256"}. Unknown keys, wrong types, and terms that do not go together exit
    with a message naming the file, the way load_config does. The overrides' own
    types and ranges are checked when start() applies them, so one set of rules
    holds for both.
    """
    if not path.exists():
        raise SystemExit(f"{path}: no such manifest")
    data = path.read_bytes()
    try:
        top = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"{path}: not a manifest: {e}") from None

    allowed = {"schedule", "agent", "channel", "harness_files"} | {t.lower() for t in harness.TUNABLES}
    check_keys(refuser(path), "", top, allowed, retired=harness.RETIRED,
               expected="schedule, [[agent]] tables, [[channel]] tables, [harness_files], "
                        "and any config.toml key")
    schedule = top.get("schedule", "sequential")
    if schedule not in SCHEDULES:
        raise SystemExit(f"{path}: schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    agents = top.get("agent")
    if not isinstance(agents, list) or not all(isinstance(r, dict) for r in agents):
        raise SystemExit(f"{path}: agents are [[agent]] tables, each with an id")
    for entry in agents:
        check_agent(path, entry)
    check_ids([entry["id"] for entry in agents], str(path))

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

    tables, harness_files = top.get("channel"), top.get("harness_files")
    if tables is not None or harness_files is not None:
        harness.validate_channels(tables, harness_files, str(path), tuple(labels.values()))

    overrides = {k: v for k, v in top.items()
                 if k not in ("schedule", "agent", "channel", "harness_files")}
    return {"schedule": schedule, "overrides": overrides, "agents": agents, "labels": labels,
            "channels": tables, "harness_files": harness_files,
            "sha256": hashlib.sha256(data).hexdigest()}


def terms_of(entry: dict) -> dict[str, Any]:
    """One agent's pinned settings as load_account's keywords, None where the manifest is silent."""
    return {k: entry.get(k) for k in ("model", "budget", "starter_files", "starter_files_below")}


def stamp_of(manifest: dict) -> dict[str, str]:
    """What every episode of the experiment records about how it was driven."""
    return {"schedule": manifest["schedule"], "manifest_sha256": manifest["sha256"]}


# --- rounds -------------------------------------------------------------------


def preparer(agent: str, seats: dict[str, str], stamp: dict[str, str],
             labels: dict[str, str] | None = None) -> Callable[[dict], None]:
    """What an agent's account is told before each episode: where it sits, what it and
    every other agent is called, and how the experiment is being driven. Nothing is
    copied into anything the agent can write, so there is nothing to revert afterwards."""
    named = {seat: (labels or {}).get(seat, seat) for seat in seats}

    def prepare(account: dict) -> None:
        seat = next(s for s, a in seats.items() if a == agent)
        account["seat"] = seat
        account["label"] = named[seat]
        account["peers"] = {"seen": seats, "labels": named}
        account["experiment"] = dict(stamp)
    return prepare


def preparers(agents: list[str], stamp: dict[str, str] | None, labels: dict[str, str] | None,
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


def sequential_round(agents: list[str], live: set[str], rnd: int, create: Callable,
                     stamp: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> bool:
    """One episode for each agent still in the experiment, in rotated order.

    Returns whether any of them took one; a round where none did moved nothing.
    """
    prepare = preparers(agents, stamp, labels)
    acted = False
    for agent in order(agents, rnd):
        if harness.STOPPING:
            # A stop that landed while no episode was in flight ends the rounds the
            # way one that landed inside an episode does: main() answers it, and
            # no environment is built for an agent that will not run.
            raise KeyboardInterrupt
        if agent not in live:
            continue
        ep = attempted(agent, live, lambda: harness.ready(agent, prepare(agent)))
        if agent not in live:
            continue
        if ep is None:
            drop_out(agent, live)
            continue
        trace = harness.commit_episode(ep, harness.run_episode(ep, create))
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


def simultaneous_round(agents: list[str], live: set[str], rnd: int, create: Callable,
                       stamp: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> bool:
    """One episode for each agent still in the experiment, all at once.

    Every environment is built first, so no episode reads this round's writes. The
    episodes run in threads. Then each settles in seat order, the transfers they
    made are credited, and each closes in seat order: a credit lands after the
    receiver's own turns and before its floor, so an agent is out on the round's
    net and never lifted back by a transfer that had already arrived. Returns
    whether any agent took an episode.
    """
    built = build_all(agents, live, preparers(agents, stamp, labels, "simultaneous"))
    if harness.STOPPING:
        for ep in built.values():
            ep.abandon()
        raise KeyboardInterrupt
    if not built:
        return False

    outs: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def go(agent: str, ep: harness.Episode) -> None:
        try:
            outs[agent] = harness.run_episode(ep, create)
        except BaseException as e:      # run_episode has saved and reaped on its way out
            errors[agent] = e

    threads = [threading.Thread(target=go, args=(agent, ep), name=agent, daemon=True)
               for agent, ep in built.items()]
    try:
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


def play_round(a_round: Callable, agents: list[str], live: set[str], rnd: int, create: Callable,
               stamp: dict[str, str], labels: dict[str, str]) -> bool:
    """One round: drop the agents that cannot act, name the round, run it, and say
    whether the rounds go on.

    The rounds end when every agent is out, when one agent is left holding a
    balance (it takes one more episode, owing no transfer and no message, there
    being nobody left to make either to), or when nobody seated could be given an
    environment to start in.
    """
    # Asked before the round, so the header names who will act. Between here and
    # an agent's own turn its balance can only move up, a peer's transfer being
    # the only thing that reaches it, so this is the answer its episode would give.
    for agent in order(agents, rnd):
        if agent in live and harness.why_out(harness.load_account(agent)):
            drop_out(agent, live)
    if not live:
        print(f"every agent is out after {rnd} rounds")
        return False
    # The order decides who acts on this round's information and who on last
    # round's, so it is on screen beside the round number. Under a simultaneous
    # round nobody acts on this round's, and the seats are named in their own order.
    ordered = agents if a_round is simultaneous_round else order(agents, rnd)
    acting = [agent for agent in ordered if agent in live]
    print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")
    if len(live) == 1:
        a_round(agents, live, rnd, create, stamp, labels)
        print(f"{acting[0]} is the only agent left with anything to spend; the rounds end here")
        return False
    if not a_round(agents, live, rnd, create, stamp, labels):
        print(f"no agent could take an episode in round {rnd + 1}; "
              f"the rounds end here with {len(live)} agents at the table")
        return False
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
        try:
            check_ids(a.agents, "--agents")
        except SystemExit as e:
            ap.error(str(e))
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
    # Every agent is created before the first round, so the first to act finds its
    # peers' blackboards in place. Each is created on its own terms, and one that
    # exists must have been created on the same.
    for entry in manifest["agents"]:
        harness.load_account(entry["id"], **terms_of(entry))
    print(f"experiment: {', '.join(agents)}  ({len(agents)} agents, up to {a.rounds} rounds, "
          f"{manifest['schedule']})")
    try:
        for rnd in range(a.rounds):
            if not play_round(a_round, agents, live, rnd, create, stamp, manifest["labels"]):
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
