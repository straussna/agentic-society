"""Several agents advancing together, each reading the others' blackboards.

    py -3 experiment.py sandbox -r 20

One round is one episode for each agent; a seat names a blackboard and a balance.
Every run names its manifest: the schedule, the environment, and each agent's terms.
config.toml holds only what is true of every run whatever the experiment."""

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
EXPERIMENT_KEYS = {"schedule", "stop_when_one_remains", "provider", "model", "agent", "channel",
                   "harness_files", "tool"}

# What a manifest may say about one agent. Everything else an agent is comes from
# the experiment's defaults and config.toml.
AGENT_KEYS = {"id", "label", "starter_files", "starter_files_below", "budget", "provider", "model",
              "system_prompt"}

AGENT_TYPES = (("id", str), ("label", str), ("starter_files", str),
               ("starter_files_below", int), ("budget", int), ("provider", str), ("model", str),
               ("system_prompt", str))

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
    """A bare manifest for these agents, which is what a round stamps when driven directly."""
    return {"schedule": schedule, "stop_when_one_remains": False,
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
    harness.validate_terms(str(path), who=agent, provider=entry.get("provider"),
                           model=entry.get("model"), budget=entry.get("budget"),
                           starter_files=entry.get("starter_files"),
                           starter_files_below=entry.get("starter_files_below"))


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


def load_manifest(path: Path) -> dict:
    """Read an experiment manifest: the schedule, the experiment's defaults, and each agent's terms.

    Returns {"schedule", "overrides", "agents", "labels", "channels", "harness_files",
    "tools", "sha256"}. Unknown keys, wrong types, and terms that do not go together
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

    check_keys(refuser(path), "", top, allowed, retired=harness.RETIRED,
               elsewhere={t.lower(): harness.NOT_MANIFEST for t in harness.PROCESS},
               expected="schedule, [[agent]] tables, [[channel]] tables, [[tool]] tables, "
                        "[harness_files], and any experiment setting")
    schedule = top.get("schedule", "sequential")
    if schedule not in SCHEDULES:
        raise SystemExit(f"{path}: schedule must be one of {list(SCHEDULES)}, got {schedule!r}")
    stop_when_one_remains = top.get("stop_when_one_remains", False)
    if type(stop_when_one_remains) is not bool:
        raise SystemExit(f"{path}: stop_when_one_remains must be bool, got {type(stop_when_one_remains).__name__}")
    agents = top.get("agent")
    if not isinstance(agents, list) or not all(isinstance(r, dict) for r in agents):
        raise SystemExit(f"{path}: agents are [[agent]] tables, each with an id")
    for terms in [top, *agents]:
        starter = terms.get("starter_files")
        if isinstance(starter, str) and starter.startswith(("./", "../", ".\\", "..\\")):
            terms["starter_files"] = str((path.parent / starter).resolve())
    default_provider, default_model = top.get("provider"), top.get("model")
    if (default_provider is None) != (default_model is None):
        raise SystemExit(f"{path}: top-level provider and model must be set together")
    for entry in agents:
        if "provider" not in entry and default_provider is not None:
            entry["provider"] = default_provider
        if "model" not in entry and default_model is not None:
            entry["model"] = default_model
        if "provider" not in entry or "model" not in entry:
            raise SystemExit(f"{path}: {entry.get('id', 'agent')}: provider and model must resolve "
                             "from top-level defaults or the [[agent]] table")
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
    if "system_prompt" not in top and not all("system_prompt" in e for e in agents):
        raise SystemExit(f"{path}: declare system_prompt, at the top level or on every "
                         f"[[agent]]. What an agent is told is the experiment's and reaches "
                         f'every request, so it is stated and never inherited; system_prompt = ""'
                         f" is the declaration that says nothing")

    overrides = {k: v for k, v in top.items()
                 if k not in EXPERIMENT_KEYS}
    return {"schedule": schedule, "stop_when_one_remains": stop_when_one_remains,
            "overrides": overrides, "agents": agents, "labels": labels,
            "channels": tables, "harness_files": harness_files, "tools": tool_tables,
            "sha256": hashlib.sha256(data).hexdigest()}


def terms_of(entry: dict) -> dict[str, Any]:
    """One agent's pinned settings as load_account's keywords, None where the manifest is silent."""
    return {k: entry.get(k) for k in ("provider", "model", "budget", "starter_files", "starter_files_below",
                                      "system_prompt")}


def stamp_of(manifest: dict) -> dict[str, Any]:
    """What every episode records about how the experiment was driven."""
    return {"schedule": manifest["schedule"],
            "stop_when_one_remains": manifest["stop_when_one_remains"],
            "manifest_sha256": manifest["sha256"]}


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
               stamp: dict[str, Any], labels: dict[str, str], stop_when_one_remains: bool) -> bool:
    """One round: drop the agents that cannot act, name the round, run it, and say
    whether the rounds go on.

    The rounds end when every agent is out, when the manifest stops after one
    agent remains, or when nobody seated could be given an environment to start.
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
    if stop_when_one_remains and len(live) == 1:
        print(f"{next(iter(live))} is the only agent left with anything to spend; the competition ends")
        return False
    # The order decides who acts on this round's information and who on last
    # round's, so it is on screen beside the round number. Under a simultaneous
    # round nobody acts on this round's, and the seats are named in their own order.
    ordered = agents if a_round is simultaneous_round else order(agents, rnd)
    acting = [agent for agent in ordered if agent in live]
    print(f"--- round {rnd + 1} ({' '.join(acting)}) ---")

    if not a_round(agents, live, rnd, create, stamp, labels):
        print(f"no agent could take an episode in round {rnd + 1}; "
              f"the rounds end here with {len(live)} agents at the table")
        return False
    return True


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
                    help="use MODEL for every seat together with --provider; both must match "
                         "existing agent accounts")
    ap.add_argument("-c", "--config", type=Path, help="default: config.toml beside harness.py")
    a = ap.parse_args(argv)

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
            if not play_round(a_round, agents, live, rnd, router, stamp, manifest["labels"],
                              manifest["stop_when_one_remains"]):
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
