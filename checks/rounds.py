"""Experiments: rotation, retries, interrupts, manifests, and simultaneous rounds."""

from __future__ import annotations

import hashlib
import re
import subprocess
import threading
import experiment
import harness

from checks.fake import DEFAULT, Err, fake, per_agent, run, say, stopping_at
from checks.lanes import (
    ALL_OWED,
    HostBox,
    PERSONA_LABELS,
    RecordingBox,
    agent_of,
    digest_name,
    elements_of,
    episodes_taken,
    ground_truth,
    manifest_file,
    plant,
    put_out,
    quiet,
    recording,
    refused,
    rooted,
    seated,
    seats_manifest,
    temp_root,
    trace_on_disk,
    turn_cost,
)


def check_the_experiment_rotates_and_validates():
    """Order rotates by round, and an experiment with no manifest or no rounds is refused."""
    ids = ["g01", "g02", "g03"]
    assert [experiment.order(ids, r) for r in range(4)] == [
        ["g01", "g02", "g03"], ["g02", "g03", "g01"],
        ["g03", "g01", "g02"], ["g01", "g02", "g03"]], "a fixed order is a standing advantage"
    for bad in ([],                                        # every run names its experiment
                ["--rounds", "5"],                         # including this one
                ["--manifest", "c.toml", "--rounds", "0"]):
        with quiet():
            refused(lambda: experiment.main(bad), code=2, because=f"accepted bad experiment: {bad}")
    with quiet():
        refused(lambda: experiment.main(["--manifest", "no-such-experiment.toml"]), "no such manifest",
                because="a manifest that is not there was accepted")


def check_an_experiment_gives_a_failed_environment_one_more_go():
    """An agent whose environment will not build sits out one attempt, not the experiment.

    Building an environment reads other agents' trees and asks Docker for a
    container, so a failure can be the daemon and not the agent. None of it is
    billed, and the round retries the build alone.
    """
    with temp_root() as root:
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
            experiment.sequential_round(ids, live, 0, fake(*DEFAULT))
        took = episodes_taken(ids)

    assert live == {"g01", "g02"}, f"only the agent that failed twice is out: {live}"
    assert took == {"g01": 1, "g02": 1, "g03": 0}, took
    assert "could not build an environment" in buf.getvalue(), buf.getvalue()
    assert "could not start an episode container" not in buf.getvalue(), \
        "the container started; it is the environment that did not"
    assert buf.getvalue().count(f"(1 of {experiment.ATTEMPTS})") == 2, buf.getvalue()


def check_an_interrupt_ends_the_whole_experiment():
    """Ctrl+C ends every remaining round; a fault ends one agent's part in them.

    An interrupt is the experimenter: the agent whose episode it landed in keeps
    its seat and the rounds stop. A fault is the agent, and only that agent drops out.
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
        took = episodes_taken(ids)
        first = ground_truth("g01")["episodes"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no agent is ejected for it: {sorted(live)}"

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            experiment.sequential_round(ids, live, 0, fake(run("echo one"), Err(400)))
        took = episodes_taken(ids)
    assert live == {"g02", "g03"}, sorted(live)
    assert took == {"g01": 1, "g02": 1, "g03": 1}, "the rest of the experiment takes its round"

    # And main answers an interrupt by ending the rounds, not the round.
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(run("echo one"), KeyboardInterrupt())
        with quiet() as buf:
            code = experiment.main(["--manifest", str(seats_manifest(root, ids)), "--rounds", "5"])
        took = episodes_taken(ids)
    assert code == 130, code
    assert took == {"g01": 1, "g02": 0, "g03": 0}, "no round after the one it landed in"
    assert "interrupted" in buf.getvalue(), buf.getvalue()


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
        took = episodes_taken(ids)
        first = ground_truth("g01")["episodes"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no agent is ejected for it: {sorted(live)}"


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
            code = experiment.main(["--manifest", str(seats_manifest(root, ids)), "--rounds", "5"])
        took = episodes_taken(ids)
        rested = {r: harness.load_account(r)["remaining"] for r in ids}
    assert code == 0, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, \
        f"one episode each, then nothing left to ask for: {took}"
    assert set(rested.values()) == {0}, rested
    assert buf.getvalue().count("drops out: nothing left to spend") == 3, buf.getvalue()
    assert "every agent is out after 1 rounds" in buf.getvalue(), buf.getvalue()


def check_a_sole_agent_runs_requested_rounds_unless_the_manifest_stops_at_a_winner():
    """The manifest decides whether a lone funded seat continues or ends the experiment."""
    with temp_root(channels=ALL_OWED) as root:
        ids = seated(root, "g01", g02={}, g03={})
        put_out("g02")
        put_out("g03")
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(["--manifest", str(seats_manifest(root, ids)), "--rounds", "5"])
        took = episodes_taken(ids)
        alone = harness.load_account("g01")["episodes"][-1]
    assert code == 0, code
    assert took == {"g01": 5, "g02": 0, "g03": 0}, took
    assert "competition ends" not in buf.getvalue(), buf.getvalue()
    assert alone["transfer"]["penalty"] == 0, alone["transfer"]
    assert alone["channels"]["mail"]["penalty"] == 0, alone["channels"]["mail"]
    assert not alone["channels"]["blackboard"]["posted"] and alone["channels"]["blackboard"]["penalty"] > 0

    with temp_root(channels=ALL_OWED) as root:
        ids = seated(root, "g01", g02={}, g03={})
        put_out("g02")
        put_out("g03")
        manifest = seats_manifest(root, ids)
        manifest.write_text('stop_when_one_remains = true\n' + manifest.read_text(encoding="utf-8"),
                            encoding="utf-8", newline="\n")
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(["--manifest", str(manifest), "--rounds", "5"])
        took = episodes_taken(ids)
    assert code == 0, code
    assert took == {"g01": 0, "g02": 0, "g03": 0}, took
    assert "g01 is the only agent left with anything to spend; the competition ends" in buf.getvalue(), buf.getvalue()

def check_a_manifest_is_validated():
    """A manifest names a schedule, the experiment's defaults, and each agent's terms, or is refused."""
    other_model = "claude-opus-5"
    good = (f'schedule = "simultaneous"\ngrace_episodes = 1\nsystem_prompt = ""\n'
            f'[[agent]]\nid = "g01"\nstarter_files = "s"\nstarter_files_below = 400000\n'
            f'[[agent]]\nid = "g02"\nbudget = 7\nmodel = "{other_model}"\n')
    with rooted(HostBox) as root:
        plant(root, "s")
        two = '[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n'
        for bad in ('colour = "red"\n' + two,                    # an unknown key
                    'schedule = "random"\n' + two,               # an unknown schedule
                    '[[agent]]\nid = "g01"\n[[agent]]\nid = "g01"\n',  # an agent twice
                    'image = "x"\n' + two,                       # config.toml's, not an experiment's
                    'max_turns = 5\n' + two,                     # and so is this
                    '[[agent]]\nid = "1"\n[[agent]]\nid = "g02"\n',  # a bare number is a seat
                    '[[agent]]\nid = "g01"\nstarter_files = "s"\n[[agent]]\nid = "g02"\n',       # starter_files alone
                    '[[agent]]\nid = "g01"\nstarter_files = "nope"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nbudget = 0\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nbudget = "many"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\nmodel = "no-such"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nid = "g01"\ncolour = "red"\n[[agent]]\nid = "g02"\n',
                    '[[agent]]\nstarter_files = "s"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',  # no id
                    'agent = 5\n',                                 # agents are tables
                    two,                                       # no system_prompt
                    'not toml ==\n'):
            p = manifest_file(root, bad)
            refused(lambda: experiment.load_manifest(p), str(p), because=f"accepted bad manifest: {bad!r}")
        refused(lambda: experiment.load_manifest(root / "experiments" / "missing.toml"),
                because="a missing manifest was ignored")

        p = manifest_file(root, good)
        m = experiment.load_manifest(p)
        expected_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert m["schedule"] == "simultaneous"
    want = {"grace_episodes": 1, "system_prompt": ""}
    assert m["overrides"] == want, "everything else is an experiment default"
    assert [e["id"] for e in m["agents"]] == ["g01", "g02"]
    assert m["sha256"] == expected_sha
    assert experiment.terms_of(m["agents"][0]) == {"provider": "anthropic", "model": "claude-sonnet-5", "budget": None, "starter_files": "s",
                                                   "starter_files_below": 400000,
                                                   "system_prompt": None}
    assert experiment.terms_of(m["agents"][1]) == {"provider": "anthropic", "model": other_model, "budget": 7, "starter_files": None,
                                                   "starter_files_below": None, "system_prompt": None}
    short = experiment.shorthand(["a", "b"])
    assert short["schedule"] == "sequential" and short["overrides"] == {} and short["sha256"] == ""
    assert [e["id"] for e in short["agents"]] == ["a", "b"]
    assert experiment.stamp_of(m) == {"schedule": "simultaneous", "stop_when_one_remains": False,
                                      "manifest_sha256": m["sha256"]}


def check_a_manifest_gives_each_agent_its_own_starter_files():
    """Each agent is created on its own terms, the experiment's defaults apply to all, and the
    terms are pinned: a manifest that later says otherwise is refused."""
    text = ('grace_episodes = 2\nsystem_prompt = ""\n'
            '[[agent]]\nid = "g01"\nstarter_files = "a"\nstarter_files_below = 500000\n'
            '[[agent]]\nid = "g02"\nstarter_files = "b"\nstarter_files_below = 600000\nbudget = 600000\n'
            '[[agent]]\nid = "g03"\n')
    with temp_root() as root:
        plant(root, "a")
        plant(root, "b", m1="bravo\n")
        p = manifest_file(root, text)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        asked = []

        def start(config=None, overrides=None, requirements=(), **kw):
            asked.append((overrides, set(requirements)))
            harness.apply_config(overrides or {}, "manifest")
            return fake(*DEFAULT)

        harness.start = start
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "1"])
        accounts = {r: ground_truth(r) for r in ("g01", "g02", "g03")}
        starter = {r: (harness.mirror(r, "notes") / "m1").read_text(encoding="utf-8")
                   if (harness.mirror(r, "notes") / "m1").exists() else None for r in accounts}
        traces = {r: trace_on_disk(r, 1) for r in accounts}

        p.write_text(p.read_text(encoding="utf-8").replace('starter_files = "a"',
                                                           'starter_files = "b"'),
                     encoding="utf-8", newline="\n")
        with quiet():
            refused(lambda: experiment.main(["--manifest", str(p), "--rounds", "1"]), "starter_files",
                    because="an agent was re-created on different terms")
    assert code == 0, buf.getvalue()
    assert asked[0] == ({"grace_episodes": 2, "system_prompt": ""},
                        {("anthropic", "claude-sonnet-5")}) and len(asked) == 2, asked
    assert {r: m["starter_files"] for r, m in accounts.items()} == {"g01": "a", "g02": "b", "g03": ""}
    assert accounts["g02"]["initial"] == 600000 and accounts["g01"]["initial"] == harness.BUDGET
    assert starter == {"g01": "alpha\n", "g02": "bravo\n", "g03": None}, starter
    for r, t in traces.items():
        assert t["provenance"]["starter_files"] == accounts[r]["starter_files"], (r, t["provenance"])
        assert t["provenance"]["schedule"] == "sequential"
        assert t["provenance"]["manifest_sha256"] == digest
        assert t["provenance"]["grace_episodes"] == 2, "the experiment's default reached every agent"
    assert "sequential" in buf.getvalue(), buf.getvalue()


def check_a_simultaneous_round_builds_every_environment_before_any_episode_runs():
    """Under a simultaneous schedule every container is up and loaded before the first API call."""
    with recording() as events, temp_root(BOX=RecordingBox) as root:
        ids = seated(root, "g01", g02={}, g03={})
        def opened():
            RecordingBox.note("create", threading.current_thread().name)
        create = per_agent(default=DEFAULT, on_open=opened)

        live = set(ids)
        with quiet():
            acted = experiment.simultaneous_round(ids, live, 0, create)
        took = episodes_taken(ids)
    first = next(i for i, (what, _) in enumerate(events) if what == "create")
    before = [what for what, _ in events[:first]]
    assert before.count("start") == before.count("load") == 3 and set(before) == {"start", "load"}, \
        f"every environment is built before any episode runs: {events}"
    assert [agent_of(n) for what, n in events if what == "start"] == ids, "in seat order"
    assert sorted(agent_of(n) for what, n in events if what == "close") == ids, "and every one reaped"
    assert acted and took == {"g01": 1, "g02": 1, "g03": 1} and live == set(ids), (took, live)


def check_a_simultaneous_round_runs_its_episodes_at_once():
    """The episodes of a simultaneous round are in flight together, not one after another."""
    gate = threading.Barrier(3, timeout=10)
    def requested(_):
        gate.wait()
    create = per_agent(default=(say(),), on_request=requested)

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet():
            experiment.simultaneous_round(ids, live, 0, create)
        stops = {r: harness.load_account(r)["episodes"][0]["stop"] for r in ids}
    assert set(stops.values()) == {"no_tool_call"}, \
        f"an episode that waited alone would have erred instead: {stops}"


def check_a_simultaneous_round_reads_last_round_and_not_this_one():
    """Nobody in a simultaneous round reads what the round writes; all of it arrives at the next.

    A transfer made in the round is credited to its receiver in the same round, after
    the receiver's own turns, and shows in g and n at the next episode.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={})
        first = per_agent(g01=(run("echo hello > 1/RESULT", "echo psst > out/2",
                                   "echo '2 250' > out/transfer"), say()),
                          g02=(run(f"cat {digest_name()}", "ls 1"), say()))
        live = set(ids)
        with quiet():
            experiment.simultaneous_round(ids, live, 0, first)
        g02_first = ground_truth("g02")["episodes"][0]
        said, listed = (c["result"] for c in trace_on_disk("g02", 1)["turns"][0]["tools"])
        # g01 withdraws the line, so the transfer is made once and not every round it stands.
        second = per_agent(g01=(run("rm out/transfer", f"cat {digest_name()}"), say()),
                           g02=(run(f"cat {digest_name()}"), say()))
        with quiet():
            experiment.simultaneous_round(ids, live, 1, second)
        later = trace_on_disk("g02", 2)
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
                experiment.simultaneous_round(ids, live, 0, per_agent(g01=(run(transfer), say()),
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
        def requested(turn):
            if turn == 1:
                gate.wait()
                harness.STOPPING = True
        return per_agent(default=(run("echo one"), run("echo two"), say()),
                         on_request=requested)

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
        p = manifest_file(root, 'schedule = "simultaneous"\nsystem_prompt = ""\n'
                          + "".join(f'[[agent]]\nid = "{r}"\n' for r in ids))
        harness.start = lambda config=None, **kw: stopping_create()
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "5"])
        took = episodes_taken(ids)
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
                                          per_agent(default=(run("echo one"), run("echo two"), say())))
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "nothing was echoed"
    assert all(any(ln.startswith(r) for r in ids) for ln in lines), \
        f"a line that does not say whose it is: {[ln for ln in lines if not any(ln.startswith(r) for r in ids)]}"
    summary = [i for i, ln in enumerate(lines) if re.match(r"^g0\d\s+ep1\s", ln)]
    assert [lines[i].split()[0] for i in summary] == ids, "summaries in seat order"
    assert summary and summary[0] > max(i for i, ln in enumerate(lines) if "| " in ln), \
        "and after every echoed line"


def check_a_simultaneous_round_drops_an_agent_whose_environment_fails_twice_and_seats_the_rest():
    """An environment that will not build costs its agent one attempt, then its seat; the rest run."""
    with recording() as events, temp_root(BOX=RecordingBox) as root:
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
            experiment.simultaneous_round(ids, live, 0, per_agent(default=DEFAULT))
        took = episodes_taken(ids)
    assert live == {"g01", "g02"}, f"only the agent that failed twice is out: {live}"
    assert took == {"g01": 1, "g02": 1, "g03": 0}, took
    assert buf.getvalue().count(f"(1 of {experiment.ATTEMPTS})") == 2, buf.getvalue()
    starts = [n for what, n in events if what == "start"]
    closes = [n for what, n in events if what == "close"]
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

    calls = []
    with recording() as events, temp_root(BOX=StoppingBox) as root:
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
        took = episodes_taken(ids)
    assert not calls, "no episode was started"
    assert took == {"g01": 0, "g02": 0, "g03": 0}, took
    starts = [n for what, n in events if what == "start"]
    closes = [n for what, n in events if what == "close"]
    assert len(starts) == 2 and sorted(starts) == sorted(closes), events
    assert live == set(ids)


def check_labels_are_validated():
    """A label is letters, digits, '.', '_' and '-', distinct after defaults, and not a path."""
    def two(a: str = "", b: str = "") -> str:
        return f'system_prompt = ""\n[[agent]]\nid = "g01"\n{a}[[agent]]\nid = "g02"\n{b}'

    with temp_root() as root:
        for bad in (two('label = "no space"\n'), two('label = "same"\n', 'label = "same"\n'),
                    two('label = "2"\n'),                      # the other agent's default
                    two('label = 3\n'), two('label = ""\n'), two('label = ".."\n')):
            refused(lambda: experiment.load_manifest(manifest_file(root, bad)), "label",
                    because=f"accepted a bad label: {bad!r}")
        m = experiment.load_manifest(manifest_file(root, two('label = "Studio"\n', 'label = "Game"\n')))
        assert m["labels"] == PERSONA_LABELS, m["labels"]
        assert experiment.load_manifest(manifest_file(root, two()))["labels"] == {"1": "1", "2": "2"}
        assert experiment.shorthand(["a", "b"])["labels"] == {"1": "1", "2": "2"}


def check_a_manifest_table_replaces_the_whole_set():
    """A manifest's [[channel]] tables are the whole table; its [harness_files] overlay one key at a time."""
    agents = '[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n'
    with temp_root() as root:
        p = manifest_file(root, 'schedule = "sequential"\nsystem_prompt = ""\n'
                                '[[channel]]\nname = "notes"\nwriter = "self"\n'
                                'readers = "self"\npath = "state"\n' + agents)
        m = experiment.load_manifest(p)
        table, hf = harness.validate_channels(m["channels"], m["harness_files"], str(p),
                                              tuple(m["labels"].values()))
        assert [c.name for c in table] == ["notes"], "one table declared, one channel in force"
        assert hf == {"balance": "n", "digest": "m"}, "no [harness_files], no change"
        assert [c.name for c in harness.channels()] == ["notes", "blackboard", "mail", "transfer"], \
            "loading a manifest sets nothing; start() does"

        p = manifest_file(root, 'system_prompt = ""\n[harness_files]\ndigest = ""\n' + agents,
                          name="d.toml")
        m = experiment.load_manifest(p)
        assert m["channels"] is None, "no [[channel]], and the table in force stays"
        table, hf = harness.validate_channels(m["channels"], m["harness_files"], str(p))
        assert len(table) == 4 and hf == {"balance": "n", "digest": ""}, (table, hf)


def check_a_manifest_declares_what_the_harness_says():
    """Invariant 2: a declared prompt reaches the request, the account and the trace.

    Every manifest declares one, at the top level or on each seat: what an agent is
    told is stated and never inherited, and "" is the declaration that says nothing.
    The experiment's own is what a seat is told unless the seat declares its own, and
    the prompt is pinned like the other four settings: an agent asked to run on a
    different one is refused, episodes either side of it not being one experiment.
    """
    mine, ours = "You are studio 1.", "You are one of several studios."
    text = ('system_prompt = "' + ours + '"\n'
            '[[agent]]\nid = "g01"\nsystem_prompt = "' + mine + '"\n'
            '[[agent]]\nid = "g02"\n')
    with temp_root() as root:
        # Declared and never inherited: a manifest silent on it is refused, and the
        # empty string is the declaration that says nothing.
        silent = manifest_file(root, '[[agent]]\nid = "g01"\n', name="s.toml")
        refused(lambda: experiment.load_manifest(silent), "system_prompt",
                because="a manifest declaring no prompt was accepted")
        empty = manifest_file(root, 'system_prompt = ""\n[[agent]]\nid = "g01"\n', name="e.toml")
        assert experiment.load_manifest(empty)["overrides"] == {"system_prompt": ""}

        p = manifest_file(root, text)
        seen = []

        def start(config=None, overrides=None, requirements=(), **kw):
            harness.apply_config(overrides or {}, "manifest")
            return fake(*DEFAULT, seen=seen)

        harness.start = start
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "1"])
        sent = {r["system"] for r in seen if r.get("kind") == "session"}
        accounts = {r: ground_truth(r) for r in ("g01", "g02")}
        traces = {r: trace_on_disk(r, 1) for r in accounts}

        # A seat's own declaration is one of its pinned settings.
        original = p.read_text(encoding="utf-8")
        p.write_text(original.replace(mine, mine + " Again."), encoding="utf-8", newline="\n")
        with quiet():
            refused(lambda: experiment.main(["--manifest", str(p), "--rounds", "1"]), "system_prompt",
                    because="an agent was re-created on a different system prompt")
        # The experiment's default is read at creation like every other setting, so a
        # seat that took it keeps what it was told and a later manifest does not resay it.
        p.write_text(original.replace(ours, ours + " Again."), encoding="utf-8", newline="\n")
        with quiet():
            assert experiment.main(["--manifest", str(p), "--rounds", "1"]) == 0
        kept = ground_truth("g02")["system_prompt"]
    assert code == 0, buf.getvalue()
    assert sent == {mine, ours}, "each seat was told what it declared"
    assert harness.SYSTEM not in sent, "and nothing was told the shipped default"
    assert {r: a["system_prompt"] for r, a in accounts.items()} == {"g01": mine, "g02": ours}
    assert kept == ours, "the account is what the agent is told, not the manifest of the day"
    for r, t in traces.items():
        want = accounts[r]["system_prompt"]
        assert t["provenance"]["system"] == want, (r, t["provenance"]["system"])
        assert t["provenance"]["system_sha256"] == t["system_sha256"] == harness.system_sha256(want)
