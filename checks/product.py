"""Experiment-level progress, cost, outcomes, and human-takeover branches."""

from __future__ import annotations

import hashlib
import json
import threading

import experiment
import harness
import product
import providers
from checks.fake import Err, fake, per_agent, run, say
from checks.lanes import episodes_taken, ground_truth, manifest_file, quiet, seated, temp_root


def check_product_controls_are_declared_and_validated():
    """Identifiers, cost boundaries, reveals, and quality tiers survive manifest loading."""
    text = ('experiment_id = "pilot-7"\nsystem_prompt = ""\n'
            '[cost]\nmaximum = 1000\nreserved_completion = 100\nwarning = 700\n'
            'ceiling_policy = "stop"\n[reveal]\nat_completion = ["winners", "scores"]\n'
            'experimenter_only = ["evidence"]\n'
            '[[agent]]\nid = "p1"\nquality_tier = "premium"\n')
    with temp_root() as root:
        loaded = experiment.load_manifest(manifest_file(root, text))
        assert loaded["experiment_id"] == "pilot-7"
        assert loaded["cost"] == {"maximum": 1000, "reserved_completion": 100,
                                  "warning": 700, "ceiling_policy": "stop"}
        assert loaded["reveal"]["at_completion"] == ["winners", "scores"]
        assert loaded["agents"][0]["quality_tier"] == "premium"
        for bad in ('experiment_id = "../bad"\n',
                    '[cost]\nmaximum = 100\nreserved_completion = 100\n',
                    '[cost]\nmaximum = 100\nceiling_policy = "downgrade"\n',
                    '[reveal]\nat_completion = ["narration"]\n'):
            path = manifest_file(root, bad + 'system_prompt = ""\n[[agent]]\nid = "p1"\n',
                                 name=hashlib.sha256(bad.encode()).hexdigest()[:8] + ".toml")
            try:
                experiment.load_manifest(path)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"accepted invalid product controls: {bad!r}")


def check_product_records_cost_progress_and_outcome_from_accounts_and_traces():
    """Human spend is excluded, progress is durable, and outcomes cite trace bytes."""
    with temp_root() as root:
        accounts = {
            "auto": {"provider": "anthropic", "remaining": 70,
                     "episodes": [{"episode": 1, "spent": 30}],
                     "product": {"quality_tier": "premium"}},
            "player": {"provider": "human", "remaining": 90,
                       "episodes": [{"episode": 1, "spent": 999}],
                       "product": {"quality_tier": "interactive"}},
        }
        for agent, account in accounts.items():
            trace = harness.trace_path(agent, 1)
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"agent": agent}), encoding="utf-8")
        load = accounts.__getitem__
        cost = product.cost(accounts, load,
                            {"maximum": 130, "reserved_completion": 100, "warning": 20},
                            providers.is_interactive)
        assert cost["autonomous_spend"] == 30 and cost["warning_reached"]
        assert cost["ceiling_reached"], "the reserve is protected before another round starts"
        product.progress(root, "pilot", "preparing_round", 1)
        product.progress(root, "pilot", "waiting_player", 1, {"agents": ["player"]})
        recorded = product.records(root, "pilot")["progress"]
        assert [event["phase"] for event in recorded["events"]] == ["preparing_round",
                                                                    "waiting_player"]
        assert recorded["latest"] == recorded["events"][-1], recorded
        assert recorded["latest"]["agents"] == ["player"], recorded
        outcome = product.outcome(root, "pilot", ["auto", "player"],
                                  {"1": "A", "2": "P"}, {"player"}, "one_remains", load,
                                  harness.trace_path,
                                  {"at_completion": ["winners"], "experimenter_only": ["evidence"]})
        assert outcome["winners"] == ["P"] and outcome["survivors"] == ["P"]
        assert outcome["scores"] == {"A": 70, "P": 90}
        assert len(outcome["evidence"]) == 2
        assert all(item["trace_sha256"] == hashlib.sha256(
            harness.trace_path(item["agent"], item["episode"]).read_bytes()).hexdigest()
                   for item in outcome["evidence"]), outcome["evidence"]
        public = product.revealed(outcome, "at_completion")
        assert public["winners"] == ["P"] and "scores" not in public
        assert "evidence" not in public, "experimenter-only evidence is not player output"
        assert product.revealed(outcome, "at_completion", experimenter=True)["evidence"]


def check_an_experiment_branch_is_new_lineage_and_leaves_sources_unchanged():
    """Every seat forks at one boundary and the selected new seat becomes interactive."""
    with temp_root() as root:
        source = manifest_file(
            root, 'experiment_id = "source"\nsystem_prompt = ""\n'
                  '[[agent]]\nid = "source-a"\nlabel = "A"\n'
                  '[[agent]]\nid = "source-b"\nlabel = "B"\n', name="source.toml")
        for agent in ("source-a", "source-b"):
            with quiet():
                harness.run_once(agent, fake(say()))
        source_bytes = {path: path.read_bytes() for agent in ("source-a", "source-b")
                        for path in (harness.records_dir(agent) / "account.json",
                                     harness.trace_path(agent, 1))}
        output = root / "experiments" / "branches" / "takeover.toml"
        with quiet():
            actual = experiment.branch_experiment(source, 1, "takeover", "2", output)
        assert actual == output
        branch = experiment.load_manifest(output)
        assert branch["experiment_id"] == "takeover"
        assert [entry["id"] for entry in branch["agents"]] == ["takeover-01", "takeover-02"]
        assert branch["agents"][0]["provider"] == "anthropic"
        assert (branch["agents"][1]["provider"], branch["agents"][1]["model"]) == \
               ("human", "interactive")
        lineage = product.records(root, "takeover")["lineage"]
        assert lineage["source_experiment_id"] == "source" and lineage["source_round"] == 1
        assert len(lineage["traces"]) == 2
        assert all(path.read_bytes() == before for path, before in source_bytes.items())


def check_an_unreadable_record_is_told_from_an_absent_one():
    """read() is None only where there is no file. A progress.json that does not read
    stops the next progress write, is left as it is, and shows as no progress."""
    with temp_root() as root:
        base = product.directory(root, "pilot")
        assert product.read(base / "progress.json") is None, "no file is no record"
        assert product.records(root, "pilot") == {"progress": None, "outcome": None,
                                                  "lineage": None}
        base.mkdir(parents=True)
        for data in (b"{not json", b"[1, 2]", b"\xff\xfe{}", b""):
            (base / "progress.json").write_bytes(data)
            for call in (lambda: product.read(base / "progress.json"),
                         lambda: product.progress(root, "pilot", "preparing_round", 1)):
                try:
                    call()
                except ValueError as e:
                    assert "progress.json" in str(e), str(e)
                else:
                    raise AssertionError(f"an unreadable record was taken for none: {data!r}")
            assert (base / "progress.json").read_bytes() == data, "and it is left as it was"
            assert not (base / "progress.jsonl").exists(), "and no event was written"
            assert product.records(root, "pilot")["progress"] is None, data


def check_progress_appends_each_event_and_reads_its_history_back():
    """Each event is one line appended to progress.jsonl; progress.json holds the header
    and the latest alone. A line a crash cut short costs that line and no other, and a
    progress.json holding its own events starts the log with them."""
    with temp_root() as root:
        base = product.directory(root, "pilot")
        for phase in ("preparing_round", "waiting_autonomous", "round_completed"):
            event = product.progress(root, "pilot", phase, 1, {"agents": ["a"]})
        head = json.loads((base / "progress.json").read_text(encoding="utf-8"))
        assert head == {"version": product.VERSION, "experiment_id": "pilot", "latest": event}, head
        lines = (base / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["phase"] for line in lines] == [
            "preparing_round", "waiting_autonomous", "round_completed"], lines

        with (base / "progress.jsonl").open("ab") as log:
            log.write(b'{"at":"2026-01-01T00:00:00Z","pha')
        product.progress(root, "pilot", "completed", 1)
        recorded = product.records(root, "pilot")["progress"]
        assert [event["phase"] for event in recorded["events"]] == [
            "preparing_round", "waiting_autonomous", "round_completed", "completed"], recorded
        assert recorded["latest"]["phase"] == "completed" and \
            {"version", "experiment_id"} <= set(recorded), recorded

        single = product.directory(root, "single")
        earlier = [{"at": "2026-01-01T00:00:00Z", "phase": phase, "round": 1}
                   for phase in ("preparing_round", "round_completed")]
        product.atomic(single / "progress.json", {"version": product.VERSION,
                                                  "experiment_id": "single",
                                                  "latest": earlier[-1], "events": earlier})
        assert product.records(root, "single")["progress"]["events"] == earlier
        product.progress(root, "single", "preparing_round", 2)
        assert [(event["phase"], event["round"])
                for event in product.records(root, "single")["progress"]["events"]] == [
            ("preparing_round", 1), ("round_completed", 1), ("preparing_round", 2)]
        assert "events" not in json.loads((single / "progress.json").read_text(encoding="utf-8"))


def check_a_round_writes_each_phase_as_it_reaches_it():
    """Progress follows the round as it goes: the round waits on an episode before it
    runs, resolves its actions after it ran and before it settles, and settles it
    before it is committed, so no phase is written once the round has left it.

    g01's transfer reaches g02's account when g01 settles, and not before. In a
    simultaneous round an episode that ends while a peer's still runs leaves the round
    waiting, and the last to end leaves it resolving actions.

    Where a person plays g02, the round waits on a player for g02's episode: a
    simultaneous round names its players apart from its autonomous agents, and g01's
    episode ending while g02's still runs leaves the round waiting on the player.
    """
    ran, committed, waiting = "ran", "committed", "waiting"
    for player in (False, True):
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            if player:
                account = harness.load_account("g02")
                account["provider"], account["model"] = "human", "interactive"
                harness.save_account("g02", account)
            seen: list[tuple] = []
            g01_ended = threading.Event()
            held_back: list[bool] = []

            def progress(phase, round_number, detail):
                def state(agent):
                    if harness.trace_path(agent, round_number).exists():
                        return "committed"
                    return "ran" if harness.raw_path(agent, round_number).exists() else "waiting"
                seen.append((phase, round_number, detail, {agent: state(agent) for agent in ids},
                             ground_truth("g02").get("received", 0)))
                if detail == {"completed_autonomous": "g01"}:
                    g01_ended.set()

            def after_g01(n):
                if threading.current_thread().name == "g02" and n == 1:
                    held_back.append(g01_ended.wait(10))

            live = set(ids)
            with quiet():
                experiment.sequential_round(ids, live, 0,
                                            fake(run("echo '2 250' > out/transfer"), say()),
                                            progress=progress)
                sequential = seen[:]
                seen.clear()
                experiment.simultaneous_round(ids, live, 1, per_agent(default=(say(),),
                                                                      on_request=after_g01),
                                              progress=progress)

        g02_waits = "waiting_player" if player else "waiting_autonomous"
        assert sequential == [
            ("waiting_autonomous", 1, {"agents": ["g01"]}, {"g01": waiting, "g02": waiting}, 0),
            ("resolving_actions", 1, {"agents": ["g01"]}, {"g01": ran, "g02": waiting}, 0),
            ("settling_round", 1, {"agents": ["g01"]}, {"g01": ran, "g02": waiting}, 250),
            (g02_waits, 1, {"agents": ["g02"]}, {"g01": committed, "g02": waiting}, 250),
            ("resolving_actions", 1, {"agents": ["g02"]}, {"g01": committed, "g02": ran}, 250),
            ("settling_round", 1, {"agents": ["g02"]}, {"g01": committed, "g02": ran}, 250)], \
            (player, sequential)
        assert held_back == [True], f"g02's episode ran on after g01's had ended: {player} {held_back}"
        unrun = {"g01": waiting, "g02": waiting}
        started = ([("waiting_autonomous", 2, {"agents": ["g01"]}, unrun, 250),
                    ("waiting_player", 2, {"agents": ["g02"]}, unrun, 250)] if player else
                   [("waiting_autonomous", 2, {"agents": ids}, unrun, 250)])
        ended = [(g02_waits, 2, {"completed_autonomous": "g01"}, {"g01": ran, "g02": waiting}, 250)]
        # g02's episode ending is written only where g02 is autonomous: a player's end
        # finds the round already waiting on it.
        if not player:
            ended.append(("resolving_actions", 2, {"completed_autonomous": "g02"},
                          {"g01": ran, "g02": ran}, 250))
        assert seen == started + ended + [
            ("resolving_actions", 2, {"agents": ids}, {"g01": ran, "g02": ran}, 250),
            ("settling_round", 2, {"agents": ids}, {"g01": ran, "g02": ran}, 250)], (player, seen)


def check_a_progress_record_that_fails_costs_no_episode_its_spend():
    """A progress write that fails between an episode's billing and its commit, or a
    second Ctrl+C that lands in one, is raised once the round's billed episodes are
    committed and an agent whose episode put it out has left the table, never in place
    of either.

    g01's episode ends on an API error after a billed turn, which drops it out. A
    simultaneous round also writes from each autonomous episode's own thread as it ends,
    and a write that fails there alone is raised the same way. A second Ctrl+C lands in
    the main thread, never there.
    """
    sequential = (experiment.sequential_round, lambda: fake(run("echo one"), Err(400)),
                  {"g01": 1, "g02": 0})
    simultaneous = (experiment.simultaneous_round,
                    lambda: per_agent(g01=(run("echo one"), Err(400)), default=(say(),)),
                    {"g01": 1, "g02": 1})
    # Where the write fails: at a phase, or at every event whose detail has this key.
    cases = [(at, failure, played) for at in ("resolving_actions", "settling_round")
             for failure in (ValueError, KeyboardInterrupt) for played in (sequential, simultaneous)]
    cases.append(("completed_autonomous", ValueError, simultaneous))
    for at, failure, (a_round, router, took) in cases:
        def failing(reached, round_number, detail, at=at, failure=failure):
            if at == reached or at in (detail or {}):
                raise failure("progress.json does not read")

        case = (a_round.__name__, at, failure.__name__)
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            live = set(ids)
            raised: BaseException | None = None
            try:
                with quiet():
                    a_round(ids, live, 0, router(), progress=failing)
            except (ValueError, KeyboardInterrupt) as e:
                raised = e
            taken = episodes_taken(ids)
            traced = {agent: harness.trace_path(agent, 1).exists() for agent in ids}
            first = ground_truth("g01")["episodes"][0]
        assert type(raised) is failure and "progress.json" in str(raised), (case, raised)
        assert taken == took, (case, taken)
        assert traced == {agent: bool(n) for agent, n in took.items()}, \
            f"every billed episode is traced: {case} {traced}"
        assert first["stop"] == "api_error" and first["spent"] > 0, (case, first)
        assert live == {"g02"}, f"g01 left the table before the raise: {case} {live}"
