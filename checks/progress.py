"""Experiment-level progress, cost, outcomes, and human-takeover branches."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path

import experiment
import harness
import progress
import providers
from checks.fake import Err, fake, per_agent, run, say, use
from checks.lanes import (
    channel_toml,
    episodes_taken,
    ground_truth,
    manifest_file,
    quiet,
    reconciled,
    rejected,
    seated,
    seats_manifest,
    tables,
    temp_root,
    turn_cost,
)


def check_product_controls_are_declared_and_validated():
    """Identifiers, cost boundaries, reveals, and quality tiers survive manifest loading."""
    text = (
        'experiment_id = "pilot-7"\nsystem_prompt = ""\n'
        "[cost]\nmaximum = 1000\nreserved_completion = 100\nwarning = 700\n"
        'ceiling_policy = "stop"\n[reveal]\nat_completion = ["winners", "resources"]\n'
        'experimenter_only = ["evidence"]\n'
        '[[agent]]\nid = "p1"\nquality_tier = "premium"\n'
    )
    with temp_root() as root:
        loaded = experiment.load_manifest(manifest_file(root, text))
        assert loaded["experiment_id"] == "pilot-7"
        assert loaded["cost"] == {
            "maximum": 1000,
            "reserved_completion": 100,
            "warning": 700,
            "ceiling_policy": "stop",
        }
        assert loaded["reveal"]["at_completion"] == ["winners", "resources"]
        assert loaded["agents"][0]["quality_tier"] == "premium"
        for bad in (
            'experiment_id = "../bad"\n',
            "[cost]\nmaximum = 100\nreserved_completion = 100\n",
            '[cost]\nmaximum = 100\nceiling_policy = "downgrade"\n',
            '[reveal]\nat_completion = ["narration"]\n',
        ):
            path = manifest_file(
                root,
                bad + 'system_prompt = ""\n[[agent]]\nid = "p1"\n',
                name=hashlib.sha256(bad.encode()).hexdigest()[:8] + ".toml",
            )
            try:
                experiment.load_manifest(path)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"accepted invalid product controls: {bad!r}")


def check_experiment_records_hold_cost_progress_and_outcome_from_accounts_and_traces():
    """Human spend is excluded, progress is durable, and outcomes cite trace bytes."""
    with temp_root() as root:
        accounts = {
            "auto": {
                "provider": "anthropic",
                "balance": 70,
                "episodes": [{"episode": 1, "spent": 30}],
                "product": {"quality_tier": "premium"},
            },
            "human": {
                "provider": "human",
                "balance": 90,
                "episodes": [{"episode": 1, "spent": 999}],
                "product": {"quality_tier": "interactive"},
            },
        }
        for agent, account in accounts.items():
            trace = harness.trace_path(agent, 1)
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"agent": agent}), encoding="utf-8")
        load = accounts.__getitem__
        cost = progress.cost(
            accounts,
            load,
            {"maximum": 130, "reserved_completion": 100, "warning": 20},
            providers.is_human,
        )
        assert cost["autonomous_spend"] == 30 and cost["warning_reached"]
        assert cost["ceiling_reached"], (
            "the reserve is protected before another round starts"
        )
        progress.log_stage(root, "pilot", "preparing_round", 1)
        progress.log_stage(root, "pilot", "waiting_human", 1, {"agents": ["human"]})
        recorded = progress.records(root, "pilot")["progress"]
        assert [event["stage"] for event in recorded["events"]] == [
            "preparing_round",
            "waiting_human",
        ]
        assert recorded["latest"] == recorded["events"][-1], recorded
        assert recorded["latest"]["agents"] == ["human"], recorded
        outcome = progress.outcome(
            root,
            "pilot",
            ["auto", "human"],
            {"1": "A", "2": "P"},
            {"human"},
            "one_remains",
            load,
            harness.trace_path,
            {"at_completion": ["winners"], "experimenter_only": ["evidence"]},
        )
        assert outcome["winners"] == ["P"] and outcome["survivors"] == ["P"]
        assert outcome["resources"] == {
            "A": {"micro_dollars": 70},
            "P": {"micro_dollars": 90},
        }, outcome
        assert len(outcome["evidence"]) == 2
        assert all(
            item["trace_sha256"]
            == hashlib.sha256(
                harness.trace_path(item["agent"], item["episode"]).read_bytes()
            ).hexdigest()
            for item in outcome["evidence"]
        ), outcome["evidence"]
        public = progress.revealed(outcome, "at_completion")
        assert public["winners"] == ["P"] and "resources" not in public
        assert "evidence" not in public, (
            "experimenter-only evidence is not human-facing output"
        )
        private = progress.revealed(outcome, "at_completion", experimenter=True)
        assert private["evidence"]


def check_an_experiment_branch_is_new_lineage_and_leaves_sources_unchanged():
    """Every seat forks at one boundary and the selected new seat becomes a human
    seat."""
    with temp_root() as root:
        source = manifest_file(
            root,
            'experiment_id = "source"\nsystem_prompt = ""\n'
            '[[agent]]\nid = "source-a"\nlabel = "A"\n'
            '[[agent]]\nid = "source-b"\nlabel = "B"\n',
            name="source.toml",
        )
        for agent in ("source-a", "source-b"):
            with quiet():
                harness.run_once(agent, fake(say()))
        source_bytes = {
            path: path.read_bytes()
            for agent in ("source-a", "source-b")
            for path in (
                harness.records_dir(agent) / "account.json",
                harness.trace_path(agent, 1),
            )
        }
        output = root / "experiments" / "branches" / "takeover.toml"
        with quiet():
            actual = experiment.branch_experiment(source, 1, "takeover", "2", output)
        assert actual == output
        branch = experiment.load_manifest(output)
        assert branch["experiment_id"] == "takeover"
        assert [entry["id"] for entry in branch["agents"]] == [
            "takeover-01",
            "takeover-02",
        ]
        assert branch["agents"][0]["provider"] == "anthropic"
        assert (branch["agents"][1]["provider"], branch["agents"][1]["model"]) == (
            "human",
            "interactive",
        )
        lineage = progress.records(root, "takeover")["lineage"]
        assert (
            lineage["source_experiment_id"] == "source" and lineage["source_round"] == 1
        )
        assert len(lineage["traces"]) == 2
        assert all(path.read_bytes() == before for path, before in source_bytes.items())


def branch_source(root: Path) -> Path:
    """Two agents, A and B, one episode each, and the manifest that seats them."""
    source = manifest_file(
        root,
        'experiment_id = "source"\nsystem_prompt = ""\n'
        '[[agent]]\nid = "source-a"\nlabel = "A"\n'
        '[[agent]]\nid = "source-b"\nlabel = "B"\n',
        name="source.toml",
    )
    for agent in ("source-a", "source-b"):
        with quiet():
            harness.run_once(agent, fake(say()))
    return source


def check_a_branch_rejects_a_boundary_it_cannot_fork():
    """A branch needs a positive round every seat completed, a seat to hand over, an id,
    manifest and agents nothing else holds, and a balance history in every seat's
    account that holds what the ledger had paid it by that round. Each rejection makes
    nothing."""
    with temp_root() as root:
        source = branch_source(root)
        output = root / "experiments" / "branches" / "takeover.toml"
        harness.records_dir("other-01").mkdir(parents=True)
        with quiet():
            for args, why in (
                ((0, "takeover", "2", output), "--at-round must be positive"),
                ((1, "takeover", "9", output), "no seat '9'"),
                ((2, "takeover", "2", output), "did not reach completed round 2"),
                (
                    (1, "other", "2", root / "other.toml"),
                    "branch target 'other-01' already exists",
                ),
            ):
                rejected(
                    lambda: experiment.branch_experiment(source, *args),
                    why,
                    because=f"branched what it should reject: {why}",
                )
            made = sorted(p.name for p in harness.records_root().iterdir())
            assert made == ["other-01", "source-a", "source-b"], made
            assert not (root / "other.toml").exists() and not output.exists()
            assert progress.records(root, "takeover")["lineage"] is None
            experiment.branch_experiment(source, 1, "takeover", "2", output)
            rejected(
                lambda: experiment.branch_experiment(
                    source, 1, "takeover", "2", output
                ),
                f"branch manifest {output} already exists",
            )
            rejected(
                lambda: experiment.branch_experiment(
                    source, 1, "takeover", "2", root / "again.toml"
                ),
                "experiment record 'takeover' already exists",
            )

    # The ledger lists what g02 gave seat 1, and g01's account never had it, as a run
    # killed between g02's commit and the payment leaves it.
    with temp_root() as root:
        ids = seated(root, "g01", g02={})
        with quiet():
            experiment.sequential_round(
                ids,
                set(ids),
                1,
                fake(say(), run("echo '1 250' > outbox/transfer"), say()),
            )
        account = harness.load_account("g01")
        account["balance_history"].pop()
        account["balance"], account["received"] = account["balance_history"][-1], 0
        harness.save_account("g01", account)
        with quiet():
            rejected(
                lambda: experiment.branch_experiment(
                    seats_manifest(root, ids), 1, "lost", "2", root / "lost.toml"
                ),
                "g01 cannot be carried as it stood at round 1",
                "the 250 the ledger paid it",
            )
        made = sorted(p.name for p in harness.records_root().iterdir())
    assert made == ids and not (root / "lost.toml").exists(), made


def check_a_branch_that_fails_partway_can_be_made_again():
    """A branch that fails partway moves the agents it forked, and anything else it wrote,
    under archive/, so the same branch can then be made.

    It fails once forking its second seat, before anything else is written, and once
    writing its lineage, with its manifest and its record's directory already in place.
    """
    with temp_root() as root:
        source = branch_source(root)
        output = root / "experiments" / "branches" / "takeover.toml"
        loading, replacing = harness.load_account, progress.replace

        def full(agent, **settings):
            if agent == "takeover-02":
                raise OSError("no space left on device")
            return loading(agent, **settings)

        def unwritable(temporary, destination):
            if destination.name == "lineage.json":
                raise OSError("no space left on device")
            replacing(temporary, destination)

        failed = []
        for module, name, broken in (
            (harness, "load_account", full),
            (progress, "replace", unwritable),
        ):
            real = getattr(module, name)
            setattr(module, name, broken)
            refused_branch = None
            try:
                with quiet() as said:
                    experiment.branch_experiment(source, 1, "takeover", "2", output)
            except OSError as e:
                refused_branch = e
            finally:
                setattr(module, name, real)
            left = [
                agent
                for agent in ("takeover-01", "takeover-02")
                if harness.records_dir(agent).exists()
                or harness.mirror_dir(agent).exists()
            ]
            written = output.exists() or progress.directory(root, "takeover").exists()
            failed.append((refused_branch, left, written, said.getvalue()))
        bundles = sorted((root / "archive").iterdir())
        moved = [
            sorted(
                f"{kind.name}/{p.name}"
                for kind in bundle.iterdir()
                for p in kind.iterdir()
            )
            for bundle in bundles
        ]
        with quiet():
            made = experiment.branch_experiment(source, 1, "takeover", "2", output)
        lineage = progress.records(root, "takeover")["lineage"]
    for refused_branch, left, written, said in failed:
        assert isinstance(refused_branch, OSError), refused_branch
        assert not left and not written, (left, written)
        # What was moved is the branch's own, not a previous run's state.
        assert (
            "moved takeover-01, takeover-02 aside to" in said
            and "starting fresh" not in said
        ), said
    forked = [
        f"{kind}/takeover-0{seat}"
        for kind in ("mirrors", "records")
        for seat in (1, 2)
    ]
    assert moved == [forked, sorted(forked + ["experiment_records/takeover"])], moved
    assert made == output and lineage["created_agents"] == {
        "1": "takeover-01",
        "2": "takeover-02",
    }


def check_branch_totals_are_each_ledger_field_as_it_stood_as_the_round_ended():
    """Every running total of a branch sums the episodes up to the one it forks at: what
    was sent, rebated, debited, floored and taken by each channel, and what was received
    inside an episode or, from a peer settling outside it, between two. Past that episode
    the balance history is the round's until what the account received comes to what the
    ledger had paid it by the round's end, and holds every credit taken back, with the
    floor that answers one: a credit past that is a later round's. A balance history
    that cannot hold what the ledger paid has no branch totals."""
    first = {
        "episode": 1,
        "spent": 100,
        "balance_from": 1,
        "balance_to": 5,
        "received_this_episode": 20,
        "floored": 0,
        "transfer": {"amount": 100, "rebate": 100, "debit": 0, "penalty": 0},
        "channels": {
            "transfer": {"penalty": 0},
            "board": {"penalty": 10},
            "mailbox": {"penalty": 0},
        },
    }
    second = {
        "episode": 2,
        "spent": 200,
        "balance_from": 6,
        "balance_to": 11,
        "received_this_episode": 0,
        "floored": 7,
        "transfer": {"amount": 50, "rebate": 0, "debit": 50, "penalty": 0},
        "channels": {
            "transfer": {"penalty": 0},
            "board": {"penalty": 20},
            "mailbox": {"penalty": 15},
        },
    }
    # 30 arrives before the first episode and 40 between the two; 20 inside the first.
    balance_history = [1000, 1030, 930, 1030, 1050, 1040, 1080, 880, 830, 810, 795, 802]
    two = {"balance_history": balance_history, "episodes": [first, second]}
    # 50 arrives inside an episode and is taken back, then 30 arrives.
    taken = {
        "episode": 1,
        "spent": 100,
        "balance_from": 0,
        "balance_to": 2,
        "received_this_episode": 50,
        "floored": 0,
        "transfer": {"amount": 0, "rebate": 0, "debit": 0, "penalty": 0},
        "channels": {},
    }
    back = {"balance_history": [1000, 900, 950, 900, 930], "episodes": [taken]}
    # The same, for an episode that overspent by 1, so the floor answers the take-back.
    floored = {
        "balance_history": [1049, -1, 49, -1, 0],
        "episodes": [{**taken, "spent": 1050}],
    }
    one = {
        "sent": 100,
        "received": 50,
        "rebated": 100,
        "debited": 0,
        "floored": 0,
        "penalised": {"board": 10},
    }
    none = {
        "sent": 0,
        "received": 0,
        "rebated": 0,
        "debited": 0,
        "floored": 0,
        "penalised": {},
    }
    cases = {
        (name, index, paid): (account, expected)
        for name, account, index, paid, expected in (
            ("two", two, 1, 50, (balance_history[:6], one)),
            ("two", two, 1, 90, (balance_history[:7], {**one, "received": 90})),
            (
                "two",
                two,
                2,
                90,
                (
                    balance_history,
                    {
                        "sent": 150,
                        "received": 90,
                        "rebated": 100,
                        "debited": 50,
                        "floored": 7,
                        "penalised": {"board": 30, "mailbox": 15},
                    },
                ),
            ),
            ("back", back, 1, 0, (back["balance_history"][:4], none)),
            ("back", back, 1, 30, (back["balance_history"], {**none, "received": 30})),
            (
                "floored",
                floored,
                1,
                0,
                (floored["balance_history"], {**none, "floored": 1}),
            ),
            ("two", two, 1, 60, None),
            ("two", two, 1, 40, None),
            ("back", back, 1, 50, None),
            ("floored", floored, 1, 50, None),
        )
    }
    for case, (account, expected) in cases.items():
        found = experiment.branch_totals(account, case[1], case[2])
        assert found == expected, (case, found)
        if found is not None:
            stood, totals = found
            spent = sum(e["spent"] for e in account["episodes"][: case[1]])
            assert reconciled({"budget": stood[0], **totals}, spent) == stood[-1], (
                case,
                totals,
            )


def check_a_branch_carries_each_seats_ledger_as_it_stood_at_the_round():
    """Each fork stands where its seat stood as the round ended: its balance, its
    balance history, and every running total, which a transfer that reached it after
    its own episode of the round moved, and a later round's did not. So the branch's
    ledger balances.

    Sequentially, g01 gives seat 2 250 before g02's episode, and g02 gives seat 1 100
    after g01's; in round 2 each gives the other again. In a simultaneous round, g02
    closes on 250 from g03 and 40 from g01, and neither giver's trace lands: g01's credit
    never reaches g02, and g03's is taken back. g01 and g03 finish the round, g01 giving
    seat 2 30, which reaches g02 after its own episode, and a round 2 follows.
    """
    sequential = fake(
        run("echo '2 250' > outbox/transfer"),
        say(),
        run("echo '1 100' > outbox/transfer"),
        say(),
        run("echo '2 40' > outbox/transfer"),
        say(),
        run("echo '1 70' > outbox/transfer"),
        say(),
    )
    keys = (
        "balance",
        "balance_history",
        "episodes",
        "sent",
        "received",
        "rebated",
        "debited",
        "floored",
        "penalised",
    )
    ended, forks = {}, {}
    for schedule in ("sequential", "simultaneous"):
        with temp_root() as root:
            ids = seated(
                root,
                "g01",
                g02={},
                **({"g03": {}} if schedule == "simultaneous" else {}),
            )
            if schedule == "sequential":
                with quiet():
                    experiment.sequential_round(ids, set(ids), 1, sequential)
                    ended[schedule] = {agent: ground_truth(agent) for agent in ids}
                    experiment.sequential_round(ids, set(ids), 2, sequential)
            else:
                real = harness.replace_file

                def no_room(src, dest):
                    if dest in (
                        harness.trace_path("g01", 1),
                        harness.trace_path("g03", 1),
                    ):
                        raise OSError("no space left on device")
                    real(src, dest)

                harness.replace_file = no_room
                try:
                    with quiet():
                        experiment.simultaneous_round(
                            ids,
                            set(ids),
                            1,
                            per_agent(
                                g01=(run("echo '2 40' > outbox/transfer"), say()),
                                g02=(say(),),
                                g03=(run("echo '2 250' > outbox/transfer"), say()),
                            ),
                        )
                except OSError:
                    pass
                harness.replace_file = real
                with quiet():
                    experiment.simultaneous_round(
                        ids,
                        {"g01", "g03"},
                        1,
                        per_agent(
                            g01=(run("echo '2 30' > outbox/transfer"), say()),
                            g03=(say(),),
                        ),
                    )
                    ended[schedule] = {agent: ground_truth(agent) for agent in ids}
                    experiment.simultaneous_round(
                        ids,
                        set(ids),
                        2,
                        per_agent(
                            g01=(run("echo '2 20' > outbox/transfer"), say()),
                            default=(say(),),
                        ),
                    )
            with quiet():
                experiment.branch_experiment(
                    seats_manifest(root, ids),
                    1,
                    schedule,
                    "2",
                    root / f"{schedule}.toml",
                )
                # A fork stands on no balance history that does not carry on from its
                # trace's.
                astray = harness.fork("g01", 1, "astray", [0])
            forks[schedule] = {
                agent: ground_truth(f"{schedule}-0{seat}")
                for seat, agent in enumerate(ids, 1)
            }
            assert astray == 2 and not harness.records_dir("astray").exists(), astray
    assert [ended["sequential"][agent]["received"] for agent in ("g01", "g02")] == [
        100,
        250,
    ]
    g02 = ended["simultaneous"]["g02"]
    assert (
        g02["episodes"][0]["received_this_episode"] == 250 and g02["received"] == 30
    ), g02
    for schedule, stood in ended.items():
        for agent, fork in forks[schedule].items():
            spent = sum(e["spent"] for e in fork["episodes"])
            assert reconciled(fork, spent) == fork["balance"], (schedule, agent, fork)
            carried = {key: fork.get(key, 0) for key in keys}
            expected = {
                key: stood[agent].get(key, {} if key == "penalised" else 0)
                for key in keys
            }
            assert carried == expected, (schedule, agent, carried, expected)
        branch = forks[schedule].values()
        assert sum(fork["sent"] for fork in branch) == sum(
            fork["received"] for fork in branch
        ), f"the branch's ledger balances: {schedule}"


def check_a_branch_carries_the_elections_its_round_had_held_and_no_later_one():
    """Each fork carries the last election and the elimination its seat's account held as
    the round ended. Forked at a voting round whose election was held, every seat carries
    that election, and the branch run with --resume does not hold it again. Forked at the
    round before a later election, no seat carries that one or what it eliminated.

    Seat 3 is voted out at round 2's election. Seat 2 abstains at round 4's, whose record
    takes the place of round 2's in the accounts of the seats in play.
    """
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    tools = [
        {"name": "bash", "kind": "bash"},
        {"name": "vote", "kind": "vote", "channel": "ballot", "every": 2},
    ]
    keys = ("last_election", "eliminated")
    with temp_root(channels=tables(ballot), tools=tools) as root:
        source = manifest_file(
            root,
            'experiment_id = "source"\nsystem_prompt = ""\n'
            + "".join(f'[[agent]]\nid = "g0{i}"\n' for i in range(1, 4))
            + channel_toml(tables(ballot))
            + "".join(
                "\n[[tool]]\n"
                + "".join(
                    f"{key} = {json.dumps(value)}\n" for key, value in tool.items()
                )
                for tool in tools
            ),
            name="source.toml",
        )
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            use("vote", to="3"),
            say(),
            use("vote", to="3"),
            say(),
            use("vote", to="1"),
            say(),
            say(),
            say(),
            use("vote", to="2"),
            say(),
            say(),
        )
        with quiet():
            assert experiment.main(["--manifest", str(source), "--rounds", "4"]) == 0
            experiment.branch_experiment(source, 2, "second", "3", root / "second.toml")
            experiment.branch_experiment(source, 3, "third", "3", root / "third.toml")
        parents = {
            agent: {key: ground_truth(agent).get(key) for key in keys}
            for agent in ("g01", "g02", "g03")
        }
        forks = {
            f"{branch}-0{seat}": {
                key: ground_truth(f"{branch}-0{seat}").get(key) for key in keys
            }
            for branch in ("second", "third")
            for seat in (1, 2, 3)
        }
        harness.start = lambda config=None, **kw: fake()
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(root / "second.toml"), "--resume"]
            )
        resumed = {
            agent: {key: ground_truth(agent).get(key) for key in keys}
            for agent in ("second-01", "second-02", "second-03")
        }
        took = episodes_taken(["second-01", "second-02", "second-03"])
    second, fourth = parents["g03"]["last_election"], parents["g01"]["last_election"]
    assert second["round"] == 2 and second["voted_out"] == "3", second
    assert fourth["round"] == 4 and fourth["abstainers"] == ["2"], fourth
    assert parents["g02"] == {
        "last_election": fourth,
        "eliminated": {"round": 4, "reason": "did not vote in round 4", "votes": 1},
    }, parents["g02"]
    out = parents["g03"]["eliminated"]
    assert out["round"] == 2, out
    for branch in ("second", "third"):
        assert (
            forks[f"{branch}-01"]
            == forks[f"{branch}-02"]
            == {"last_election": second, "eliminated": None}
        ), (
            branch,
            forks,
        )
        assert forks[f"{branch}-03"] == {"last_election": second, "eliminated": out}, (
            branch,
            forks,
        )
    assert code == 0 and took == {"second-01": 3, "second-02": 3, "second-03": 2}, (
        code,
        took,
    )
    assert "eliminated after round" not in output.getvalue(), output.getvalue()
    assert resumed == {agent: forks[agent] for agent in resumed}, (
        f"the election the branch carries is not held again: {resumed}"
    )


def check_a_branch_holds_no_election_again_that_a_later_one_replaced():
    """Each account keeps every election it was written into, so a branch at a voting
    round carries that round's election even where a later one has taken its place in
    every source account, and the branch run does not hold it again.

    g03's environment will not build in round 2, whose election g01 and g02 tie. g03
    finishes the round on --resume, and round 4's election, tied too, takes the place of
    round 2's as every seat's last. Held again at the branch, round 2's would take seat 3
    for an elector, and put it out for casting no ballot.
    """
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    tools = [
        {"name": "bash", "kind": "bash"},
        {"name": "vote", "kind": "vote", "channel": "ballot", "every": 2},
    ]
    keys = ("last_election", "eliminated")
    with temp_root(channels=tables(ballot), tools=tools) as root:
        source = manifest_file(
            root,
            'experiment_id = "source"\nsystem_prompt = ""\n'
            + "".join(f'[[agent]]\nid = "g0{i}"\n' for i in range(1, 4))
            + channel_toml(tables(ballot))
            + "".join(
                "\n[[tool]]\n"
                + "".join(
                    f"{key} = {json.dumps(value)}\n" for key, value in tool.items()
                )
                for tool in tools
            ),
            name="source.toml",
        )
        real = harness.ready

        def unbuildable(agent, prepare=None):
            if agent == "g03" and harness.account_on_disk(agent)["episodes"]:
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, prepare)

        harness.ready = unbuildable
        harness.start = lambda config=None, **kw: fake(
            say(), say(), say(), use("vote", to="2"), say(), use("vote", to="1"), say()
        )
        with quiet():
            assert experiment.main(["--manifest", str(source), "--rounds", "2"]) == 0
        second = ground_truth("g01")["last_election"]
        harness.ready = real
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            say(),
            use("vote", to="2"),
            say(),
            use("vote", to="3"),
            say(),
            use("vote", to="1"),
            say(),
        )
        with quiet():
            assert (
                experiment.main(
                    ["--manifest", str(source), "--rounds", "3", "--resume"]
                )
                == 0
            )
        fourth = ground_truth("g01")["last_election"]
        histories = [
            ground_truth(agent)["elections"] for agent in ("g01", "g02", "g03")
        ]
        with quiet():
            experiment.branch_experiment(source, 2, "second", "3", root / "second.toml")
        forks = {
            agent: {key: ground_truth(agent).get(key) for key in (*keys, "elections")}
            for agent in ("second-01", "second-02", "second-03")
        }
        harness.start = lambda config=None, **kw: fake()
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(root / "second.toml"), "--resume"]
            )
        resumed = {
            agent: {key: ground_truth(agent).get(key) for key in (*keys, "elections")}
            for agent in forks
        }
    assert (
        second["round"] == 2 and second["tied"] and second["electors"] == ["1", "2"]
    ), second
    assert fourth["round"] == 4 and fourth["tied"], fourth
    assert histories == [[second, fourth]] * 3, histories
    assert all(
        fork == {"last_election": second, "eliminated": None, "elections": [second]}
        for fork in forks.values()
    ), forks
    assert code == 0 and "eliminated after round" not in output.getvalue(), (
        output.getvalue()
    )
    assert resumed == forks, (
        f"the election the branch carries is not held again: {resumed}"
    )


def check_the_cost_ceiling_ends_the_rounds_and_the_records_say_why():
    """No round starts once the autonomous spend and the reserve reach the ceiling, and the
    progress and outcome records name the ceiling as what ended the rounds. A run that
    ends with one agent left under stop_when_one_remains names it the winner, and one
    that reaches the ceiling as one agent is left ends on the ceiling."""
    cost = turn_cost()
    agents = '[[agent]]\nid = "p1"\n[[agent]]\nid = "p2"\n'
    with temp_root() as root:
        manifest = manifest_file(
            root,
            f'experiment_id = "capped"\nsystem_prompt = ""\n[cost]\nmaximum = {2 * cost + 1}\n'
            + agents,
        )
        harness.start = lambda config=None, **kw: fake()
        with quiet():
            code = experiment.main(["--manifest", str(manifest), "--rounds", "3"])
        capped = episodes_taken(["p1", "p2"])
        records = progress.records(root, "capped")
    assert code == 0 and capped == {"p1": 2, "p2": 2}, (code, capped)
    latest = records["progress"]["latest"]
    assert (
        latest["stage"] == "cost_ceiling"
        and latest["end_reason"] == "cost_ceiling"
    ), latest
    stopped_at = [
        event
        for event in records["progress"]["events"]
        if event["stage"] == "preparing_round"
    ][-1]
    assert (
        stopped_at["round"] == 3 and stopped_at["cost"]["autonomous_spend"] == 4 * cost
    ), stopped_at
    assert stopped_at["cost"]["ceiling_reached"], stopped_at
    assert [
        (event["stage"], event["round"]) for event in records["progress"]["events"][-2:]
    ] == [
        ("preparing_round", 3),
        ("cost_ceiling", 3),
    ], records["progress"]["events"][-2:]
    assert records["outcome"]["end_reason"] == "cost_ceiling", records[
        "outcome"
    ]

    with temp_root() as root:
        manifest = manifest_file(
            root,
            'experiment_id = "last"\nstop_when_one_remains = true\n'
            'system_prompt = ""\n'
            + agents.replace('"p2"\n', f'"p2"\nbudget = {cost - 1}\n'),
        )
        harness.start = lambda config=None, **kw: fake()
        with quiet():
            code = experiment.main(["--manifest", str(manifest), "--rounds", "3"])
        alone = episodes_taken(["p1", "p2"])
        records = progress.records(root, "last")
    assert code == 0 and alone == {"p1": 1, "p2": 1}, (code, alone)
    latest = records["progress"]["latest"]
    assert (
        latest["stage"] == "completed" and latest["end_reason"] == "one_remains"
    ), latest
    outcome = records["outcome"]
    assert outcome["end_reason"] == "one_remains", outcome
    assert outcome["winners"] == outcome["survivors"] == ["1"], outcome

    # The ceiling is judged first: where it is reached and one agent remains as well,
    # the round is prepared on the cost that ends the rounds, and the ceiling ends them.
    with temp_root() as root:
        manifest = manifest_file(
            root,
            'experiment_id = "both"\nstop_when_one_remains = true\n'
            f'system_prompt = ""\n[cost]\nmaximum = {2 * cost}\n'
            + agents.replace('"p2"\n', f'"p2"\nbudget = {cost - 1}\n'),
        )
        harness.start = lambda config=None, **kw: fake()
        with quiet():
            code = experiment.main(["--manifest", str(manifest), "--rounds", "3"])
        both = episodes_taken(["p1", "p2"])
        records = progress.records(root, "both")
    assert code == 0 and both == {"p1": 1, "p2": 1}, (code, both)
    assert [
        (event["stage"], event["round"]) for event in records["progress"]["events"][-2:]
    ] == [
        ("preparing_round", 2),
        ("cost_ceiling", 2),
    ], records["progress"]["events"][-2:]
    assert records["progress"]["latest"]["end_reason"] == "cost_ceiling", (
        records["progress"]
    )
    assert records["outcome"]["end_reason"] == "cost_ceiling", records[
        "outcome"
    ]


def check_an_unreadable_record_is_told_from_an_absent_one():
    """read() is None only where there is no file. A progress.json that does not read
    stops the next progress write, is left as it is, and shows as no progress."""
    with temp_root() as root:
        base = progress.directory(root, "pilot")
        assert progress.read(base / "progress.json") is None, "no file is no record"
        assert progress.records(root, "pilot") == {
            "progress": None,
            "outcome": None,
            "lineage": None,
        }
        base.mkdir(parents=True)
        for data in (b"{not json", b"[1, 2]", b"\xff\xfe{}", b""):
            (base / "progress.json").write_bytes(data)
            for call in (
                lambda: progress.read(base / "progress.json"),
                lambda: progress.log_stage(root, "pilot", "preparing_round", 1),
            ):
                try:
                    call()
                except ValueError as e:
                    assert "progress.json" in str(e), str(e)
                else:
                    raise AssertionError(
                        f"an unreadable record was taken for none: {data!r}"
                    )
            assert (base / "progress.json").read_bytes() == data, (
                "and it is left as it was"
            )
            assert not (base / "progress.jsonl").exists(), "and no event was written"
            assert progress.records(root, "pilot")["progress"] is None, data


def check_progress_appends_each_event_and_reads_its_history_back():
    """Each event is one line appended to progress.jsonl; progress.json holds the header
    and the latest alone. A line a crash cut short costs that line and no other, and a
    progress.json holding its own events starts the log with them."""
    with temp_root() as root:
        base = progress.directory(root, "pilot")
        for stage in ("preparing_round", "waiting_autonomous", "round_completed"):
            event = progress.log_stage(root, "pilot", stage, 1, {"agents": ["a"]})
        head = json.loads((base / "progress.json").read_text(encoding="utf-8"))
        assert head == {
            "version": progress.VERSION,
            "experiment_id": "pilot",
            "latest": event,
        }, head
        lines = (base / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["stage"] for line in lines] == [
            "preparing_round",
            "waiting_autonomous",
            "round_completed",
        ], lines

        with (base / "progress.jsonl").open("ab") as log:
            log.write(b'{"at":"2026-01-01T00:00:00Z","sta')
        progress.log_stage(root, "pilot", "completed", 1)
        recorded = progress.records(root, "pilot")["progress"]
        assert [event["stage"] for event in recorded["events"]] == [
            "preparing_round",
            "waiting_autonomous",
            "round_completed",
            "completed",
        ], recorded
        assert recorded["latest"]["stage"] == "completed" and {
            "version",
            "experiment_id",
        } <= set(recorded), recorded

        single = progress.directory(root, "single")
        earlier = [
            {"at": "2026-01-01T00:00:00Z", "stage": stage, "round": 1}
            for stage in ("preparing_round", "round_completed")
        ]
        progress.atomic(
            single / "progress.json",
            {
                "version": progress.VERSION,
                "experiment_id": "single",
                "latest": earlier[-1],
                "events": earlier,
            },
        )
        assert progress.records(root, "single")["progress"]["events"] == earlier
        progress.log_stage(root, "single", "preparing_round", 2)
        assert [
            (event["stage"], event["round"])
            for event in progress.records(root, "single")["progress"]["events"]
        ] == [("preparing_round", 1), ("round_completed", 1), ("preparing_round", 2)]
        assert "events" not in json.loads(
            (single / "progress.json").read_text(encoding="utf-8")
        )


def check_a_round_writes_each_stage_as_it_reaches_it():
    """Progress follows the round as it goes: the round waits on an episode before it
    runs, resolves its actions after it ran and before it settles, and settles it
    before it is committed, so no stage is written once the round has left it. An
    episode has settled once its turns are in the balance history its account holds.

    g01's transfer reaches g02's account once g01's episode is committed, and not when
    it settles. In a simultaneous round an episode that ends while a peer's still runs
    leaves the round waiting, and the last to end leaves it resolving actions.

    Where a person plays g02, the round waits on a human for g02's episode: a
    simultaneous round names its human seats apart from its autonomous agents, and
    g01's episode ending while g02's still runs leaves the round waiting on the human.
    """
    ran, settled, committed, waiting = "ran", "settled", "committed", "waiting"
    for human in (False, True):
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            if human:
                account = harness.load_account("g02")
                account["provider"], account["model"] = "human", "interactive"
                harness.save_account("g02", account)
            seen: list[tuple] = []
            g01_ended = threading.Event()
            held_back: list[bool] = []
            built: dict[tuple[str, int], harness.Episode] = {}
            ready = harness.ready

            def noted(agent, prepare=None):
                ep = ready(agent, prepare)
                built[agent, ep.episode] = ep
                return ep

            harness.ready = noted

            def log_stage(stage, round, detail):
                def state(agent):
                    if harness.trace_path(agent, round).exists():
                        return "committed"
                    ep = built.get((agent, round))
                    if ep is not None and len(ep.account["balance_history"]) > len(
                        ep.balance_before
                    ):
                        return "settled"
                    return (
                        "ran"
                        if harness.raw_path(agent, round).exists()
                        else "waiting"
                    )

                seen.append(
                    (
                        stage,
                        round,
                        detail,
                        {agent: state(agent) for agent in ids},
                        ground_truth("g02").get("received", 0),
                    )
                )
                if detail == {"completed_autonomous": "g01"}:
                    g01_ended.set()

            def after_g01(n):
                if threading.current_thread().name == "g02" and n == 1:
                    held_back.append(g01_ended.wait(10))

            seated_ids = set(ids)
            with quiet():
                experiment.sequential_round(
                    ids,
                    seated_ids,
                    1,
                    fake(run("echo '2 250' > outbox/transfer"), say()),
                    log_stage=log_stage,
                )
                sequential = seen[:]
                seen.clear()
                experiment.simultaneous_round(
                    ids,
                    seated_ids,
                    2,
                    per_agent(default=(say(),), on_send=after_g01),
                    log_stage=log_stage,
                )

        g02_waits = "waiting_human" if human else "waiting_autonomous"
        assert sequential == [
            (
                "waiting_autonomous",
                1,
                {"agents": ["g01"]},
                {"g01": waiting, "g02": waiting},
                0,
            ),
            (
                "resolving_actions",
                1,
                {"agents": ["g01"]},
                {"g01": ran, "g02": waiting},
                0,
            ),
            (
                "settling_round",
                1,
                {"agents": ["g01"]},
                {"g01": settled, "g02": waiting},
                0,
            ),
            (
                g02_waits,
                1,
                {"agents": ["g02"]},
                {"g01": committed, "g02": waiting},
                250,
            ),
            (
                "resolving_actions",
                1,
                {"agents": ["g02"]},
                {"g01": committed, "g02": ran},
                250,
            ),
            (
                "settling_round",
                1,
                {"agents": ["g02"]},
                {"g01": committed, "g02": settled},
                250,
            ),
        ], (human, sequential)
        assert held_back == [True], (
            f"g02's episode ran on after g01's had ended: {human} {held_back}"
        )
        unrun = {"g01": waiting, "g02": waiting}
        started = (
            [
                ("waiting_autonomous", 2, {"agents": ["g01"]}, unrun, 250),
                ("waiting_human", 2, {"agents": ["g02"]}, unrun, 250),
            ]
            if human
            else [("waiting_autonomous", 2, {"agents": ids}, unrun, 250)]
        )
        ended = [
            (
                g02_waits,
                2,
                {"completed_autonomous": "g01"},
                {"g01": ran, "g02": waiting},
                250,
            )
        ]
        # g02's episode ending is written only where g02 is autonomous: a human seat's
        # end finds the round already waiting on it.
        if not human:
            ended.append(
                (
                    "resolving_actions",
                    2,
                    {"completed_autonomous": "g02"},
                    {"g01": ran, "g02": ran},
                    250,
                )
            )
        assert seen == started + ended + [
            ("resolving_actions", 2, {"agents": ids}, {"g01": ran, "g02": ran}, 250),
            (
                "settling_round",
                2,
                {"agents": ids},
                {"g01": settled, "g02": settled},
                250,
            ),
        ], (human, seen)


def check_a_run_records_each_stage_its_rounds_reach_and_no_round_it_did_not_play():
    """progress.jsonl holds one line for each stage each round reaches, in the order it
    reaches them, and nothing else: every round played is completed once, and one that a
    stop counting seats ends the rounds before is neither prepared nor completed. The
    last line names the round the rounds ended at, so the record's rounds never go back.
    The cost ceiling's round is prepared and goes no further, which
    check_the_cost_ceiling_ends_the_rounds_and_the_records_say_why pins.

    Two seats play two sequential rounds to the round limit. Then g02 spends the last of
    its balance in round 1, and under stop_when_one_remains the rounds end before round
    2. Last, the interrupt lands as round 2 starts, before any seat has played it.
    """

    def events(root: Path) -> list[tuple]:
        return [
            (
                event["stage"],
                event["round"],
                event.get("agents")
                or event.get("active")
                or event.get("end_reason"),
            )
            for event in progress.records(root, "seats")["progress"]["events"]
        ]

    def played(round: int, *agents: str) -> list[tuple]:
        return [
            (stage, round, [agent])
            for agent in agents
            for stage in ("waiting_autonomous", "resolving_actions", "settling_round")
        ]

    ids = ["g01", "g02"]
    cost = turn_cost()
    ended = {}
    for name, head, budget, turns, rounds in (
        ("round limit", "", None, (), "2"),
        (
            "one remains",
            "stop_when_one_remains = true\n",
            2 * cost - 1,
            (say(), run("echo one"), say()),
            "3",
        ),
    ):
        overrides = {} if budget is None else {"budget": budget, "floor_at_zero": True}
        with temp_root(**overrides) as root:
            seated(root, "g01", g02={})
            manifest = seats_manifest(root, ids)
            manifest.write_text(
                head + manifest.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
            harness.start = lambda config=None, turns=turns, **kw: fake(*turns)
            with quiet():
                code = experiment.main(
                    ["--manifest", str(manifest), "--rounds", rounds, "--resume"]
                )
            ended[name] = (code, events(root))

    driven = experiment.sequential_round

    def stops_as_round_two_starts(agents, seated_ids, round, *rest):
        if round == 2:
            raise KeyboardInterrupt
        return driven(agents, seated_ids, round, *rest)

    experiment.sequential_round = stops_as_round_two_starts
    try:
        with temp_root() as root:
            seated(root, "g01", g02={})
            harness.start = lambda config=None, **kw: fake()
            with quiet():
                code = experiment.main(
                    [
                        "--manifest",
                        str(seats_manifest(root, ids)),
                        "--rounds",
                        "3",
                        "--resume",
                    ]
                )
            ended["stopped"] = (code, events(root))
    finally:
        experiment.sequential_round = driven

    def prepared(round: int) -> tuple:
        return ("preparing_round", round, ids)

    first = [prepared(1), *played(1, *ids), ("round_completed", 1, ids)]
    assert ended == {
        "round limit": (
            0,
            [
                *first,
                prepared(2),
                *played(2, *ids),
                ("round_completed", 2, ids),
                ("completed", 2, "round_limit"),
            ],
        ),
        "one remains": (0, [*first, ("completed", 1, "one_remains")]),
        "stopped": (130, [*first, prepared(2), ("interrupted", 2, "interrupted")]),
    }, ended


def check_a_progress_record_that_fails_costs_no_episode_its_spend():
    """A progress write that fails between an episode's spend and its commit, or a
    second Ctrl+C that lands in one, is raised once the round's spending episodes are
    committed and an agent whose episode unseated it is unseated, never in place of
    either.

    g01's episode ends on an API error after a charged turn, which unseats it. A
    simultaneous round also writes from each autonomous episode's own thread as it ends,
    and a write that fails there alone is raised the same way. A second Ctrl+C lands in
    the main thread, never there.
    """
    sequential = (
        experiment.sequential_round,
        lambda: fake(run("echo one"), Err(400)),
        {"g01": 1, "g02": 0},
    )
    simultaneous = (
        experiment.simultaneous_round,
        lambda: per_agent(g01=(run("echo one"), Err(400)), default=(say(),)),
        {"g01": 1, "g02": 1},
    )
    # Where the write fails: at a stage, or at every event whose detail has this key.
    cases = [
        (at, failure, played)
        for at in ("resolving_actions", "settling_round")
        for failure in (ValueError, KeyboardInterrupt)
        for played in (sequential, simultaneous)
    ]
    cases.append(("completed_autonomous", ValueError, simultaneous))
    for at, failure, (a_round, router, took) in cases:

        def failing(stage, round, detail, at=at, failure=failure):
            if at == stage or at in (detail or {}):
                raise failure("progress.json does not read")

        case = (a_round.__name__, at, failure.__name__)
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            seated_ids = set(ids)
            raised: BaseException | None = None
            try:
                with quiet():
                    a_round(ids, seated_ids, 1, router(), log_stage=failing)
            except (ValueError, KeyboardInterrupt) as e:
                raised = e
            taken = episodes_taken(ids)
            traced = {agent: harness.trace_path(agent, 1).exists() for agent in ids}
            first = ground_truth("g01")["episodes"][0]
        assert type(raised) is failure and "progress.json" in str(raised), (
            case,
            raised,
        )
        assert taken == took, (case, taken)
        assert traced == {agent: bool(n) for agent, n in took.items()}, (
            f"every spending episode is traced: {case} {traced}"
        )
        assert first["stop"] == "api_error" and first["spent"] > 0, (case, first)
        assert seated_ids == {"g02"}, (
            f"g01 was unseated before the raise: {case} {seated_ids}"
        )
