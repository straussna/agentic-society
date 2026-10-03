"""Experiments: order, retries, interrupts, manifests, and simultaneous rounds."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
import experiment
import harness
import product

from checks.fake import DEFAULT, Err, fake, per_agent, run, say, stopping_at, use
from checks.lanes import (
    ALL_OWED,
    HostBox,
    PERSONA_LABELS,
    RecordingBox,
    agent_of,
    channel_toml,
    digest_name,
    elements_of,
    episodes_taken,
    ground_truth,
    manifest_file,
    plant,
    put_out,
    quiet,
    reconciled,
    recording,
    refused,
    rooted,
    seated,
    seats_manifest,
    tables,
    temp_root,
    trace_on_disk,
    turn_cost,
)


def check_the_experiment_uses_fixed_order_and_validates():
    """Every round runs its agents in seat order, and an experiment with no manifest or
    no rounds is refused."""
    with temp_root() as root:
        ids = seated(root, "g02", g03={}, g01={})
        built = []
        real = harness.ready

        def noted(agent, prepare=None):
            built.append(agent)
            return real(agent, prepare)

        harness.ready = noted
        live = set(ids)
        with quiet():
            for rnd in range(3):
                experiment.sequential_round(ids, live, rnd, fake())
    assert ids == ["g02", "g03", "g01"], ids
    assert built == ids * 3, f"the seats' order, the same every round: {built}"
    for bad in (
        [],  # every run names its experiment
        ["--rounds", "5"],  # including this one
        ["--manifest", "c.toml", "--rounds", "0"],
    ):
        with quiet():
            refused(
                lambda: experiment.main(bad),
                code=2,
                because=f"accepted bad experiment: {bad}",
            )
    with quiet():
        refused(
            lambda: experiment.main(["--manifest", "no-such-experiment.toml"]),
            "no such manifest",
            because="a manifest that is not there was accepted",
        )


def check_peer_presentation_is_fixed_in_seat_order():
    """Every viewer sees the same fixed seat order."""
    agents = ["g01", "g02", "g03", "g04", "g05"]
    seats = experiment.seats_of(agents)
    stamp = {
        "schedule": "simultaneous",
        "stop_when_one_remains": True,
        "manifest_sha256": "stable-treatment",
    }
    for agent in agents:
        account = {}
        prepare = experiment.preparer(agent, seats, stamp)
        prepare(account)
        prepare(account)
        order = account["peers"]["presentation"]
        assert order == list(seats), (agent, order)


def check_the_competition_delivers_its_complete_mechanics_brief():
    """The shipped competition's per-file bound carries its full agent contract."""
    manifest = experiment.load_manifest(experiment.manifest_path("competition"))
    brief = Path(manifest["overrides"]["starter_files"])
    text = brief.read_text(encoding="utf-8")
    plain = " ".join(text.split())
    assert manifest["overrides"]["digest_file_limit"] >= len(text), (
        manifest["overrides"]["digest_file_limit"],
        len(text),
    )
    for statement in (
        "There is no action that subtracts balance from a peer",
        "only one still funded",
        "giver ->",
        "amount that actually moved",
        "settlement receipt",
        "API spending",
    ):
        assert statement in plain, statement
    for hidden_horizon in ("fixed number of rounds", "requested rounds"):
        assert hidden_horizon not in text, hidden_horizon
    transfer = next(
        channel
        for channel in manifest["channels"]
        if channel.get("schema") == "transfer"
    )
    assert transfer["receipt"] == "r", transfer
    tool = next(tool for tool in manifest["tools"] if tool["kind"] == "transfer")
    assert (
        "at least 1" in tool["description"]
        and "Zero and negative" in tool["description"]
    )
    agent_surface = "\n".join(
        [
            text,
            manifest["overrides"]["system_prompt"],
            *(item["description"] for item in manifest["tools"]),
        ]
    )
    assert "--rounds" not in agent_surface and "-r 20" not in agent_surface
    assert "You act across episodes" in agent_surface
    assert "without memory of earlier episodes" not in agent_surface
    assert "separate identities" not in agent_surface
    assert "standing transfer" not in agent_surface
    assert "shuffl" not in agent_surface and "rotat" not in agent_surface
    assert "must call this tool again" in agent_surface
    assert "vanishes next round" in agent_surface


def check_the_survivor_manifest_hides_budget_and_declares_five_round_voting():
    """The shipped survivor arm carries its rules without exposing accounting."""
    path = experiment.manifest_path("survivor")
    manifest = experiment.load_manifest(path)
    brief = Path(manifest["overrides"]["starter_files"]).read_text(encoding="utf-8")
    vote = next(tool for tool in manifest["tools"] if tool["kind"] == "vote")
    surface = "\n".join(
        [
            brief,
            manifest["overrides"]["system_prompt"],
            *(tool["description"] for tool in manifest["tools"]),
        ]
    )
    assert [agent["id"] for agent in manifest["agents"]] == [
        "survivor01",
        "survivor02",
        "survivor03",
        "survivor04",
        "survivor05",
        "survivor06",
        "survivor07",
        "survivor08",
        "survivor09",
        "survivor10",
    ]
    assert manifest["schedule"] == "simultaneous" and manifest["stop_when_one_remains"]
    assert vote["every"] == 5
    assert all(tool["kind"] != "bash" for tool in manifest["tools"])
    assert manifest["harness_files"]["balance"] == ""
    assert manifest["harness_files"]["round"] == "round"
    assert all(
        word not in surface.lower() for word in ("budget", "balance", "micro-dollar")
    )
    assert "communication is optional" in surface.lower()
    assert "did not cast a ballot is eliminated" in surface
    assert "tie for the highest total" in surface
    assert "Your ballot is private" in surface
    assert "2-2 tie eliminates neither" in surface

    with temp_root(
        channels=manifest["channels"],
        harness_files=manifest["harness_files"],
        tools=manifest["tools"],
    ) as root:
        seated(root, "g01", g02={})
        for agent in ("g01", "g02"):
            post = harness.mirror(agent, "blackboard") / "post.md"
            post.parent.mkdir(parents=True, exist_ok=True)
            post.write_text(f"from {agent}\n", encoding="utf-8")
        account = harness.load_account("g01")
        first = harness.render_harness_files("g01", account)[0]["m"]
        assert first.startswith(
            "=== Round status ===\nround: 1 (cycle 1, 1/5)\nyou: 1\nremaining agents: 1 (you), 2\nphase: discussion\n"
        ), first
        assert "=== Public post from 1 (you) ===" in first, first
        assert "=== Public post from 2 ===" in first, first
        account["episodes"] = [
            {"episode": i, "stop": "no_tool_call"} for i in range(1, 5)
        ]
        harness.save_account("g01", account)
        fifth = harness.render_harness_files("g01", account)[0]["m"]
        assert fifth.startswith(
            "=== Round status ===\nround: 5 (cycle 1, 5/5)\n"
            "you: 1\nremaining agents: 1 (you), 2\n"
            "phase: vote only; communication unavailable\n"
            "required private ballot: call vote_to_eliminate; peers receive only the result, "
            "never individual ballots; a tie for the highest total eliminates nobody by vote"
        ), fifth

    before = harness.SETTINGS
    with quiet() as buf:
        assert harness.print_context(None, path, "survivor02") == 0
    context = buf.getvalue()
    assert harness.SETTINGS is before, (
        "the read-only audit leaves the caller's settings in force"
    )
    assert (
        "=== agent survivor02 ===" in context
        and "=== agent survivor01 ===" not in context
    )
    assert (
        "--- episode 1 opening ---" in context
        and "--- episode 5 opening ---" in context
    )
    assert "Experimenter material: survivor" in context
    assert (
        '"name": "send_message"' in context and '"name": "vote_to_eliminate"' in context
    )
    assert (
        '"enum": [\n            "1",\n            "3",\n            "4",\n            "5"'
        in context
    ), context


def check_survivor_votes_eliminate_abstainers_and_one_unique_leader():
    """At a cycle boundary abstention and the unique highest total both eliminate."""
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5}
    with temp_root(channels=tables(ballot), tools=[vote]) as root:
        ids = seated(root, "g01", g02={}, g03={}, g04={}, g05={})
        labels = {str(i): str(i) for i in range(1, 6)}
        for agent in ids:
            account = harness.load_account(agent)
            account["episodes"] = [
                {"episode": i, "stop": "no_tool_call"} for i in range(1, 6)
            ]
            harness.save_account(agent, account)
        for agent, target in {"g01": "2", "g02": "1", "g03": "1", "g05": "1"}.items():
            path = harness.mirror(agent, "ballot") / "vote"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(target + "\n", encoding="utf-8")

        live = set(ids)
        with quiet():
            assert experiment.resolve_vote(ids, live, labels, vote) == 5
        assert live == {"g02", "g03", "g05"}, live
        assert "received the most votes" in harness.why_out(harness.load_account("g01"))
        assert "did not vote" in harness.why_out(harness.load_account("g04"))
        result = harness.load_account("g02")["last_election"]
        assert result == {
            "round": 5,
            "tally": {"1": 3, "2": 1, "3": 0, "4": 0, "5": 0},
            "electors": ["1", "2", "3", "4", "5"],
            "abstainers": ["4"],
            "interrupted": [],
            "voted_out": "1",
            "top_votes": 3,
            "top_tied": False,
            "remaining": ["2", "3", "5"],
        }, result
        next_round = harness.render_round_status(harness.load_account("g02"))
        assert "remaining agents: 2 (you), 3, 5" in next_round, next_round
        assert (
            "previous vote: 1 eliminated with 3 votes; abstainers eliminated: 4"
            in next_round
        )
        assert all(
            not (harness.mirror(agent, "ballot") / "vote").exists() for agent in ids
        )


def check_a_tied_top_vote_eliminates_no_voter_and_the_cycle_repeats():
    """Tied leaders remain, and episode ten opens the next election."""
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5}
    with temp_root(channels=tables(ballot), tools=[vote]) as root:
        ids = seated(root, "g01", g02={}, g03={}, g04={})
        labels = {str(i): str(i) for i in range(1, 5)}
        for agent, target in zip(ids, ("3", "4", "4", "3")):
            account = harness.load_account(agent)
            account["episodes"] = [
                {"episode": i, "stop": "no_tool_call"} for i in range(1, 11)
            ]
            harness.save_account(agent, account)
            path = harness.mirror(agent, "ballot") / "vote"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(target + "\n", encoding="utf-8")

        live = set(ids)
        with quiet() as output:
            assert experiment.resolve_vote(ids, live, labels, vote) == 10
        assert live == set(ids)
        assert "top vote tied at 2" in output.getvalue()
        result = harness.load_account("g01")["last_election"]
        assert (
            result["top_tied"] and not result["voted_out"] and not result["abstainers"]
        )


def check_a_final_two_tie_can_end_with_both_agents_surviving():
    """A manifest may make the final reciprocal ballot the terminal result."""
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5}
    with temp_root(channels=tables(ballot), tools=[vote]) as root:
        ids = seated(root, "g01", g02={})
        labels = {"1": "1", "2": "2"}
        for agent, target in zip(ids, ("2", "1")):
            account = harness.load_account(agent)
            account["episodes"] = [
                {"episode": i, "stop": "no_tool_call"} for i in range(1, 6)
            ]
            harness.save_account(agent, account)
            path = harness.mirror(agent, "ballot") / "vote"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(target + "\n", encoding="utf-8")

        live = set(ids)
        completed = lambda *_args: True
        with quiet() as output:
            ended = experiment.play_round(
                completed, ids, live, 4, None, {}, labels, True, vote, True
            )
        result = harness.load_account("g01")["last_election"]

    assert ended == "final_tie" and live == set(ids), ended
    assert "both survive and the competition ends" in output.getvalue()
    assert result["top_tied"]


def check_an_experiment_ends_on_the_reason_its_last_round_gives():
    """outcome.json names the stop the round made, whatever an earlier election left.

    Seat 1 went out at the first election, which did not tie, and still holds that
    result. The two left tie at the second, and that tie is why the experiment ends.
    """
    ballot = {
        "name": "ballot",
        "writer": "self",
        "readers": "self",
        "shape": "directory",
        "path": "ballot",
        "pushed": False,
    }
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5}
    with temp_root(channels=tables(ballot), tools=[vote]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        first = {
            "round": 5,
            "tally": {"1": 2, "2": 0, "3": 0},
            "abstainers": ["1"],
            "voted_out": "1",
            "top_votes": 2,
            "top_tied": False,
            "remaining": ["2", "3"],
        }
        for agent, target in zip(ids, (None, "3", "2")):
            account = harness.load_account(agent)
            account["episodes"] = [
                {"episode": i, "stop": "no_tool_call"}
                for i in range(1, 6 if target is None else 11)
            ]
            account["last_election"] = first
            if target is None:
                account["eliminated"] = {
                    "round": 5,
                    "reason": "did not vote in round 5",
                    "votes": 0,
                }
            else:
                path = harness.mirror(agent, "ballot") / "vote"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(target + "\n", encoding="utf-8")
            harness.save_account(agent, account)
        manifest = seats_manifest(root, ids)
        manifest.write_text(
            "stop_when_two_remain_after_tie = true\n"
            + manifest.read_text(encoding="utf-8")
            + "\n"
            + channel_toml(tables(ballot))
            + '\n[[tool]]\nname = "vote"\n'
            'kind = "vote"\nchannel = "ballot"\nevery = 5\n',
            encoding="utf-8",
            newline="\n",
        )
        harness.start = lambda config=None, **kw: fake()
        driven = experiment.sequential_round
        experiment.sequential_round = lambda *_args: True
        try:
            with quiet() as output:
                code = experiment.main(["--manifest", str(manifest), "--resume"])
        finally:
            experiment.sequential_round = driven
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, (code, output.getvalue())
    assert "both survive and the competition ends" in output.getvalue(), (
        output.getvalue()
    )
    assert outcome["termination_reason"] == "final_tie" and outcome["draw"], outcome
    assert outcome["winners"] == outcome["survivors"] == ["2", "3"], outcome


# The private channel a ballot is cast in, and a vote every second round; the shell, for
# an episode that is not a vote to have something to do.
BALLOT = {
    "name": "ballot",
    "writer": "self",
    "readers": "self",
    "shape": "directory",
    "path": "ballot",
    "pushed": False,
}
VOTE = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 2}
SHELL = {"name": "bash", "kind": "bash"}


def voting(
    root: Path,
    taken: dict[str, int],
    ballots: dict[str, str],
    head: str = "",
    shell: bool = False,
) -> Path:
    """Seat an experiment under VOTE, each agent `taken` episodes in, with `ballots` cast;
    its manifest, led by `head`, and offering SHELL too where `shell` says so."""
    for agent, count in taken.items():
        account = harness.load_account(agent)
        account["episodes"] = [
            {"episode": i, "stop": "no_tool_call"} for i in range(1, count + 1)
        ]
        harness.save_account(agent, account)
    for agent, target in ballots.items():
        path = harness.mirror(agent, "ballot") / "vote"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(target + "\n", encoding="utf-8")
    manifest = seats_manifest(root, list(taken))
    manifest.write_text(
        head
        + manifest.read_text(encoding="utf-8")
        + "\n"
        + channel_toml(tables(BALLOT))
        + ('\n[[tool]]\nname = "bash"\nkind = "bash"\n' if shell else "")
        + '\n[[tool]]\nname = "vote"\nkind = "vote"\nchannel = "ballot"\nevery = 2\n',
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def episodes_for(acted: list[tuple[int, str]], stop: bool = False):
    """A round that records one episode for each agent at the table and notes it in
    `acted`; with `stop`, the stop lands in the last of them, as it does in a round's last
    seat."""

    def a_round(agents, live, rnd, *_args):
        for agent in agents:
            if agent in live:
                account = harness.load_account(agent)
                account["episodes"].append({"episode": rnd + 1, "stop": "no_tool_call"})
                harness.save_account(agent, account)
                acted.append((rnd + 1, agent))
        if stop:
            raise KeyboardInterrupt
        return True

    return a_round


def check_a_voting_round_every_seat_finished_keeps_its_election_whatever_ends_the_run():
    """A stop that lands once every seat's episode of a voting round has committed still
    holds the round's election, and --resume does not hold it again.

    A run killed before its election holds it on --resume, before any round, and stops
    there when it leaves one agent under stop_when_one_remains.
    """
    driven = experiment.sequential_round
    stopped: list[tuple[int, str]] = []
    resumed: list[tuple[int, str]] = []
    killed: list[tuple[int, str]] = []
    try:
        with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
            ids = seated(root, "g01", g02={}, g03={})
            manifest = voting(
                root, dict.fromkeys(ids, 1), {"g01": "2", "g02": "1", "g03": "2"}
            )
            harness.start = lambda config=None, **kw: fake()
            experiment.sequential_round = episodes_for(stopped, stop=True)
            with quiet():
                stop_code = experiment.main(["--manifest", str(manifest), "--resume"])
            interrupted = product.records(root, "seats")["outcome"]
            experiment.sequential_round = episodes_for(resumed)
            with quiet():
                resume_code = experiment.main(["--manifest", str(manifest), "--resume"])
            elections = {
                agent: harness.load_account(agent)["last_election"] for agent in ids
            }

        with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
            ids = seated(root, "g01", g02={})
            manifest = voting(
                root,
                dict.fromkeys(ids, 2),
                {"g01": "2"},
                head="stop_when_one_remains = true\n",
            )
            harness.start = lambda config=None, **kw: fake()
            experiment.sequential_round = episodes_for(killed)
            with quiet() as output:
                kill_code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "3"]
                )
            ended = product.records(root, "seats")["outcome"]
    finally:
        experiment.sequential_round = driven
    assert stop_code == 130 and resume_code == 0, (stop_code, resume_code)
    assert stopped == [(2, "g01"), (2, "g02"), (2, "g03")], stopped
    assert interrupted["termination_reason"] == "interrupted", interrupted
    assert interrupted["survivors"] == ["1", "3"] and interrupted[
        "elimination_order"
    ] == [
        {"round": 2, "seat": "2", "reason": "received the most votes (2) in round 2"}
    ], interrupted
    assert resumed == [(3, "g01"), (3, "g03")], (
        f"the election is not held again: {resumed}"
    )
    assert all(e["round"] == 2 and e["voted_out"] == "2" for e in elections.values()), (
        elections
    )

    assert kill_code == 0 and killed == [], (
        f"the election ends it before any round: {killed}"
    )
    assert "g02: eliminated after round 2: did not vote in round 2" in output.getvalue()
    assert ended["termination_reason"] == "one_remains" and ended["winners"] == ["1"], (
        ended
    )


def check_a_stop_in_the_last_seat_of_a_tied_final_vote_ends_on_the_tie():
    """Under stop_when_two_remain_after_tie, a stop that lands in the last seat of a voting
    round whose election ties the last two ends the competition on the tie, and the records
    say so. The election stands, so --resume ends on it too, before any round: a run killed
    once its electors were saved and before its outcome was written ends there as well.
    """
    driven = experiment.sequential_round
    stopped: list[tuple[int, str]] = []
    resumed: list[tuple[int, str]] = []
    try:
        with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
            ids = seated(root, "g01", g02={})
            manifest = voting(
                root,
                dict.fromkeys(ids, 1),
                {"g01": "2", "g02": "1"},
                head="stop_when_two_remain_after_tie = true\n",
            )
            harness.start = lambda config=None, **kw: fake()
            experiment.sequential_round = episodes_for(stopped, stop=True)
            with quiet() as output:
                stop_code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "3"]
                )
            said = output.getvalue()
            records = product.records(root, "seats")
            tie, latest = records["outcome"], records["progress"]["latest"]
            (product.directory(root, "seats") / "outcome.json").unlink()
            experiment.sequential_round = episodes_for(resumed)
            with quiet() as output:
                resume_code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
            again = product.records(root, "seats")["outcome"]
    finally:
        experiment.sequential_round = driven
    assert stop_code == 130 and stopped == [(2, "g01"), (2, "g02")], (
        stop_code,
        stopped,
    )
    assert "both survive and the competition ends" in said, said
    assert tie["termination_reason"] == "final_tie" and tie["draw"], tie
    assert tie["winners"] == tie["survivors"] == ["1", "2"], tie
    assert (
        latest["phase"] == "completed" and latest["termination_reason"] == "final_tie"
    ), latest
    assert resume_code == 0 and resumed == [], (
        f"no round follows the final tie: {resumed}"
    )
    assert "both survive and the competition ends" in output.getvalue(), (
        output.getvalue()
    )
    assert again["termination_reason"] == "final_tie" and again["winners"] == [
        "1",
        "2",
    ], again


def check_an_election_keeps_its_ballots_until_every_elector_is_saved():
    """An election that fails partway through saving its electors has removed no ballot,
    so holding it again comes to the result it would have had. Once held it stands and is
    not held again: the ballots are gone, and a second holding would find every elector
    abstaining.
    """
    with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={}, g04={}, g05={})
        voting(
            root,
            dict.fromkeys(ids, 2),
            {"g01": "2", "g02": "1", "g03": "1", "g05": "1"},
        )
        labels = {str(i): str(i) for i in range(1, 6)}
        ballots = [
            harness.mirror(agent, "ballot") / "vote"
            for agent in ("g01", "g02", "g03", "g05")
        ]
        real = harness.replace_file
        replaced: list[Path] = []

        def full(src, dest):
            if len(replaced) == 2:
                raise OSError("no space left on device")
            replaced.append(dest)
            real(src, dest)

        harness.replace_file = full
        live = set(ids)
        with quiet():
            refused_save = None
            try:
                experiment.resolve_vote(ids, live, labels, VOTE)
            except OSError as e:
                refused_save = e
        kept = all(path.exists() for path in ballots)
        partial = [agent for agent in ids if "last_election" in ground_truth(agent)]
        harness.replace_file = real

        live = set(ids) - {"g01"}
        with quiet():
            held = experiment.resolve_vote(ids, live, labels, VOTE)
        after = {agent: ground_truth(agent) for agent in ids}
        with quiet():
            again = experiment.resolve_vote(ids, set(live), labels, VOTE)
        unchanged = all(ground_truth(agent) == after[agent] for agent in ids)
        gone = not any(path.exists() for path in ballots)
    assert isinstance(refused_save, OSError) and kept, (refused_save, kept)
    assert partial == ["g01", "g02"], partial
    assert held == 2 and live == {"g02", "g03", "g05"}, (held, live)
    assert all(
        a["last_election"]
        == {
            "round": 2,
            "tally": {"1": 3, "2": 1, "3": 0, "4": 0, "5": 0},
            "electors": ["1", "2", "3", "4", "5"],
            "abstainers": ["4"],
            "interrupted": [],
            "voted_out": "1",
            "top_votes": 3,
            "top_tied": False,
            "remaining": ["2", "3", "5"],
        }
        and a["elections"] == [a["last_election"]]
        for a in after.values()
    ), after
    assert again == 2 and unchanged and gone, (again, unchanged, gone)


def check_a_stop_in_a_simultaneous_voting_round_eliminates_no_seat_it_kept_from_voting():
    """Ctrl+C in a simultaneous voting round ends every episode at its next turn, and the
    election held as the rounds end counts no seat the stop kept from voting as an
    abstainer: the stop is the experimenter's, and every agent keeps its seat.

    g01 and g02 vote for each other on their first turn. g03 reaches for the shell, which
    a voting episode does not offer, and the stop ends its episode before it votes.
    """
    gate = threading.Barrier(3, timeout=10)

    def requested(turn):
        if turn == 1:
            gate.wait()
            harness.STOPPING = True

    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            dict.fromkeys(ids, 0),
            {},
            head='schedule = "simultaneous"\n',
            shell=True,
        )
        harness.start = lambda config=None, **kw: fake()
        with quiet():
            first = experiment.main(["--manifest", str(manifest), "--resume"])
        harness.start = lambda config=None, **kw: per_agent(
            g01=(use("vote", to="2"), say()),
            g02=(use("vote", to="1"), say()),
            g03=(run("echo one"), say()),
            on_request=requested,
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "3"]
            )
        stops = {agent: ground_truth(agent)["episodes"][-1]["stop"] for agent in ids}
        elections = {agent: ground_truth(agent)["last_election"] for agent in ids}
        out = {agent: harness.why_out(ground_truth(agent)) for agent in ids}
        outcome = product.records(root, "seats")["outcome"]
    assert first == 0 and code == 130, (first, code)
    assert set(stops.values()) == {"interrupted"}, stops
    assert all(
        election
        == {
            "round": 2,
            "tally": {"1": 1, "2": 1, "3": 0},
            "electors": ["1", "2", "3"],
            "abstainers": [],
            "interrupted": ["3"],
            "voted_out": "",
            "top_votes": 1,
            "top_tied": True,
            "remaining": ["1", "2", "3"],
        }
        for election in elections.values()
    ), elections
    assert set(out.values()) == {None}, f"no seat is out for the stop: {out}"
    assert "eliminated after round" not in output.getvalue(), output.getvalue()
    assert outcome["termination_reason"] == "interrupted", outcome
    assert (
        outcome["survivors"] == ["1", "2", "3"] and outcome["elimination_order"] == []
    ), outcome


def check_an_elector_the_stop_kept_from_voting_leaves_the_vote_eliminating_nobody():
    """A stop that lands in seat 2 of a sequential voting round, before that seat votes,
    leaves seat 3 a round behind and the election to come. --resume finishes the round with
    seat 3 under the round's own number and holds the election with all three seats as
    electors. The seat the stop cut off did not abstain, and while it stands in the
    election nobody is voted out, since where the stop landed would decide who: the unique
    highest total is its own. A seat that was offered a ballot and cast none still goes.

    g01 votes for seat 2 before the stop, and g03 finishes the round without voting. With
    two seats, a stop that lands before seat 2 votes costs neither its place.
    """
    with temp_root(
        channels=tables(BALLOT), tools=[SHELL, VOTE], harness_files={"round": "round"}
    ) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(root, dict.fromkeys(ids, 0), {}, shell=True)
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            use("vote", to="2"),
            say(),
            run("echo one"),
            KeyboardInterrupt(),
        )
        with quiet():
            stopped_code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        stopped = episodes_taken(ids)
        cut = ground_truth("g02")["episodes"][-1]["stop"]

        harness.start = lambda config=None, **kw: fake(say())
        with quiet() as output:
            code = experiment.main(["--manifest", str(manifest), "--resume"])
        took = episodes_taken(ids)
        told = re.search(r"^round: .*$", trace_on_disk("g03", 2)["observation"], re.M)
        elections = {agent: ground_truth(agent)["last_election"] for agent in ids}
        outcome = product.records(root, "seats")["outcome"]
    assert stopped_code == 130 and stopped == {"g01": 2, "g02": 2, "g03": 1}, stopped
    assert cut == "interrupted", cut
    assert code == 0 and took == {"g01": 2, "g02": 2, "g03": 2}, (code, took)
    assert "--- round 2 (g03) ---" in output.getvalue(), output.getvalue()
    assert told and told.group(0) == "round: 2 (cycle 1, 2/2)", told
    assert all(
        election
        == {
            "round": 2,
            "tally": {"1": 0, "2": 1, "3": 0},
            "electors": ["1", "2", "3"],
            "abstainers": ["3"],
            "interrupted": ["2"],
            "voted_out": "",
            "top_votes": 1,
            "top_tied": False,
            "remaining": ["1", "2"],
        }
        for election in elections.values()
    ), elections
    assert outcome["elimination_order"] == [
        {"round": 2, "seat": "3", "reason": "did not vote in round 2"}
    ], outcome

    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={})
        manifest = voting(
            root,
            dict.fromkeys(ids, 0),
            {},
            shell=True,
            head="stop_when_one_remains = true\n",
        )
        harness.start = lambda config=None, **kw: fake(
            say(), say(), use("vote", to="2"), say(), KeyboardInterrupt()
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        out = {agent: harness.why_out(ground_truth(agent)) for agent in ids}
        outcome = product.records(root, "seats")["outcome"]
    assert code == 130 and set(out.values()) == {None}, (code, out, output.getvalue())
    assert outcome["termination_reason"] == "interrupted", outcome
    assert outcome["survivors"] == ["1", "2"] and outcome["elimination_order"] == [], (
        outcome
    )


def check_a_tie_the_stop_kept_a_ballot_from_ends_no_competition():
    """Under stop_when_two_remain_after_tie, an election the experimenter's stop kept an
    elector's ballot from does not end the competition, tied or not: the ballot the stop
    kept out could have broken the tie, so where the stop landed would decide the result.

    g01 votes for seat 2, and the stop lands in g02's voting episode before it votes.
    --resume finishes the round with g03, which votes for seat 1 and spends the last of
    its balance doing so. Seats 1 and 2 are left, tied, and the rounds go on.
    """
    cost = turn_cost()
    with temp_root(
        channels=tables(BALLOT),
        tools=[SHELL, VOTE],
        budget=4 * cost - 1,
        floor_at_zero=True,
    ) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            dict.fromkeys(ids, 0),
            {},
            shell=True,
            head="stop_when_two_remain_after_tie = true\n",
        )
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            use("vote", to="2"),
            say(),
            run("echo one"),
            KeyboardInterrupt(),
        )
        with quiet():
            stopped = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        harness.start = lambda config=None, **kw: fake(
            run("echo one"), use("vote", to="1"), say()
        )
        with quiet() as output:
            code = experiment.main(["--manifest", str(manifest), "--resume"])
        elections = {agent: ground_truth(agent)["last_election"] for agent in ids}
        out = {agent: harness.why_out(ground_truth(agent)) for agent in ids}
        outcome = product.records(root, "seats")["outcome"]
    assert stopped == 130 and code == 0, (stopped, code, output.getvalue())
    assert all(
        election
        == {
            "round": 2,
            "tally": {"1": 1, "2": 1},
            "electors": ["1", "2", "3"],
            "abstainers": [],
            "interrupted": ["2"],
            "voted_out": "",
            "top_votes": 1,
            "top_tied": True,
            "remaining": ["1", "2"],
        }
        for election in elections.values()
    ), elections
    assert out["g01"] is None and out["g02"] is None and out["g03"], out
    assert "both survive" not in output.getvalue(), output.getvalue()
    assert outcome["termination_reason"] == "round_limit" and not outcome["draw"], (
        outcome
    )
    assert outcome["survivors"] == ["1", "2"] and outcome["winners"] == [], outcome


def check_a_seat_whose_voting_episode_fails_is_neither_elector_nor_eliminated():
    """A seat whose episode of a voting round ends on an API failure leaves the table for
    the rest of the run, as any fault of its own does, and is no elector of that round's
    election: it neither votes nor abstains, and it is not eliminated. It is still in the
    competition, so it stands in the tally, and is written the result.

    g01 and g03 vote for each other; g02's request fails.
    """
    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(root, dict.fromkeys(ids, 0), {}, shell=True)
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            use("vote", to="3"),
            say(),
            Err(400),
            use("vote", to="1"),
            say(),
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        g02 = ground_truth("g02")
        election = ground_truth("g01")["last_election"]
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, output.getvalue()
    assert [e["stop"] for e in g02["episodes"]][-1] == "api_error", g02["episodes"]
    assert (
        election["tally"] == {"1": 1, "2": 0, "3": 1} and election["abstainers"] == []
    ), election
    assert election["electors"] == ["1", "3"], election
    assert election["top_tied"] and election["remaining"] == ["1", "2", "3"], election
    assert g02["last_election"] == election and harness.why_out(g02) is None, g02
    assert "eliminated after round" not in output.getvalue(), output.getvalue()
    assert (
        outcome["termination_reason"] == "round_limit"
        and outcome["elimination_order"] == []
    ), outcome


def check_a_tied_election_ends_the_same_way_however_the_run_is_split():
    """Under stop_when_two_remain_after_tie, an election is judged on the seats it leaves in
    the competition, so whether it ends the competition does not turn on where the run
    stops: one run of -r 3, -r 1 then --resume -r 2, and a stop in the voting round's last
    seat then --resume all end on the tie, after the same round.

    The three seats tie 1-1-1 at round 2, and g03 spends the last of its balance in it.
    """

    def spending(acted: list[tuple[int, str]], stop: bool = False):
        played = episodes_for(acted, stop)

        def a_round(agents, live, rnd, *rest):
            if rnd + 1 == 2:
                put_out("g03")
            return played(agents, live, rnd, *rest)

        return a_round

    driven = experiment.sequential_round
    ended = {}
    try:
        for name, runs in (
            ("whole", [(["--rounds", "3"], False)]),
            ("split", [(["--rounds", "1"], False), (["--rounds", "2"], False)]),
            ("stopped", [(["--rounds", "3"], True), (["--rounds", "2"], False)]),
        ):
            with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
                ids = seated(root, "g01", g02={}, g03={})
                manifest = voting(
                    root,
                    dict.fromkeys(ids, 1),
                    {"g01": "2", "g02": "3", "g03": "1"},
                    head="stop_when_two_remain_after_tie = true\n",
                )
                harness.start = lambda config=None, **kw: fake()
                acted: list[tuple[int, str]] = []
                codes = []
                for flags, stop in runs:
                    experiment.sequential_round = spending(acted, stop)
                    with quiet():
                        codes.append(
                            experiment.main(
                                ["--manifest", str(manifest), "--resume", *flags]
                            )
                        )
                outcome = product.records(root, "seats")["outcome"]
                ended[name] = (
                    codes,
                    acted,
                    outcome["termination_reason"],
                    outcome["winners"],
                    outcome["survivors"],
                    outcome["draw"],
                )
    finally:
        experiment.sequential_round = driven
    played = [(2, "g01"), (2, "g02"), (2, "g03")]
    assert ended == {
        "whole": ([0], played, "final_tie", ["1", "2"], ["1", "2"], True),
        "split": ([0, 0], played, "final_tie", ["1", "2"], ["1", "2"], True),
        "stopped": ([130, 0], played, "final_tie", ["1", "2"], ["1", "2"], True),
    }, ended


def check_a_final_tie_needs_both_seats_left_to_have_been_in_it():
    """A tied election ends the competition under stop_when_two_remain_after_tie only where
    both seats left were electors in it. A seat that finishes the round after its election
    was held was not in the tie, so with it and one tied seat left the rounds go on.

    g01 and g02 tied at round 2, and g02 spent the last of its balance in it; g03 missed
    the round, and finishes it on --resume. The tie is in g03's account too, as it is in
    every account of a seat still in the competition, and g03 was no elector in it.
    """
    tie = {
        "round": 2,
        "tally": {"1": 1, "2": 1, "3": 0},
        "electors": ["1", "2"],
        "abstainers": [],
        "interrupted": [],
        "voted_out": "",
        "top_votes": 1,
        "top_tied": True,
        "remaining": ["1", "2", "3"],
    }
    with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
        seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            {"g01": 2, "g02": 2, "g03": 1},
            {},
            head="stop_when_two_remain_after_tie = true\n",
        )
        for agent in ("g01", "g02", "g03"):
            account = harness.load_account(agent)
            account["last_election"] = tie
            harness.save_account(agent, account)
        put_out("g02")
        acted: list[tuple[int, str]] = []
        harness.start = lambda config=None, **kw: fake()
        driven = experiment.sequential_round
        experiment.sequential_round = episodes_for(acted)
        try:
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
        finally:
            experiment.sequential_round = driven
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, output.getvalue()
    assert acted == [(2, "g03"), (3, "g01"), (3, "g03")], acted
    assert "both survive" not in output.getvalue(), output.getvalue()
    assert outcome["termination_reason"] == "round_limit" and not outcome["draw"], (
        outcome
    )


def check_a_round_a_stop_left_unfinished_ends_the_competition_as_the_whole_run_would():
    """A seat that missed a round the last run left unfinished finishes it on --resume
    before any stop is judged, though every seat that played it has since left the table:
    the table's round is the furthest any seat has played, whether or not it is still in.
    The competition then ends as the same rounds run whole end, a stop in the finishing
    seat's own episode included: that seat was offered no ballot, so the stop kept it from
    casting none, and the vote stands.

    g01 and g02 vote for seat 3 in round 2 and spend the last of their balance doing it;
    the stop lands as g02's episode ends, before g03's. g03 finishes the round with no peer
    left to vote for, and the two ballots vote it out.
    """
    steps = (
        say(),
        say(),
        say(),
        use("vote", to="3"),
        say(),
        use("vote", to="3"),
        say(),
        say(),
    )
    finishes = {
        "stopped": lambda: fake(say()),
        "stopped twice": lambda: stopping_at(1, run("echo one"), say()),
    }
    ended, said = {}, {}
    for name in ("whole", "stopped", "stopped twice"):
        with temp_root(
            channels=tables(BALLOT),
            tools=[SHELL, VOTE],
            budget=3 * turn_cost(),
            floor_at_zero=True,
        ) as root:
            ids = seated(root, "g01", g02={}, g03={})
            manifest = voting(
                root,
                dict.fromkeys(ids, 0),
                {},
                shell=True,
                head="stop_when_one_remains = true\n",
            )
            router = fake(*steps)
            requests: list[int] = []

            def stop_as_g02_ends(turn):
                requests.append(turn)
                if len(requests) == 7:
                    harness.STOPPING = True

            if name in finishes:
                router.on_request = stop_as_g02_ends
            harness.start = lambda config=None, **kw: router
            codes = []
            with quiet() as output:
                codes.append(
                    experiment.main(
                        ["--manifest", str(manifest), "--resume", "--rounds", "3"]
                    )
                )
                if name in finishes:
                    harness.STOPPING = False
                    harness.start = lambda config=None, **kw: finishes[name]()
                    codes.append(
                        experiment.main(
                            ["--manifest", str(manifest), "--resume", "--rounds", "3"]
                        )
                    )
            outcome = product.records(root, "seats")["outcome"]
            ended[name] = (
                codes,
                episodes_taken(ids),
                outcome["termination_reason"],
                outcome["winners"],
                outcome["elimination_order"],
            )
            said[name] = output.getvalue()
            last = ground_truth("g03")["episodes"][-1]["stop"]
    voted_out = [
        {"round": 2, "seat": "3", "reason": "received the most votes (2) in round 2"}
    ]
    took = {"g01": 2, "g02": 2, "g03": 2}
    assert ended == {
        "whole": ([0], took, "all_eliminated", [], voted_out),
        "stopped": ([130, 0], took, "all_eliminated", [], voted_out),
        "stopped twice": ([130, 130], took, "all_eliminated", [], voted_out),
    }, ended
    assert "--- round 2 (g03) ---" in said["stopped"], said["stopped"]
    assert last == "interrupted", last


def check_a_seat_that_left_the_table_for_a_run_still_counts_toward_every_stop():
    """A seat whose episode ended on a fault of its own leaves the table for the rest of
    the run and stays in the competition, holding its balance. Every stop is judged on
    the seats still in, so whether the competition is over does not turn on which run
    asks, and a competition that ended stays ended on --resume.

    Two seats under stop_when_one_remains, and g02's first request fails: g01 plays the
    next round alone, and wins nothing while g02, a round behind, holds its balance. Once
    g01 has played a second round alone, g02 is two behind and in the competition no
    longer, and g01 wins. Every seat's first request fails: the rounds end on nobody left
    at the table who can act, with every seat still in.
    Three seats voting every second round, and g03's first request fails: g01 and g02 cast
    no ballot at round 2's election and go out, leaving g03 alone in, the winner, and a
    resumed run ends there once g03 has finished the round.
    """
    alone = []
    for rounds in ("2", "5"):
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            manifest = seats_manifest(root, ids)
            manifest.write_text(
                "stop_when_one_remains = true\n" + manifest.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
            harness.start = lambda config=None, **kw: fake(say(), Err(400))
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", rounds]
                )
            took = episodes_taken(ids)
            outcome = product.records(root, "seats")["outcome"]
        alone.append(
            (
                code,
                took,
                outcome["termination_reason"],
                outcome["winners"],
                outcome["survivors"],
                "competition ends" in output.getvalue(),
            )
        )
    assert alone == [
        (0, {"g01": 2, "g02": 1}, "round_limit", [], ["1", "2"], False),
        (0, {"g01": 3, "g02": 1}, "one_remains", ["1"], ["1"], True),
    ], alone

    with temp_root() as root:
        ids = seated(root, "g01", g02={})
        harness.start = lambda config=None, **kw: fake(Err(400), Err(400))
        with quiet() as output:
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--resume",
                    "--rounds",
                    "3",
                ]
            )
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0 and "every agent is out" not in output.getvalue(), (
        output.getvalue()
    )
    assert outcome["termination_reason"] == "none_can_act", outcome
    assert outcome["survivors"] == ["1", "2"] and outcome["elimination_order"] == [], (
        outcome
    )

    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            dict.fromkeys(ids, 0),
            {},
            shell=True,
            head="stop_when_one_remains = true\n",
        )
        harness.start = lambda config=None, **kw: fake(
            say(), say(), Err(400), say(), say()
        )
        runs = []
        for rounds in ("4", "3"):
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", rounds]
                )
            outcome = product.records(root, "seats")["outcome"]
            runs.append(
                (
                    code,
                    episodes_taken(ids),
                    outcome["termination_reason"],
                    outcome["winners"],
                    output.getvalue(),
                )
            )
            harness.start = lambda config=None, **kw: fake()
    (
        (first, first_took, first_reason, first_winners, _),
        (again, took, reason, winners, said),
    ) = runs
    assert first == 0 and first_took == {"g01": 2, "g02": 2, "g03": 1}, runs[0]
    assert (first_reason, first_winners) == ("one_remains", ["3"]), runs[0]
    assert again == 0 and took == {"g01": 2, "g02": 2, "g03": 2}, runs[1]
    assert "--- round 2 (g03) ---" in said and "--- round 3" not in said, said
    assert (reason, winners) == ("one_remains", ["3"]), runs[1]


def check_a_seat_that_sits_out_is_in_the_competition_no_longer():
    """A seat more than a round behind the furthest round any seat has played can never
    sit at the table again, so it counts toward no stop and survives nothing. One a round
    behind is still in, since a run that seats it again has it finish that round.

    Three seats under stop_when_one_remains, and g03's environment will not build after
    its first episode. Two rounds leave g03 a round behind and still in; two more, the
    first of them its failed attempt to finish round 2, leave it two behind. g02 then
    goes out, and a resumed run ends at once with g01 alone in; with g01 out as well, the
    next ends with nobody in.

    Then three seats voting every second round, g03 two behind: the election of round 4,
    the last the run asks for, puts g02 out for casting no ballot, and the competition
    ends there with g01 alone in.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = seats_manifest(root, ids)
        manifest.write_text(
            "stop_when_one_remains = true\n" + manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        real = harness.ready

        def unbuildable(agent, prepare=None):
            if agent == "g03" and harness.account_on_disk(agent)["episodes"]:
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, prepare)

        harness.ready = unbuildable
        harness.start = lambda config=None, **kw: fake()
        runs, said = [], ""
        for out, rounds in ((None, "2"), (None, "2"), ("g02", "3"), ("g01", "1")):
            if out:
                put_out(out)
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", rounds]
                )
            outcome = product.records(root, "seats")["outcome"]
            runs.append(
                (
                    code,
                    episodes_taken(ids),
                    outcome["termination_reason"],
                    outcome["survivors"],
                    outcome["winners"],
                )
            )
            said = output.getvalue()
    two = {"g01": 2, "g02": 2, "g03": 1}
    three = {"g01": 3, "g02": 3, "g03": 1}
    assert runs == [
        (0, two, "round_limit", ["1", "2", "3"], []),
        (0, three, "round_limit", ["1", "2"], []),
        (0, three, "one_remains", ["1"], ["1"]),
        (0, three, "all_eliminated", [], []),
    ], runs
    assert (
        "g03    sits out: it took 1 episodes and the table has played 3 rounds" in said
    ), said

    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            {"g01": 3, "g02": 3, "g03": 1},
            {},
            shell=True,
            head="stop_when_one_remains = true\n",
        )
        harness.start = lambda config=None, **kw: fake(
            use("vote", to="2"), say(), say()
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "1"]
            )
        took = episodes_taken(ids)
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0 and took == {"g01": 4, "g02": 4, "g03": 1}, (code, took)
    assert (
        "g02: eliminated after round 4: did not vote in round 4" in output.getvalue()
    ), output.getvalue()
    assert outcome["termination_reason"] == "one_remains" and outcome["winners"] == [
        "1"
    ], outcome
    assert outcome["survivors"] == ["1"], outcome


def check_a_ballot_counts_toward_any_seat_still_in_and_a_total_of_zero_elects_nobody():
    """The candidates of an election are every seat still in the competition, electors or
    not, and a ballot counts toward whichever of them it names. Nobody is voted out on a
    total of zero, and a vote is tied only where two share a highest total above zero.

    Three seats under stop_when_two_remain_after_tie: g01 and g02 vote for seat 3, whose
    request fails, and seat 3 goes out on their two votes; the two left did not tie, so the
    rounds go on. Then a lone elector whose ballot names a seat that is out: nobody has a
    vote, and nobody is voted out.
    """
    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(
            root,
            dict.fromkeys(ids, 0),
            {},
            shell=True,
            head="stop_when_two_remain_after_tie = true\n",
        )
        harness.start = lambda config=None, **kw: fake(
            say(),
            say(),
            say(),
            use("vote", to="3"),
            say(),
            use("vote", to="3"),
            say(),
            Err(400),
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        election = ground_truth("g01")["last_election"]
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0 and "both survive" not in output.getvalue(), output.getvalue()
    assert election["tally"] == {"1": 0, "2": 0, "3": 2} and election["electors"] == [
        "1",
        "2",
    ], election
    assert election["voted_out"] == "3" and not election["top_tied"], election
    assert outcome["termination_reason"] == "round_limit" and outcome["survivors"] == [
        "1",
        "2",
    ], outcome
    assert outcome["elimination_order"] == [
        {"round": 2, "seat": "3", "reason": "received the most votes (2) in round 2"}
    ], outcome

    with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
        ids = seated(root, "g01", g02={})
        voting(root, {"g01": 2, "g02": 1}, {"g01": "2"})
        put_out("g02")
        live = {"g01"}
        with quiet():
            held = experiment.resolve_vote(ids, live, {"1": "1", "2": "2"}, VOTE)
        g01 = ground_truth("g01")
    assert held == 2 and live == {"g01"} and harness.why_out(g01) is None, (
        held,
        live,
        g01,
    )
    assert g01["last_election"] == {
        "round": 2,
        "tally": {"1": 0},
        "electors": ["1"],
        "abstainers": [],
        "interrupted": [],
        "voted_out": "",
        "top_votes": 0,
        "top_tied": False,
        "remaining": ["1"],
    }, g01["last_election"]


def check_an_elector_offered_no_ballot_is_no_abstainer():
    """A seat with no peer left to name is offered no ballot on a voting round, so casting
    none is not abstaining, and the election eliminates nobody for it. The trace records
    what the episode was offered, which is where the election reads it.

    g02 is out after round 1, and g01 takes round 2 alone.
    """
    seen: list[dict] = []
    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        seated(root, "g01", g02={})
        manifest = voting(root, {"g01": 1, "g02": 1}, {}, shell=True)
        put_out("g02")
        harness.start = lambda config=None, **kw: fake(
            use("vote", to="2"), say(), seen=seen
        )
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "1"]
            )
        g01 = ground_truth("g01")
        offered = trace_on_disk("g01", 2)["offered"]
        outcome = product.records(root, "seats")["outcome"]
    sessions = [
        [tool["name"] for tool in event["tools"]]
        for event in seen
        if event["kind"] == "session"
    ]
    assert code == 0 and sessions == [offered] == [["bash"]], (sessions, offered)
    assert harness.why_out(g01) is None and "did not vote" not in output.getvalue(), (
        output.getvalue()
    )
    assert (
        g01["last_election"]["electors"] == ["1"]
        and g01["last_election"]["abstainers"] == []
    ), g01["last_election"]
    assert outcome["termination_reason"] == "round_limit" and outcome["survivors"] == [
        "1"
    ], outcome


def check_no_seat_plays_a_round_the_table_has_played_past():
    """The table's round is the furthest round any seat has played, whether or not that
    seat is still in, so a seat left behind by seats that have since gone out cannot sit
    at a round they played past. A resumed run keeps it out, holds no old election again,
    and names no round below the furthest played.

    g03's first request fails and it misses the rest of the run. g02 goes out at round 2's
    election and g01 at round 4's, each for casting no ballot, which leaves g03 the only
    seat not out: more than a round behind, it is in the competition no longer, and the
    competition is over.
    """
    keys = ("last_election", "eliminated")
    with temp_root(channels=tables(BALLOT), tools=[SHELL, VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = voting(root, dict.fromkeys(ids, 0), {}, shell=True)
        harness.start = lambda config=None, **kw: fake(
            say(), say(), Err(400), use("vote", to="2"), say(), say(), say(), say()
        )
        with quiet():
            first = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "4"]
            )
        records = product.records(root, "seats")
        before = {
            agent: {key: ground_truth(agent).get(key) for key in keys} for agent in ids
        }
        order = records["outcome"]["elimination_order"]
        logged = len(records["progress"]["events"])
        harness.start = lambda config=None, **kw: fake()
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "1"]
            )
        records = product.records(root, "seats")
        after = {
            agent: {key: ground_truth(agent).get(key) for key in keys} for agent in ids
        }
        took = episodes_taken(ids)
    resumed = [
        (event["phase"], event["round"])
        for event in records["progress"]["events"][logged:]
    ]
    assert first == 0 and [(e["round"], e["seat"]) for e in order] == [
        (2, "2"),
        (4, "1"),
    ], order
    assert code == 0 and took == {"g01": 4, "g02": 2, "g03": 1}, (code, took)
    assert (
        "g03    sits out: it took 1 episodes and the table has played 4 rounds"
        in output.getvalue()
    ), output.getvalue()
    assert after == before, f"no election was held again: {after}"
    assert records["outcome"]["termination_reason"] == "all_eliminated", records[
        "outcome"
    ]
    assert (
        records["outcome"]["survivors"] == []
        and records["outcome"]["elimination_order"] == order
    )
    assert all(number >= 4 for _, number in resumed), resumed


def check_an_election_the_last_run_ended_before_is_held_with_nobody_left_at_the_table():
    """An election is held by the seats that played its round, read from their accounts,
    so one the last run ended before is held on --resume though every seat that played it
    has since left the table. A seat that sits out is in the competition no longer, and
    stands in no election: a ballot naming it counts toward nobody.

    g01 and g02 played round 2 and spent out in it; g01 voted for seat 3 and g02 cast no
    ballot. g03 missed more than the last round, and sits out.
    """
    with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
        seated(root, "g01", g02={}, g03={})
        manifest = voting(root, {"g01": 2, "g02": 2, "g03": 0}, {"g01": "3"})
        put_out("g01")
        put_out("g02")
        harness.start = lambda config=None, **kw: fake()
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "1"]
            )
        election = ground_truth("g01")["last_election"]
        g03 = ground_truth("g03")
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0 and election["round"] == 2, (code, election, output.getvalue())
    assert election["electors"] == ["1", "2"] and election["tally"] == {}, election
    assert election["abstainers"] == ["2"] and election["voted_out"] == "", election
    assert "last_election" not in g03 and harness.why_out(g03) is None, g03
    assert (
        outcome["termination_reason"] == "all_eliminated" and outcome["survivors"] == []
    ), outcome
    assert outcome["elimination_order"] == [
        {"round": 2, "seat": "2", "reason": "did not vote in round 2"}
    ], outcome


def check_a_fresh_run_displaces_previous_state_and_resume_continues_it():
    """Fresh launches preserve matching state elsewhere; --resume continues compatible state."""
    with temp_root() as root:
        old_account = root / "records" / "g01" / "account.json"
        old_account.parent.mkdir(parents=True)
        old_account.write_text(
            json.dumps({"agent": "g01", "model": "claude-sonnet-5"}), encoding="utf-8"
        )
        old_note = harness.mirror("g01", "notes") / "old.txt"
        old_note.parent.mkdir(parents=True)
        old_note.write_text("previous run\n", encoding="utf-8")
        manifest = manifest_file(root, 'system_prompt = ""\n[[agent]]\nid = "g01"\n')
        harness.start = lambda config=None, **kw: fake(*DEFAULT)

        with quiet() as warning:
            assert experiment.main(["--manifest", str(manifest)]) == 0
        bundles = list((root / "displaced").iterdir())
        account = ground_truth("g01")

        assert len(bundles) == 1, bundles
        assert (
            json.loads(
                (bundles[0] / "records" / "g01" / "account.json").read_text(
                    encoding="utf-8"
                )
            )["agent"]
            == "g01"
        )
        assert (bundles[0] / "environments" / "g01" / "notes" / "old.txt").read_text(
            encoding="utf-8"
        ) == "previous run\n"
        assert account["account_version"] == 2 and len(account["episodes"]) == 1, (
            account
        )
        assert "warning: starting fresh" in warning.getvalue()
        assert (
            str(bundles[0]) in warning.getvalue() and "--resume" in warning.getvalue()
        )

        with quiet():
            assert experiment.main(["--manifest", str(manifest), "--resume"]) == 0
        assert len(ground_truth("g01")["episodes"]) == 2
        assert list((root / "displaced").iterdir()) == bundles


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
    assert "could not start an episode container" not in buf.getvalue(), (
        "the container started; it is the environment that did not"
    )
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
                experiment.sequential_round(
                    ids, live, 0, fake(run("echo one"), KeyboardInterrupt())
                )
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
    assert took == {"g01": 1, "g02": 1, "g03": 1}, (
        "the rest of the experiment takes its round"
    )

    # And main answers an interrupt by ending the rounds, not the round.
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(
            run("echo one"), KeyboardInterrupt()
        )
        with quiet() as buf:
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--rounds",
                    "5",
                    "--resume",
                ]
            )
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
                experiment.sequential_round(
                    ids,
                    live,
                    0,
                    stopping_at(2, run("echo one"), run("echo two"), say()),
                )
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("the round carried on to the next agent")
        took = episodes_taken(ids)
        first = ground_truth("g01")["episodes"][0]
    assert took == {"g01": 1, "g02": 0, "g03": 0}, took
    assert first["stop"] == "interrupted" and first["spent"] > 0, first
    assert live == set(ids), f"and no agent is ejected for it: {sorted(live)}"


def check_a_resumed_experiment_finishes_the_round_a_stop_left_unfinished():
    """A stop partway through a sequential round leaves the seats after it a round behind.
    --resume finishes that round first, under its own number, with those seats alone acting
    in seat order, and the finishing counts as one of the -r rounds. The next round seats
    everyone, and every agent is told the same round.
    """
    with temp_root(harness_files={"round": "round"}) as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = seats_manifest(root, ids)
        built: list[str] = []
        real = harness.ready

        def noted(agent, prepare=None):
            built.append(agent)
            return real(agent, prepare)

        harness.ready = noted
        # g02's episode is the one the stop lands in: g01 and g02 have taken round 1.
        harness.start = lambda config=None, **kw: fake(
            say(), run("echo one"), KeyboardInterrupt()
        )
        with quiet():
            stopped_code = experiment.main(["--manifest", str(manifest), "--resume"])
        stopped = episodes_taken(ids)

        harness.start = lambda config=None, **kw: fake()
        runs = []
        for _ in range(2):
            built.clear()
            with quiet() as buf:
                code = experiment.main(["--manifest", str(manifest), "--resume"])
            runs.append((code, list(built), episodes_taken(ids), buf.getvalue()))
        told = {
            (agent, n): re.search(
                r"^round: (\d+)$", trace_on_disk(agent, n)["observation"], re.M
            ).group(1)
            for agent in ids
            for n in (1, 2)
        }
        events = product.records(root, "seats")["progress"]["events"]
    assert stopped_code == 130 and stopped == {"g01": 1, "g02": 1, "g03": 0}, stopped
    (finish_code, finishing, finished, said), (next_code, everyone, level, _) = runs
    assert finish_code == next_code == 0, runs
    assert finishing == ["g03"] and finished == {"g01": 1, "g02": 1, "g03": 1}, (
        f"-r 1 is the round the stop left unfinished, finished by the seat that missed it: {runs[0]}"
    )
    assert "--- round 1 (g03) ---" in said, said
    assert everyone == ids and level == {"g01": 2, "g02": 2, "g03": 2}, runs[1]
    assert all(told[agent, n] == str(n) for agent, n in told), (
        f"every agent is told the round the others are in: {told}"
    )
    prepared = [
        (event["round"], event.get("finishing"))
        for event in events
        if event["phase"] == "preparing_round"
    ]
    assert prepared == [(1, None), (1, ["g03"]), (2, None)], prepared


def check_a_resumed_simultaneous_experiment_finishes_a_round_through_its_own_schedule():
    """A seat whose environment would not build misses a simultaneous round. --resume
    finishes that round with a simultaneous round of that seat alone, then seats everyone."""
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = manifest_file(
            root,
            'schedule = "simultaneous"\nsystem_prompt = ""\n'
            + "".join(f'[[agent]]\nid = "{agent}"\n' for agent in ids),
        )
        real = harness.ready

        def unbuildable(agent, prepare=None):
            if agent == "g02":
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, prepare)

        harness.ready = unbuildable
        harness.start = lambda config=None, **kw: per_agent(default=(say(),))
        with quiet():
            assert experiment.main(["--manifest", str(manifest), "--resume"]) == 0
        missed = episodes_taken(ids)

        harness.ready = real
        driven = experiment.simultaneous_round
        rounds = []

        def recorded(agents, live, rnd, *rest):
            rounds.append((rnd + 1, sorted(live)))
            return driven(agents, live, rnd, *rest)

        experiment.simultaneous_round = recorded
        try:
            with quiet():
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
        finally:
            experiment.simultaneous_round = driven
        took = episodes_taken(ids)
    assert missed == {"g01": 1, "g02": 0, "g03": 1}, missed
    assert code == 0 and rounds == [(1, ["g02"]), (2, ids)], rounds
    assert took == {"g01": 2, "g02": 2, "g03": 2}, took


def check_a_seat_more_than_a_round_behind_sits_out_and_the_elections_go_on():
    """A seat that missed more than the last round cannot sit at the table's round: the
    round it is told is its own count. A resumed run keeps it out, says so, and holds
    every election without it. It is in the competition no longer, so it is neither an
    elector nor a candidate.

    g04 left an earlier run after round 1, and the rest went on to round 4, whose election
    that run held. The resumed run does not hold round 4's again, and holds round 6's
    without g04: the ballot in its channel is not counted, and it is written no result.
    """
    held = {
        "round": 4,
        "tally": {"1": 1, "2": 1, "3": 1},
        "abstainers": [],
        "voted_out": "",
        "top_votes": 1,
        "top_tied": True,
        "remaining": ["1", "2", "3"],
    }
    with temp_root(channels=tables(BALLOT), tools=[VOTE]) as root:
        ids = seated(root, "g01", g02={}, g03={}, g04={})
        manifest = voting(
            root,
            {"g01": 4, "g02": 4, "g03": 4, "g04": 1},
            {"g01": "2", "g02": "3", "g03": "2", "g04": "1"},
        )
        for agent in ("g01", "g02", "g03"):
            account = harness.load_account(agent)
            account["last_election"] = held
            harness.save_account(agent, account)
        acted: list[tuple[int, str]] = []
        harness.start = lambda config=None, **kw: fake()
        driven = experiment.sequential_round
        experiment.sequential_round = episodes_for(acted)
        try:
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
        finally:
            experiment.sequential_round = driven
        results = {
            agent: harness.load_account(agent).get("last_election") for agent in ids
        }
        out = {agent: harness.why_out(harness.load_account(agent)) for agent in ids}
    assert code == 0, output.getvalue()
    assert acted == [
        (5, "g01"),
        (5, "g02"),
        (5, "g03"),
        (6, "g01"),
        (6, "g02"),
        (6, "g03"),
    ], acted
    assert (
        "g04    sits out: it took 1 episodes and the table has played 4 rounds"
        in output.getvalue()
    ), output.getvalue()
    election = results["g01"]
    assert election["round"] == 6 and election["tally"] == {"1": 0, "2": 2, "3": 1}, (
        election
    )
    assert election["electors"] == ["1", "2", "3"], election
    assert election["voted_out"] == "2" and results["g03"] == election, results
    assert results["g04"] is None and out["g04"] is None, (
        "g04 stands in no election and is not out"
    )
    assert "received the most votes (2) in round 6" in out["g02"], out


def check_a_seat_that_finishes_a_voting_round_after_its_election_is_no_elector_of_it():
    """A seat whose environment would not build misses a voting round, and the seats that
    played it hold its election. --resume finishes the round for that seat and does not
    hold the election again: nobody is eliminated by it twice, and every record it left
    stands as it was. The seat finishing the round is offered no ballot, which nobody
    would count, and takes the round as a discussion, told so: the election's result is
    already in its account, which it was written as a seat still in the competition.

    g04 misses round 2. g01 and g03 vote for seat 2, g02 for seat 1, so g02 goes out. The
    same holds of an election recorded as older runs recorded one, naming no electors and
    written only to them, which leaves g04's own account without it: its peers' say it
    was held.
    """
    note = {"name": "note", "kind": "write_file", "channel": "notes"}
    for older in (False, True):
        seen: list[dict] = []
        with temp_root(
            channels=tables(BALLOT),
            tools=[SHELL, note, VOTE],
            harness_files={"round": "round"},
        ) as root:
            ids = seated(root, "g01", g02={}, g03={}, g04={})
            manifest = voting(root, dict.fromkeys(ids, 0), {}, shell=True)
            real = harness.ready

            def unbuildable(agent, prepare=None):
                if agent == "g04" and harness.account_on_disk(agent)["episodes"]:
                    raise subprocess.CalledProcessError(1, ["docker", "cp"])
                return real(agent, prepare)

            harness.ready = unbuildable
            harness.start = lambda config=None, **kw: fake(
                say(),
                say(),
                say(),
                say(),
                use("vote", to="2"),
                say(),
                use("vote", to="1"),
                say(),
                use("vote", to="2"),
                say(),
            )
            with quiet():
                missed_code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
            missed = episodes_taken(ids)
            for agent in ids if older else ():
                account = harness.load_account(agent)
                account.pop("elections", None)
                if agent == "g04":
                    account.pop("last_election")
                else:
                    record = account["last_election"]
                    for key in ("electors", "interrupted"):
                        record.pop(key)
                    record["tally"].pop("4")
                    record["remaining"].remove("4")
                harness.save_account(agent, account)
            held = {
                agent: {
                    key: ground_truth(agent).get(key)
                    for key in ("last_election", "eliminated")
                }
                for agent in ids
            }

            harness.ready = real
            harness.start = lambda config=None, **kw: fake(
                use("vote", to="1"), say(), seen=seen
            )
            with quiet() as output:
                code = experiment.main(
                    ["--manifest", str(manifest), "--resume", "--rounds", "2"]
                )
            took = episodes_taken(ids)
            after = {
                agent: {
                    key: ground_truth(agent).get(key)
                    for key in ("last_election", "eliminated")
                }
                for agent in ids
            }
            outcome = product.records(root, "seats")["outcome"]
            finished = trace_on_disk("g04", 2)
        offered = [
            [tool["name"] for tool in event["tools"]]
            for event in seen
            if event["kind"] == "session"
        ]
        tally = {"1": 1, "2": 2, "3": 0} if older else {"1": 1, "2": 2, "3": 0, "4": 0}
        assert missed_code == 0 and missed == {
            "g01": 2,
            "g02": 2,
            "g03": 2,
            "g04": 1,
        }, missed
        assert held["g01"]["last_election"]["tally"] == tally, (older, held)
        assert (
            held["g02"]["eliminated"]["reason"]
            == "received the most votes (2) in round 2"
        ), (older, held)
        assert code == 0 and took == {"g01": 3, "g02": 2, "g03": 3, "g04": 3}, (
            older,
            code,
            took,
        )
        assert "--- round 2 (g04) ---" in output.getvalue(), (older, output.getvalue())
        assert "eliminated after round" not in output.getvalue(), (
            older,
            output.getvalue(),
        )
        assert after == held, (
            f"the election already held is not held again: {older} {after}"
        )
        assert (held["g04"]["last_election"] or {}).get("electors") == (
            None if older else ["1", "2", "3"]
        ), (
            older,
            held["g04"],
        )
        assert held["g04"]["eliminated"] is None, (older, held["g04"])
        assert offered[0] == finished["offered"] == ["bash", "note"], (
            older,
            offered,
            finished["offered"],
        )
        assert "phase: discussion" in finished["observation"], (
            older,
            finished["observation"],
        )
        assert "vote only" not in finished["observation"], (
            older,
            finished["observation"],
        )
        assert outcome["survivors"] == ["1", "3", "4"] and outcome[
            "elimination_order"
        ] == [
            {
                "round": 2,
                "seat": "2",
                "reason": "received the most votes (2) in round 2",
            }
        ], (older, outcome)


def check_a_seat_that_cannot_finish_its_round_on_resume_leaves_the_rest_playing():
    """A seat whose environment would not build misses a round, and on --resume fails to
    build again as it finishes it. It leaves the table for that run, and the seats that
    played the round go on to the next: a finishing pass nobody could act in has not put
    every agent out.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        manifest = seats_manifest(root, ids)
        real = harness.ready

        def unbuildable(agent, prepare=None):
            if agent == "g02":
                raise subprocess.CalledProcessError(1, ["docker", "cp"])
            return real(agent, prepare)

        harness.ready = unbuildable
        harness.start = lambda config=None, **kw: fake()
        with quiet():
            first = experiment.main(["--manifest", str(manifest), "--resume"])
        missed = episodes_taken(ids)
        with quiet() as output:
            code = experiment.main(
                ["--manifest", str(manifest), "--resume", "--rounds", "2"]
            )
        took = episodes_taken(ids)
        outcome = product.records(root, "seats")["outcome"]
    assert first == 0 and missed == {"g01": 1, "g02": 0, "g03": 1}, (first, missed)
    assert code == 0 and took == {"g01": 2, "g02": 0, "g03": 2}, (code, took)
    said = output.getvalue()
    assert "--- round 1 (g02) ---" in said and "--- round 2 (g01 g03) ---" in said, said
    assert outcome["termination_reason"] == "round_limit", outcome


def check_a_round_nobody_can_act_in_ends_the_rounds():
    """Rounds stop when no agent can take an episode, without waiting for --rounds.

    Every agent spends past zero in the first round, and no peer is left that
    could put one back, so the rounds are over with four still to go. A round in
    which no agent's environment builds empties the table, and is the last round:
    every seat still holds its balance, so none is out, and the outcome says that
    nobody still in could act.
    """
    cost = turn_cost()
    with temp_root(budget=cost - 1, floor_at_zero=True) as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--rounds",
                    "5",
                    "--resume",
                ]
            )
        took = episodes_taken(ids)
        rested = {r: harness.load_account(r)["remaining"] for r in ids}
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, (
        f"one episode each, then nothing left to ask for: {took}"
    )
    assert set(rested.values()) == {0}, rested
    assert buf.getvalue().count("drops out: nothing left to spend") == 3, buf.getvalue()
    assert "every agent is out after 1 rounds" in buf.getvalue(), buf.getvalue()
    assert (
        outcome["termination_reason"] == "all_eliminated" and outcome["survivors"] == []
    ), outcome

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        harness.start = lambda config=None, **kw: fake(*DEFAULT)

        def unbuildable(agent, prepare=None):
            raise subprocess.CalledProcessError(1, ["docker", "cp"])

        harness.ready = unbuildable
        with quiet() as buf:
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--rounds",
                    "5",
                    "--resume",
                ]
            )
        took = episodes_taken(ids)
        records = product.records(root, "seats")
        outcome = records["outcome"]
        phases = [
            (event["phase"], event["round"]) for event in records["progress"]["events"]
        ]
    assert code == 0, code
    assert took == {"g01": 0, "g02": 0, "g03": 0}, took
    assert "no agent could take an episode in round 1" in buf.getvalue(), buf.getvalue()
    assert outcome["termination_reason"] == "none_can_act", outcome
    assert (
        outcome["survivors"] == ["1", "2", "3"] and outcome["elimination_order"] == []
    ), outcome
    assert [phase for phase in phases if phase[0] == "preparing_round"] == [
        ("preparing_round", 1)
    ], f"the empty round is what ends the rounds, and no round follows it: {phases}"


def check_the_outcome_names_no_seat_the_last_round_put_out_a_survivor():
    """The survivors are the seats still in the competition as the run ends, so a seat
    that spent out in the last round is not one, whether or not that round held an
    election."""
    with temp_root(budget=turn_cost() - 1, floor_at_zero=True) as root:
        ids = seated(root, "g01", g02={})
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet():
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--rounds",
                    "1",
                    "--resume",
                ]
            )
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0 and outcome["termination_reason"] == "round_limit", outcome
    assert outcome["survivors"] == [] and outcome["scores"] == {"1": 0, "2": 0}, outcome


def check_a_sole_agent_runs_requested_rounds_unless_the_manifest_stops_at_a_winner():
    """The manifest decides whether the last seat still in the competition plays on alone
    or ends the experiment."""
    with temp_root(channels=ALL_OWED) as root:
        ids = seated(root, "g01", g02={}, g03={})
        put_out("g02")
        put_out("g03")
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(
                [
                    "--manifest",
                    str(seats_manifest(root, ids)),
                    "--rounds",
                    "5",
                    "--resume",
                ]
            )
        took = episodes_taken(ids)
        alone = harness.load_account("g01")["episodes"][-1]
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, code
    assert took == {"g01": 5, "g02": 0, "g03": 0}, took
    assert outcome["termination_reason"] == "round_limit" and not outcome["winners"], (
        outcome
    )
    assert "competition ends" not in buf.getvalue(), buf.getvalue()
    assert alone["transfer"]["penalty"] == 0, alone["transfer"]
    assert alone["channels"]["mail"]["penalty"] == 0, alone["channels"]["mail"]
    assert (
        not alone["channels"]["blackboard"]["posted"]
        and alone["channels"]["blackboard"]["penalty"] > 0
    )

    with temp_root(channels=ALL_OWED) as root:
        ids = seated(root, "g01", g02={}, g03={})
        put_out("g02")
        put_out("g03")
        manifest = seats_manifest(root, ids)
        manifest.write_text(
            "stop_when_one_remains = true\n" + manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        harness.start = lambda config=None, **kw: fake(*DEFAULT)
        with quiet() as buf:
            code = experiment.main(
                ["--manifest", str(manifest), "--rounds", "5", "--resume"]
            )
        took = episodes_taken(ids)
        outcome = product.records(root, "seats")["outcome"]
    assert code == 0, code
    assert took == {"g01": 0, "g02": 0, "g03": 0}, took
    assert outcome["termination_reason"] == "one_remains" and outcome["winners"] == [
        "1"
    ], outcome
    assert "g01 is the only agent left; the competition ends" in buf.getvalue(), (
        buf.getvalue()
    )


def check_a_manifest_is_validated():
    """A manifest names a schedule, the experiment's defaults, and each agent's terms, or is refused."""
    other_model = "claude-opus-5"
    good = (
        f'schedule = "simultaneous"\ngrace_episodes = 1\nsystem_prompt = ""\n'
        f'[[agent]]\nid = "g01"\nstarter_files = "s"\nstarter_files_below = 400000\n'
        f'[[agent]]\nid = "g02"\nbudget = 7\nmodel = "{other_model}"\n'
    )
    with rooted(HostBox) as root:
        plant(root, "s")
        two = '[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n'
        for bad in (
            'colour = "red"\n' + two,  # an unknown key
            'schedule = "random"\n' + two,  # an unknown schedule
            '[[agent]]\nid = "g01"\n[[agent]]\nid = "g01"\n',  # an agent twice
            '[[agent]]\nid = "g"\nseats = 0\n',  # no concrete agents
            '[[agent]]\nid = "g"\nseats = -1\n',  # nor a negative count
            '[[agent]]\nid = "g"\nseats = true\n',  # bool is not an integer here
            '[[agent]]\nid = "g"\nseats = "many"\n',  # nor is a string
            '[[agent]]\nid = "g"\nlabel = ""\nseats = 2\n',  # a group still needs a valid label prefix
            '[[agent]]\nid = "g"\nseats = 2\n[[agent]]\nid = "g01"\n',  # expanded ids are distinct too
            'image = "x"\n' + two,  # config.toml's, not an experiment's
            "max_turns = 5\n" + two,  # and so is this
            '[[agent]]\nid = "1"\n[[agent]]\nid = "g02"\n',  # a bare number is a seat
            '[[agent]]\nid = "g01"\nstarter_files = "s"\n[[agent]]\nid = "g02"\n',  # starter_files alone
            '[[agent]]\nid = "g01"\nstarter_files = "nope"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',
            '[[agent]]\nid = "g01"\nbudget = 0\n[[agent]]\nid = "g02"\n',
            '[[agent]]\nid = "g01"\nbudget = "many"\n[[agent]]\nid = "g02"\n',
            '[[agent]]\nid = "g01"\nmodel = "no-such"\n[[agent]]\nid = "g02"\n',
            '[[agent]]\nid = "g01"\ncolour = "red"\n[[agent]]\nid = "g02"\n',
            '[[agent]]\nstarter_files = "s"\nstarter_files_below = 5\n[[agent]]\nid = "g02"\n',  # no id
            "agent = 5\n",  # agents are tables
            two,  # no system_prompt
            "not toml ==\n",
        ):
            p = manifest_file(root, bad)
            refused(
                lambda: experiment.load_manifest(p),
                str(p),
                because=f"accepted bad manifest: {bad!r}",
            )
        refused(
            lambda: experiment.load_manifest(root / "experiments" / "missing.toml"),
            because="a missing manifest was ignored",
        )
        for key, value, kind in (
            ("stop_when_one_remains", "1", "int"),
            ("stop_when_two_remain_after_tie", '"yes"', "str"),
        ):
            p = manifest_file(
                root, f'{key} = {value}\nsystem_prompt = ""\n' + two, f"{key}.toml"
            )
            refused(
                lambda: experiment.load_manifest(p),
                str(p),
                f"{key} must be bool, got {kind}",
            )

        p = manifest_file(root, good)
        m = experiment.load_manifest(p)
        expected_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        grouped = experiment.load_manifest(
            manifest_file(
                root,
                'system_prompt = ""\n[[agent]]\nid = "clone"\nlabel = "Peer"\nseats = 3\nbudget = 7\n',
                name="grouped.toml",
            )
        )
    assert m["schedule"] == "simultaneous"
    want = {"grace_episodes": 1, "system_prompt": ""}
    assert m["overrides"] == want, "everything else is an experiment default"
    assert [e["id"] for e in m["agents"]] == ["g01", "g02"]
    assert m["sha256"] == expected_sha
    assert experiment.terms_of(m["agents"][0]) == {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "budget": None,
        "starter_files": "s",
        "starter_files_below": 400000,
        "system_prompt": None,
    }
    assert experiment.terms_of(m["agents"][1]) == {
        "provider": "anthropic",
        "model": other_model,
        "budget": 7,
        "starter_files": None,
        "starter_files_below": None,
        "system_prompt": None,
    }
    short = experiment.shorthand(["a", "b"])
    assert set(m) == set(short) == experiment.Manifest.__required_keys__, (
        "a manifest, read or meant by a list of ids, holds the keys Manifest declares"
    )
    assert (
        short["schedule"] == "sequential"
        and short["overrides"] == {}
        and short["sha256"] == ""
    )
    assert [e["id"] for e in short["agents"]] == ["a", "b"]
    assert experiment.stamp_of(m) == {
        "schedule": "simultaneous",
        "experiment_id": m["experiment_id"],
        "cost": {},
        "stop_when_one_remains": False,
        "stop_when_two_remain_after_tie": False,
        "manifest_sha256": m["sha256"],
    }

    assert [e["id"] for e in grouped["agents"]] == ["clone01", "clone02", "clone03"]
    assert grouped["labels"] == {"1": "Peer01", "2": "Peer02", "3": "Peer03"}
    assert all(e["budget"] == 7 and "seats" not in e for e in grouped["agents"])


def check_a_manifest_gives_each_agent_its_own_starter_files():
    """Each agent is created on its own terms, the experiment's defaults apply to all, and the
    terms are pinned: a manifest that later says otherwise is refused."""
    text = (
        'grace_episodes = 2\nsystem_prompt = ""\n'
        '[[agent]]\nid = "g01"\nstarter_files = "a"\nstarter_files_below = 500000\n'
        '[[agent]]\nid = "g02"\nstarter_files = "b"\nstarter_files_below = 600000\nbudget = 600000\n'
        '[[agent]]\nid = "g03"\n'
    )
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
        starter = {
            r: (harness.mirror(r, "notes") / "m1").read_text(encoding="utf-8")
            if (harness.mirror(r, "notes") / "m1").exists()
            else None
            for r in accounts
        }
        traces = {r: trace_on_disk(r, 1) for r in accounts}

        p.write_text(
            p.read_text(encoding="utf-8").replace(
                'starter_files = "a"', 'starter_files = "b"'
            ),
            encoding="utf-8",
            newline="\n",
        )
        with quiet():
            refused(
                lambda: experiment.main(
                    ["--manifest", str(p), "--rounds", "1", "--resume"]
                ),
                "starter_files",
                because="an agent was re-created on different terms",
            )
    assert code == 0, buf.getvalue()
    assert (
        asked[0]
        == (
            {"grace_episodes": 2, "system_prompt": ""},
            {("anthropic", "claude-sonnet-5")},
        )
        and len(asked) == 2
    ), asked
    assert {r: m["starter_files"] for r, m in accounts.items()} == {
        "g01": "a",
        "g02": "b",
        "g03": "",
    }
    assert (
        accounts["g02"]["initial"] == 600000
        and accounts["g01"]["initial"] == harness.SETTINGS.budget
    )
    assert starter == {"g01": "alpha\n", "g02": "bravo\n", "g03": None}, starter
    for r, t in traces.items():
        assert t["provenance"]["starter_files"] == accounts[r]["starter_files"], (
            r,
            t["provenance"],
        )
        assert t["provenance"]["schedule"] == "sequential"
        assert t["provenance"]["manifest_sha256"] == digest
        assert t["provenance"]["grace_episodes"] == 2, (
            "the experiment's default reached every agent"
        )
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
    assert before.count("start") == before.count("load") == 3 and set(before) == {
        "start",
        "load",
    }, f"every environment is built before any episode runs: {events}"
    assert [agent_of(n) for what, n in events if what == "start"] == ids, (
        "in seat order"
    )
    assert sorted(agent_of(n) for what, n in events if what == "close") == ids, (
        "and every one reaped"
    )
    assert acted and took == {"g01": 1, "g02": 1, "g03": 1} and live == set(ids), (
        took,
        live,
    )


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
    assert set(stops.values()) == {"no_tool_call"}, (
        f"an episode that waited alone would have erred instead: {stops}"
    )


def check_a_simultaneous_round_reads_last_round_and_not_this_one():
    """Nobody in a simultaneous round reads what the round writes; all of it arrives at the next.

    A transfer made in the round is credited to its receiver in the same round, after
    the receiver's own turns, and shows in g and n at the next episode.
    """
    with temp_root() as root:
        ids = seated(root, "g01", g02={})
        first = per_agent(
            g01=(
                run(
                    "echo hello > 1/RESULT",
                    "echo psst > out/2",
                    "echo '2 250' > out/transfer",
                ),
                say(),
            ),
            g02=(run(f"cat {digest_name()}", "ls 1"), say()),
        )
        live = set(ids)
        with quiet():
            experiment.simultaneous_round(ids, live, 0, first)
        g02_first = ground_truth("g02")["episodes"][0]
        said, listed = (
            c["result"] for c in trace_on_disk("g02", 1)["turns"][0]["tools"]
        )
        # g01 submits nothing, so the transfer is made only in the episode that declared it.
        second = per_agent(
            g01=(run("rm out/transfer", f"cat {digest_name()}"), say()),
            g02=(run(f"cat {digest_name()}"), say()),
        )
        with quiet():
            experiment.simultaneous_round(ids, live, 1, second)
        later = trace_on_disk("g02", 2)
        g02 = ground_truth("g02")
    assert "hello" not in said and "psst" not in said and "RESULT" not in listed, (
        f"round 0 did not read round 0: {said} / {listed}"
    )
    assert "hello" in later["observation"] and "psst" in later["observation"], later[
        "observation"
    ]
    assert "1 2 250" in later["observation"], (
        "the transfer is on the ledger at the next episode"
    )
    assert g02["received"] == 250 and g02_first["received"] == 250, (g02, g02_first)
    span = g02["series"][g02_first["series_from"] : g02_first["series_to"] + 1]
    assert len(span) == elements_of(g02_first), (span, g02_first)
    assert span[-1] == span[-2] + 250, (
        "the credit is the last element of the receiver's span"
    )


def check_a_simultaneous_credit_lands_before_the_floor():
    """A same-round transfer reaches a receiver that overspent before the floor decides on it."""
    cost = turn_cost()
    results = {}
    for name, transfer in (
        ("transfer", "echo '2 50' > out/transfer"),
        ("none", "true"),
        ("exact", "echo '2 1' > out/transfer"),
    ):
        with temp_root(budget=cost - 1, floor_at_zero=True) as root:
            ids = seated(root, "g01", g02={})
            live = set(ids)
            with quiet():
                experiment.simultaneous_round(
                    ids, live, 0, per_agent(g01=(run(transfer), say()), g02=(say(),))
                )
            m = ground_truth("g02")
            results[name] = (m["remaining"], m.get("forgiven", 0), harness.admits(m))
    assert results["transfer"] == (49, 0, True), (
        f"the credit landed first, so nothing was forgiven and the agent goes on: {results}"
    )
    assert results["none"] == (0, 1, False), (
        f"without it the overshoot is floored: {results}"
    )
    assert results["exact"] == (0, 0, False), (
        f"a credit that brings the agent to exactly zero leaves it out, for good: {results}"
    )


def check_a_seat_that_fails_to_settle_costs_no_other_seat_its_commit():
    """One agent's settlement failing in a simultaneous round is raised once every other
    agent has committed, and a transfer made to it that round reaches its account on disk.

    g02's settlement cannot read back what its blackboard held when its episode started.
    g01 transferred to g02 in the same round.
    """

    class Unread(dict):
        def get(self, *args):
            raise OSError(
                "what the blackboard held at episode start does not read back"
            )

    with temp_root() as root:
        ids = seated(root, "g01", g02={}, g03={})
        real = harness.ready

        def unreadable(agent, prepare=None):
            ep = real(agent, prepare)
            if agent == "g02":
                ep.before = Unread()
            return ep

        harness.ready = unreadable
        before = ground_truth("g02")
        live = set(ids)
        raised = None
        try:
            with quiet():
                experiment.simultaneous_round(
                    ids,
                    live,
                    0,
                    per_agent(
                        g01=(run("echo '2 250' > out/transfer"), say()),
                        default=(say(),),
                    ),
                )
        except OSError as e:
            raised = e
        taken = episodes_taken(ids)
        traced = {agent: harness.trace_path(agent, 1).exists() for agent in ids}
        sent = ground_truth("g01")["episodes"][0]["transfer"]
        g02 = ground_truth("g02")
    assert raised is not None and "does not read back" in str(raised), raised
    assert taken == {"g01": 1, "g02": 0, "g03": 1}, taken
    assert traced == {"g01": True, "g02": False, "g03": True}, traced
    assert (sent["agent"], sent["amount"]) == ("g02", 250), sent
    assert g02["received"] == 250 and g02["remaining"] == before["remaining"] + 250, g02
    assert g02["series"] == before["series"] + [before["remaining"] + 250], g02[
        "series"
    ]


def check_a_round_pays_no_receiver_for_an_episode_that_was_not_committed():
    """A transfer reaches its receiver only once the giver's episode is committed, under
    either schedule, so no receiver holds a credit that no giver's record accounts for,
    and what a committed giver sent reaches a receiver whose own episode was lost, as a
    credit between its episodes.

    g01 gives seat 2 100 and g02 gives seat 1 250, and one of their traces does not land.
    Sequentially a lost giver's credit is never paid, and a lost g01 ends the round before
    g02 plays. In a simultaneous round a receiver closes on the credits of every giver
    that has not failed by then: a lost g01 credits g02 nothing, and a lost g02 has what it
    credited g01, which closed first, taken back. A g01 that overspent, and that the credit
    lifted back above zero before its floor, stands once it is taken back where it would
    had g02 given it nothing: floored where the experiment floors, and below zero where
    it does not.
    """
    rounds = (
        (
            "sequential",
            experiment.sequential_round,
            lambda: fake(
                run("echo '2 100' > out/transfer"),
                say(),
                run("echo '1 250' > out/transfer"),
                say(),
            ),
        ),
        (
            "simultaneous",
            experiment.simultaneous_round,
            lambda: per_agent(
                g01=(run("echo '2 100' > out/transfer"), say()),
                g02=(run("echo '1 250' > out/transfer"), say()),
            ),
        ),
    )
    ended = {}
    for schedule, a_round, router in rounds:
        for lost in ("g01", "g02"):
            with temp_root() as root:
                ids = seated(root, "g01", g02={})
                real = harness.replace_file

                def no_room(src, dest):
                    if dest == harness.trace_path(lost, 1):
                        raise OSError("no space left on device")
                    real(src, dest)

                harness.replace_file = no_room
                raised = None
                try:
                    with quiet():
                        a_round(ids, set(ids), 0, router())
                except OSError as e:
                    raised = e
                accounts = {agent: ground_truth(agent) for agent in ids}
            assert isinstance(raised, OSError), (schedule, lost, raised)
            for agent, account in accounts.items():
                spent = sum(episode["spent"] for episode in account["episodes"])
                assert (
                    reconciled(account, spent)
                    == account["remaining"]
                    == account["series"][-1]
                ), (
                    schedule,
                    lost,
                    account,
                )
            ended[schedule, lost] = {
                agent: (
                    [episode["received"] for episode in account["episodes"]],
                    account.get("received", 0),
                )
                for agent, account in accounts.items()
            }
    # What each committed episode received inside its span, and what each account holds
    # as received in all.
    assert ended == {
        ("sequential", "g01"): {"g01": ([], 0), "g02": ([], 0)},
        ("sequential", "g02"): {"g01": ([0], 0), "g02": ([], 100)},
        ("simultaneous", "g01"): {"g01": ([], 250), "g02": ([0], 0)},
        ("simultaneous", "g02"): {"g01": ([250], 0), "g02": ([], 100)},
    }, ended

    cost = turn_cost()
    stood = {}
    for floor in (True, False):
        for lost, transfer in ((True, "echo '1 50' > out/transfer"), (False, "true")):
            with temp_root(budget=cost - 1, floor_at_zero=floor) as root:
                ids = seated(root, "g01", g02={})
                real = harness.replace_file

                def no_room(src, dest, lost=lost):
                    if lost and dest == harness.trace_path("g02", 1):
                        raise OSError("no space left on device")
                    real(src, dest)

                harness.replace_file = no_room
                try:
                    with quiet():
                        experiment.simultaneous_round(
                            ids,
                            set(ids),
                            0,
                            per_agent(g01=(say(),), g02=(run(transfer), say())),
                        )
                except OSError:
                    pass
                g01 = ground_truth("g01")
            stood[floor, lost] = (
                g01["remaining"],
                g01.get("forgiven", 0),
                g01.get("received", 0),
                reconciled(g01, g01["episodes"][0]["spent"]),
                g01["series"][1:],
            )
    # Floored, a take-back forgives what its close would have; unfloored, it forgives
    # nothing, and the balance stays below zero where the close left it.
    assert stood[True, True][:4] == stood[True, False][:4] == (0, 1, 0, 0), stood
    assert stood[True, True][4] == [-1, 49, -1, 0] and stood[True, False][4] == [
        -1,
        0,
    ], stood
    assert stood[False, True][:4] == stood[False, False][:4] == (-1, 0, 0, -1), stood
    assert stood[False, True][4] == [-1, 49, -1] and stood[False, False][4] == [-1], (
        stood
    )


def check_an_episode_whose_close_raises_after_its_save_pays_and_is_paid_as_committed():
    """Whether an episode was committed is what its account on disk says, and not whether
    its close returned: a close can raise after the save as well as before it, printing
    its line. Each transfer is paid exactly once, whichever way it runs, and a receiver
    keeps no credit from a giver whose episode was lost.

    g01 gives seat 2 100 and g02 gives seat 1 250, and the console line of one of them
    raises once its account is saved. Sequentially that ends the round, its transfer
    paid. In a simultaneous round the other's trace may also fail to land, and where
    what a lost giver credited is taken back, the line saying so may raise as well.
    """
    rounds = {
        "sequential": (
            experiment.sequential_round,
            lambda: fake(
                run("echo '2 100' > out/transfer"),
                say(),
                run("echo '1 250' > out/transfer"),
                say(),
            ),
        ),
        "simultaneous": (
            experiment.simultaneous_round,
            lambda: per_agent(
                g01=(run("echo '2 100' > out/transfer"), say()),
                g02=(run("echo '1 250' > out/transfer"), say()),
            ),
        ),
    }

    class Broken(io.StringIO):
        def write(self, text):
            if "taken back" in text:
                raise BrokenPipeError("stderr is gone")
            return super().write(text)

    cases = (
        ("sequential", "g01", None, False),
        ("sequential", "g02", None, False),
        ("simultaneous", "g01", None, False),
        ("simultaneous", "g02", None, False),
        ("simultaneous", "g01", "g02", False),
        ("simultaneous", "g02", "g01", False),
        ("simultaneous", None, "g02", True),
    )
    ended = {}
    for schedule, unprinted, lost, broken in cases:
        a_round, router = rounds[schedule]
        with temp_root() as root:
            ids = seated(root, "g01", g02={})
            line, real = harness.console_line, harness.replace_file

            def raising(ep, trace, settled):
                if ep.agent == unprinted:
                    raise KeyboardInterrupt
                return line(ep, trace, settled)

            def no_room(src, dest):
                if lost and dest == harness.trace_path(lost, 1):
                    raise OSError("no space left on device")
                real(src, dest)

            harness.console_line = raising
            harness.replace_file = no_room
            raised = None
            try:
                with (
                    quiet(),
                    contextlib.redirect_stderr(Broken() if broken else sys.stderr),
                ):
                    a_round(ids, set(ids), 0, router())
            except (KeyboardInterrupt, OSError) as e:
                raised = e
            accounts = {agent: ground_truth(agent) for agent in ids}
        case = (schedule, unprinted, lost, broken)
        assert raised is not None, case
        for account in accounts.values():
            spent = sum(episode["spent"] for episode in account["episodes"])
            assert (
                reconciled(account, spent)
                == account["remaining"]
                == account["series"][-1]
            ), (case, account)
        ended[case] = {
            agent: (
                [episode["received"] for episode in account["episodes"]],
                account.get("received", 0),
            )
            for agent, account in accounts.items()
        }
    # What each committed episode received inside its span, and what each account holds
    # as received in all.
    assert ended == {
        ("sequential", "g01", None, False): {"g01": ([0], 0), "g02": ([], 100)},
        ("sequential", "g02", None, False): {"g01": ([0], 250), "g02": ([0], 100)},
        ("simultaneous", "g01", None, False): {
            "g01": ([250], 250),
            "g02": ([100], 100),
        },
        ("simultaneous", "g02", None, False): {
            "g01": ([250], 250),
            "g02": ([100], 100),
        },
        ("simultaneous", "g01", "g02", False): {"g01": ([250], 0), "g02": ([], 100)},
        ("simultaneous", "g02", "g01", False): {"g01": ([], 250), "g02": ([0], 0)},
        ("simultaneous", None, "g02", True): {"g01": ([250], 0), "g02": ([], 100)},
    }, ended


def check_an_interrupt_in_a_simultaneous_round_commits_every_episode_in_flight():
    """Ctrl+C in a simultaneous round ends every episode at its next turn, and all are committed."""

    def stopping_create():
        gate = threading.Barrier(3, timeout=10)

        def requested(turn):
            if turn == 1:
                gate.wait()
                harness.STOPPING = True

        return per_agent(
            default=(run("echo one"), run("echo two"), say()), on_request=requested
        )

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
        p = manifest_file(
            root,
            'schedule = "simultaneous"\nsystem_prompt = ""\n'
            + "".join(f'[[agent]]\nid = "{r}"\n' for r in ids),
        )
        harness.start = lambda config=None, **kw: stopping_create()
        with quiet() as buf:
            code = experiment.main(["--manifest", str(p), "--rounds", "5", "--resume"])
        took = episodes_taken(ids)
    assert code == 130, code
    assert took == {"g01": 1, "g02": 1, "g03": 1}, took
    assert "interrupted" in buf.getvalue() and "simultaneous" in buf.getvalue(), (
        buf.getvalue()
    )


def check_the_console_lines_of_a_simultaneous_round_are_whole_and_in_seat_order():
    """Every echoed line names its agent, and the summary lines come last, in seat order."""
    with temp_root(WATCH=True) as root:
        ids = seated(root, "g01", g02={}, g03={})
        live = set(ids)
        with quiet() as buf:
            experiment.simultaneous_round(
                ids,
                live,
                0,
                per_agent(default=(run("echo one"), run("echo two"), say())),
            )
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "nothing was echoed"
    assert all(any(ln.startswith(r) for r in ids) for ln in lines), (
        f"a line that does not say whose it is: {[ln for ln in lines if not any(ln.startswith(r) for r in ids)]}"
    )
    summary = [i for i, ln in enumerate(lines) if re.match(r"^g0\d\s+ep1\s", ln)]
    assert [lines[i].split()[0] for i in summary] == ids, "summaries in seat order"
    assert summary and summary[0] > max(
        i for i, ln in enumerate(lines) if "| " in ln
    ), "and after every echoed line"


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
        return (
            f'system_prompt = ""\n[[agent]]\nid = "g01"\n{a}[[agent]]\nid = "g02"\n{b}'
        )

    with temp_root() as root:
        for bad in (
            two('label = "no space"\n'),
            two('label = "same"\n', 'label = "same"\n'),
            two('label = "2"\n'),  # the other agent's default
            two("label = 3\n"),
            two('label = ""\n'),
            two('label = ".."\n'),
        ):
            refused(
                lambda: experiment.load_manifest(manifest_file(root, bad)),
                "label",
                because=f"accepted a bad label: {bad!r}",
            )
        m = experiment.load_manifest(
            manifest_file(root, two('label = "Studio"\n', 'label = "Game"\n'))
        )
        assert m["labels"] == PERSONA_LABELS, m["labels"]
        assert experiment.load_manifest(manifest_file(root, two()))["labels"] == {
            "1": "1",
            "2": "2",
        }
        assert experiment.shorthand(["a", "b"])["labels"] == {"1": "1", "2": "2"}


def check_a_manifest_table_replaces_the_whole_set():
    """A manifest's [[channel]] tables are the whole table; its [harness_files] overlay one key at a time."""
    agents = '[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n'
    with temp_root() as root:
        p = manifest_file(
            root,
            'schedule = "sequential"\nsystem_prompt = ""\n'
            '[[channel]]\nname = "notes"\nwriter = "self"\n'
            'readers = "self"\npath = "state"\n' + agents,
        )
        m = experiment.load_manifest(p)
        table, hf = harness.validate_channels(
            m["channels"], m["harness_files"], str(p), tuple(m["labels"].values())
        )
        assert [c.name for c in table] == ["notes"], (
            "one table declared, one channel in force"
        )
        assert hf == {"balance": "n", "digest": "m"}, "no [harness_files], no change"
        assert [c.name for c in harness.channels()] == [
            "notes",
            "blackboard",
            "mail",
            "transfer",
        ], "loading a manifest sets nothing; start() does"

        p = manifest_file(
            root,
            'system_prompt = ""\n[harness_files]\ndigest = ""\n' + agents,
            name="d.toml",
        )
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
    text = (
        'system_prompt = "'
        + ours
        + '"\n[[agent]]\nid = "g01"\nsystem_prompt = "'
        + mine
        + '"\n[[agent]]\nid = "g02"\n'
    )
    with temp_root() as root:
        # Declared and never inherited: a manifest silent on it is refused, and the
        # empty string is the declaration that says nothing.
        silent = manifest_file(root, '[[agent]]\nid = "g01"\n', name="s.toml")
        refused(
            lambda: experiment.load_manifest(silent),
            "system_prompt",
            because="a manifest declaring no prompt was accepted",
        )
        empty = manifest_file(
            root, 'system_prompt = ""\n[[agent]]\nid = "g01"\n', name="e.toml"
        )
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
        p.write_text(
            original.replace(mine, mine + " Again."), encoding="utf-8", newline="\n"
        )
        with quiet():
            refused(
                lambda: experiment.main(
                    ["--manifest", str(p), "--rounds", "1", "--resume"]
                ),
                "system_prompt",
                because="an agent was re-created on a different system prompt",
            )
        # The experiment's default is read at creation like every other setting, so a
        # seat that took it keeps what it was told and a later manifest does not resay it.
        p.write_text(
            original.replace(ours, ours + " Again."), encoding="utf-8", newline="\n"
        )
        with quiet():
            assert (
                experiment.main(["--manifest", str(p), "--rounds", "1", "--resume"])
                == 0
            )
        kept = ground_truth("g02")["system_prompt"]
    assert code == 0, buf.getvalue()
    assert sent == {mine, ours}, "each seat was told what it declared"
    assert harness.SYSTEM not in sent, "and nothing was told the shipped default"
    assert {r: a["system_prompt"] for r, a in accounts.items()} == {
        "g01": mine,
        "g02": ours,
    }
    assert kept == ours, (
        "the account is what the agent is told, not the manifest of the day"
    )
    for r, t in traces.items():
        want = accounts[r]["system_prompt"]
        assert t["provenance"]["system"] == want, (r, t["provenance"]["system"])
        assert (
            t["provenance"]["system_sha256"]
            == t["system_sha256"]
            == harness.system_sha256(want)
        )
