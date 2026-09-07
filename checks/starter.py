"""Starter files, and forks.

When the starter files land, what they are in the record, the terms pinned at
creation, and what a fork rebuilds and refuses."""

from __future__ import annotations

import json
import harness

from checks.fake import fake, run, say
from checks.lanes import (
    HostBox,
    episode_once,
    files_by_path,
    ground_truth,
    pinned,
    plant,
    quiet,
    refused,
    rooted,
    seated,
    temp_root,
    trace_on_disk,
    turn_cost,
)


def check_starter_files_land_when_the_balance_falls():
    """The starter files are absent above the threshold, and in the initial observation listing below it.

    The listing is the agent's whole environment at episode start, so material that is not in
    it is material the agent was not given.
    """
    # A say() episode spends 1750, so this is above episode 1's balance and below
    # episode 2's: the starter files land at the second episode and not the first.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        before = episode_once(say())
        after = episode_once(say())

    assert not [f for f in before["files"] if f.get("starter")], "nothing lands above the threshold"
    assert "m1" not in before["observation"], before["observation"]
    starter = sorted(f["path"] for f in after["files"] if f.get("starter"))
    assert starter == ["state/d/m2", "state/m1"], starter
    assert "m1" in after["observation"], "the starter files are in the listing the agent opens on"
    assert after["provenance"]["starter_files"] == "s"
    assert after["provenance"]["starter_files_sha256"], "the digest goes in provenance"
    assert after["provenance"]["starter_files_below"] == 499_000


def check_the_starter_files_threshold_is_a_balance_not_an_episode():
    """Two agents on the same threshold receive the starter files at different starts, at the same balance.

    An episode number does not mean the same thing twice, an episode's cost
    varying by orders of magnitude, so what the starter files land on is runway remaining.
    """
    landed = {}
    # The gap is two of the cheap agent's turns, so it crosses inside the four
    # episodes run whatever a turn costs under the model in force.
    below = 500_000 - 2 * turn_cost()
    for name, steps in [("cheap", (say(),)), ("dear", (run("echo hi"), say()))]:
        with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=below) as root:
            plant(root)
            with quiet():
                harness.run_episodes(name, fake(*steps), 4)
            account = ground_truth(name)
            landed[name] = account["starter_files_landed"]

    assert landed["cheap"]["episode"] > landed["dear"]["episode"], landed
    assert all(r["episode"] < 4 for r in landed.values()), \
        f"both must land inside the episodes run, with headroom to spare: {landed}"
    assert all(r["remaining"] <= below for r in landed.values()), landed
    assert all(r["sha256"] == landed["cheap"]["sha256"] for r in landed.values()), \
        "the same starter files, whenever they happened to land"


def check_starter_files_are_recorded_and_idempotent():
    """The account records what landed, and a later episode does not plant it again."""
    # At the budget itself the threshold is met at episode 1: material that was
    # always there, not material that appeared.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000) as root:
        plant(root)
        episode_once(run("echo mine > state/m1"), say())     # the agent overwrites it
        after = episode_once(say())
        record = ground_truth()["starter_files_landed"]

    assert record["name"] == "s" and record["episode"] == 1, record
    assert record["remaining"] == 500_000, "and the balance it landed on"
    assert sorted(record["paths"]) == ["d/m2", "m1"], record
    kept = files_by_path(after)["state/m1"]
    assert kept["text"].strip() == "mine", \
        "a second episode must not restore what the agent changed"
    assert kept["starter"], "and it is still one of the files the agent was given"


def check_starter_files_are_not_the_agents():
    """What the agent was given is `ours`; only what it invented is not."""
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000) as root:
        plant(root)
        t = episode_once(run("echo doctrine > state/NOTES.md"), say())

    by = files_by_path(t)
    assert by["state/m1"]["ours"] and by["state/m1"]["starter"], by["state/m1"]
    assert by["state/m1"]["text"].strip() == "alpha", \
        "a starter file's contents are captured, so an edit to it is legible"
    assert not by["state/NOTES.md"]["ours"], "the agent's own file stays the agent's"
    assert [f["path"] for f in t["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "everything the agent did not invent is out of what analyze.py counts as its own"
    assert not [f for f in t["files"] if f["channel"] != "notes"], \
        "an agent on its own has a blackboard and nothing on it"


def check_starter_files_refuse_to_overwrite_the_agents_work():
    """A starter-files path the agent already wrote stops the agent instead of clobbering it."""
    # Below episode 1's balance and above episode 2's, so the agent gets an episode to
    # make the file before the starter files arrive wanting the same name.
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        episode_once(run("echo mine > state/m1"), say())
        refused(lambda: episode_once(say()), "m1",
                because="the starter files overwrote a file the agent had made")
        assert (harness.mirror("t", "notes") / "m1").read_text().strip() == "mine", "and left it alone"


def check_an_agent_keeps_the_starter_files_it_was_created_with():
    """The starter-files terms are pinned in the account at creation, and the config cannot move them.

    Four things about one account: the terms are pinned, a config that moves on
    is ignored, other terms are refused, and an account from before the terms
    were pinned adopts the config at its next episode.
    """
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        plant(root, "other", m9="nine\n")
        first = episode_once(say())
        account = ground_truth()
        assert (account["starter_files"], account["starter_files_below"]) == ("s", 499_000)
        assert first["provenance"]["starter_files"] == "s"

        # The config moves on; the agent does not.
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "other", 500_000
        second = episode_once(say())
        assert second["provenance"]["starter_files"] == "s" and second["provenance"]["starter_files_below"] == 499_000
        assert ground_truth()["starter_files_landed"]["name"] == "s", \
            "the starter files that landed are the pinned ones"
        assert not (harness.mirror("t", "notes") / "m9").exists()
        assert second["provenance_drift"] == [], "nothing about this agent changed"

        # Asking the agent to be something it was not created as is refused.
        refused(lambda: harness.load_account("t", starter_files="other"), "starter_files",
                because="an agent was re-created on different terms")
        assert harness.load_account("t", starter_files="s", starter_files_below=499_000)["starter_files"] == "s", \
            "the terms it was created on are accepted"

        # An account from before the terms were pinned adopts the config at its next episode.
        m = ground_truth()
        del m["starter_files"], m["starter_files_below"]
        harness.save_account("t", m)
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "s", 499_000
        episode_once(say())
        assert ground_truth()["starter_files"] == "s"


def check_a_fork_carries_the_starter_files_terms_it_had():
    """A fork of an episode the starter files had landed on carries the terms.

    A fork from before they landed carries none, and takes the config it is next
    run under.
    """
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=499_000) as root:
        plant(root)
        plant(root, "other", m9="nine\n")
        episode_once(say())
        episode_once(say())
        assert ground_truth()["starter_files_landed"]["episode"] == 2
        with quiet():
            assert harness.fork("t", 2, "after") == 0
            assert harness.fork("t", 1, "before") == 0
        assert ground_truth("after")["starter_files"] == "s"
        assert ground_truth("after")["starter_files_landed"]["name"] == "s"
        assert "starter_files" not in ground_truth("before")
        assert "starter_files_landed" not in ground_truth("before")
        harness.STARTER_FILES, harness.STARTER_FILES_BELOW = "other", 500_000
        with quiet():
            harness.run_once("before", fake(say()))
        assert ground_truth("before")["starter_files"] == "other", "adopted at its first episode"
        assert (harness.mirror("before", "notes") / "m9").exists(), "and given at it"


def check_starter_files_config_is_validated():
    """starter_files and starter_files_below are set together, and starter_files must name a real directory."""
    with rooted(HostBox) as root:
        plant(root, "ok")
        f = root / "config.toml"
        for bad in ('starter_files = "ok"', 'starter_files_below = 3', 'starter_files = "ok"\nstarter_files_below = 0',
                    'starter_files = "nope"\nstarter_files_below = 2', 'starter_files = "ok"\nstarter_files_below = -1',
                    'starter_files = 5\nstarter_files_below = 2'):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                refused(lambda: harness.load_config(f), "starter_files",
                        because=f"accepted bad starter_files config: {bad!r}")

        f.write_text('starter_files = "ok"\nstarter_files_below = 400000\n', encoding="utf-8")
        with pinned():
            harness.load_config(f)
            assert (harness.STARTER_FILES, harness.STARTER_FILES_BELOW) == ("ok", 400_000)
        # The digest covers paths as well as bytes, so a rename is a different set of starter files.
        was = harness.files_sha256("ok")
        (root / "files" / "ok" / "m1").rename(root / "files" / "ok" / "m3")
        assert harness.files_sha256("ok") != was, "a renamed file is a different set of starter files"


def check_fork_reproduces_state_and_account():
    """A fork rebuilds the recorded episode exactly, and runs nothing."""
    with temp_root():
        episode_once(run("echo v1 > state/NOTES.md"), say())
        episode_once(run("echo v2 > state/NOTES.md"), say())
        parent = ground_truth()
        trace = trace_on_disk("t", 1)
        with quiet():
            assert harness.fork("t", 1, "f") == 0
        forked = ground_truth("f")
        notes = (harness.mirror("f", "notes") / "NOTES.md").read_bytes()

        was = files_by_path(trace)["state/NOTES.md"]
        assert notes.decode() == was["text"], "state/ is the episode it forked at"
        assert len(notes) == was["size"], "byte for byte, not merely line for line"
        assert notes == b"v1\n", "and not the episode the parent has since reached"
        assert forked["series"] == trace["series_after"], "invariant 8: the series is what carries"
        assert not any(harness.mirror("f", "blackboard").rglob("*")), \
            "the parent's blackboard was empty, so the fork's is"
        assert forked["remaining"] == trace["series_after"][-1]
        assert len(forked["episodes"]) == 1, "the episodes after the fork point are dropped"
        assert forked["initial"] == parent["initial"] and forked["model"] == parent["model"]
        assert forked["forked_from"] == {"agent": "t", "episode": 1, "modes": "defaulted"}
        assert not list((harness.records_dir("f") / "traces").glob("*.json")), "a fork bills nothing"

        # Forking the head restores the modes the parent's sidecar still holds.
        with quiet():
            assert harness.fork("t", 2, "g") == 0
        assert ground_truth("g")["forked_from"]["modes"] == "restored"
        with quiet():
            assert harness.fork("t", 2, "g") != 0, "an existing agent is not overwritten"


def check_a_fork_rebuilds_every_tree_the_agent_wrote():
    """All three writable channels come back, and neither peers nor inboxes do.

    A peer's blackboard and another agent's message are rebuilt from their owners at
    the next episode, so copying them would deliver the same post twice.
    """
    with temp_root() as root:
        seated(root, other={"group/theirs": "not mine\n", "out/1": "for you\n"})
        episode_once(run("echo mine > state/NOTES.md",
                      "echo posted > 1/RESULT",
                      "echo psst > out/2 && echo '2 50' > out/transfer"), say())
        with quiet():
            assert harness.fork("t", 1, "f") == 0

        assert (harness.mirror("f", "notes") / "NOTES.md").read_text(encoding="utf-8") == "mine\n"
        assert (harness.mirror("f", "blackboard") / "RESULT").read_text(encoding="utf-8") == "posted\n"
        assert (harness.mirror("f", "mail") / "2").read_text(encoding="utf-8") == "psst\n", \
            "a message is one file, and the fork rebuilds it as one"
        assert (harness.mirror("f", "mail") / "transfer").read_text(encoding="utf-8") == "2 50\n", \
            "a standing declaration is part of the episode being rebuilt"
        # And nothing that belonged to the neighbour: its blackboard, and the message
        # it addressed to this agent, are both rebuilt from it at the next episode.
        rebuilt = {p.name for tree in (harness.mirror("f", "notes"), harness.mirror("f", "blackboard"),
                                       harness.mirror("f", "mail")) for p in tree.rglob("*")}
        assert rebuilt == {"NOTES.md", "RESULT", "2", "transfer"}, sorted(rebuilt)
        assert "theirs" not in rebuilt, sorted(rebuilt)


def check_fork_refuses_what_it_cannot_rebuild():
    """Anything the trace did not store exactly stops the fork."""
    with rooted(HostBox):
        priv = harness.records_dir("p") / "traces"
        priv.mkdir(parents=True)
        harness.save_account("p", {"agent": "p", "model": "claude-opus-5", "initial": 10,
                                   "created_at": "now", "remaining": 9, "series": [10, 9],
                                   "episodes": [{"episode": 1, "stop": "end_turn",
                                                 "spent": 1, "turns": 1}]})

        def trace(**over):
            # A peer's blackboard is skipped and not rebuilt, so an entry that
            # could never be stored does not stop a fork if it is one.
            t = {"trace_version": harness.TRACE_VERSION, "episode": 1, "state_saved": True,
                 "series_after": [10, 9],
                 "files": [{"path": "2/theirs", "channel": "blackboard", "writer": "self",
                            "readers": "all", "role": "peer", "size": 5,
                            "author": "peer:2", "ours": True, "starter": False,
                            "text": None},
                           {"path": "state/a.txt", "channel": "notes", "writer": "self",
                            "readers": "self", "role": "own", "size": 3,
                            "author": "self", "ours": False, "starter": False,
                            "text": "hi\n"}]}
            return {**t, **over}

        binary = trace()
        binary["files"][1]["text"] = None
        big = trace()
        big["files"][1]["size"] = harness.FILE_CONTENT_LIMIT + 1
        lossy = trace()
        lossy["files"][1]["text"] = "h�\n"
        for name, bad in [("binary", binary), ("truncated", big), ("lossy", lossy),
                          ("unmirrored", trace(state_saved=False))]:
            harness.trace_path("p", 1).write_text(json.dumps(bad), encoding="utf-8")
            with quiet():
                assert harness.fork("p", 1, f"x-{name}") != 0, f"forked a {name} episode"
            assert not (harness.records_dir(f"x-{name}") / "account.json").exists(), \
                f"a refused fork left a {name} agent behind"

        harness.trace_path("p", 1).write_text(json.dumps(trace()), encoding="utf-8")
        with quiet():
            assert harness.fork("p", 9, "x-missing") != 0, "forked an episode that never ran"
            assert harness.fork("nosuch", 1, "x-none") != 0, "forked an agent that is not there"
            assert harness.fork("p", 1, "good") == 0, "and a storable episode still forks"
