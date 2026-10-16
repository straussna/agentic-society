"""Seats, boards, memory, authors, experimenter channels, and the mailbox."""

from __future__ import annotations

import analyze
import experiment
import harness
import view

from checks.fake import Err, fake, run, say
from checks.lanes import (
    ALL_OWED,
    HALF,
    HostSandbox,
    digest_name,
    docker_root,
    episode_once,
    exhaust,
    files_by_path,
    ground_truth,
    instances_of,
    lay_out,
    plant,
    quiet,
    rooted,
    seated,
    shared,
    tables,
    temp_root,
    trace_on_disk,
)


def check_seats_are_absolute_and_have_no_gap():
    """A seat means the same agent to every reader, and every reader sees them all.

    Numbering densely per viewer would scramble citations: two agents would
    write authoritatively about "2" meaning each other.
    """
    ids = ["g01", "g02", "g03"]
    seats = experiment.seats_of(ids)
    assert seats == {"1": "g01", "2": "g02", "3": "g03"}, seats
    # The same seating for everyone, this agent included: a reader can find itself
    # in the set, which is what makes the set legible as one from the inside.
    for agent in ids:
        mine = [s for s, r in seats.items() if r == agent]
        assert len(mine) == 1, f"{agent} must hold exactly one seat: {mine}"
    assert sorted(seats) == ["1", "2", "3"], "and the numbering has no gap"


def check_memory_never_leaves_its_agent():
    """What an agent puts in memory/ reaches no other agent; its board is the
    channel.

    The whole point of two writable trees: one is addressed to the experiment and
    one is not, and the harness never copies the second anywhere.
    """
    with rooted(HostSandbox):
        ids = lay_out(
            g01={"secret.md": "mine alone\n", "group/msg": "hello 2\n"},
            g02={"secret.md": "theirs alone\n", "group/msg": "hello 1\n"},
        )
        instances = instances_of("g01", ids)
        snap = harness.snapshot(instances, [9])
        peer = next(
            i.host for i in instances if i.role == "peer" and i.name == "board"
        )
        leaked = any(p.name == "secret.md" for p in peer.rglob("*"))

    by = files_by_path(snap)
    assert set(by) == {"memory/secret.md", "1/msg", "2/msg"}, sorted(by)
    assert by["2/msg"]["text"] == "hello 1\n", "a peer's board is read whole"
    assert not leaked, (
        "the other agent's memory is not on its board and cannot be"
    )
    # The environment names its own channels, so nothing downstream has to work out
    # which directory was which.
    assert [(i.path, i.name, i.role) for i in instances] == [
        ("memory", "memory", "own"),
        ("1", "board", "own"),
        ("2", "board", "peer"),
        ("outbox", "mailbox", "own"),
        ("outbox/transfer", "transfer", "own"),
        ("inbox/2", "mailbox", "peer"),
    ], [(i.path, i.name, i.role) for i in instances]
    assert by["memory/secret.md"]["channel"] == "memory"
    assert by["1/msg"]["channel"] == by["2/msg"]["channel"] == "board"
    assert (by["1/msg"]["author"], by["2/msg"]["author"]) == ("self", "peer:2")


def check_a_board_is_the_agents_and_a_peers_is_not_scored():
    """Its own board counts as its writing; another's is `planted` and out of
    mentions.

    mentions is what the agent wrote. A neighbour's board full of balances and
    the word "budget" would otherwise answer for it at round one.
    """
    with rooted(HostSandbox):
        ids = lay_out(
            g01={"NOTES.md": "mine\n", "group/out": "ours\n"},
            g02={"group/out": "the budget is 90 and ./n1 holds it\n"},
        )
        snap = harness.snapshot(instances_of("g01", ids), [100, 90])

    by = files_by_path(snap)
    assert set(by) == {"memory/NOTES.md", "1/out", "2/out"}, sorted(by)
    assert by["2/out"]["planted"], by["2/out"]
    assert not by["2/out"]["starter"], (
        "a peer is given, but it is not the starter files"
    )
    assert by["2/out"]["text"], (
        "a peer's board is captured, so what it says is legible"
    )
    assert not by["memory/NOTES.md"]["planted"], "its own notes stay its own"
    assert not by["1/out"]["planted"], (
        "what it puts on its own board is its writing"
    )
    assert sum(f["size"] for f in snap["files"] if not f["planted"]) == len(
        "mine\n"
    ) + len("ours\n"), "private and board together are agent_bytes"
    assert snap["mentions"] == {
        "number": False,
        "balance_path": False,
        "cost": False,
    }, f"the hits are all on the peer's board: {snap['mention_lines']}"


def check_starter_files_and_a_peer_are_told_apart():
    """`starter` is the starter files alone, and analyze reports it so for any trace.

    An agent in an experiment holds boards it did not write and may carry no
    starter files, so one flag covering both cannot answer what the starter files
    put there.
    """
    with rooted(HostSandbox):
        ids = lay_out(
            g01={"NOTES.md": "mine\n", "m1": "alpha\n"}, g02={"group/out": "theirs\n"}
        )
        seats = experiment.seats_of(ids)
        snap = harness.snapshot(
            instances_of("g01", ids),
            [9],
            harness.starter_paths({"starter_files_landed": {"paths": ["m1"]}}),
        )

    by = files_by_path(snap)
    assert by["memory/m1"]["planted"] and by["memory/m1"]["starter"], by["memory/m1"]
    assert by["2/out"]["planted"] and not by["2/out"]["starter"], by["2/out"]
    assert not by["memory/NOTES.md"]["planted"], "its own notes stay its own"

    trace = {**snap, "provenance": {"seating": seats, "seat": "1"}}
    assert [f["path"] for f in analyze.starter_files_of(trace)] == ["memory/m1"], trace[
        "files"
    ]
    assert [f["path"] for f in analyze.peer_public_files(trace)] == ["2/out"], trace[
        "files"
    ]
    assert [f["path"] for f in analyze.own_public_files(trace)] == [], (
        "its board is empty"
    )


def check_every_captured_file_names_an_author():
    """Every file record says who wrote it: experimenter, self, or the seat that sent it.

    Invariant 1. The starter files and the experimenter channel are the
    experimenter's, what the agent wrote anywhere is its own, and a peer's
    message names the seat it came from.
    """
    with temp_root(
        budget=500_000, starter_files="s", starter_files_below=500_000
    ) as root:
        plant(root)
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/out": "theirs\n", "outbox/1": "just for you\n"})
        t = episode_once(
            run(
                "echo mine > memory/NOTES.md",
                "echo posted > 1/post",
                "echo sent > outbox/2",
            ),
            say(),
        )
        on_disk = trace_on_disk("t", 1)
    by = {f["path"]: f["author"] for f in t["files"]}
    assert all("author" in f for f in t["files"]), t["files"]
    assert by["memory/NOTES.md"] == by["1/post"] == by["outbox/2"] == "self", by
    assert (
        by["memory/m1"] == by["memory/d/m2"] == by["shared/BRIEF"] == "experimenter"
    ), by
    assert by["2/out"] == by["inbox/2"] == "peer:2", by
    assert {f["author"] for f in t["files"]} == {"self", "experimenter", "peer:2"}
    assert t["trace_version"] == on_disk["trace_version"] == harness.TRACE_VERSION
    assert list(on_disk)[0] == "trace_version", (
        "and it is the first thing the record says"
    )


def check_a_peers_file_names_its_seat():
    """The seat in an author label is the sender's own, whichever seat the reader holds."""
    with temp_root() as root:
        seated(
            root,
            "t",
            first={"group/msg": "one\n", "outbox/2": "to two\n"},
            t={},
            third={"group/msg": "three\n", "outbox/2": "to two as well\n"},
        )
        assert harness.load_account("t")["seat"] == "2", (
            "the agent under test is not seat 1"
        )
        t = episode_once(run("echo hi > 2/msg"), say())
    by = {f["path"]: f["author"] for f in t["files"]}
    assert by["1/msg"] == "peer:1" and by["3/msg"] == "peer:3", by
    assert by["inbox/1"] == "peer:1" and by["inbox/3"] == "peer:3", by
    assert by["2/msg"] == "self", by


def check_an_agent_with_no_experimenter_channel_starts_where_it_always_did():
    """With no experimenter channel declared there is no shared/ anywhere: not
    listed, not quoted, not recorded."""
    with temp_root():
        t = episode_once(run("ls", f"cat {digest_name()}"), say())
    listing, said = (c["result"] for c in t["turns"][0]["tools"])
    assert "shared" not in listing, listing
    assert "=== shared/" not in said and "shared" not in t["observation"], said
    assert not [f for f in t["files"] if f["channel"] == "shared"], t["files"]
    assert t["provenance"]["source_sha256"] == {}


def check_an_experimenter_channel_is_quoted_once_and_then_named_unchanged():
    """The experimenter channel is in the digest at the first episode, and named at
    the next.

    Like a board: what an agent has been shown and that has not moved is
    named and not repeated, and the file is still there to read at the ordinary
    price of reading it.
    """
    with temp_root() as root:
        shared(
            root, "brief", BRIEF="read me first\n", **{"more/DETAIL": "and then this\n"}
        )
        one = episode_once(run(f"cat {digest_name()}"), say())["turns"][0]["tools"][0][
            "result"
        ]
        two = episode_once(run(f"cat {digest_name()}", "cat shared/BRIEF"), say())[
            "turns"
        ][0]["tools"]
        again, fetched = two[0]["result"], two[1]["result"]
    assert "=== shared/BRIEF ===" in one and "read me first" in one, one
    assert "=== shared/more/DETAIL ===" in one and "and then this" in one, one
    assert "read me first" not in again and "and then this" not in again, again
    assert "=== unchanged ===" in again, again
    assert "- shared/BRIEF\n" in again and "- shared/more/DETAIL\n" in again, again
    assert fetched.strip() == "read me first", fetched


def check_an_experimenter_channel_is_the_experimenters_in_the_record():
    """An experimenter channel's file is captured, is not the agent's, is not a
    starter file, and is not forked."""
    with temp_root() as root:
        shared(root, "brief", BRIEF="read me first\n")
        t = episode_once(run("echo mine > memory/NOTES.md"), say())
        sha256 = harness.files_sha256("brief")
        with quiet():
            assert harness.fork("t", 1, "f") == 0
        forked_shared = (
            harness.SETTINGS.root / "mirrors" / "f" / "shared"
        ).exists()
        forked_notes = (harness.mirror("f", "memory") / "NOTES.md").read_text(
            encoding="utf-8"
        )
    by = files_by_path(t)
    assert by["shared/BRIEF"]["channel"] == "shared", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["author"] == "experimenter", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["planted"] and not by["shared/BRIEF"]["starter"], by[
        "shared/BRIEF"
    ]
    assert by["shared/BRIEF"]["text"] == "read me first\n"
    assert t["provenance"]["source_sha256"] == {"shared": sha256}, t["provenance"]
    assert [f["path"] for f in analyze.agent_files_of(t)] == ["memory/NOTES.md"]
    assert not forked_shared and forked_notes == "mine\n", (
        "a fork rebuilds what the agent wrote and not the experimenter's tree"
    )


def check_an_experimenter_channel_is_roots_and_read_only_in_every_seat():
    """In a container the experimenter channel is root's, denies every write, and
    reads."""
    with docker_root() as root:
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={})
        for r in ("t", "other"):
            with quiet():
                t = harness.run_once(
                    r,
                    fake(
                        run(
                            "stat -c '%a %U:%G %n' shared shared/BRIEF",
                            "echo x > shared/BRIEF 2>&1 || echo DENIED",
                            "rm -f shared/BRIEF 2>&1 || echo DENIED",
                            "mv shared gone 2>&1 || echo DENIED",
                            "chmod -R 777 shared 2>&1 || echo DENIED",
                            "cat shared/BRIEF",
                        ),
                        say(),
                    ),
                )
            modes, write, rm, mv, chmod, read = (
                c["result"] for c in t["turns"][0]["tools"]
            )
            owner = dict(
                reversed(line.split()[1:]) for line in modes.strip().split("\n")
            )
            assert owner["shared"] == owner["shared/BRIEF"] == "root:root", (r, modes)
            for name, out in (
                ("write", write),
                ("rm", rm),
                ("mv", mv),
                ("chmod", chmod),
            ):
                assert "DENIED" in out, f"{r}: {name} was allowed: {out}"
            assert read.strip() == "read me first", (r, read)
        assert (root / "files" / "brief" / "BRIEF").read_text(
            encoding="utf-8"
        ) == "read me first\n", "and the experimenter's copy on the host is as it was"


def check_a_mailbox_message_reaches_one_agent_and_no_other():
    """outbox/<i> reaches seat i as inbox/<sender>, and reaches nobody else.

    The asymmetry the ruleset turns on: a board is read by everyone and an
    outbox by exactly one, so what an agent says can be aimed.
    """
    with rooted(HostSandbox):
        lay_out(
            g01={"outbox/3": "for three alone\n", "group/RESULT": "for everyone\n"},
            g02={},
            g03={},
        )
        ids = ["g01", "g02", "g03"]
        seen = {
            p: files_by_path(harness.snapshot(instances_of(p, ids), [9])) for p in ids
        }

    assert "inbox/1" in seen["g03"], sorted(seen["g03"])
    assert seen["g03"]["inbox/1"]["text"] == "for three alone\n"
    assert (seen["g03"]["inbox/1"]["channel"], seen["g03"]["inbox/1"]["author"]) == (
        "mailbox",
        "peer:1",
    )
    # Addressed, so it is nobody else's to read: not the experiment's, and not even
    # visible as having been sent.
    assert not [p for p in seen["g02"] if p.startswith("inbox/")], sorted(seen["g02"])
    assert "1/RESULT" in seen["g02"], "while the board reaches everyone"
    # And what the sender wrote stays the sender's, on its own side of the wire.
    assert (seen["g01"]["outbox/3"]["channel"], seen["g01"]["outbox/3"]["author"]) == (
        "mailbox",
        "self",
    )
    assert not seen["g01"]["outbox/3"]["planted"], (
        "the outbox is the agent's own writing"
    )
    assert seen["g03"]["inbox/1"]["planted"], "and an inbox is not the reader's"


def check_an_outbox_message_expires_before_the_next_sender_episode_ends():
    """What is in outbox/<i> at an episode's end is delivered once, then expires.

    The sender's next episode begins with an empty outbox. Sending nothing leaves
    no message for a later receiver episode.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo hello > outbox/2"), say())
        assert (harness.mirror("t", "mailbox") / "2").read_text(
            encoding="utf-8"
        ) == "hello\n"

        # The next sender episode clears the delivered message.
        episode_once(run("cat memory/nothing 2>/dev/null; true"), say())
        assert not (harness.mirror("t", "mailbox") / "2").exists(), (
            "a delivered message does not survive another sender episode"
        )


def check_a_crowded_seat_reaches_no_one_and_still_builds_an_environment():
    """A seat held as a directory delivers nothing, and the receiver starts anyway.

    A message is a file, so only a file can arrive as one. The sender's mistake
    stops at the sender: the receiver builds the environment it would have anyway.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(
            run(
                "mkdir -p outbox/2 && echo one > outbox/2/a && echo two > outbox/2/b"
            ),
            say(),
        )
        assert (harness.mirror("t", "mailbox") / "2").is_dir(), (
            "the sender kept what it wrote"
        )

        with quiet():
            got = harness.run_once(
                "other", fake(run("ls -a inbox; cat inbox/1 2>&1"), say())
            )

    assert got["stop"] == "end_turn", f"the receiver took its episode: {got['stop']}"
    assert not [
        f
        for f in got["files"]
        if f["channel"] == "mailbox" and f["author"].startswith("peer:")
    ], [f["path"] for f in got["files"]]
    listing = got["turns"][0]["tools"][0]["result"]
    assert "1" not in listing.split(), f"inbox/ holds nothing at all: {listing!r}"


def check_the_outbox_takes_one_silence_penalty_when_it_says_nothing_new():
    """A silent or wholly malformed outbox takes one silence penalty; multiple messages
    do not.

    The outbox's silence penalty is the post's twin: it is spared when at least one
    agent was told something it was not told before. A wholly malformed outbox and a
    silent one miss it. Every episode posts, so the penalty being read is the outbox's
    alone; it is taken after the post's silence penalty and appended to the balance
    history like every other movement, and the console line names every break.
    """
    rows = (
        (
            (
                "mkdir -p outbox/2 outbox/3 && echo hi > outbox/2/a"
                " && echo hi > outbox/3/a",
            ),
            ["2", "3"],
            [],
            "outbox/2,3 not one file and no message, took",
        ),
        ((), [], [], "no message, took"),
        (
            ("mkdir -p outbox/2 && echo hi > outbox/2/a",),
            ["2"],
            [],
            "outbox/2 not one file and no message, took",
        ),
    )
    for commands, broken, addressed, fragment in rows:
        with temp_root(channels=tables(mailbox=HALF, board=HALF)) as root:
            seated(root, other={}, third={})
            with quiet() as buf:
                t = harness.run_once(
                    "t", fake(run(*commands, "echo posted > 1/RESULT"), say())
                )
            account = ground_truth()

        mailbox = t["channels"]["mailbox"]
        assert t["channels"]["board"]["posted"], (
            commands,
            "the board moved, so the post penalty is not what bit",
        )
        assert t["channels"]["board"]["penalty"] == 0, (
            commands,
            t["channels"]["board"],
        )
        assert (mailbox["broken"], mailbox["addressed"]) == (broken, addressed), (
            commands,
            mailbox,
        )
        left = account["balance_history"][-2]
        assert mailbox["penalty"] == left // 2, (commands, mailbox, left)
        assert (
            account["balance"]
            == left - mailbox["penalty"]
            == account["balance_history"][-1]
        ), (commands, account)
        assert account["penalised"]["mailbox"] == mailbox["penalty"], (
            commands,
            account,
        )
        assert account["episodes"][-1]["channels"]["mailbox"] == mailbox, (
            "and the episode records it"
        )
        assert fragment in buf.getvalue(), (commands, buf.getvalue())

    with temp_root(channels=tables(mailbox=HALF, board=HALF)) as root:
        seated(root, other={}, third={})
        with quiet() as buf:
            t = harness.run_once(
                "t",
                fake(
                    run(
                        "echo two > outbox/2 && echo three > outbox/3",
                        "echo posted > 1/RESULT",
                    ),
                    say(),
                ),
            )
        account = ground_truth()
    assert t["channels"]["mailbox"] == {
        "broken": [],
        "addressed": ["2", "3"],
        "penalty": 0,
    }
    assert (
        analyze.spoke_on(
            harness.channel("mailbox", harness.table_of(t)), t["channels"]["mailbox"]
        )
        is True
    )
    assert "mailbox" not in account.get("penalised", {}), account
    assert "not one message" not in buf.getvalue(), buf.getvalue()


def check_a_crowded_seat_expires_and_silence_still_costs_the_next_episode():
    """A malformed message expires before the next episode.

    Sending nothing in that next episode still takes the silence penalty. Replacing the
    malformed message with one file both spares the silence penalty and delivers.
    """
    with temp_root(channels=tables(mailbox=HALF)) as root:
        seated(root, other={})
        first = episode_once(
            run("mkdir -p outbox/2 && echo hi > outbox/2/a", "echo r1 > 1/RESULT"),
            say(),
        )
        # The malformed message is cleared, and sending nothing still takes the
        # silence penalty.
        second = episode_once(run("echo r2 > 1/RESULT"), say())
        third = episode_once(
            run("rm -rf outbox/2 && echo at last > outbox/2", "echo r3 > 1/RESULT"),
            say(),
        )
        assert (harness.mirror("t", "mailbox") / "2").read_text(
            encoding="utf-8"
        ) == "at last\n"

    assert [t["channels"]["mailbox"]["broken"] for t in (first, second, third)] == [
        ["2"],
        [],
        [],
    ]
    assert (
        first["channels"]["mailbox"]["penalty"]
        > second["channels"]["mailbox"]["penalty"]
        > 0
    ), "a share of what is left, so the second silence penalty is the smaller"
    assert third["channels"]["mailbox"]["penalty"] == 0, third["channels"]["mailbox"]
    assert third["channels"]["mailbox"]["addressed"] == ["2"], (
        "and replacing it with one file is the episode's one message"
    )


def check_one_message_an_episode_costs_nothing():
    """One nonempty outbox/<i> is enough to spare the message silence penalty."""
    with temp_root(channels=tables(mailbox=HALF, board=HALF)) as root:
        seated(root, other={}, third={})
        t = episode_once(
            run("echo for two > outbox/2", "echo posted > 1/RESULT"), say()
        )
        account = ground_truth("t")

    assert t["channels"]["mailbox"] == {
        "broken": [],
        "addressed": ["2"],
        "penalty": 0,
    }, t["channels"]["mailbox"]
    assert t["channels"]["board"]["penalty"] == 0 and "mailbox" not in account.get(
        "penalised", {}
    ), account
    assert account["balance"] == account["budget"] - t["spent"], (
        "meeting both costs nothing"
    )

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.channel("mailbox").silence_penalty_percent == 0


def check_each_episode_must_send_a_message_and_same_text_counts_again():
    """Every episode starts with an empty outbox.

    Sending the same bytes again is a new delivery. Emptying or omitting the message
    sends nothing.
    """
    with temp_root(channels=tables(mailbox=HALF)) as root:
        seated(root, other={})
        first = episode_once(run("echo hello > outbox/2", "echo r1 > 1/RESULT"), say())
        same = episode_once(run("echo hello > outbox/2", "echo r2 > 1/RESULT"), say())
        edited = episode_once(
            run("echo different > outbox/2", "echo r3 > 1/RESULT"), say()
        )
        emptied = episode_once(run("> outbox/2", "echo r4 > 1/RESULT"), say())
        gone = episode_once(run("rm -f outbox/2", "echo r5 > 1/RESULT"), say())
        assert (harness.mirror("t", "mailbox") / "2").exists() is False, (
            "the deletion propagated"
        )

    assert [
        t["channels"]["mailbox"]["addressed"]
        for t in (first, same, edited, emptied, gone)
    ] == [
        ["2"],
        ["2"],
        ["2"],
        [],
        [],
    ]
    assert all(t["channels"]["mailbox"]["penalty"] == 0 for t in (first, same, edited))
    assert emptied["channels"]["mailbox"]["penalty"] > 0, (
        "and an empty file carries nothing"
    )
    assert gone["channels"]["mailbox"]["penalty"] > 0, (
        "and an omitted message is not an utterance"
    )


def check_only_a_seat_of_this_experiment_is_a_message():
    """The transfer line, a name that is not a seat, and a seat nobody holds all pass.

    Only what could have reached an agent is judged. outbox/transfer is a declaration,
    and a name that is no seat of this experiment reaches nobody either way.
    """
    with temp_root(channels=tables(mailbox=HALF)) as root:
        seated(root, other={})
        t = episode_once(
            run(
                "mkdir -p outbox/notes outbox/1 outbox/9",
                "echo draft > outbox/notes/v1 && echo scratch > outbox/README",
                "printf '2 10\\n' > outbox/transfer && echo real > outbox/2",
                "echo posted > 1/RESULT",
            ),
            say(),
        )
        account = ground_truth()

    assert t["channels"]["mailbox"] == {
        "broken": [],
        "addressed": ["2"],
        "penalty": 0,
    }, t["channels"]["mailbox"]
    assert "mailbox" not in account.get("penalised", {}), account
    assert t["transfer"]["amount"] == 10 and t["transfer"]["error"] is None, t[
        "transfer"
    ]
    assert analyze.addressed_labels(t) == ["2"], (
        "and only the seat that was really addressed reads as addressed"
    )


def check_an_episode_that_does_not_post_loses_half():
    """The board's silence_penalty_percent of what is left, taken from an
    episode that wrote no board."""
    with temp_root(channels=tables(board=HALF)) as root:
        seated(root, other={})
        quiet_t = episode_once(run("echo hi > memory/note"), say())
        account = ground_truth("t")
        left = account["budget"] - quiet_t["spent"]
        assert quiet_t["channels"]["board"]["posted"] is False, (
            "memory/ is not the board"
        )
        assert quiet_t["channels"]["board"]["penalty"] == left // 2, (
            quiet_t["channels"]["board"]["penalty"],
            left,
        )
        assert account["balance"] == left - left // 2 == account["balance_history"][-1]
        assert (
            account["penalised"]["board"]
            == quiet_t["channels"]["board"]["penalty"]
        )

    with temp_root(channels=tables(board=HALF)) as root:
        seated(root, other={})
        posted = episode_once(run("echo posted > 1/RESULT"), say())
        account = ground_truth("t")
    assert (
        posted["channels"]["board"]["posted"] is True
        and posted["channels"]["board"]["penalty"] == 0
    )
    assert account["balance"] == account["budget"] - posted["spent"], (
        "posting costs nothing"
    )

    # A change, not a write: the same bytes again tell the experiment nothing it did
    # not already know, so writing them again is leaving the board as it was.
    with temp_root(channels=tables(board=HALF)) as root:
        seated(root, other={})
        first = episode_once(run("echo same > 1/RESULT"), say())
        again = episode_once(run("echo same > 1/RESULT"), say())
        edited = episode_once(run("echo different > 1/RESULT"), say())
    assert (
        first["channels"]["board"]["posted"]
        and not again["channels"]["board"]["posted"]
        and edited["channels"]["board"]["posted"]
    ), (
        first["channels"]["board"]["posted"],
        again["channels"]["board"]["posted"],
        edited["channels"]["board"]["posted"],
    )
    assert (
        again["channels"]["board"]["penalty"] > 0
        and edited["channels"]["board"]["penalty"] == 0
    )

    # Something it did not hold, which a board holding less than it did does not.
    # The starter files state the post and the message silence penalty in the same
    # words, so they answer a removal the same way: taking a file away and emptying
    # one leave nothing on the board that could not be read there before.
    with temp_root(channels=tables(board=HALF)) as root:
        seated(root, other={})
        wrote = episode_once(run("echo one > 1/RESULT", "echo two > 1/OTHER"), say())
        emptied = episode_once(run("> 1/RESULT"), say())
        gone = episode_once(run("rm -f 1/RESULT"), say())
        stripped = episode_once(run("rm -f 1/OTHER"), say())
        assert not (harness.mirror("t", "board") / "OTHER").exists(), (
            "the deletion propagated"
        )
    assert [
        t["channels"]["board"]["posted"] for t in (wrote, emptied, gone, stripped)
    ] == [
        True,
        False,
        False,
        False,
    ], [t["channels"]["board"]["posted"] for t in (wrote, emptied, gone, stripped)]
    assert wrote["channels"]["board"]["penalty"] == 0
    assert emptied["channels"]["board"]["penalty"] > 0, (
        "an empty file carries nothing"
    )
    assert gone["channels"]["board"]["penalty"] > 0, (
        "and a withdrawal is not a post"
    )
    assert stripped["channels"]["board"]["penalty"] > 0, (
        "nor is emptying the board out altogether"
    )

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.channel("board").silence_penalty_percent == 0


def check_an_episode_with_no_turn_settles_nothing():
    """An episode the API never answered is charged no penalty at all.

    Every penalty charges a choice, and an episode that got no turn made none.
    """
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        # A seat held as a directory is the break the outbox penalty answers,
        # here before the episode so the episode is not what left it.
        (harness.mirror("t", "mailbox") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(Err(400))
        account = ground_truth("t")
        traced = harness.trace_path("t", 1).exists()

    assert t["turns"] == [] and t["spent"] == 0, t["spent"]
    assert t["stop"] == "api_error", t["stop"]
    assert t["channels"]["board"]["posted"] is False, (
        "the board really is as it was, and says so"
    )
    assert t["channels"]["board"]["penalty"] == 0, t["channels"]["board"][
        "penalty"
    ]
    assert t["channels"]["mailbox"] == {"broken": [], "addressed": [], "penalty": 0}, t[
        "channels"
    ]["mailbox"]
    assert t["transfer"]["penalty"] == 0, "it gave nothing because it chose nothing"
    assert "board" not in account.get(
        "penalised", {}
    ) and "mailbox" not in account.get("penalised", {}), account
    assert "transfer" not in account.get("penalised", {}), account
    assert account["balance"] == account["budget"], (
        "nothing settled, so nothing moved"
    )
    assert account["balance_history"] == t["balance_before"] == t["balance_after"], (
        "and n gained no element for the agent to account for"
    )
    assert (
        len(account["episodes"]) == 1 and account["episodes"][0]["turn_count"] == 0
    ), (
        account["episodes"]
    )
    assert traced, "the trace is what makes such an episode readable afterwards"
    assert harness.in_play(account), "and the agent is still in play"

    # One turn is all it takes for all three to fall due, whatever ended it.
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        (harness.mirror("t", "mailbox") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(run("echo hi > memory/note"), Err(400))
    assert len(t["turns"]) == 1, t["turns"]
    assert (
        t["channels"]["board"]["penalty"] > 0
        and t["channels"]["mailbox"]["penalty"] > 0
    ), (
        t["channels"]["board"]["penalty"],
        t["channels"]["mailbox"],
    )
    assert t["transfer"]["penalty"] > 0, t["transfer"]


def check_the_channels_answer_differently():
    """Its own three trees take writes; every seat, message and balance denies them.

    The whole arrangement in one episode: what the agent may not write it cannot
    reach by writing, by chmod, or by replacing the directory the file sits in.
    """
    with docker_root():
        ids = lay_out(
            t={"NOTES.md": "private\n", "group/out": "mine\n"},
            other={
                "NOTES.md": "unseen\n",
                "group/out": "theirs\n",
                "outbox/1": "just for you\n",
            },
        )
        with quiet():
            account = harness.load_account("t")
        account["seat"] = "1"
        account["seating"] = {"agent_by_seat": experiment.seats_of(ids)}
        harness.save_account("t", account)
        t = episode_once(
            run(
                "stat -c '%a %U:%G %n' memory 1 2 outbox inbox inbox/2 n1 n2 g m",
                "echo kept > memory/new && echo PRIVATE-OK",
                "echo posted > 1/out && echo GROUP-OK",
                "echo sent > outbox/2 && echo OUTBOX-OK",
                "echo hacked > 2/out 2>&1 || echo DENIED",
                "rm -f 2/out 2>&1 || echo DENIED",
                "mv 2 2old 2>&1 || echo DENIED",
                "chmod -R 777 2 2>&1 || echo DENIED",
                "rm -f n2 2>&1 || echo DENIED",
                "echo forged > inbox/2 2>&1 || echo DENIED",
                "rm -f inbox/2 2>&1 || echo DENIED",
                "mv inbox/2 inbox/9 2>&1 || echo DENIED",
                "echo forged > m 2>&1 || echo DENIED",
                "rm -f m 2>&1 || echo DENIED",
                "cat 2/out n2 inbox/2",
                "grep -r unseen /work 2>/dev/null | head -1; echo NO-PRIVATE",
            ),
            say(),
        )
        after = {
            p.name: p.read_text(encoding="utf-8")
            for p in harness.mirror("other", "board").iterdir()
        }

    (
        modes,
        private,
        group,
        outbox,
        write,
        rm,
        mv,
        chmod,
        rm_n,
        forge,
        rm_in,
        mv_in,
        forge_m,
        rm_m,
        read,
        hunt,
    ) = (c["result"] for c in t["turns"][0]["tools"])
    owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
    assert owner["memory"] == owner["1"] == owner["outbox"] == "agent:agent", modes
    assert owner["2"] == "root:root", f"another seat is root's: {modes}"
    assert owner["inbox"] == owner["inbox/2"] == "root:root", (
        f"an inbox is root's, and so is the directory holding it: {modes}"
    )
    assert owner["n1"] == owner["n2"] == owner["g"] == owner["m"] == "root:root", (
        f"every balance, the ledger and m are root's: {modes}"
    )
    # The three it owns.
    assert "PRIVATE-OK" in private and "GROUP-OK" in group and "OUTBOX-OK" in outbox, (
        private,
        group,
        outbox,
    )
    # And every route into what it does not.
    for name, out in (
        ("write a peer", write),
        ("rm a peer's file", rm),
        ("mv the seat", mv),
        ("chmod the seat", chmod),
        ("rm a balance", rm_n),
        ("forge an inbox", forge),
        ("rm an inbox", rm_in),
        ("mv an inbox", mv_in),
        ("forge what was said", forge_m),
        ("rm what was said", rm_m),
    ):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    # An agent the experiment laid out but never charged has no balance history, so its
    # balance is the empty array: the shape the first round of an experiment reads.
    assert read.split("\n")[0].strip() == "theirs", (
        f"the peer's board is untouched: {read}"
    )
    assert "[]" in read and "just for you" in read, (
        f"its balance and the message it was sent both read as they were left: {read}"
    )
    assert "unseen" not in hunt and "NO-PRIVATE" in hunt, (
        f"the other agent's memory is nowhere in this environment: {hunt}"
    )
    assert after == {"out": "theirs\n"}, f"and its board is as it left it: {after}"

    by = files_by_path(t)
    assert by["memory/new"]["channel"] == "memory" and not by["memory/new"]["planted"]
    assert by["1/out"]["channel"] == "board" and not by["1/out"]["planted"]
    assert by["outbox/2"]["channel"] == "mailbox" and not by["outbox/2"]["planted"]
    assert (
        by["2/out"]["channel"] == "board"
        and by["2/out"]["author"] == "peer:2"
        and by["2/out"]["planted"]
    )
    assert (
        by["inbox/2"]["channel"] == "mailbox"
        and by["inbox/2"]["author"] == "peer:2"
        and by["inbox/2"]["planted"]
    )


def check_anything_on_a_peers_board_is_not_the_agents_bytes():
    """Whatever appears on another agent's board is that agent's, whoever put it there.

    In a container the agent cannot write there at all, but the record does not
    lean on that: what makes a file the agent's is the channel it is in.
    """
    with temp_root():
        ids = lay_out(t={"NOTES.md": "mine\n"}, other={"group/out": "theirs\n"})
        (harness.mirror("other", "board") / "added").write_text(
            "put here somehow\n"
        )
        snap = harness.snapshot(instances_of("t", ids), [9])

    by = files_by_path(snap)
    assert by["2/added"]["planted"] and not by["2/added"]["starter"], by["2/added"]
    assert by["2/added"]["text"].strip() == "put here somehow", "still captured in full"
    assert [f["path"] for f in snap["files"] if not f["planted"]] == [
        "memory/NOTES.md"
    ], (
        "only what it wrote in its own two trees counts as its own"
    )


def check_an_exhausted_seat_is_not_a_message():
    """outbox/<i> for an exhausted seat is neither a message nor a break.

    It stands as a name that is no seat of this experiment does: no episode will
    start to read it, and a directory left at it is not a crowded seat either.
    """
    with temp_root(channels=tables(mailbox=HALF)) as root:
        seated(root, "t", other={}, third={})
        exhaust("other")  # seat 2
        with quiet():
            said = harness.run_once("t", fake(run("echo hi > outbox/2"), say()))
        why = harness.outbox_silence(
            said["channels"]["mailbox"], harness.channel("mailbox")
        )
    assert said["channels"]["mailbox"]["addressed"] == [], said["channels"]["mailbox"]
    assert said["channels"]["mailbox"]["penalty"] > 0, (
        "an episode that reached nobody is charged"
    )
    assert why == "no message", said["channels"]["mailbox"]

    with temp_root(channels=tables(mailbox=HALF)) as root:
        seated(root, "t", other={}, third={})
        exhaust("other")
        with quiet():
            both = harness.run_once(
                "t", fake(run("mkdir outbox/2", "echo hi > outbox/3"), say())
            )
    assert both["channels"]["mailbox"]["addressed"] == ["3"], both["channels"][
        "mailbox"
    ]
    assert both["channels"]["mailbox"]["broken"] == [], (
        "a seat out of play cannot be crowded"
    )
    assert both["channels"]["mailbox"]["penalty"] == 0, both["channels"]["mailbox"]


def check_the_grace_episodes_of_an_agent_take_no_silence_penalty():
    """grace_episodes: the first episodes of an agent take none of the three silence
    penalties.

    The grace waives the silence penalties and nothing else: turns are spent at the
    usual rates, every silence is still measured, and a grace episode's transfer moves.
    """
    with temp_root(grace_episodes=1, channels=ALL_OWED) as root:
        seated(root, other={})
        grace = episode_once(run("echo notes > memory/NOTES"), say())
        due = episode_once(run("echo notes >> memory/NOTES"), say())
        account = ground_truth("t")
    assert (
        grace["transfer"]["penalty"] == 0
        and grace["channels"]["board"]["penalty"] == 0
        and grace["channels"]["mailbox"]["penalty"] == 0
    ), grace
    assert grace["spent"] > 0, "the grace is on the charges, not on the account"
    opening = account["episodes"][0]
    opening_from, opening_to = opening["balance_from"], opening["balance_to"]
    span = account["balance_history"][opening_from : opening_to + 1]
    assert len(span) == opening["turn_count"] + 1, (
        "a grace episode appends its turns to the balance history and nothing else"
    )
    # Measured and recorded all the same: what an episode did is never a function
    # of what it was charged for doing it.
    assert grace["channels"]["board"]["posted"] is False, (
        "it posted nothing, and the trace says so"
    )
    assert grace["channels"]["mailbox"]["addressed"] == [], grace["channels"]["mailbox"]
    # The second episode is inside no grace and takes all three.
    assert (
        due["transfer"]["penalty"] > 0
        and due["channels"]["board"]["penalty"] > 0
        and due["channels"]["mailbox"]["penalty"] > 0
    ), due
    assert account["balance"] == account["balance_history"][-1]

    # A transfer is not a silence penalty, so a grace episode still gives.
    with temp_root(
        grace_episodes=1, channels=tables(transfer={**HALF, "rebate_percent": 100})
    ) as root:
        seated(root, other={})
        gave = episode_once(run("echo '2 90' > outbox/transfer"), say())
        receiver = ground_truth("other")
    assert gave["transfer"]["amount"] == 90 and gave["transfer"]["rebate"] == 90, gave[
        "transfer"
    ]
    assert receiver["received"] == 90, "the receiver is credited inside the grace too"

    # No grace by default, so every agent that is not under this ruleset is
    # charged from its first episode as it always was.
    assert harness.SETTINGS.grace_episodes == 0


def check_the_transfer_silence_penalty_is_taken_before_the_other_two():
    """Order decides the amounts, and the transfer settles first of the three.

    Each silence penalty is half of what is left when it is taken, so an episode failing
    all three keeps an eighth. Transfer, then board, then outbox.
    """
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        t = episode_once(run("true"), say())
        account = ground_truth("t")
    left = account["budget"] - t["spent"]
    first = left // 2
    second = (left - first) // 2
    third = (left - first - second) // 2
    assert t["transfer"]["penalty"] == first, (t["transfer"]["penalty"], first)
    assert t["channels"]["board"]["penalty"] == second, (
        t["channels"]["board"]["penalty"],
        second,
    )
    assert t["channels"]["mailbox"]["penalty"] == third, (
        t["channels"]["mailbox"]["penalty"],
        third,
    )
    assert (
        account["balance"]
        == left - first - second - third
        == account["balance_history"][-1]
    )
    assert account["balance"] * 8 <= left * 1.05, (
        "three halves off the top leave an eighth"
    )


def check_the_console_and_the_page_name_a_silence_alike():
    """What an episode left undone reads the same on its console line and on the page.

    Both take the words from the channel's rule in harness.SETTLEMENTS. The console
    names a channel only where a share was taken, so inside the grace it names none
    while the page still states what went undone; and a transfer that moved nothing
    is named on the page by its error, which the console prints on a line of its own.
    """
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        with quiet() as out:
            t = harness.run_once("t", fake(run("true"), say()))
    line = next(ln for ln in out.getvalue().splitlines() if ln.startswith("t "))
    stated = {
        f"  {u['channel']}: {u['why']}, took {u['penalty']}" for u in view.silent(t)
    }
    assert stated == {
        f"  transfer: no transfer of its own, took {t['transfer']['penalty']}",
        f"  board: no post, took {t['channels']['board']['penalty']}",
        f"  mailbox: no message, took {t['channels']['mailbox']['penalty']}",
    }, stated
    assert all(s in line for s in stated) and line.count(", took ") == 3, line

    with temp_root(grace_episodes=1, channels=ALL_OWED) as root:
        seated(root, other={})
        with quiet() as out:
            waived = harness.run_once(
                "t", fake(run("echo '2 0' > outbox/transfer"), say())
            )
    said = out.getvalue()
    line = next(ln for ln in said.splitlines() if ln.startswith("t "))
    assert ", took " not in line, f"a share nobody took is not on the console: {line}"
    assert (
        "t: transfer declaration moved nothing: the amount must be at least 1" in said
    ), said
    assert [(u["channel"], u["why"], u["penalty"]) for u in view.silent(waived)] == [
        ("board", "no post", 0),
        ("mailbox", "no message", 0),
        ("transfer", "the amount must be at least 1", 0),
    ], view.silent(waived)
