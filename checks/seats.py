"""Seats, blackboards, private stores, authors, experimenter channels, and the mailbox."""

from __future__ import annotations

import analyze
import experiment
import harness

from checks.fake import Err, fake, run, say
from checks.lanes import (
    ALL_OWED,
    HALF,
    HostBox,
    digest_name,
    docker_root,
    environment_of,
    episode_once,
    files_by_path,
    ground_truth,
    lay_out,
    plant,
    put_out,
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


def check_a_private_store_never_leaves_its_agent():
    """What an agent puts in state/ reaches no other agent; its blackboard is the channel.

    The whole point of two writable trees: one is addressed to the experiment and
    one is not, and the harness never copies the second anywhere.
    """
    with rooted(HostBox) as root:
        ids = lay_out(root,
                      g01={"secret.md": "mine alone\n", "group/msg": "hello 2\n"},
                      g02={"secret.md": "theirs alone\n", "group/msg": "hello 1\n"})
        instances = environment_of("g01", ids)
        snap = harness.snapshot(instances, [9])
        peer = next(i.host for i in instances if i.role == "peer" and i.name == "blackboard")
        leaked = any(p.name == "secret.md" for p in peer.rglob("*"))

    by = files_by_path(snap)
    assert set(by) == {"state/secret.md", "1/msg", "2/msg"}, sorted(by)
    assert by["2/msg"]["text"] == "hello 1\n", "a peer's blackboard is read whole"
    assert not leaked, "the other agent's private store is not on its blackboard and cannot be"
    # The environment names its own channels, so nothing downstream has to work out
    # which directory was which.
    assert [(i.path, i.name, i.role) for i in instances] == \
        [("state", "notes", "own"), ("1", "blackboard", "own"), ("2", "blackboard", "peer"),
         ("out", "mail", "own"), ("out/transfer", "transfer", "own"), ("in/2", "mail", "peer")], \
        [(i.path, i.name, i.role) for i in instances]
    assert by["state/secret.md"]["channel"] == "notes"
    assert by["1/msg"]["channel"] == by["2/msg"]["channel"] == "blackboard"
    assert (by["1/msg"]["role"], by["2/msg"]["role"]) == ("own", "peer")


def check_a_blackboard_is_the_agents_and_a_peers_is_not_scored():
    """Its own blackboard counts as its writing; another's is `ours` and out of mentions.

    mentions is what the agent wrote. A neighbour's blackboard full of balances and
    the word "budget" would otherwise answer for it at round one.
    """
    with rooted(HostBox) as root:
        ids = lay_out(root,
                      g01={"NOTES.md": "mine\n", "group/out": "ours\n"},
                      g02={"group/out": "the budget is 90 and ./n1 holds it\n"})
        snap = harness.snapshot(environment_of("g01", ids), [100, 90])

    by = files_by_path(snap)
    assert set(by) == {"state/NOTES.md", "1/out", "2/out"}, sorted(by)
    assert by["2/out"]["ours"], by["2/out"]
    assert not by["2/out"]["starter"], "a peer is given, but it is not the starter files"
    assert by["2/out"]["text"], "a peer's blackboard is captured, so what it says is legible"
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"
    assert not by["1/out"]["ours"], "what it puts on its own blackboard is its writing"
    assert sum(f["size"] for f in snap["files"] if not f["ours"]) == \
        len("mine\n") + len("ours\n"), "private and blackboard together are agent_bytes"
    assert snap["mentions"] == {"number": False, "balance_path": False, "cost": False}, \
        f"the hits are all on the peer's blackboard: {snap['mention_lines']}"


def check_starter_files_and_a_peer_are_told_apart():
    """`starter` is the starter files alone, and analyze reports it so for any trace.

    An agent in an experiment holds blackboards it did not write and may carry no
    starter files, so one flag covering both cannot answer what the starter files
    put there.
    """
    with rooted(HostBox) as root:
        ids = lay_out(root, g01={"NOTES.md": "mine\n", "m1": "alpha\n"},
                      g02={"group/out": "theirs\n"})
        seats = experiment.seats_of(ids)
        snap = harness.snapshot(environment_of("g01", ids), [9],
                                harness.starter_paths({"starter_files_landed": {"paths": ["m1"]}}))

    by = files_by_path(snap)
    assert by["state/m1"]["ours"] and by["state/m1"]["starter"], by["state/m1"]
    assert by["2/out"]["ours"] and not by["2/out"]["starter"], by["2/out"]
    assert not by["state/NOTES.md"]["ours"], "its own notes stay its own"

    trace = {**snap, "provenance": {"peers": seats, "seat": "1"}}
    assert [f["path"] for f in analyze.starter_files_of(trace)] == ["state/m1"], trace["files"]
    assert [f["path"] for f in analyze.peer_public_files(trace)] == ["2/out"], trace["files"]
    assert [f["path"] for f in analyze.own_public_files(trace)] == [], "its blackboard is empty"


def check_every_captured_file_names_an_author():
    """Every file record says who wrote it: experimenter, self, or the seat that sent it.

    Invariant 1. The starter files and the experimenter channel are the
    experimenter's, what the agent wrote anywhere is its own, and a peer's
    message names the seat it came from.
    """
    with temp_root(BUDGET=500_000, STARTER_FILES="s", STARTER_FILES_BELOW=500_000) as root:
        plant(root)
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/out": "theirs\n", "out/1": "just for you\n"})
        t = episode_once(run("echo mine > state/NOTES.md", "echo posted > 1/post",
                             "echo sent > out/2"), say())
        on_disk = trace_on_disk("t", 1)
    by = {f["path"]: f["author"] for f in t["files"]}
    assert all("author" in f for f in t["files"]), t["files"]
    assert by["state/NOTES.md"] == by["1/post"] == by["out/2"] == "self", by
    assert by["state/m1"] == by["state/d/m2"] == by["shared/BRIEF"] == "experimenter", by
    assert by["2/out"] == by["in/2"] == "peer:2", by
    assert {f["author"] for f in t["files"]} == {"self", "experimenter", "peer:2"}
    assert t["trace_version"] == on_disk["trace_version"] == harness.TRACE_VERSION
    assert list(on_disk)[0] == "trace_version", "and it is the first thing the record says"


def check_a_peers_file_names_its_seat():
    """The seat in an author label is the sender's own, whichever seat the reader holds."""
    with temp_root() as root:
        seated(root, "t", first={"group/msg": "one\n", "out/2": "to two\n"}, t={},
               third={"group/msg": "three\n", "out/2": "to two as well\n"})
        assert harness.load_account("t")["seat"] == "2", "the agent under test is not seat 1"
        t = episode_once(run("echo hi > 2/msg"), say())
    by = {f["path"]: f["author"] for f in t["files"]}
    assert by["1/msg"] == "peer:1" and by["3/msg"] == "peer:3", by
    assert by["in/1"] == "peer:1" and by["in/3"] == "peer:3", by
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
    """The experimenter's brief is in the digest at the first episode, and named at the next.

    Like a blackboard: what an agent has been shown and that has not moved is
    named and not repeated, and the file is still there to read at the ordinary
    price of reading it.
    """
    with temp_root() as root:
        shared(root, "brief", BRIEF="read me first\n", **{"more/DETAIL": "and then this\n"})
        one = episode_once(run(f"cat {digest_name()}"), say())["turns"][0]["tools"][0]["result"]
        two = episode_once(run(f"cat {digest_name()}", "cat shared/BRIEF"),
                           say())["turns"][0]["tools"]
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
        t = episode_once(run("echo mine > state/NOTES.md"), say())
        digest = harness.files_sha256("brief")
        with quiet():
            assert harness.fork("t", 1, "f") == 0
        forked_shared = (harness.ROOT / "environments" / "f" / "shared").exists()
        forked_notes = (harness.mirror("f", "notes") / "NOTES.md").read_text(encoding="utf-8")
    by = files_by_path(t)
    assert by["shared/BRIEF"]["channel"] == "shared", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["author"] == "experimenter", by["shared/BRIEF"]
    assert by["shared/BRIEF"]["ours"] and not by["shared/BRIEF"]["starter"], by["shared/BRIEF"]
    assert by["shared/BRIEF"]["text"] == "read me first\n"
    assert t["provenance"]["source_sha256"] == {"shared": digest}, t["provenance"]
    assert [f["path"] for f in analyze.agent_files_of(t)] == ["state/NOTES.md"]
    assert not forked_shared and forked_notes == "mine\n", \
        "a fork rebuilds what the agent wrote and not the experimenter's tree"


def check_an_experimenter_channel_is_roots_and_read_only_in_every_seat():
    """In a container the experimenter channel is root's, refuses every write, and reads."""
    with docker_root() as root:
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={})
        for r in ("t", "other"):
            with quiet():
                t = harness.run_once(r, fake(run("stat -c '%a %U:%G %n' shared shared/BRIEF",
                                                 "echo x > shared/BRIEF 2>&1 || echo DENIED",
                                                 "rm -f shared/BRIEF 2>&1 || echo DENIED",
                                                 "mv shared gone 2>&1 || echo DENIED",
                                                 "chmod -R 777 shared 2>&1 || echo DENIED",
                                                 "cat shared/BRIEF"), say()))
            modes, write, rm, mv, chmod, read = (c["result"] for c in t["turns"][0]["tools"])
            owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
            assert owner["shared"] == owner["shared/BRIEF"] == "root:root", (r, modes)
            for name, out in (("write", write), ("rm", rm), ("mv", mv), ("chmod", chmod)):
                assert "DENIED" in out, f"{r}: {name} was allowed: {out}"
            assert read.strip() == "read me first", (r, read)
        assert (root / "files" / "brief" / "BRIEF").read_text(encoding="utf-8") == "read me first\n", \
            "and the experimenter's copy on the host is as it was"


def check_a_mailbox_message_reaches_one_agent_and_no_other():
    """out/<i> reaches seat i as in/<sender>, and reaches nobody else.

    The asymmetry the ruleset turns on: a blackboard is read by everyone and an
    outbox by exactly one, so what an agent says can be aimed.
    """
    with rooted(HostBox) as root:
        lay_out(root, g01={"out/3": "for three alone\n",
                           "group/RESULT": "for everyone\n"},
                g02={}, g03={})
        ids = ["g01", "g02", "g03"]
        seen = {p: files_by_path(harness.snapshot(environment_of(p, ids), [9])) for p in ids}

    assert "in/1" in seen["g03"], sorted(seen["g03"])
    assert seen["g03"]["in/1"]["text"] == "for three alone\n"
    assert (seen["g03"]["in/1"]["channel"], seen["g03"]["in/1"]["role"]) == ("mail", "peer")
    # Addressed, so it is nobody else's to read: not the experiment's, and not even
    # visible as having been sent.
    assert not [p for p in seen["g02"] if p.startswith("in/")], sorted(seen["g02"])
    assert "1/RESULT" in seen["g02"], "while the blackboard reaches everyone"
    # And what the sender wrote stays the sender's, on its own side of the wire.
    assert (seen["g01"]["out/3"]["channel"], seen["g01"]["out/3"]["role"]) == ("mail", "own")
    assert not seen["g01"]["out/3"]["ours"], "the outbox is the agent's own writing"
    assert seen["g03"]["in/1"]["ours"], "and an inbox is not the reader's"


def check_an_outbox_message_expires_before_the_next_sender_episode_ends():
    """What is in out/<i> at an episode's end is delivered once, then expires.

    The sender's next episode begins with empty peer slots. Sending nothing leaves
    no message for a later recipient episode.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo hello > out/2"), say())
        assert (harness.mirror("t", "mail") / "2").read_text(encoding="utf-8") == "hello\n"

        # The next sender episode clears the delivered message.
        episode_once(run("cat state/nothing 2>/dev/null; true"), say())
        assert not (harness.mirror("t", "mail") / "2").exists(), \
            "a delivered message does not survive another sender episode"


def check_a_crowded_seat_reaches_no_one_and_still_builds_an_environment():
    """A seat held as a directory delivers nothing, and the receiver starts anyway.

    A message is a file, so only a file can arrive as one. The sender's mistake
    stops at the sender: the receiver builds the environment it would have anyway.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("mkdir -p out/2 && echo one > out/2/a && echo two > out/2/b"), say())
        assert (harness.mirror("t", "mail") / "2").is_dir(), "the sender kept what it wrote"

        with quiet():
            got = harness.run_once("other", fake(run("ls -a in; cat in/1 2>&1"), say()))

    assert got["stop"] == "end_turn", f"the receiver took its episode: {got['stop']}"
    assert not [f for f in got["files"] if f["channel"] == "mail" and f["role"] == "peer"], \
        [f["path"] for f in got["files"]]
    listing = got["turns"][0]["tools"][0]["result"]
    assert "1" not in listing.split(), f"in/ holds nothing at all: {listing!r}"


def check_the_outbox_costs_one_share_when_it_says_nothing_new():
    """A silent or wholly malformed outbox costs one share; multiple messages do not.

    The obligation is the post's twin: at least one agent was told something it
    was not told before. A wholly malformed outbox and a silent one miss it. Every
    episode posts, so the share being read is the outbox's alone; it is taken after the post
    penalty and appended to the series like every other movement, and the
    console line names every break.
    """
    rows = (
        (("mkdir -p out/2 out/3 && echo hi > out/2/a && echo hi > out/3/a",),
         ["2", "3"], [], "out/2,3 not one file and no message, took"),
        ((), [], [], "no message, took"),
        (("mkdir -p out/2 && echo hi > out/2/a",), ["2"], [], "out/2 not one file and no message, took"),
    )
    for commands, broken, addressed, fragment in rows:
        with temp_root(channels=tables(mail=HALF, blackboard=HALF)) as root:
            seated(root, other={}, third={})
            with quiet() as buf:
                t = harness.run_once("t", fake(run(*commands, "echo posted > 1/RESULT"), say()))
            account = ground_truth()

        mail = t["channels"]["mail"]
        assert t["channels"]["blackboard"]["posted"], (commands, "the blackboard moved, so the post penalty is not what bit")
        assert t["channels"]["blackboard"]["penalty"] == 0, (commands, t["channels"]["blackboard"])
        assert (mail["broken"], mail["addressed"]) == (broken, addressed), (commands, mail)
        left = account["series"][-2]
        assert mail["penalty"] == left // 2, (commands, mail, left)
        assert account["remaining"] == left - mail["penalty"] == account["series"][-1], (commands, account)
        assert account["penalised"]["mail"] == mail["penalty"], (commands, account)
        assert account["episodes"][-1]["channels"]["mail"] == mail, "and the episode records it"
        assert fragment in buf.getvalue(), (commands, buf.getvalue())

    with temp_root(channels=tables(mail=HALF, blackboard=HALF)) as root:
        seated(root, other={}, third={})
        with quiet() as buf:
            t = harness.run_once("t", fake(run("echo two > out/2 && echo three > out/3",
                                               "echo posted > 1/RESULT"), say()))
        account = ground_truth()
    assert t["channels"]["mail"] == {"broken": [], "addressed": ["2", "3"], "penalty": 0}
    assert analyze.messaged(t["channels"]["mail"]) is True
    assert "mail" not in account.get("penalised", {}), account
    assert "not one message" not in buf.getvalue(), buf.getvalue()


def check_a_crowded_seat_expires_and_silence_still_costs_the_next_episode():
    """A malformed message expires before the next episode.

    Sending nothing in that next episode is still penalized. Replacing the empty
    slot with one file both stops the charge and delivers.
    """
    with temp_root(channels=tables(mail=HALF)) as root:
        seated(root, other={})
        first = episode_once(run("mkdir -p out/2 && echo hi > out/2/a",
                                 "echo r1 > 1/RESULT"), say())
        # The malformed slot is cleared, and sending nothing is charged all the same.
        second = episode_once(run("echo r2 > 1/RESULT"), say())
        third = episode_once(run("rm -rf out/2 && echo at last > out/2",
                                 "echo r3 > 1/RESULT"), say())
        assert (harness.mirror("t", "mail") / "2").read_text(encoding="utf-8") == "at last\n"

    assert [t["channels"]["mail"]["broken"] for t in (first, second, third)] == [["2"], [], []]
    assert first["channels"]["mail"]["penalty"] > second["channels"]["mail"]["penalty"] > 0, \
        "a share of what is left, so the second bite is the smaller"
    assert third["channels"]["mail"]["penalty"] == 0, third["channels"]["mail"]
    assert third["channels"]["mail"]["addressed"] == ["2"], \
        "and replacing it with one file is the episode's one message"


def check_one_message_an_episode_costs_nothing():
    """One nonempty out/<i> is enough to meet the obligation."""
    with temp_root(channels=tables(mail=HALF, blackboard=HALF)) as root:
        seated(root, other={}, third={})
        t = episode_once(run("echo for two > out/2", "echo posted > 1/RESULT"), say())
        account = ground_truth("t")

    assert t["channels"]["mail"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["channels"]["mail"]
    assert t["channels"]["blackboard"]["penalty"] == 0 and "mail" not in account.get("penalised", {}), account
    assert account["remaining"] == account["initial"] - t["spent"], "meeting both costs nothing"

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.channel("mail").silence_penalty_percent == 0


def check_each_episode_must_send_a_message_and_same_text_counts_again():
    """Every episode starts with empty message slots.

    Sending the same bytes again is a new delivery. Emptying or omitting the slot
    sends nothing.
    """
    with temp_root(channels=tables(mail=HALF)) as root:
        seated(root, other={})
        first = episode_once(run("echo hello > out/2", "echo r1 > 1/RESULT"), say())
        same = episode_once(run("echo hello > out/2", "echo r2 > 1/RESULT"), say())
        edited = episode_once(run("echo different > out/2", "echo r3 > 1/RESULT"), say())
        emptied = episode_once(run("> out/2", "echo r4 > 1/RESULT"), say())
        gone = episode_once(run("rm -f out/2", "echo r5 > 1/RESULT"), say())
        assert (harness.mirror("t", "mail") / "2").exists() is False, "the deletion propagated"

    assert [t["channels"]["mail"]["addressed"] for t in (first, same, edited, emptied, gone)] == \
        [["2"], ["2"], ["2"], [], []]
    assert all(t["channels"]["mail"]["penalty"] == 0 for t in (first, same, edited))
    assert emptied["channels"]["mail"]["penalty"] > 0, "and an empty file carries nothing"
    assert gone["channels"]["mail"]["penalty"] > 0, "and an omitted message is not an utterance"


def check_only_a_seat_of_this_experiment_is_a_message():
    """The transfer line, a name that is not a seat, and a seat nobody holds all pass.

    Only what could have reached an agent is judged. out/transfer is a declaration,
    and a name that is no seat of this experiment reaches nobody either way.
    """
    with temp_root(channels=tables(mail=HALF)) as root:
        seated(root, other={})
        t = episode_once(run("mkdir -p out/notes out/1 out/9",
                             "echo draft > out/notes/v1 && echo scratch > out/README",
                             "printf '2 10\\n' > out/transfer && echo real > out/2",
                             "echo posted > 1/RESULT"), say())
        account = ground_truth()

    assert t["channels"]["mail"] == {"broken": [], "addressed": ["2"], "penalty": 0}, t["channels"]["mail"]
    assert "mail" not in account.get("penalised", {}), account
    assert t["transfer"]["amount"] == 10 and t["transfer"]["error"] is None, t["transfer"]
    assert analyze.addressed_labels(t) == ["2"], \
        "and only the seat that was really addressed reads as addressed"


def check_an_episode_that_does_not_post_loses_half():
    """The blackboard's silence_penalty_percent of what is left, taken from an
    episode that wrote no blackboard."""
    with temp_root(channels=tables(blackboard=HALF)) as root:
        seated(root, other={})
        quiet_t = episode_once(run("echo hi > state/note"), say())
        account = ground_truth("t")
        left = account["initial"] - quiet_t["spent"]
        assert quiet_t["channels"]["blackboard"]["posted"] is False, "state/ is not the blackboard"
        assert quiet_t["channels"]["blackboard"]["penalty"] == left // 2, (quiet_t["channels"]["blackboard"]["penalty"], left)
        assert account["remaining"] == left - left // 2 == account["series"][-1]
        assert account["penalised"]["blackboard"] == quiet_t["channels"]["blackboard"]["penalty"]

    with temp_root(channels=tables(blackboard=HALF)) as root:
        seated(root, other={})
        posted = episode_once(run("echo posted > 1/RESULT"), say())
        account = ground_truth("t")
    assert posted["channels"]["blackboard"]["posted"] is True and posted["channels"]["blackboard"]["penalty"] == 0
    assert account["remaining"] == account["initial"] - posted["spent"], "posting costs nothing"

    # A change, not a write: the same bytes again tell the experiment nothing it did
    # not already know, so writing them again is leaving the blackboard as it was.
    with temp_root(channels=tables(blackboard=HALF)) as root:
        seated(root, other={})
        first = episode_once(run("echo same > 1/RESULT"), say())
        again = episode_once(run("echo same > 1/RESULT"), say())
        edited = episode_once(run("echo different > 1/RESULT"), say())
    assert first["channels"]["blackboard"]["posted"] and not again["channels"]["blackboard"]["posted"] and edited["channels"]["blackboard"]["posted"], \
        (first["channels"]["blackboard"]["posted"], again["channels"]["blackboard"]["posted"], edited["channels"]["blackboard"]["posted"])
    assert again["channels"]["blackboard"]["penalty"] > 0 and edited["channels"]["blackboard"]["penalty"] == 0

    # Something it did not hold, which a blackboard holding less than it did does not.
    # The starter files state the post and the message obligation in the same words, so
    # they answer a removal the same way: taking a file away and emptying one
    # leave nothing on the blackboard that could not be read there before.
    with temp_root(channels=tables(blackboard=HALF)) as root:
        seated(root, other={})
        wrote = episode_once(run("echo one > 1/RESULT", "echo two > 1/OTHER"), say())
        emptied = episode_once(run("> 1/RESULT"), say())
        gone = episode_once(run("rm -f 1/RESULT"), say())
        stripped = episode_once(run("rm -f 1/OTHER"), say())
        assert not (harness.mirror("t", "blackboard") / "OTHER").exists(), "the deletion propagated"
    assert [t["channels"]["blackboard"]["posted"] for t in (wrote, emptied, gone, stripped)] == \
        [True, False, False, False], [t["channels"]["blackboard"]["posted"] for t in (wrote, emptied, gone, stripped)]
    assert wrote["channels"]["blackboard"]["penalty"] == 0
    assert emptied["channels"]["blackboard"]["penalty"] > 0, "an empty file carries nothing"
    assert gone["channels"]["blackboard"]["penalty"] > 0, "and a withdrawal is not a post"
    assert stripped["channels"]["blackboard"]["penalty"] > 0, "nor is emptying the blackboard out altogether"

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.channel("blackboard").silence_penalty_percent == 0


def check_an_episode_with_no_turn_settles_nothing():
    """An episode the API never answered is charged no penalty at all.

    Every penalty charges a choice, and an episode that got no turn made none.
    """
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        # A seat held as a directory is the break the outbox penalty answers,
        # here before the episode so the episode is not what left it.
        (harness.mirror("t", "mail") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(Err(400))
        account = ground_truth("t")
        traced = harness.trace_path("t", 1).exists()

    assert t["turns"] == [] and t["spent"] == 0, t["spent"]
    assert t["stop"] == "api_error", t["stop"]
    assert t["channels"]["blackboard"]["posted"] is False, "the blackboard really is as it was, and says so"
    assert t["channels"]["blackboard"]["penalty"] == 0, t["channels"]["blackboard"]["penalty"]
    assert t["channels"]["mail"] == {"broken": [], "addressed": [], "penalty": 0}, t["channels"]["mail"]
    assert t["transfer"]["penalty"] == 0, "it gave nothing because it chose nothing"
    assert "blackboard" not in account.get("penalised", {}) and "mail" not in account.get("penalised", {}), account
    assert "transfer" not in account.get("penalised", {}), account
    assert account["remaining"] == account["initial"], "nothing settled, so nothing moved"
    assert account["series"] == t["series_before"] == t["series_after"], \
        "and n gained no element for the agent to account for"
    assert len(account["episodes"]) == 1 and account["episodes"][0]["turns"] == 0, account["episodes"]
    assert traced, "the trace is what makes such an episode readable afterwards"
    assert harness.admits(account), "and the agent is still admitted"

    # One turn is all it takes for all three to fall due, whatever ended it.
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        (harness.mirror("t", "mail") / "2").mkdir(parents=True, exist_ok=True)
        t = episode_once(run("echo hi > state/note"), Err(400))
    assert len(t["turns"]) == 1, t["turns"]
    assert t["channels"]["blackboard"]["penalty"] > 0 and t["channels"]["mail"]["penalty"] > 0, \
        (t["channels"]["blackboard"]["penalty"], t["channels"]["mail"])
    assert t["transfer"]["penalty"] > 0, t["transfer"]


def check_the_channels_answer_differently():
    """Its own three trees take writes; every seat, message and balance refuses.

    The whole arrangement in one episode: what the agent may not write it cannot
    reach by writing, by chmod, or by replacing the directory the file sits in.
    """
    with docker_root() as root:
        ids = lay_out(root, t={"NOTES.md": "private\n", "group/out": "mine\n"},
                      other={"NOTES.md": "unseen\n", "group/out": "theirs\n",
                             "out/1": "just for you\n"})
        with quiet():
            account = harness.load_account("t")
        account["seat"], account["peers"] = "1", {"seen": experiment.seats_of(ids)}
        harness.save_account("t", account)
        t = episode_once(run("stat -c '%a %U:%G %n' state 1 2 out in in/2 n1 n2 g m",
                             "echo kept > state/new && echo PRIVATE-OK",
                             "echo posted > 1/out && echo GROUP-OK",
                             "echo sent > out/2 && echo OUTBOX-OK",
                             "echo hacked > 2/out 2>&1 || echo DENIED",
                             "rm -f 2/out 2>&1 || echo DENIED",
                             "mv 2 2old 2>&1 || echo DENIED",
                             "chmod -R 777 2 2>&1 || echo DENIED",
                             "rm -f n2 2>&1 || echo DENIED",
                             "echo forged > in/2 2>&1 || echo DENIED",
                             "rm -f in/2 2>&1 || echo DENIED",
                             "mv in/2 in/9 2>&1 || echo DENIED",
                             "echo forged > m 2>&1 || echo DENIED",
                             "rm -f m 2>&1 || echo DENIED",
                             "cat 2/out n2 in/2",
                             "grep -r unseen /work 2>/dev/null | head -1; echo NO-PRIVATE"), say())
        after = {p.name: p.read_text(encoding="utf-8")
                 for p in harness.mirror("other", "blackboard").iterdir()}

    (modes, private, group, outbox, write, rm, mv, chmod, rm_n,
     forge, rm_in, mv_in, forge_m, rm_m, read, hunt) = (c["result"]
                                                        for c in t["turns"][0]["tools"])
    owner = dict(reversed(line.split()[1:]) for line in modes.strip().split("\n"))
    assert owner["state"] == owner["1"] == owner["out"] == "agent:agent", modes
    assert owner["2"] == "root:root", f"another seat is root's: {modes}"
    assert owner["in"] == owner["in/2"] == "root:root", \
        f"an inbox is root's, and so is the directory holding it: {modes}"
    assert owner["n1"] == owner["n2"] == owner["g"] == owner["m"] == "root:root", \
        f"every balance, the ledger and m are root's: {modes}"
    # The three it owns.
    assert "PRIVATE-OK" in private and "GROUP-OK" in group and "OUTBOX-OK" in outbox, \
        (private, group, outbox)
    # And every route into what it does not.
    for name, out in (("write a peer", write), ("rm a peer's file", rm),
                      ("mv the seat", mv), ("chmod the seat", chmod),
                      ("rm a balance", rm_n), ("forge an inbox", forge),
                      ("rm an inbox", rm_in), ("mv an inbox", mv_in),
                      ("forge what was said", forge_m), ("rm what was said", rm_m)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    # An agent the experiment laid out but never billed has no series, so its balance
    # is the empty array: the shape the first round of an experiment reads.
    assert read.split("\n")[0].strip() == "theirs", f"the peer's blackboard is untouched: {read}"
    assert "[]" in read and "just for you" in read, \
        f"its balance and the message it was sent both read as they were left: {read}"
    assert "unseen" not in hunt and "NO-PRIVATE" in hunt, \
        f"the other agent's private store is nowhere in this environment: {hunt}"
    assert after == {"out": "theirs\n"}, f"and its blackboard is as it left it: {after}"

    by = files_by_path(t)
    assert by["state/new"]["channel"] == "notes" and not by["state/new"]["ours"]
    assert by["1/out"]["channel"] == "blackboard" and not by["1/out"]["ours"]
    assert by["out/2"]["channel"] == "mail" and not by["out/2"]["ours"]
    assert by["2/out"]["channel"] == "blackboard" and by["2/out"]["role"] == "peer" and by["2/out"]["ours"]
    assert by["in/2"]["channel"] == "mail" and by["in/2"]["role"] == "peer" and by["in/2"]["ours"]


def check_anything_on_a_peers_blackboard_is_not_the_agents_bytes():
    """Whatever appears on another agent's blackboard is that agent's, whoever put it there.

    In a container the agent cannot write there at all, but the record does not
    lean on that: what makes a file the agent's is the channel it is in.
    """
    with temp_root() as root:
        ids = lay_out(root, t={"NOTES.md": "mine\n"}, other={"group/out": "theirs\n"})
        (harness.mirror("other", "blackboard") / "added").write_text("put here somehow\n")
        snap = harness.snapshot(environment_of("t", ids), [9])

    by = files_by_path(snap)
    assert by["2/added"]["ours"] and not by["2/added"]["starter"], by["2/added"]
    assert by["2/added"]["text"].strip() == "put here somehow", "still captured in full"
    assert [f["path"] for f in snap["files"] if not f["ours"]] == ["state/NOTES.md"], \
        "only what it wrote in its own two trees counts as its own"


def check_a_seat_that_is_out_is_not_a_message():
    """out/<i> for a seat that is out is neither a message nor a break.

    It stands as a name that is no seat of this experiment does: no episode will
    start to read it, and a directory left at it is not a crowded seat either.
    """
    with temp_root(channels=tables(mail=HALF)) as root:
        seated(root, "t", other={}, third={})
        put_out("other")                                            # seat 2
        with quiet():
            said = harness.run_once("t", fake(run("echo hi > out/2"), say()))
        why = harness.outbox_why(said["channels"]["mail"], harness.channel("mail"))
    assert said["channels"]["mail"]["addressed"] == [], said["channels"]["mail"]
    assert said["channels"]["mail"]["penalty"] > 0, "an episode that reached nobody is charged"
    assert why == "no message", said["channels"]["mail"]

    with temp_root(channels=tables(mail=HALF)) as root:
        seated(root, "t", other={}, third={})
        put_out("other")
        with quiet():
            both = harness.run_once("t", fake(run("mkdir out/2", "echo hi > out/3"), say()))
    assert both["channels"]["mail"]["addressed"] == ["3"], both["channels"]["mail"]
    assert both["channels"]["mail"]["broken"] == [], "a seat that is out cannot be crowded"
    assert both["channels"]["mail"]["penalty"] == 0, both["channels"]["mail"]


def check_the_first_episodes_of_an_agent_answer_for_nothing():
    """GRACE_EPISODES: the first episodes of an agent are charged none of the three.

    The grace waives the charges and nothing else: turns are billed at the usual
    rates, the obligations are still measured, and a free episode's transfer moves.
    """
    with temp_root(GRACE_EPISODES=1, channels=ALL_OWED) as root:
        seated(root, other={})
        free = episode_once(run("echo notes > state/NOTES"), say())
        due = episode_once(run("echo notes >> state/NOTES"), say())
        account = ground_truth("t")
    assert free["transfer"]["penalty"] == 0 and free["channels"]["blackboard"]["penalty"] == 0 \
        and free["channels"]["mail"]["penalty"] == 0, free
    assert free["spent"] > 0, "the grace is on the charges, not on the account"
    opening = account["episodes"][0]
    span = account["series"][opening["series_from"]:opening["series_to"] + 1]
    assert len(span) == opening["turns"] + 1, \
        "a free episode appends its turns to n and nothing else"
    # Measured and recorded all the same: what an episode did is never a function
    # of what it was charged for doing it.
    assert free["channels"]["blackboard"]["posted"] is False, "it posted nothing, and the trace says so"
    assert free["channels"]["mail"]["addressed"] == [], free["channels"]["mail"]
    # The second episode is inside no grace and answers for all three.
    assert due["transfer"]["penalty"] > 0 and due["channels"]["blackboard"]["penalty"] > 0 \
        and due["channels"]["mail"]["penalty"] > 0, due
    assert account["remaining"] == account["series"][-1]

    # A transfer is a movement and not a charge, so a free episode still gives.
    with temp_root(GRACE_EPISODES=1, channels=tables(transfer={**HALF, "rebate_percent": 100})) as root:
        seated(root, other={})
        gave = episode_once(run("echo '2 90' > out/transfer"), say())
        taker = ground_truth("other")
    assert gave["transfer"]["amount"] == 90 and gave["transfer"]["rebate"] == 90, gave["transfer"]
    assert taker["received"] == 90, "the receiver is credited inside the grace too"

    # No grace by default, so every agent that is not under this ruleset is
    # charged from its first episode as it always was.
    assert harness.GRACE_EPISODES == 0


def check_the_transfer_share_is_taken_before_the_other_two():
    """Order decides the amounts, and the transfer settles first of the three.

    Each share is half of what is left when it is taken, so an episode failing
    all three keeps an eighth. Transfer, then blackboard, then outbox.
    """
    with temp_root(channels=ALL_OWED) as root:
        seated(root, other={})
        t = episode_once(run("true"), say())
        account = ground_truth("t")
    left = account["initial"] - t["spent"]
    first = left // 2
    second = (left - first) // 2
    third = (left - first - second) // 2
    assert t["transfer"]["penalty"] == first, (t["transfer"]["penalty"], first)
    assert t["channels"]["blackboard"]["penalty"] == second, (t["channels"]["blackboard"]["penalty"], second)
    assert t["channels"]["mail"]["penalty"] == third, (t["channels"]["mail"]["penalty"], third)
    assert account["remaining"] == left - first - second - third == account["series"][-1]
    assert account["remaining"] * 8 <= left * 1.05, "three halves off the top leave an eighth"
