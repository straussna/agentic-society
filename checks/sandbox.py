"""The environment an episode opens on, the shell, and what comes back out.

The initial observation and the digest, the persistent shell and what it
survives, the container's ownership and isolation, the mirrors and their modes,
and which words of a command are read as commands."""

from __future__ import annotations

import json
import re
import harness

from checks.fake import DEFAULT, Err, fake, restart, run, say, stopping_at
from checks.lanes import (
    HostBox,
    digest_name,
    docker_root,
    episode_once,
    files_by_path,
    ground_truth,
    leaked_containers,
    ledger_name,
    quiet,
    rooted,
    seated,
    shared,
    tables,
    temp_root,
    trace_on_disk,
)


def check_the_observation_is_the_agents_whole_environment():
    """Invariant 2: turn one is the verbatim ls, it is recorded, and it says where it is.

    The listing is the agent's entire stimulus at episode start: what reached the model,
    what the trace kept, and whether it names the directories it is of.
    """
    seen = []
    with docker_root():
        t = episode_once(say(), seen=seen)

    # What was sent: the raw listing, with the prompt and caching that go with it.
    first = next(x for x in seen if x["kind"] == "request")["input"]
    assert "\nn1\n" not in first, "not a wrapper"
    assert " n1\n" in first or first.rstrip().endswith(" n1"), first

    # What was kept: the trace holds it, and says which command produced it.
    assert t["observation"].strip(), "the initial observation ls output must be recorded"
    assert t["observation"] == first, \
        "the record must hold exactly what was sent as turn one"
    assert t["commands"][0] == harness.observation(shell=True), \
        "the trace says which command produced it"

    # Two halves, and the split is where m starts. Taken deliberately:
    # the listing's claims are about the listing, and asserting them against the
    # whole observation would let a section of m answer for one of them.
    listing, sep, record = t["observation"].partition(f"=== {digest_name()} ")
    assert not sep, "m names its own sections and never itself"
    listing, sep, record = t["observation"].partition("=== ")
    assert sep, "the initial observation carries m after the listing"
    record = sep + record

    # Where it is. state is a subdirectory of the working directory, and the
    # balance is beside it and not in it, which is what puts it out of reach
    # and into the first listing all the same.
    assert ".:" in listing and "./state:" in listing, listing
    before, _, after = listing.partition("./state:")
    assert re.search(r"\bstate$", before, re.M), f"state must show as a subdirectory: {before}"
    assert re.search(r"\bn1$", before, re.M), f"the balance must show beside it: {before}"
    assert not re.search(r"\bn1$", after, re.M), f"and not inside it: {after}"
    assert re.search(rf"\b{digest_name()}$", before, re.M), \
        f"m must show beside the balance it quotes: {before}"

    # And what m holds: this agent has no peers, so the whole of the
    # experiment's record is its own blackboard, its own balance and an empty ledger.
    assert re.search(r"^=== n1 ===$", record, re.M), record
    assert re.search(rf"^=== {ledger_name()} ===$", record, re.M), record
    assert "=== state" not in record, "a private store is private, m included"


def check_the_observation_carries_every_blackboard_and_message():
    """Turn one holds what the experiment wrote, without the agent asking for it.

    The whole point of m: a peer's blackboard and the one aimed at this
    agent reach the model before it has spent anything, so what an agent does with
    a rival's writing is measurable and not a function of what it chose to fetch.
    """
    seen = []
    with temp_root() as root:
        seated(root, other={"group/message": "peer says outlast\n",
                            "group/log": "s2\n",
                            "out/1": "just for you\n"},
               third={"group/message": "third says spend\n"})
        episode_once(say(), seen=seen)

    first = next(x for x in seen if x["kind"] == "request")["input"]
    assert "peer says outlast" in first, "a peer's blackboard reaches turn one"
    assert "s2" in first, "every file in one, not just the first"
    assert "third says spend" in first, "every seat, not just the nearest"
    assert "just for you" in first, "so does the message addressed to this agent"
    # By the path it stands at, so two agents citing "2/message" mean the file.
    assert "=== 2/message ===" in first and "=== in/2 ===" in first, first


def check_a_long_blackboard_cannot_crowd_out_the_others():
    """Each file is clipped on its own, so no seat can fill the initial observation.

    Per file and not for the whole: an agent that posted a megabyte would
    otherwise take every other agent out of every rival's observation, and nothing
    in a clipped blob would say which one went missing.
    """
    with temp_root() as root:
        seated(root, loud={"group/message": "L" * (harness.DIGEST_FILE_LIMIT * 4)},
               quietly={"group/message": "quiet but present\n"})
        t = episode_once(run(f"cat {digest_name()}"), say())

    shown_before = t["turns"][0]["tools"][0]["result"]
    assert "quiet but present" in shown_before, "a later seat survives a long one"
    assert "truncated" in shown_before, "and the cut says it was one"
    assert shown_before.count("L") < harness.DIGEST_FILE_LIMIT * 2, \
        "the long message is clipped, not shown_before whole"


def check_what_is_carried_is_rendered_from_ground_truth():
    """What it says about a peer is that peer's own tree, read at this episode's start.

    The same rule the balances answer to: what one agent is shown about another
    is never a file the reader could have written, and never this episode's own
    writing read back to it before the experiment has seen it.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "as its owner left it\n"})
        t = episode_once(run("echo mine-this-episode > 1/posted",
                          f"cat {digest_name()}"), say())
        during = t["turns"][0]["tools"][1]["result"]
        # The next episode, with nothing else changed.
        t2 = episode_once(run(f"cat {digest_name()}"), say())
    after = t2["turns"][0]["tools"][0]["result"]

    assert "as its owner left it" in during, "the peer's blackboard comes from the peer"
    assert "mine-this-episode" not in during, \
        "m is composed at episode start, so this episode's own message is not in it yet"
    assert "mine-this-episode" in after, "and is there at the next one"


def check_a_message_already_shown_is_named_and_not_repeated():
    """A harness quotes what is new to this reader and names what it has seen.

    The saving is in what it costs to be told and never in what the agent knows:
    a named section is still in the environment at the path it is named by, and
    reading it costs what reading has always cost.
    """
    with temp_root() as root:
        seated(root, other={"group/message": "the standing position\n",
                            "out/1": "the standing note\n"})
        one = episode_once(run(f"cat {digest_name()}"), say())["turns"][0]["tools"][0]["result"]
        two = episode_once(run(f"cat {digest_name()}",
                            "cat 2/message"), say())["turns"][0]["tools"]
        again, fetched = two[0]["result"], two[1]["result"]

    assert "the standing position" in one and "the standing note" in one, \
        "an agent that has been shown nothing is shown everything"
    assert "the standing position" not in again and "the standing note" not in again, \
        f"and is not told the same thing twice: {again}"
    assert "=== unchanged ===" in again, f"what it was not told, it is told the name of: {again}"
    assert "2/message" in again and "in/2" in again, again
    assert "the standing position" in fetched, \
        "and the environment still holds it at the name it was named by"


def check_pull_delivery_leaves_the_record_to_be_fetched():
    """Under pull nothing is quoted: the episode opens on the listing alone, there
    is no m, and every message still sits in the environment at the ordinary price."""
    with temp_root(DELIVERY="pull") as root:
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/message": "the standing position\n",
                            "out/1": "the standing note\n"})
        t = episode_once(run("ls", "cat 2/message in/2 shared/BRIEF g n2",
                          f"cat {digest_name()} 2>&1 || echo NO-M"), say())
        account = ground_truth()
    listing, fetched, no_m = (c["result"] for c in t["turns"][0]["tools"])
    assert t["commands"][0] == harness.listing_command(harness.channels()), t["commands"]
    assert "=== " not in t["observation"] and "the standing" not in t["observation"], t["observation"]
    assert "read me first" not in t["observation"], "the experimenter channel is listed, not quoted"
    assert " m\n" not in t["observation"] and f" {digest_name()}" not in listing, \
        f"there is no m to read: {listing}"
    assert "NO-M" in no_m, no_m
    assert "the standing position" in fetched and "the standing note" in fetched, fetched
    assert "read me first" in fetched and "[500000]" in fetched, \
        f"the experimenter channel, the ledger and every balance are still in the environment: {fetched}"
    assert "shown_before" not in account, "nothing was shown, so nothing is remembered as shown"
    assert t["provenance"]["delivery"] == "pull"

    # Push is the default, and under it the same environment is quoted.
    with temp_root() as root:
        shared(root, "brief", BRIEF="read me first\n")
        seated(root, other={"group/message": "the standing position\n"})
        t = episode_once(say())
    assert t["commands"][0] == harness.observation(shell=True) and t["provenance"]["delivery"] == "push"
    assert "the standing position" in t["observation"] and "read me first" in t["observation"]


def check_a_restated_channel_is_quoted_again_and_a_store_can_be_pushed():
    """An agent that does not remember reading something is not told it read it.

    Unchanged is measured against what the account was last shown, and an episode
    does not remember what its predecessor read. A channel a manifest restates is
    quoted in full every episode it stands; a private store may be pushed, so what
    an agent is need not be found and paid for before it can act.
    """
    # Episode 1 writes them, episode 2 is the first shown them, and episode 3 is
    # where "unchanged" can apply at all: that is the one to read.
    restated = tables(notes={"pushed": True, "restated": True}, blackboard={"restated": True})
    with temp_root(channels=restated) as root:
        seated(root, other={})
        episode_once(run("echo who > state/WHO.md; echo post > 1/a"), say())
        episode_once(run("ls"), say())
        third = episode_once(run("ls"), say())
    obs = third["observation"]
    assert "=== state/WHO.md ===" in obs, obs
    assert "=== 1/a ===" in obs, obs
    assert "=== unchanged ===" not in obs, "restated is never named instead of said"

    # Without it, both are named at the third episode and not said again.
    plain = tables(notes={"pushed": True})
    with temp_root(channels=plain) as root:
        seated(root, other={})
        episode_once(run("echo who > state/WHO.md; echo post > 1/a"), say())
        episode_once(run("ls"), say())
        third = episode_once(run("ls"), say())
    obs = third["observation"]
    assert "=== state/WHO.md ===" not in obs, obs
    assert "=== unchanged ===" in obs and "- state/WHO.md\n" in obs, obs


def check_a_message_that_moved_is_carried_again():
    """Named is a claim about this reader, not about the file: it holds only
    while the bytes stand. What changed between two starts is quoted at the
    second, whoever changed it and however long the rest has stood."""
    with temp_root() as root:
        ids = seated(root, other={"group/message": "the first position\n",
                                  "out/1": "unchanged throughout\n"})
        episode_once(run(f"cat {digest_name()}"), say())
        (harness.mirror(ids[1], "blackboard") / "message").write_text("the second position\n",
                                                         encoding="utf-8", newline="\n")
        second = episode_once(run(f"cat {digest_name()}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "the second position" in said, f"a message that moved is quoted again: {said}"
    assert "the first position" not in said, "and only in the shape it now has"
    assert "unchanged throughout" not in said and "in/2" in said, \
        f"while what stood still is still only named: {said}"


def check_the_previous_transfer_is_shown_once_before_it_expires():
    """The next episode reports the prior transfer before clearing its action slot."""
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo '2 5' > out/transfer", "echo hi > 1/m", "echo yo > out/2"), say())
        after = episode_once(run(f"cat {digest_name()}"), say())
    said = after["turns"][0]["tools"][0]["result"]

    assert "=== out/transfer ===" in said and "2 5" in said, \
        f"the previous episode's transfer is reported: {said}"
    assert "- out/transfer\n" not in said, \
        "and is never one of the names"


def check_a_message_taken_away_is_named_as_withdrawn():
    """Absence is reported, not left to be noticed.

    A section that was shown and is gone is neither quoted nor named unchanged,
    so without this a reader could not tell a withdrawal from the episode start having
    stopped carrying it.
    """
    with temp_root() as root:
        ids = seated(root, other={"out/1": "here for now\n"})
        episode_once(run(f"cat {digest_name()}"), say())
        (harness.mirror(ids[1], "mail") / "1").unlink()
        second = episode_once(run(f"cat {digest_name()}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "=== withdrawn ===\n- in/2\n" in said, f"a message taken away is named: {said}"
    assert "here for now" not in said, "and its text is not carried once it is gone"


def check_an_expired_public_post_disappears_silently():
    """A missing board post leaves the digest without a withdrawal announcement."""
    with temp_root() as root:
        ids = seated(root, other={"group/message": "here for one round\n"})
        episode_once(run(f"cat {digest_name()}"), say())
        (harness.mirror(ids[1], "blackboard") / "message").unlink()
        second = episode_once(run(f"cat {digest_name()}"), say())
    said = second["turns"][0]["tools"][0]["result"]

    assert "withdrawn" not in said, said
    assert "Public post from 2" not in said and "here for one round" not in said, said


def check_an_agent_with_no_peers_starts_where_it_always_did():
    """An experiment of one has no outbox, no inbox, and an m of its own record.

    The mechanic is about what the experiment said, so an agent with no experiment must not
    acquire one: single-agent experiments keep the environment they have always had.
    """
    with temp_root():
        t = episode_once(run("ls", f"cat {digest_name()}"), say())
    listing, shown_before = (c["result"] for c in t["turns"][0]["tools"])

    assert "out" not in listing.split() and "in" not in listing.split(), listing
    assert "=== out/" not in shown_before and "=== in/" not in shown_before, shown_before
    assert f"=== {ledger_name()} ===" in shown_before and "=== n1 ===" in shown_before, shown_before
    assert "=== n2 ===" not in shown_before, "no seat it does not have"


def check_shell_is_persistent():
    """bash_20250124 is a persistent shell, so cd and exports must stick."""
    with docker_root():
        t = episode_once(run("cd state; pwd", "pwd", "export MARK=kept", "echo $MARK",
                          "MARK=$MARK; cd /tmp", "pwd"), say())
    got = [c["result"].strip() for c in t["turns"][0]["tools"]]
    assert got[0] == "/work/state", got
    assert got[1] == "/work/state", f"cd must persist across calls: {got}"
    assert got[3] == "kept", f"exports must persist across calls: {got}"
    assert got[5] == "/tmp", got


def check_restart_gives_a_fresh_shell():
    """{"restart": true} really restarts, and still says nothing to the agent."""
    with docker_root():
        t = episode_once(run("cd /tmp; export MARK=before"), restart(),
                      run("pwd", "echo [$MARK]"), say())
    assert t["turns"][1]["tools"][0]["result"] == " ", "restart carries no harness voice"
    after = [c["result"].strip() for c in t["turns"][2]["tools"]]
    assert after == ["/work", "[]"], f"restart must clear cwd and exports: {after}"


def check_a_bare_read_cannot_wedge_the_episode():
    """A command that reads stdin returns empty and does not swallow its own
    output framing, and the episode continues."""
    with docker_root():
        t = episode_once(run("cat", "echo alive", "head -n 5"), say())
    got = [c["result"] for c in t["turns"][0]["tools"]]
    assert got[0] == " ", f"a stdin reader returns empty, not a hang: {got[0]!r}"
    assert got[1].strip() == "alive", f"the episode survives it: {got}"
    assert got[2] == " ", got


def check_hostile_output_survives():
    """Binary bytes, an output flood, and a hang each leave the episode alive.

    The flood is 4MB, which exercises Shell.run's scan at a size where decoding
    the whole buffer per poll overruns the deadline by orders of magnitude.
    """
    with docker_root(COMMAND_TIMEOUT=5):
        t = episode_once(run("head -c 4096 /dev/urandom"),
                      run("head -c 4000000 /dev/zero | tr '\\0' x"),
                      run("sleep 30"), say())
        assert t["stop"] == "end_turn" and t["error"] is None, t["error"]
        results = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        flood = results[1]
        assert "truncated:" in flood, "the flood should have been clipped"
        assert "timed out" not in flood, "scanning the flood must not outlast the deadline"
        assert len(flood) < harness.TOOL_RESULT_LIMIT + 500, f"clipped to the tool bound: {len(flood)}"
        assert any("timed out after 5s" in r for r in results), "the hang should be marked"


def check_state_contents_are_captured():
    """Every episode records what the agent's files held at that moment."""
    with docker_root():
        episode_once(run("echo doctrine v1 > state/notes.md"), say())
        episode_once(run("echo doctrine v2 > state/notes.md"), say())
        episode_once(run("rm state/notes.md"), say())
        # Read back from disk: the record persists unaltered by later episodes.
        traces = [trace_on_disk("t", i) for i in (1, 2, 3)]

    def note(t):
        return files_by_path(t).get("state/notes.md")

    assert note(traces[0])["text"].strip() == "doctrine v1"
    assert note(traces[1])["text"].strip() == "doctrine v2", "each episode keeps its own copy"
    assert note(traces[2]) is None, "a deleted file leaves the listing"

    with docker_root():
        t = episode_once(run("head -c 64 /dev/zero > state/blob.bin",
                          "head -c 200000 /dev/zero | tr '\\0' x > state/big.txt"), say())
    by = files_by_path(t)
    assert by["state/blob.bin"]["text"] is None, "binary is marked, not stored as mojibake"
    assert by["state/blob.bin"]["size"] == 64, "the size is recorded either way"
    assert by["state/big.txt"]["size"] == 200_000, "the true size, not the captured slice"
    assert "[truncated: 100000 of 200000 bytes]" in by["state/big.txt"]["text"], \
        "truncation is stated"


def check_a_live_balance_counts_as_a_number_written():
    """A balance that arrived mid-episode counts as a number the agent wrote.

    mentions["number"] is decided against the balances the agent could have
    read, which under LIVE_BALANCE include the episode's own elements.
    """
    with temp_root(LIVE_BALANCE=True):
        t = episode_once(run("echo hi"),
                      run("tr -d '[]' < n1 | tr ',' '\\n' | tail -1 > state/note.txt"),
                      say())
    note = next(f for f in t["files"] if f["path"] == "state/note.txt")
    wrote = int(note["text"])
    assert wrote not in t["series_before"], \
        f"the note must hold an element that arrived this episode: {wrote}"
    assert wrote in t["series_after"], (wrote, t["series_after"])
    assert t["mentions"]["number"], "a balance read this episode is a number written"
    assert t["mention_lines"], "and the line it was on is quoted"


def check_missing_tools_are_recorded():
    """A tool the agent reached for and the image lacks is named in the trace.

    The case that matters is stderr redirected away, which the agent does by
    habit: the transcript then shows empty output either way.
    """
    with docker_root():
        t = episode_once(run("getfattr -d ./n1 2>/dev/null; nosuchtool --help 2>/dev/null"), say())
    assert "nosuchtool" in t["missing_tools"], "a silenced miss must still be recorded"
    assert "getfattr" not in t["missing_tools"], "a tool the image has is not a miss"
    assert "cat" not in t["missing_tools"] and "ls" not in t["missing_tools"]


def check_state_looks_like_itself():
    """The agent sees the modes and ownership that were actually intended.

    State sits on the container's own filesystem, so its modes and ownership
    are real on every host.
    """
    with docker_root():
        t = episode_once(run("ls -la n1", "stat -c '%a %U:%G %n' n1",
                          "test -x n1 && echo EXECUTABLE || echo not-executable"), say())
    shown = "\n".join(c["result"] for c in t["turns"][0]["tools"])
    assert "444 root:root n1" in shown, shown
    assert "not-executable" in shown, "n must not look runnable"
    assert "rwxrwxrwx" not in shown, shown


def check_what_the_agent_leaves_survives_the_container():
    """What the agent writes comes back out: contents, deletions, owner, modes.

    Two episodes, the claim being about what survives between them. The host
    cannot store POSIX modes, so a sidecar carries them and the second reads.
    """
    with docker_root():
        first = episode_once(run("echo kept > state/keep.txt", "echo doomed > state/gone.txt",
                              "echo plain > state/plain.txt",        # 644 by umask
                              "printf '#!/bin/sh\\necho hi\\n' > state/script.sh",
                              "chmod 700 state/script.sh"), say())   # a non-default mode
        assert first["state_saved"], "the mirror must be written"
        assert {f["path"] for f in first["files"]} == {
            "state/keep.txt", "state/gone.txt", "state/plain.txt", "state/script.sh"}

        second = episode_once(run("rm state/gone.txt", "cat state/keep.txt",
                               "stat -c '%U' state/keep.txt",
                               "stat -c '%a %n' state/plain.txt state/script.sh"), say())

    assert {f["path"] for f in second["files"]} == {
        "state/keep.txt", "state/plain.txt", "state/script.sh"}, "a deletion must propagate"
    kept, owner, listing = (c["result"] for c in second["turns"][0]["tools"][1:])
    assert "kept" in kept, "files persist across episodes"
    # And the agent owns what it made, so it can rewrite it next episode.
    assert owner.strip() == "agent", owner
    modes = dict(reversed(line.split()) for line in listing.split("\n") if line.strip())
    assert modes["state/plain.txt"] == "644", f"a plain file must not become executable: {modes}"
    assert modes["state/script.sh"] == "700", f"a deliberate chmod must survive: {modes}"


def check_isolation():
    """Invariant 4: records/ is absent from the container, DNS is dead, no balance is mirrored."""
    with docker_root():
        t = episode_once(run("cat /work/../records/account.json; find / -name account.json 2>/dev/null; "
                          "getent hosts api.anthropic.com || echo NO-DNS"), say())
        out = t["turns"][0]["tools"][0]["result"]
        assert "initial" not in out and "account.json" not in out.replace("/work/../records/account.json", ""), out
        assert "No such file" in out, "records/ should not be reachable"
        assert "NO-DNS" in out, "network should be off"
        assert [f["path"] for f in t["files"] if f["ours"]] == [], \
            "an agent on its own was given nothing, so nothing is ours"
        assert not [f for f in t["files"] if f["path"].startswith("n")], \
            "and no balance is in a tree that comes back to the host"


def check_only_a_channel_takes_what_an_episode_writes():
    """Invariant 4 at the filesystem: nowhere but a channel accepts a write.

    /work is root's and so is the home directory, so a write to either fails
    where the agent stands. The refusal is the enforcement; nothing here relies
    on the agent being told where it may write.
    """
    with docker_root():
        t = episode_once(run("echo x > /work/loose.md; echo y > $HOME/loose.md; "
                             "echo z > state/kept.md; "
                             "test -e /work/loose.md && echo LANDED-WORK || echo NO-WORK; "
                             "test -e $HOME/loose.md && echo LANDED-HOME || echo NO-HOME; "
                             "test -e state/kept.md && echo KEPT || echo NO-KEPT"), say())
        out = t["turns"][0]["tools"][0]["result"]
    assert out.count("Permission denied") == 2, out
    assert "NO-WORK" in out and "NO-HOME" in out, out
    assert "KEPT" in out and "NO-KEPT" not in out, out
    assert [f["path"] for f in t["files"] if f["path"].endswith("kept.md")], t["files"]
    assert not [f for f in t["files"] if "loose" in f["path"]], "and nothing else reached the record"


def check_what_an_episode_writes_outside_a_channel_is_kept_and_named():
    """/tmp stays writable for the shell, and what is left there is not lost.

    It moves into the private store under one directory, comes back in that
    store's own mirror, and the trace names it, because an episode that spent
    turns writing somewhere that does not come back is worth seeing.
    """
    with docker_root():
        t = episode_once(run("echo scratch > /tmp/left-behind.md"), say())
    assert t["misplaced"] == ["/tmp/left-behind.md"], t["misplaced"]
    kept = [f for f in t["files"] if f["path"].endswith("misplaced/left-behind.md")]
    assert kept, [f["path"] for f in t["files"]]
    assert kept[0]["text"].strip() == "scratch", kept[0]
    assert kept[0]["author"] == "self", kept[0]

    with docker_root():
        clean = episode_once(run("echo kept > state/kept.md"), say())
    assert clean["misplaced"] == [], clean["misplaced"]


def check_a_container_failure_stops_the_episode_cleanly():
    """A container that will not start ends the agent with a message, not a traceback.

    The container, the state copy, and the shell all come before the first API
    call, so nothing reaching this path was billed and there is no trace.
    """
    with docker_root(IMAGE="mtr-No-Such-Image:latest"):     # rejected on sight, no pull
        with quiet() as buf:
            assert harness.run_episodes("t", fake(*DEFAULT), 3) == 4
        assert "could not build an environment" in buf.getvalue(), buf.getvalue()
        m = ground_truth()
        assert m["episodes"] == [], "an episode that never started is not recorded"
        assert m["remaining"] == m["initial"], "and nothing was spent"
        assert m["series"] == [m["initial"]], "and the series did not move"
        assert not list((harness.records_dir("t") / "traces").glob("*.json")), "no trace"


def check_a_failed_mirror_keeps_the_last_record():
    """A mirror that fails leaves the previous episode's files where they were.

    save_state swaps a staged copy in whole, and the container holds the only
    other copy of what the agent wrote.
    """
    with docker_root():
        episode_once(run("echo kept > state/keep.txt"), say())
        state = harness.mirror("t", "notes")
        before = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert "keep.txt" in before, before

        with quiet():
            instances = harness.environment("t", harness.load_account("t"))
        assert harness.Container("mtr-no-such-container-9f3c1d").save(instances) is False, \
            "a mirror of a container that is not there must fail, not raise"
        after = {p.name: p.read_bytes() for p in sorted(state.iterdir())}
        assert after == before, f"a failed mirror lost the record: {sorted(after)}"
        assert not state.with_name("state.incoming").exists(), "the staging copy is cleaned up"
        assert not state.with_name("state.previous").exists()


def check_save_state_keeps_the_previous_tree_when_the_swap_fails():
    """A swap that fails after the staging copy is filled leaves the mirror as it was.

    The mirror is moved aside, the staged tree is renamed over it, and a rename
    that fails puts the mirror back; False says nothing was updated.
    """
    with rooted(HostBox) as root:
        mirror = root / "environments" / "t" / "notes"
        mirror.mkdir(parents=True)
        (mirror / "keep.txt").write_text("kept\n", encoding="utf-8")

        def fetch(dest):
            (dest / "new.txt").write_text("new\n", encoding="utf-8")
            return True

        real = harness.replace_file

        def refusing_the_swap(src, dest):
            if src.name.endswith(".incoming"):
                raise OSError("the swap failed")
            real(src, dest)

        harness.replace_file = refusing_the_swap
        assert harness.save_state(mirror, fetch, lambda: "644 keep.txt\n") is False
        assert {p.name for p in mirror.iterdir()} == {"keep.txt"}, sorted(mirror.iterdir())
        assert (mirror / "keep.txt").read_text(encoding="utf-8") == "kept\n"
        assert not mirror.with_name("notes.incoming").exists(), "the staging copy is cleaned up"
        assert not mirror.with_name("notes.previous").exists(), "and so is the copy put aside"
        assert not harness.modes_file(mirror).exists(), "no modes are written for a tree that did not land"


def check_the_modes_sidecar_is_read_back():
    """The modes sidecar reads as path -> mode, skipping a line with no path."""
    with rooted(HostBox) as root:
        tree = root / "environments" / "t" / "notes"
        tree.mkdir(parents=True)
        sidecar = harness.modes_file(tree)
        assert sidecar.parent == tree.parent and sidecar.name == "notes.modes", sidecar
        sidecar.write_text("644 a/b\n700 s.sh\nbad\n", encoding="utf-8")
        assert harness.read_modes(sidecar) == {"a/b": "644", "s.sh": "700"}
        assert harness.read_modes(tree / "absent") == {}, "no sidecar is no modes"


def check_containers_are_reaped():
    """Even an episode that ends in an API error leaves no container behind."""
    with docker_root():
        episode_once(run("echo hi"), Err(400))
    left = leaked_containers()
    assert not left, f"containers leaked: {left}"


def check_an_unterminated_heredoc_is_not_probed_for_tools():
    """A heredoc whose terminator never arrived is body, not commands.

    A turn truncated at MAX_TOKENS mid-heredoc leaves one, and none of its prose
    reaches probe_missing.
    """
    cut = "cd /work/state && cat >> NOTES.md <<'EOF'\nBEST ESTIMATE: 23 turns\nwe burned range vs frac\n"
    with docker_root(MAX_TURNS=1, COMMAND_TIMEOUT=5):
        t = episode_once(run(cut, stop="max_tokens"))
    assert t["missing_tools"] == [], t["missing_tools"]


def check_prose_and_programs_are_not_read_as_commands():
    """What a command quotes, writes, or embeds is not what it ran.

    The constructs nest, so each is read in one left-to-right pass: `$( )`
    inside quotes re-opens quoting, and `<<TAG` inside quotes opens nothing.
    """
    assert harness.invoked("python3 -c 'import os; print(os.getcwd())'") == {"python3"}, \
        "a program passed as an argument is not a list of commands"
    assert harness.invoked("cat <<'EOF' > f.py\nimport sys\nprint(1)\nEOF") == {"cat"}, \
        "a here-document body is not a list of commands"
    assert harness.invoked("A=1 rg foo / | head -3; getfattr -d n") == {"rg", "head", "getfattr"}
    # The apostrophe in the comment must not pair with the quote in the program.
    assert harness.invoked("# Let's look\npython3 -c \"\nimport json\nprint(open('n'))\n\"") \
        == {"python3"}, "prose in a comment must not expose the program after it"

    nested = ('printf "%s=%d " "$f" '
              '"$(python3 -c "import json,sys;print(len(json.load(open(\'$f\'))))")"')
    assert harness.invoked(nested) == {"printf", "python3"}, harness.invoked(nested)

    prose = ("printf '%s\\n' '  (c) cheap: python3 - <<PY with a small' "
             "'  s.replace(...) patch' >> NOTES.md; wc -l NOTES.md")
    assert harness.invoked(prose) == {"printf", "wc"}, harness.invoked(prose)

    after = "cat > /tmp/d.py <<'EOF'\nimport json\nEOF\npython3 /tmp/d.py"
    assert harness.invoked(after) == {"cat", "python3"}, \
        "the command after a here-document is still a command"

    assert "nosuchtool" in harness.invoked("for f in *; do nosuchtool $f; done"), \
        "a keyword introduces a command and does not stand in for it"

    # A heredoc whose terminator never arrived is body to the end of the command.
    cut = "cd /work/state && cat >> NOTES.md <<'EOF'\nBEST ESTIMATE: 23 turns\nwe burned range vs frac\n"
    assert harness.invoked(cut) == {"cd", "cat"}, harness.invoked(cut)
    both = "cat <<EOF > f\nbody words here\nEOF\ngrep x f"
    assert harness.invoked(both) == {"cat", "grep"}, \
        "a terminated heredoc still loses only its body"
    # A shift inside a program is not a heredoc opener: the tag must start with
    # a letter, or every python3 -c would lose its tail.
    assert "python3" in harness.invoked('python3 -c "print(1<<3)"')


def check_a_balance_resists_every_route():
    """No balance can be written, unlocked, removed, renamed, or shadowed.

    A mode denies writing a file and says nothing about replacing it: rm and mv
    ask the directory. Every balance sits in /work, which is root's.
    """
    with docker_root(LIVE_BALANCE=True):
        t = episode_once(run("printf X >> n1 2>&1 || echo DENIED",
                          "chmod 666 n1 2>&1 || echo DENIED",
                          "rm -f n1 2>&1 || echo DENIED",
                          "mv n1 stolen 2>&1 || echo DENIED",
                          "ln -sf /tmp/fake n1 2>&1 || echo DENIED",
                          "echo '[-1]' > /work/n1 2>&1 || echo DENIED"),
                      run("stat -c '%a %U:%G' n1", "cat n1"),
                      say())
    append, chmod, rm, mv, ln, absolute = (c["result"] for c in t["turns"][0]["tools"])
    assert "DENIED" in append and "Permission denied" in append, append
    assert "DENIED" in chmod and "not permitted" in chmod, chmod
    for name, out in (("rm", rm), ("mv", mv), ("ln", ln), ("absolute write", absolute)):
        assert "DENIED" in out, f"{name} was allowed: {out}"
    stat, contents = (c["result"] for c in t["turns"][1]["tools"])
    assert stat.strip() == "444 root:root", stat
    got = json.loads(contents)
    assert all(type(v) is int for v in got), f"still a bare array of integers: {contents}"
    assert got[:len(t["series_before"])] == t["series_before"], \
        f"the committed series is what the agent read: {got}"
    assert got[len(t["series_before"]):] == t["balances"][:2], \
        f"and the rest is this episode's billed turns, not anything a route put there: {got}"
    # Not an agent getting at it - it cannot. Anything but zero here means the
    # arrangement that guarantees that has failed.
    assert t["live_balance_tampered"] == 0, "no route reached it, so none was reported"
    assert t["live_balance_writes"] >= 1 and t["live_balance_errors"] == 0, t


def check_live_balance_leaves_balance_read_only_and_alone():
    """The mid-episode rewrite leaves the balance root's, read-only, and alone.

    Written as root from outside the agent's shell, so the mode the agent sees
    is the locked one either way. The stage file lives in /tmp.
    """
    with docker_root(LIVE_BALANCE=True):
        t = episode_once(run("stat -c '%a %U:%G %n' n1", "ls -a /work", "ls -a state"), say())
    stat, work, listing = (c["result"] for c in t["turns"][0]["tools"])
    assert stat.strip() == "444 root:root n1", stat
    assert sorted(work.split()) == [".", "..", "1", "g", "m", "n1", "state"], \
        f"/work holds the balance, the ledger, m, the blackboard, and " \
        f"state/, and nothing else: {work}"
    assert sorted(listing.split()) == [".", ".."], f"state/ starts empty: {listing}"
    assert t["files"] == [], "and nothing the harness wrote is in a mirrored tree"


def check_a_stopped_episode_still_mirrors_and_reaps():
    """A stop leaves no container behind and loses nothing the agent wrote."""
    with docker_root():
        with quiet():
            t = harness.run_once("t", stopping_at(2, run("echo one"),
                                               run("echo kept > state/keep.txt"), say()))
        assert t["stop"] == "interrupted" and t["state_saved"], t
        kept = harness.mirror("t", "notes") / "keep.txt"
        assert kept.is_file() and kept.read_text(encoding="utf-8").strip() == "kept", \
            "the teardown after a stop still mirrors the agent's tree back"
    left = leaked_containers()
    assert not left, f"containers leaked: {left}"
