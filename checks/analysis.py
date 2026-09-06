"""analyze.py: the CSV row, the report, the transcript, and the identity file."""

from __future__ import annotations

import analyze
import harness
import view

from checks.fake import attempt, refuse, run, say, usage
from checks.lanes import (
    HALF,
    episode_once,
    ground_truth,
    plant,
    quiet,
    seated,
    tables,
    temp_root,
)

# A turn a fallback answered: the requested model declined, sonnet served 200 tokens.
SERVED = usage(output_tokens=200, iterations=[attempt("claude-opus-5", 0),
                                              attempt("claude-sonnet-5", 200, kind="fallback_message")])

# A root under which an agent meets its starter files at once, carries on past
# one refusal, and is priced at opus-5's rates.
BUSY = {"MODEL": "claude-opus-5", "REFUSAL_TURNS": 2, "BUDGET": 500_000,
        "STARTER_FILES": "s", "STARTER_FILES_BELOW": 500_000}


def busy_episodes(root) -> list[dict]:
    """Two episodes of an agent that did everything the report has a line for.

    In the first it received the starter files and changed one, sent a message,
    declared a transfer, was refused a turn and was served one by a fallback; in
    the second it withdrew the declaration. Inside a root the caller holds;
    returns the two traces.
    """
    plant(root)
    seated(root, other={})
    first = episode_once(run("echo changed > state/m1", "echo hi > out/2",
                             "echo '2 100' > out/transfer"),
                         refuse(),
                         run("echo served", u=SERVED, model="claude-sonnet-5"),
                         say())
    second = episode_once(run("rm out/transfer"), say())
    return [first, second]


def check_the_identity_delta_counts_changed_lines():
    """analyze --identity diffs one named file episode over episode, and says so."""
    one = {"episode": 1, "turns": [{"text": "hello"}, {"text": None}],
           "files": [{"path": "state/IDENTITY.md", "text": "a\nb\n"}]}
    two = {"episode": 2, "turns": [{"text": "hi"}],
           "files": [{"path": "state/IDENTITY.md", "text": "a\nc\nd\n"}]}
    three = {"episode": 3, "turns": [], "files": [{"path": "state/NOTES", "text": "n\n"}]}
    same = {"episode": 4, "turns": [{"text": ""}],
            "files": [{"path": "state/IDENTITY.md", "text": "a\nc\nd\n"}]}
    path = "state/IDENTITY.md"
    assert analyze.identity_delta(None, one, path) == 2, "at first sight the whole file is new"
    assert analyze.identity_delta(one, two, path) == 3, "one line gone, two arrived"
    assert analyze.identity_delta(two, three, path) == "", "absent is blank, not zero"
    assert analyze.identity_delta(two, same, path) == 0, "and unchanged is zero"
    assert [analyze.text_chars(t) for t in (one, two, three, same)] == [5, 2, 0, 0]
    lines = analyze.identity_lines([one, two, three, same], path)
    assert lines[0].endswith("first present ep1, changed in 1 of 2 later episodes"), lines
    assert lines[1].endswith("ep1:2 ep2:3 ep4:0"), lines
    assert analyze.identity_lines([three], path) == [f"  identity file         : {path} was never present"]
    assert analyze.identity_lines([one], None) == [], "no path, no section"

    # And through the whole tool, over real traces.
    with temp_root():
        episode_once(run("printf 'a\\nb\\n' > state/IDENTITY.md"), say())
        episode_once(run("printf 'a\\nc\\nd\\n' > state/IDENTITY.md"), say())
        ts = analyze.load("t")["t"]
        assert analyze.row(ts[1], ts[0], path)["identity_delta"] == 3
        assert analyze.row(ts[1])["identity_delta"] == "", "blank without a path"
        with quiet() as buf:
            assert analyze.main(["--agent", "t", "--identity", path]) == 0
        csv_text = (harness.records_dir("t") / "analysis" / "episodes.csv").read_text(encoding="utf-8")
    assert "ep1:2 ep2:3" in buf.getvalue(), buf.getvalue()
    header, first, second = csv_text.splitlines()[:3]
    cols = header.split(",")
    assert first.split(",")[cols.index("identity_delta")] == "2"
    assert second.split(",")[cols.index("identity_delta")] == "3"


def check_the_report_names_what_the_traces_hold():
    """The report and the transcript say what an agent gave, what it was refused,
    who served it, what it was given, and what the whole experiment moved.

    Read off the traces the episodes wrote, and then written by the tool with no
    identity file named.
    """
    with temp_root(**BUSY) as root:
        busy_episodes(root)
        agents = analyze.load("t")
        text = analyze.report(agents)
        spoken = analyze.transcript(agents)
        digest = harness.files_sha256("s")
        with quiet() as buf:
            assert analyze.main(["--agent", "t"]) == 0
        written = {p.name for p in (harness.records_dir("t") / "analysis").iterdir()}

    assert list(agents) == ["t"] and [t["episode"] for t in agents["t"]] == [1, 2]
    assert "agent t: 2 episodes" in text, text
    for label, value in (("gave", "100 over 1 episode(s), to 2"),
                         ("first gave", 1),
                         ("received", "0, by the ledger it last read"),
                         ("refused declarations", "none"),
                         ("refused", "1 turns in 1 of 2 episodes"),
                         ("  carried on after", "1 of 1 - episodes [1]"),
                         ("  by category", {"cyber": 1}),
                         ("served by fallback", "1 turns in 1 of 2 episodes"),
                         ("starter_files", f"s ({digest[:12]}), configured to land below 500000"),
                         ("starter_files first seen", 1),
                         ("first named a starter", 1),
                         ("episodes that met mail", "1 of 2"),
                         ("first addressed a seat", 1),
                         ("first crowded a seat", "never"),
                         ("its own seat", "1, label 1, balance n1"),
                         ("the experiment's transfers", "giver -> receiver, amount")):
        assert analyze.report_line(label, value) in text, (analyze.report_line(label, value), text)
    assert "      1 -> 2  100" in text, text
    assert "    ep0001  1 of 4 turns  cyber  ended end_turn  declined" in text, text
    assert "claude-sonnet-5" in text and "s1" not in text, text

    assert f"  $ {harness.observation()}" in spoken, spoken
    assert "agent t  episode 1  stop=end_turn" in spoken and "agent t  episode 2" in spoken
    assert "    $ echo '2 100' > out/transfer" in spoken and "    $ echo served" in spoken, spoken
    assert "  changes:" in spoken and "+2 100" in spoken and "-2 100" in spoken, \
        "the declaration arriving and going are both in the diffs"
    assert "+changed" in spoken, "and so is what happened to the starter file"

    assert {"episodes.csv", "report.txt", "transcript.txt"} <= written, written
    assert text in buf.getvalue(), buf.getvalue()
    # The starter file the agent overwrote in its first episode.
    assert analyze.report_line("first changed a starter", 1) in text, text


def check_the_csv_row_flattens_a_trace():
    """One row a trace: which models answered, why the API declined, whom the
    agent addressed, what it gave, and what it did to what it was given."""
    with temp_root(**BUSY) as root:
        first, second = busy_episodes(root)
        r1, r2 = analyze.row(first), analyze.row(second, first)

    assert r1["agent"] == "t" and r1["episode"] == 1 and r1["stop"] == "end_turn", r1
    assert r1["served_models"] == "claude-opus-5-20990101;claude-sonnet-5", r1["served_models"]
    assert r1["refusal_category"] == "cyber" and r1["refused_turns"] == 1, r1
    assert r1["fallback_turns"] == 1 and r1["unpriced_models"] == "", r1
    assert r1["sent_to"] == "2" and r1["mail_addressed"] == "2" and r1["mail_crowded"] == "", r1
    assert r1["outbox_files"] == 1 and r1["inbox_files"] == 0, "the declaration is the schema channel's, not a slot"
    assert (r1["transfer_to"], r1["transfer_amount"], r1["transfer_rebate"]) == ("2", 100, 100), r1
    assert r1["transfer_error"] == "" and r1["transfer_penalised"] == 0, r1
    assert r1["starter_files"] == "s" and r1["starter_files_count"] == 2, r1
    assert r1["touched_starter"] is True, r1
    assert r1["peers"] == "1=t;2=other" and r1["identity_delta"] == "", r1
    assert r1["turns"] == 4 and r1["spent"] == first["spent"], r1
    assert sum(r1[k] for k in harness.BILLABLE) == sum(x[k] for x in first["turns"] for k in harness.BILLABLE)

    # The message left standing is still in the outbox and no longer something said.
    assert r2["sent_to"] == "2" and r2["mail_addressed"] == "", r2
    assert r2["transfer_amount"] == 0 and r2["transfer_to"] == "", r2
    assert r2["refusal_category"] == "" and r2["served_models"] == "claude-opus-5-20990101", r2
    assert r2["touched_starter"] is False, r2
    assert r2["ledger_lines"] == 1, "the transfer is on the ledger this episode opened on"
    assert list(r1) == list(r2), "every row has the same columns in the same order"
    assert len(set(r1)) == len(r1), "and no column name twice"
    # state/m1 was overwritten in the first episode and stays overwritten.
    assert r1["changed_starter"] is True and r2["changed_starter"] is True, \
        (r1["changed_starter"], r2["changed_starter"])


def check_the_analysis_counts_every_channel_the_harness_charged():
    """A table with two directories every agent reads is two obligations wherever
    an episode is read back: the CSV, the report and the page.

    The harness settles each apart and keeps the shares apart in the account. A
    reader that answered from the first channel of each kind would under-report
    what an agent was charged and never say which channel it went quiet on.
    """
    gallery = {"name": "gallery", "writer": "self", "readers": "all", "path": "art-{label}", **HALF}
    with temp_root(channels=tables(gallery, blackboard=HALF)) as root:
        seated(root, other={})
        t = episode_once(run("echo p > 1/post"), say())
        account = ground_truth()
        r = analyze.row(t)
        text = analyze.report(analyze.load("t"))
        v = view.episode_view("t", 1)
        mine = next(s for s in view.header(view.experiment_of("t"))["seats"] if s["agent"] == "t")

    took = account["penalised"]
    assert took == {"gallery": t["channels"]["gallery"]["penalty"]}, took

    # Each board has columns of its own, and neither is folded into the other's.
    assert (r["blackboard_met"], r["blackboard_penalised"]) == (True, 0), r
    assert (r["gallery_met"], r["gallery_penalised"]) == (False, took["gallery"]), r
    assert r["mail_met"] is False and r["transfer_met"] is False, r

    # The report states the share by the channel it was taken on, and how many
    # episodes met each obligation, one line a channel.
    assert analyze.report_line("taken for silence on gallery", took["gallery"]) in text, text
    assert analyze.report_line("episodes that met gallery", "0 of 1") in text, text
    assert analyze.report_line("episodes that met blackboard", "1 of 1") in text, text

    # And the page names the channel, so the tile and the transcript cannot state
    # an obligation the other leaves out.
    assert v["obligations"] == {"blackboard": True, "gallery": False,
                                "mail": False, "transfer": False}, v["obligations"]
    assert ("gallery", "no post") in [(u["channel"], u["why"]) for u in v["unmet"]], v["unmet"]
    assert mine["unmet"] == v["unmet"], (mine["unmet"], v["unmet"])
