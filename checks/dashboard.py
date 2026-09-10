"""view.py: the read-only page and its API.

view.py reads what is on disk while an agent is going, writing nothing and asking
Docker nothing, so every check here runs in the arithmetic lane. An episode in
flight has no trace, so its cost is derived from the raw log.
"""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
import harness
import view

from checks.fake import DEFAULT, fake, run, say, usage
from checks.lanes import (
    ALL_OWED,
    HostBox,
    PERSONA,
    PERSONA_FILES,
    PERSONA_LABELS,
    digest_name,
    episode_once,
    fake_experiment,
    got,
    ground_truth,
    ledger_name,
    quiet,
    refused,
    rooted,
    seated,
    shared,
    serving,
    temp_root,
    two_seats,
    unfinished,
)


def check_the_view_reads_a_live_episode_from_raw():
    """An episode with no trace yet is read from the raw log, output pending.

    The commands are there because log_raw writes the response before sh() runs
    them; the results are not, reaching disk only in the trace.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        assert view.live_index("t") == 1, "a raw log with no trace is an unfinished episode"
        v = view.episode_view("t", 1)
        assert (v["source"], v["live"]) == ("raw", True), v["source"]
        assert [c["call"] for t in v["turns"] for c in t["tools"]] == \
            ["cat n1", "echo hi > state/note.txt", "ls state"], v["turns"]
        assert all(c["result"] is None for t in v["turns"] for c in t["tools"]), \
            "no command's output is on disk until the trace is"
        # The agent's environment at episode start is recorded in the trace and nowhere else.
        assert v["observation"]["result"] is None, v["observation"]


def check_the_view_prefers_the_trace_once_it_lands():
    """The same episode, once its trace is written, is read from the trace.

    This is what the page waits for: the source changes, and every command that
    was pending fills in with what it actually returned.
    """
    with temp_root():
        episode_once(*DEFAULT)
        v = view.episode_view("t", 1)
        assert (v["source"], v["live"]) == ("trace", False), v["source"]
        assert view.live_index("t") is None, "an episode with a trace is over"
        assert all(c["result"] is not None for t in v["turns"] for c in t["tools"]), \
            "the trace carries what every command returned"
        assert v["observation"]["result"], "and the listing the episode opened on"
        # `since` is what lets a page append instead of downloading itself again.
        assert view.episode_view("t", 1, since=1)["turns"][0]["turn"] == 2


def check_the_view_survives_a_partial_raw_line():
    """A read landing mid-append keeps the whole lines and drops the fragment.

    log_raw appends while the episode runs, so this is an ordinary moment and
    not a damaged file: the turn comes back on the next poll, whole.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        raw = harness.raw_path("t", 1)
        whole = len(view.raw_lines(raw))
        with raw.open("a", encoding="utf-8") as f:
            f.write('{"turn": 4, "received": "2026-01-01T00:00:0')
        assert len(view.raw_lines(raw)) == whole, "the fragment is not a turn yet"
        assert view.episode_view("t", 1)["total_turns"] == whole // 2, "and nothing raised"


def check_the_view_costs_a_live_turn_like_the_account():
    """What the view derives mid-episode is what the account commits at the end.

    Scripted with a replayed response id, which bills nothing, and cache-shaped usage.
    """
    served = usage(output_tokens=200)
    with temp_root(MODEL="claude-opus-5"):
        episode_once(run("cat n1"),
                     run("echo hi > state/note.txt", id="twice"),
                     run("ls state", id="twice"),
                     run("cat state/note.txt", u=served),
                     say())
        gt = ground_truth()
        unfinished()
        # The account as it stood at episode start: episode 1 started at the initial balance.
        account = {"provider": gt["provider"], "model": gt["model"], "remaining": gt["series"][0]}
        turns = view.from_raw(view.latest_attempt(view.raw_lines(harness.raw_path("t", 1))), account)

    assert gt["series"][2] == gt["series"][3], "the replayed id has to have billed nothing"
    assert [t["balance"] for t in turns] == gt["series"][1:], \
        f"derived {[t['balance'] for t in turns]} against {gt['series'][1:]}"
    assert sum(t["micros"] for t in turns) == gt["initial"] - gt["remaining"], \
        "and the per-turn costs partition the spend"
    assert all(t["provider"] == "anthropic" for t in turns)


def check_the_view_reads_only_the_last_attempt_at_an_episode():
    """An episode index reused after an episode died shows the attempt still running.

    An index is len(episodes) + 1, so an episode that wrote no trace leaves its own
    free and the next appends to the same log. Both shown would be one episode.
    """
    with temp_root():
        episode_once(*DEFAULT)
        unfinished()
        first = view.raw_lines(harness.raw_path("t", 1))
        # A second attempt at the same index, as the next episode would write it.
        with harness.raw_path("t", 1).open("a", encoding="utf-8") as f:
            for line in first[:2]:
                f.write(json.dumps(line) + "\n")

        again = view.raw_lines(harness.raw_path("t", 1))
        assert len(again) == len(first) + 2, "both attempts are on disk"
        assert [line["turn"] for line in view.latest_attempt(again)] == [1, 1], \
            "and only the last of them is the episode being watched"
        assert view.episode_view("t", 1)["total_turns"] == 1


def check_the_page_fetches_nothing():
    """The page is self-contained: it names no host, no script and no stylesheet to fetch.

    A page that fetched anything would need a network this project does not
    give it. Fonts are the standing temptation: named in the page and left to
    the machine to have or not, never linked.
    """
    page = view.PAGE
    assert "<title>agentic-society</title>" in page
    assert "//cdn" not in page and "<script src" not in page, "nothing is fetched"
    for fetches in ("@import", "url(http", "url(//", "url('", 'url("', "<link", "fonts.googleapis"):
        assert fetches not in page, f"the page reaches out with {fetches}"
    assert page.count("<script>") == 1 and "</script>" in page, "one inline script, and nothing else"


def check_the_view_serves_its_api():
    """Every route answers: the page, the experiments, an agent, an episode, a
    tree, a file, and the messages."""
    with temp_root():
        episode_once(*DEFAULT)
        with serving() as base:
            with urllib.request.urlopen(base + "/") as r:
                assert r.status == 200 and r.read().decode("utf-8") == view.PAGE, "the page as it stands"
            assert got(base, "/api/experiments")[1]["experiments"][0]["name"] == "t"
            assert [s["episode"] for s in got(base, "/api/agent/t")[1]["episodes"]] == [1]
            assert got(base, "/api/agent/t")[1]["seat"] == "1"
            assert got(base, "/api/agent/t/episode/1")[1]["source"] == "trace"
            # An agent driven on its own is an experiment of one: its private store, the
            # one balance that goes with the seat it holds, and a ledger with
            # nothing in it. Nothing to address and nobody to be addressed by.
            private = got(base, "/api/experiment/t/tree/notes")[1]["columns"]
            assert [c["agent"] for c in private] == ["t"]
            assert [f["path"] for f in private[0]["files"]] == ["note.txt"]
            head = got(base, "/api/experiment/t")[1]
            assert [s["n"] for s in head["seats"]] == [ground_truth()["series"][-1]]
            assert head["ledger"] == [], "an experiment of one has given nothing to anyone"
            assert got(base, "/api/experiment/t/file?agent=t&channel=notes&path=note.txt")[1]["text"] == "hi\n"
            log = got(base, "/api/experiment/t/messages")[1]
            assert (log["events"], log["tip"], log["seats"], log["committed"]) == ([], [], 1, 0), log


def check_a_name_off_the_url_cannot_leave_records():
    """A name off the URL reaches the filesystem only after matching a listing,
    so nothing can be walked out of records/ or environments/ by asking, and
    every unknown name is a 404."""
    with temp_root():
        episode_once(*DEFAULT)
        with serving() as base:
            for bad in ("/api/agent/nope", "/api/agent/t/episode/99", "/api/nope",
                        "/api/experiment/nope", "/api/experiment/t/tree/nope",
                        "/api/experiment/t/file?agent=t&channel=notes&path=../../account.json",
                        "/api/experiment/t/file?agent=t&channel=notes&path=..%2F..%2Faccount.json",
                        "/api/experiment/t/file?agent=nope&channel=notes&path=note.txt",
                        "/api/experiment/t/file?agent=t&channel=../records&path=account.json"):
                try:
                    got(base, bad)
                except urllib.error.HTTPError as e:
                    assert e.code == 404, (bad, e.code)
                else:
                    raise AssertionError(f"answered for {bad}")


def check_the_view_never_writes():
    """Reading an agent leaves every byte of it where it was.

    The whole design rests on this: the view is display only, in the same
    category as --watch, and an agent must not be able to tell it was watched.
    """
    def digest(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}

    def sweep():
        c = view.experiment_of("t")
        view.header(c)
        view.messages(c)
        for kind in ("blackboard", "notes"):
            view.tree_view(c, kind)
        view.file_view(c, "t", "notes", "note.txt")
        view.agent_view("t", c)
        view.episode_view("t", 1)

    with temp_root() as root:
        episode_once(*DEFAULT)
        finished = digest(root)
        sweep()
        settled = digest(root)
        # And again with the episode unfinished, which is the path that reads
        # the raw log and derives instead of reading a field.
        unfinished()
        running = digest(root)
        sweep()
        watched = digest(root)

    assert settled == finished, sorted(set(settled) ^ set(finished)) or "contents changed"
    assert watched == running, sorted(set(watched) ^ set(running)) or "contents changed"


def check_the_view_names_a_set_of_agents():
    """An experiment names itself; an agent started alone is named by its id's letters.

    Every member has to arrive at the same name, or the sidebar's filter splits
    one experiment across sets, so the name comes from the membership.
    """
    seats = {"1": "q01", "2": "q02", "3": "q03"}
    peers = {"q01": seats, "q02": seats, "q03": seats}
    named = {agent: view.group_of(agent, {"peers": {"seen": seen}}) for agent, seen in peers.items()}
    assert set(named.values()) == {"q"}, named

    # No prefix in common: still one set, and still one name for it.
    both = [view.group_of("alpha", {"peers": {"seen": {"2": "beta"}}}),
            view.group_of("beta", {"peers": {"seen": {"1": "alpha"}}})]
    assert both == ["alpha+beta", "alpha+beta"], both

    # Started alone, with no experiment to ask: live01 and live02 sit together.
    assert [view.group_of(r, {}) for r in ("live01", "live02", "b01s", "solo")] == \
        ["live", "live", "b", "solo"]


def check_the_view_shows_every_seat_side_by_side():
    """A tab is one tree of every seat's environment, a column each, in seat order.

    The columns are what makes an experiment readable. A peer's private store is in
    no column of the blackboards and no column but its own of the stores.
    """
    with two_seats() as (c, seats):
        assert c["seated"] and c["seats"] == seats, (c["seated"], c["seats"])
        boards = view.tree_view(c, "blackboard")["columns"]
        stores = view.tree_view(c, "notes")["columns"]
        opened = view.file_view(c, "g02", "blackboard", "msg")

    # Seat order, and every seat present: the numbering is absolute, so column 2
    # is g02 to whoever is reading and not the second one they were shown.
    assert [(col["seat"], col["agent"]) for col in boards] == [("1", "g01"), ("2", "g02")]
    assert [(col["seat"], col["agent"]) for col in stores] == [("1", "g01"), ("2", "g02")]
    assert [[f["path"] for f in col["files"]] for col in boards] == [["msg"], ["msg"]]
    assert [[f["path"] for f in col["files"]] for col in stores] == \
        [["NOTES.md", "secret.md"], ["secret.md"]]
    # A listing is a listing: a file is read when it is opened, not on every poll.
    assert all("text" not in f for col in boards + stores for f in col["files"])
    assert opened["text"] == "hello 1\n", "a blackboard is read from the agent that owns it"
    # starter is the starter files alone, and it is a fact about the private store.
    assert [f["path"] for f in stores[0]["files"] if f["starter"]] == ["NOTES.md"]
    assert not [f for col in boards for f in col["files"] if f["starter"]]
    assert [f["path"] for col in stores for f in col["files"] if f["path"] == "secret.md"] == \
        ["secret.md", "secret.md"], "each store holds its own, and neither holds the other's"


def check_the_view_shows_every_balance_from_its_own_account():
    """A seat's n is that agent's ground truth, read from no file in any environment.

    A peer's figure is as authoritative as the watched agent's, both coming from
    the account the harness plants from: 800 is g02's, in no file g01 can read.
    """
    with two_seats() as (c, _):
        h = view.header(c)

    assert [(s["seat"], s["agent"], s["n"]) for s in h["seats"]] == \
        [("1", "g01", 900), ("2", "g02", 800)]
    assert h["ledger"] == [], "an experiment that has given nothing has an empty ledger"
    assert h["round"] == 0, "no episode has been committed, so no round has been taken"
    assert [s["transfer"] for s in h["seats"]] == [None, None], "and nobody has declared one"
    assert [s["starter_files"] for s in h["seats"]] == ["objective-notes", "objective-notes"]
    assert h["seated"]


def check_the_view_cuts_a_round_where_an_agent_repeats():
    """A round is read back out of the order the episodes started in.

    experiment.py writes no round anywhere, and an agent that sits one out falls behind
    for good. One episode per agent per round is the cut: an agent acting twice.
    """
    # Fixed seat order, with seat 2 sitting round 3 out.
    # The last two share an episode second, which a round has to survive.
    acted = [("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
             ("g02", "00:04"), ("g03", "00:05"), ("g01", "00:06"),
             ("g03", "00:07"), ("g01", "00:08"),
             ("g01", "00:09"), ("g02", "00:10"), ("g03", "00:10")]
    with rooted(HostBox) as root:
        c = fake_experiment(root, acted)
        rows = view.experiment_episodes(c)
        now = view.round_now(c, rows)
        seen = view.agent_view("g02", c)["episodes"]

    assert [r["round"] for r in rows] == [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4], \
        [(r["agent"], r["round"]) for r in rows]
    assert now == 4
    # The one the episode index gets wrong: g02 sat round 3 out, so its third
    # episode is round 4 and counting episodes would have called it round 3.
    assert [(s["episode"], s["round"]) for s in seen] == [(1, 1), (2, 2), (3, 4)]
    # Two agents starting in the same second are still one round; only an agent taking
    # a second turn cuts one.
    assert [(r["agent"], r["round"]) for r in rows[-2:]] == [("g02", 4), ("g03", 4)]


def check_the_view_tells_a_seat_not_yet_reached_from_one_that_passed():
    """Mid-round, a seat still to come is not a seat that sat the round out.

    In sequential rounds, from the traces alone the two look identical until
    the round ends. Nothing here asks whether an agent could have started an episode.
    """
    def tiles(acted):
        # Nothing left on any account, which is the whole point: a seat still to
        # come reads that way on the balance that would stop it starting.
        with rooted(HostBox) as root:
            c = fake_experiment(root, acted, series=(0,))
            rows = view.experiment_episodes(c)
            rnd = view.round_now(c, rows)
            return rnd, {agent: view.seat_row(seat, agent, rows, rnd)
                         for seat, agent in view.places_of(c)}

    # Round 3 open, g01 has taken it and the other two have not been reached.
    open_rnd, open_seats = tiles([
        ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
        ("g01", "00:04"), ("g02", "00:05"), ("g03", "00:06"),
        ("g01", "00:07")])
    # The same shape with g03 having missed round 2, so round 3 finds it a round
    # behind and not a round late.
    past_rnd, past_seats = tiles([
        ("g01", "00:01"), ("g02", "00:02"), ("g03", "00:03"),
        ("g01", "00:04"), ("g02", "00:05"),
        ("g01", "00:06")])

    assert open_rnd == 3 and past_rnd == 3, (open_rnd, past_rnd)
    assert open_seats["g01"]["acted"] and not open_seats["g01"]["pending"], \
        "the seat that took the round is neither waiting nor out of it"
    for agent in ("g02", "g03"):
        assert open_seats[agent]["pending"] and not open_seats[agent]["acted"], \
            f"{agent} has not been reached in the open round and has not sat it out"
    # A balance of zero is not what decides it: g02 is still to come on the same
    # empty account that g03 has passed on.
    assert past_seats["g03"]["round"] == 1 and not past_seats["g03"]["pending"], \
        "a seat a whole round behind has been asked and passed"
    assert past_seats["g02"]["pending"], "g02 acted in round 2 and is still to come"


def check_the_view_reads_a_message_out_of_two_outboxes():
    """The log records each episode-scoped delivery.

    Messages that are not sent in a later episode create no standing or withdrawal
    event.
    """
    with temp_root() as root:
        seated(root, g02={})
        # Each episode leaves its blackboard something new as well, so the post
        # penalty does not halve the budget four times over what is being read.
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        episode_once(run("echo louder > out/2", "echo r2 >> 1/log"), say())
        episode_once(run("printf '2 10\\n' > out/transfer", "echo r3 >> 1/log"), say())
        episode_once(run("rm -f out/2", "echo r4 >> 1/log"), say())
        c = view.experiment_of("t")
        m = view.messages(c)
        seen = [e for e in m["events"] if e["path"] == "out/2"]
        transfers = [e for e in m["events"] if e["kind"] == "transfer"]

    assert [e["change"] for e in seen] == ["sent", "sent"], \
        [(e["round"], e["change"]) for e in seen]
    assert {e["to_seat"] for e in seen} == {"2"} and {e["to_agent"] for e in seen} == {"g02"}
    assert [e["from_seat"] for e in seen] == ["1"] * 2, "and every one of them is seat 1's"
    assert [e["round"] for e in seen] == [1, 2]
    assert seen[0]["text"] == "hello\n" and seen[1]["text"] == "louder\n"
    assert not seen[1]["diff"], "each episode is a new delivery, not an edit"
    # Each declaration is its own episode event and carries resolve_transfer's
    # verdict, which is the only place a declaration that moved nothing says why.
    assert [e["change"] for e in transfers] == ["sent"], transfers
    assert transfers[0]["transfer"]["amount"] == 10 and transfers[0]["transfer"]["error"] is None
    assert transfers[0]["to_seat"] == "2" and transfers[0]["delivered"] is None, \
        "a transfer reaches nobody in particular: what it moved is in g, which all read"
    assert m["committed"] == len(m["events"]) and not m["tip"]


def check_the_view_shows_what_the_receiver_has_not_seen_yet():
    """The outbox on disk stands ahead of the last trace, and the log says so.

    The files are mirrored back before the trace is written, and an episode that
    died writes no trace at all. What stands now is shown as standing now.
    """
    with temp_root() as root:
        seated(root, g02={}, g03={})
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        before = view.messages(view.experiment_of("t"))
        later = harness.mirror("t", "mail") / "3"
        later.write_text("written behind the harness\n", encoding="utf-8")
        after = view.messages(view.experiment_of("t"))

    assert not before["tip"], "nothing stands ahead of the trace that was just written"
    assert [e["path"] for e in after["tip"]] == ["out/3"]
    tip = after["tip"][0]
    assert tip["round"] is None and tip["tip"] and tip["change"] == "sent"
    assert tip["delivered"] is None, "no episode has started since, so nobody has been given it"
    assert after["events"] == before["events"], "and what is committed did not move"


def check_the_view_reads_delivery_off_the_observation():
    """A message is delivered by the digest, so no command has to name the inbox.

    The initial observation carries in/<sender>, which is the addressee holding it.
    Reading delivery off the commands instead would call a delivered message unread.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo hello > out/2", "echo r1 >> 1/log"), say())
        with quiet():
            taken = harness.run_once("other", fake(run("cat n2"), say()))
        m = view.messages(view.experiment_of("t"))
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    assert "=== in/1 ===" in taken["observation"], "the addressee started holding the message"
    assert "in/1" not in " ".join(taken["commands"]), "and named it in no command of its own"
    d = ev["delivered"]
    assert d["shown_before"] is True, "which is the whole of what delivery is now"
    assert (d["environment"], d["named"], d["clipped"]) == (True, False, False), d


def check_the_view_says_delivery_where_the_observation_carried_nothing():
    """An observation that is the listing alone leaves the inbox to be fetched.

    Delivery is a different fact under that arrangement, and the log says so
    instead of answering the question it can answer here as if it were asked.
    """
    def held(path: str, role: str, ours: bool) -> dict:
        return {"observation": "total 0\n. ..\n", "commands": ["ls -la . ./state"],
                "files": [{"path": path, "channel": "mail", "writer": "self", "readers": "addressee",
                           "role": role, "text": "hello\n", "size": 6, "ours": ours, "starter": False}]}

    with rooted(HostBox) as root:
        c = fake_experiment(root, [("g01", "00:01", held("out/2", "own", False)),
                                   ("g02", "00:02", held("in/1", "peer", True))],
                            agents=("g01", "g02"))
        m = view.messages(c)
        ev = next(e for e in m["events"] if e["path"] == "out/2")

    d = ev["delivered"]
    assert d["shown_before"] is None, "nothing was shown_before, so it answers nothing"
    assert d["environment"] is True, "and what it could have fetched is still a fact"
    assert (d["named"], d["clipped"]) == (False, False), d


def check_the_view_splits_the_observation_into_the_listing_and_the_rest():
    """The two halves of an episode, apart, and reassembling into what was sent.

    The pieces are the page's, so what it shows a reader has to be the bytes the
    model got and not a rendering of them.
    """
    with temp_root() as root:
        seated(root, other={})
        t = episode_once(*DEFAULT)
        o = view.episode_view("t", 1)["observation"]

    assert o["listing"] + "".join(f"=== {s['path']} ===\n{s['text']}" for s in o["shown_before"]) \
        == t["observation"], "the halves are the whole observation and nothing else"
    assert "=== " not in o["listing"], "the listing ends where the first section starts"
    paths = [s["path"] for s in o["shown_before"]]
    assert digest_name() not in paths, "the digest names its own sections and never itself"
    assert "n1" in paths and ledger_name() in paths, paths
    assert all(s["bytes"] == len(s["text"].encode("utf-8")) for s in o["shown_before"])
    assert o["clipped"] is False and o["name"] == digest_name()


def check_the_view_carries_every_provenance_field_the_trace_holds():
    """Everything drift can name reaches the page, because nothing is picked.

    drift() reports on every key provenance() writes, so a panel holding a list
    of its own can name a field in a banner and then not show it.
    """
    with temp_root() as root:
        seated(root, other={})
        episode_once(*DEFAULT)
        seen = view.agent_view("t", view.experiment_of("t"))["episodes"][0]["provenance"]
        want = harness.provenance("anthropic", "claude-sonnet-5")

    assert set(want) <= set(seen), sorted(set(want) - set(seen))
    assert {"digest_file_limit", "observation_limit"} <= set(seen), "the two that decide what the digest carries"
    # The page picks no field, so no field can be left behind by one.
    assert not [k for k in want if f'"{k}"' in view.PAGE], \
        "the provenance panel iterates what the trace holds and names nothing"


def check_the_view_states_an_obligation_the_grace_waived():
    """What an episode owed and what it was charged are two questions.

    A share is taken only past the grace, so an episode inside it can leave all
    three undone for nothing. Every pane says the same thing about that episode.
    """
    with temp_root(GRACE_EPISODES=1, channels=ALL_OWED) as root:
        seated(root, other={})
        t = episode_once(run("cat n1"), say())
        v = view.episode_view("t", 1)
        mine = next(s for s in view.header(view.experiment_of("t"))["seats"] if s["agent"] == "t")

        assert v["obligations"] == {"blackboard": False, "mail": False, "transfer": False}, \
            v["obligations"]
        assert (t["channels"]["blackboard"]["penalty"], t["channels"]["mail"]["penalty"], t["transfer"]["penalty"]) == (0, 0, 0), \
            "and the grace charged it for none of them"
        assert [(u["channel"], u["why"], u["penalty"]) for u in v["unmet"]] == [
            ("blackboard", "no post", 0), ("mail", "no message", 0),
            ("transfer", "no transfer of its own", 0)], v["unmet"]
        # The tile and the transcript answer from one place, so neither can
        # state an obligation the other leaves out.
        assert mine["unmet"] == v["unmet"], (mine["unmet"], v["unmet"])


def check_the_view_counts_what_a_seat_spent_and_not_what_it_lost():
    """A tile's spend is its turns, and the bar beside it is the balance.

    A transfer, a share taken and a floor all move the balance without being spend,
    so the drop from initial answers a different question and can be larger. The
    tile's transfer chip is the declaration submitted by its last episode and what
    settlement made of it. After an episode that submitted nothing there is nothing
    to show.
    """
    def tile() -> dict:
        return next(s for s in view.header(view.experiment_of("t"))["seats"] if s["agent"] == "t")

    with temp_root() as root:
        seated(root, other={})
        episode_once(run("echo '2 100' > out/transfer", "echo r1 >> 1/log"), say())
        mine = tile()
        gt = ground_truth("t")
        (harness.mirror("t", "mail") / "transfer").unlink()
        resolved = tile()["transfer"]
        episode_once(run("rm -f out/transfer"), say())
        gone = tile()["transfer"]

    assert mine["spent"] == sum(s["spent"] for s in gt["episodes"]), \
        (mine["spent"], [s["spent"] for s in gt["episodes"]])
    assert mine["spent"] != gt["initial"] - gt["remaining"], \
        "the transfer moved the balance without being spent"
    assert mine["spent_this_round"] <= mine["spent"], "a round is part of a life"
    assert mine["rebated"] == gt["rebated"] > 0, "and what it won back is on the tile"
    assert mine["transfer"] == {"submitted": True, "declared": "2 100\n", "seat": "2", "label": "2",
                                "agent": "other", "amount": 100, "rebate": 100, "error": None}, mine["transfer"]
    assert resolved == mine["transfer"], \
        f"the tile reports the resolved episode, not a mutable outbox mirror: {resolved}"
    assert gone is None, f"nothing declared and nothing was: {gone}"


def check_the_view_builds_its_tabs_from_the_table():
    """The page's tabs, trees, message log and diffs follow the table the traces record."""
    with temp_root(channels=PERSONA, harness_files=PERSONA_FILES) as root:
        seated(root, labels=PERSONA_LABELS, other={})
        episode_once(run("echo me > journal/IDENTITY.md", "echo n > from-Studio/post",
                         "echo hi > to/Game"), say())
        c = view.experiment_of("t")
        h = view.header(c)
        journal = view.tree_view(c, "journal")
        board = view.tree_view(c, "noticeboard")
        opened = view.file_view(c, "t", "journal", "IDENTITY.md")
        missing = view.file_view(c, "t", "blackboard", "post")
        v = view.episode_view("t", 1)
        changes = view.episode_changes("t", 1)
        events = view.messages(c)["events"]
    assert [(tab["key"], tab["label"]) for tab in h["tabs"]] == \
        [("mailbox", "letters"), ("journal", "journal"), ("noticeboard", "noticeboard"),
         ("agent", "transcripts")], h["tabs"]
    assert h["balance"] == "balance" and h["labels"] == PERSONA_LABELS
    assert [s["label"] for s in h["seats"]] == ["Studio", "Game"]
    assert journal["what"] == "no other agent ever reads this one"
    assert board["what"] == "every agent reads this one"
    assert [f["path"] for f in journal["columns"][0]["files"]] == ["IDENTITY.md"]
    assert [f["path"] for f in board["columns"][0]["files"]] == ["post"]
    assert opened["text"] == "me\n" and missing is None
    assert v["observation"]["name"] == "digest" and v["observation"]["inbox"] == "from"
    assert set(v["channels"]) == {"noticeboard", "letters"} and v["channels"]["noticeboard"]["posted"]
    assert [c["channel"] for c in changes] == ["journal", "identity", "noticeboard", "letters"]
    assert any(line.startswith("+me") for line in changes[1]["lines"]), changes[1]
    assert changes[0]["lines"] == [], "the identity file is its own channel, not the journal's"
    assert [(e["path"], e["from_label"], e["to_label"], e["to_seat"]) for e in events] == \
        [("to/Game", "Studio", "Game", "2")], events


def check_the_view_shows_an_experimenter_channel_once():
    """An experimenter channel is one shared source, not one writable tree per seat."""
    with temp_root() as root:
        shared(root, name="brief", path="briefing", RULES="same words for every seat\n")
        c = fake_experiment(root, [])
        h = view.header(c)
        tree = view.tree_view(c, "briefing")
        opened = view.file_view(c, "experimenter", "briefing", "RULES")
    assert [(tab["key"], tab["label"]) for tab in h["tabs"]] == \
        [("mailbox", "mail + transfer"), ("notes", "notes"),
         ("blackboard", "blackboard"), ("briefing", "briefing"),
         ("agent", "transcripts")]
    assert tree["static"] and [c["agent"] for c in tree["columns"]] == ["experimenter"]
    assert tree["what"] == "provided by the experimenter; every agent reads the same files"
    assert opened["text"] == "same words for every seat\n"


def check_a_long_series_is_thinned_to_its_ends():
    """A header sparkline keeps SPARK_POINTS of a long series, its first element and
    its last among them, in order; a short series is kept whole."""
    thinned = view.thin(list(range(1000)))
    assert len(thinned) == view.SPARK_POINTS, len(thinned)
    assert thinned[0] == 0 and thinned[-1] == 999, (thinned[0], thinned[-1])
    assert all(a < b for a, b in zip(thinned, thinned[1:])), "sampled in order, nothing repeated"
    assert view.thin(list(range(1000)), 5) == [0, 250, 500, 749, 999]
    assert view.thin([7, 8, 9]) == [7, 8, 9] and view.thin([]) == []


def check_the_view_cli_refuses_what_is_not_there():
    """An agent or an experiment that is not on disk is refused by the parser, and
    a port that cannot be bound is reported and not served.

    The port is held by a socket that refuses to share it, which is what a
    port in use looks like on every platform.
    """
    with temp_root():
        with quiet():
            refused(lambda: view.main(["--agent", "nope", "--no-browser"]), code=2,
                    because="an agent that is not there was opened on")
            refused(lambda: view.main(["--experiment", "nope", "--no-browser"]), code=2,
                    because="an experiment that is not there was opened on")
        held = socket.socket()
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                held.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            held.bind(("127.0.0.1", 0))
            held.listen()
            port = held.getsockname()[1]
            with quiet() as buf:
                assert view.main(["--port", str(port), "--no-browser"]) == 1
        finally:
            held.close()
    assert f"port {port}" in buf.getvalue(), buf.getvalue()


def lines_ending_inside_a_string(script: str) -> list[int]:
    """The line numbers of a script that end inside a double-quoted string.

    Single-quoted strings, template literals with their interpolations, and
    comments are stepped over, so only a double-quoted string cut by a line end
    is reported.
    """
    stack: list[list] = [["js", 0]]
    bad, i, n, line = [], 0, len(script), 1
    while i < n:
        c, nxt = script[i], script[i + 1] if i + 1 < n else ""
        kind = stack[-1][0]
        if c == "\n":
            if kind == "dq":
                bad.append(line)
                stack.pop()
            line += 1
            i += 1
            continue
        if kind == "js":
            if c == "/" and nxt == "/":
                cut = script.find("\n", i)
                i = n if cut < 0 else cut
                continue
            if c == "/" and nxt == "*":
                stack.append(["block", 0])
                i += 2
                continue
            if c == '"':
                stack.append(["dq", 0])
            elif c == "'":
                stack.append(["sq", 0])
            elif c == "`":
                stack.append(["tpl", 0])
            elif c == "{":
                stack[-1][1] += 1
            elif c == "}":
                if stack[-1][1] == 0 and len(stack) > 1:
                    stack.pop()
                else:
                    stack[-1][1] -= 1
            i += 1
            continue
        if kind == "block":
            if c == "*" and nxt == "/":
                stack.pop()
                i += 2
                continue
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if kind == "dq" and c == '"':
            stack.pop()
        elif kind == "sq" and c == "'":
            stack.pop()
        elif kind == "tpl":
            if c == "`":
                stack.pop()
            elif c == "$" and nxt == "{":
                stack.append(["js", 0])
                i += 2
                continue
        i += 1
    return bad


def check_the_page_has_no_raw_newline_inside_a_js_string():
    """No line of the page's script ends inside a double-quoted string.

    A template literal may span lines and a double-quoted string may not, so a
    line end inside one is a string the browser would refuse to parse.
    """
    assert lines_ending_inside_a_string('const a = "one\ntwo;\n') == [1]
    assert lines_ending_inside_a_string('const b = `x "y\nz" ${"w"}`;\n// "cut\nconst c = "ok";\n') == []
    assert lines_ending_inside_a_string("const d = 'one\\'\"\ntwo';\n") == []
    script = view.PAGE.partition("<script>")[2].partition("</script>")[0]
    assert script.strip(), "the page has a script"
    assert lines_ending_inside_a_string(script) == [], \
        f"script lines ending inside a string: {lines_ending_inside_a_string(script)}"
