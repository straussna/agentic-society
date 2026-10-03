"""Interactive provider, durable coordination, and browser control routes."""

from __future__ import annotations

import json
import subprocess
import sys
import builtins
import contextlib
import http.server
import io
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

import harness
import product
import view
from interaction import (
    InteractionCancelled,
    InteractionConflict,
    InteractionError,
    InteractionStore,
    InvalidSubmission,
    StaleRequest,
    Submission,
    UnreadableRecord,
)
from interaction import cli
from providers import SessionContext, ToolResult, ToolSpec, Usage
from providers.human import HumanProvider, HumanSession

from checks.lanes import episode_once, temp_root


TOOLS = (
    ToolSpec(
        "remember",
        "Keep private memory",
        {
            "type": "object",
            "properties": {"body": {"type": "string"}},
            "required": ["body"],
            "additionalProperties": False,
        },
    ),
)


def submission(request, body="hello"):
    return {
        "version": 1,
        "request_id": request.request_id,
        "submission_id": "human-call-1",
        "action": "tool_calls",
        "tool_calls": [{"id": "call-1", "name": "remember", "input": {"body": body}}],
    }


def end_submission(request):
    return {
        "version": 1,
        "request_id": request.request_id,
        "submission_id": "human-end-1",
        "action": "end_turn",
        "tool_calls": [],
    }


def raised(
    error: type[BaseException], call: Callable[[], Any], because: str
) -> BaseException:
    """What `call` raised, which must be an `error`; `because` says what it did instead."""
    try:
        call()
    except error as caught:
        return caught
    raise AssertionError(because)


def submission_file(root: Path, agent: str, request_id: str) -> Path:
    return root / "interactions" / "submissions" / agent / f"{request_id}.json"


def check_interaction_store_preserves_requests_and_first_submission():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "Seat A", 2, 3, "system", "observation", TOOLS)
        on_disk = store.current("a")
        assert on_disk == request
        assert on_disk.available_tools == TOOLS, (
            "declared tools retain their order and exact schemas"
        )
        undeclared = submission(request)
        undeclared["tool_calls"][0]["name"] = "not_offered"
        try:
            store.submit("a", request.request_id, undeclared)
        except InvalidSubmission:
            pass
        else:
            raise AssertionError("an undeclared tool entered the interaction store")
        won = store.submit("a", request.request_id, submission(request))
        assert store.submit("a", request.request_id, submission(request)) == won, (
            "a repeated submission id returns the winner"
        )
        try:
            store.submit(
                "a",
                request.request_id,
                {**submission(request, "different"), "submission_id": "other"},
            )
        except InteractionConflict:
            pass
        else:
            raise AssertionError(
                "a conflicting client cannot replace the winning submission"
            )
        assert store.wait(request, lambda: False, 0).tool_calls[0].input == {
            "body": "hello"
        }
        assert store.request("a", request.request_id).status == "completed"
        assert store.current("a") is None


def check_new_interaction_request_isolated_from_stale_submissions_and_keeps_draft():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        old = store.publish("a", "A", 1, 1, "", "first", TOOLS)
        store.save_draft("a", old.request_id, submission(old)["tool_calls"])
        assert store.load_draft("a", old.request_id)[0]["name"] == "remember"
        new = store.publish("a", "A", 1, 2, "", (ToolResult("call-1", "ok"),), TOOLS)
        assert store.request("a", old.request_id).status == "cancelled"
        try:
            store.submit("a", old.request_id, submission(old))
        except StaleRequest:
            pass
        else:
            raise AssertionError("an old request satisfied a newer turn")
        assert store.current("a").request_id == new.request_id
        assert new.input["kind"] == "tool_results"


def check_a_store_write_retries_a_rename_a_reader_holds_up():
    """On Windows a reader holding the target open refuses the rename for as long as the
    read takes, and view.py reads the pending pointer and the request on every poll."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        real, refused = os.replace, []

        def held(source, destination) -> None:
            if len(refused) < product.RENAME_ATTEMPTS - 1:
                refused.append(destination)
                raise PermissionError(13, "held open by a reader", str(destination))
            real(source, destination)

        def never(source, destination) -> None:
            raise PermissionError(13, "held open by a reader", str(destination))

        os.replace = held
        try:
            request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        finally:
            os.replace = real
        assert len(refused) == product.RENAME_ATTEMPTS - 1, refused
        assert store.current("a") == request, "the write landed once the reader let go"
        os.replace = never
        try:
            raised(
                PermissionError,
                lambda: store.publish("a", "A", 1, 2, "", "again", TOOLS),
                "a rename refused on every attempt was taken for a write",
            )
        finally:
            os.replace = real
        assert not list((root / "interactions").rglob("*.tmp")), (
            "a write that failed left its temporary file behind"
        )
        assert store.current("a") == request, "and what was there before is still there"


def check_every_malformed_submission_envelope_is_refused_and_the_request_stays_pending():
    """The envelope is all that stands between a browser POST and a human seat's turn."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        good = submission(request)

        def calling(**call: Any) -> dict:
            return {**good, "tool_calls": [call]}

        malformed = {
            "version 2": {**good, "version": 2},
            "a version that is not a number": {**good, "version": "one"},
            "another request's id": {**good, "request_id": "another"},
            "no submission_id": {k: v for k, v in good.items() if k != "submission_id"},
            "an empty submission_id": {**good, "submission_id": ""},
            "a submission_id that is not text": {**good, "submission_id": 1},
            "no action": {k: v for k, v in good.items() if k != "action"},
            "an action other than tool_calls or end_turn": {**good, "action": "wait"},
            "tool_calls that are not a list": {**good, "tool_calls": {}},
            "a call that is not an object": {**good, "tool_calls": ["remember"]},
            "a call without an id": calling(name="remember", input={"body": "x"}),
            "a call with an empty id": calling(
                id="", name="remember", input={"body": "x"}
            ),
            "a call without a name": calling(id="call-1", input={"body": "x"}),
            "a call without input": calling(id="call-1", name="remember"),
            "a call whose input is not an object": calling(
                id="call-1", name="remember", input="x"
            ),
            "a call to a tool not offered": calling(
                id="call-1", name="forget", input={}
            ),
            "end_turn with a call": {**good, "action": "end_turn"},
            "tool_calls without a call": {**good, "tool_calls": []},
        }
        for case, value in malformed.items():
            raised(
                InvalidSubmission,
                lambda: store.submit("a", request.request_id, value),
                f"{case} was accepted",
            )
            assert store.current("a") == request, f"{case} moved the request on"
        assert not (root / "interactions" / "submissions").exists(), (
            "a refused envelope left a submission behind"
        )
        assert (
            store.submit("a", request.request_id, good).submission_id == "human-call-1"
        ), "and the request still takes a well-formed one"


class HeldBeforeTheLink(InteractionStore):
    """A store whose submitters each wait for the other after finding no winner yet.

    submit reads current() between its look for an existing submission and its link,
    so every submitter held here reaches os.link, and one of them loses the link itself.
    """

    gate: threading.Barrier | None = None

    def current(self, agent: str):
        if self.gate is not None:
            self.gate.wait()
        return super().current(agent)


def race(
    store: HeldBeforeTheLink, request_id: str, payloads: dict[str, dict]
) -> dict[str, Submission | BaseException]:
    """Every payload submitted at once, one thread each: what each client got back."""
    store.gate = threading.Barrier(len(payloads), timeout=10)
    got: dict[str, Submission | BaseException] = {}

    def client(name: str, payload: dict) -> None:
        try:
            got[name] = store.submit("a", request_id, payload)
        except Exception as error:
            got[name] = error

    threads = [
        threading.Thread(target=client, args=item, daemon=True)
        for item in payloads.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive(), "a submitter never came back from the race"
    store.gate = None
    return got


def check_two_clients_racing_one_request_leave_one_winner_and_one_conflict():
    """The browser panel and the terminal can submit one turn at once; the first link wins."""
    with temp_root() as root:
        store = HeldBeforeTheLink(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        got = race(
            store,
            request.request_id,
            {
                name: {**submission(request, name), "submission_id": name}
                for name in ("browser", "terminal")
            },
        )
        won = [name for name, outcome in got.items() if isinstance(outcome, Submission)]
        lost = [
            name
            for name, outcome in got.items()
            if isinstance(outcome, InteractionConflict)
        ]
        assert len(won) == 1 and len(lost) == 1, got
        on_disk = json.loads(
            submission_file(root, "a", request.request_id).read_text(encoding="utf-8")
        )
        assert on_disk["submission_id"] == won[0], (
            "the file holds the winner's submission"
        )
        assert store.wait(request, lambda: False, 0).tool_calls[0].input == {
            "body": won[0]
        }


def check_one_submission_sent_twice_at_once_returns_the_winner_to_both():
    """A double-click is one submission: whichever copy links first, both get it back."""
    with temp_root() as root:
        store = HeldBeforeTheLink(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        got = race(
            store,
            request.request_id,
            {"first": submission(request), "again": submission(request)},
        )
        assert all(isinstance(outcome, Submission) for outcome in got.values()), got
        assert got["first"] == got["again"], "both copies return the one that won"


def check_the_winners_submission_id_on_other_calls_or_another_action_is_a_conflict():
    """Only the winner sent again gets the winner back: a client that reused its id on
    a different turn would otherwise be told it won while its own calls were dropped."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        won = store.submit("a", request.request_id, submission(request))
        reused = {
            "other calls": submission(request, "other"),
            "another action": {
                **end_submission(request),
                "submission_id": won.submission_id,
            },
        }
        for case, value in reused.items():
            assert value["submission_id"] == won.submission_id, case
            raised(
                InteractionConflict,
                lambda: store.submit("a", request.request_id, value),
                f"the winner's submission_id with {case} was answered as the winner",
            )
        assert store.wait(request, lambda: False, 0) == won, (
            "and the winner is the turn"
        )


def check_an_unreadable_submission_ends_the_wait_instead_of_polling_forever():
    """A submission that is there and does not read can never be replaced, so waiting
    on it would hold the seat, and a simultaneous round with it, for good."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        path = submission_file(root, "a", request.request_id)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        polls = []

        def cancelled() -> bool:
            polls.append(True)
            if len(polls) > 3:
                raise AssertionError(
                    "wait polled an unreadable submission as if it were absent"
                )
            return False

        error = raised(
            UnreadableRecord,
            lambda: store.wait(request, cancelled, 0),
            "wait returned a turn from an unreadable submission",
        )
        assert str(path) in str(error), f"the error names the file: {error}"
        assert store.request("a", request.request_id).status == "cancelled", (
            "and the request it gave up on is no longer pending"
        )
        assert store.current("a") is None


def check_an_unreadable_submission_is_reported_and_never_taken_for_a_winner():
    """Neither the client that wrote it nor any other: nothing the store can read won."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        path = submission_file(root, "a", request.request_id)
        path.parent.mkdir(parents=True)
        for written in ("[]", "{not json", '{"version": 1}'):
            path.write_text(written, encoding="utf-8")
            raised(
                UnreadableRecord,
                lambda: store.submit("a", request.request_id, submission(request)),
                f"a submission holding {written} was answered as if it had won or lost",
            )
            assert path.read_text(encoding="utf-8") == written, (
                "and it is left as it was"
            )


def check_an_interaction_file_that_does_not_read_is_told_from_one_that_is_absent():
    """Absent is None to every caller; there and unreadable is an error naming the file."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        assert (
            store.current("a") is None
            and store.pending() == []
            and store.history("a") == []
        )
        assert store.load_draft("a", "nothing") == []
        pointer = root / "interactions" / "pending" / "a.json"
        for written in ("[]", '{"request_id": 5}', "{not json"):
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(written, encoding="utf-8")
            error = raised(
                UnreadableRecord,
                lambda: store.current("a"),
                f"a pending pointer holding {written} read as no request",
            )
            assert str(pointer) in str(error), error
        pointer.unlink()

        request = store.publish("b", "B", 1, 1, "", "observation", TOOLS)
        assert [r.agent for r in store.pending()] == ["b"]
        pointer.write_text("[]", encoding="utf-8")
        assert [r.agent for r in store.pending()] == ["b"], (
            "the list of pending requests leaves out the seat it cannot read, and only that one"
        )
        request_file = (
            root / "interactions" / "requests" / "b" / f"{request.request_id}.json"
        )
        request_file.write_text('{"version": 1}', encoding="utf-8")
        raised(
            UnreadableRecord,
            lambda: store.request("b", request.request_id),
            "a request missing its fields read as no request",
        )
        raised(
            UnreadableRecord,
            lambda: store.current("b"),
            "a pending pointer to an incomplete request read as nothing pending",
        )


def check_a_pending_pointer_that_does_not_read_is_written_over_and_never_ends_a_wait():
    """The pointer only says which request is pending. The next request is written over
    one that does not read, and a wait on its own request ends as it would have: stopped
    as stopped, answered with the submission that landed."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        pointer = root / "interactions" / "pending" / "a.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text("[]", encoding="utf-8")
        stopped = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        assert store.current("a") == stopped, (
            "the request's own pointer replaced the broken one"
        )

        pointer.write_text("[]", encoding="utf-8")
        raised(
            InteractionCancelled,
            lambda: store.wait(stopped, lambda: True, 0),
            "a stopped wait over an unreadable pointer ended some other way",
        )
        assert store.request("a", stopped.request_id).status == "cancelled"
        assert pointer.read_text(encoding="utf-8") == "[]", (
            "the pointer is left as it was"
        )

        taken = store.publish("a", "A", 1, 2, "", (ToolResult("call-1", "ok"),), TOOLS)
        store.submit("a", taken.request_id, submission(taken))
        pointer.write_text("[]", encoding="utf-8")
        assert store.wait(taken, lambda: False, 0).tool_calls[0].input == {
            "body": "hello"
        }, "the submission that landed is the turn"
        assert store.request("a", taken.request_id).status == "completed"

        broken = store.publish("a", "A", 2, 1, "", "observation", TOOLS)
        request_file = (
            root / "interactions" / "requests" / "a" / f"{broken.request_id}.json"
        )
        request_file.write_text('{"version": 1}', encoding="utf-8")
        raised(
            InteractionCancelled,
            lambda: store.wait(broken, lambda: True, 0),
            "a stopped wait on a request that no longer reads ended some other way",
        )
        assert not pointer.exists(), (
            "the pointer to the request it stopped is taken down"
        )
        pointer.write_text(
            json.dumps({"request_id": broken.request_id}), encoding="utf-8"
        )
        after = store.publish("a", "A", 3, 1, "", "observation", TOOLS)
        assert store.current("a") == after, (
            "a pointer to a request that does not read is written over"
        )


def answered(store: InteractionStore, episode: int, turn: int, end: bool = False):
    """Seat a's request for one turn, published, answered and taken, as a human provider's is."""
    request = store.publish("a", "A", episode, turn, "", "observation", TOOLS)
    store.submit(
        "a",
        request.request_id,
        end_submission(request) if end else submission(request, f"{episode}.{turn}"),
    )
    store.wait(request, lambda: False, 0)
    return request


def check_history_pairs_each_answered_request_with_its_submission_in_turn_order():
    """Six turns over three episodes, so a directory listing in that order by chance is
    one in 720."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        turns = [(episode, turn) for episode in (1, 2, 3) for turn in (1, 2)]
        for episode, turn in turns:
            answered(store, episode, turn, end=turn == 2)
        store.publish("a", "A", 4, 1, "", "observation", TOOLS)
        history = store.history("a")
        assert [(r.episode, r.turn) for r, _ in history] == turns, (
            "every answered turn in order, and the request nobody answered left out"
        )
        assert all(r.request_id == s.request_id for r, s in history), (
            "each with its own submission"
        )
        assert [s.action for _, s in history] == ["tool_calls", "end_turn"] * 3
        assert [s.tool_calls[0].input for _, s in history if s.tool_calls] == [
            {"body": f"{episode}.1"} for episode in (1, 2, 3)
        ]
        assert store.history("b") == []
        raised(
            InteractionError,
            lambda: store.history("../a"),
            "an agent name walked out of the store",
        )


def check_history_leaves_out_a_turn_whose_request_or_submission_does_not_read():
    """A seat's history is still served when one of its turns is not: the player page
    asks for it on every poll."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        kept, cut, lost = (answered(store, 1, turn) for turn in (1, 2, 3))
        submission_file(root, "a", cut.request_id).write_text(
            "{truncated", encoding="utf-8"
        )
        request_file = (
            root / "interactions" / "requests" / "a" / f"{lost.request_id}.json"
        )
        request_file.write_text('{"version": 1}', encoding="utf-8")
        assert [r.request_id for r, _ in store.history("a")] == [kept.request_id]


@contextlib.contextmanager
def terminal(lines: Iterable[str]):
    """builtins.input answering each of `lines` in turn and then end of input; yields stdout."""
    remaining = iter(lines)

    def typed(prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    original = builtins.input
    builtins.input = typed
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            yield out
    finally:
        builtins.input = original


def said_in_order(out: io.StringIO, *expected: str) -> None:
    """Every expected line appears in what the terminal printed, in this order."""
    printed = out.getvalue().splitlines()
    at = 0
    for line in expected:
        found = next(
            (i for i in range(at, len(printed)) if printed[i].startswith(line)), None
        )
        assert found is not None, f"{line!r} not printed after line {at}: {printed}"
        at = found + 1


def check_human_cli_drafts_and_submits_declared_calls():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        commands = iter(
            ['call remember {"body":"from terminal"}', "draft", "submit", "quit"]
        )
        original = builtins.input
        try:
            builtins.input = lambda prompt="": next(commands)
            with contextlib.redirect_stdout(io.StringIO()):
                assert cli.run("a", root / "interactions") == 0
        finally:
            builtins.input = original
        submitted = store.wait(request, lambda: False, 0)
        assert submitted.tool_calls[0].input == {"body": "from terminal"}


def check_human_cli_refuses_what_it_cannot_send_and_ends_a_turn_without_calls():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        with terminal(
            [
                "call nope {}",
                "call remember [1]",
                "call remember",
                "call remember {bad",
                'call remember {"body":"x"}',
                "remove 2",
                "remove 1",
                "draft",
                "submit",
                "bogus",
                "done",
                'call remember {"body":"late"}',
                "quit",
            ]
        ) as out:
            assert cli.run("a", root / "interactions") == 0
        said_in_order(
            out,
            "unknown tool 'nope'",
            "tool input must be a JSON object",
            "usage: call <tool-name> <JSON-object>",
            "invalid value:",
            "drafted call 1",
            "no such draft call",
            "removed",
            "empty",
            "draft is empty; use done to finish without calls",
            "unknown command",
            "submitted end_turn",
            "No pending interaction; use refresh",
        )
        assert store.load_draft("a", request.request_id) == [], (
            "the removed call left the draft"
        )
        ended = store.wait(request, lambda: False, 0)
        assert ended.action == "end_turn" and ended.tool_calls == (), ended


def check_human_cli_follows_a_newer_request_and_reports_a_turn_it_lost():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        turns: dict[str, Any] = {}

        def session():
            yield "done"
            turns["first"] = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
            yield "refresh"
            turns["second"] = store.publish(
                "a", "A", 1, 2, "", (ToolResult("call-1", "ok"),), TOOLS
            )
            yield "done"
            yield "done"
            turns["third"] = third = store.publish(
                "a", "A", 2, 1, "", "observation", TOOLS
            )
            yield "refresh"
            store.submit("a", third.request_id, submission(third, "from the browser"))
            yield "done"
            turns["fourth"] = fourth = store.publish(
                "a", "A", 3, 1, "", "observation", TOOLS
            )
            yield "refresh"
            path = submission_file(root, "a", fourth.request_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")
            yield "done"

        with terminal(session()) as out:
            assert cli.run("a", root / "interactions") == 0, (
                "end of input leaves the terminal"
            )
        said_in_order(
            out,
            "No pending interaction for a.",
            "No pending interaction; use refresh after the episode starts.",
            "A · a · episode 1 · turn 1",
            "request changed: the request is no longer pending",
            "submitted end_turn",
            "A · a · episode 2 · turn 1",
            "not submitted: a different submission already won this request",
            "A · a · episode 3 · turn 1",
            "unreadable:",
        )
        second = store.wait(turns["second"], lambda: False, 0)
        assert second.action == "end_turn", (
            "the turn that moved on took the terminal's answer"
        )
        third = store.wait(turns["third"], lambda: False, 0)
        assert third.tool_calls[0].input == {"body": "from the browser"}, (
            "and the browser kept its win"
        )


def check_human_cli_reports_a_pointer_or_draft_that_does_not_read_and_keeps_going():
    """At the start, on refresh and after a request moves on: the file is named, the
    terminal goes on as if there were none, and the request after it can be answered."""
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        pointer = root / "interactions" / "pending" / "a.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text("[]", encoding="utf-8")
        turns: dict[str, Any] = {}
        drafts = root / "interactions" / "drafts" / "a"

        def session():
            yield "refresh"
            turns["first"] = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
            yield "refresh"
            turns["second"] = second = store.publish(
                "a", "A", 1, 2, "", (ToolResult("call-1", "ok"),), TOOLS
            )
            drafts.mkdir(parents=True, exist_ok=True)
            (drafts / f"{second.request_id}.json").write_text("{cut", encoding="utf-8")
            yield "done"
            yield "draft"
            yield "done"

        with terminal(session()) as out:
            assert cli.run("a", root / "interactions") == 0, (
                "end of input leaves the terminal"
            )
        said_in_order(
            out,
            f"unreadable: {pointer}",
            "No pending interaction for a.",
            f"unreadable: {pointer}",
            "No pending interaction.",
            "A · a · episode 1 · turn 1",
            "request changed: the request is no longer pending",
            f"unreadable: {drafts / turns['second'].request_id}.json",
            "empty",
            "submitted end_turn",
        )
        ended = store.wait(turns["second"], lambda: False, 0)
        assert ended.action == "end_turn", (
            "the request after the broken files took the answer"
        )


def human_session(
    root: Path, agent: str, stopping: threading.Event | None = None
) -> tuple[HumanSession, threading.Event]:
    """A human provider session, and an event set each time it waits on a published request.

    The provider asks whether it is cancelled on every poll of its wait, and the wait
    starts only once the request is on disk, so the check never polls the store for it.
    """
    published = threading.Event()

    def cancelled() -> bool:
        published.set()
        return stopping is not None and stopping.is_set()

    return HumanProvider().open_session(
        "interactive",
        "system",
        TOOLS,
        100,
        SessionContext(agent, f"Seat {agent}", 4, root / "interactions", cancelled),
    ), published


def requesting(
    session: HumanSession, content
) -> tuple[threading.Thread, dict[str, Any]]:
    """session.request(content) on a thread: the thread, and what it returned or raised."""
    outcome: dict[str, Any] = {}

    def ask() -> None:
        try:
            outcome["response"] = session.request(content)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    return thread, outcome


def check_human_provider_returns_a_submitted_call_as_a_zero_charge_tool_use_turn():
    with temp_root() as root:
        session, published = human_session(root, "a")
        thread, outcome = requesting(session, "observation")
        assert published.wait(5), "the human provider published no request"
        store = InteractionStore(root / "interactions")
        pending = store.current("a")
        assert pending is not None and pending.input["kind"] == "initial_observation", (
            pending
        )
        store.submit("a", pending.request_id, submission(pending))
        thread.join(5)
        assert not thread.is_alive(), (
            "the provider kept waiting after the submission landed"
        )
        assert "response" in outcome, outcome
        response = outcome["response"]
        assert response.native["request_id"] == pending.request_id
        turn = response.normalize()
        assert turn.provider == "human" and turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].as_dict() == submission(pending)["tool_calls"][0]
        assert turn.usage == Usage() and turn.charges == (), (
            "a human's turn costs nothing"
        )
        assert store.request("a", pending.request_id).status == "completed"


def check_human_provider_ends_the_turn_on_an_end_turn_submission():
    with temp_root() as root:
        session, published = human_session(root, "a")
        thread, outcome = requesting(session, (ToolResult("call-1", "stored"),))
        assert published.wait(5), (
            "the human provider published no request for the tool results"
        )
        store = InteractionStore(root / "interactions")
        pending = store.current("a")
        assert pending is not None and pending.input["kind"] == "tool_results", pending
        store.submit("a", pending.request_id, end_submission(pending))
        thread.join(5)
        assert not thread.is_alive(), (
            "the provider kept waiting after the submission landed"
        )
        assert "response" in outcome, outcome
        ended = outcome["response"].normalize()
        assert ended.stop_reason == "end_turn" and ended.tool_calls == (), ended


def check_human_provider_cancels_its_request_when_the_episode_stops():
    with temp_root() as root:
        stopping = threading.Event()
        session, published = human_session(root, "b", stopping)
        thread, outcome = requesting(session, "observation")
        assert published.wait(5), "the human provider published no request"
        store = InteractionStore(root / "interactions")
        pending = store.current("b")
        assert pending is not None
        stopping.set()
        thread.join(5)
        assert not thread.is_alive(), (
            "the provider kept waiting after the episode stopped"
        )
        assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome
        assert store.request("b", pending.request_id).status == "cancelled"
        assert store.current("b") is None, (
            "nothing is left pending for a client to answer"
        )


def check_view_interaction_route_requires_origin_token_and_pending_request():
    with temp_root() as root:
        episode_once()
        store = InteractionStore(root / "interactions")
        pending = store.publish("t", "1", 2, 1, "system", "observation", TOOLS)
        httpd = view.serve(0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/api/interaction/t") as response:
                assert (
                    json.loads(response.read().decode("utf-8"))["request"]["request_id"]
                    == pending.request_id
                )
            data = json.dumps(submission(pending)).encode("utf-8")
            refused = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}",
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(refused)
            except urllib.error.HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("a POST without origin and token was accepted")
            wrong_origin = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}",
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "http://example.invalid",
                    "X-Interaction-Token": httpd.control_token,
                },
            )
            try:
                urllib.request.urlopen(wrong_origin)
            except urllib.error.HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("a cross-origin POST was accepted")
            oversized = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}",
                data=b" " * (view.MAX_INTERACTION_BODY + 1),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Origin": httpd.origin,
                    "X-Interaction-Token": httpd.control_token,
                },
            )
            try:
                urllib.request.urlopen(oversized)
            except urllib.error.HTTPError as error:
                assert error.code == 413
            else:
                raise AssertionError("an oversized POST was accepted")
            accepted = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}",
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Origin": httpd.origin,
                    "X-Interaction-Token": httpd.control_token,
                },
            )
            with urllib.request.urlopen(accepted) as response:
                assert response.status == 201
        finally:
            httpd.shutdown()
            httpd.server_close()


def posted(base: str, path: str, body: bytes, headers: dict[str, str]) -> int:
    """The status one POST against the served API is answered with, refusals included."""
    request = urllib.request.Request(
        base + path, data=body, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def check_view_interaction_route_answers_each_refusal_with_its_own_status():
    """The page reads the status to tell a player whether to fix the turn, refresh or retry."""
    with temp_root() as root:
        episode_once()
        store = InteractionStore(root / "interactions")
        old = store.publish("t", "1", 2, 1, "system", "observation", TOOLS)
        pending = store.publish(
            "t", "1", 2, 2, "system", (ToolResult("call-1", "ok"),), TOOLS
        )
        httpd = view.serve(0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        allowed = {
            "Content-Type": "application/json",
            "Origin": httpd.origin,
            "X-Interaction-Token": httpd.control_token,
        }

        def to(path: str, value: Any, *without: str, **headers: str) -> int:
            sent = {k: v for k, v in {**allowed, **headers}.items() if k not in without}
            body = (
                value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            )
            return posted(base, "/api/interaction/" + path, body, sent)

        answer = f"t/{pending.request_id}"
        try:
            statuses = {
                "a route with no request id": (to("t", submission(pending)), 404),
                "an agent the view does not know": (
                    to(f"nobody/{pending.request_id}", submission(pending)),
                    404,
                ),
                "the token without an origin": (
                    to(answer, submission(pending), "Origin"),
                    403,
                ),
                "the origin without a token": (
                    to(answer, submission(pending), "X-Interaction-Token"),
                    403,
                ),
                "a body that is not JSON by its type": (
                    to(answer, submission(pending), **{"Content-Type": "text/plain"}),
                    415,
                ),
                "a body that does not parse": (to(answer, b"{not json"), 400),
                "a body that is not an object": (
                    to(answer, [submission(pending)]),
                    400,
                ),
                "an envelope without a submission_id": (
                    to(
                        answer,
                        {
                            k: v
                            for k, v in submission(pending).items()
                            if k != "submission_id"
                        },
                    ),
                    400,
                ),
                "a request a newer one cancelled": (
                    to(f"t/{old.request_id}", submission(old)),
                    409,
                ),
                "a request id that is not a file name": (
                    to("t/a%2Fb", submission(pending)),
                    404,
                ),
                "the first submission": (to(answer, submission(pending)), 201),
                "the same submission again": (to(answer, submission(pending)), 201),
                "a different submission after it": (
                    to(
                        answer,
                        {**submission(pending, "other"), "submission_id": "other"},
                    ),
                    409,
                ),
                "the winner's submission_id on other calls": (
                    to(answer, submission(pending, "other")),
                    409,
                ),
                "the winner's submission_id on another action": (
                    to(
                        answer,
                        {
                            **end_submission(pending),
                            "submission_id": submission(pending)["submission_id"],
                        },
                    ),
                    409,
                ),
            }
            unread = store.publish("t", "1", 3, 1, "system", "observation", TOOLS)
            path = submission_file(root, "t", unread.request_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")
            statuses["a submission on disk that does not read"] = (
                to(f"t/{unread.request_id}", submission(unread)),
                500,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
    wrong = {
        case: f"{got}, not {want}"
        for case, (got, want) in statuses.items()
        if got != want
    }
    assert not wrong, wrong


def check_the_view_reports_a_pending_pointer_that_does_not_read_and_serves_the_rest():
    """The seat's own route says the file is broken; the list and the header go on without it."""
    with temp_root() as root:
        episode_once()
        (root / "interactions" / "pending").mkdir(parents=True)
        (root / "interactions" / "pending" / "t.json").write_text(
            "[]", encoding="utf-8"
        )
        httpd = view.serve(0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            try:
                urllib.request.urlopen(base + "/api/interaction/t")
            except urllib.error.HTTPError as error:
                status, said = (
                    error.code,
                    json.loads(error.read().decode("utf-8"))["error"],
                )
            else:
                raise AssertionError(
                    "an unreadable pending pointer was served as a request"
                )
            with urllib.request.urlopen(base + "/api/interaction") as response:
                listed = json.loads(response.read().decode("utf-8"))["requests"]
            with urllib.request.urlopen(base + "/api/experiment/t") as response:
                seats = json.loads(response.read().decode("utf-8"))["seats"]
        finally:
            httpd.shutdown()
            httpd.server_close()
    assert status == 500 and "t.json" in said, (status, said)
    assert listed == [], listed
    assert [seat["pending_human"] for seat in seats] == [None], seats


def check_the_view_handler_answers_only_for_the_server_serve_builds():
    """A server without the token and the origin has nothing a POST's headers could match,
    so the handler refuses to read them off it rather than letting absent equal absent."""
    httpd = view.serve(0)
    try:
        assert isinstance(httpd, view.ViewServer), type(httpd)
        assert httpd.origin == f"http://127.0.0.1:{httpd.server_address[1]}"
        assert httpd.control_token and httpd.focus is None
    finally:
        httpd.server_close()
    handler = object.__new__(view.View)
    handler.server = object.__new__(http.server.ThreadingHTTPServer)
    raised(
        TypeError,
        lambda: handler.view_server,
        "a handler read its token and origin off a server that has neither",
    )


def check_the_human_cli_answers_the_store_the_harness_publishes_to():
    """With no --root the terminal reads the repository's interactions/, the one the
    harness publishes to under its own root. The interaction package does not import
    harness, so it spells that path itself, and the two are held alike here."""
    ran = []
    original = cli.run
    cli.run = lambda agent, root: ran.append((agent, root)) or 0
    try:
        assert cli.main(["--agent", "a"]) == 0
    finally:
        cli.run = original
    with harness.using(harness.Settings()):
        expected = harness.interactions_root()
    assert ran == [("a", expected)], (ran, expected)


def check_human_py_starts_in_a_fresh_interpreter():
    """interaction is imported before providers there, the order that exposes an import cycle."""
    script = Path(__file__).resolve().parent.parent / "human.py"
    done = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "--agent" in done.stdout
