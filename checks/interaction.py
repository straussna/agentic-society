"""Interactive provider, durable coordination, and browser control routes."""

from __future__ import annotations

import json
import subprocess
import sys
import builtins
import contextlib
import io
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import harness
import view
from interaction import InteractionConflict, InteractionStore, InvalidSubmission, StaleRequest
from interaction import cli
from providers import SessionContext, ToolResult, ToolSpec, Usage
from providers.human import HumanProvider

from checks.lanes import episode_once, temp_root


TOOLS = (ToolSpec("remember", "Keep private memory", {
    "type": "object", "properties": {"body": {"type": "string"}},
    "required": ["body"], "additionalProperties": False}),)


def submission(request, body="hello"):
    return {"version": 1, "request_id": request.request_id, "submission_id": "human-call-1",
            "action": "tool_calls",
            "tool_calls": [{"id": "call-1", "name": "remember", "input": {"body": body}}]}


def end_submission(request):
    return {"version": 1, "request_id": request.request_id, "submission_id": "human-end-1",
            "action": "end_turn", "tool_calls": []}


def check_interaction_store_preserves_requests_and_first_submission():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "Seat A", 2, 3, "system", "observation", TOOLS)
        on_disk = store.current("a")
        assert on_disk == request
        assert on_disk.available_tools == TOOLS, "declared tools retain their order and exact schemas"
        undeclared = submission(request)
        undeclared["tool_calls"][0]["name"] = "not_offered"
        try:
            store.submit("a", request.request_id, undeclared)
        except InvalidSubmission:
            pass
        else:
            raise AssertionError("an undeclared tool entered the interaction store")
        won = store.submit("a", request.request_id, submission(request))
        assert store.submit("a", request.request_id, submission(request)) == won, \
            "a repeated submission id returns the winner"
        try:
            store.submit("a", request.request_id,
                         {**submission(request, "different"), "submission_id": "other"})
        except InteractionConflict:
            pass
        else:
            raise AssertionError("a conflicting client cannot replace the winning submission")
        assert store.wait(request, lambda: False, 0).tool_calls[0].input == {"body": "hello"}
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


def check_human_cli_drafts_and_submits_declared_calls():
    with temp_root() as root:
        store = InteractionStore(root / "interactions")
        request = store.publish("a", "A", 1, 1, "", "observation", TOOLS)
        commands = iter(['call remember {"body":"from terminal"}', "draft", "submit", "quit"])
        original = builtins.input
        try:
            builtins.input = lambda prompt="": next(commands)
            with contextlib.redirect_stdout(io.StringIO()):
                assert cli.run("a", root / "interactions") == 0
        finally:
            builtins.input = original
        submitted = store.wait(request, lambda: False, 0)
        assert submitted.tool_calls[0].input == {"body": "from terminal"}


def check_human_provider_returns_canonical_zero_charge_turn():
    with temp_root() as root:
        session = HumanProvider().open_session(
            "interactive", "system", TOOLS, 100,
            SessionContext("a", "Seat A", 4, root / "interactions"))
        result = {}

        def request():
            result["pending"] = session.request("observation")

        thread = threading.Thread(target=request)
        thread.start()
        store = InteractionStore(root / "interactions")
        for _ in range(100):
            if pending := store.current("a"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("human request was not published")
        store.submit("a", pending.request_id, submission(pending))
        thread.join(2)
        assert not thread.is_alive()
        pending_response = result["pending"]
        assert pending_response.native["request_id"] == pending.request_id
        turn = pending_response.normalize()
        assert turn.provider == "human" and turn.stop_reason == "tool_use"
        assert turn.tool_calls[0].as_dict() == submission(pending)["tool_calls"][0]
        assert turn.usage == Usage() and turn.charges == ()

        finished = {}
        second = threading.Thread(target=lambda: finished.update(
            pending=session.request((ToolResult("call-1", "stored"),))))
        second.start()
        for _ in range(100):
            if next_request := store.current("a"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("follow-up request was not published")
        assert next_request.input["kind"] == "tool_results"
        store.submit("a", next_request.request_id, end_submission(next_request))
        second.join(2)
        ended = finished["pending"].normalize()
        assert ended.stop_reason == "end_turn" and ended.tool_calls == ()

        stopping = threading.Event()
        cancelled_session = HumanProvider().open_session(
            "interactive", "system", TOOLS, 100,
            SessionContext("b", "Seat B", 1, root / "interactions", stopping.is_set))
        cancelled = {}

        def wait_until_cancelled():
            try:
                cancelled_session.request("observation")
            except BaseException as error:
                cancelled["error"] = error

        waiting = threading.Thread(target=wait_until_cancelled)
        waiting.start()
        for _ in range(100):
            if cancelled_request := store.current("b"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("cancellable request was not published")
        stopping.set()
        waiting.join(2)
        assert isinstance(cancelled["error"], KeyboardInterrupt)
        assert store.request("b", cancelled_request.request_id).status == "cancelled"


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
                assert json.loads(response.read().decode("utf-8"))["request"]["request_id"] == pending.request_id
            data = json.dumps(submission(pending)).encode("utf-8")
            refused = urllib.request.Request(base + f"/api/interaction/t/{pending.request_id}",
                                             data=data, method="POST",
                                             headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(refused)
            except urllib.error.HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("a POST without origin and token was accepted")
            wrong_origin = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}", data=data, method="POST",
                headers={"Content-Type": "application/json", "Origin": "http://example.invalid",
                         "X-Interaction-Token": httpd.control_token})
            try:
                urllib.request.urlopen(wrong_origin)
            except urllib.error.HTTPError as error:
                assert error.code == 403
            else:
                raise AssertionError("a cross-origin POST was accepted")
            oversized = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}",
                data=b" " * (view.MAX_INTERACTION_BODY + 1), method="POST",
                headers={"Content-Type": "application/json", "Origin": httpd.origin,
                         "X-Interaction-Token": httpd.control_token})
            try:
                urllib.request.urlopen(oversized)
            except urllib.error.HTTPError as error:
                assert error.code == 413
            else:
                raise AssertionError("an oversized POST was accepted")
            accepted = urllib.request.Request(
                base + f"/api/interaction/t/{pending.request_id}", data=data, method="POST",
                headers={"Content-Type": "application/json", "Origin": httpd.origin,
                         "X-Interaction-Token": httpd.control_token})
            with urllib.request.urlopen(accepted) as response:
                assert response.status == 201
        finally:
            httpd.shutdown()
            httpd.server_close()


def check_human_py_starts_in_a_fresh_interpreter():
    """interaction is imported before providers there, the order that exposes an import cycle."""
    script = Path(__file__).resolve().parent.parent / "human.py"
    done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "--agent" in done.stdout
