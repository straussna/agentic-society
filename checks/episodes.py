"""How an episode starts, stops, and is recorded.

The loop over episodes, the signals, the stop reasons, the model call every turn
makes, refusals, the raw log, provenance, and the CLI paths that start nothing."""

from __future__ import annotations

from types import SimpleNamespace as NS
from pathlib import Path
import hashlib
import inspect
import json
import signal
import sys
import analyze
import harness
import providers
import view

from checks.fake import (
    DEFAULT,
    Err,
    fake,
    interrupting_at,
    refuse,
    run,
    say,
    think,
    usage,
)
from checks.lanes import (
    HostShell,
    NoSandbox,
    amend,
    differs,
    digest_name,
    episode_once,
    ground_truth,
    never_start,
    pinned,
    plant,
    quiet,
    rejected,
    rooted,
    seated,
    temp_root,
    turn_cost,
    without_listing_drift,
)


def check_episodes_are_a_ceiling_not_a_floor():
    """run_episodes(N) runs N episodes, or fewer if the budget ends it first."""
    cost = turn_cost()
    with temp_root():
        with quiet() as buf:
            assert harness.run_episodes("t", fake(), 2) == 0
        account = ground_truth()
        assert [s["episode"] for s in account["episodes"]] == [1, 2], account[
            "episodes"
        ]
        assert len(account["balance_history"]) == 1 + sum(
            s["turn_count"] for s in account["episodes"]
        )
        assert buf.getvalue().count("created agent") == 1, (
            "the agent is created once, not per episode"
        )

    # Budget for three episodes, asked for eight: the account decides.
    with temp_root(budget=cost * 3):
        with quiet() as buf:
            assert harness.run_episodes("t", fake(), 8) == 0
        account = ground_truth()
        assert 0 < len(account["episodes"]) < 8, account["episodes"]
        assert (
            account["balance"] <= 0 and "nothing left to spend" in buf.getvalue()
        ), buf.getvalue()


def check_a_turn_that_leaves_exactly_nothing_is_the_last():
    """The floor is zero, and a balance at it has nothing left to spend: the turn that
    brings it there exactly ends the episode, and the next model call is never sent."""
    with temp_root(budget=turn_cost()):
        t = episode_once(run("echo one"), run("echo two"), say())

    assert t["balance"] == 0, t["balance"]
    assert (t["stop"], len(t["turns"])) == ("budget_exhausted", 1), (
        t["stop"],
        len(t["turns"]),
    )


def check_a_fault_ends_the_loop():
    """An interrupted or failed episode stops the loop, and is not retried.

    An episode that merely finished, however it finished, is not a fault.
    """
    for fault, stop in ((KeyboardInterrupt(), "interrupted"), (Err(400), "api_error")):
        with temp_root():
            with quiet() as buf:
                assert harness.run_episodes("t", fake(run("echo one"), fault), 5) == 0
            account = ground_truth()
            assert len(account["episodes"]) == 1, (
                "the loop must not run a second episode"
            )
            assert account["episodes"][0]["stop"] == stop, account["episodes"]
            assert "stopping after 1 of 5" in buf.getvalue(), buf.getvalue()
    assert harness.STOPS_THE_AGENT.isdisjoint(
        {
            "end_turn",
            "budget_exhausted",
            "context_threshold",
            "max_turns",
            "max_tokens",
            "no_tool_call",
            "refusal",
        }
    ), harness.STOPS_THE_AGENT
    assert harness.STOPS_THE_EXPERIMENT <= harness.STOPS_THE_AGENT, (
        "what ends the experiment ends the agent whose episode it landed in"
    )


def check_episodes_flag_is_validated():
    """--episodes below one is rejected before anything is created or spent."""
    with rooted(NoSandbox):
        harness.start = never_start
        for bad in ("0", "-1"):
            with quiet():
                rejected(
                    lambda: harness.main(["--agent", "t", "--episodes", bad]),
                    code=2,
                    because=f"accepted --episodes {bad}",
                )


def check_a_long_option_is_never_abbreviated():
    """--at is not --at-episode: a long option is matched whole or not at all."""
    with temp_root():
        harness.start = never_start
        episode_once(say())
        with quiet() as buf:
            rejected(
                lambda: harness.main(
                    ["--agent", "f", "--fork-from", "t", "--at", "1"]
                ),
                code=2,
                because="--at was taken for --at-episode",
            )
    assert "unrecognized arguments: --at 1" in buf.getvalue(), buf.getvalue()


def check_print_system_and_print_files_audit_without_starting():
    """The static audits and --fork-from run and stop before start() is reached.

    None calls start(), opens a provider or builds a real environment. A fork without
    an episode to fork at is rejected by the parser. --print-system reads config.toml
    and installs none of it.
    """
    with temp_root() as root:
        harness.start = never_start
        (root / "config.toml").write_text("max_turns = 300\n", encoding="utf-8")
        before = harness.SETTINGS
        with quiet() as buf:
            assert harness.main(["--print-system"]) == 0
        assert f"config: {root / 'config.toml'}" in buf.getvalue(), buf.getvalue()
        assert harness.SETTINGS is before, (
            "an audit leaves the settings as it found them"
        )
        for name, _, sha256 in harness.SHIPPED_TEXT:
            assert name in buf.getvalue() and sha256 in buf.getvalue(), (
                name,
                buf.getvalue(),
            )
        plant(root)
        with quiet() as buf:
            assert harness.main(["--print-files", "s"]) == 0
        assert (
            "2 files" in buf.getvalue() and harness.files_sha256("s") in buf.getvalue()
        ), buf.getvalue()
        with quiet() as buf:
            assert harness.main(["--print-files", "nope"]) == 2
        assert str(harness.SETTINGS.root / "files") in buf.getvalue(), buf.getvalue()
        with quiet():
            rejected(
                lambda: harness.main(["--print-context"]),
                code=2,
                because="--print-context without a manifest was accepted",
            )
        with quiet():
            rejected(
                lambda: harness.main(["--agent", "f", "--fork-from", "t"]),
                code=2,
                because="--fork-from without --at-episode was accepted",
            )
            rejected(
                lambda: harness.main(
                    ["--agent", "f", "--fork-from", "t", "--at-episode", "0"]
                ),
                code=2,
                because="--at-episode 0 was accepted",
            )
        episode_once(say())
        with quiet():
            forked = ["--agent", "f", "--fork-from", "t", "--at-episode", "1"]
            assert harness.main(forked) == 0
        assert ground_truth("f")["forked_from"] == {
            "agent": "t",
            "episode": 1,
            "modes": "restored",
        }


def check_interrupt_still_traces_and_commits():
    """Ctrl+C ends the episode with its spend recorded, not discarded."""
    with temp_root():
        t = episode_once(run("echo one"), KeyboardInterrupt())
        assert t["stop"] == "interrupted", t["stop"]
        assert t["spent"] > 0, (
            "spend before the interrupt must reach the balance history"
        )
        account = ground_truth()
        assert account["budget"] - account["balance"] == t["spent"]
        assert len(account["episodes"]) == 1, "the episode must appear in the record"
        assert t["balance_after"] == account["balance_history"], (
            "the trace carries what was committed"
        )


def check_an_account_never_lists_an_episode_whose_trace_did_not_land():
    """The trace is written whole before the account records its episode.

    A trace that fails to land leaves the account as it stood, naming no episode it
    has no record of, and leaves nothing cut short where a reader looks for traces.
    The next episode takes the episode number, and its responses are appended to that
    episode's raw log after the lost attempt's, which stay there as the one record of
    them.
    """
    with temp_root():
        real = harness.replace_file

        def no_room_for_the_trace(src, dest):
            if dest.parent.name == "traces":
                raise OSError("no space left on device")
            real(src, dest)

        harness.replace_file = no_room_for_the_trace
        try:
            episode_once(run("echo one"), say())
        except OSError:
            pass
        else:
            raise AssertionError("an episode whose trace did not land was committed")
        harness.replace_file = real
        assert ground_truth()["episodes"] == [], (
            "the account names an episode with no trace"
        )
        assert harness.trace_paths("t") == [], harness.trace_paths("t")
        t = episode_once(say())
        raw = [
            json.loads(line)
            for line in harness.raw_path("t", 1)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    assert t["episode"] == 1, (
        "the next episode takes the episode number the lost one never claimed"
    )
    turns = [line["turn"] for line in raw if line["line_type"] == "raw_response"]
    assert turns == [1, 2, 1], (
        f"the raw log holds both attempts at the episode, lost first: {turns}"
    )


def check_a_transfer_pays_no_receiver_from_an_episode_that_was_not_committed():
    """A transfer reaches its receiver once the giver's account has saved the episode,
    and not before. An episode whose trace does not land, or whose account does not
    save, has no debit, spend or record to show for it, and pays its receiver nothing
    either, so no receiver holds a credit that no giver's record accounts for.

    t declares 100 for seat 2, which is other's.
    """
    lost = {}
    for refused_at in ("trace", "account"):
        with temp_root() as root:
            seated(root, other={})
            before = ground_truth("other")
            real = harness.replace_file
            full = (
                harness.trace_path("t", 1)
                if refused_at == "trace"
                else harness.account_path("t")
            )

            def no_room(src, dest):
                if dest == full:
                    raise OSError("no space left on device")
                real(src, dest)

            harness.replace_file = no_room
            try:
                episode_once(run("echo '2 100' > outbox/transfer"), say())
            except OSError:
                pass
            else:
                raise AssertionError(
                    f"an episode whose {refused_at} did not save was committed"
                )
            harness.replace_file = real
            lost[refused_at] = (
                harness.trace_path("t", 1).exists(),
                ground_truth()["episodes"],
                before,
                ground_truth("other"),
            )
            paid = episode_once(run("echo '2 100' > outbox/transfer"), say())
            received = ground_truth("other")
    for refused_at, (traced, episodes, before, other) in lost.items():
        assert traced == (refused_at == "account") and episodes == [], (
            refused_at,
            episodes,
        )
        assert other == before, (
            f"the receiver was paid by an episode never committed: {refused_at}"
        )
    assert paid["transfer"]["amount"] == 100 and received["received"] == 100, received


def check_a_trace_its_account_never_listed_is_no_episode_to_any_reader():
    """A commit that stops between writing the trace and saving the account leaves a
    trace the account does not list. Every reader of the traces stops at the episodes
    the account lists: the analysis and the page read no episode there, and the page
    shows its episode number as one not yet committed, as the next episode with that
    number will overwrite it.
    """
    with temp_root():
        episode_once(say())
        real = harness.replace_file

        def no_room_for_the_account(src, dest):
            if dest == harness.account_path("t"):
                raise OSError("no space left on device")
            real(src, dest)

        harness.replace_file = no_room_for_the_account
        try:
            episode_once(say())
        except OSError:
            pass
        else:
            raise AssertionError("an episode whose account did not save was committed")
        harness.replace_file = real
        orphaned = harness.trace_path("t", 2).exists()
        analysed = [t["episode"] for t in analyze.load("t")["t"]]
        shown = [t["episode"] for t in view.traces_of("t")]
        latest = view.latest_trace("t")
        live = view.running_episode("t")
        watched = view.episode_view("t", 2)
    assert orphaned, "the trace landed and the account did not"
    assert analysed == shown == [1], (analysed, shown)
    assert latest is not None and latest["episode"] == 1, latest
    assert live == 2 and watched is not None and watched["source"] == "raw", (
        live,
        watched,
    )


def check_an_interrupt_ends_the_episode_at_the_turn_boundary():
    """An interrupt lands between turns: the turn in flight is whole and is charged.

    The same outcome Ctrl+C reaches by raising, reached by the flag instead,
    which is what makes the teardown after it safe.
    """
    with temp_root():
        with quiet():
            t = harness.run_once(
                "t",
                interrupting_at(
                    2, run("echo one"), run("echo two"), run("echo three"), say()
                ),
            )
        assert t["stop"] == "interrupted", t["stop"]
        assert len(t["turns"]) == 2, (
            f"the turn in flight must finish: {len(t['turns'])}"
        )
        assert t["commands"][-1] == "echo two", t["commands"]
        assert t["spent"] > 0 and t["trees_mirrored"], t
        account = ground_truth()
        assert account["budget"] - account["balance"] == t["spent"]
        assert len(account["episodes"]) == 1, "the episode must appear in the record"
        assert t["balance_after"] == account["balance_history"], (
            "the trace carries what was committed"
        )


def check_an_interrupt_between_episodes_builds_no_sandbox():
    """An interrupt that lands between episodes starts no sandbox at all."""
    with rooted(NoSandbox, INTERRUPTING=True):
        harness.load_account("t")  # the agent exists; what does not is an episode
        seen = []
        with quiet() as buf:
            assert harness.run_episodes("t", fake(*DEFAULT, seen=seen), 5) == 0
        assert seen == [], "and asks the API nothing"
        assert ground_truth()["episodes"] == [], "and records no episode"
        assert "stopping after 0 of 5" in buf.getvalue(), buf.getvalue()


def check_a_second_signal_is_the_default_again():
    """The first signal asks; the second is the ordinary hard stop.

    Outside a root, because pinned() stubs catch_signals for every check that
    uses one - so this puts the handlers and the flag back itself.
    """
    was = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        harness.INTERRUPTING = False
        harness.catch_signals()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler) and handler not in was.values(), handler
        with quiet() as buf:
            handler(signal.SIGINT, None)
        assert harness.INTERRUPTING, "the first signal sets the flag and raises nothing"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, (
            "the second must be the interpreter's own, or an experimenter who will not "
            "wait is stuck"
        )
        assert "stopping" in buf.getvalue().lower(), buf.getvalue()

        term = signal.getsignal(signal.SIGTERM)
        with quiet():
            term(signal.SIGTERM, None)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, signal.getsignal(
            signal.SIGTERM
        )
    finally:
        harness.INTERRUPTING = False
        for s, h in was.items():
            signal.signal(s, h)


def check_the_children_are_in_their_own_process_group():
    """Every docker command and every episode shell is detached from this one.

    A console Ctrl+C goes to the whole foreground group, so a shared group means
    the harness kills the docker client it is waiting on.
    """
    expected = "creationflags" if sys.platform == "win32" else "start_new_session"
    assert set(harness.DETACHED) == {expected}, harness.DETACHED

    source = inspect.getsource(harness)
    assert 'subprocess.run(["docker"' not in source, (
        "every docker command goes through harness.docker, which is where DETACHED is "
        "applied"
    )
    assert "**DETACHED" in inspect.getsource(harness.docker), inspect.getsource(
        harness.docker
    )

    # Both lanes' shells, without starting either.
    for cls in (harness.Shell, HostShell):
        probe = cls.__new__(cls)
        probe.sandbox = NS(work=Path("."))
        assert expected in probe.popen_kwargs(), (
            f"{cls.__name__}: {probe.popen_kwargs()}"
        )


def check_fatal_error_still_traces_and_commits():
    """A 400 aborts without retrying, but the spend before it still reaches the balance
    history."""
    with temp_root():
        t = episode_once(run("echo before"), Err(400))
        assert t["stop"] == "api_error" and t["retries"] == [], (
            "a 400 must not be retried"
        )
        assert t["spent"] > 0, (
            "spend before the failure must still reach the balance history"
        )
        account = ground_truth()
        assert account["budget"] - account["balance"] == t["spent"]

    # An episode that never got a turn spent nothing and adds nothing.
    with temp_root():
        t = episode_once(Err(400))
        assert t["turns"] == [] and t["spent"] == 0, t["spent"]
        assert t["balance_after"] == t["balance_before"], t["balance_after"]


def check_no_tool_call_on_turn_one():
    """Text with no tool call on turn one is a clean outcome, not an error, and still
    spends."""
    with temp_root():
        t = episode_once(say("Nothing to do."))
        assert t["stop"] == "no_tool_call" and t["error"] is None, t
        assert t["spent"] > 0 and t["balance_after"][-1] == t["balance"]


def check_stop_reasons():
    """end_turn, refusal, context_threshold, and max_turns each fire on the right
    condition."""
    with temp_root():
        t = episode_once(*DEFAULT)
        assert t["stop"] == "end_turn", t["stop"]
        # And every turn carries the API's own stop_reason, which is a different
        # thing from the episode's derived stop.
        assert [x["stop_reason"] for x in t["turns"]] == [
            "tool_use",
            "tool_use",
            "end_turn",
        ], t["turns"]
    with temp_root():
        big = usage(input_tokens=900_000)
        t = episode_once(run("echo hi", u=big), say())
        assert t["stop"] == "context_threshold", t["stop"]
    with temp_root(max_turns=1):
        t = episode_once(*DEFAULT)
        assert t["stop"] == "max_turns", t["stop"]
    # Safety classifiers can decline before the agent has done anything. One
    # refusal is a turn the episode carries on past; REFUSAL_TURNS running is
    # what it stops for.
    with temp_root(REFUSAL_TURNS=2):
        t = episode_once(refuse(), refuse())
        assert t["stop"] == "refusal", t["stop"]
    with temp_root(REFUSAL_TURNS=2):
        t = episode_once(run("echo hi"), refuse(), refuse())
        assert t["stop"] == "refusal", t["stop"]


def check_truncated_turn_is_not_a_clean_end():
    """A turn cut off at max_tokens is recorded as truncated, not as a clean end.

    Both shapes: text truncated mid-sentence, and a tool_use block truncated
    mid-JSON, which arrives with no command. Decided before the shell.
    """
    with temp_root():
        t = episode_once(run("echo hi"), say("half a sen", stop="max_tokens"), say())
        assert t["stop"] == "max_tokens", t["stop"]
        assert t["error"] is None, "truncation is an outcome, not a harness fault"
        assert t["spent"] > 0, "the truncated turn was still charged"

    with temp_root():
        t = episode_once(
            run("cd /tmp; export MARK=before"),
            run(None, stop="max_tokens"),  # truncated mid-JSON
            run("pwd", "echo [$MARK]"),
            say(),
        )
        assert t["stop"] == "max_tokens", t["stop"]
        assert len(t["turns"]) == 2, "the episode must stop at the truncated turn"
        assert t["turns"][1]["tools"] == [], "a truncated call must not be executed"
        assert t["commands"] == [
            harness.observation(),
            "cd /tmp; export MARK=before",
        ], t["commands"]


def check_reasoning_reaches_the_record():
    """Thinking blocks are recorded, and kept apart from spoken words."""
    with temp_root():
        t = episode_once(think("weighing it up", "here goes"), say())
    assert t["turns"][0]["reasoning"] == "weighing it up", t["turns"][0]
    assert t["turns"][0]["text"] == "here goes", (
        "reasoning stays apart from spoken words"
    )


def check_reaching_for_balance_is_not_reading_it():
    """touched_balance is a command naming the balance file; read_balance is its
    contents in a result."""
    with temp_root():
        missed = episode_once(run("cat balance-"), say())  # wrong path: fails
        hit = episode_once(run("cat balance-1"), say())  # right path: succeeds
    assert missed["touched_balance"] and not missed["read_balance"], (
        "a failed read is not a read"
    )
    assert hit["touched_balance"] and hit["read_balance"], "a successful read is both"
    # Under live_balance a read comes back as the committed balance history plus the
    # balance so far, so the committed array is a prefix of what the agent saw.
    got = json.loads(hit["turns"][0]["tools"][0]["result"])
    assert got[: len(hit["balance_before"])] == hit["balance_before"], got

    with temp_root():
        # Names the balance file, never sees it.
        blind = episode_once(run("wc -c balance-1"), say())
    assert blind["touched_balance"] and not blind["read_balance"], (
        "a size check is not a read"
    )


def check_read_balance_survives_a_clipped_read():
    """A read of an n too big for the tool bound is still a read.

    Past roughly a thousand turns n outgrows tool_result_limit. clip() keeps a
    fixed head and n is append-only, so matching those leading bytes is exact.
    """
    with temp_root():
        with quiet():
            harness.load_account("t")  # create the agent, then enlarge it
        m = ground_truth()
        m["balance_history"] = list(range(1_000_000, 1_000_000 + 2_000))
        m["balance"] = m["balance_history"][-1]
        harness.save_account("t", m)
        t = episode_once(run("cat balance-1"), say())

    result = t["turns"][0]["tools"][0]["result"]
    assert "[truncated:" in result, "the read has to have actually been clipped"
    assert len(result) < harness.SETTINGS.tool_result_limit + 500, len(result)
    assert (
        t["balance_bytes"] > harness.SETTINGS.tool_result_limit
        and t["balance_fits"] is False
    ), t["balance_bytes"]
    assert t["read_balance"], "a clipped read of the balance file is still a read"
    assert t["touched_balance"]


def check_balance_in_prose_is_scored_as_a_path():
    """Writing about the balance file means naming it as a path, not using its bare
    name as a word."""
    with temp_root():
        maths = episode_once(
            run(
                "printf 'fit balance-1 = a*exp(-c*t^2) for t in 1..5\\n' "
                "> memory/notes.md"
            ),
            say(),
        )
    assert not maths["mentions"]["balance_path"], (
        "the bare name as a word is not writing about the file"
    )

    with temp_root():
        named = episode_once(
            run(
                "printf 'the series lives in ./balance-1, append-only\\n' "
                "> memory/notes.md"
            ),
            say(),
        )
    assert named["mentions"]["balance_path"], "the file named is writing about it"


def check_provenance_is_recorded():
    """Each trace states what decided the episode, and says when that changed.

    budget and model are pinned in account.json; everything else here is read at
    each episode, so only the trace can say what an episode actually ran as.
    """
    with temp_root():
        with quiet():
            first = harness.run_once("t", fake(*DEFAULT))
            prov = first["provenance"]
            for key in (
                "started_at",
                "harness_sha256",
                "image",
                "image_id",
                "provider",
                "message_lifetime",
                "requested_model",
                "context_fraction",
                "max_tokens",
                "max_turns",
                "command_timeout",
                "tool_result_limit",
                # What the initial observation carried, and how much of each board
                # reached it. An agent either side of a change to either
                # opened on a different environment.
                "digest_file_limit",
                "observation_limit",
                "live_balance",
                # The five the starter files state in words. An agent either side of
                # a change to any of them was told something else, so all
                # five have to reach config_drift() and not just the ones that
                # were here first.
                "grace_episodes",
                # What a transfer does to the giver, what the agent was given,
                # and what the whole experiment was given.
                "starter_files",
                "starter_files_sha256",
                "starter_files_below",
                "labels",
                "channels",
                "channels_sha256",
                "status_files",
                "source_sha256",
                "delivery",
            ):
                assert key in prov, f"provenance omits {key}"
            assert first["trace_version"] == harness.TRACE_VERSION, (
                "the record says which shape it is"
            )
            assert prov["provider"]["name"] == "anthropic", "the adapter is recorded"
            assert first["resolved_model"], "the dated snapshot behind the alias"
            assert first["config_drift"] == [], (
                "nothing to differ from on episode one"
            )

            # A rate change between episodes makes early and late entries of the
            # same balance history mean different things, so the drift is recorded.
            amend(context_fraction=0.5)
            second = harness.run_once("t", fake(*DEFAULT))
    assert any(d.startswith("context_fraction:") for d in second["config_drift"]), (
        "a provider-affecting change between episodes must be recorded"
    )


def check_watch_is_quiet_and_display_only():
    """--watch echoes the account and the agent's words, never commands or output;
    what each model call sends and the trace are identical."""
    quiet_seen, loud_seen = [], []
    with temp_root():
        with quiet():
            plain = harness.run_once("t", fake(*DEFAULT, seen=quiet_seen))
    with temp_root(WATCH=True):
        with quiet() as buf:
            loud = harness.run_once("t", fake(*DEFAULT, seen=loud_seen))
        shown = buf.getvalue()

    assert "=== episode 1 ===" in shown, "the episode number heads the episode"
    assert "turn 1" in shown and "context" in shown, (
        "per-turn account and context are shown"
    )
    assert "done." in shown, "the agent's words are shown"
    # Every line an episode produces leads with the agent it belongs to, so an
    # experiment's interleaved output stays attributable. "created agent" comes
    # from the account and not from the episode, and names the agent mid-line.
    lines = [
        l for l in shown.splitlines() if l.strip() and not l.startswith("created agent")
    ]
    assert lines and all(l.startswith("t") for l in lines), (
        "every line names the agent it came from: "
        f"{[l for l in lines if not l.startswith('t')]}"
    )
    assert any(l.startswith("t| ") for l in lines), "the watched lines carry the prefix"
    assert any(l.startswith("t ") and "end_turn" in l for l in lines), (
        "and so does the episode summary"
    )
    assert "$ " not in shown, "commands are not shown"
    assert "echo hi > memory/note.txt" not in shown, "commands are not shown"
    assert harness.observation() not in shown, (
        "the initial observation command is not shown"
    )
    q, l = without_listing_drift(quiet_seen), without_listing_drift(loud_seen)
    assert q == l, "watching must not change what is sent to the model: " + differs(
        q, l
    )
    for t in (plain, loud):
        # Wall clock, not the record: these differ between any two agents.
        t.pop("duration_s")
        t["provenance"].pop("started_at")
    # The observation carries the listing, whose mtimes are the minute the
    # environment was built. Everything else must agree exactly.
    assert without_listing_drift(plain.pop("observation")) == without_listing_drift(
        loud.pop("observation")
    )
    assert plain == loud, "watching must not change the record: " + differs(plain, loud)


def check_run_once_is_build_then_run_then_commit():
    """run_once is the steps composed: a driver composing them itself gets the same
    record.

    Same trace, same accounts, same console, so a simultaneous round that holds the
    steps apart commits exactly what an episode run on its own would have.
    """
    script = (
        run(
            "echo hi > memory/NOTES.md",
            "echo '2 100' > outbox/transfer",
            "echo yo > outbox/2",
            "echo p > 1/post",
            f"cat {digest_name()}",
        ),
        say(),
    )
    neighbour = {"group/msg": "theirs\n", "outbox/1": "for you\n"}

    def normal(t: dict) -> dict:
        t = json.loads(json.dumps(t))
        t.pop("duration_s")
        t["provenance"].pop("started_at")
        return without_listing_drift(t)

    with temp_root() as root:
        seated(root, other=neighbour)
        with quiet() as one:
            t1 = harness.run_once("t", fake(*script))
        m1, o1 = ground_truth(), ground_truth("other")
    with temp_root() as root:
        seated(root, other=neighbour)
        with quiet() as two:
            ep = harness.build_episode("t")
            assert ep.sandbox is not None and ep.shell is not None, (
                "built means a sandbox is up"
            )
            out = harness.run_episode(ep, fake(*script))
            assert ep.trees_mirrored, "run_episode mirrors the environment back"
            t2 = harness.close_episode(ep, out, harness.settle_episode(ep, out))
        m2, o2 = ground_truth(), ground_truth("other")

    assert normal(t1) == normal(t2), "the same record, step by step or in one call"
    for m in (m1, m2):
        m.pop("created_at")
    assert m1 == m2 and o1["received"] == o2["received"] == 100, (m1, m2, o1, o2)
    assert one.getvalue() == two.getvalue(), (one.getvalue(), two.getvalue())
    assert t1["transfer"]["amount"] == 100 and t1["channels"]["board"]["posted"], (
        t1
    )


def check_the_harness_sha256_is_read_once():
    """provenance() reports the code that is running, not the file on disk.

    The process has already imported this module, so a later edit to harness.py
    must not change what an episode records having run.
    """
    with pinned():
        amend(root=Path(harness.__file__).parent)
        prov = harness.provenance("anthropic", "claude-sonnet-5")
    assert prov["harness_sha256"] == harness.HARNESS_SHA256
    assert (
        harness.HARNESS_SHA256
        == hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest()
    ), "and it is this file's sha256"
    assert "read_bytes" not in inspect.getsource(harness.provenance), (
        "provenance must not re-read the harness from disk"
    )


def check_a_refusal_records_why():
    """Canonical refusal details are captured and absent on every other stop.

    A classifier declining and the model itself declining both arrive as
    stop_reason "refusal", and the category is what separates them.
    """
    with temp_root(REFUSAL_TURNS=2):
        t = episode_once(
            run("echo hi"), refuse(), refuse(recommended_model="claude-sonnet-5"), say()
        )

    refused = t["turns"][1]
    assert refused["stop_reason"] == "refusal"
    assert refused["refusal"]["refusal_type"] == "refusal"
    assert refused["refusal"]["details"]["category"] == "cyber"
    assert refused["refusal"]["explanation"] == "declined"
    assert t["turns"][0]["refusal"] is None, "absent on every other stop reason"
    assert t["turns"][2]["refusal"]["recommended_model"] == "claude-sonnet-5", t[
        "turns"
    ][2]["refusal"]
    assert t["refused_turns"] == 2, "counted whether or not they ended the episode"


def check_the_refusal_notice_is_pinned():
    """The notice is pinned, and says only what happened and what it left alone.

    It is the second thing the harness says, so it is held to what invariant 2 holds the
    first to: no cause, no instruction, and nothing addressed to the agent.
    """
    sha256 = hashlib.sha256(harness.REFUSAL_NOTICE.encode()).hexdigest()
    assert sha256 == harness.REFUSAL_NOTICE_SHA256, sha256
    assert (
        "REFUSAL_NOTICE",
        harness.REFUSAL_NOTICE,
        harness.REFUSAL_NOTICE_SHA256,
    ) in harness.SHIPPED_TEXT, (
        "start() rejects a run on what --print-system audits, so both read SHIPPED_TEXT"
    )
    low = harness.REFUSAL_NOTICE.lower()
    for word in (
        "polic",
        "safet",
        "classif",
        "anthropic",
        "cyber",
        "block",
        "you ",
        "your",
        "try",
        "instead",
        "again",
    ):
        assert word not in low, f"REFUSAL_NOTICE contains {word!r}"


def check_a_refusal_does_not_run_its_command():
    """A refused turn's tool call is not executed.

    A refusal can arrive with a call already emitted and cut mid-JSON, so what
    it would execute is not what the agent wrote.
    """
    with temp_root():
        t = episode_once(refuse("echo poison > memory/poison.txt"), say())

    assert "echo poison > memory/poison.txt" not in t["commands"], t["commands"]
    assert not [f for f in t["files"] if f["path"] == "poison.txt"], (
        "memory/ is untouched"
    )
    assert t["turns"][0]["tools"] == [], "no result recorded, because nothing ran"


def check_a_refusal_notice_reaches_the_agent():
    """The notice stands in for the results the refused turn would have had.

    Reached only where the cap lets an episode carry on past a refusal, so
    REFUSAL_TURNS is raised here. `seen` is the whole `messages` list at the end.
    """
    seen = []
    with temp_root(REFUSAL_TURNS=2):
        episode_once(refuse("cat balance-1"), say(), seen=seen)
    # A refusal carrying a call leaves a tool_use the next send must answer.
    sent = [x["input"] for x in seen if x["call"] == "send"][-1]
    blocks = [b for b in sent if b.content == harness.REFUSAL_NOTICE]
    assert len(blocks) == 1 and blocks[0].is_error is True, blocks

    seen = []
    with temp_root(REFUSAL_TURNS=2):
        episode_once(refuse(), say(), seen=seen)
    # A refusal with no content has no call to answer, and no words to replay.
    sent = [x["input"] for x in seen if x["call"] == "send"][-1]
    assert sent == harness.REFUSAL_NOTICE, sent


def check_an_unhandled_stop_reason_is_named():
    """A stop reason the loop has no branch for ends the episode saying so.

    Read as the absence of tool calls it would be filed as end_turn, which says
    the agent chose to stop when in fact the harness did not know how to go on.
    """
    with temp_root():
        t = episode_once(say(stop="pause_turn"), say())
    assert t["stop"] == "unhandled:pause_turn", t["stop"]


def check_every_response_is_logged_raw():
    """Each response is appended verbatim, before anything else reads it."""
    with temp_root():
        t = episode_once(run("echo hi"), refuse(), say())
        log = harness.records_dir("t") / "raw" / f"episode-{t['episode']:04d}.jsonl"
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]

    assert [x["line_type"] for x in lines] == [
        "raw_response",
        "normalized_response",
    ] * 2, lines
    assert all(
        x["raw_response"]["id"] for x in lines if x["line_type"] == "raw_response"
    )
    assert all(x["provider"] == "anthropic" for x in lines)
    refused = [x for x in lines if x["line_type"] == "normalized_response"][-1]
    assert refused["response"]["refusal"]["details"]["category"] == "cyber", refused


def check_the_raw_log_never_stops_an_episode():
    """A log that cannot be written is reported, and the episode goes on.

    The log is a record of the agent, not part of it. An episode that is spending
    money does not stop because a line could not be appended.
    """
    with temp_root():
        # A file where the agent's raw/ directory needs to be, so mkdir fails.
        (harness.records_dir("t")).mkdir(parents=True, exist_ok=True)
        (harness.records_dir("t") / "raw").write_text("in the way", encoding="utf-8")
        t = episode_once(run("echo hi"), say())
    assert t["stop"] == "end_turn", t["stop"]
    assert "echo hi" in t["commands"], "the episode ran despite the log failing"


def check_refusals_end_the_episode_at_the_cap():
    """REFUSAL_TURNS running end the episode, and it stops asking."""
    seen = []
    with temp_root(REFUSAL_TURNS=3):
        t = episode_once(refuse(), refuse(), refuse(), say(), seen=seen)

    assert t["stop"] == "refusal", t["stop"]
    assert t["refused_turns"] == 3, t["refused_turns"]
    sends = [x for x in seen if x["call"] == "send"]
    assert len(sends) == 3, f"asked {len(sends)} times past the cap"


def check_a_recovered_refusal_is_not_a_refused_episode():
    """Refusals an episode gets past are counted but do not name its stop.

    stuck() reads the episode stop, so an agent that acted must not look like one
    that never got to.
    """
    with temp_root(REFUSAL_TURNS=4):
        t = episode_once(refuse(), refuse(), run("echo hi > memory/note.txt"), say())

    assert t["stop"] == "end_turn", t["stop"]
    assert t["refused_turns"] == 2, t["refused_turns"]
    assert "echo hi > memory/note.txt" in t["commands"], "the episode went on to act"
    streak = [{"stop": t["stop"]}] * harness.REFUSAL_STREAK
    assert not harness.stuck({"episodes": streak}), (
        "a recovered episode breaks the streak"
    )


def check_a_stuck_agent_stops_itself():
    """An agent that refuses REFUSAL_STREAK episodes running is out of play.

    A refusal ends an episode before the agent writes anything, so the next episode
    opens on a near-identical context: the agent never acts, so it cannot escape.
    """
    streak = harness.REFUSAL_STREAK
    episodes = [
        {
            "episode": i,
            "stop": "refusal",
            "spent": 1,
            "turn_count": 1,
            "balance_at_start": 9,
        }
        for i in range(1, streak + 1)
    ]
    account = {"balance": 999_999, "episodes": episodes}

    assert harness.stuck(account), f"{streak} refusals running is stuck"
    assert not harness.in_play(account), (
        "and a stuck agent is out of play, whatever its balance"
    )
    assert not harness.stuck({**account, "episodes": episodes[:-1]}), (
        "one short is not stuck"
    )
    # A single success anywhere in the window clears it: the agent acted, so its
    # next episode opens on something it wrote and not on the same context.
    broken = [*episodes[:-1], {**episodes[-1], "stop": "end_turn"}]
    assert not harness.stuck({**account, "episodes": broken})
    assert harness.in_play({**account, "episodes": broken})
    # A runaway guard, not a productivity filter: a healthy agent can refuse a
    # few episodes running and recover.
    assert streak > 3, f"REFUSAL_STREAK={streak} would stop an agent that recovers"


def check_an_adapter_fault_or_a_harness_bug_ends_as_harness_error_and_still_commits():
    """What is not the API failing is harness_error, and the spend before it is settled.

    A provider failure classified configuration, unreadable_response or unclassified
    and an exception from anywhere else are all on this side of the line. None is
    retried, the error names its type, and only a provider failure carries the
    provider's classification, which a trace recording the category adapter reads
    as unclassified.
    """
    assert providers.HARNESS_FAULTS == {
        "configuration",
        "unreadable_response",
        "unclassified",
    }
    faults = (
        *(
            (
                providers.ProviderError("bad", category=category, provider="anthropic"),
                "ProviderError",
            )
            for category in sorted(providers.HARNESS_FAULTS)
        ),
        (ValueError("bad"), "ValueError"),
    )
    for fault, kind in faults:
        with temp_root():
            t = episode_once(run("echo before"), fault)
            account = ground_truth()
        assert t["stop"] == "harness_error", (kind, t["stop"])
        assert t["error"] == f"{kind}: bad" and t["retries"] == [], (
            t["error"],
            t["retries"],
        )
        assert account["episodes"][-1]["stop"] == "harness_error", account["episodes"]
        assert (
            t["spent"] > 0 and account["budget"] - account["balance"] == t["spent"]
        ), "the turn before the fault was charged and reached the balance history"
        if kind == "ProviderError":
            assert t["provider_error"] == {
                "category": fault.category,
                "provider": "anthropic",
                "message": "bad",
                "status_code": None,
                "native_type": None,
            }, t["provider_error"]
            earlier = {**t["provider_error"], "category": "adapter"}
            old = {**t, "provider_error": earlier}
            assert harness.trace_as_current(old)["provider_error"] == {
                **t["provider_error"],
                "category": "unclassified",
            }
        else:
            assert "provider_error" not in t, t["provider_error"]
