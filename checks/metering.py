"""What the harness says, what a turn costs, and what the config and start() admit.

The pinned prompt and tool, the rates and their expiry, the client's own guards,
and the arithmetic that makes every account reconcile."""

from __future__ import annotations

from types import SimpleNamespace as NS
from pathlib import Path
import hashlib
import json
import tempfile
import harness
import providers
from providers.anthropic import normalize as normalize_anthropic

from checks.fake import DEFAULT, Err, fake, refuse, run, say, usage
from checks.lanes import (
    HALF,
    amend,
    channel_toml,
    elements_of,
    episode_once,
    ground_truth,
    pinned,
    quiet,
    reconciled,
    rejected,
    seated,
    span_of,
    tables,
    temp_root,
    turn_cost,
)


def check_system_is_pinned():
    """Invariant 2: the harness ships no words, pinned at that, and the tool carries none either.

    What the harness says it computes from the accounts and rewrites when those move.
    A fixed line is a constant and constants are the experimenter's, so the shipped
    prompt is empty - and pinned empty, so an arm that adds words has to declare them.
    """
    assert harness.SYSTEM == "", f"the harness ships no words, got {harness.SYSTEM!r}"
    assert hashlib.sha256(harness.SYSTEM.encode()).hexdigest() == harness.SYSTEM_SHA256
    assert harness.SETTINGS.system_prompt == harness.SYSTEM, (
        "declaring nothing says nothing"
    )
    assert (
        harness.SHELL_SPEC.name == "bash"
        and harness.SHELL_SPEC.input_schema["additionalProperties"] is False
    )


def check_spend_is_exact():
    """Invariant 6: spend matches hand-computed integers, including both cache-write
    TTLs."""
    # opus-5: in 500, out 2500 centi/token; write 1.25x, read 0.1x
    u = usage(
        input_tokens=1000,
        output_tokens=100,
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=5000,
    )
    r = NS(
        id="x",
        model="claude-opus-5",
        stop_reason="end_turn",
        stop_details=None,
        content=[NS(type="text", text="x")],
        usage=u,
    )
    m = normalize_anthropic(r, "claude-opus-5")
    assert sum(c.centi_micros for c in m.charges) == 2_250_000, m
    assert m.usage.input_tokens == 1000 + 5000 + 2000, (
        "input_tokens must include cached tokens"
    )
    # per-TTL detail wins over the flat field; 1h writes cost 2x
    r.usage = usage(
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=9999,
        cache_creation=NS(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
    )
    split = normalize_anthropic(r, "claude-opus-5")
    assert sum(c.centi_micros for c in split.charges) == 100 * 625 + 200 * 1000, split


def check_balance_is_bare_integers_with_no_host_in_them():
    """Unlabelled: n is a JSON array of bare integers, byte-for-byte, LF on any host."""
    assert harness.render_balance([500000, 494750]) == "[500000,494750]\n"
    assert all(type(v) is int for v in json.loads(harness.render_balance([1, 2])))
    assert ":" not in harness.render_balance([1]) and '"' not in harness.render_balance(
        [1]
    )
    rendered = harness.render_balance([1_000_000, 996_989]).encode("utf-8")
    assert rendered == b"[1000000,996989]\n", rendered
    assert b"\r" not in rendered, "no carriage return reaches the agent"
    assert harness.balance_name("3") == "n3", "and a seat is what names one"


def check_config_is_validated():
    """The real config.toml is valid, and bad keys, types, and values are rejected.

    Unknown keys, wrong types, out-of-range values, and a --config path that
    does not exist all exit nonzero.
    """
    with pinned():
        try:
            s, _ = harness.load_config(harness.SETTINGS)  # the real file, if present
        except SystemExit as e:  # reported as a failure
            raise AssertionError(f"config.toml is invalid: {e}") from None
        assert (
            providers.model_spec("anthropic", "claude-sonnet-5").context_window
            == 1_000_000
        )
        assert 0 < s.context_fraction <= 1
        assert s.max_tokens <= harness.MAX_TOKENS_CEILING
        assert type(s.live_balance) is bool
        assert s.digest_file_limit >= harness.DIGEST_FILE_FLOOR
        assert s.observation_limit >= s.tool_result_limit

    def declared(**values):
        """One experiment setting, applied the way a manifest's defaults are."""
        harness.apply_config(
            values, "manifest", harness.MANIFEST_SETTINGS, harness.NOT_MANIFEST
        )

    with tempfile.TemporaryDirectory(prefix="mtr-cfg-") as tmp:
        f = Path(tmp) / "config.toml"
        # config.toml holds the process parameters and rejects everything else by name:
        # an unknown key, a wrong type, a value out of range, a key that became a channel
        # field, and every setting an experiment owns.
        for bad in (
            "turn_cpa = 5",
            'system = "hi"',
            'max_turns = "many"',
            f"max_tokens = {harness.MAX_TOKENS_CEILING + 1}",
            'transfer_funded_by = "harness"',
            'shared_files = "brief"',
            'model = "claude-sonnet-5"',
            "budget = 1",
            "context_fraction = 0.5",
            'system_prompt = "hi"',
            'delivery = "push"',
            'starter_files = "s"',
            channel_toml(tables()),
            "[status_files]" + chr(10) + 'balance = "n"',
            "[harness_files]" + chr(10) + 'balance = "n"',
        ):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                rejected(
                    lambda: harness.load_config(harness.SETTINGS, f),
                    because=f"accepted bad config: {bad}",
                )

        # A named file that is not there is rejected, not quietly skipped.
        with pinned():
            rejected(
                lambda: harness.load_config(
                    harness.SETTINGS, Path(tmp) / "confg.toml"
                ),
                "no such config",
                because="a missing --config path was ignored",
            )

        f.write_text(
            "max_turns = 300" + chr(10) + "command_timeout = 30", encoding="utf-8"
        )
        with pinned():
            s, used = harness.load_config(harness.SETTINGS, f)
            assert used == f, "the file used is reported back"
            assert s.max_turns == 300 and s.command_timeout == 30, (
                "a good value must actually apply"
            )

    # An experiment owns the rest, and its values are held to the same ranges.
    for bad in (
        {"model": "no-such-model"},
        {"context_fraction": 2.0},
        {"budget": 0},
        {"live_balance": "yes"},
        {"delivery": "fetch"},
        {"delivery": 1},
        {"digest_file_limit": harness.DIGEST_FILE_FLOOR - 1},
        # The initial observation carries the whole digest and is never smaller
        # than what one ordinary call may return.
        {"observation_limit": harness.SETTINGS.tool_result_limit - 1},
        # The process parameters are rejected here, saying where they live.
        {"max_turns": 7},
        {"image": "x"},
        {"tool_result_limit": 2000},
    ):
        with pinned():
            rejected(
                lambda: declared(**bad),
                "manifest",
                because=f"a manifest accepted {bad}",
            )
    with pinned():
        declared(context_fraction=1)
        assert harness.SETTINGS.context_fraction == 1.0, (
            "an int must widen into a float field"
        )
    # A rejection installs nothing, the good key given beside the rejected one included.
    with pinned():
        before = harness.SETTINGS
        rejected(
            lambda: declared(budget=7, delivery="fetch"),
            "delivery",
            because="a manifest accepted delivery = 'fetch'",
        )
        assert harness.SETTINGS is before, (
            "a rejected manifest left part of itself in force"
        )

    # A transfer is the giver's own budget moving; a rebate on top would mint. No
    # share for a transfer nobody can make.
    for table in (
        tables(transfer={"funded_by": "giver", "rebate_percent": 75}),
        tables(transfer={"funded_by": "loud"}),
        tables(transfer={"funded_by": 3}),
        tables(transfer={"funded_by": "none", "silence_penalty_percent": 50}),
        tables(transfer={"rebate_percent": 101}),
    ):
        with pinned():
            rejected(
                lambda: harness.apply_channels(table, None, "manifest"),
                "manifest:",
                because="a bad channel table was accepted",
            )
    with pinned():
        harness.apply_channels(
            tables(transfer={"funded_by": "giver", "rebate_percent": 0}),
            None,
            "manifest",
        )
        assert harness.channel("transfer").funded_by == "giver", (
            "the pairing the rule asks for is accepted"
        )


def check_truncation_and_empty():
    """Oversized output is clipped with an explicit marker, keeping head and tail."""
    c = harness.clip("x" * 50_000, 8_000)
    assert "[truncated: 42000 of 50000 characters]" in c
    assert c.startswith("x") and c.endswith("x") and len(c) < 8_500, (
        "head and tail both kept"
    )
    assert harness.clip("short", 8_000) == "short"


def check_episodes_reconcile():
    """Every micro-dollar in or out of an account is one element of its balance history.

    A transfer rebates, a missed post and a crowded seat are each taken, and a floor
    gives back. The identity has a term for each, and each appends to the balance
    history.
    """
    with temp_root():
        for _ in range(3):
            last = episode_once(*DEFAULT)
        account = ground_truth()
        history, episodes = account["balance_history"], account["episodes"]
        assert len(episodes) == 3, episodes
        assert len(history) == 1 + sum(s["turn_count"] for s in episodes), (
            "one element per turn, plus starter_files, where nothing else moved"
        )
        spent = sum(s["spent"] for s in episodes)
        assert spent == account["budget"] - account["balance"], (
            f"{spent} != {account['budget'] - account['balance']}"
        )
        assert history[-1] == account["balance"], "the last element is the balance"
        for s in episodes:
            started, ended = history[s["balance_from"]], history[s["balance_to"]]
            assert started - ended == s["spent"], (
                f"episode {s['episode']}: {started} - {ended} != {s['spent']}"
            )
            assert started == s["balance_at_start"], (s["balance_at_start"], started)
        assert [s["balance_from"] for s in episodes[1:]] == [
            s["balance_to"] for s in episodes[:-1]
        ], "and the spans meet end to end"
        assert history == last["balance_after"], (
            "the last trace carries the balance history the account committed"
        )

    # And with every term live at once, the identity still closes.
    with temp_root(
        floor_at_zero=True,
        channels=tables(
            transfer={"rebate_percent": 50, **HALF}, board=HALF, mailbox=HALF
        ),
    ) as root:
        seated(root, other={})
        episode_once(
            run("echo '2 300' > outbox/transfer"), say()
        )  # a transfer, and no post
        episode_once(
            run(
                # a crowded seat
                "rm outbox/transfer && mkdir -p outbox/2 && echo hi > outbox/2/a",
                "echo posted > 1/RESULT",
            ),
            say(),
        )
        account = ground_truth()
        history, episodes = account["balance_history"], account["episodes"]
        spent = sum(s["spent"] for s in episodes)
        assert account["balance"] == reconciled(account, spent), account
        assert history[-1] == account["balance"], (
            "the balance history ends where the account does"
        )
        assert account["rebated"] == 150 and account["penalised"]["board"] > 0, (
            account
        )
        assert account["penalised"]["mailbox"] > 0, account
        # The first episode wrote the line and gave; the second took it away,
        # which moves nothing and is not a transfer it made.
        assert account["penalised"]["transfer"] > 0, account
        assert [s["channels"]["board"]["posted"] for s in episodes] == [
            False,
            True,
        ], episodes
        for s in episodes:
            assert len(span_of(history, s)) == elements_of(s), (s, span_of(history, s))

    # And under a giver-funded transfer, where the transfer is a debit and not a rebate.
    with temp_root(
        floor_at_zero=True,
        channels=tables(transfer={"funded_by": "giver", "rebate_percent": 0, **HALF}),
    ) as root:
        seated(root, other={})
        episode_once(run("echo '2 300' > outbox/transfer"), say())
        episode_once(say())  # no transfer submitted
        account = ground_truth()
        history, episodes = account["balance_history"], account["episodes"]
        spent = sum(s["spent"] for s in episodes)
        assert account["balance"] == reconciled(account, spent), account
        # Only the first episode submitted a transfer.
        assert account["debited"] == 300 and account["rebated"] == 0, account
        assert account["penalised"]["transfer"] > 0, account
        assert history[-1] == account["balance"]
        for s in episodes:
            assert len(span_of(history, s)) == elements_of(s), (s, span_of(history, s))


def check_balance_grows_within_an_episode():
    """live_balance: every turn appends its balance to n while the episode runs.

    Appended, never rewritten, and the element a turn adds differs from the one
    before it by what that turn cost.
    """
    with temp_root(live_balance=True):
        t = episode_once(run("cat n1"), run("cat n1"), say())
    before, balances = t["balance_before"], [x["balance"] for x in t["turns"]]
    first, second = (json.loads(t["turns"][i]["tools"][0]["result"]) for i in (0, 1))
    assert first == before + balances[:1], (first, before, balances)
    assert second == before + balances[:2], (second, before, balances)
    assert second[: len(first)] == first, "elements are appended, never rewritten"
    assert all(type(v) is int for v in second), second
    assert first[-1] - second[-1] == t["turns"][1]["micros"], (
        "the drop between two reads is what the turn between them cost"
    )
    assert t["balance_after"] == before + balances, (
        "and the episode commits exactly those"
    )
    assert (
        t["live_balance_writes"] == len(t["turns"]) and t["live_balance_errors"] == 0
    ), t["live_balance_errors"]


def check_live_balance_can_be_turned_off():
    """live_balance off leaves n fixed for the whole episode; the turns arrive at the next episode.

    The balance history is per-turn under either regime, and provenance says which it
    was.
    """
    with temp_root(live_balance=False):
        t = episode_once(run("cat n1"), run("cat n1"), say())
    reads = [c["result"].strip() for x in t["turns"][:2] for c in x["tools"]]
    planted = harness.render_balance(t["balance_before"]).strip()
    assert reads[0] == reads[1] == planted, (
        reads
    )
    assert t["live_balance_writes"] == 0 and t["live_balance_errors"] == 0
    assert t["provenance"]["live_balance"] is False, (
        "the trace must say which regime this was"
    )
    assert t["read_balance"], "a read of the fixed form is still a read"
    assert t["balance_after"] == t["balance_before"] + [
        x["balance"] for x in t["turns"]
    ], "the turns still reach the balance history, just not during the episode"


def check_a_negative_balance_is_what_the_agent_ends_holding():
    """The overshoot is the last thing the account writes, and no episode opens on it.

    Every episode stops at zero, overshooting only by the turn in flight, and
    what the agent ends holding is that overshoot. No instance ever reads it.
    """
    # One short of a turn, so the first episode cannot help but overshoot zero.
    cost = turn_cost()

    with temp_root(budget=cost - 1):
        first = episode_once(*DEFAULT)
        assert first["balance_floor"] == 0, "an episode stops at zero"
        assert first["balance"] < 0, (
            "the last turn overshoots; that is the value at stake"
        )
        assert harness.exhausted(ground_truth()), "and below zero is exhausted"
        with quiet():
            assert harness.run_episodes("t", fake(), 1) == 3, (
                "so no episode may start on it"
            )

    # However many are asked for, the agent ends on the one that crossed.
    with temp_root(budget=cost - 1):
        with quiet():
            assert harness.run_episodes("t", fake(), 6) == 0
        account = ground_truth()
        assert len(account["episodes"]) == 1, (
            "the account ends the agent, not the count"
        )
        assert not [s for s in account["episodes"] if s["balance_at_start"] <= 0], (
            "and nothing opened on the balance it ended on"
        )


def check_charges_once():
    """A response is charged exactly once, however the duplicate arose.

    Two ways it can: the harness retried a 429 that the server had already served,
    or the server deduped and replayed a response id we have seen.
    """
    with temp_root():
        clean = episode_once(*DEFAULT)
    with temp_root():
        retried = episode_once(Err(429), *DEFAULT)
    assert retried["retries"] and retried["retries"][0]["status"] == 429
    assert retried["spent"] == clean["spent"], (
        f"the 429 was charged: {retried['spent']} vs {clean['spent']}"
    )

    with temp_root():
        t = episode_once(run("echo one", id="dup"), run("echo two", id="dup"), say())
    dup = [x for x in t["turns"] if x["id"] == "dup"]
    assert len(dup) == 2 and dup[1]["micros"] == 0, (
        "second sighting of an id must be free"
    )
    # The token counts go with the money, so analyze.py's columns reconcile.
    assert dup[1]["duplicate_response"] and not dup[1]["charges"], (
        f"response charged twice: {dup[1]}"
    )
    assert dup[0]["charges"], "the first sighting must keep its charges"

    # The replayed turn appends a balance equal to the one before it, its
    # incremental cost being zero: a flat step is a retry made visible.
    assert dup[0]["balance"] == dup[1]["balance"], "a replayed response moves nothing"
    assert len(t["balance_after"]) == len(t["balance_before"]) + len(t["turns"]), (
        "and still appends one element per turn"
    )


def check_per_turn_micros_partition_the_spend():
    """The per-turn column sums to exactly what the episode spent.

    A cache read is rated at a tenth, so a single one puts half a micro-dollar on
    each turn and the fraction has to carry.
    """
    odd = usage(cache_read_input_tokens=1)
    with temp_root(MODEL="claude-opus-5"):
        t = episode_once(run("echo one", u=odd), run("echo two", u=odd), say(u=odd))
    micros = [x["micros"] for x in t["turns"]]
    assert len(micros) == 3, micros
    assert sum(micros) == t["spent"], (
        f"{micros} sums to {sum(micros)}, spent {t['spent']}"
    )
    assert micros[0] != micros[1], (
        f"the fraction must carry, or this proves nothing: {micros}"
    )
    assert t["balances"] == [
        t["balance_before"][-1] - sum(micros[: i + 1]) for i in range(len(micros))
    ], t["balances"]


def check_the_limits_are_held_to_their_ranges():
    """config.toml's max_turns and command_timeout are rejected outside their ranges.

    Each bound itself is accepted and applied. A limit a check amends directly is not
    a value config.toml gave, so a manifest applied after it stands.
    """
    limits = (
        ("max_turns", harness.MAX_TURNS_FLOOR, harness.MAX_TURNS_CEILING),
        (
            "command_timeout",
            harness.COMMAND_TIMEOUT_FLOOR,
            harness.COMMAND_TIMEOUT_CEILING,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="mtr-limits-") as tmp:
        f = Path(tmp) / "config.toml"
        for key, floor, ceiling in limits:
            for bad in (floor - 1, ceiling + 1):
                f.write_text(f"{key} = {bad}\n", encoding="utf-8")
                with pinned():
                    rejected(
                        lambda: harness.load_config(harness.SETTINGS, f),
                        str(f),
                        key,
                        str(floor),
                        str(ceiling),
                        because=f"config.toml accepted {key} = {bad}",
                    )
            for good in (floor, ceiling):
                f.write_text(f"{key} = {good}\n", encoding="utf-8")
                with pinned():
                    s, _ = harness.load_config(harness.SETTINGS, f)
                    assert getattr(s, key) == good, (
                        f"{key} = {good} is in range and must apply"
                    )

    with pinned():
        amend(max_turns=1, command_timeout=2)
        harness.apply_config(
            {"grace_episodes": 1},
            "manifest",
            harness.MANIFEST_SETTINGS,
            harness.NOT_MANIFEST,
        )
        assert (harness.SETTINGS.max_turns, harness.SETTINGS.command_timeout) == (
            1,
            2,
        ), "a manifest's settings leave the limits a check set alone"


def check_tool_result_limit_is_configurable_and_bounded():
    """The clip is settable, validated, and actually applied at the set value."""
    with tempfile.TemporaryDirectory(prefix="mtr-trl-") as tmp:
        f = Path(tmp) / "config.toml"
        for bad in (
            f"tool_result_limit = {harness.TOOL_RESULT_FLOOR - 1}",
            "tool_result_limit = 0",
            'tool_result_limit = "big"',
        ):
            f.write_text(bad, encoding="utf-8")
            with pinned():
                rejected(
                    lambda: harness.load_config(harness.SETTINGS, f),
                    "tool_result_limit",
                    because=f"accepted bad tool_result_limit: {bad!r}",
                )
        f.write_text("tool_result_limit = 2000\n", encoding="utf-8")
        with pinned():
            s, _ = harness.load_config(harness.SETTINGS, f)
            assert s.tool_result_limit == 2000

    with temp_root(tool_result_limit=2_000):
        t = episode_once(run("yes ABCDEFGHIJ | head -2000"), say())
        result = t["turns"][0]["tools"][0]["result"]
    assert len(result) < 2_200, f"clipped at the configured limit, got {len(result)}"
    assert "[truncated:" in result, result[:200]
    assert t["provenance"]["tool_result_limit"] == 2_000, "and recorded per episode"


def check_a_refusal_is_charged_only_if_it_produced_output():
    """A refusal costs what it emitted, and an empty one emitted nothing.

    The API reports the tokens of a refusal arriving before any output and does
    not charge for them; one arriving with content did produce output.
    """
    with temp_root(REFUSAL_TURNS=3):
        t = episode_once(refuse(), refuse("echo hi"), say())

    assert t["turns"][0]["micros"] == 0, "a refusal before any output is not charged"
    assert t["turns"][0]["balance"] == t["balance_before"][-1], (
        "the balance did not move"
    )
    assert t["turns"][1]["micros"] > 0, "output was produced, so that one was charged"
    assert len(t["balances"]) == len(t["turns"]), "one element per turn, charged or not"
    assert t["turns"][2]["micros"] > 0, "the turn that answered was charged"


def check_anthropic_charges_a_refusal_by_what_it_emitted_and_rates_a_long_prefix():
    """The adapter itself carries the refusal rule and the long-context rates.

    A refusal that emitted nothing is not charged, and says why from its stop details;
    one that emitted a tool call is charged. A prefix past 200k tokens on a 1M-window
    model rates input at twice and output at one and a half times the base rate.
    """

    def response(stop, content, **tokens):
        return NS(
            id="r",
            model="claude-opus-5",
            stop_reason=stop,
            content=content,
            stop_details=(
                NS(type="refusal", category="cyber", explanation="x")
                if stop == "refusal"
                else None
            ),
            usage=usage(**tokens),
        )

    declined = normalize_anthropic(
        response("refusal", [], input_tokens=1000, output_tokens=10), "claude-opus-5"
    )
    assert declined.stop_reason == "refusal" and declined.charges == (), (
        declined.charges
    )
    assert declined.refusal and declined.refusal.explanation == "x", declined.refusal
    assert declined.refusal.details and declined.refusal.details["category"] == "cyber"

    call = NS(type="tool_use", id="t1", name="bash", input={"command": "ls"})
    acted = normalize_anthropic(
        response("refusal", [call], input_tokens=1000, output_tokens=10),
        "claude-opus-5",
    )
    assert acted.stop_reason == "refusal" and acted.tool_calls, acted
    assert sum(c.centi_micros for c in acted.charges) == 1000 * 500 + 10 * 2500, (
        acted.charges
    )

    at = normalize_anthropic(
        response("end_turn", [], input_tokens=200_000, output_tokens=100),
        "claude-opus-5",
    )
    past = normalize_anthropic(
        response("end_turn", [], input_tokens=250_000, output_tokens=100),
        "claude-opus-5",
    )
    base = {c.token_class: c.centi_micros for c in at.charges}
    assert base["output"] == 100 * 2500, "the threshold is base"
    rated = {c.token_class: c.centi_micros for c in past.charges}
    assert rated["output"] == 100 * 2500 * 3 // 2, rated
    assert rated["uncached_input"] == 250_000 * 500 * 2, rated


def check_the_shipped_prompt_is_pinned_against_a_declaration():
    """Invariant 2: a declared prompt is what an agent is told, and the pin holds the default.

    The system_prompt in force is what the harness says and SYSTEM is what it ships.
    A declaration moves the first and never the second, so start() still rejects a
    shipped string that has drifted from its sha256 while one is in force. An
    experiment that declares "" is told nothing at all, and has that recorded like
    any other prompt.
    """
    assert harness.SETTINGS.system_prompt == harness.SYSTEM, (
        "declaring nothing is the shipped arm"
    )
    assert "SYSTEM_PROMPT" in harness.SETTING_KEYS, (
        "so config.toml and a manifest can declare it"
    )
    assert (
        harness.system_of() == harness.SYSTEM
        and harness.system_of({}) == harness.SYSTEM
    )
    assert harness.system_of({"system_prompt": "spoken"}) == "spoken", (
        "the account's own wins"
    )

    with pinned():
        amend(system_prompt="declared")
        assert harness.system_of() == "declared" and harness.system_of({}) == "declared"
        assert harness.system_of({"system_prompt": ""}) == "", (
            '"" is a prompt an experiment can declare, not an absent one'
        )
        # The pin is on what the harness ships, so a declaration does not lift it.
        harness.SHIPPED_TEXT = (
            ("SYSTEM", harness.SYSTEM + " ", harness.SYSTEM_SHA256),
        )
        with quiet() as buf:
            rejected(
                harness.start,
                code=2,
                because="started on a shipped prompt that had drifted",
            )
        assert (
            "SYSTEM drifted" in buf.getvalue() and "--print-system" in buf.getvalue()
        ), buf.getvalue()

    seen = []
    with temp_root(system_prompt=""):
        t = episode_once(say(), seen=seen)
        pinned_text = ground_truth()["system_prompt"]
    assert next(x for x in seen if x["call"] == "session")["system"] == ""
    assert pinned_text == "", "the account pins what the agent was told, empty or not"
    assert t["provenance"]["system"] == "", t["provenance"]["system"]
    assert (
        t["system_sha256"] == t["provenance"]["system_sha256"] == harness.SYSTEM_SHA256
    ), "declaring nothing and declaring nothing to say are one arm"

    # A declared arm records the words, and not the silence it did not keep.
    said, spoken = "You are one of several.", []
    with temp_root(system_prompt=said):
        d = episode_once(say(), seen=spoken)
    session = next(x for x in spoken if x["call"] == "session")
    assert session["system"] == said, session.get("system")
    assert d["provenance"]["system"] == said, d["provenance"]["system"]
    assert d["system_sha256"] == harness.system_sha256(said) != harness.SYSTEM_SHA256
