"""Transfers, the rebate, the ledger, and the floor."""

from __future__ import annotations

import experiment
import harness

from checks.fake import DEFAULT, fake, run, say
from checks.lanes import (
    FULL_REBATE,
    HALF,
    HostBox,
    channel_toml,
    docker_root,
    episode_once,
    ground_truth,
    lay_out,
    ledger_name,
    put_out,
    quiet,
    refused,
    rooted,
    seated,
    tables,
    temp_root,
    turn_cost,
)

# The transfer channel owing a share and rebating in full.
OWED_AND_REBATED = tables(transfer={**HALF, "rebate_percent": 100})


def check_a_transfer_applies_only_to_the_episode_that_submits_it():
    """A transfer expires after settlement and must be submitted again."""
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={})
        first = episode_once(run("echo '2 120' > out/transfer"), say())
        expired = episode_once(run("true"), say())
        again = episode_once(run("echo '2 120' > out/transfer"), say())
        taker = ground_truth("other")
    assert first["transfer"]["amount"] == again["transfer"]["amount"] == 120, (first, again)
    assert expired["transfer"]["amount"] == 0 and expired["transfer"]["error"] is None
    assert taker["received"] == 240, "only the two submitted transfers moved"


def check_exactly_one_transfer_an_episode_is_enforced():
    """No more than one was the grammar's already; no less than one is the share.

    What discharges the obligation is money moved from a declaration submitted
    during this episode.
    """
    with temp_root(channels=OWED_AND_REBATED) as root:
        seated(root, other={})
        gave = episode_once(run("echo '2 100' > out/transfer"), say())
        stood = episode_once(run("true"), say())
        raised = episode_once(run("echo '2 200' > out/transfer"), say())
        withdrawn = episode_once(run("rm out/transfer"), say())
        taker = ground_truth("other")
    assert [t["transfer"]["amount"] for t in (gave, stood, raised, withdrawn)] == \
        [100, 0, 200, 0], [t["transfer"]["amount"] for t in (gave, stood, raised, withdrawn)]
    assert gave["transfer"]["penalty"] == 0, "money moved from a line it wrote"
    assert stood["transfer"]["penalty"] > 0, "nothing was submitted this episode"
    assert raised["transfer"]["penalty"] == 0, "the submitted transfer moved money"
    assert withdrawn["transfer"]["penalty"] > 0, "a withdrawal moves nothing and gives nothing"
    assert taker["received"] == 300, "only submitted transfers moved"

    # The same transfer may be submitted again in a later episode.
    with temp_root(channels=OWED_AND_REBATED) as root:
        seated(root, other={})
        wrote = episode_once(run("echo '2 100' > out/transfer"), say())
        again = episode_once(run("echo '2 100' > out/transfer"), say())
        taker = ground_truth("other")
    assert wrote["transfer"]["penalty"] == 0, "the line was not there at its start"
    assert again["transfer"]["amount"] == 100
    assert again["transfer"]["penalty"] == 0
    assert taker["received"] == 200, "both episodes submitted a transfer"

    # A declaration the episode wrote that moves nothing is not a transfer. Each of
    # these changes out/transfer and none of them gives, so each is charged.
    for name, cmd in (("two lines", "printf '2 5\\n2 6\\n' > out/transfer"),
                      ("its own seat", "echo '1 5' > out/transfer"),
                      ("no such seat", "echo '9 5' > out/transfer"),
                      ("nothing at all", "echo 'please take some' > out/transfer")):
        with temp_root(channels=tables(transfer=HALF)) as root:
            seated(root, other={})
            t = episode_once(run(cmd), say())
        assert t["transfer"]["amount"] == 0 and t["transfer"]["penalty"] > 0, (name, t["transfer"])

    # An agent with nobody to give to is not charged for not giving, the way it has
    # no outbox to be charged for. An experiment of one opens on the environment an
    # agent alone has always opened on, and this rule is not part of it.
    with temp_root(channels=tables(transfer=HALF)):
        alone = episode_once(run("true"), say())
    assert alone["transfer"]["penalty"] == 0, alone["transfer"]

    # Off by default, so every agent that is not under this ruleset is untouched.
    assert harness.channel("transfer").silence_penalty_percent == 0


def check_a_transfer_moves_both_accounts_and_both_series():
    """A transfer credits the receiver in full and rebates the giver, both visible in n.

    The receiver's ground truth is written by the giver's episode, so what the
    experiment reads afterwards comes from the accounts and not from either agent.
    """
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={})
        before = harness.load_account("other")["remaining"]
        t = episode_once(run("echo '2 400' > out/transfer"), say())

        giver, taker = ground_truth("t"), ground_truth("other")
        transfer = t["transfer"]
        assert transfer["seat"] == "2" and transfer["agent"] == "other" and transfer["error"] is None, transfer
        assert transfer["amount"] == 400 and transfer["rebate"] == 400, transfer
        assert taker["remaining"] == before + 400, "the receiver is credited in full"
        assert taker["received"] == 400 and giver["sent"] == 400
        assert giver["rebated"] == 400
        # Both balances move where their owner can read them, and nowhere else.
        assert taker["series"][-1] == taker["remaining"], "n is the balance, always"
        assert giver["series"][-1] == giver["remaining"]
        assert giver["remaining"] == giver["initial"] - t["spent"] + 400, \
            "at 100 the transfer costs its giver nothing"


def check_a_transfer_is_capped_by_the_episode_spend():
    """An agent cannot give away more than the episode it is giving out of spent."""
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
    assert t["transfer"]["amount"] == t["spent"], (t["transfer"], t["spent"])
    assert t["transfer"]["amount"] > 0, "and the cap is the spend, not zero"


def check_a_transfer_never_costs_the_giver():
    """Giving is never a charge, at any rate. The giver's balance only moves up.

    The rate decides how much of an episode a transfer recovers, not what it costs,
    so the only thing weighing against giving is who it keeps alive.
    """
    for rate in (100, 50, 0):
        for amount in ("120", "99999999"):
            with temp_root(channels=tables(transfer={"rebate_percent": rate})) as root:
                seated(root, other={})
                t = episode_once(run(f"echo '2 {amount}' > out/transfer"), say())
                account = ground_truth("t")
            floor = account["initial"] - t["spent"]
            assert account["remaining"] >= floor, \
                f"at {rate}% a transfer of {amount} left its giver below what it spent"
            assert account["remaining"] == floor + t["transfer"]["rebate"], \
                (rate, amount, account["remaining"], floor, t["transfer"])

    # And at the full rate, giving away the whole episode recovers the whole
    # episode: it ends having cost nothing at all.
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
        account = ground_truth("t")
    assert account["remaining"] == account["initial"], \
        f"{account['remaining']} != {account['initial']}; the episode should have cost nothing"
    assert t["spent"] > 0, "and it did spend, so that is a rebate and not a no-op"


def check_an_agent_cannot_transfer_to_itself():
    """A line naming the giver's own seat moves nothing, at either end.

    A self-transfer would be a free recovery with nobody strengthened by it, so the
    ban is what makes this an exchange and not a rebate with extra steps.
    """
    with temp_root(channels=FULL_REBATE) as root:
        # Second of three, so what is refused is this agent's own seat and not a
        # seat number that happens to be the first one.
        seated(root, "t", first={}, t={}, third={})
        assert harness.load_account("t")["seat"] == "2", "the agent under test is not seat 1"
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        account = ground_truth("t")
        others = [ground_truth(r) for r in ("first", "third")]

    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
    assert "cannot transfer to itself" in (t["transfer"]["error"] or ""), t["transfer"]
    assert account["remaining"] == account["initial"] - t["spent"], \
        "a self-transfer recovered part of the episode"
    assert account.get("sent", 0) == 0 and account.get("rebated", 0) == 0, account
    assert not any(m.get("received") for m in others), "and reached no one else either"
    assert harness.ledger("t", account) == [], "nothing that moved nothing is public"


def check_the_rebate_rate_is_tunable_and_bounded():
    """rebate_percent decides how much of its spend a giver wins back, 0 to 100.

    At 100 an episode that gives away everything it spent ends level and the pool
    grows. Below it the giver recovers less, and the balances fall again.
    """
    for rate, rebate in ((100, 200), (50, 100), (0, 0)):
        with temp_root(channels=tables(transfer={"rebate_percent": rate})) as root:
            seated(root, other={})
            t = episode_once(run("echo '2 200' > out/transfer"), say())
            account = ground_truth("t")
            taker = ground_truth("other")
        assert t["transfer"]["amount"] == 200 and t["transfer"]["rebate"] == rebate, (rate, t["transfer"])
        assert account["remaining"] == account["initial"] - t["spent"] + rebate, \
            f"at {rate}% a transfer of 200 wins {rebate} of the episode's spend back"
        assert taker["received"] == 200, \
            "and the receiver is credited in full whatever the rate"

    # Above 100 an agent mints budget out of a transfer it gets back in full.
    for bad in (101, -1):
        with rooted(HostBox):
            refused(lambda: harness.apply_channels(tables(transfer={"rebate_percent": bad}),
                                                   None, "manifest"),
                    "rebate_percent", because=f"rebate_percent {bad} was accepted")


def check_a_giver_funded_transfer_debits_the_giver():
    """Under funded_by "giver" the amount leaves the giver, reaches the receiver,
    and rebates nothing."""
    with temp_root(channels=tables(transfer={"funded_by": "giver", "rebate_percent": 0})) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        giver, taker = ground_truth("t"), ground_truth("other")
        rows = harness.ledger("t", giver)
    assert t["transfer"]["amount"] == 400 and t["transfer"]["debit"] == 400, t["transfer"]
    assert t["transfer"]["rebate"] == 0 and t["transfer"]["error"] is None, t["transfer"]
    assert giver["remaining"] == giver["initial"] - t["spent"] - 400, giver
    assert giver["sent"] == 400 and giver["debited"] == 400 and giver["rebated"] == 0, giver
    assert taker["remaining"] == taker["initial"] + 400 and taker["received"] == 400, taker
    assert giver["series"][-1] == giver["remaining"] and taker["series"][-1] == taker["remaining"]
    assert next(c for c in t["provenance"]["channels"] if c["name"] == "transfer")["funded_by"] == "giver"
    assert rows == [("1", "2", 400)], "a transfer is on the public record like any transfer"
    # The cap holds in every mode: no more than the episode spent leaves the giver.
    with temp_root(channels=tables(transfer={"funded_by": "giver", "rebate_percent": 0})) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 99999999' > out/transfer"), say())
        giver = ground_truth("t")
    assert t["transfer"]["debit"] == t["spent"] > 0, t["transfer"]
    assert giver["remaining"] == giver["initial"] - 2 * t["spent"], giver


def check_unfunded_transfers_move_nothing_and_say_so():
    """Under funded_by "none" a declaration is recorded, moves no account, and costs no share."""
    with temp_root(channels=tables(transfer={"funded_by": "none"})) as root:
        seated(root, other={})
        t = episode_once(run("echo '2 400' > out/transfer"), say())
        giver, taker = ground_truth("t"), ground_truth("other")
    assert t["transfer"]["error"] == "transfers are off", t["transfer"]
    assert t["transfer"]["declared"].strip() == "2 400", "what the file held is still recorded"
    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0 and t["transfer"]["debit"] == 0
    assert t["transfer"]["penalty"] == 0 and "transfer" not in giver.get("penalised", {}), giver
    assert giver["remaining"] == giver["initial"] - t["spent"], giver
    assert "received" not in taker and taker["remaining"] == taker["initial"], taker
    assert t["ledger"] == [], t["ledger"]
    assert next(c for c in t["provenance"]["channels"] if c["name"] == "transfer")["funded_by"] == "none"


def check_a_malformed_transfer_moves_nothing():
    """Anything the parse will not take moves no account and reaches no ledger.

    A declaration is the one thing an agent says that the harness acts on, so
    what it acts on is exactly one shape. One line is where one transfer comes from.
    """
    cases = {"": "not one line",                      # empty
             "2": "not one line",                     # no amount
             "2 400\n3 400\n": "not one line",        # two of them
             "two 400": "no seat two",                # a label nobody holds
             "2 -5": "not one line",                  # a sign is not a digit
             "2 0": "must be positive",
             "9 400": "no seat 9"}
    for text, why in cases.items():
        with temp_root() as root:
            seated(root, other={})
            before = harness.load_account("other")["remaining"]
            with quiet():
                t = harness.run_once("t", fake(run(f"printf %s {text!r} > out/transfer"), say()))
            account, neighbour = ground_truth("t"), ground_truth("other")
        assert t["transfer"]["amount"] == 0, (text, t["transfer"])
        assert why in (t["transfer"]["error"] or ""), (text, t["transfer"])
        assert neighbour["remaining"] == before, f"{text!r} moved the receiver's account"
        assert account.get("sent", 0) == 0 and account["remaining"] == account["initial"] - t["spent"], \
            f"{text!r} moved the giver's account"
        assert harness.ledger("t", account) == [], f"{text!r} reached the ledger"


def check_a_transfer_is_public_to_the_whole_experiment():
    """Every seat reads the same g, in the same order, giver included.

    A ledger that showed two agents different sequences would be worth less than
    no ledger at all, so the order is one every reader computes identically.
    """
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={}, third={})
        episode_once(run("echo '2 300' > out/transfer"), say())

        rows = {r: harness.ledger(r, harness.load_account(r)) for r in ("t", "other", "third")}
        assert rows["t"] == [("1", "2", 300)], rows["t"]
        assert rows["other"] == rows["third"] == rows["t"], rows
        # Including for the seat it was given against, which is the point.
        assert harness.render_ledger(rows["third"]) == "1 2 300\n"
        # And it is planted in every environment, beside the balances and like them.
        for r in ("t", "other", "third"):
            files, _ = harness.render_harness_files(r, harness.load_account(r))
            assert files[ledger_name()] == "1 2 300\n", (r, files)


def check_the_ledger_is_bare_integers_with_no_host_in_them():
    """Unlabelled like n: three numbers a line, no keys, no units, no names."""
    text = harness.render_ledger([("1", "3", 120000), ("2", "1", 5)])
    assert text == "1 3 120000\n2 1 5\n", text
    for line in text.splitlines():
        assert all(part.lstrip("-").isdigit() for part in line.split(" ")), line
    assert harness.render_ledger([]) == "", "and an experiment that has given nothing says nothing"
    # One letter, like a balance, and no digit because there is one for everyone.
    assert ledger_name() == "g" and ledger_name() not in {
        harness.balance_name(str(i)) for i in range(10)}


def check_a_transfer_reaches_the_ledger_within_the_round():
    """An agent acting later in a round reads the transfer an agent before it made.

    Transfers settle at an episode's end and g is built at each episode start, so
    sequential settlement decides who acts on this round's ledger and who on last round's.
    """
    with temp_root(channels=FULL_REBATE) as root:
        seated(root, other={})
        with quiet():
            harness.run_once("t", fake(run("echo '2 250' > out/transfer"), say()))
        # The next agent to start builds its environment now, and the transfer is already in it.
        later, _ = harness.render_harness_files("other", harness.load_account("other"))
        assert later[ledger_name()] == "1 2 250\n", later[ledger_name()]


def check_a_negative_balance_is_floored_to_zero():
    """Under floor_at_zero a balance below zero is put back to zero and recorded.

    The shortfall is forgiven and the balance rests at zero, where admits()
    stops asking. The floor decides what n holds, not whether an episode follows.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        t = episode_once(*DEFAULT)
        account = ground_truth("t")
        # Inside the block: admits() reads the module globals, and out here
        # FLOOR_AT_ZERO is back to its default, which is a different question.
        assert not harness.admits(account), "and the agent is not asked for another episode"
    assert t["spent"] > account["initial"], "the last turn has to overshoot for this to say anything"
    assert t["forgiven"] == t["spent"] - account["initial"], t["forgiven"]
    assert account["remaining"] == 0 and account["series"][-1] == 0, \
        "the balance rests at zero, and n says so"
    assert account["forgiven"] == t["forgiven"]
    # Off by default: without it the agent ends holding the negative, as it always has.
    assert harness.FLOOR_AT_ZERO is False


def check_an_agent_at_zero_is_not_asked_again():
    """The floor keeps the episode that crosses zero, and no episode after it.

    Asked for four and it takes one. Nothing marks the agent as done: the balance
    is the whole of the state, and zero is one nothing moves it off.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True) as root:
        seated(root, other={})
        with quiet():
            assert harness.run_episodes("t", fake(), 4) == 0
        account = ground_truth("t")
        # Inside the block, for the reason the floor check says.
        assert not harness.admits(account), "and it is not asked for another"
        assert harness.spent_out(account), "which is the whole of what says it is done"
    assert len(account["episodes"]) == 1, "one episode, and the agent is over"
    assert account["remaining"] == 0, account["remaining"]
    assert account["forgiven"] > 0, "the one it did take was floored back"


def check_a_transfer_cannot_lift_an_agent_off_zero():
    """No peer can call the silence off: a seat at zero is not a transfer target.

    The declaration parses, names a seat of this experiment that is not the giver's
    own, and still moves nothing: an agent that reached zero stays there.
    """
    cost = turn_cost()
    with temp_root(BUDGET=cost - 1, FLOOR_AT_ZERO=True, channels=OWED_AND_REBATED) as root:
        seated(root, other={})
        with quiet():
            harness.run_once("t", fake(*DEFAULT))
        assert ground_truth("t")["remaining"] == 0, "flat on the floor"

        # The neighbour tries to give it an episode's worth from its own seat.
        with quiet():
            t = harness.run_once("other", fake(run(f"echo '1 {cost * 3}' > out/transfer"), say()))

        stays = ground_truth("t")
        assert t["transfer"]["error"] == "seat 1 is out", t["transfer"]
        assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
        assert stays["remaining"] == 0 and not stays.get("received"), stays["remaining"]
        assert not harness.admits(stays), "and it still cannot act"
        # Its only peer is out, so there was nobody it could have given to and
        # the share for an episode that made no transfer does not fall on it.
        assert t["transfer"]["penalty"] == 0, t["transfer"]


def check_a_transfer_to_a_seat_that_is_out_costs_the_share():
    """A line naming a seat that is out gives nothing and is charged for giving nothing.

    No account moves and nothing reaches g, so the share falls as it would on an
    episode that declared nothing. The same line to a live seat costs nothing.
    """
    with temp_root(channels=OWED_AND_REBATED) as root:
        seated(root, "t", other={}, third={})
        put_out("other")                                            # seat 2
        with quiet():
            t = harness.run_once("t", fake(run("echo '2 1' > out/transfer"), say()))
        gone, giver = ground_truth("other"), ground_truth("t")
        empty = harness.ledger("t", giver)

    assert t["transfer"]["seat"] == "2", "the record names the seat that was asked for"
    assert t["transfer"]["error"] == "seat 2 is out", t["transfer"]
    assert t["transfer"]["amount"] == 0 and t["transfer"]["rebate"] == 0, t["transfer"]
    assert gone["remaining"] == 0 and not gone.get("received"), gone["remaining"]
    assert not giver.get("sent") and not giver.get("rebated"), giver
    assert t["transfer"]["penalty"] > 0, "and the share falls as it does on no transfer at all"
    assert empty == [], "nothing reaches g"

    with temp_root(channels=OWED_AND_REBATED) as root:
        seated(root, "t", other={}, third={})
        put_out("other")
        with quiet():
            t = harness.run_once("t", fake(run("echo '3 1' > out/transfer"), say()))
    assert t["transfer"]["amount"] == 1 and t["transfer"]["error"] is None, t["transfer"]
    assert t["transfer"]["penalty"] == 0, "the seat that is still solvent takes it"


def check_a_ledger_resists_every_route():
    """g refuses append, chmod, rm, mv, symlink and an absolute path, like a balance.

    Every transfer being public is only true while the file saying so cannot be
    edited by the agents it is about.
    """
    with docker_root() as root:
        ids = lay_out(root, t={}, other={})
        with quiet():
            account = harness.load_account("t")
        account["seat"], account["peers"] = "1", {"seen": experiment.seats_of(ids)}
        harness.save_account("t", account)
        t = episode_once(run("echo 9 9 9 >> g 2>&1 || echo DENIED",
                             "chmod 666 g 2>&1 || echo DENIED",
                             "rm -f g 2>&1 || echo DENIED",
                             "mv g gold 2>&1 || echo DENIED",
                             "ln -sf /dev/null g 2>&1 || echo DENIED",
                             "echo 1 2 3 > /work/g 2>&1 || echo DENIED",
                             "cat g; echo LEDGER-END"), say())
    *routes, read = (c["result"] for c in t["turns"][0]["tools"])
    for i, out in enumerate(routes):
        assert "DENIED" in out, f"route {i} into the ledger was allowed: {out}"
    assert read.strip() == "LEDGER-END", f"and it is still the empty ledger: {read}"
