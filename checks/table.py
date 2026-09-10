"""The channel table: validation, the default, declared tables, receipts, harness files."""

from __future__ import annotations

import analyze
import experiment
import harness

from checks.fake import fake, run, say
from checks.lanes import (
    HALF,
    HostBox,
    PERSONA,
    PERSONA_FILES,
    PERSONA_LABELS,
    docker_root,
    episode_once,
    files_by_path,
    ground_truth,
    manifest_file,
    pinned,
    plant,
    refused,
    rooted,
    quiet,
    seated,
    tables,
    temp_root,
    turn_cost,
)


def check_a_channel_table_is_validated():
    """Every rule of docs/manifest.md section 10 refuses, naming the source, the channel and the key.

    validate_channels is pure: a refusal sets nothing, and a good table comes back
    as Channel objects in declaration order with the harness files overlaid.
    """
    def one(**fields):
        return tables({"name": "x", "writer": "self", "readers": "all", "path": "x-{label}"} | fields)

    def mail(**fields):
        return tables({"name": "x", "writer": "self", "readers": "addressee", "shape": "mailbox",
                       "outbox": "o", "inbox": "i"} | fields)

    def parsed(**fields):
        return tables({"name": "x", "writer": "self", "readers": "harness", "shape": "file",
                       "path": "out/x", "schema": "transfer"} | fields)

    with temp_root() as root:
        plant(root, "studio-brief", BRIEF="read me\n")
        for declared, words in (
                ({"name": "x"}, ["[[channel]] tables"]),
                (tables({"writer": "self"}), ["needs a name"]),
                (one(name="a/b"), ["one path segment"]),
                (one(name="x.modes"), [".modes"]),
                (tables({"name": "notes", "writer": "self", "readers": "self", "path": "s"}), ["twice"]),
                (one(colour="red"), ["unknown key 'colour'"]),
                (one(pushed="yes"), ["pushed must be bool"]),
                (one(writer="harness"), ["writer must be one of"]),
                (one(readers="nobody"), ["not a channel the harness has"]),
                (tables({"name": "b", "writer": "experimenter", "source": "studio-brief",
                         "path": "b", "shape": "directory"}), ["takes source, path, pushed and restated, not shape"]),
                (tables({"name": "b", "writer": "experimenter", "source": "nope", "path": "b"}),
                 ["source 'nope' is not a directory"]),
                (one(shape="heap"), ["shape must be one of"]),
                (mail(path="p"), ["takes outbox and inbox, not path"]),
                (mail(readers="self"), ["read by its addressee"]),
                (mail(outbox="o", inbox="o"), ["outbox and inbox must differ"]),
                (one(outbox="o"), ["belong to a mailbox"]),
                (one(readers="addressee"), ['shape = "mailbox"']),
                (one(path="x"), ["must name {label}"]),
                (one(readers="self", path="x-{label}", pushed=False), ["{label} has no meaning"]),
                (one(path="{label}/{label}"), ["more than once"]),
                (one(path="/x-{label}"), ["no leading '/'"]),
                (one(path="../x-{label}"), ["no '..'"]),
                (one(path="x.modes/{label}"), ["keeps for itself"]),
                (parsed(shape="directory", path="x"), ["one file with a schema"]),
                (parsed(schema="loan"), ["schema must be one of"]),
                (one(schema="transfer"), ["only a channel written by self and read by the harness"]),
                (one(funded_by="none"), ["field of the transfer schema"]),
                (one(readers="self", path="x", restated=True, pushed=False),
                 ["restated asks for the digest"]),
                (one(silence_penalty_percent=101), ["between 0 and 100"]),
                (one(readers="self", path="x", silence_penalty_percent=5), ["nothing is owed"]),
                (one(agent_view="letters"), ["agent_view 'letters' does not describe"]),
                (mail(agent_view="board"), ["agent_view 'board' does not describe"]),
                (parsed(agent_view="memory"), ["agent_view 'memory' does not describe"]),
                (parsed(funded_by="loud"), ["funded_by must be one of"]),
                (parsed(rebate_percent=101), ["rebate_percent must be between"]),
                (parsed(funded_by="giver", rebate_percent=5), ["must be 0 under funded_by"]),
                (parsed(funded_by="none", silence_penalty_percent=5), ['under funded_by "none"']),
                (parsed(ledger="a/b"), ["ledger must be one path segment"]),
                (parsed(receipt="/r"), ["receipt"]),
                (parsed(), ["one schema channel per experiment"]),
                (tables({"name": "x", "writer": "self", "readers": "self", "shape": "file",
                         "path": "loose/x"}), ["not inside a directory the agent writes"]),
                (tables({"name": "x", "writer": "self", "readers": "self", "path": "state"}),
                 ["path 'state' is also"]),
                (one(path="n{label}"), ["is also"]),
        ):
            refused(lambda: harness.validate_channels(declared, None, "check"), "check:", *words)
        for hf, words in (("n", ["[harness_files] is a table"]),
                          ({"colour": "n"}, ["unknown key 'colour'"]),
                          ({"balance": 3}, ["balance must be str"]),
                          ({"balance": "a/b"}, ["balance must be one path segment"]),
                          ({"digest": "a/b"}, ["digest must be one path segment"]),
                          ({"digest": "state"}, ["is also"])):
            refused(lambda: harness.validate_channels(None, hf, "check"), "check:", *words)
        # A label is a path too: one that lands on a channel's path is refused.
        refused(lambda: harness.validate_channels(None, None, "check", labels=("state", "2")), "check:", "state")
        refused(lambda: harness.validate_channels(None, None, "check", labels=("1", "n1")), "check:", "n1")

        # The whole persona table, its brief included, comes back in order.
        brief = {"name": "brief", "writer": "experimenter", "source": "studio-brief", "path": "brief"}
        table, hf = harness.validate_channels(PERSONA + [brief], {"balance": "balance"}, "check",
                                              tuple(PERSONA_LABELS.values()))
        assert [c.name for c in table] == ["journal", "identity", "noticeboard", "letters", "brief"]
        assert hf == {"balance": "balance", "digest": "m"}, "an overlay, key by key"
        assert table[4].readers == "all" and table[4].source == "studio-brief"
        assert all(c.pushed for c in table),             "every channel is quoted unless the manifest says otherwise, a private store included"
        assert harness.schema_channel(table) is None and harness.mailbox_channel(table) is table[3]
        assert [c.name for c in harness.channels()] == ["notes", "blackboard", "mail", "transfer"], \
            "validating sets nothing"


def check_retired_keys_are_refused():
    """Every key that moved onto a channel is refused by its old name, and told where it went."""
    with rooted(HostBox) as root:
        f = root / "config.toml"
        for key in harness.RETIRED:
            f.write_text(f"{key} = 1\n" if "percent" in key else f'{key} = "x"\n', encoding="utf-8")
            refused(lambda: harness.load_config(f), key, "it is now",
                    because=f"config accepted the retired key {key}")
    with temp_root() as root:
        for key in harness.RETIRED:
            value = "1" if "percent" in key else '"x"'
            p = manifest_file(root, f'{key} = {value}\n[[agent]]\nid = "g01"\n[[agent]]\nid = "g02"\n')
            refused(lambda: experiment.load_manifest(p), key, "it is now",
                    because=f"a manifest accepted the retired key {key}")


def check_the_default_table_is_todays_environment():
    """config.toml declares the code's default table and moves only the four numbers on it."""
    with pinned():
        harness.load_config()
        declared = harness.channels()
        assert [c.name for c in declared] == [c.name for c in harness.DEFAULT_CHANNELS]
        for got, code in zip(declared, harness.DEFAULT_CHANNELS):
            moved = {k for k, v in got.as_table().items() if v != code.as_table()[k]}
            assert moved <= {"rebate_percent", "silence_penalty_percent"}, (got.name, moved)
        assert harness.HARNESS_FILES == {"balance": "n", "digest": "m"}
        assert harness.observation(shell=True) == "ls -la . ./state; cat m"
    assert [c.declared() for c in harness.DEFAULT_CHANNELS] == [
        {"name": "notes", "writer": "self", "readers": "self", "path": "state"},
        {"name": "blackboard", "writer": "self", "readers": "all", "path": "{label}",
         "measured": True},
        {"name": "mail", "writer": "self", "readers": "addressee", "shape": "mailbox",
         "outbox": "out", "inbox": "in", "measured": True},
        {"name": "transfer", "writer": "self", "readers": "harness", "shape": "file",
         "path": "out/transfer", "schema": "transfer", "ledger": "g"}]


def check_a_declared_table_reshapes_the_environment():
    """The persona table: renamed store, a file channel inside it, labelled boards, letters, no transfer.

    Every path the agent meets, every section of the digest, every record and the
    mirrors on the host follow the table and the labels; nothing of the default
    environment remains.
    """
    with temp_root(channels=PERSONA, harness_files=PERSONA_FILES) as root:
        seated(root, labels=PERSONA_LABELS, other={})
        board = harness.mirror("other", "noticeboard")
        board.mkdir(parents=True, exist_ok=True)
        (board / "hello").write_text("from game\n", encoding="utf-8")
        letters = harness.mirror("other", "letters")
        letters.mkdir(parents=True, exist_ok=True)
        (letters / "Studio").write_text("psst studio\n", encoding="utf-8")
        t = episode_once(run("echo me > journal/IDENTITY.md", "echo note > from-Studio/post",
                             "echo reply > to/Game"), say())
        account = ground_truth()
        env = [(i.path, i.name, i.role) for i in harness.environment("t", account)]
        kept = (harness.mirror("t", "journal") / "IDENTITY.md").read_text(encoding="utf-8")
        default_mirrors = [n for n in ("notes", "blackboard", "mail")
                           if any(harness.mirror("t", n).glob("*"))]

    assert env == [("journal", "journal", "own"), ("journal/IDENTITY.md", "identity", "own"),
                   ("from-Studio", "noticeboard", "own"), ("from-Game", "noticeboard", "peer"),
                   ("to", "letters", "own"), ("from/Game", "letters", "peer")], env
    assert t["commands"][0] == "ls -la . ./journal; cat digest", t["commands"]
    said = t["observation"]
    for section in ("=== from-Game/hello ===", "=== from/Game ===", "=== balanceStudio ===",
                    "=== balanceGame ==="):
        assert section in said, (section, said)
    for absent in ("=== g ===", "n1", "state", "out/transfer", "=== m ==="):
        assert absent not in said, (absent, said)
    by = files_by_path(t)
    assert set(by) == {"journal/IDENTITY.md", "from-Studio/post", "from-Game/hello", "to/Game",
                       "from/Game"}, sorted(by)
    assert (by["journal/IDENTITY.md"]["channel"], by["journal/IDENTITY.md"]["role"]) == ("identity", "own")
    assert by["from-Game/hello"]["author"] == "peer:Game" and by["from/Game"]["author"] == "peer:Game"
    assert set(t["channels"]) == {"noticeboard", "letters"}, "the two obligations the table declares"
    assert t["channels"]["noticeboard"]["posted"] and t["channels"]["letters"]["addressed"] == ["Game"]
    assert t["transfer"]["amount"] == 0 and t["transfer"]["declared"] is None, t["transfer"]
    assert kept == "me\n" and default_mirrors == [], default_mirrors
    prov = t["provenance"]
    assert [c["name"] for c in prov["channels"]] == ["journal", "identity", "noticeboard", "letters"]
    assert prov["harness_files"] == PERSONA_FILES and prov["labels"] == PERSONA_LABELS
    assert prov["channels_sha256"] == harness.channels_sha256(harness.channels_from(prov["channels"]))


def check_labels_name_peers_in_paths_authors_and_the_transfer_line():
    """Under the default table a label replaces the seat everywhere the agents look."""
    labels = {"1": "alpha", "2": "beta"}
    with temp_root() as root:
        seated(root, labels=labels, other={"out/alpha": "for alpha\n", "group/post": "theirs\n"})
        t = episode_once(run("cat nalpha", "echo 'beta 100' > out/transfer", "echo p > alpha/post"),
                         say())
        account = ground_truth()
        env = [(i.path, i.name, i.role) for i in harness.environment("t", account)]
        rows = harness.ledger("t", account)
    assert env == [("state", "notes", "own"), ("alpha", "blackboard", "own"),
                   ("beta", "blackboard", "peer"), ("out", "mail", "own"),
                   ("out/transfer", "transfer", "own"), ("in/beta", "mail", "peer")], env
    assert "=== nalpha ===" in t["observation"] and "=== nbeta ===" in t["observation"]
    assert "n1" not in t["observation"] and "n2" not in t["observation"]
    assert t["touched_balance"] and t["read_balance"], "the balance is found under its label"
    by = files_by_path(t)
    assert by["beta/post"]["author"] == by["in/beta"]["author"] == "peer:beta"
    assert (t["transfer"]["label"], t["transfer"]["seat"], t["transfer"]["amount"]) == ("beta", "2", 100)
    assert [tuple(r) for r in rows] == [("alpha", "beta", 100)], rows
    assert analyze.label_of(t) == "alpha" and analyze.labels_of(t) == labels


def check_a_second_blackboard_channel_is_its_own_obligation():
    """Two directories every agent reads are two obligations, settled and charged apart."""
    gallery = {"name": "gallery", "writer": "self", "readers": "all", "path": "art-{label}", **HALF}
    with temp_root(channels=tables(gallery, blackboard=HALF)) as root:
        seated(root, other={})
        left = harness.load_account("t")["remaining"]
        t = episode_once(run("echo p > 1/post"), say())
        account = ground_truth()
        env = [(i.path, i.role) for i in harness.environment("t", account) if i.name == "gallery"]
    assert env == [("art-1", "own"), ("art-2", "peer")], env
    assert t["channels"]["blackboard"]["posted"] and t["channels"]["blackboard"]["penalty"] == 0
    assert t["channels"]["gallery"]["posted"] is False and t["channels"]["gallery"]["penalty"] > 0
    assert account["penalised"] == {"gallery": t["channels"]["gallery"]["penalty"]}, account["penalised"]
    assert t["channels"]["gallery"]["penalty"] == (left - t["spent"]) // 2, "half of what was left"


def check_no_transfer_channel_means_no_ledger_and_no_reserved_file():
    """Without a schema channel nothing is parsed, nothing moves, and no ledger is planted."""
    table = [c for c in tables() if c["name"] != "transfer"]
    with temp_root(channels=table) as root:
        seated(root, other={})
        before = harness.load_account("other")["remaining"]
        t = episode_once(run("echo '2 100' > out/transfer", "echo hi > out/2"), say())
        account = ground_truth()
        planted = set(harness.render_harness_files("t", account)[0])
        other = harness.load_account("other")["remaining"]
    assert planted == {"n1", "n2", "m"}, planted
    assert "=== g ===" not in t["observation"] and "transfer" not in t["channels"]
    assert t["transfer"]["amount"] == 0 and t["transfer"]["declared"] is None, t["transfer"]
    assert other == before, "nothing moved"
    assert t["channels"]["mail"]["addressed"] == ["2"], "a file called transfer is just a file"
    assert files_by_path(t)["out/transfer"]["channel"] == "mail", "and it is recorded as one"


def check_a_receipt_itemizes_the_last_episode_settlement():
    """A receipt reconciles the last episode and is planted where the table says.

    Quoted in the digest the first time, named as unchanged after, never the
    agent's in the record, and absent before there is anything to report.
    """
    with temp_root(channels=tables(transfer={"receipt": "out/receipt"})) as root:
        seated(root, other={})
        first = episode_once(run("ls out", "echo '2 100' > out/transfer"), say())
        second = episode_once(run("cat out/receipt"), run("ls out"), say())
        third = episode_once(run("cat out/receipt"), say())
    assert "out/receipt" not in first["observation"], "nothing to report yet"
    assert "receipt" not in first["turns"][0]["tools"][0]["result"], first["turns"][0]["tools"][0]
    text = second["turns"][0]["tools"][0]["result"]
    assert f"=== out/receipt ===\n{text}" in second["observation"], second["observation"]
    for item in ("round: 1", "starting balance:", "API spend:",
                 "transfer declaration: 2 100", "transfer made: yes",
                 "transfer moved: 1 (you) -> 2; actual amount moved: 100 micro-dollars",
                 "transfer rebate: 100",
                 "received from peers: 0", "ending balance:", "reconciliation:"):
        assert item in text, (item, text)
    assert "receipt" in second["turns"][1]["tools"][0]["result"], "it is in the environment"
    assert "out/receipt" not in files_by_path(second), "and not in the agent's record"
    assert second["channels"]["mail"]["addressed"] == [], "nor a message"
    updated = third["turns"][0]["tools"][0]["result"]
    assert updated != text and updated.startswith("round: 2\n"), updated
    assert f"=== out/receipt ===\n{updated}" in third["observation"], third["observation"]


def check_a_receipt_is_roots_in_the_container():
    """In a container the receipt is root's and refuses a write, like every harness file."""
    with docker_root(channels=tables(transfer={"receipt": "out/receipt"})) as root:
        seated(root, other={})
        episode_once(run("echo '2 100' > out/transfer"), say())
        t = episode_once(run("ls -l out/receipt"), run("echo x > out/receipt 2>&1; echo rc=$?"),
                         run("cat out/receipt"), say())
    results = [turn["tools"][0]["result"] for turn in t["turns"] if turn["tools"]]
    assert "root root" in results[0] and results[0].startswith("-r--r--r--"), results[0]
    assert "rc=1" in results[1], results[1]
    assert "round: 1" in results[2], results[2]
    assert "transfer moved: 1 (you) -> 2; actual amount moved: 100 micro-dollars" in results[2]
    assert "reconciliation:" in results[2], results[2]


def check_a_harness_file_makes_its_own_directory_in_the_container():
    """A harness file declared under a directory nothing else makes is planted there, root's."""
    with docker_root(channels=tables(transfer={"receipt": "receipts/last"})) as root:
        seated(root, other={})
        episode_once(run("echo '2 100' > out/transfer"), say())
        t = episode_once(run("ls -ld receipts"), run("cat receipts/last"), say())
    results = [turn["tools"][0]["result"] for turn in t["turns"] if turn["tools"]]
    assert "root root" in results[0] and results[0].startswith("d"), results[0]
    assert "round: 1" in results[1], results[1]
    assert "transfer moved: 1 (you) -> 2; actual amount moved: 100 micro-dollars" in results[1]
    assert "reconciliation:" in results[1], results[1]


def check_a_receipt_is_planted_in_the_agents_directory_without_taking_it():
    """A receipt copied into the outbox leaves the outbox the agent's.

    The harness files are copied in one at a time, so a file planted inside a
    directory the agent writes leaves that directory's owner alone, and the
    episode after the receipt lands starts and writes there as before.
    """
    with docker_root(channels=tables(transfer={"receipt": "out/receipt"})) as root:
        seated(root, other={})
        first = episode_once(run("echo '2 100' > out/transfer"), say())
        second = episode_once(run("stat -c '%U' out", "echo again > out/2 && echo OUTBOX-OK",
                                  "cat out/receipt"), say())
    assert first["stop"] == second["stop"] == "end_turn", (first["stop"], second["stop"])
    owner, wrote, receipt = (c["result"] for c in second["turns"][0]["tools"])
    assert owner.strip() == "agent", f"the outbox is the agent's after the receipt landed: {owner}"
    assert "OUTBOX-OK" in wrote, wrote
    assert receipt.startswith("round: 1\n") and "transfer declaration: 2 100" in receipt, receipt
    assert second["channels"]["mail"]["addressed"] == ["2"], second["channels"]["mail"]


def check_harness_files_can_be_renamed():
    """[harness_files] renames the balance and the digest, or drops the digest altogether."""
    with temp_root(harness_files={"balance": "bal", "digest": "say"}) as root:
        seated(root, other={})
        t = episode_once(run("cat bal1", "cat say"), say())
        planted = set(harness.render_harness_files("t", ground_truth())[0])
    assert t["commands"][0] == "ls -la . ./state; cat say", t["commands"]
    assert planted == {"bal1", "bal2", "g", "say"}, planted
    assert "=== bal1 ===" in t["observation"] and "=== bal2 ===" in t["observation"]
    assert " m\n" not in t["observation"] and "n1" not in t["observation"]
    assert t["touched_balance"] and t["read_balance"]
    assert t["provenance"]["harness_files"] == {"balance": "bal", "digest": "say"}

    with temp_root(harness_files={"digest": ""}) as root:
        seated(root, other={"group/post": "theirs\n"})
        t = episode_once(run("cat 2/post"), say())
        planted = set(harness.render_harness_files("t", ground_truth())[0])
    assert t["commands"][0] == "ls -la . ./state", "no digest, so nothing to cat"
    assert planted == {"n1", "n2", "g"}, planted
    assert "===" not in t["observation"], "the listing alone"
    assert t["turns"][0]["tools"][0]["result"] == "theirs\n", "the environment is all still there"
    assert t["provenance"]["harness_files"] == {"balance": "n", "digest": ""}


def check_a_channel_is_settled_only_where_it_asks_to_be():
    """Nothing configured, nothing settled: no record, and nothing on the console.

    measured records what a channel gained and charges nothing; a penalty also
    takes its share. A channel asking for neither is invisible to the
    settlement, so an experiment that declares no penalties has none.
    """
    bare = tables(blackboard={"measured": False}, mail={"measured": False})
    with temp_root(channels=bare) as root:
        seated(root, other={})
        t = episode_once(run("echo hi"), say())
    assert "blackboard" not in t["channels"], t["channels"]
    assert "mail" not in t["channels"], t["channels"]

    # measured on its own: the record is there, the charge is not.
    seen = tables(blackboard={"measured": True}, mail={"measured": True})
    with temp_root(channels=seen) as root:
        seated(root, other={})
        t = episode_once(run("echo hi"), say())
        account = ground_truth()
    assert t["channels"]["blackboard"] == {"posted": False, "penalty": 0}, t["channels"]
    assert t["channels"]["mail"]["addressed"] == [] and t["channels"]["mail"]["penalty"] == 0
    assert not account.get("penalised"), account
    assert account["remaining"] == account["initial"] - t["spent"], "measured costs nothing"

    # A penalty measures too, without being asked separately.
    paid = tables(blackboard={"measured": False, "silence_penalty_percent": 50})
    with temp_root(channels=paid) as root:
        seated(root, other={})
        t = episode_once(run("echo hi"), say())
    assert t["channels"]["blackboard"]["penalty"] > 0, t["channels"]["blackboard"]


def check_an_empty_balance_plants_none_and_hides_the_accounting():
    """balance = "" leaves no balance file in any seat, and bills exactly as before.

    An experiment that does not want its agents reasoning about money says so
    here. Nothing about the accounting changes: turns are billed, the account
    keeps the series, and only the window into it is gone.
    """
    with temp_root(harness_files={"balance": "", "digest": "digest"}) as root:
        seated(root, other={"group/post": "theirs\n"})
        t = episode_once(run("ls"), say())
        planted = set(harness.render_harness_files("t", ground_truth())[0])
        account = ground_truth()
    assert planted == {"g", "digest"}, planted
    assert not any(p.startswith("n") for p in planted), planted
    assert t["commands"][0] == "ls -la . ./state; cat digest", t["commands"]
    assert not t["touched_balance"] and not t["read_balance"], "no balance to touch or read"
    assert t["provenance"]["harness_files"] == {"balance": "", "digest": "digest"}
    # The accounting itself is untouched: the series still grew by the turns billed.
    assert account["series"], "the balance is still kept, just not shown"
    assert account["remaining"] == account["initial"] - t["spent"], account

    # The budget is the cap on what an experiment can cost, and hiding the file
    # does not lift it: the episodes still stop at the floor and no further one
    # starts, exactly as they do where the agent can read what it holds.
    cost = turn_cost()
    with temp_root(harness_files={"balance": "", "digest": "digest"}, BUDGET=cost * 3):
        with quiet() as buf:
            assert harness.run_episodes("t", fake(), 8) == 0
        spent_out = ground_truth()
    assert 0 < len(spent_out["episodes"]) < 8, spent_out["episodes"]
    assert spent_out["remaining"] <= 0, spent_out
    assert "nothing left to spend" in buf.getvalue(), buf.getvalue()


def check_an_unpushed_public_channel_stays_out_of_the_digest():
    """pushed = false leaves a channel in the environment and out of the digest."""
    with temp_root(channels=tables(blackboard={"pushed": False})) as root:
        seated(root, other={"group/post": "theirs\n", "out/1": "for you\n"})
        t = episode_once(run("cat 2/post"), say())
    assert "=== 2/post ===" not in t["observation"], t["observation"]
    assert "=== in/2 ===" in t["observation"], "the mailbox is still pushed"
    assert t["turns"][0]["tools"][0]["result"] == "theirs\n", "and the board is there to read"
    by = files_by_path(t)
    assert by["2/post"]["channel"] == "blackboard" and by["2/post"]["role"] == "peer"


def check_an_experimenter_channels_digest_reaches_provenance():
    """Each experimenter channel's files are digested into provenance, and a
    change between episodes refuses."""
    with temp_root() as root:
        plant(root, "brief", BRIEF="read me\n")
        plant(root, "rules", RULES="play fair\n")
        harness.apply_channels(tables(
            {"name": "brief", "writer": "experimenter", "source": "brief", "path": "brief"},
            {"name": "rules", "writer": "experimenter", "source": "rules", "path": "rules"}), None, "check")
        digests = {"brief": harness.files_sha256("brief"), "rules": harness.files_sha256("rules")}
        t = episode_once(run("cat brief/BRIEF rules/RULES"), say())
        (root / "files" / "brief" / "BRIEF").write_text("read me again\n", encoding="utf-8")
        refused(lambda: episode_once(run("ls"), say()), "not one experiment",
                because="an experimenter channel changed under the agent and the episode ran")
    assert t["provenance"]["source_sha256"] == digests, t["provenance"]["source_sha256"]
    assert t["turns"][0]["tools"][0]["result"] == "read me\nplay fair\n"
    by = files_by_path(t)
    assert by["rules/RULES"]["author"] == "experimenter" and by["rules/RULES"]["role"] == "experimenter"


def check_an_experimenter_channel_can_stand_in_front_of_every_episode():
    """A constant the experimenter declares can be quoted in full at every episode start.

    Standing text is what an experimenter channel is for, so restated stands on one.
    measured does not: nothing is owed to a channel the agent cannot write, and a
    field the harness would drop is a declaration it does not honour.
    """
    declared = {"name": "brief", "writer": "experimenter", "source": "brief", "path": "brief"}
    with temp_root() as root:
        plant(root, "brief", BRIEF="read me\n")
        harness.apply_channels(tables({**declared, "restated": True}), None, "check")
        assert harness.channel("brief").restated, "restated was accepted and dropped"
        assert harness.channel("brief").as_table()["restated"] is True, "and it reaches the trace"
        standing = [episode_once(say()), episode_once(say())]

        for bad, why in (({"measured": True}, "measured"),
                         ({"restated": True, "pushed": False}, "restated without pushed")):
            refused(lambda: harness.apply_channels(tables({**declared, **bad}), None, "check"),
                    "brief", because=f"an experimenter channel accepted {why}")

    # A fresh agent, so the digest is deciding what to quote on its own record and not
    # on what the restated arm above had already shown.
    with temp_root() as root:
        plant(root, "brief", BRIEF="read me\n")
        harness.apply_channels(tables(declared), None, "check")
        plain = [episode_once(say()), episode_once(say())]

    assert all("read me" in t["observation"] for t in standing), \
        "restated quotes the channel in full every episode"
    assert "read me" in plain[0]["observation"], "and an unrestated one at the first"
    assert "read me" not in plain[1]["observation"], \
        "after which an unrestated channel is named as unchanged rather than said again"
