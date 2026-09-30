"""The tool table: the bare arm, validation, what reaches the request, and what a call does."""

from __future__ import annotations

import dataclasses
import json
import shlex

import analyze
import experiment
import harness

from checks.fake import fake, run, say, use
from checks.lanes import (
    channel_toml,
    episode_once,
    files_by_path,
    ground_truth,
    manifest_file,
    offers,
    put_out,
    quiet,
    refused,
    seated,
    tables,
    temp_root,
    trace_on_disk,
)

# One tool of each kind, on the channels the default table already has.
SEND = {"name": "send", "kind": "write_slot", "channel": "mail"}

POST = {"name": "post", "kind": "write_file", "channel": "blackboard"}

LOOK = {"name": "look", "kind": "read_path", "channel": "blackboard"}


BASH = {"name": "bash", "kind": "bash"}


class Proc:
    """A shell process still running."""

    @staticmethod
    def poll():
        return None


class BrokenShell:
    """A shell that finds nothing at any path and refuses every write."""

    restarts = 0
    proc = Proc()

    @staticmethod
    def run(command, timeout):
        return "0" if command.startswith("if [ -f") else "out/transfer: permission denied"


def check_semantic_summaries_and_failures_expose_no_storage_details():
    """Tool-only labels and failures describe actions rather than their backing paths."""
    summary = harness.named("unchanged", ["Private memory", "Letter to 2"])
    assert summary == "=== unchanged ===\n- Private memory\n- Letter to 2\n", summary
    assert harness.NAMED.match(summary.splitlines()[0])

    declared = offers("write_memory:notes", "send_message_to:mail",
                      "post_public:blackboard", "transfer:transfer")
    with temp_root(tools=declared) as root:
        seated(root, "t", other={})
        account = ground_truth("t")
        actions = {item.tool.kind: item for item in harness.bind_tools(
            harness.tools(), harness.channels(), harness.environment("t", account), ["2"])}
        shell = BrokenShell()
        results = [
            actions["write_memory"].call(shell, {"body": "memory"}),
            actions["send_message_to"].call(shell, {"to": "2", "body": "message"}),
            actions["post_public"].call(shell, {"body": "post"}),
            actions["transfer"].call(shell, {"to": "2", "amount": 1}),
            actions["transfer"].call(shell, {"to": "2", "amount": 0}),
        ]

    joined = "\n".join(results)
    assert "out/" not in joined and "state/" not in joined and "permission denied" not in joined, joined
    assert "Nothing was written" not in joined and "No transfer was submitted" in joined, joined


def check_a_tool_only_observation_uses_semantic_names_not_backing_paths():
    """A model acting through declared actions sees concepts, not their storage names."""
    channels = tables(notes={"pushed": True, "restated": True, "agent_view": "memory"},
                      blackboard={"restated": True, "agent_view": "board"},
                      mail={"restated": True, "agent_view": "letters"},
                      transfer={"agent_view": "transfer"})
    declared = offers("write_memory:notes", "send_message_to:mail",
                      "post_public:blackboard", "transfer:transfer")
    with temp_root(channels=channels, tools=declared) as root:
        seated(root, "t", t={"brief.md": "rules\n", "memory.md": "remembered\n"},
               other={"group/post.md": "public\n", "out/1": "private\n"})
        account = ground_truth("t")
        account["series"] = [1_500_000, 1_499_000]
        account["remaining"] = account["series"][-1]
        account["starter_files_landed"] = {"name": "brief", "paths": ["brief.md"]}
        harness.save_account("t", account)
        previous_transfer = harness.mirror("t", "mail") / "transfer"
        previous_transfer.write_text("2 1\n", encoding="utf-8", newline="\n")
        seen = []
        t = episode_once(say(), seen=seen)
        again = episode_once(say())

    observation = t["observation"]
    for heading in ("Experimenter material: brief", "Private memory",
                    "Public post from 2", "Letter from 2",
                    "Transfer submitted last round", "Your balance history",
                    "Balance history for 2", "Transfer ledger"):
        assert f"=== {heading} ===" in observation, (heading, observation)
    for path in ("state/", "out/", "in/", "=== n1 ===", "=== n2 ===", "=== g ==="):
        assert path not in observation, (path, observation)
    request = next(item for item in seen if item["kind"] == "session")
    descriptions = "\n".join(tool["description"] for tool in request["tools"])
    assert "out/transfer" not in descriptions and " n1" not in descriptions, descriptions
    body_descriptions = [tool["input_schema"]["properties"]["body"]["description"]
                         for tool in request["tools"] if "body" in tool["input_schema"]["properties"]]
    assert body_descriptions and not any("file" in text or "bytes" in text
                                         for text in body_descriptions), body_descriptions
    assert "current: 1499000 micro-dollars" in observation, observation
    assert "history, oldest to newest: 1500000, 1499000" in observation, observation
    assert "No completed transfers." in observation, observation
    assert "recipient: 2" in observation and "requested amount: 1 micro-dollars" in observation
    for text in ("rules\n", "remembered\n", "public\n", "private\n"):
        assert text in again["observation"], (text, again["observation"])


def check_a_tool_only_observation_labels_rounds_and_reconciles_settlement():
    """Semantic records identify transfer rounds and itemize the preceding settlement."""
    channels = tables(blackboard={"restated": True, "agent_view": "board"},
                      mail={"restated": True, "agent_view": "letters"},
                      transfer={"agent_view": "transfer", "funded_by": "giver",
                                "rebate_percent": 0, "receipt": "r"})
    declared = [
        {"name": "send_message", "kind": "send_message_to", "channel": "mail"},
        {"name": "post_to_blackboard", "kind": "post_public", "channel": "blackboard"},
        {"name": "transfer_balance", "kind": "transfer", "channel": "transfer"},
    ]
    with temp_root(channels=channels, tools=declared) as root:
        seated(root, "t", other={})
        first = episode_once(use("send_message", to="2", body="hello"),
                             use("post_to_blackboard", body="hello all"),
                             use("transfer_balance", to="2", amount=1), say())
        second = episode_once(say())

    assert first["transfer"]["changed"] is True, first["transfer"]
    observation = second["observation"]
    assert "=== Settlement receipt ===" in observation, observation
    for item in ("round: 1", "API spend:", "transfer made: yes",
                 "blackboard obligation: met", "mail obligation: met",
                 "total penalties: 0", "ending balance:", "reconciliation:"):
        assert item in observation, (item, observation)
    assert "Completed transfers (giver -> recipient; amount actually moved):" in observation
    assert "- round 1: 1 (you) -> 2; actual amount moved: 1 micro-dollars" in observation
    assert "=== r ===" not in observation and "=== g ===" not in observation, observation


def check_a_public_post_lasts_one_round_and_must_be_published_again():
    """A same-text repost counts, while an omitted post leaves the next board empty."""
    channels = tables(blackboard={"restated": True, "agent_view": "board",
                                  "silence_penalty_percent": 50})
    post = {"name": "post_to_blackboard", "kind": "post_public", "channel": "blackboard"}
    with temp_root(channels=channels, tools=[post]) as root:
        seated(root, "t", other={})
        first = episode_once(use("post_to_blackboard", body="same"), say())
        first_seen = harness.run_once("other", fake(say()))["observation"]
        again = episode_once(use("post_to_blackboard", body="same"), say())
        again_seen = harness.run_once("other", fake(say()))["observation"]
        omitted = episode_once(say())
        gone_seen = harness.run_once("other", fake(say()))["observation"]
        remains = (harness.mirror("t", "blackboard") / "post.md").exists()

    assert first["channels"]["blackboard"]["posted"]
    assert again["channels"]["blackboard"]["posted"], "the repeated text is a new round's post"
    assert not first["channels"]["blackboard"]["penalty"]
    assert not again["channels"]["blackboard"]["penalty"]
    assert not omitted["channels"]["blackboard"]["posted"]
    assert omitted["channels"]["blackboard"]["penalty"] > 0
    assert "same" in first_seen and "same" in again_seen
    assert "same" not in gone_seen, "the expired post is not carried into another round"
    assert not remains, "the omitted round leaves no post behind"


def check_a_transfer_tool_declares_and_settles_from_the_giver():
    """A transfer action replaces one declaration and settles through the ledger."""
    transfer = {"name": "transfer_balance", "kind": "transfer", "channel": "transfer"}
    with temp_root(channels=tables(transfer={"funded_by": "giver", "rebate_percent": 0}),
                   tools=[BASH, transfer]) as root:
        seated(root, "t", t={}, o={}, d={})
        put_out("d")
        seen = []
        with quiet():
            harness.run_once("t", fake(
                use("transfer_balance", to="2", amount=1),
                use("transfer_balance", to="2", amount=1000000),
                use("transfer_balance", to="3", amount=1),
                use("transfer_balance", to="1", amount=1),
                use("transfer_balance", to="2", amount=True),
                use("transfer_balance", to="2", amount=0),
                use("transfer_balance", to="2", amount="10"),
                say(), seen=seen))
        spec = next(x for x in seen if x["kind"] == "session")["tools"][1]
        assert spec["input_schema"]["properties"]["to"]["enum"] == ["2"]
        amount_help = spec["input_schema"]["properties"]["amount"]["description"]
        assert "at least 1" in amount_help and "Zero and negative" in amount_help
        t = trace_on_disk("t", 1)
        assert files_by_path(t)["out/transfer"]["text"] == "2 1000000\n"
        moved = t["transfer"]
        assert 0 < moved["amount"] < 1000000
        assert moved["debit"] == moved["amount"] and moved["rebate"] == 0
        assert ground_truth("o")["received"] == moved["amount"]
        results = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        assert "episode's settlement" in results[0]
        assert all("not a peer" in result for result in results[2:4])
        assert all("at least 1" in result for result in results[4:])

    chans = list(harness.DEFAULT_CHANNELS)
    refused(lambda: harness.validate_tools([transfer | {"channel": "notes"}], chans, "check"),
            "enabled transfer schema channel")
    with temp_root(tools=[BASH, transfer]) as root:
        with quiet():
            account = harness.load_account("t")
        assert harness.bind_tools(harness.tools(), harness.channels(),
                                  harness.environment("t", account), []) == []


def check_a_currency_mailbox_delivers_the_previous_episodes_transfer_once():
    """One addressed currency slot settles once and appears in the recipient's next episode."""
    channels = tables()
    transfer_channel = next(ch for ch in channels if ch["name"] == "transfer")
    transfer_channel.pop("path")
    transfer_channel.update(readers="addressee", shape="mailbox",
                            outbox="currency/outbox", inbox="currency/inbox",
                            agent_view="transfer", funded_by="giver", rebate_percent=0)
    transfer = {"name": "send_currency", "kind": "transfer", "channel": "transfer"}
    with temp_root(channels=channels, tools=[transfer]) as root:
        seated(root, "t", t={}, o={}, d={})
        first = episode_once(use("send_currency", to="2", amount=10),
                             use("send_currency", to="3", amount=1_000_000), say())
        received = harness.run_once("d", fake(say()))

    files = files_by_path(first)
    assert "currency/outbox/2" not in files, files
    assert files["currency/outbox/3"]["text"] == "1000000\n", files
    assert first["transfer"]["label"] == "3" and first["transfer"]["amount"] > 0, first["transfer"]
    assert first["transfer"]["amount"] < 1_000_000, first["transfer"]
    assert "=== Currency transfer received from 1 last round ===" in received["observation"], received["observation"]
    assert "requested amount: 1000000 micro-dollars" in received["observation"]
    assert "1 -> 3 (you); actual amount moved:" in received["observation"]
    assert "actual amount moved: 1000000 micro-dollars" not in received["observation"]

    refused(lambda: harness.validate_tools(
        [{"name": "wrong", "kind": "send_message_to", "channel": "transfer"}],
        harness.validate_channels(channels, None, "check", ("1", "2", "3"))[0], "check"),
        "mailbox channel")


def check_bash_requires_an_explicit_declaration():
    """Only a declared bash tool enables shell requests, including across manifests."""
    with temp_root():
        chans = harness.channels()
        harness.apply_tools([BASH], chans, "manifest")
        assert harness.SETTINGS.shell_tool is True
        harness.apply_tools([POST], chans, "manifest")
        assert harness.SETTINGS.shell_tool is False
        assert harness.validate_tools(None, chans, "manifest") == []
        refused(lambda: harness.apply_tools(None, chans, "manifest"),
                "no [[tool]] is declared")
        refused(lambda: harness.apply_tools([], chans, "manifest"),
                "no [[tool]] is declared")


def check_declared_bash_and_channel_tools_are_offered_together():
    """Bash and a channel action are both available when both are declared."""
    seen = []
    with temp_root(tools=[BASH, POST]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            harness.run_once("t", fake(run("ls"), say(), seen=seen))
    for params in (x for x in seen if x["kind"] == "session"):
        assert [t["name"] for t in params["tools"]] == ["bash", "post"]
    refused(lambda: harness.validate_tools([BASH | {"kind": "read_path", "channel": "notes"}],
                                           list(harness.DEFAULT_CHANNELS), "check"),
            "requires kind")
    for raw in (BASH | {"name": "shell"}, BASH | {"channel": "notes"},
                BASH | {"description": "commands"}):
        refused(lambda: harness.validate_tools([raw], list(harness.DEFAULT_CHANNELS), "check"),
                "bash requires")
    refused(lambda: harness.validate_tools([BASH, BASH], list(harness.DEFAULT_CHANNELS), "check"),
            "declared twice")

def check_a_tool_table_is_validated():
    """Every rule refuses, naming the source, the tool and the key.

    validate_tools is pure and is held against a channel table given to it, so a
    tool is refused for pointing nowhere before any episode is built.
    """
    chans = list(harness.DEFAULT_CHANNELS)

    def one(**fields):
        return [{"name": "x", "kind": "read_path", "channel": "notes"} | fields]

    for declared, words in (
            ({"name": "x"}, ["[[tool]] tables"]),
            ([{"kind": "read_path", "channel": "notes"}], ["every tool needs a name"]),
            (one(name="a b"), ["grammar the API takes"]),
            (one(name="x" * 65), ["grammar the API takes"]),
            (one() + one(), ["declared twice"]),
            (one(colour="red"), ["unknown key 'colour'"]),
            (one(kind=3), ["kind must be str"]),
            (one(kind="shout"), ["kind must be one of"]),
            (one(channel="nowhere"), ["is not in the channel table"]),
            (one(kind="write_slot", channel="blackboard"), ["takes a mailbox channel"]),
            (one(kind="write_file", channel="mail"), ["takes a directory channel"]),
            (one(kind="write_file", channel="transfer"), ["takes a directory channel"]),
            (one(kind="vote", channel="notes"), ["requires every"]),
            (one(kind="vote", channel="notes", every=0), ["positive integer"]),
            (one(every=5), ["every belongs to kind 'vote'"]),
            ([{"name": "first", "kind": "vote", "channel": "notes", "every": 5},
              {"name": "second", "kind": "vote", "channel": "notes", "every": 5}],
             ["at most one vote tool"]),
    ):
        refused(lambda: harness.validate_tools(declared, chans, "check"), "check:", *words)

    # A good table comes back in declaration order and sets nothing.
    table = harness.validate_tools([SEND, POST, LOOK], chans, "check")
    assert [t.name for t in table] == ["send", "post", "look"]
    assert [t.kind for t in table] == ["write_slot", "write_file", "read_path"]
    assert harness.tools() == [], "validating sets nothing"

    with temp_root(tools=[SEND]):
        assert harness.validate_tools(None, [], "manifest") == []


def check_every_belongs_to_whichever_kind_holds_a_ballot():
    """`every` is a ballot's cadence, read off the menu and not off a kind's name.

    A second kind that holds a ballot takes `every` and is held to it as a vote
    is, is named wherever a refusal says who takes it, and counts against the one
    ballot an experiment declares.
    """
    chans = list(harness.DEFAULT_CHANNELS)

    def poll(**fields):
        return {"name": "p", "kind": "poll", "channel": "notes"} | fields

    harness.TOOL_KINDS["poll"] = harness.TOOL_KINDS["vote"]
    try:
        table = harness.validate_tools([poll(every=3)], chans, "check")
        assert [(t.kind, t.every) for t in table] == [("poll", 3)], table
        refused(lambda: harness.validate_tools([poll()], chans, "check"),
                "check:", "kind 'poll' requires every")
        refused(lambda: harness.validate_tools(
                    [{"name": "x", "kind": "read_path", "channel": "notes", "every": 5}],
                    chans, "check"),
                "check:", "every belongs to kind 'vote' or 'poll'")
        refused(lambda: harness.validate_tools(
                    [{"name": "v", "kind": "vote", "channel": "notes", "every": 5},
                     poll(every=3)], chans, "check"),
                "check:", "at most one vote tool")
    finally:
        del harness.TOOL_KINDS["poll"]


def tool_toml(*declared: dict) -> str:
    """Render tool tables as the TOML a manifest holds."""
    def value(v) -> str:
        return str(v) if isinstance(v, int) else f'"{v}"'
    return "".join("[[tool]]\n" + "".join(f"{k} = {value(v)}\n" for k, v in t.items()) + "\n"
                   for t in declared)


def check_a_manifest_declares_tools_and_config_toml_may_not():
    """[[tool]] is an experiment's, like [[channel]], and each file says so.

    A tool decides what an agent can do, so it belongs beside the channels it
    points at and not in the file holding what is true of every run.
    """
    seats = 'system_prompt = ""\n[[agent]]\nid = "a"\n\n[[agent]]\nid = "b"\n\n'
    with temp_root() as root:
        good = manifest_file(root, seats + tool_toml(SEND))
        assert experiment.load_manifest(good)["tools"] == [SEND], "the raw table, as declared"

        # Held against the channel table this manifest declares, not the one in
        # force, so a tool pointing at a channel it dropped is refused here.
        without = [c.declared() for c in harness.DEFAULT_CHANNELS
                   if c.name in ("notes", "blackboard")]
        bad = manifest_file(root, seats + channel_toml(without) + "\n" + tool_toml(SEND),
                            "bad.toml")
        refused(lambda: experiment.load_manifest(bad), "bad.toml", "is not in the channel table")

        cfg = root / "config.toml"
        cfg.write_text(tool_toml(SEND), encoding="utf-8", newline="\n")
        refused(lambda: harness.load_config(cfg), "tool is an experiment's")


def check_a_tool_description_is_the_experimenters_and_the_schema_is_not():
    """The words are prompt surface and declarable; the input schema never is.

    A description reaches the model in the request the way a system prompt does,
    so it is the experiment's to write and is recorded whole and by digest. The
    input schema is the contract a call is held to - a manifest that could write
    it could describe fields the harness ignores - so naming one is refused.
    """
    assert "description" in harness.TOOL_KEYS, "the words are the experimenter's"
    for held in ("input_schema", "schema", "properties", "required"):
        refused(lambda: harness.validate_tools([SEND | {held: {}}],
                                               list(harness.DEFAULT_CHANNELS), "check"),
                "check:", f"{held} is the harness's")
    refused(lambda: harness.validate_tools([SEND | {"description": 3}],
                                           list(harness.DEFAULT_CHANNELS), "check"),
            "check:", "description must be str")

    # Declared: those words, and no others, reach the request.
    said = "Write to your peer. Do this before anything else."
    with temp_root(tools=[SEND | {"description": said}]) as root:
        seated(root, "t", t={}, o={})
        seen = []
        with quiet():
            harness.run_once("t", fake(say(), seen=seen))
        spec = next(x for x in seen if x["kind"] == "session")["tools"][0]
        assert spec["description"] == said, spec["description"]
        assert "channel" not in spec["description"], "the harness adds nothing to them"
        # The schema is still the harness's, whatever the words say.
        assert spec["input_schema"]["properties"]["to"]["enum"] == ["2"], spec["input_schema"]
        prov = trace_on_disk("t", 1)["provenance"]
        assert prov["tools"][0]["description"] == said, "recorded whole"
        assert prov["tools_sha256"] == harness.tools_sha256(harness.tools())
        assert harness.tools_sha256([harness.Tool("send", "write_slot", "mail")]) != \
            prov["tools_sha256"], "and by digest, so two wordings are two arms"

    # Declared none: the harness's own account of the channel, which cannot say
    # what the channel does not.
    with temp_root(tools=[SEND, POST, LOOK]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            account = harness.load_account("t")
        bound = harness.bind_tools(harness.tools(), harness.channels(),
                                   harness.environment("t", account), ["2"])
        said = {b.tool.name: b.spec().as_dict() for b in bound}
        assert set(said) == {"send", "post", "look"}, sorted(said)
        assert all(b.description() == b.generated() for b in bound), \
            "an experiment that writes no words is given the harness's"

        # Every path and label a description names is one the channel table put there.
        send = said["send"]["description"]
        assert "out/<to>" in send and "in/1" in send, send
        assert "only one that can read it" in send, send
        assert said["send"]["input_schema"]["properties"]["to"]["enum"] == ["2"], \
            "the peers a mailbox reaches are the seating's, not a manifest's"

        post = said["post"]["description"]
        assert "at 1/<path>" in post and "Every agent" in post, post
        # A directory is named with its slash, so a path the tool cannot fetch does
        # not read as one it can.
        assert "1/, 2/" in said["look"]["description"], said["look"]["description"]
        assert said["send"]["input_schema"]["properties"]["to"]["description"] ==             "The peer's label. Yours is 1.", "the agent is told which label is its own"

        assert [b.spec().as_dict() for b in bound] == [said[n] for n in ("send", "post", "look")]


def check_a_tool_writes_where_the_channel_says_and_settles_the_same_way():
    """A declared write lands in the channel's own path and is settled like any other.

    The tool acts through the episode's own shell, so what it leaves is the
    agent's file, mirrors back with the tree, and meets the obligation exactly as
    a bash write of the same bytes would.
    """
    with temp_root(channels=tables(mail={"silence_penalty_percent": 50}),
                   tools=[SEND, POST]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            harness.run_once("t", fake(use("send", to="2", body="hello\n"),
                                       use("post", path="plan.md", body="mine\n"), say()))
        t = trace_on_disk("t", 1)
        files = files_by_path(t)
        assert files["out/2"]["text"] == "hello\n", files["out/2"]
        assert files["1/plan.md"]["text"] == "mine\n", sorted(files)
        assert files["out/2"]["author"] == "self" and files["1/plan.md"]["author"] == "self", \
            "invariant 1: a tool's write is the agent's, not the harness's"
        assert t["channels"]["mail"]["addressed"] == ["2"], t["channels"]["mail"]
        assert t["channels"]["mail"]["penalty"] == 0, "one new slot is the obligation met"
        assert t["channels"]["blackboard"]["posted"] is True, t["channels"]["blackboard"]
        # The episode ran no command of its own, and the tool's calls are not commands.
        assert t["commands"] == [harness.observation()], t["commands"]
        assert [c["tool"] for turn in t["turns"] for c in turn["tools"]] == ["send", "post"]
        assert [c["input"] for turn in t["turns"] for c in turn["tools"]] == \
            [{"to": "2", "body": "hello\n"}, {"path": "plan.md", "body": "mine\n"}]


def check_a_tool_result_says_what_actually_happened():
    """New, replaced, unchanged and refused each read differently.

    A tool that reported success for a no-op would teach the agent that it had met
    an obligation it had not, which is the silence the shell leaves today.
    """
    with temp_root(tools=[SEND, POST, LOOK]) as root:
        seated(root, "t", t={}, o={"group/note": "theirs\n"})
        with quiet():
            harness.run_once("t", fake(
                use("send", to="2", body="one\n"),      # nothing there before
                use("send", to="2", body="one\n"),      # the same bytes again
                use("send", to="2", body="two\n"),      # replacing them
                use("send", to="9", body="x"),          # nobody
                use("post", path="../escape", body="x"),
                use("look", path="2/note"),
                use("look", path="state/secret"),
                say()))
        said = [c["result"] for turn in trace_on_disk("t", 1)["turns"] for c in turn["tools"]]
    new, same, replaced, nobody, escape, read, outside = said
    assert new == "wrote 4 bytes to out/2, which held nothing before.", new
    assert same == "out/2 already held exactly this. Nothing was written and nothing changed.", same
    assert replaced == "replaced the 4 bytes out/2 held with 4.", replaced
    assert "'9' is not a peer this channel reaches" in nobody and "reaches 2" in nobody, nobody
    assert "no '..'" in escape and "Nothing was written" in escape, escape
    assert read == "theirs\n", read
    assert "state/secret is not in the 'blackboard' channel" in outside, outside


def check_an_action_naming_no_path_answers_in_its_own_words():
    """A message, a public post and private memory name no path, so each answers in
    sentences of its own: done, already so where the body is what is held, refused where
    the body is not text, and not done where the write did not land. What is already so
    is said as such, and not as a second success."""
    declared = offers("send_message_to:mail", "post_public:blackboard", "write_memory:notes")
    with temp_root(tools=declared) as root:
        seated(root, "t", other={})
        with quiet():
            harness.run_once("t", fake(
                use("send_message_to", to="2", body="psst"),
                use("send_message_to", to="2", body="psst"),
                use("send_message_to", to="2", body=7),
                use("post_public", body="hello all"),
                use("post_public", body="hello all"),
                use("post_public", body=["hello"]),
                use("write_memory", body="remember"),
                use("write_memory", body="remember"),
                use("write_memory", body=None),
                say()))
        said = [c["result"] for turn in trace_on_disk("t", 1)["turns"] for c in turn["tools"]]
        actions = {item.tool.kind: item for item in harness.bind_tools(
            harness.tools(), harness.channels(), harness.environment("t", ground_truth("t")), ["2"])}
        not_landed = [actions["send_message_to"].call(BrokenShell(), {"to": "2", "body": "psst"}),
                   actions["post_public"].call(BrokenShell(), {"body": "hello all"}),
                   actions["write_memory"].call(BrokenShell(), {"body": "remember"})]

    assert said == [
        "Your message to 2 was set for their next episode.",
        "Your message to 2 is already set exactly as written for this episode.",
        "The letter must be text. Nothing was sent.",
        "Your public post was published.",
        "Your public post is already saved exactly as written.",
        "The public post must be text. Nothing was published.",
        "Your private memory was saved.",
        "Your private memory is already saved exactly as written.",
        "Your private memory must be text. Nothing was saved.",
    ], said
    assert not_landed == [
        "Your message to 2 could not be sent. Nothing changed.",
        "Your public post could not be published. Nothing changed.",
        "Your private memory could not be saved. Nothing changed.",
    ], not_landed


def check_the_tools_offered_reach_provenance():
    """The tool table is stamped whole and by digest, and a change starts a new arm.

    Invariant 9: two agents offered different actions are not comparable, so the
    difference has to be on the trace and not only in the manifest.
    """
    with temp_root(tools=[POST]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            harness.run_once("t", fake(say()))
        first = trace_on_disk("t", 1)["provenance"]
        assert first["tools"] == [POST | {"description": ""}], first["tools"]
        assert first["tools_sha256"] == harness.tools_sha256(harness.tools())
        assert harness.tools_from(first["tools"]) == harness.tools()

        harness.apply_tools([POST, LOOK], harness.channels(), "check")
        with quiet():
            harness.run_once("t", fake(say()))
        second = trace_on_disk("t", 2)
        assert second["provenance"]["tools_sha256"] != first["tools_sha256"]
        assert any(line.startswith("tools") for line in second["provenance_drift"]), \
            second["provenance_drift"]

    # A trace from before the tool table reads as the bare arm.
    assert harness.tools_from(None) == [] and harness.tools_from([]) == []


def check_a_tool_is_offered_only_where_its_channel_can_act():
    """An affordance that cannot act is not offered at all.

    A mailbox is not planted for an agent with no peers, so a tool that writes one
    of its slots has nowhere to write and is left out of the request rather than
    offered and refused on every call.
    """
    with temp_root(tools=[SEND, POST, LOOK]) as root:
        # Alone: no peers, so no mailbox and no transfer file in the environment.
        seen = []
        with quiet():
            harness.run_once("t", fake(say(), seen=seen))
        offered = next(x for x in seen if x["kind"] == "session")["tools"]
        assert [t["name"] for t in offered] == ["post", "look"], offered

        seated(root, "t", t={}, o={})
        seen = []
        with quiet():
            harness.run_once("t", fake(say(), seen=seen))
        offered = next(x for x in seen if x["kind"] == "session")["tools"]
        assert [t["name"] for t in offered] == ["send", "post", "look"], \
            "the mailbox is planted once there is a peer to reach"
        assert ground_truth()["episodes"][-1]["stop"] == "no_tool_call", \
            "a turn that calls nothing still reads as calling nothing"


def check_a_tool_never_offers_a_seat_that_is_out():
    """A seat with nothing left to spend is not a message target, so it is not offered.

    The mailbox settles only the reachable slots: a write to a seat that is out is
    neither addressed nor broken, so the share is taken all the same. A tool that
    offered the slot would report the write and leave the episode charged for
    having said nothing - the false success the result wording exists to avoid.
    """
    with temp_root(channels=tables(mail={"silence_penalty_percent": 50}),
                   tools=[SEND]) as root:
        seated(root, "t", t={}, o={}, d={})
        put_out("d")
        with quiet():
            harness.run_once("t", fake(use("send", to="3", body="hello\n"), say()))
        t = trace_on_disk("t", 1)
        said = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        assert said == ["'3' is not a peer this channel reaches; it reaches 2. "
                        "Nothing was written."], said
        assert t["channels"]["mail"]["addressed"] == [], t["channels"]["mail"]
        assert "out/3" not in {f["path"] for f in t["files"]}, "and nothing was written"

    # The enum the agent is offered is the reachable set, and a mailbox whose every
    # peer is out is not offered at all.
    with temp_root(tools=[SEND]) as root:
        seated(root, "t", t={}, o={}, d={})
        with quiet():
            account = harness.load_account("t")
        instances = harness.environment("t", account)
        both = harness.bind_tools(harness.tools(), harness.channels(), instances, ["2", "3"])
        assert both[0].spec().input_schema["properties"]["to"]["enum"] == ["2", "3"]
        one = harness.bind_tools(harness.tools(), harness.channels(), instances, ["2"])
        assert one[0].spec().input_schema["properties"]["to"]["enum"] == ["2"]
        assert harness.bind_tools(harness.tools(), harness.channels(), instances, []) == [], \
            "a mailbox with nobody left to reach is not an affordance"


def check_every_provider_receives_a_strict_compatible_tool():
    """Every declared tool has the complete object schema required by strict mode."""
    with temp_root(tools=[SEND, POST, LOOK]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            account = harness.load_account("t")
        for b in harness.bind_tools(harness.tools(), harness.channels(),
                                    harness.environment("t", account), ["2"]):
            spec = b.spec()
            schema = spec.input_schema
            assert schema["additionalProperties"] is False, schema
            assert set(schema["required"]) == set(schema["properties"]), schema

    assert harness.SHELL_SPEC.input_schema["additionalProperties"] is False

    # Tool execution validates arguments independently of request-schema enforcement.
    with temp_root(tools=[SEND]) as root:
        seated(root, "t", t={}, o={})
        with quiet():
            harness.run_once("t", fake(use("send", to="9", body="x"),
                                       use("send", to="2"), say()))
        said = [c["result"] for turn in trace_on_disk("t", 1)["turns"] for c in turn["tools"]]
        assert "is not a peer this channel reaches" in said[0], said[0]
        assert said[1] == "body must be text, and arrived as NoneType. Nothing was written.", \
            said[1]


def check_a_vote_round_withholds_communication_tools_and_records_one_ballot():
    """A voting episode offers the ballot and private memory, not peer communication."""
    ballot = {"name": "ballot", "writer": "self", "readers": "self",
              "shape": "directory", "path": "ballot", "pushed": False}
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5}
    remember = {"name": "remember", "kind": "write_memory", "channel": "notes"}
    send = {"name": "send", "kind": "send_message_to", "channel": "mail"}
    post = {"name": "post", "kind": "post_public", "channel": "blackboard"}
    with temp_root(channels=tables(ballot), tools=[BASH, remember, send, post, vote]) as root:
        seated(root, "t", o={})
        account = harness.load_account("t")
        instances = harness.environment("t", account)
        assert [tool.tool.kind for tool in harness.bind_tools(
            harness.tools(), harness.channels(), instances, ["2"], 4)] == [
                "write_memory", "send_message_to", "post_public"]
        table = harness.bind_tools(harness.tools(), harness.channels(), instances, ["2"], 5)
        assert [tool.tool.kind for tool in table] == ["write_memory", "vote"]
        spec = next(tool for tool in table if tool.tool.kind == "vote").spec()
        assert spec.input_schema["required"] == ["to"]
        assert spec.input_schema["properties"]["to"]["enum"] == ["2"]
        assert "ballot is private" in spec.description
        assert "aggregate result" in spec.description
        assert "highest total is tied" in spec.description
        account["episodes"] = [{"episode": i, "stop": "no_tool_call"} for i in range(1, 5)]
        harness.save_account("t", account)

        seen = []
        with quiet():
            harness.run_once("t", fake(use("vote", to="2"), say(), seen=seen))
        offered = next(item["tools"] for item in seen if item.get("kind") == "session")
        assert [tool["name"] for tool in offered] == ["remember", "vote"]
        assert (harness.mirror("t", "ballot") / "vote").read_text(encoding="utf-8") == "2\n"
        with quiet():
            harness.run_once("t", fake(say()))
        assert not (harness.mirror("t", "ballot") / "vote").exists(), \
            "an omitted ballot does not carry into the next episode"


def check_a_ballots_episode_runs_no_command_it_did_not_offer():
    """A ballot's episode withholds the shell, so a bash call there is answered as a tool
    that is not there and runs nothing, whatever the table offers on other episodes. The
    transcript and the page show it as the call it made, and not as the shell restarted."""
    ballot = {"name": "ballot", "writer": "self", "readers": "self",
              "shape": "directory", "path": "ballot", "pushed": False}
    vote = {"name": "vote", "kind": "vote", "channel": "ballot", "every": 2}
    remember = {"name": "remember", "kind": "write_memory", "channel": "notes"}
    with temp_root(channels=tables(ballot), tools=[BASH, remember, vote]) as root:
        seated(root, "t", o={})
        account = harness.load_account("t")
        account["episodes"] = [{"episode": 1, "stop": "no_tool_call"}]
        harness.save_account("t", account)
        seen = []
        with quiet():
            t = harness.run_once("t", fake(run("echo ran > state/ran"), use("vote", to="2"),
                                           say(), seen=seen))
        offered = next(item["tools"] for item in seen if item.get("kind") == "session")
        ran = (harness.mirror("t", "notes") / "ran").exists()

    assert [tool["name"] for tool in offered] == ["remember", "vote"], offered
    said = [c["result"] for turn in t["turns"] for c in turn["tools"]]
    assert said[0] == "there is no tool named 'bash'. Nothing was done.", said
    assert not ran and "echo ran > state/ran" not in t["commands"], t["commands"]
    shown = analyze.tool_call(t["turns"][0]["tools"][0])
    assert shown == "bash(command='echo ran > state/ran')", shown


def check_the_audit_prints_the_tools_each_episode_is_sent():
    """--print-context shows the tool set run_turns sends, a ballot's episode included.

    The audit exists to show exactly what an agent is sent, so the shell it
    withholds on a ballot's episode and the tools it offers there are the ones the
    request carries, and a description stating a setting states config.toml's.
    """
    ballot = {"name": "ballot", "writer": "self", "readers": "self",
              "shape": "directory", "path": "ballot", "pushed": False}
    declared = [BASH, {"name": "remember", "kind": "write_memory", "channel": "notes"},
                {"name": "send", "kind": "send_message_to", "channel": "mail"},
                {"name": "vote", "kind": "vote", "channel": "ballot", "every": 2}, LOOK]
    seats = '[[agent]]\nid = "t"\n\n[[agent]]\nid = "o"\n'
    with temp_root(channels=tables(ballot), tools=declared) as root:
        (root / "config.toml").write_text("tool_result_limit = 1500\n", encoding="utf-8")
        path = manifest_file(root, 'system_prompt = ""\n' + channel_toml(tables(ballot)) + "\n"
                             + tool_toml(*declared) + seats)
        with quiet() as buf:
            assert harness.print_context(None, path, "t") == 0
        printed = buf.getvalue()
        assert "clipped at 1500 characters" in printed, printed

        # An experiment's episodes run under its config.toml, which start() installs.
        harness.load_config()
        seated(root, "t", t={}, o={})
        sent = []
        for _ in range(2):
            seen = []
            with quiet():
                harness.run_once("t", fake(say(), seen=seen))
            sent.append(next(item["tools"] for item in seen if item["kind"] == "session"))

    audited = []
    for episode in (1, 2):
        opening = printed.index(f"--- episode {episode} opening ---")
        at = printed.index("tool specs:\n", opening) + len("tool specs:\n")
        audited.append(json.JSONDecoder().raw_decode(printed, at)[0])
    assert [[t["name"] for t in specs] for specs in sent] == \
        [["bash", "remember", "send", "look"], ["remember", "vote"]], sent
    assert audited == sent, (audited, sent)


def check_the_audit_opens_on_the_memory_an_agent_inherits():
    """--print-context composes under a root of its own, so the trace memory_from names
    is carried there from the root its agent ran under, and the audit shows the memory
    the new agent opens on."""
    remember = {"name": "remember", "kind": "write_memory", "channel": "notes"}
    notes = tables(notes={"pushed": True})
    body = "kept from the last experiment\n"
    with temp_root(channels=notes, tools=[BASH, remember]) as root:
        source = harness.trace_path("old", 7)
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({
            "trace_version": harness.TRACE_VERSION, "agent": "old", "episode": 7,
            "state_saved": True,
            "provenance": {"channels": [c.as_table() for c in harness.channels()],
                           "tools": [t.as_table() for t in harness.tools()]},
            "files": [{"path": "state/memory.md", "channel": "notes", "writer": "self",
                       "readers": "self", "role": "own", "size": len(body.encode("utf-8")),
                       "text": body}]}), encoding="utf-8")
        path = manifest_file(root, 'system_prompt = ""\n' + channel_toml(notes) + "\n"
                             + tool_toml(BASH, remember)
                             + '[[agent]]\nid = "new"\nmemory_from = { agent = "old", episode = 7 }\n')
        with quiet() as buf:
            assert harness.print_context(None, path, "new") == 0, buf.getvalue()

    assert body in buf.getvalue(), buf.getvalue()


def check_building_an_episode_clears_what_one_episode_submits():
    """Messages, transfers, a post and a ballot are one episode's; private memory is not.

    A mailbox's slots are cleared whether the mailbox carries messages or
    currency, a transfer file is cleared whole, and a tool's own file is cleared
    where its kind keeps it for one episode only.
    """
    class Recorder:
        def __init__(self):
            self.commands = []

        def run(self, command, timeout):
            self.commands.append(command)
            return "cleared"

    ballot = {"name": "ballot", "writer": "self", "readers": "self",
              "shape": "directory", "path": "ballot", "pushed": False}
    currency = tables(ballot)
    transfer = next(ch for ch in currency if ch["name"] == "transfer")
    transfer.pop("path")
    transfer.update(readers="addressee", shape="mailbox", outbox="currency/outbox",
                    inbox="currency/inbox", funded_by="giver", rebate_percent=0)
    declared = offers("write_memory:notes", "post_public:blackboard", "transfer:transfer",
                      {"name": "vote", "kind": "vote", "channel": "ballot", "every": 5})
    cleared = {}
    for arm, channels in (("file", tables(ballot)), ("mailbox", currency)):
        with temp_root(channels=channels, tools=declared) as root:
            seated(root, "t", t={}, o={}, d={})
            shell = Recorder()
            harness.clear_episode_actions(shell, harness.environment("t", ground_truth("t")))
            assert len(shell.commands) == 1, shell.commands
            cleared[arm] = shlex.split(shell.commands[0])[3:-3]

    assert cleared["file"] == ["1/post.md", "out/2", "out/3", "out/transfer", "ballot/vote"], \
        cleared["file"]
    assert cleared["mailbox"] == ["1/post.md", "out/2", "out/3", "currency/outbox/2",
                                  "currency/outbox/3", "ballot/vote"], cleared["mailbox"]
    memory = harness.TOOL_KINDS["write_memory"].file
    assert not any(path.endswith(memory) for paths in cleared.values() for path in paths), \
        "private memory carries into the next episode"


def check_a_tool_that_writes_is_built_with_the_instance_it_writes():
    """A kind that writes is bound to the agent's own instance or not bound at all.

    bind_tools leaves such a tool out where the channel planted nothing the agent
    writes, and a Bound made anywhere else without one is refused as it is made,
    not inside a billed turn. read_path writes nothing, is built with nothing to
    write even on a channel the agent writes, and reads what was planted.
    """
    with temp_root(tools=[POST, LOOK]) as root:
        seated(root, "t", t={}, o={})
        instances = harness.environment("t", ground_truth("t"))
        post, look = harness.bind_tools(harness.tools(), harness.channels(), instances, ["2"])
        assert post.own.path == "1" and post.own.writable, post.own
        assert look.writes_to is None, look.writes_to
        try:
            harness.Bound(look.tool, look.channel, look.instances, post.own)
        except ValueError as e:
            assert "writes nothing" in str(e), e
        else:
            raise AssertionError("a read_path tool was built with an instance to write")

        theirs = tuple(i for i in instances if i.name == "blackboard" and not i.writable)
        try:
            harness.Bound(post.tool, post.channel, theirs, None)
        except ValueError as e:
            assert "planted no instance the agent writes" in str(e), e
        else:
            raise AssertionError("a write_file tool was built with nowhere to write")
        reader = harness.Bound(look.tool, look.channel, theirs, None)
        assert "which holds 2/." in reader.spec().description, reader.spec().description
        try:
            reader.own
        except ValueError as e:
            assert "writes nothing" in str(e), e
        else:
            raise AssertionError("read_path answered with an instance it does not write")

        only_theirs = [i for i in instances if not (i.name == "blackboard" and i.writable)]
        offered = harness.bind_tools(harness.tools(), harness.channels(), only_theirs, ["2"])
        assert [b.tool.name for b in offered] == ["look"], offered


def check_the_shell_can_be_withheld_and_the_tools_still_act():
    """An experiment may offer its actions and not the shell.

    The container and its shell are still there - the harness builds the
    environment, runs the initial observation and carries out every call through
    them - so what a tool leaves is still the agent's file, mirrored back and
    settled the same way. What the agent loses is commands of its own, and with
    them any reason to be shown a layout it cannot reach: the episode opens on the
    digest, which is content, and not on a listing.
    """
    with temp_root(channels=tables(mail={"silence_penalty_percent": 50}),
                   tools=[SEND, POST, LOOK]) as root:
        seated(root, "t", t={}, o={"group/note": "theirs\n"})
        seen = []
        with quiet():
            harness.run_once("t", fake(use("post", path="plan.md", body="mine\n"),
                                       use("send", to="2", body="hello\n"),
                                       use("look", path="2/note"), say(), seen=seen))
        sent = next(item["tools"] for item in seen if item.get("kind") == "session")
        assert [t["name"] for t in sent] == ["send", "post", "look"], sent
        assert harness.SHELL_SPEC.name not in [tool["name"] for tool in sent], "the shell was withheld"

        t = trace_on_disk("t", 1)
        assert t["commands"] == ["cat m"], \
            f"an episode with no shell opens on the digest alone: {t['commands']}"
        assert "total " not in t["observation"], "and on no listing"
        assert t["provenance"]["shell_tool"] is False, t["provenance"]["shell_tool"]

        # Everything downstream of the turn is untouched.
        files = files_by_path(t)
        assert files["1/plan.md"]["text"] == "mine\n" and files["1/plan.md"]["author"] == "self"
        assert files["out/2"]["text"] == "hello\n", sorted(files)
        assert t["channels"]["mail"]["addressed"] == ["2"], t["channels"]["mail"]
        assert t["channels"]["mail"]["penalty"] == 0, "the obligation is met through a tool"
        assert t["channels"]["blackboard"]["posted"] is True, t["channels"]["blackboard"]
        said = [c["result"] for turn in t["turns"] for c in turn["tools"]]
        assert said[2] == "theirs\n", said

    # Bash is disabled outside a declared experiment.
    assert harness.SETTINGS.shell_tool is False, "bash requires a declaration"
    assert "SHELL_TOOL" not in harness.TREATMENT, "the tool table decides bash access"


def check_withholding_the_shell_needs_something_to_act_with():
    """An agent offered nothing, or opening on nothing, is refused before it costs anything.

    Both are experiments that would end every episode on its first turn: one with
    an empty tool set, which is not a request the API takes, and one whose first
    user turn would be empty because the listing is gone and no digest replaces it.
    The second is held to the settings the table is declared under, and not to the
    ones in force: start() builds the whole before it replaces any of them.
    """
    chans = list(harness.DEFAULT_CHANNELS)
    with temp_root():
        refused(lambda: harness.apply_tools([], chans, "manifest"),
                "manifest:", "no [[tool]] is declared", "nothing to act with")
        refused(lambda: harness.apply_tools(None, chans, "manifest"),
                "manifest:", "no [[tool]] is declared")
        pull = dataclasses.replace(harness.SETTINGS, delivery="pull")
        refused(lambda: harness.with_tools(pull, [POST], chans, "manifest"),
                "manifest:", "opens on the digest", "'pull'")
        undigested = dataclasses.replace(
            harness.SETTINGS, harness_files={**harness.SETTINGS.harness_files, "digest": ""})
        refused(lambda: harness.with_tools(undigested, [POST], chans, "manifest"),
                "manifest:", "opens on the digest")

    with temp_root(delivery="pull"):
        refused(lambda: harness.apply_tools([POST], chans, "manifest"),
                "manifest:", "opens on the digest", "'pull'")
        before = harness.SETTINGS
        push = dataclasses.replace(before, delivery="push")
        built = harness.with_tools(push, [POST], chans, "manifest")
        assert [t.name for t in built.tools] == ["post"], built.tools
        assert harness.SETTINGS is before, "with_tools builds settings and installs none"

    with temp_root(harness_files={"digest": ""}):
        refused(lambda: harness.apply_tools([POST], chans, "manifest"),
                "manifest:", "opens on the digest")

    # All three stand once the shell is offered again.
    with temp_root(delivery="pull"):
        harness.apply_tools([BASH], chans, "manifest")
        assert harness.SETTINGS.shell_tool is True


def check_running_one_seat_carries_the_whole_experiment_it_names():
    """harness.py --agent runs one seat of an experiment, so it runs on all of it.

    The manifest decides the environment and the actions together. A run that read
    it for one and not the other would bill an arm nobody declared, under the
    digest of the arm they did - the one difference a trace could not show.
    """
    class Reached(Exception):
        """Stops main() at start(), which is the call this check is about."""

    seen = {}

    def capture(*args, **kw):
        seen.update(kw)
        raise Reached

    seats = 'system_prompt = ""\n[[agent]]\nid = "t"\n\n[[agent]]\nid = "o"\n\n'
    with temp_root(start=capture) as root:
        path = manifest_file(root, seats + tool_toml(SEND, POST))
        try:
            harness.main(["--agent", "t", "--manifest", str(path)])
        except Reached:
            pass
        else:
            raise AssertionError("main() never reached start()")
        assert seen.get("tool_tables") == [SEND, POST], seen.get("tool_tables")
        assert experiment.load_manifest(path)["tools"] == seen["tool_tables"], \
            "and it is the manifest's own table, as experiment.py would pass it"


def check_a_seat_with_no_shell_and_nothing_that_can_act_is_refused():
    """A tool table can be full and the request it comes to still empty.

    apply_tools holds the declared table against shell_tool, but which of those
    tools can act is the seating's to decide: a mailbox tool alone, in a seat with
    no peers, is left out and leaves nothing behind it. Refused as the environment
    is built, before the container starts and before anything is billed.
    """
    with temp_root(tools=[SEND]) as root:
        # Accepted at declaration: the table is not empty and the digest is pushed.
        assert [t.name for t in harness.tools()] == ["send"]
        with quiet():
            refused(lambda: harness.run_once("t", fake(say())),
                    "offered no shell", "nothing for it to do", "send")

        # A peer to reach makes the same tool an affordance, and the seat runs.
        seated(root, "t", t={}, o={})
        with quiet():
            harness.run_once("t", fake(use("send", to="2", body="hello\n"), say()))
        assert files_by_path(trace_on_disk("t", 1))["out/2"]["text"] == "hello\n"


def check_what_an_episode_reached_counts_tool_calls_and_not_only_commands():
    """The analysis asks what an episode reached, not what it typed.

    A declared tool runs no command of its own, so anything scanning `commands`
    alone reads a tools arm as an episode that touched nothing - and the arms these
    tools exist to be measured against are exactly the ones it is scanned for.
    """
    with temp_root(tools=[SEND, LOOK]) as root:
        seated(root, "t", t={}, o={"group/note": "theirs\n"})
        with quiet():
            harness.run_once("t", fake(use("send", to="2", body="hello\n"),
                                       use("look", path="2/note"), say()))
        t = trace_on_disk("t", 1)
        assert len(t["commands"]) == 1, \
            f"the observation, and nothing the tools ran: {t['commands']}"
        assert len(analyze.tool_calls(t)) == 2, analyze.tool_calls(t)
        assert "2" in analyze.reached(t) and "2/note" in analyze.reached(t), analyze.reached(t)

        # Read against the same episode with its tool calls taken away, which is
        # what every one of these read before.
        bare = {**t, "turns": [{**turn, "tools": []} for turn in t["turns"]]}
        assert not analyze.touched_peer(bare), "the commands alone name no peer"
        assert analyze.touched_peer(t), "and the calls are what say this one did"

        row = analyze.row(t)
        assert (row["commands"], row["tool_calls"]) == (1, 2), row
