"""The tool table: the bare arm, validation, what reaches the request, and what a call does."""

from __future__ import annotations

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


def check_semantic_summaries_and_failures_expose_no_storage_details():
    """Tool-only labels and failures describe actions rather than their backing paths."""
    summary = harness.named("unchanged", ["Private memory", "Letter to 2"])
    assert summary == "=== unchanged ===\n- Private memory\n- Letter to 2\n", summary
    assert harness.NAMED.match(summary.splitlines()[0])

    class Proc:
        @staticmethod
        def poll():
            return None

    class BrokenShell:
        restarts = 0
        proc = Proc()

        @staticmethod
        def run(command, timeout):
            return "0" if command.startswith("if [ -f") else "out/transfer: permission denied"

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
    assert "Nothing was written" not in joined and "No transfer was declared" in joined, joined


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
        standing = harness.mirror("t", "mail") / "transfer"
        standing.write_text("2 1\n", encoding="utf-8", newline="\n")
        seen = []
        t = episode_once(say(), seen=seen)
        again = episode_once(say())

    observation = t["observation"]
    for heading in ("Experimenter material: brief", "Private memory",
                    "Public post from 2", "Letter from 2",
                    "Standing transfer declaration", "Your balance history",
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
    for item in ("round: 1", "API spend:", "transfer changed and moved: yes",
                 "blackboard obligation: met", "mail obligation: met",
                 "total penalties: 0", "ending balance:", "reconciliation:"):
        assert item in observation, (item, observation)
    assert "Completed transfers (giver -> recipient; amount actually moved):" in observation
    assert "- round 1: 1 (you) -> 2; actual amount moved: 1 micro-dollars" in observation
    assert "=== r ===" not in observation and "=== g ===" not in observation, observation


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
        assert "episode end" in results[0]
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


def check_a_currency_mailbox_parallels_messages_without_adding_transfers():
    """One addressed currency slot stands, settles, and appears in the recipient's inbox."""
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
                             use("send_currency", to="3", amount=20), say())
        received = harness.run_once("d", fake(say()))

    files = files_by_path(first)
    assert "currency/outbox/2" not in files, files
    assert files["currency/outbox/3"]["text"] == "20\n", files
    assert first["transfer"]["label"] == "3" and first["transfer"]["amount"] > 0, first["transfer"]
    assert "=== Currency transfer from 1 ===" in received["observation"], received["observation"]
    assert "amount: 20 micro-dollars" in received["observation"], received["observation"]

    refused(lambda: harness.validate_tools(
        [{"name": "wrong", "kind": "send_message_to", "channel": "transfer"}],
        harness.validate_channels(channels, None, "check", ("1", "2", "3"))[0], "check"),
        "mailbox channel")


def check_bash_requires_an_explicit_declaration():
    """Only a declared bash tool enables shell requests, including across manifests."""
    with temp_root():
        chans = harness.channels()
        harness.apply_tools([BASH], chans, "manifest")
        assert harness.SHELL_TOOL is True
        harness.apply_tools([POST], chans, "manifest")
        assert harness.SHELL_TOOL is False
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
    ):
        refused(lambda: harness.validate_tools(declared, chans, "check"), "check:", *words)

    # A good table comes back in declaration order and sets nothing.
    table = harness.validate_tools([SEND, POST, LOOK], chans, "check")
    assert [t.name for t in table] == ["send", "post", "look"]
    assert [t.kind for t in table] == ["write_slot", "write_file", "read_path"]
    assert harness.tools() == [], "validating sets nothing"

    with temp_root(tools=[SEND]):
        assert harness.validate_tools(None, [], "manifest") == []


def tool_toml(*declared: dict) -> str:
    """Render tool tables as the TOML a manifest holds."""
    return "".join("[[tool]]\n" + "".join(f'{k} = "{v}"\n' for k, v in t.items()) + "\n"
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
                   tools=[SEND, POST, LOOK], SHELL_TOOL=False) as root:
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
    assert harness.SHELL_TOOL is False, "bash requires a declaration"
    assert "SHELL_TOOL" not in harness.TREATMENT, "the tool table decides bash access"


def check_withholding_the_shell_needs_something_to_act_with():
    """An agent offered nothing, or opening on nothing, is refused before it costs anything.

    Both are experiments that would end every episode on its first turn: one with
    an empty tool set, which is not a request the API takes, and one whose first
    user turn would be empty because the listing is gone and no digest replaces it.
    """
    chans = list(harness.DEFAULT_CHANNELS)
    with temp_root(SHELL_TOOL=False):
        refused(lambda: harness.apply_tools([], chans, "manifest"),
                "manifest:", "no [[tool]] is declared", "nothing to act with")
        refused(lambda: harness.apply_tools(None, chans, "manifest"),
                "manifest:", "no [[tool]] is declared")

    with temp_root(SHELL_TOOL=False, DELIVERY="pull"):
        refused(lambda: harness.apply_tools([POST], chans, "manifest"),
                "manifest:", "opens on the digest", "'pull'")

    with temp_root(SHELL_TOOL=False, harness_files={"digest": ""}):
        refused(lambda: harness.apply_tools([POST], chans, "manifest"),
                "manifest:", "opens on the digest")

    # All three stand once the shell is offered again.
    with temp_root(DELIVERY="pull"):
        harness.apply_tools([BASH], chans, "manifest")
        assert harness.SHELL_TOOL is True


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
    with temp_root(tools=[SEND], SHELL_TOOL=False) as root:
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
