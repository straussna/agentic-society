"""Behavior-only memory inheritance between experiments."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys

import experiment
import harness
from checks.lanes import HostBox, manifest_file, offers, plant, quiet, refused, rooted


def source_trace(agent: str, episode: int, body: str) -> None:
    """Record one complete private memory at an episode boundary."""
    data = body.encode("utf-8")
    trace = {
        "trace_version": harness.TRACE_VERSION,
        "agent": agent,
        "episode": episode,
        "state_saved": True,
        "provenance": {
            "channels": [channel.as_table() for channel in harness.channels()],
            "tools": [tool.as_table() for tool in harness.tools()],
        },
        "files": [{
            "path": "state/memory.md",
            "channel": "notes",
            "writer": "self",
            "readers": "self",
            "role": "own",
            "size": len(data),
            "text": body,
        }],
    }
    path = harness.trace_path(agent, episode)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(trace), encoding="utf-8")


def check_memory_from_copies_only_behavioral_memory_into_a_fresh_agent():
    """The declared memory crosses exactly; account history and transient channels do not."""
    with rooted(HostBox, tools=offers("write_memory:notes")):
        body = "A complete inherited memory.\nUnicode: →\n"
        source_trace("old", 7, body)
        account = harness.load_account("new")
        entry = {"id": "new", "memory_from": {"agent": "old", "episode": 7}}

        experiment.inherit_memory(entry, account)
        inherited = harness.account_on_disk("new")
        copied = harness.mirror("new", "notes") / "memory.md"

        assert copied.read_bytes() == body.encode("utf-8")
        assert inherited["episodes"] == [] and inherited["series"] == [harness.BUDGET]
        assert inherited["memory_from"] == entry["memory_from"]
        record = inherited["memory_inherited"]
        assert record["agent"] == "old" and record["episode"] == 7
        assert record["memories"][0]["sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert not harness.mirror("new", "blackboard").joinpath("post.md").exists()

        prepare = experiment.preparer("new", {"1": "new"},
                                      {"schedule": "simultaneous", "manifest_sha256": "m"})
        prepare(inherited)
        provenance = harness.provenance(inherited["provider"], inherited["model"],
                                        experiment=inherited["experiment"])
        assert provenance["memory_from"] == record

        experiment.inherit_memory(entry, inherited)
        assert copied.read_bytes() == body.encode("utf-8"), "resuming recopied the memory"


def check_memory_from_refuses_any_source_it_cannot_copy_exactly():
    """Every memory_from that cannot give the exact recorded memory is refused, and leaves
    the new agent's memory as it found it and nothing recorded as inherited: a changed
    source on an agent that already has one, a missing trace, another trace version, an
    episode that did not save its state, unmatched write_memory tools, two memories for
    one tool, a text copy that is not the file, and a memory already there that differs.
    """
    body = "A complete inherited memory.\n"

    def rewritten(change):
        def source():
            source_trace("old", 7, body)
            path = harness.trace_path("old", 7)
            trace = json.loads(path.read_text(encoding="utf-8"))
            change(trace)
            path.write_text(json.dumps(trace), encoding="utf-8")
        return source

    def pinned_elsewhere():
        source_trace("old", 7, body)
        account = harness.load_account("new")
        account["memory_from"] = {"agent": "old", "episode": 6}
        harness.save_account("new", account)

    def planted_differently():
        source_trace("old", 7, body)
        mine = harness.mirror("new", "notes") / "memory.md"
        mine.parent.mkdir(parents=True, exist_ok=True)
        mine.write_text("what the new agent already remembers\n", encoding="utf-8")

    cases = {
        "was created with memory_from": pinned_elsewhere,
        "does not exist": lambda: None,
        "trace version": rewritten(lambda t: t.update(trace_version=harness.TRACE_VERSION - 1)),
        "did not save its state": rewritten(lambda t: t.update(state_saved=False)),
        "matching write_memory tools": rewritten(lambda t: t["provenance"].update(tools=[])),
        "several memories": rewritten(lambda t: t["files"].append(dict(t["files"][0]))),
        "exact text copy": rewritten(lambda t: t["files"][0].update(size=len(body) + 1)),
        "would overwrite": planted_differently,
    }
    for why, arrange in cases.items():
        with rooted(HostBox, tools=offers("write_memory:notes")), quiet():
            arrange()
            destination = harness.mirror("new", "notes") / "memory.md"
            held = destination.read_bytes() if destination.exists() else None
            entry = {"id": "new", "memory_from": {"agent": "old", "episode": 7}}
            refused(lambda: experiment.inherit_memory(entry, harness.load_account("new")), why,
                    because=f"inherited from a source it should refuse: {why}")
            now = destination.read_bytes() if destination.exists() else None
            recorded = harness.account_on_disk("new")
        assert now == held, f"{why}: the memory changed from {held!r} to {now!r}"
        assert "memory_inherited" not in recorded, f"{why}: {recorded.get('memory_inherited')}"


def check_harness_run_as_a_file_runs_its_cli_in_the_module_experiment_imports():
    """inherit_memory reads the globals of the harness experiment.py imports, so
    `harness.py` run as a file runs its CLI in that module and not in a copy of it.

    The ROOT set here is the imported module's; a copy would read its own and find
    no starter files by that name.
    """
    argv = sys.argv
    with rooted(HostBox) as root:
        plant(root, "only-here", m1="alpha\n")
        sys.argv = [harness.__file__, "--print-files", "only-here"]
        try:
            with quiet() as output:
                runpy.run_path(harness.__file__, run_name="__main__")
        except SystemExit as e:
            code = e.code
        else:
            raise AssertionError("harness.py run as a file did not exit through its CLI")
        finally:
            sys.argv = argv
    assert code == 0, (code, output.getvalue())
    assert "1 files, 6 bytes" in output.getvalue(), output.getvalue()


def check_memory_from_is_strict_agent_grammar():
    """A source and positive episode are required, and a new seat cannot be its own source."""
    with rooted(HostBox) as root:
        good = manifest_file(
            root,
            'system_prompt = ""\n[[agent]]\nid = "new"\n'
            'memory_from = { agent = "old", episode = 7 }\n',
        )
        loaded = experiment.load_manifest(good)
        assert loaded["agents"][0]["memory_from"] == {"agent": "old", "episode": 7}

        bad = (
            'memory_from = {}',
            'memory_from = { agent = "old", episode = 0 }',
            'memory_from = { agent = "old", episode = true }',
            'memory_from = { agent = "new", episode = 1 }',
            'memory_from = { agent = "old", episode = 1, history = true }',
        )
        for index, declaration in enumerate(bad):
            path = manifest_file(
                root,
                f'system_prompt = ""\n[[agent]]\nid = "new"\n{declaration}\n',
                name=f"bad-{index}.toml",
            )
            try:
                experiment.load_manifest(path)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"accepted invalid agent memory grammar: {declaration}")
