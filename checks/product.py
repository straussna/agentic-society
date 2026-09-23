"""Experiment-level progress, cost, outcomes, and human-takeover branches."""

from __future__ import annotations

import hashlib
import json

import experiment
import harness
import product
from checks.fake import fake, say
from checks.lanes import manifest_file, quiet, temp_root


def check_product_controls_are_declared_and_validated():
    """Identifiers, cost boundaries, reveals, and quality tiers survive manifest loading."""
    text = ('experiment_id = "pilot-7"\nsystem_prompt = ""\n'
            '[cost]\nmaximum = 1000\nreserved_completion = 100\nwarning = 700\n'
            'ceiling_policy = "stop"\n[reveal]\nat_completion = ["winners", "scores"]\n'
            'experimenter_only = ["evidence"]\n'
            '[[agent]]\nid = "p1"\nquality_tier = "premium"\n')
    with temp_root() as root:
        loaded = experiment.load_manifest(manifest_file(root, text))
        assert loaded["experiment_id"] == "pilot-7"
        assert loaded["cost"] == {"maximum": 1000, "reserved_completion": 100,
                                  "warning": 700, "ceiling_policy": "stop"}
        assert loaded["reveal"]["at_completion"] == ["winners", "scores"]
        assert loaded["agents"][0]["quality_tier"] == "premium"
        for bad in ('experiment_id = "../bad"\n',
                    '[cost]\nmaximum = 100\nreserved_completion = 100\n',
                    '[cost]\nmaximum = 100\nceiling_policy = "downgrade"\n',
                    '[reveal]\nat_completion = ["narration"]\n'):
            path = manifest_file(root, bad + 'system_prompt = ""\n[[agent]]\nid = "p1"\n',
                                 name=hashlib.sha256(bad.encode()).hexdigest()[:8] + ".toml")
            try:
                experiment.load_manifest(path)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"accepted invalid product controls: {bad!r}")


def check_product_records_cost_progress_and_outcome_from_accounts_and_traces():
    """Human spend is excluded, progress is durable, and outcomes cite trace bytes."""
    with temp_root() as root:
        accounts = {
            "auto": {"provider": "anthropic", "remaining": 70,
                     "episodes": [{"episode": 1, "spent": 30}],
                     "product": {"quality_tier": "premium"}},
            "player": {"provider": "human", "remaining": 90,
                       "episodes": [{"episode": 1, "spent": 999}],
                       "product": {"quality_tier": "interactive"}},
        }
        for agent, account in accounts.items():
            trace = harness.trace_path(agent, 1)
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({"agent": agent}), encoding="utf-8")
        load = accounts.__getitem__
        cost = product.cost(accounts, load,
                            {"maximum": 130, "reserved_completion": 100, "warning": 20})
        assert cost["autonomous_spend"] == 30 and cost["warning_reached"]
        assert cost["ceiling_reached"], "the reserve is protected before another round starts"
        first = product.progress(root, "pilot", "preparing_round", 1)
        second = product.progress(root, "pilot", "waiting_player", 1, {"agents": ["player"]})
        assert len(first["events"]) == 1 and len(second["events"]) == 2
        assert product.records(root, "pilot")["progress"]["latest"]["phase"] == "waiting_player"
        outcome = product.outcome(root, "pilot", ["auto", "player"],
                                  {"1": "A", "2": "P"}, {"player"}, "one_remains", load,
                                  {"at_completion": ["winners"], "experimenter_only": ["evidence"]})
        assert outcome["winners"] == ["P"] and outcome["survivors"] == ["P"]
        assert outcome["scores"] == {"A": 70, "P": 90}
        assert len(outcome["evidence"]) == 2
        assert all(len(item["trace_sha256"]) == 64 for item in outcome["evidence"])
        public = product.revealed(outcome, "at_completion")
        assert public["winners"] == ["P"] and "scores" not in public
        assert "evidence" not in public, "experimenter-only evidence is not player output"
        assert product.revealed(outcome, "at_completion", experimenter=True)["evidence"]


def check_an_experiment_branch_is_new_lineage_and_leaves_sources_unchanged():
    """Every seat forks at one boundary and the selected new seat becomes interactive."""
    with temp_root() as root:
        source = manifest_file(
            root, 'experiment_id = "source"\nsystem_prompt = ""\n'
                  '[[agent]]\nid = "source-a"\nlabel = "A"\n'
                  '[[agent]]\nid = "source-b"\nlabel = "B"\n', name="source.toml")
        for agent in ("source-a", "source-b"):
            with quiet():
                harness.run_once(agent, fake(say()))
        source_bytes = {path: path.read_bytes() for agent in ("source-a", "source-b")
                        for path in (harness.records_dir(agent) / "account.json",
                                     harness.trace_path(agent, 1))}
        output = root / "experiments" / "branches" / "takeover.toml"
        with quiet():
            actual = experiment.branch_experiment(source, 1, "takeover", "2", output)
        assert actual == output
        branch = experiment.load_manifest(output)
        assert branch["experiment_id"] == "takeover"
        assert [entry["id"] for entry in branch["agents"]] == ["takeover-01", "takeover-02"]
        assert branch["agents"][0]["provider"] == "anthropic"
        assert (branch["agents"][1]["provider"], branch["agents"][1]["model"]) == \
               ("human", "interactive")
        lineage = product.records(root, "takeover")["lineage"]
        assert lineage["source_experiment_id"] == "source" and lineage["source_round"] == 1
        assert len(lineage["traces"]) == 2
        assert all(path.read_bytes() == before for path, before in source_bytes.items())
