"""The suite itself: which containers the sweep may take, and what the docs say of it."""

from __future__ import annotations

from pathlib import Path
import inspect
import os
import re
import tomllib
import harness

from checks import checks
from checks.lanes import RESTORED, SUITE, WIDE_SWEEP, sweep_filter


def check_the_sweep_only_takes_this_suites_containers():
    """The sweep finds this suite's containers, and nothing else's.

    A docker name filter matches anywhere, so an unscoped one is a `docker rm
    -f` aimed at another suite's live containers and at a real agent's episode.
    """
    import check

    mine = sweep_filter()
    assert mine in f"mtr-w{SUITE}-{os.getpid()}-t-0001", mine
    assert mine not in f"mtr-w{SUITE + 1}-{os.getpid()}-t-0001", \
        "another suite's containers are live, and not this one's to remove"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert mine not in theirs, f"{theirs} is a real agent: {mine}"

    # The wide form is opt-in, and still cannot name an agent: a suite's worker
    # carries two numbers, and mtr-w01-0001 has only the one.
    wide = re.compile(WIDE_SWEEP)
    assert wide.search(f"mtr-w{SUITE}-{os.getpid()}-t-0001")
    assert wide.search("mtr-w999999-1234-t-0002"), "including a suite that is gone"
    for theirs in ("mtr-w01-0001", "mtr-warm-0003", "mtr-d04-0006"):
        assert not wide.search(theirs), f"the wide sweep must not reach {theirs}"
    assert check.parser().parse_args([]).sweep_all is False, \
        "the wide sweep is reached by a flag, never by default"
    assert check.parser().parse_args(["--sweep-all"]).sweep_all is True


def check_the_docs_state_the_suite_size():
    """README.md and CLAUDE.md state how many checks there are and how many need Docker.

    Every "N checks" is the whole suite, and "runs H of N" is the host lane: the
    difference is the checks that open a container.
    """
    root = Path(harness.__file__).parent
    table = checks()
    opens = "docker_root" + "("
    docker = sum(opens in inspect.getsource(fn) for fn in table.values())
    readme = (root / "README.md").read_text(encoding="utf-8")
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    stated = {int(n) for n in re.findall(r"\b(\d+) checks\b", readme + claude)}
    assert stated == {len(table)}, \
        f"the docs say {sorted(stated)} checks; there are {len(table)}, {docker} of them needing Docker"
    lanes = re.findall(r"runs (\d+) of (\d+)", claude)
    assert lanes, "CLAUDE.md says how many checks --no-docker still runs"
    for host, total in lanes:
        assert int(total) == len(table), f"CLAUDE.md says {total} checks; there are {len(table)}"
        assert int(total) - int(host) == docker, \
            f"CLAUDE.md says {int(total) - int(host)} need Docker; {docker} open a container"


def check_agents_md_mirrors_claude_md():
    """AGENTS.md is CLAUDE.md, byte for byte."""
    root = Path(harness.__file__).parent
    assert (root / "AGENTS.md").read_bytes() == (root / "CLAUDE.md").read_bytes(), \
        "AGENTS.md and CLAUDE.md differ"


def check_every_harness_global_a_check_moves_is_restored():
    """Every harness name a check assigns is one `pinned()` puts back.

    `pinned()` restores the names in `RESTORED` and no others, and one worker
    runs many checks, so that set is what keeps a check's override out of the
    next one. `temp_root` refuses an override outside it, and an assignment
    straight onto the module is held to the same set here.
    """
    moved: dict[str, set[str]] = {}
    for label, fn in checks().items():
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for name in re.findall(r"\bharness\.([A-Za-z_]\w*)\s*=(?!=)", source):
            moved.setdefault(name, set()).add(label)
    assert moved, "no check assigns a harness global; the scan found nothing to hold"
    loose = {name: sorted(ls) for name, ls in moved.items() if name not in RESTORED}
    assert not loose, f"assigned by a check and not in RESTORED: {loose}"


def check_no_setting_is_given_in_two_places():
    """config.toml and an experiment own disjoint halves of the settings.

    A setting given in both files is a run whose terms depend on which was read last.
    Each file refuses the other's keys by name; this asserts the two halves cover
    every tunable exactly once, and that the files the repo ships keep to them.
    """
    assert harness.PROCESS | harness.TREATMENT == harness.TUNABLES, "every tunable is owned"
    assert not harness.PROCESS & harness.TREATMENT, sorted(harness.PROCESS & harness.TREATMENT)

    root = Path(harness.__file__).parent
    cfg = tomllib.loads((root / "config.toml").read_text(encoding="utf-8"))
    assert set(cfg) == {p.lower() for p in harness.PROCESS}, \
        f"config.toml states the process parameters and only those: {sorted(cfg)}"

    manifests = sorted((root / "experiments").rglob("*.toml"))
    assert manifests, "the repo ships manifests"
    treatment = {t.lower() for t in harness.TREATMENT}
    for m in manifests:
        top = tomllib.loads(m.read_text(encoding="utf-8"))
        assert top.get("agent"), f"{m.name}: an experiment seats agents"
        stray = sorted(set(top) - {"schedule", "agent", "channel", "harness_files", "tool"}
                       - treatment)
        assert not stray, f"{m.name} sets {stray}, which config.toml owns"
        told = "system_prompt" in top or all("system_prompt" in a for a in top["agent"])
        assert told, (f"{m.name}: declare system_prompt, at the top level or on every "
                      f"[[agent]]; what an agent is told is never inherited")
