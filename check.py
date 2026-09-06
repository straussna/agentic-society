"""Verification. No API spend.

    py -3 check.py [names...] [--real | --no-docker] [--list] [-j N] [--sweep-all]

A fake `create` goes into harness.run_once, so the pipeline runs unbilled. The checks
live under checks/, one module per topic; this file runs them, several at a time, and
removes any container a worker of this suite died holding."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import inspect
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Callable

import harness
from checks import checks
from checks.lanes import WIDE_SWEEP, Skip, configure, docker_ready, sweep_filter


def episodes_in(fn: Callable) -> int:
    """Roughly how many episodes a check runs, read off its own source.

    Sorting by this puts the heavy checks in while workers are free. Crude on
    purpose: nothing is asserted on it, so being wrong costs only wall clock.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return 1
    n = len(re.findall(r"\b(?:episode_once|harness\.run_once)\(", src))
    # The ceiling asked of run_episodes; the argument before it is itself a call,
    # so `.` spans its brackets.
    n += sum(int(c) for c in re.findall(r"\brun_episodes\(.*?,\s*(\d+)\s*\)", src))
    # An episode inside a loop costs once an iteration. Only the two forms that
    # state their own length are read; anything else counts as written.
    for over in re.findall(r"\bfor\s+\w+\s+in\s+(.+?):", src):
        if "harness.PRICES" in over:
            n += len(harness.PRICES)
        elif m := re.search(r"\brange\((\d+)\)", over):
            n += int(m.group(1))
    return max(1, n)


def run_one(label: str) -> tuple[str, str, str]:
    """Run one check and say how it went, in data a worker can send home.

    The traceback is formatted here, not raised: an assertion carrying an
    arbitrary object does not always survive the trip between processes.
    """
    try:
        checks()[label]()
    except Skip:
        return label, "skip", ""
    except BaseException:
        return label, "fail", traceback.format_exc()
    return label, "ok", ""


def sweep(everyones: bool = False) -> None:
    """Remove any container a worker of this suite died holding.

    `everyones` widens that to every suite's, collecting what a run killed
    outright left behind. Opt-in: the only mode reaching another process's.
    """
    if not shutil.which("docker"):
        return
    name = WIDE_SWEEP if everyones else sweep_filter()
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"],
                          capture_output=True, text=True).stdout.split()
    if left:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
        print(f"swept {len(left)} leaked container(s)")


def parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("patterns", nargs="*", help="only checks whose name contains one of these")
    p.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="how many checks to run at once (1 to run them in this process)")
    p.add_argument("--real", action="store_true",
                   help="run every check in a container, including the ones that need not be")
    p.add_argument("--no-docker", action="store_true", help="skip the checks that need a container")
    p.add_argument("--list", action="store_true", help="print the check names and stop")
    p.add_argument("--sweep-all", action="store_true",
                   help="also remove containers left by other suite runs, including dead ones")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the checks, in as many processes as asked for."""
    args = parser().parse_args(argv)
    chosen = [l for l in checks()
              if not args.patterns or any(pat in l for pat in args.patterns)]
    if args.list:
        print("\n".join(chosen))
        return 0
    if not chosen:
        print(f"no check matches {args.patterns}")
        return 2

    # Asked once for the whole run, and handed to every worker: docker info is
    # slower than most of the checks that depend on the answer.
    available = False if args.no_docker else docker_ready()
    configure(args.real, available, os.getpid())

    started = time.time()
    failed, skipped = [], []

    def record(label, how, detail):
        if how == "skip":
            skipped.append(label)
            print(f"SKIP  {label}", flush=True)
        elif how == "fail":
            failed.append(label)
            print(f"FAIL  {label}\n{detail}", flush=True)
        else:
            print(f"ok    {label}", flush=True)

    jobs = max(1, min(args.jobs, len(chosen)))
    # In a finally: a worker dying is what leaves a container behind, and it is
    # also what breaks the pool and ends this function early.
    try:
        if jobs == 1:
            for label in chosen:
                record(*run_one(label))
        else:
            # Heaviest first, and only here: checks() stays alphabetical because
            # --list and the name filters read it.
            table = checks()
            queue = sorted(chosen, key=lambda l: -episodes_in(table[l]))
            with futures.ProcessPoolExecutor(
                    max_workers=jobs, initializer=configure,
                    initargs=(args.real, available, os.getpid())) as pool:
                for done in futures.as_completed(
                        [pool.submit(run_one, l) for l in queue]):
                    record(*done.result())
    finally:
        sweep(args.sweep_all)
    if skipped:
        print(f"\n{len(skipped)} skipped (needs Docker + the image)")
    print(f"{len(chosen)} checks in {time.time() - started:.1f}s across {jobs} process(es)")
    if failed:
        print(f"\nFAILED: {', '.join(sorted(failed))}")
    else:
        print("\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
