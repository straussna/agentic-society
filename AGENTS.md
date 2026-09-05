# Running check.py

`check.py` is the verification suite. Nothing in it bills an API. Some of it
starts Docker containers, and that is the only part that is slow.

**Agent the cheapest thing that answers the question you actually have.** Work down
this ladder and stop at the first rung that covers what you changed.

| what you changed | agent |
|---|---|
| one behaviour, and you know its name | `py -3 check.py <name-fragment>` |
| pricing, metering, refusals, traces, starter files, forks, experiment, manifests, simultaneous rounds, the shared files, transfer modes, push and pull delivery, author labels, transfers, the ledger, group and mailbox messages, what `m` carries, the initial observation, the three penalties, the grace, the floor | `py -3 check.py --no-docker` |
| anything, before handing work over | `py -3 check.py` |
| `harness.py`'s episode path — the container, the shell, `load_state`/`save_state`, `run_once` | `py -3 check.py --real` |

Rough costs: a name filter is seconds, `--no-docker` about 25s, the full agent
about 40s, `--real` two to four minutes.

`py -3 check.py --list` prints every name. Fragments match anywhere, and several
can be given at once: `py -3 check.py refusal fallback`.

## Why there are two lanes

Most checks are arithmetic — what a turn cost, what reached the series, which
stop an episode ended on. A container proves none of that, so those episodes agent
in a directory and a bash process on this machine.

What only a container can show — modes, ownership, the dead network, what the
image has and lacks — takes a real one. That includes the checks that an inbox
and the transfer ledger really are root's and really do refuse every route into
them. Those are the checks that skip when Docker is down, and the reason
`--no-docker` still runs 151 of 171.

`--real` puts every check in a container. It is what says the two lanes still
agree, so run it after changing how an episode is set up or torn down. It is not
the default, and it is not what to run to check an assertion you just edited.

## Things that will waste your time

Do not run the full suite repeatedly to watch a number. Run it once when the
work is done.

Docker Desktop slows down markedly after a few hundred containers. A full agent
that took 40s on a fresh daemon can take two minutes later in a long episode.
That is the daemon, not a regression — restart Docker rather than hunting it.

A suite run only removes containers carrying its own pid, so two runs at once
leave each other alone and no agent of `check.py` can touch a live experiment.
Nothing collects what an agent killed outright leaves behind: `--sweep-all` does,
and is the only mode that reaches a container this process did not make.

Two checks are wall-clock sensitive by design: `hostile_output_survives` (a 4MB
flood against a deadline) and anything setting `COMMAND_TIMEOUT`. Running with `-j` above
the core count makes them fail for contention rather than for cause. The default
`-j` is already sized for this machine; lower it before raising it.

# Stopping an agent early

One `Ctrl+C` ends the agent cleanly. It does not raise: it sets a flag that the
turn loop reads where it reads the account floor, so the episode ends the way an
exhausted budget ends it — the turn in flight finishes, its spend is committed,
the agent's trees are mirrored back, the trace is written and the container is
reaped. An experiment ends every remaining round, and every agent keeps its seat, so it
can be started again from where it stopped. Under a simultaneous round every episode in
flight ends at its next turn the same way, and all of them are committed before the
rounds end.

The cost is latency: worst case one whole turn, which is one API call plus the
commands it asks for. Press `Ctrl+C` a second time to stop waiting — the handler
puts the default back before it returns, so the second one is the ordinary hard
stop. That abandons the episode: the container leaks and the spend never reaches
`account.json`. `SIGTERM` behaves the same way.

Nothing in the repo reaps a container left by a hard kill — `check.py`'s sweep is
scoped to its own pid and cannot match `mtr-<agent>-<index>`. Remove those by hand.

# Editing harness.py

`harness.py` hashes itself at import and records the digest in every trace, and
`check_the_harness_digest_is_read_once` compares that against the file on disk.
So do not edit `harness.py` while a suite agent or an experiment is in flight — the
running process will disagree with the file and the check fails for a reason
that has nothing to do with the change.
