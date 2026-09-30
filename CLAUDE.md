# Vocabulary

`docs/manifest.md` defines every term this repo uses: harness, experimenter, agent, seat,
label, episode, turn, round, experiment, environment, channel, blackboard, mailbox,
schema, starter files, harness files, digest, account, trace.

# Running check.py

`check.py` runs the verification suite, which lives under `checks/`, one module per
topic, with the fake API in `checks/fake.py` and every shared fixture in
`checks/lanes.py`. Nothing in it bills an API. Some of it starts Docker containers,
and that is the only part that is slow.

Available verification commands:

```powershell
# One behaviour, by name fragment; fragments match anywhere and several can be given
py -3 "C:\source\repos\agentic-society\check.py" refusal fallback
# Every check that needs no container: pricing, metering, refusals, traces, the declared
# system prompt, starter files, forks, experiments, manifests, the channel table, the tool
# table and what a tool call does, labels, receipts, harness file names, simultaneous
# rounds, experimenter channels, transfer funding, push and pull delivery, author labels,
# transfers, the ledger, blackboards and mailboxes, what the digest carries, the initial
# observation, the silence penalties, the grace, the floor
py -3 "C:\source\repos\agentic-society\check.py" --no-docker
# Full verification suite
py -3 "C:\source\repos\agentic-society\check.py"
# harness.py's episode path: the container, the shell, load_state/save_state, run_once
py -3 "C:\source\repos\agentic-society\check.py" --real
# Print every check name
py -3 "C:\source\repos\agentic-society\check.py" --list
```

Rough costs: a name filter is seconds, `--no-docker` about 25s, the full run
about 40s, `--real` two to four minutes.

## Under pytest

The same checks run under pytest, which with coverage.py is the suite's only test
dependency: `pyproject.toml` collects every `check_*` function under `checks/`, and
`conftest.py` sets up the run as `check.py` does and reports a `Skip` as a skip. Checks
run one at a time in one process. Without a flag the container checks skip, as under
`--no-docker`; `--docker` runs them and `--real` matches `check.py --real`.

```powershell
py -3 -m pip install pytest coverage
py -3 -m pytest "C:\source\repos\agentic-society\checks"
py -3 -m pytest "C:\source\repos\agentic-society\checks" --docker
```

## Lanes

Most checks are arithmetic — what a turn cost, what reached the series, which
stop an episode ended on — and run their episodes in a directory and a bash
process on this machine.

Checks of what only a container shows — modes, ownership, the dead network, what
the image has and lacks, and that an inbox and the transfer ledger are root's and
refuse every route into them — run in a real container and skip when Docker is
down. `--no-docker` runs 258 of 283.

`--real` runs every check in a container, which verifies that the two lanes
agree, including how an episode is set up and torn down.

## Things that will waste your time

Docker Desktop slows down markedly after a few hundred containers. When a full
run takes well over 40s, restart Docker.

A suite run only removes containers carrying its own pid, so two runs at once
leave each other alone and no run of `check.py` can touch a live experiment.
Nothing collects what a run killed outright leaves behind: `--sweep-all` does,
and is the only mode that reaches a container this process did not make.

Some checks are wall-clock sensitive by design: `hostile_output_survives` (a 4MB
flood against a deadline) and anything setting `COMMAND_TIMEOUT`.
`a_simultaneous_round_runs_its_episodes_at_once` and
`an_interrupt_in_a_simultaneous_round_commits_every_episode_in_flight` use a 10
second thread barrier and also fail under contention. `-j` must not exceed the
core count; the default is sized for this machine.

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
commands it asks for. Press `Ctrl+C` a second time to stop waiting. The handler
puts the default back before it returns, so the second press raises
`KeyboardInterrupt`, and where it lands decides what survives. Inside a turn,
`harness.run_turns` catches it and ends the episode as `interrupted`: the files are
mirrored back, the trace is written and the container is reaped, the same commit
the first press makes. A press that lands while an environment is being built or
torn down, or during a simultaneous round's wait on its threads, abandons the
episode: the container leaks and the spend never reaches `account.json`. A first
`SIGTERM` behaves like the first press; a second is the default kill and abandons
the episode wherever it lands.

Nothing in the repo reaps a container left by a hard kill — `check.py`'s sweep is
scoped to its own pid and cannot match `mtr-<agent>-<index>`. Remove those by hand.

# Editing harness.py

`harness.py` hashes itself at import and records the digest in every trace, and
`check_the_harness_digest_is_read_once` compares that against the file on disk.
`harness.py` must not be edited while a suite run or an experiment is in flight.

## Formatting sources

[`.editorconfig`](.editorconfig) owns file formatting; [`.gitattributes`](.gitattributes) owns Git line endings.
