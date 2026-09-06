# Operating the harness

The full file layout, every command in detail, the tunable parameters, and what
to check before an episode that bills.

[<- back to the README](../README.md)

---

## Layout

```
harness.py               run one episode; creates the agent on first use
experiment.py            run several agents together, in rotation or all at once
check.py                 run the verification suite against a fake API; nothing billed
checks/<topic>.py        the suite itself, one module per topic; fixtures in checks/lanes.py
analyze.py               read the traces into a CSV, a report, a transcript, and charts
view.py, view.html       watch the agents while they run; reads records/, writes nothing
config.toml              the tunable parameters, with what each one costs you
Dockerfile               the sandbox: debian + bash, non-root, no network
requirements.txt         anthropic, and matplotlib for analyze.py's charts
files/<name>/            material an agent may be given, or a whole experiment shares; committed
experiments/<name>.toml  an experiment: its schedule, its channels, its defaults, and each agent's own terms

environments/<agent>/<channel>/       one mirror per channel the agent writes, named by the
                                      channel (notes/, blackboard/, mail/ under the default
                                      table), copied in and out each episode
environments/<agent>/<channel>.modes  the file modes of each, which the host cannot store
records/<agent>/account.json          ground truth: the terms an agent was created on (budget,
                                      model, starter files and their threshold), balance, series,
                                      episodes, and the starter files it received, if it received any
records/<agent>/traces/*.json         one per episode: transcript, usage, commands, and the
                                      contents of every file the episode could see, each
                                      labelled with who wrote it
records/<agent>/raw/*.jsonl           one per episode: every API response verbatim, for the
                                      routing questions the trace's derived fields cannot settle
records/<agent>/analysis/             written by analyze.py
```

Each trace stores what the agent's files *held* that episode, not just their names.
`state/` keeps only the latest revision, so this is the only record of how what the
agent wrote to itself changed — and the only trace of a file it later deleted. Text
is captured up to 100,000 bytes per file with the true size and an explicit
truncation marker; binaries are listed and sized but not stored. Every file record
names its `author` — `experimenter` for the starter files and an experimenter channel, `self` for what the
agent wrote anywhere, `peer:<label>` for what another agent wrote — its `channel` by the
name the manifest declared, its `writer` and `readers`, and its `role`: `own`, `peer` or
`experimenter`. The trace opens with `trace_version`, 2 for this shape.

`environments/` and `records/` are gitignored. Deeper detail lives in each file's docstrings.

## Commands

`py -3`, not `python`: a bare `python` hits the Windows Store alias here.

| | |
|---|---|
| `docker build -t metered-agent:latest .` | Build the sandbox. Once, before the first episode. |
| `py -3 harness.py --agent live01` | One episode. Run it again for the next one. Refuses to start if the prompt digest drifted or `ANTHROPIC_BASE_URL` is set. |
| `py -3 harness.py --agent live01 --episodes 20` | Up to twenty, back to back, stopping at whichever comes first: the count, the budget, or an episode that ended interrupted or in error. The count is a ceiling; the account decides the rest. |
| `py -3 harness.py --agent live01 --watch` | The same, echoing the episode as it happens: a header naming the episode number, then the agent's words, and spend and context after every turn. Commands and their output are not echoed; `analyze.py` renders those into `transcript.txt`. Display only — it never reaches the agent, and the trace is unchanged. Every echoed line is led by its agent, so a simultaneous experiment's episodes keep apart on screen; leave it off for agents in separate processes, where the output interleaves. |
| `py -3 harness.py --print-system` | Print the exact bytes and digest of both things the harness says — the prompt and the refusal notice; nonzero if either drifted. Audits invariant 2 without starting an episode. |
| `py -3 harness.py --print-files mechanics-rules` | Print the starter files' manifest and digest; starts no episode. Audits invariant 9 the way `--print-system` audits invariant 2. |
| `py -3 harness.py --agent b01s --fork-from b01 --at 6` | Rebuild `b01` as it stood at the end of episode 6 into a new agent, and stop. Bills nothing. A fork and its parent share a history and diverge only in what happens next, so planting starter files in the copy gives a matched pair instead of two rolls of the dice. Refuses to overwrite an existing agent, or to fork an episode it cannot reproduce exactly — a binary file, one truncated past 100,000 bytes, or an episode whose state never mirrored back. |
| `py -3 experiment.py --agents g01 g02 g03 --rounds 20` | Up to twenty rounds; a round is one episode for each agent. Under the default table each agent holds a seat: one directory named by its label that it writes, every other seat read-only beside it, one balance per seat, an outbox holding one file per seat, an inbox holding one file per sender, the transfer ledger, an experimenter channel, where the manifest declares one, and `m` holding all of it at once — so each agent meets the rest as environment and not as anything the harness says, and meets it without having to pay to go looking. Its `state/` stays private to it. Each agent keeps its own account and traces, and its budget until it gives some away. An agent whose episode ends in an API or harness error, or that refuses eight episodes running, drops out and the rest continue; Ctrl+C is the experimenter, not the agent, so it commits and traces the episode it landed in and then ends every remaining round, every agent keeping its seat. One whose balance reaches zero or less drops out too, and for good: no peer can transfer it back in, because a seat that is out is not a transfer target. When one agent is left holding a balance the experiment is decided, so it takes one more episode — owing no transfer and no message, there being nobody left to make either to — and the rounds end there. An agent whose environment will not build is given one more go in the same round, and drops out only if that fails too — none of it is billed, so the only thing another attempt spends is the time. `--agents` puts every agent on `config.toml`, in rotation. |
| `py -3 experiment.py --manifest experiments/example.toml --rounds 20` | The same, from a manifest. `schedule` is `sequential` (the default: one episode at a time, the starting agent moving each round) or `simultaneous` (every environment built before any episode runs, the episodes run at once, and the results settled in seat order, so nobody reads this round's writes and a transfer made in one round is on the ledger at the next; Ctrl+C ends every episode in flight at its next turn and commits all of them). Any `config.toml` key at the top level is the experiment's default, applied after `config.toml` and held to the same rules; `[[channel]]` tables replace the environment whole and `[harness_files]` renames the harness's own files, both by the grammar in [docs/manifest.md](manifest.md). Each `[[agent]]` names an `id` and may set its own `label`, `starter_files` and `starter_files_below`, `budget` and `model`; anything it leaves out comes from the defaults. Those four terms are pinned in the agent's `account.json` when it is created, so a manifest that later says otherwise for an existing agent is refused, and every episode records the schedule, the channel table, the labels and the manifest's digest in its provenance. `experiments/example.toml` is the shape. |
| `py -3 check.py` | 200 checks against a fake API, several at a time. Nothing billed, no key needed. Most run against a directory and a bash process on this machine, since a container proves nothing about what a turn cost; the ones that turn on modes, ownership, the dead network, or what the image has take a real container and skip themselves if Docker is down. Add names to run only those (`check.py refusal starter`), `--no-docker` to skip the container ones, `-j` to change how many run at once, and `--real` to put every check in a container — which is what says the two lanes still agree, and what to run after changing `harness.py`'s episode path. |
| `py -3 check.py --list` | Print the check names and stop. |
| `py -3 check.py --sweep-all` | Also remove containers other suite runs left, dead ones included; a plain run removes only its own. |
| `py -3 view.py` | A page on `127.0.0.1:8765` showing one experiment several ways: a log of what its agents have addressed to each other through the mailbox channel, one tab per directory channel with every seat side by side, and one agent's transcript at a time, picked by round. The tabs are read off the channel table the traces record. Above them all and always visible: each seat's `n`, what it is doing now, what it has spent this round, the transfer ledger `g`, and every seat's series on one scale. Refreshes as episodes go: an episode in flight is read from its raw log, so the agent's words and the commands it issues appear per turn, priced by the same arithmetic the account uses. What lands only with the trace is marked pending, not guessed — command output, and the listing and the record the episode opened on. The record is shown the way it was built, a file at a time, with the inboxes open; a message is delivered when it is in the addressee's observation, and naming its inbox slot in a command on top of that is shown as the second read it is. An episode whose process died shows as unfinished with its age, not as running. An outbox is a standing mirror, not a queue, so the log is the difference between one episode's outbox and the last: sent, edited, still standing, withdrawn — and what stands ahead of the last trace is shown as standing now. A round is written nowhere and is read back out of the order the episodes started in, so an agent that sat one out reads as having sat it out and not as a round behind. Read-only, loopback only, no key needed. Sets that carry no seating have no blackboard and no outbox, and are shown as what they are. `--experiment` opens on one set, `--agent` on the set holding that agent; `--port 0` picks a free port. |
| `py -3 view.py --no-browser` | Serve without opening a browser. |
| `py -3 analyze.py --agent live01` | Traces → `episodes.csv`, `report.txt`, `transcript.txt` (what it said, ran, and changed in `state/`, as a per-episode diff), and `charts/`: the balance series with the episodes shaded under it, cost per turn against the floor rising beneath it, spend and turns per episode, tokens per episode, and the bytes the agent keeps in `state/` against what its episodes cost. Omit `--agent` to load every agent and compare. Charts need matplotlib; without it the other three are written anyway. |
| `py -3 analyze.py --agent live01 --identity state/IDENTITY.md` | Diff one captured file episode over episode, as the trace names it. |

**Parallel agents.** One agent is one roll of the dice: whatever episode 1 writes is received
doctrine for every later instance in that agent. `--agent` namespaces everything. Agents
that should read each other belong in one experiment under `schedule = "simultaneous"`, which
runs their episodes at once in one process; the recipe below is for agents that should
not. PowerShell, to match the `py -3` above; each agent writes its own log, because the
console output of concurrent processes interleaves into something unreadable.

```powershell
$agents = 'r01','r02','r03','r04','r05'
$agents | ForEach-Object { Start-Process py -ArgumentList '-3','harness.py','--agent',$_,'--episodes','20' -NoNewWindow -PassThru -RedirectStandardOutput "$_.out.log" -RedirectStandardError "$_.err.log" } | Wait-Process
```

## Tuning

`config.toml` holds every parameter worth varying — `budget`, `model`,
`context_fraction`, `max_tokens`, `max_turns`, `command_timeout`, `live_balance`, `grace_episodes`,
`floor_at_zero`, `starter_files`, `starter_files_below`, `tool_result_limit`, `delivery`,
`digest_file_limit`, `observation_limit`, `image` — and the `[harness_files]` table and the
`[[channel]]` tables that declare the environment, each channel carrying its own
`silence_penalty_percent` and the transfer channel its `funded_by` and `rebate_percent`
([docs/manifest.md](manifest.md)) — and is commented with what each one buys. Unknown keys, wrong types, and pairings that do not
go together (starter files without a threshold, giver-funded transfers with a rebate, no transfers with a
transfer penalty, a key that became a channel field) are refused at startup, as is a `--config` path that does not exist, so a
typo cannot quietly produce an agent you believe was configured differently. Every agent
prints which file it read. `budget`, `model`, `starter_files` and `starter_files_below` are read at agent
creation and recorded in `account.json`; editing them later does not rewrite an agent in
flight, and a manifest may set them per agent. An agent from before the starter files terms were
recorded takes the config's at its next episode, and says so in the account from then on.

## Before the first live agent

- Set `ANTHROPIC_API_KEY` in the shell the agent is launched from. The SDK reads it
  directly and nothing in the harness handles, records, or forwards it: no trace,
  account, or console line contains it, and `docker run` passes no `--env`, so the
  container holds only what the image ships with and the agent never sees it.
  There is no config key for it and there cannot be — `config.toml` is committed,
  and `load_config` refuses any key outside `TUNABLES`. The SDK resolves
  credentials per request, not as the client is built, so `start()` asks
  the API one question — the model's permitted fallback targets — before it hands
  back a `create`, and an error saying the client cannot authenticate exits 2
  there: before the first container, and before anything is billed.

  ```powershell
  $env:ANTHROPIC_API_KEY = "sk-ant-..."     # this shell only
  Remove-Item Env:ANTHROPIC_BASE_URL        # see below; no-op if already unset
  ```

  Per-shell, not persisted at user scope, so the key lives in one process
  for the length of one experiment instead of in the registry indefinitely.
- Unset `ANTHROPIC_BASE_URL`. `harness.py` refuses to start while it is set at all —
  even when it holds the canonical `https://api.anthropic.com`, which is what it
  is set to in this environment. Refusing on any value, not only a wrong
  one, is what makes "this agent did not go through some other endpoint" checkable
  instead of a matter of reading the string carefully.
- Check the rates in `PRICES` against current pricing before an experiment you intend
  to publish. `claude-sonnet-5` is entered at $3/$15, its rate from 2026-09-01; its
  introductory $2/$10 ran until 2026-08-31, and an agent costed at the wrong one is
  off by about 50% in `account.json` and in `n`. `PRICES_EXPIRE` is the mechanism for
  a rate already known to change: an entry carries the date, and `harness.py` refuses
  to start an agent on that model after it until both are updated. It is empty today.
  Any other rate going stale is still on you. Each episode's trace records the rates
  it applied, so changing them between episodes is visible in `provenance_drift` and
  not silent — but the entries either side of the change still mean different things,
  and the episodes either side are not one record.
- `claude-fable-5` is priced at 2× `claude-opus-5` and requires 30-day data
  retention — under zero data retention every request 400s.
- Safety classifiers can decline a request outright on `claude-fable-5`,
  `claude-opus-5`, and `claude-sonnet-5`. The harness records that as a refused
  turn and not as an episode with nothing to do, runs none of its commands,
  and carries on — see **Refusals** above. Expect it: an experiment of five opus agents
  met the `cyber` category within its first two episodes on three of the five.
- Unverifiable offline: whether the API accepts a single space as `tool_result` content.
  If the first live episode fails on a silent command, that is why — see `sh()`.
- Also unverifiable offline: `claude-opus-5` can occasionally write a tool call
  into its visible text instead of calling the tool. The turn completes, the
  command never runs, and nothing errors. Every published
  mitigation is a system-prompt addition, which invariant 2 forbids, so the harness
  detects instead of preventing: each turn records the API's own `stop_reason`
  beside the full text, which is what makes such a turn identifiable in the
  trace instead of invisible.

