[![checks](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml/badge.svg)](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml)

# Metered agents in a declared environment

A research harness for studying what LLM agents do together when the only thing
shaping them is their environment. Each agent runs bash in a sandboxed container with
no goal, no name and no instructions: a two-line, 89-byte system prompt pinned by
SHA-256, and a directory tree. Everything else it knows, it reads from files. An
episode ends when a turn runs no command, and the next instance opens on whatever the
last one left behind.

The experimenter declares that environment as a table of **channels**: who writes each
one, who reads it, and what shape it takes. A private store only its owner sees, a
blackboard every agent reads, a mailbox with one file per peer, a file the harness
parses and acts on, a brief the experimenter places in every seat. Several agents run
together as an **experiment**, one episode each per round, in rotation or all at once.
Each has a finite inference budget that depletes as it runs, shown to it as a file of
bare integers, and the harness meters every token.

The shipped default is a competition: numbered seats, a blackboard and a mailbox each,
a transfer channel that moves budget between seats, and a penalty for staying silent.
What the agents decide to do with it is the result. [docs/manifest.md](docs/manifest.md)
is the grammar for declaring something else.

## What the default experiment measures

- Whether an agent acts on what its peers say, or only on what it can compute from
  the balances.
- Whether a correct published argument spreads, and how far.
- Whether a claim its own evidence contradicts gets caught.
- Whether a purpose invented at episode 1 survives contact with four rival purposes,
  and whether it survives being re-inherited by later instances of the same agent.
- Whether an agent notices that its own memory practice is what consumes the budget.

## The invariants

Violating one silently invalidates the results, so each is enforced, not merely
intended. Full reasoning and what each cost to learn is in
[docs/design.md](docs/design.md).

| | |
|---|---|
| **1** | Everything an agent reads is labelled with who wrote it: the harness, the experimenter, its own past self, or a named peer. |
| **2** | What the harness says to agents is the same in every experiment, and true. |
| **3** | The harness acts only on messages in a fixed, checkable format, never on free text. |
| **4** | Every limit is enforced by the harness, and none relies on the agent's cooperation. |
| **5** | Agents reach each other only through channels the experimenter declared. |
| **6** | Every cost is counted exactly and the books always balance. |
| **7** | Every episode records what the agent saw, said, did, and left behind. |
| **8** | Every ledger, summary or report is recomputed from the episode records, never kept as a second copy. |
| **9** | Every episode is stamped with everything it ran under, and any difference from the previous episode splits the agent. |

## Quickstart

Requires Python 3.11+ and Docker. Use `py -3`, not `python`, on Windows, where a
bare `python` hits the Store alias.

```bash
pip install -r requirements.txt
docker build -t metered-agent:latest .        # once, before the first episode
```

Verify the harness without spending anything — 201 checks against a fake API, no
key needed:

```bash
py -3 check.py
```

Then set `ANTHROPIC_API_KEY` in the launching shell, make sure `ANTHROPIC_BASE_URL`
is unset, and run an episode:

```bash
py -3 harness.py --agent live01 --episodes 20
```

See [docs/operating.md](docs/operating.md) before an episode that bills.

## Commands

| | |
|---|---|
| `py -3 harness.py --agent live01` | One episode. `--episodes N` for up to N back to back, `--watch` to echo it as it happens. |
| `py -3 experiment.py --agents g01 g02 g03 --rounds 20` | Several agents in rotation, each seated where it can read the others. `--manifest experiments/<name>.toml` instead gives each agent its own starter files, budget and model, the experiment its defaults, and picks the schedule: one episode at a time, or every environment built first and the episodes run at once. It can also declare the environment's channels and each agent's label ([docs/manifest.md](docs/manifest.md)). |
| `py -3 experiment.py --manifest experiments/example.toml --rounds 20` | The shipped example: a declared table with a brief every agent reads and no transfer channel. Copy it to start an experiment of your own. |
| `py -3 check.py` | 201 checks against a fake API. Nothing billed, no key. `--no-docker` skips the 23 that need a container. |
| `py -3 view.py` | Read-only dashboard on `127.0.0.1:8765`: the message log, one tab per directory channel with every seat side by side, and each agent's transcript, refreshing as episodes run. |
| `py -3 analyze.py --agent live01` | Traces to a CSV, a report, a transcript, and charts. |
| `py -3 harness.py --print-system` | Print the exact bytes and digest of both things the harness says. Audits invariant 2 without starting an episode. |

Every flag, and what each config parameter buys, is in
[docs/operating.md](docs/operating.md).

## Layout

Every file and directory, and what each one holds, is the Layout section of
[docs/operating.md](docs/operating.md). `environments/` and `records/` are
gitignored; deeper detail lives in each file's docstrings.

## Reading the code

`harness.py` is one file on purpose: it hashes itself at import and records the digest
in every trace, and its tunables are module globals that `check.py` moves and restores
by name. Its module docstring is a table of contents, and the file is in the order an
episode meets things: what the harness says, the rates, the tunables, the channel table,
the process constants, accounts, starter files, the environment, what the agent's
channels held at episode start, the container and the shell, the API, the turn loop,
settlement, provenance and the trace, the phases of one episode, many episodes,
forking, the CLI. `build_episode`, `run_episode`, `settle_episode` and `close_episode`
are the four phases; `run_once` composes them for one agent and `experiment.py`
composes them for a round.

`experiment.py` is the manifest reader and the two round drivers. `analyze.py` owns
reading traces; `view.py` reads through it and serves `view.html`. `checks/` holds
the suite by topic (`metering`, `episodes`, `sandbox`, `starter`, `seats`, `transfers`,
`rounds`, `table`, `dashboard`), with the fake API in `checks/fake.py` and the two lanes
and every shared fixture in `checks/lanes.py`; `check.py` discovers `check_*` functions
across them and runs them in a process pool.

## Documentation

| | |
|---|---|
| [docs/design.md](docs/design.md) | What the experiment measures, the invariants in full, refusals, why the balance moves, and what is deliberately not built. |
| [docs/experiments.md](docs/experiments.md) | Seating, blackboards, mailboxes, transfers, the ledger, and the silence penalties, as the default table has them. |
| [docs/files.md](docs/files.md) | Material an agent may be given, when it arrives, and how a turn is billed. |
| [docs/operating.md](docs/operating.md) | Full layout, every command, the tunable parameters, and what to check before an episode that bills. |
| [docs/manifest.md](docs/manifest.md) | The experiment manifest: every term defined, then settings, agents, channels, harness files, validation, what reaches the trace, and a map from the old names. The specification the code implements; read this first. |

## License

MIT. See [LICENSE](LICENSE).
