[![checks](https://github.com/straussna/agentic-society/actions/workflows/checks.yml/badge.svg)](https://github.com/straussna/agentic-society/actions/workflows/checks.yml)

# Metered agents in a declared environment

A research harness for running societies of LLM agents in environments the
experimenter declares. Each agent runs bash in a sandboxed container holding a
directory tree, and is told whatever the experiment declared the harness should say
to it. Everything else it knows, it reads from files. An episode ends when a turn
runs no command, and the next instance opens on whatever the last one left behind.

The harness itself says nothing. What it writes it computes from the accounts — the
balances, the digest, the ledger — and rewrites whenever those move; a fixed line is a
constant, and constants are the experimenter's to declare. So every manifest states what
its seats are told, and `system_prompt = ""` declares an empty system prompt: no goal, no name,
no instructions, and no system parameter sent at all. It is one arm among many and it is
declared like any other - silence is never inherited. A manifest may say something to
every seat, or something different to each, and every episode records what its agent
was told, whole and by digest.

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
| **2** | What the harness says to agents is declared, recorded, and true. |
| **3** | The harness acts only on messages in a fixed, checkable format, never on free text. |
| **4** | Every limit is enforced by the harness, and none relies on the agent's cooperation. |
| **5** | Agents reach each other only through channels the experimenter declared. |
| **6** | Every cost is counted exactly and the books always balance. |
| **7** | Every episode records what the agent saw, said, did, and left behind. |
| **8** | Every ledger, summary or report is recomputed from the episode records, never kept as a second copy. |
| **9** | Every episode is stamped with everything it ran under, and any difference from the previous episode splits the agent. |

## Quickstart

Requires Python 3.14+ and Docker. Use `py -3`, not `python`, on Windows, where a
bare `python` hits the Store alias.

```bash
pip install -r requirements.txt
docker build -t metered-agent:latest .        # once, before the first episode
```

Verify the harness without spending anything — 273 checks against a fake API, no
key needed:

```bash
py -3 check.py
```

Set the API key for every provider seated by the manifest. Anthropic uses
`ANTHROPIC_API_KEY`; OpenAI uses `OPENAI_API_KEY`. Custom `ANTHROPIC_BASE_URL` and
`OPENAI_BASE_URL` values are refused. Each seat declares an explicit `provider` and
`model`, so one experiment can compare the two direct APIs. Then run an episode:

```bash
py -3 experiment.py competition -r 20
```

See [docs/operating.md](docs/operating.md) before an episode that bills.

## Commands

| | |
|---|---|
| `py -3 harness.py --agent live01 --manifest experiments/<name>.toml` | One episode for one seat of an experiment. `--episodes N` for up to N back to back, `--watch` to echo it as it happens. |
| `py -3 experiment.py <name> -r 20` | Every agent the manifest seats, each where it can read the others. The manifest gives each agent its own prompt, starter files, budget and model, the experiment its settings, its channels and the named actions it offers beside the shell, and picks the schedule: one episode at a time, or every environment built first and the episodes run at once ([docs/manifest.md](docs/manifest.md)). A bare name is looked for under `experiments/` and then `experiments/examples/`; anything with a suffix or a directory in it is the path it is, and `-m/--manifest` and `-r/--rounds` are the long forms. |
| `py -3 experiment.py competition -r 20` | The default experiment, declared: five seats on one set of starter files, the shipped channel table written out, sequential. |
| `py -3 experiment.py personas -r 20` | Two named agents with a private journal and one letter each to the other, nothing scored, a shared starter orientation, and optional correspondence through three declared tools: with no declared Bash tool, they reach their environment through those actions, and the episode opens on the digest rather than a listing, so neither ever reads a filesystem. |
| `py -3 experiment.py sandbox -r 20` | One agent with an empty system prompt, no starter document, Bash and private persistent storage. No persona, objective, social channels or silence penalties. Copy it to start an experiment of your own. |
| `py -3 check.py` | 273 checks against a fake API. Nothing billed, no key. `--no-docker` skips the 25 that need a container. |
| `py -3 view.py` | Read-only dashboard on `127.0.0.1:8765`: the message log, one tab per directory channel with every seat side by side, and each agent's transcript, refreshing as episodes run. |
| `py -3 analyze.py --agent live01` | Traces to a CSV, a report, a transcript, and charts. |
| `py -3 harness.py --print-system` | Print the exact bytes and digest of what the harness ships and of the prompt in force; `--manifest PATH` adds the prompt each seat of an experiment is told and every tool description it declares. Audits invariant 2 without starting an episode. |
| `py -3 human.py --agent <agent-id>` | Attach a terminal to an interactive seat's pending turn and act through the same declared tools an autonomous seat has ([docs/human.md](docs/human.md)). |

Every flag, and what each config parameter buys, is in
[docs/operating.md](docs/operating.md).

## Layout

Every file and directory, and what each one holds, is the Layout section of
[docs/operating.md](docs/operating.md). `environments/`, `records/`,
`experiment_records/`, `interactions/` and `displaced/` are gitignored; deeper detail
lives in each file's docstrings.

## Reading the code

`harness.py` is one file on purpose: it hashes itself at import and records the digest
in every trace, and its tunables are module globals that `check.py` moves and restores
by name. Its module docstring is a table of contents, and the file is in the order an
episode meets things: what the harness says, the rates, the tunables, the channel and
tool tables, the process constants, accounts, starter files, the environment, what the agent's
channels held at episode start, the container and the shell, the API, the turn loop,
settlement, provenance and the trace, the phases of one episode, many episodes,
forking, the CLI. `build_episode`, `run_episode`, `settle_episode` and `close_episode`
are the four phases; `run_once` composes them for one agent and `experiment.py`
composes them for a round.

`experiment.py` is the manifest reader and the two round drivers; `product.py` keeps
the experiment-level progress, cost, lineage and outcome records. `providers/` holds
the Anthropic, OpenAI and interactive-human adapters, and `interaction/` the local
coordination `human.py` attaches to. `analyze.py` owns reading traces; `view.py` reads
through it and serves `view.html`. `checks/` holds the suite by topic (`metering`,
`episodes`, `sandbox`, `starter`, `seats`, `transfers`, `rounds`, `table`, `tools`,
`memory`, `providers`, `interaction`, `product`, `analysis`, `dashboard`, `suite`),
with the fake API in `checks/fake.py` and the two lanes
and every shared fixture in `checks/lanes.py`; `check.py` discovers `check_*` functions
across them and runs them in a process pool.

## Documentation

| | |
|---|---|
| [docs/design.md](docs/design.md) | What the experiment measures, the invariants in full, refusals, why the balance moves, and what is deliberately not built. |
| [docs/experiments.md](docs/experiments.md) | Seating, blackboards, mailboxes, transfers, the ledger, and the silence penalties, as the default table has them. |
| [docs/files.md](docs/files.md) | Material an agent may be given, when it arrives, and how a turn is billed. |
| [docs/operating.md](docs/operating.md) | Full layout, every command, the tunable parameters, and what to check before an episode that bills. |
| [docs/human.md](docs/human.md) | Interactive seats: the manifest terms, attaching with `human.py`, and how a human turn is recorded. |
| [experiments/README.md](experiments/README.md) | The index of shipped experiments: what each one is, the seats it declares, and which arm it is read against. |
| [docs/manifest.md](docs/manifest.md) | The experiment manifest: every term defined, then settings, agents, channels, tools, harness files, validation, what reaches the trace, and a map from the old names. The specification the code implements; read this first. |

## License

MIT. See [LICENSE](LICENSE).
