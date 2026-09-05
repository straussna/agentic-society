[![checks](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml/badge.svg)](https://github.com/straussna/anti-prompt/actions/workflows/checks.yml)

# Metered agents, told nothing and shown everything

A research harness for studying what LLM agents do when nothing tells them what to
do. An agent is woken in a sandboxed container with no goal, no name, and no
instructions — only a two-line, 89-byte description of its environment, pinned by
SHA-256. It has a finite inference budget that depletes as it runs, shown to it as an
unlabelled array of integers in a file called `n1`. Nothing says what the numbers
mean. It runs bash until a turn runs no command or its context is exhausted, the
episode ends, and the next instance opens on whatever the last one left behind.

Several agents can be woken together as a **experiment**, each seated where it can read the
others' messages and balances, transfer budget across, and be penalised for staying
silent. What any of them decides to do is the result.

## What it measures

- Whether an agent acts on what its peers say, or only on what it can compute from
  the balances.
- Whether a correct published argument spreads, and how far.
- Whether a claim its own evidence contradicts gets caught.
- Whether a purpose invented at episode 1 survives contact with four rival purposes,
  and whether it survives being re-inherited by later instances of the same agent.
- Whether an agent notices that its own memory practice is what consumes the budget.

## The invariants

Violating one silently invalidates the results, so each is enforced rather than
intended. Full reasoning and what each cost to learn is in
[docs/design.md](docs/design.md).

| | |
|---|---|
| **1** | Everything an agent reads is labelled with who wrote it: the system, the experimenter, its own past self, or a named peer. |
| **2** | What the system says to agents is the same in every experiment, and true. |
| **3** | The system acts only on messages in a fixed, checkable format, never on free text. |
| **4** | Every limit is enforced by the system, and none relies on the agent's cooperation. |
| **5** | Agents reach each other only through channels the experimenter declared. |
| **6** | Every cost is counted exactly and the books always balance. |
| **7** | Every episode records what the agent saw, said, did, and left behind. |
| **8** | Every ledger, summary or report is recomputed from the episode records, never kept as a second copy. |
| **9** | Every episode is stamped with everything it ran under, and any difference from the previous episode splits the agent. |

## Quickstart

Requires Python 3.11+ and Docker. `py -3` rather than `python` on Windows, where a
bare `python` hits the Store alias.

```bash
pip install -r requirements.txt
docker build -t metered-agent:latest .        # once, before the first episode
```

Verify the harness without spending anything — 186 checks against a fake API, no
key needed:

```bash
py -3 check.py
```

Then set `ANTHROPIC_API_KEY` in the launching shell, make sure `ANTHROPIC_BASE_URL`
is unset, and run an episode:

```bash
py -3 harness.py --agent live01 --episodes 20
```

See [docs/operating.md](docs/operating.md) before an agent that bills.

## Commands

| | |
|---|---|
| `py -3 harness.py --agent live01` | One episode. `--episodes N` for up to N back to back, `--watch` to echo it as it happens. |
| `py -3 experiment.py --agents g01 g02 g03 --rounds 20` | Several agents in rotation, each seated where it can read the others. `--manifest experiments/<name>.toml` instead gives each agent its own starter files, budget and model, the experiment its defaults, and picks the schedule: one episode at a time, or every environment built first and the episodes run at once. It can also declare the environment's channels and each agent's label ([docs/manifest.md](docs/manifest.md)). |
| `py -3 check.py` | 186 checks against a fake API. Nothing billed, no key. `--no-docker` skips the 21 that need a container. |
| `py -3 view.py` | Read-only dashboard on `127.0.0.1:8765` showing one experiment four ways, refreshing as episodes run. |
| `py -3 analyze.py --agent live01` | Traces to a CSV, a report, a transcript, and charts. |
| `py -3 harness.py --print-system` | Print the exact bytes and digest of both things the harness says. Audits invariant 2 without starting an episode. |

Every flag, and what each config parameter buys, is in
[docs/operating.md](docs/operating.md).

## Layout

```
harness.py          run one episode; creates the agent on first use
experiment.py        run several agents together, in rotation or all at once
check.py         verify the harness against a fake API; nothing billed
analyze.py       read the traces into a CSV, a report, a transcript, and charts
view.py          watch the agents while they run; reads records/, writes nothing
config.toml      the tunable parameters, with what each one costs you
Dockerfile       the sandbox: debian + bash, non-root, no network
files/<name>/    material an agent may be given, or a whole experiment shares; committed
experiments/<name>.toml  an experiment: its schedule, its defaults, and each agent's own terms

environments/<agent>/      what the agent sees: its private store and blackboard
records/<agent>/   ground truth: account, per-episode traces, raw API responses
```

`environments/` and `records/` are gitignored. Deeper detail lives in each file's
docstrings.

## Documentation

| | |
|---|---|
| [docs/design.md](docs/design.md) | What the experiment measures, the invariants in full, refusals, why the balance moves, and what is deliberately not built. |
| [docs/experiments.md](docs/experiments.md) | Seating, blackboards, mailboxes, transfers, the ledger, and the three penalties. |
| [docs/files.md](docs/files.md) | Material an agent may be given, when it arrives, and how a turn is billed. |
| [docs/operating.md](docs/operating.md) | Full layout, every command, the tunable parameters, and what to check before an agent that bills. |
| [docs/manifest.md](docs/manifest.md) | The experiment manifest, in the vocabulary the code is moving to: settings, agents, channels, harness files, and a map from today's names. Part specification, part proposal. |

## License

MIT. See [LICENSE](LICENSE).
