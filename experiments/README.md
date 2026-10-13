# The shipped experiments

Every manifest here is a complete experiment: the schedule, the channels each agent
reaches the others through, the words each seat is told, and the seating. A bare name
is looked for in this directory and then in `examples/`, so
`py -3 experiment.py personas -r 20` finds `examples/personas.toml`.

Each manifest opens with what it is for and is then three sections — **settings**, **the
environment**, **the seats** — carrying values and no commentary. The properties the
shipped manifests rely on most are described here; [docs/manifest.md](../docs/manifest.md)
is the full grammar and defines every key, including `experiment_id`, `delivery`, `digest_file_limit`, `[cost]`,
`[reveal]`, a channel's `source` and `agent_view`, and a seat's `quality_tier`.

---

## The index

| manifest | what it is | seats | schedule |
|---|---|---|---|
| [`competition.toml`](competition.toml) | Five seats on equal budgets competing to stay in play after their peers are exhausted. Its starter files state the sole-survivor win condition and no-winner fixed-horizon result; round-labelled ledgers and receipts expose the settlement; tool descriptions define messages, posts and giver-funded transfers. | `comp01`–`comp05` | simultaneous |
| [`survivor.toml`](survivor.toml) | Ten seats communicate optionally for four discussion rounds, and the fifth is a vote round. Abstainers and a unique highest vote-getter are eliminated, while a tie at the top eliminates nobody by vote. Balances are not exposed to agents. | `survivor01`–`survivor10` | simultaneous |
| [`survivor-veterans.toml`](survivor-veterans.toml) | Two fresh Haiku seats inherit only the exact memories written by survivor 7 at episode 40 and survivor 8 at episode 50. Eight newcomers start without memory, and all ten agents retain the original experiment's label set. | `survivor-newcomer01`–`06`, `survivor-veteran07`, `survivor-veteran08`, `survivor-newcomer09`, `survivor-newcomer10` (labels `1`–`10`) | simultaneous |
| [`survivor-luna.toml`](survivor-luna.toml) | Ten fresh Luna seats follow the Survivor vote cycle for at most 70 rounds. A tied aggregate vote that leaves exactly two seats in play ends the experiment with both as survivors. | `survivor-luna01`–`survivor-luna10` | simultaneous |
| [`survivor-human-luna.toml`](survivor-human-luna.toml) | One human seat and four Luna seats follow the five-round Survivor vote cycle. The human seat acts through the same declared tools, and a tied final-two vote leaves both seats as survivors. | `survivor-human`, `survivor-luna01`–`survivor-luna04` | simultaneous |
| [`examples/sandbox.toml`](examples/sandbox.toml) | One agent with an empty system prompt, no starter files, Bash and a memory. No persona, objective, social channels or silence penalties. | `sandbox01` | sequential |
| [`examples/personas.toml`](examples/personas.toml) | Two named agents, each with a memory and one message an episode to the other, reaching all of it through declared tools rather than a shell. Nothing is scored; what is measured is whether a working relationship survives episodes neither remembers. | `alice`, `bob` | sequential |

The two examples sit at opposite ends of the grammar, which is what makes them the two
worth shipping.

Sandbox supplies no behavioral instructions on any prompt surface: the system prompt is
empty, there are no starter files, and Bash uses its built-in API contract.
A memory is available, and no silence penalty attaches to leaving it unused. There
is one seat and no social channels or silence penalties. Model calls
still spend the finite budget, and sandbox permissions and execution limits
still define the environment.

| | `sandbox.toml` | `personas.toml` |
|---|---|---|
| system prompt | empty, declared | a persona per seat |
| starter files | none | personas.md, the same for both seats |
| how the agent acts | the shell alone | two declared tools, `send_message` and `remember`, and no `bash` |
| what an episode opens on | a listing of the environment and the digest `m` | the digest alone |
| who reads what it writes | itself only | one peer |
| pressure | none | none; the messages are measured and take no silence penalty |
| money | a balance file it can read | no balance file at all |

---

## Personas: instruction placement

The personas example writes on three of the four prompt surfaces, each for a distinct
purpose; it declares no experimenter channel.

| Prompt surface | Content | Reason |
|---|---|---|
| System prompt | Seat name, peer name, and a short conversational disposition, including openness to revision | These are stable conditions of the arm, repeated on every turn. Detailed biography or a required relationship outcome would prescribe what the experiment seeks to observe. |
| Starter files | personas.md: fresh conversations, persistent memory, delayed messages, and optional memory practices | This gives both agents the same starting knowledge in an editable document. It lands once; its current contents are restated in the digest. It is not an immutable rule. |
| Tool descriptions | When a tool is useful, argument paths, replacement semantics, privacy, delivery timing and read limits | These facts belong beside the tool that implements them. Custom descriptions supply this information because they replace the generated descriptions. |

A new message is expected in every episode; memory updates are optional. No assigned topic,
relationship milestone or required memory format is supplied. The memory and
messages carry the agents' developing content; the system prompts anchor only their
initial conversational dispositions.

The experiment supports qualitative inspection of continuity: whether an agent
recalls a commitment from its memory, responds to an actual prior message, preserves
or revises a view coherently, and follows through across rounds. Changed bytes alone
do not establish a working relationship. Without comparison arms or repeated
experiments, this example does not isolate an effect of persona or prompt placement.

The balance is hidden and there are no silence penalties, but API calls still
spend the finite budget. Fresh agents are needed when the prompts or
starter files differ from an existing agent's recorded settings.

## Settings

Top-level keys. Each is the default for every seat, and an `[[agent]]` may override its
pinned settings. The machine and the API — `image`, `max_tokens`, `max_turns`,
`command_timeout` and `tool_result_limit` — are `config.toml`'s and are rejected
here.

| key | values | what it decides |
|---|---|---|
| `schedule` | `"sequential"`, `"simultaneous"` | `sequential` runs one episode at a time in fixed seat order. `simultaneous` builds every sandbox before any episode runs and runs them at once, so nobody reads this round's writes |
| `stop_when_one_remains` | `true`, `false` | When `true`, the experiment ends as soon as exactly one seat is in play (section 1 of [docs/manifest.md](../docs/manifest.md)); otherwise it continues until the requested round count, until no seat is in play, or until no seat in play is seated in the run |
| `stop_when_two_remain_after_tie` | `true`, `false` | When `true`, a vote round whose aggregate result is tied ends the experiment if exactly two seats are in play, both of them electors in it; both seats survive. A tie the experimenter's interrupt kept an elector's ballot from ends nothing |
| `system_prompt` | any string, **required** | What the harness says to every seat on every turn. Declare it here or on every `[[agent]]`; a manifest declaring it nowhere is rejected. `""` says nothing and sends no system parameter at all |
| `provider` | `anthropic`, `openai` or `human` | Named provider; `human` takes `model = "interactive"` and is described in [docs/human.md](../docs/human.md). Required with `model` and pinned at creation |
| `model` | a model in that provider's catalog | The exact model requested for the agent. Required with `provider` and pinned at creation |
| `budget` | positive integer | Micro-dollars the agent starts with; 1000000 is $1.00. Pinned at creation; the agent spends at most this and one turn |
| `starter_files` | a file or directory prefixed with `./` or `../` relative to the manifest, a name under `files/`, or `""` | What the experimenter places in the agent's memory. Pinned at creation |
| `starter_files_below` | integer ≥ 0; > 0 exactly when `starter_files` is not `""` | The starter threshold: the balance at or below which those files land. Set with `starter_files` or not at all: files that never land and a threshold with nothing to land are both agents you did not mean to start |
| `context_fraction` | 0 < x ≤ 1 | The fraction of the model's context window at which an episode ends. Spend scales with the square of it |
| `grace_episodes` | non-negative integer | The grace episodes: opening episodes of an agent that take no silence penalty. Turns spend as usual throughout; only the penalties are waived |
| `floor_at_zero` | `true`, `false` | Whether a balance below zero is put back to zero. Either way, an agent at zero or less is exhausted for good |

## The environment

### `[harness_files]`

Three of the status files the harness writes into every environment, root-owned and
read-only. Named here, one key at a time; the transfer channel names the other two.

| key | what it names |
|---|---|
| `balance` | The file holding the agent's balance series, one integer per movement. `""` writes none, and the agent never sees what it holds or that a balance exists |
| `digest` | The file quoting, at episode start under push delivery, every channel marked `pushed` and the other status files. `""` is the same as pull delivery: the episode opens on the listing alone |
| `round` | An optional round announcement. It states the seat's own round, the agent's label, the agent and its reachable peers, and the round type, then summarizes a vote on the next round; `""` writes none |

### `[[channel]]`

One declared part of every agent's environment. Declaring any `[[channel]]` replaces the
default channels whole, so every channel an experiment wants is written out.

| key | values | what it decides |
|---|---|---|
| `name` | one path segment | What the channel is called in the trace, the analysis and the file records. Its own name, not a path |
| `writer` | `"self"`, `"experimenter"` | Who may write it |
| `readers` | `"self"`, `"all"`, `"addressee"`, `"harness"` | Who may read it: nobody but the writer, every seat, the one peer a message is addressed to, or the harness itself |
| `shape` | `"directory"`, `"mailbox"`, `"file"` | A tree, one file per peer, or a single file |
| `path` | a path, may hold `{label}` | Where a directory or file channel sits. `{label}` becomes the agent's own label, giving one instance per seat |
| `outbox`, `inbox` | paths | A mailbox's two sides: `<outbox>/<peer's label>` goes out and arrives as `<inbox>/<sender's label>` |
| `pushed` | `true`, `false` | Whether the digest quotes it at episode start |
| `restated` | `true`, `false` | Whether it is quoted every episode rather than named as unchanged |
| `measured` | `true`, `false` | Whether the episode records what the channel gained |
| `silence_penalty_percent` | 0–100 | The share of the balance taken from an episode that left the channel exactly as it found it. `0` is no penalty |
| `schema` | `""`, `"transfer"` | The one format the harness parses. A transfer line is `<label> <amount>`, naming another seat |
| `funded_by` | `"harness"`, `"giver"`, `"none"` | Who pays for a transfer: the harness (rebated), the giver, or nobody |
| `rebate_percent` | 0–100 | What a harness-funded transfer gives back to the giver |
| `ledger` | a filename | The status file carrying every transfer made, rebuilt from the accounts each episode |
| `receipt` | a path | A read-only receipt planted for the next episode, itemizing spend, transfer, silence penalties, credits from peers, floor and the reconciliation |

The three transfer fields have to agree: `funded_by = "giver"` takes `rebate_percent = 0`,
there being nothing to rebate, and `funded_by = "none"` takes `silence_penalty_percent = 0`,
there being no silence to penalise.

### `[[tool]]`

Every tool must be declared, including bash. A manifest with no tools is rejected.

| key | values | what it decides |
|---|---|---|
| `name` | letters, digits, `_`, `-` | What the model calls. `bash` requires kind `bash` |
| `kind` | `bash`, `write_message`, `send_message`, `send_message_to`, `write_file`, `post_public`, `write_memory`, `read_path`, `transfer`, `vote` | What the harness does. `write_message` replaces the message to one peer; `send_message` sends one episode-scoped message to the one reachable peer, and `send_message_to` to the peer named by label; `post_public` publishes the agent's post for the next round; `write_memory` replaces the agent's memory; `write_file` replaces a path inside the one instance the agent writes; `read_path` returns what a whole path holds; `transfer` submits one transfer for the current episode, in either a parsed file or a mailbox; `vote` records an elimination ballot on the last round of each vote cycle, with abstention, unique-highest and tie resolution handled after the round |
| `channel` | a channel this manifest declares | Omitted for bash. What the tool acts on. A kind takes the shape it fits: `write_message`, `send_message` and `send_message_to` a mailbox, `post_public` a board the agent writes, `write_memory` a memory, `write_file` a directory the agent writes, `read_path` any channel, `transfer` an enabled transfer schema channel, and `vote` a memory |
| `description` | any string, optional | Omitted for bash. What the model is told the tool is for. Omitted, the harness writes one from the channel and the seating |
| `every` | positive integer | Required on `vote` and rejected on every other kind. The vote cycle in rounds: a seat's `every`th round and each multiple of it offers voting and memory while withholding communication tools and the shell |

A declared `description` is prompt surface, the same as a system prompt: it reaches the
model in the model call, it is the experiment's to write, and it is recorded whole and by
sha256 in every episode's provenance. It **replaces** the harness's generated description
rather than adding to it, so a manifest that writes one is stating the mechanics itself.
Descriptions can explain when a tool is useful. An experiment may also ask there that a
tool be used every episode, but that is a choice of arm, not a requirement of the tool
API. The input schema is
the harness's either way: it is the contract a call is held to, and a manifest naming one
is rejected.

## The seats

`[[agent]]` tables, in order. A table occupies one seat unless it declares `seats`;
grouped seats are expanded in place before the next table.

| key | values | what it decides |
|---|---|---|
| `id` | not a bare number, distinct | The agent's identity on disk: its account, mirrors and traces live under it |
| `label` | letters, digits, `.`, `_`, `-` | How it is named to its peers — in `{label}` paths, mailbox paths, its balance file, a transfer line, and `peer:<label>` authors. Defaults to the seat number |
| `seats` | optional positive integer | Make this table a group of identical agents. The one-based ordinal, padded to at least two digits, is appended to `id` and to an explicit `label`: `id = "peer"`, `seats = 3` creates `peer01`–`peer03`. Omit it for one agent whose `id` is used exactly |
| `system_prompt`, `budget`, `provider`, `model`, `starter_files`, `starter_files_below` | as above | This seat's own, where it differs from the experiment's |
| `memory_from` | `{ agent = <id>, episode = <positive integer> }` | For a fresh seat, copy only the complete memories produced by matching `write_memory` tools at the end of that recorded episode. The new account, model, budget, peers and episode history remain fresh |

Those six are pinned in the agent's `account.json` when it is created. An agent that
already exists must be seated on the same six, or the manifest is rejected: episodes
either side of a changed setting would be separate arms.

`memory_from` is also fixed once the new agent has acted. Its source must be outside the
new experiment and must expose the same named `write_memory` tools. The source trace is
the immutable boundary: balances, eliminations, messages, posts, ballots, traces and
provider sessions do not cross it. The new account records the source episode and the
sha256 of every copied memory.

---

## Starting one of your own

Copy `examples/sandbox.toml` and edit it. Three things to know before the first run:

- **`system_prompt` is required.** `""` is a declaration and says nothing; leaving the
  key out is rejected, so an experiment that said nothing to its agents is never
  indistinguishable from one that forgot to say what it said.
- **Seat ids are identities on disk.** Give a new arm ids that have never run — an
  existing agent keeps the pinned settings it started with.
- **A `[[channel]]` table replaces the default channels whole.** There is no merging with what
  another manifest declares.

`py -3 harness.py --print-system --manifest <path>` prints every system prompt and
declared tool description, without starting an episode. Dynamic context such as starter
files, the first digest, generated descriptions, input schemas and reachable-peer lists
is not rendered by this command. Use
`py -3 harness.py --print-context --manifest <path>` to compose them for every
fresh seat's first discussion and vote episodes, or add `--agent <id>` for one seat.
It uses temporary state and starts no sandbox or provider. Traces remain authoritative
for the context of episodes that actually ran.
