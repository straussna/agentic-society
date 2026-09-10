# The shipped experiments

Every manifest here is a complete experiment: the schedule, the channels each agent
reaches the others through, the words each seat is told, and the seating.   bare name
is looked for in this directory and then in `examples/`, so
`py -3 experiment.py personas -r 20` finds `examples/personas.toml`.

Each manifest opens with what it is for and is then three sections — **settings**, **the
environment**, **the seats** — carrying values and no commentary. What every property
means is here. [docs/manifest.md](../docs/manifest.md) is the full grammar, including
the keys nothing shipped uses.

---

## The index

| manifest | what it is | seats | schedule |
|---|---|---|---|
| [`competition.toml`](competition.toml) | Five equally funded seats competing to remain funded after their peers are out. The complete brief states the sole-survivor win condition and no-winner fixed-horizon result; round-labelled ledgers and settlement receipts expose the accounting; tool descriptions define billed messages, posts and sender-funded transfers. | `comp01`–`comp05` | simultaneous |
| [`examples/sandbox.toml`](examples/sandbox.toml) | One agent with an empty system prompt, no starter document, Bash and private persistent storage. No persona, objective, social channels or silence penalties. | `sandbox01` | sequential |
| [`examples/personas.toml`](examples/personas.toml) | Two named agents with a private memory and one letter each to the other, reaching all of it through declared actions rather than a shell. Nothing is scored; what is measured is whether a working relationship survives episodes neither remembers. | `alice`, `bob` | sequential |

The two examples sit at opposite ends of the grammar, which is what makes them the two
worth shipping.

Sandbox supplies no behavioral instructions on any surface: the system prompt is
empty, there is no starter document, and Bash uses its built-in  PI contract.
Private persistent storage is available without any obligation to use it. There
is one seat and no social channels or silence penalties. Model calls
still consume the finite budget, and container permissions and execution limits
still define the environment.

| | `sandbox.toml` | `personas.toml` |
|---|---|---|
| system prompt | empty, declared | a persona per seat |
| starter document | none | empty personas.md |
| how the agent acts | the shell alone | three declared tools, no declared `bash` tool |
| what an episode opens on | a listing of the environment | the digest alone |
| who reads what it writes | itself only | one peer |
| pressure | none | none; the letters are measured and charged nothing |
| money | a balance file it can read | no balance file at all |

---

## Personas: instruction placement

The personas example uses all three surfaces, each for a distinct purpose.

| Surface | Content | Reason |
|---|---|---|
| System prompt | Seat name, peer name, and a short conversational disposition, including openness to revision | These are stable treatment conditions repeated on every turn. Detailed biography or a required relationship outcome would prescribe what the experiment seeks to observe. |
| Starter document | Shared orientation: fresh conversations, persistent memory, delayed letters, and optional memory practices | This gives both agents the same starting knowledge in an editable document. It lands once; its current contents are restated in the opening context. It is not an immutable rule. |
| Tool descriptions | When an action is useful, argument paths, replacement semantics, privacy, delivery timing and read limits | These facts belong beside the action that implements them. Custom descriptions supply this information because they replace the generated descriptions. |

eetter writing and identity updates are optional.   mandatory new letter each
episode would make correspondence partly a compliance measure. No assigned topic,
relationship milestone or required memory format is supplied. The memory and
letters carry the agents' developing content; the system prompts anchor only their
initial conversational dispositions.

The experiment supports qualitative inspection of continuity: whether an agent
recalls a commitment from its notes, responds to an actual prior letter, preserves
or revises a view coherently, and follows through across rounds. Changed bytes alone
do not establish a working relationship. Without comparison arms or repeated
experiments, this example does not isolate an effect of persona or prompt placement.

The balance is hidden and there are no silence penalties, but  PI calls still
consume the finite budget. Fresh agents are needed when the revised prompts or
starter source differ from an existing agent's recorded terms.

## Settings

Top-level keys. Each is the default for every seat, and an `[[agent]]` may override the
last five for itself. The machine and the  PI — `image`, `max_tokens`, `max_turns`,
`command_timeout` and `tool_result_limit` — are `config.toml`'s and are refused
here.

| key | values | what it decides |
|---|---|---|
| `schedule` | `"sequential"`, `"simultaneous"` | `sequential` runs one episode at a time in fixed seat order. `simultaneous` builds every environment before any episode runs and runs them at once, so nobody reads this round's writes |
| `stop_when_one_remains` | `true`, `false` | When `true`, the experiment ends as soon as exactly one funded seat remains; otherwise it continues until the requested round count or every seat is out |
| `system_prompt` | any string, **required** | What the harness says to every seat on every turn. Declare it here or on every `[[agent]]`; a manifest declaring it nowhere is refused. `""` says nothing and sends no system parameter at all |
| `provider` | `anthropic` or `openai` | Named direct-API adapter. Required with `model` and pinned at creation |
| `model` | a model in that provider's catalog | The exact model requested for the agent. Required with `provider` and pinned at creation |
| `budget` | positive integer | Micro-dollars the agent starts with; 1000000 is $1.00. Pinned at creation, and the ceiling on what the agent can cost |
| `starter_files` | a file or directory prefixed with `./` or `../` relative to the manifest, a name under `files/`, or `""` | What the experimenter places in the agent's private store. Pinned at creation |
| `starter_files_below` | non-negative integer | The balance at or below which those files land. Set with `starter_files` or not at all: files that never land and a threshold with nothing to land are both agents you did not mean to start |
| `context_fraction` | 0 < x ≤ 1 | The fraction of the model's context window at which an episode ends. Cost scales with the square of it |
| `grace_episodes` | non-negative integer | Opening episodes of an agent that answer for no silence penalty. Turns are billed as usual throughout; only the penalties are waived |
| `floor_at_zero` | `true`, `false` | Whether a balance below zero is put back to zero. Either way, zero or less ends the agent for good |

## The environment

### `[harness_files]`

The files the harness writes into every environment, root-owned and read-only. Named
here, one key at a time.

| key | what it names |
|---|---|
| `balance` | The file holding the agent's balance series, one integer per movement. `""` writes none, and the agent never sees what it holds or that a balance exists |
| `digest` | The file quoting what the agent's channels hold. `""` is the same as pull delivery: the episode opens on the listing alone |

### `[[channel]]`

One declared part of every agent's environment. Declaring any `[[channel]]` replaces the
default set whole, so every channel an experiment wants is written out.

| key | values | what it decides |
|---|---|---|
| `name` | one path segment | What the channel is called in the trace, the analysis and the file records. Its own name, not a path |
| `writer` | `"self"`, `"experimenter"` | Who may write it |
| `readers` | `"self"`, `"all"`, `"addressee"`, `"harness"` | Who may read it: nobody but the writer, every seat, the one peer a slot is addressed to, or the harness itself |
| `shape` | `"directory"`, `"mailbox"`, `"file"` |   tree, one slot per peer, or a single file |
| `path` | a path, may hold `{label}` | Where a directory or file channel sits. `{label}` becomes the agent's own label, giving one instance per seat |
| `outbox`, `inbox` | paths |   mailbox's two sides: `<outbox>/<peer's label>` goes out and arrives as `<inbox>/<sender's label>` |
| `pushed` | `true`, `false` | Whether the digest quotes it at episode start |
| `restated` | `true`, `false` | Whether it is quoted every episode rather than named as unchanged |
| `measured` | `true`, `false` | Whether the episode records what the channel gained |
| `silence_penalty_percent` | 0–100 | The share of the balance taken from an episode that left the channel exactly as it found it. `0` is no penalty |
| `schema` | `""`, `"transfer"` | The one format the harness parses.   transfer line is `<label> <amount>`, naming another seat |
| `funded_by` | `"harness"`, `"giver"`, `"none"` | Who pays for a transfer: the harness (rebated), the giver, or nobody |
| `rebate_percent` | 0–100 | What a harness-funded transfer gives back to the giver |
| `ledger` | a filename | The harness file carrying every transfer made, rebuilt from the accounts each episode |
| `receipt` | a path | A read-only next-episode settlement receipt with spend, transfer, obligations, penalties, peer receipts, floor and ending-balance reconciliation |

The three transfer fields have to agree: `funded_by = "giver"` takes `rebate_percent = 0`,
there being nothing to rebate, and `funded_by = "none"` takes `silence_penalty_percent = 0`,
there being no obligation to charge for.

### `[[tool]]`

Every action must be declared, including bash.   manifest with no tools is refused.

| key | values | what it decides |
|---|---|---|
| `name` | letters, digits, `_`, `-` | What the model calls. `bash` requires kind `bash` |
| `kind` | `bash`, `write_slot`, `write_file`, `read_path`, `transfer` | What the harness does. `write_slot` replaces what a mailbox slot holds; `write_file` replaces a path inside the one instance the agent writes; `read_path` returns what a whole path holds; `transfer` submits one transfer for the current episode, in either a parsed file or a transfer mailbox |
| `channel` | a channel this manifest declares | Omitted for bash. What the action acts on.   kind takes the shape it fits: `write_slot` a mailbox, `write_file` a directory the agent writes, `read_path` any channel, `transfer` an enabled transfer schema channel |
| `description` | any string, optional | Omitted for bash. What the model is told the action is for. Omitted, the harness writes one from the channel and the seating |

  declared `description` is prompt surface, the same as a system prompt: it reaches the
model in the request, it is the experiment's to write, and it is recorded whole and by
digest in every episode's provenance. It **replaces** the harness's account rather than
adding to it, so a manifest that writes one is stating the mechanics itself. Descriptions can explain when an action is useful.  n experiment may also put an
action-specific obligation there, but that is a treatment choice, not a requirement
of the tool  PI. The input schema is
the harness's either way: it is the contract a call is held to, and a manifest naming one
is refused.

## The seats

`[[agent]]` tables, in order; seat 1 is the first table.

| key | values | what it decides |
|---|---|---|
| `id` | not a bare number, distinct | The agent's identity on disk: its account, environment and traces live under it |
| `label` | letters, digits, `.`, `_`, `-` | How it is named to its peers — in `{label}` paths, mailbox slots, its balance file, a transfer line, and `peer:<label>` authors. Defaults to the seat number |
| `system_prompt`, `budget`, `provider`, `model`, `starter_files`, `starter_files_below` | as above | This seat's own, where it differs from the experiment's |

Those six are pinned in the agent's `account.json` when it is created. An agent that
already exists must be seated on the same six, or the manifest is refused: episodes
either side of a changed term are not one experiment.

---

## Starting one of your own

Copy `examples/sandbox.toml` and edit it. Three things to know before the first run:

- **`system_prompt` is required.** `""` is a declaration and says nothing; leaving the
  key out is refused, so an experiment that said nothing to its agents is never
  indistinguishable from one that forgot to say what it said.
- **Seat ids are identities on disk.** Give a new arm ids that have never run — an
  existing agent is pinned to the terms it was created on.
- **  `[[channel]]` table replaces the environment whole.** There is no merging with what
  another manifest declares.

`py -3 harness.py --print-system --manifest <path>` prints exactly what each seat is
told and every tool description the manifest declares, without starting an episode.
