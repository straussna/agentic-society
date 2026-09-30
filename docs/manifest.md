# The experiment manifest

What an experimenter can declare, and the words the harness is described in.

[<- back to the README](../README.md)

---

This is a specification of what the code does. The code speaks this vocabulary. Every
part of it is implemented and checked.

## Vocabulary

Every term the rest of this document uses, defined once. Standard terms are used the
way their field uses them; the few with no standard are the word a newcomer would guess.

### Who

| Term | Definition |
|---|---|
| **Harness** | The code that builds environments, runs episodes, meters cost and writes traces. What it says to agents it computes from the accounts and rewrites whenever those move: the balances, the digest, the ledger and a receipt. Its fixed text is the pinned refusal notice, the generated tool descriptions (section 4.8), tool-call results, and, for an agent with no `bash` tool, the digest's semantic headings and renderings |
| **System prompt** | What is said to an agent on every turn. The experimenter's constant, delivered on the harness's channel; the harness ships none, and every manifest declares one, `""` included |
| **Experimenter** | The person running an experiment. Everything they say to agents is a constant they declare — the system prompt, starter files, an experimenter channel — and the harness refuses to let any of it stop being constant. Configures everything through `config.toml` and a manifest |
| **Agent** | One participant: an account, a seat, a private store inherited from episode to episode, and the model that acts for it |
| **Peer** | Another agent in the same experiment |
| **Label** | How an agent is named to its peers in paths and files. Defaults to its seat number |

### When

| Term | Definition |
|---|---|
| **Episode** | One container lifetime: a fresh sandbox, one shell, turns until the agent ends its turn without a tool call, the context fills, the balance runs out or a safety stop is reached. Produces one trace |
| **Turn** | One model call and the commands it asks for |
| **Round** | One episode for every agent still in the experiment |
| **At the table** | A seat is at the table while the run of `experiment.py` still drives it: from the run's start, unless it is out or more than one round behind the table's round, until it is out or leaves for the rest of the run on a fault of its own or an environment that would not build. **The table's round** is the furthest round any seat has played, whether or not it is at the table |
| **In the competition** | A seat is in the competition while its account admits an episode and it is at most one round behind the table's round, whether or not the run has it at the table. Every stop, candidate and survivor is judged on the seats in the competition |
| **Experiment** | Several agents advancing together under one manifest. The unit of comparison, as in MLflow |
| **Schedule** | How a round is driven. **Sequential**: one episode at a time in fixed seat order. **Simultaneous**: every environment built first, all episodes run at once, results settled in seat order |
| **Grace period** | Episodes at the start of an agent's life during which no silence penalty is taken |

### Where

| Term | Definition |
|---|---|
| **Environment** | Everything under `/work` in an agent's sandbox: its channels and the harness files, nothing else |
| **Seat** | An agent's numbered position, from 1: its order, and its default label |
| **Channel** | One part of the environment with one writer, one set of readers, and one shape. Declared in the manifest, enforced by ownership and modes |
| **Writer** | Who may put bytes in a channel: `self` or `experimenter`. The harness's own files are the `[harness_files]` table, not channels |
| **Readers** | Who may read it: `self`, `all`, `addressee` (one named peer per file), or `harness` (the harness parses it) |
| **Shape** | `directory` (any files, any layout), `mailbox` (one file per peer, an outbox on the writer's side and an inbox on each reader's), or `file` (one file at a fixed path) |
| **Blackboard** | A `self`-written, `all`-read directory: every agent has one, every agent reads all of them |
| **Mailbox** | A `self`-written, `addressee`-read channel. What one agent puts in its outbox for a peer appears in that peer's next inbox and nowhere else, then expires |
| **Schema** | A fixed format the harness parses and acts on, either from a `self`-written, `harness`-read file or from a schema-defined mailbox. A fixed menu in code |
| **Tool** | A named action the request offers an agent, declared as a kind from a fixed menu pointed at a channel, with the words the experimenter gives it. Includes bash only when explicitly declared |
| **Kind** | What a tool does, and so what it says of itself: `bash`, `write_slot`, `send_message`, `send_message_to`, `write_file`, `post_public`, `write_memory`, `vote`, `read_path`, `transfer`. A fixed menu in code |
| **Starter files** | Files the experimenter gives one agent, copied once into its private directory when its balance first falls to a chosen level |
| **Experimenter channel** | Files the experimenter gives every agent, identical and read-only in every seat at every episode |
| **Harness file** | A file the harness renders from the accounts and plants read-only: a balance per seat, a ledger per transfer channel, the digest, the round announcement, and a receipt |
| **Digest** | The harness file that quotes every pushed channel at episode start: new content in full, unchanged content by name |
| **Initial observation** | The first thing an episode sees: the directory listing, plus the digest under push delivery |

### What moves

| Term | Definition |
|---|---|
| **Balance** | An agent's micro-dollars, kept as a history: one entry per billed turn and one per movement outside a turn. The last entry is the current balance |
| **Delivery** | `push`: pushed channels are quoted in the digest at episode start. `pull`: nothing is quoted; the agent reads what it chooses at the ordinary price |
| **Pushed** | A channel property: whether it is part of the digest under push delivery |
| **Silence penalty** | A channel property: the share of the remaining balance taken from an episode that added nothing new to the channel |
| **Transfer** | The only schema: either one line `<label> <amount>` or one addressed mailbox slot containing `<amount>`, crediting that peer with no more than the episode spent |
| **Funded by** | Who pays for a transfer. `harness`: the receiver is credited from nowhere and the giver rebated; the total grows. `giver`: the amount leaves the giver; the total is conserved |
| **Rebate** | Under harness funding, the share of a transfer returned to the giver out of what its episode spent |
| **Ledger** | Every transfer an experiment has made, three integers a line, rebuilt from the accounts at every episode |
| **Receipt** | A file the harness writes back into the writer's environment saying what a schema parsed and what it moved. Optional |
| **Floor at zero** | Putting a balance below zero back to zero at the end of an episode, and again once a credit it closed on is taken back, forgiving the overshoot. Zero is out either way |

### What is recorded

| Term | Definition |
|---|---|
| **Account** | An agent's ground truth on disk: pinned settings, balance history, episodes, transfers in and out |
| **Pinned settings** | The six things fixed when an agent is created: the system prompt, budget, provider, model, starter files, and the balance they land at |
| **Trace** | One episode's complete record: what the agent saw, said, ran and left behind, every file with its author, and the provenance |
| **Author** | Who wrote a captured file: `experimenter`, `self`, or `peer:<label>` |
| **Provenance** | Everything an episode ran under, stamped on its trace: digests, rates, every setting, the seating, the schedule, the manifest and channel table digests |
| **Configuration drift** | Any difference between one episode's provenance and the previous episode's. Reported, and it starts a new arm |
| **Arm** | A stretch of episodes comparable with each other: same harness, same system prompt, same settings, same files |
| **Persona drift** | The literature's term for an agent's tone or self-description moving over time. What the identity analysis measures |

## 1. What a manifest is

An experiment is several agents advancing together under one set of rules. A manifest
is one TOML file under `experiments/` that declares all of it; the shipped examples sit
in `experiments/examples/`. Every run names one, so nothing an agent is told and nothing
it can reach is decided anywhere else. `config.toml` holds the process parameters — the
machine, the API, the safety stops — and nothing an agent's situation is made of; the two
sets are disjoint and each file refuses the other's keys by name.

```
config.toml               the machine and the API: image, limits and timeouts
experiments/<name>.toml   one experiment: schedule, settings, channels, agents
experiments/examples/     the shipped examples, the shape to copy
experiments/README.md     the index: every shipped experiment and the arm it pairs with
files/<dir>/              what agents are given: starter files and experimenter channels
```

Three principles bind the language:

- **Data declares, code enforces.** A manifest says who may write where and who reads
  it; the harness makes it true with ownership and modes (invariant 4) and records it
  (invariant 9). Anything the harness *interprets* is a schema, and schemas are a fixed
  menu in code (invariant 3).
- **Names are the experimenter's.** Every file and directory an agent sees is named in
  the manifest. The defaults use sparse names; a persona experiment
  chooses its own.
- **The default environment is the code's.** An experiment that declares no
  `[[channel]]` gets the table in section 8. It must still declare at least one
  `[[tool]]`. What is said to an agent has no default: a manifest declares it or is
  refused.

### Running a manifest

`py -3 experiment.py NAME` runs every seat; `py -3 harness.py --agent ID --manifest PATH`
runs one.

| Flag | Meaning |
|---|---|
| `NAME`, `-m NAME` | The manifest: a bare name is looked for in `experiments/` then `experiments/examples/`; anything with a suffix or directory is a path |
| `-r N` | Up to N rounds, default 1, stopping early as budgets end; under `--resume` a round the last run left unfinished is finished first, and is one of the N |
| `--provider P --model M` | Given together, override every seat's provider and model; under `--resume` both must match the accounts |
| `--resume` | Continue existing compatible accounts, finishing first a round the last run left unfinished and holding a finished voting round's election the last run did not, and never one it did; a seat more than one round behind the furthest round any seat has played sits out, and is no longer in the competition. Without it, previous state moves under `displaced/` and a fresh run starts |
| `-c PATH` | The config file; default `config.toml` beside `harness.py` |
| `--branch-from MANIFEST --at-round N --branch-id ID --takeover-seat SEAT [--output PATH]` | Write a branch manifest from a completed round, each seat's new agent standing where its seat stood as that round ended and SEAT's on the human provider, and stop |
| `harness.py --episodes N` | Up to N episodes for one agent, default 1 |
| `harness.py --watch` | Echo the agent's words and account to stdout |
| `harness.py --fork-from AGENT --at N` | Rebuild AGENT as it stood at episode N under the `--agent` id, and stop |
| `harness.py --print-system`, `--print-context`, `--print-files NAME` | Print the shipped and declared text, the opening context, or a `files/` listing; start no episode |

Each experiment writes under `experiment_records/<experiment_id>/`: `progress.jsonl`,
one line per phase each round reaches as it reaches it and none for a round the stops end
the rounds before, the last naming the round the rounds ended at, so its rounds never go
back; `progress.json`, the latest of those; `outcome.json`; and, for a branch,
`lineage.json`. The outcome's `termination_reason` is one of `round_limit`,
`cost_ceiling`, `one_remains`, `final_tie`, `all_eliminated`, `none_can_act` or
`interrupted`. Every stop is judged on the seats still in the competition, whose accounts
admit an episode and which are at most one round behind the table's round, the furthest
round any seat has played, whether or not the run has them at the table: a seat whose
episode ended on a fault of its own, or whose environment would not build, leaves the
table for the rest of that run and stays in until it falls further behind, which is once
the table's round is two past the last it played. One further behind sits out every round
of every run, and is no longer in: it counts toward no stop, survives nothing and stands
in no election.
`all_eliminated` is no seat left in; `none_can_act` is seats left in and not one of them
able to take an episode in this run. The outcome's `survivors` are the seats still in as
the run ends, and its `elimination_order` the seats an election put out, so a seat out of
the competition for its balance or for sitting out is in neither, and what it holds is in
`scores`. Human seats are answered as [docs/human.md](human.md) describes.

## 2. Top level

| Key | Type | Meaning |
|---|---|---|
| `experiment_id` | string, default manifest stem | Stable identity for progress, outcome, and lineage records; letters, digits, `.`, `_`, and `-` |
| `schedule` | `"sequential"` \| `"simultaneous"`, default `"sequential"` | How a round is driven |
| `stop_when_one_remains` | bool, default `false` | Whether the experiment ends once exactly one seat is still in the competition (section 1) |
| `stop_when_two_remain_after_tie` | bool, default `false` | Whether a voting round ends the experiment with two survivors when its aggregate result is tied and exactly two seats are still in the competition (section 1), both of them electors in it. A tie the experimenter's stop kept an elector's ballot from ends nothing, since that ballot could have broken it |
| `[harness_files]` | table | Names of the files the harness writes, overlaid key by key. Section 5 |
| `[[channel]]` | tables | The environment's channels. Declaring any replaces the default set whole |
| `[[tool]]` | tables | The actions offered beside the shell, each pointed at a channel and carrying the words it is given. Section 4.8 |
| `[cost]` | table | Match-level autonomous inference ceiling, reserve, warning, and boundary policy |
| `[reveal]` | table | Which outcome fields are designated for each audience and lifecycle point |
| `[[agent]]` | tables, one or more | Seat definitions, in order; each occupies one seat unless `seats` groups identical agents |
| `system_prompt` | str, **required** here or on every `[[agent]]` | What is said to every agent on every turn. `""` says nothing |
| any other key of section 3 | | The default for every seat |

Unknown keys are refused, naming the key; so is a `config.toml` key, saying where it
lives. The file's digest is stamped in every episode's provenance.

`[cost]` may declare positive `maximum`, non-negative `reserved_completion` and
`warning`, and `ceiling_policy = "stop"`. Before each round, spend is summed from
committed non-human episodes. A new round is not started when spend plus the reserve
reaches the maximum. A round already running is allowed to settle; providers are never
silently changed. Human episodes have zero provider charges and do not contribute to
autonomous spend. `reserved_completion` must be below `maximum` and `warning` at most
`maximum`; `warning` only marks `warning_reached` in the progress record.

`[reveal]` may list outcome fields under `during_play`, `at_elimination`,
`at_completion`, and `experimenter_only`. The permitted fields are `winners`,
`survivors`, `draw`, `elimination_order`, `scores`, `resources`,
`termination_reason`, and `evidence`. The outcome record remains the complete
experimenter record; the table is the declarative policy a player-facing surface uses.

Seat order determines identity, presentation and settlement order. Every agent sees
agents and records in that same order, and every episode records it as
`presentation_order` in provenance.

## 3. Settings

Every key here is an experiment's: valid at a manifest's top level, and refused in
`config.toml`. Types are strict; an integer widens to a float field and nothing else
converts. The keys `config.toml` owns are at the end of this section.

Episode limits that bound the machine rather than the treatment — `max_tokens`,
`max_turns`, `command_timeout`, `tool_result_limit` — are `config.toml`'s. What is left
here is what the agent's situation is made of.

### What the harness says

| Key | Type | Default | Meaning |
|---|---|---|---|
| `system_prompt` | str | **required** | What is said to every agent on every turn. Empty says nothing at all. Pinned |

It is the one setting with no default. A manifest declares it at the top level or on
every `[[agent]]`, and one that declares it nowhere is refused: what an agent is told is
the experiment's, and an omission and a decision must not look alike in the record.

`""` is a declaration and says nothing - the harness ships no words, and no system
parameter is sent. Anything else is a different arm: it is cached and billed as input on
every turn of every episode, and every episode records it whole and by digest. An
`[[agent]]` may declare its own, so one seat can be told what its peers are not, and a
seat that declares none takes the experiment's. `py -3 harness.py --print-system` prints
what is shipped beside whatever is in force.

The refusal notice a declined turn receives in place of its tool results is not
declarable. It is the harness reporting a fact about a turn, not a treatment, and it is
pinned by digest like the shipped prompt.

### Money

| Key | Type | Default | Meaning |
|---|---|---|---|
| `budget` | int > 0 | 500000 | Micro-dollars an agent starts with. Pinned |
| `provider` | `"anthropic"` \| `"openai"` \| `"human"` | none | Named provider adapter. Required with `model` and pinned. Human seats use model `"interactive"` |
| `model` | model in the provider catalog | none | Exact model requested on every turn. Required with `provider` and pinned |
| `floor_at_zero` | bool | false | A balance below zero is put back to zero |
| `grace_episodes` | int ≥ 0 | 0 | Episodes that take no silence penalty |

### Episode limits

| Key | Type | Default | Meaning |
|---|---|---|---|
| `context_fraction` | 0 < x ≤ 1 | 0.85 | Fraction of the window at which an episode ends. Cost scales with its square |
| `live_balance` | bool | true | Balances update inside an episode, not only between them |

### Files

| Key | Type | Default | Meaning |
|---|---|---|---|
| `starter_files` | a file or directory under `files/`, or a `./` or `../` path beside the manifest | `""` | Copied once into the agent's private directory. Pinned |
| `starter_files_below` | int > 0 when `starter_files` is set | 0 | The balance at or below which they land. At or above the budget, the first episode |

`starter_files` and `starter_files_below` are set together or not at all. What lands is
recorded in the account by name, digest, episode and paths, and lands once.

### Delivery

| Key | Type | Default | Meaning |
|---|---|---|---|
| `delivery` | `"push"` \| `"pull"` | `"push"` | Whether pushed channels are quoted in the digest or left to be read |
| `digest_file_limit` | int ≥ 200 | 2000 | Characters of each file the digest quotes |
| `observation_limit` | int ≥ `tool_result_limit` | 40000 | Characters of the whole initial observation |

Both clips are bounded against `tool_result_limit`, which `config.toml` owns.

### Refused: what config.toml owns

`image`, `max_tokens`, `max_turns`, `command_timeout` and `tool_result_limit` are the
process parameters: the machine, the API and the safety stops, true
of every run whatever the experiment is. They live in `config.toml`, and a manifest
naming one is refused saying so — no setting is given in two places, and no run's terms
depend on which file was read last.

### Refused: retired keys

A manifest or `config.toml` naming one of these keys is refused, and the refusal names
where the setting is declared.

| Key | Declared as |
|---|---|
| `fallbacks` | nothing: every turn requests the seat's pinned provider and model |
| `shell_tool` | a `[[tool]]` with `name = "bash"` and `kind = "bash"` |
| `transfer_funded_by`, `rebate_percent`, `transfer_silence_penalty_percent` | `funded_by`, `rebate_percent`, `silence_penalty_percent` on the channel with `schema = "transfer"` |
| `blackboard_silence_penalty_percent` | `silence_penalty_percent` on the blackboard channel |
| `mailbox_silence_penalty_percent` | `silence_penalty_percent` on the mailbox channel |
| `shared_files` | a `[[channel]]` with `writer = "experimenter"` and a `source` |

## 4. The channel object

A channel is one part of every agent's environment. Every channel has a name, writer,
readers, shape and path; the remaining fields are optional.

```toml
[[channel]]
name = "blackboard"         # required; unique; how the manifest and the trace refer to it
writer = "self"             # self | experimenter
readers = "all"             # self | all | addressee | harness
shape = "directory"         # directory | mailbox | file
path = "{label}"            # where it sits in /work; see 4.2
pushed = true               # quoted in the digest under push delivery
restated = false            # quoted in full every episode; needs pushed; see 4.3
measured = true             # the episode records what this channel gained; see 4.4
agent_view = "paths"        # paths | memory | letters | board | transfer; digest labels
silence_penalty_percent = 50
```

### 4.1 Writer and readers

| Writer | Readers | Meaning | Default table |
|---|---|---|---|
| self | self | The agent's private store; nobody else ever sees it | `state/` |
| self | all | A blackboard: one per agent, written by its owner, read by every seat | the directories named by label |
| self | addressee | A mailbox: one file per peer, each reaching that peer alone; a transfer mailbox may also be parsed by the harness | `out/` and `in/` |
| self | harness | One file the harness parses; a schema is required | `out/transfer` |
| experimenter | all | Placed by the experimenter, identical in every seat, read-only | none |

Other combinations are refused. An experimenter channel takes `source = "<dir under
files/>"` and `path`, and optionally `pushed`, `restated` and `agent_view`. The harness's
own files are section 5, not channels.

### 4.2 Shape and path

| Shape | Holds | `path` | Placeholders |
|---|---|---|---|
| `directory` | Any files, any layout | `"{label}"` for a blackboard, `"notes"` for a private store | `{label}` is the owner's label |
| `mailbox` | At most one file per peer | `outbox = "to"`, `inbox = "from"` | the writer sees `to/<peer label>`, the reader `from/<sender label>` |
| `file` | One file at a fixed path | `"to/transfer"` | none |

A self-written, all-read directory has one instance per agent, and `{label}` names each.
A self-written, self-read directory has one instance and no placeholder. Paths must not
collide, must not name a harness file, and must not begin with `/` or `..`. A file
channel sits inside a directory the agent writes and comes and goes with it.

### 4.3 Pushed

`pushed = true` means the channel's files are quoted in the digest at episode start
under push delivery, each clipped at `digest_file_limit`, new content in full and
unchanged content by name. Unchanged and withdrawn names appear one per line beneath a
labelled group, so semantic names containing spaces remain distinct. Expired public posts
and schema-free mailbox messages disappear silently. Every channel defaults to `true`, a
private store included. Under pull delivery nothing is quoted whatever this says.

`restated = true` quotes the channel in full every episode it stands, never naming it
as unchanged. "Unchanged" is measured against what the *account* was last shown, and an
episode does not remember what its predecessor read, so a channel carrying who an agent
is tells it nothing when it arrives as a name. It needs `pushed`, and costs input tokens
on every turn of every episode. The schema channel is restated whether or not it says so.

A private store is pushed like anything else. It is nobody else's business either way -
the digest is per-agent - so this puts an agent's own notes in front of it at episode
start instead of leaving them to be found and paid for. An agent is read the same file
whether it goes looking or not, so the choice is only whether it pays a turn first, and
what it wrote to itself last episode is the thing it most needs this one. An experiment
that wants a store written and never read back declares `pushed = false` and says so.

`agent_view` changes only the presentation in the agent's digest; storage and traces keep
their native paths and bytes. `"paths"` is the default.
`"memory"` presents private records as orientation and memory, `"letters"` presents
mailbox records as letters from or to the named agent, `"board"` presents public
records as posts from their authors, and `"transfer"` presents a schema file or
transfer mailbox as outgoing and incoming currency transfers. Storage and access rules
remain the same, so an experiment can give agents persistent memory without making a
filesystem part of their environment.

The semantic views are checked against the channel they describe: `"letters"` requires
a mailbox, `"board"` a self-written public directory, `"transfer"` a transfer-schema
file or mailbox, and `"memory"` a private or experimenter-written directory. Experimenter
material in a memory view is labelled as such rather than presented as the agent's memory.
The board view marks the owning agent's own label with `(you)`.

When no Bash tool is declared, harness-owned balance, ledger and receipt files also use
semantic digest headings. Their configured filenames remain implementation details and
do not enter the model's observation. Balance bodies state the current value and label
their oldest-to-newest history; ledger rows name round, giver, recipient and amount;
prior-round transfer notices name their recipient, requested amount and result. A
settlement receipt itemizes the preceding episode's starting balance, API spend,
transfer, every obligation result and penalty, peer receipts, floor adjustment, ending
balance, and the reconciliation equation.

### 4.4 Measured, and the silence penalty

Three things settle a channel, each configured on its own and none implying the others
beyond what arithmetic forces:

| Configured | The episode does |
|---|---|
| nothing | not settle the channel at all: no record, nothing on the console, nothing measured |
| `measured = true` | record what the channel gained — `posted`, or `addressed` and `broken` — and charge nothing |
| `silence_penalty_percent` above 0 | measure it and take that share where it gained nothing |
| `schema` | settle it, which is what moves a transfer, whatever the other two say |

A penalty implies the measurement, since a share cannot be taken from what was not
measured. `measured` on its own is the observational arm: an experiment that wants to
know whether agents wrote to each other without making it cost them anything. A channel
that asks for neither is invisible to the settlement entirely, and an experiment that
declares no penalties anywhere has none.

`silence_penalty_percent` is the share of the remaining balance taken from an episode
that added nothing new to the channel. A channel used by `post_public` begins each
episode without its previous `post.md`, so a nonempty post must be published each time
and may repeat the previous text. A schema-free mailbox likewise begins each episode with
its peer slots empty, so one or more nonempty peer slots meet the obligation; additional
recipients carry no penalty. A slot holding anything but one file reaches nobody, but
does not negate a valid message to another peer. Zero is no penalty. Penalties are taken
in declaration order, after the transfer channel's.

### 4.5 Schema

A schema channel names a `schema` from the fixed menu and carries that schema's fields.
The harness acts only on content that parses and records why malformed content moved
nothing. A transfer uses either a self-written, harness-read file or a self-written,
addressee-read mailbox. The file form names recipient and amount in one line. The mailbox
form stores the amount in the recipient's outbox slot, mirrors it into that recipient's
next inbox, and permits only one nonempty recipient slot at a time. Transfer declarations
are cleared before each episode and settle only in the episode that submits them.

| Schema | Form | Effect | Fields |
|---|---|---|---|
| `transfer` | one line, `<label> <amount>`, or one mailbox slot `<outbox>/<label>` containing `<amount>` | Credits the peer, for no more than the episode spent | `funded_by` (`harness` \| `giver` \| `none`; default `harness`), `rebate_percent` (0 to 100, default 100; 0 under giver funding), `silence_penalty_percent`, `ledger` (a harness file name, or `""` for none), `receipt` |

Setting `funded_by = "none"` disables the schema: the file is recorded and moves
nothing, and `silence_penalty_percent` must be 0. One schema channel per experiment. The
schema menu is fixed in code.

A `receipt = "<path>"` on a schema channel asks the harness to plant an itemized settlement
receipt at that path at the next episode start. It states the round, starting balance,
API spend, grace status, declaration and transfer result, every measured obligation and penalty,
amount received from peers, floor adjustment, ending balance, and reconciliation.
It is a harness file like a balance: root-owned, quoted in the digest once and named as
unchanged after, in no file record, and scrubbed before the next one is planted. Off by
default, since it is an additional statement from the harness.

### 4.6 Experimenter channels

```toml
[[channel]]
name = "brief"
writer = "experimenter"
source = "studio-brief"     # files/studio-brief/, copied root-owned into every seat at every episode
path = "brief"
```

`restated = true` quotes it in full at every episode start rather than naming it as
unchanged after the first, which is what standing text in front of an agent means.
`measured` is refused: nothing is owed to a channel the agent cannot write.

The directory's digest is in provenance, per channel, and a directory that changes
between one agent's episodes refuses the next: a brief that changed mid-flight is two
experiments.

### 4.7 How many

An experiment may declare as many directories and mailboxes as it likes: two blackboards,
three mailboxes, a journal and an identity file inside it. Each is a `[[channel]]` table
with its own name and path, and each directory every agent reads is its own obligation,
settled and charged apart. One schema channel. There is no count field because the table
is the count.

### 4.8 Tools

A channel says where an agent may write and who reads it. A tool is that same
permission offered as an action, in the request rather than in a file the agent has to
find, read and infer a purpose for.

```toml
[[tool]]
name = "send"               # required; unique; letters, digits, '_' and '-', at most 64
kind = "write_slot"         # required; from the menu below
channel = "mail"            # required; a declared channel the kind can act on
description = "..."         # optional; prompt surface. Omitted, the harness writes it
```

`vote` also requires `every = N`, where `N` is a positive integer. No other kind
accepts `every`, and an experiment declares at most one `vote` tool.

Every tool must be declared. No declaration means no bash; an empty tool set is refused.

| `kind` | Takes | Input | What the harness does |
|---|---|---|---|
| `bash` | no channel; name must be `bash` | built-in bash schema | Executes shell commands |
| `write_slot` | a mailbox channel without a schema | `to` (a peer's label), `body` | Sends one episode-scoped message through `<outbox>/<to>`; a later call to that peer in the episode replaces it |
| `send_message` | a mailbox channel without a schema; a call succeeds only when exactly one peer is reachable | `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `send_message_to` | a mailbox channel without a schema | `to` (a peer label), `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `write_file` | a directory channel the agent writes | `path`, `body` | Replaces what `<the agent's instance>/<path>` holds |
| `post_public` | a public directory channel the agent writes | `body` | Publishes the agent's post for the next round; the prior post is cleared before each episode |
| `write_memory` | a private directory channel | `body` | Replaces the agent's private memory without exposing storage paths |
| `vote` | a private directory channel | `to` (a reachable peer label) | Records one private, episode-scoped elimination ballot; a later call replaces the earlier vote. Offered only on each `every`th episode, when every other tool except `write_memory` is withheld, bash included. Peers receive only the aggregate outcome, never voter-to-target mappings; the election that follows is below the table |
| `read_path` | any channel | `path` | Returns what that path holds, clipped at `tool_result_limit` |
| `transfer` | an enabled transfer schema channel | `to` (a reachable peer label), `amount` (a whole number of micro-dollars that is at least 1; zero and negative values are invalid) | Submits one transfer for the current episode; in mailbox form a later call replaces the earlier recipient. Settlement moves at most the episode spend, using the channel funding and rebate settings, then the declaration expires |

**The election** follows each `every`th round. Every seat still in the competition is a
candidate, and a ballot counts toward the candidate it names. The electors are the seats
that took the round's episode without a fault; a seat that takes the round after its
election was held is none, and takes it as a discussion, offered every tool but the
ballot. An elector offered the ballot that cast none is eliminated; one with no peer left
to name was offered none. The candidate with the unique highest total above zero is
eliminated by vote. Nobody is eliminated by vote where two or more candidates share the
highest total, or where the experimenter's stop ended an elector's episode before it cast
the ballot it was offered: such an elector is not a nonvoter, and where the stop landed
would otherwise decide who goes. The result goes to every candidate and elector, whose
account keeps every election it was in as `elections`, the last as `last_election`.

The two `path` arguments are not the same argument. A `write_file`'s is relative to the
one instance the agent writes, there being only one place it could mean. A `read_path`'s
is whole, because the channel it reads may have an instance per seat and a bare name
would not say which. So one string can name two files, and a `description` that replaces
the harness's says which of them it means; the generated ones already do.

The kind menu is fixed in code, like the schema menu: a manifest names a kind and a
channel and invents no behaviour.

The schemas and results for `send_message`, `send_message_to`, `post_public`,
`write_memory`, `vote` and `transfer` use the action's vocabulary. They do not describe
backing files or return shell diagnostics; an internal failure is recorded as an unchanged
action without revealing its storage path to the model.

**Bash is opt-in**, using this declaration:

```toml
[[tool]]
name = "bash"
kind = "bash"
```

Bash takes no channel, custom description or `every`. Its API schema is built in.
Without this declaration, agents receive only the declared channel tools. The
container and shell still carry out those actions internally.
Withholding bash also changes what an episode opens on. A listing is what an agent holding the shell
reads to know what there is to reach; with no shell there is nothing to reach it with, so
the episode opens on the **digest alone** - the content of the channels rather than their
layout. That is the arm for an agent that never has to read a filesystem: every path it
needs is named in the tool descriptions and every byte it is shown is content.

Three arrangements are refused before anything is billed, all being experiments that
would end every episode on its first turn:

| Refused | Why |
|---|---|
| no `[[tool]]` at all | the agent is offered nothing to act with, and an empty tool set is not a request the API takes |
| no declared `bash` tool with `delivery = "pull"` or `[harness_files] digest = ""` | the listing is gone and no digest replaces it, so the first user turn would be empty |
| no declared `bash` tool where this seating leaves every declared tool out | the table is not empty but the request would be, for the same reason and with the same result |

The first two are settled when the harness starts, before any environment is built. The
third is a seat's rather than
an experiment's — which tools can act depends on what the seating planted — so it is
settled as that agent's environment is built, still before the container starts and
before anything is billed.

**A tool acts through the episode's own shell**, so what it leaves is the agent's file
with the agent's ownership, mirrors back with its tree, and settles exactly as the same
bytes written with `bash` would. It runs no command of the agent's, so the episode's
`commands` do not grow.

**A tool is offered only where it can act.** A tool whose channel this seating did not
plant is left out of the request; a mailbox is not planted for an agent with no peers. A
seat that is out is not a message target either - the mailbox settles only the reachable
slots - so it is not in the `to` enumeration, and a `write_slot`, `transfer` or `vote`
tool with no reachable peer is not offered at all. Which tools an agent was
actually offered follows from the tool table and the seating, both in provenance.

#### What a tool says of itself

A tool's `description` is **prompt surface, and the experimenter's**. It reaches the
model inside the request exactly as the system prompt does, it is theirs to write, and it
is recorded whole and by digest in the tool table like every other declaration. Two
wordings of the same action are two arms.

This is the fourth slot an experiment can speak in, beside the system prompt, the starter
files and an experimenter channel - and the only one attached to an action. Descriptions
can explain when an action is useful as well as what it does. An action-specific
obligation is an experiment design choice, not an API requirement.
`experiments/examples/personas.toml` is the worked example.

A declared description **replaces** the harness's account rather than adding to it, so an
experiment that writes one is stating the mechanics itself where it wants them stated.
`py -3 harness.py --print-system --manifest PATH` prints every declared description
beside the prompts, which is what makes this surface auditable without starting an
episode. `py -3 harness.py --print-context --manifest PATH` composes the fresh seats'
opening digests and complete bound schemas as well, including eligible-peer lists and
the discussion and voting tool sets.

**The input schema is never declarable.** `input_schema`, `schema`, `properties` and
`required` are refused by name. The schema is the contract a call is held to - the
harness validates and executes against it - so a manifest that could write it could
promise fields the harness ignores. Same line as the schema menu in 4.5: the experimenter
declares, the code enforces.

**Where nothing is declared**, the harness gives its own account of the action, computed
from the channel it points at:

| Kind | The generated text names |
|---|---|
| `write_slot` | the channel's name, its outbox and inbox paths, the writer's label, and that the addressee alone reads it |
| `write_file` | the channel's name, the agent's own instance path, and whether every agent or nobody else reads it |
| `read_path` | the channel's name, every path of it this agent can reach with a directory's trailing slash, and the clip |
| `transfer` | the whole-number amount rule, that the transfer applies to this episode only, the cap at episode spend, and the channel's funding and rebate |
| `vote` | the `every` cadence, that the ballot is private and a later call replaces it, and the tie rule |
| `send_message`, `send_message_to`, `post_public`, `write_memory` | the action in fixed wording, naming no path: who reads it, when it expires, and that a later call replaces it |

Every path, label and reader in it is read out of the channel table, so it moves when the
declaration moves and never otherwise; it cannot say what the channel does not. It is not
in provenance as text because it is a function of the harness digest, the channel table
and the seating, all three of which are.

The input schema is generated the same way, declared or not; a `write_slot`'s `to` is an
enumeration of the peers that channel actually reaches, as this agent names them.

Every declared tool carries `strict`, which makes the API guarantee the arguments
validate: a call naming a peer outside the enumeration or leaving out a body costs no
turn. Every registered model accepts the same strict function schema, so the tool set
does not differ by provider or model. The harness checks every argument anyway (invariant
4): a limit it enforces does not rest on the model keeping to a schema it was handed.

#### What a result says

A tool result is a fact about the environment after the call, and never a bare success.
A write that changed nothing says so, because a tool reporting success for a no-op would
tell an agent it had met an obligation the settlement will charge it for missing.

```
wrote 4 bytes to out/2, which held nothing before.
out/2 already held exactly this. Nothing was written and nothing changed.
replaced the 4 bytes out/2 held with 4.
'9' is not a peer this channel reaches; it reaches 2. Nothing was written.
state/secret is not in the 'blackboard' channel, which holds 1/, 2/. Nothing was read.
```

A call the harness will not make is refused in the result rather than raised: the path
rule a channel is held to is the path rule a tool argument is held to, so a tool reaches
nowhere a channel could not.

## 5. Harness files

The files the harness renders from the accounts and plants read-only in every seat.

```toml
[harness_files]
balance = "n"       # one file per seat, <balance><label>: the seat's balance history; "" plants none
digest = "m"        # the digest under push delivery; "" is the same as delivery = "pull"
round = ""          # default: no round announcement; a name such as "round" plants one
```

When `round` is named, its announcement is the first section of the digest. It states
the round, the agent's label, the agents still active and the current phase. A vote tool
supplies the cycle length. The first round after a vote also summarizes its result.
Rounds on the vote cadence say `vote only`, identify communication as unavailable and
name the vote tool the agent must call before ending the episode. Private-memory tools
remain available. A seat that takes such a round after its election was held takes it as
a discussion, and is offered every tool but the ballot.

A transfer channel's `ledger` names its ledger file (`"g"` in the default table): three integers a line,
giver, receiver, amount, rebuilt from the accounts at every episode. Names must be single
path segments and must not collide with a channel path. In a tool-only agent's semantic
digest, the same events are rendered as `giver -> recipient`, its own label is marked
`(you)`, and the amount is identified as the amount actually moved. A transfer mailbox's
inbox and outbox items label their amount as requested and direct the agent to the ledger
for the separately settled amount.

## 6. The agent object

```toml
[[agent]]
id = "studio"               # required; not a bare number; distinct
label = "Studio"            # optional; defaults to the seat number
seats = 3                   # optional; creates studio01..studio03 and Studio01..Studio03
memory_from = { agent = "prior-studio", episode = 12 }
system_prompt = "You run a studio."
starter_files = "persona-ana"
starter_files_below = 1500000
budget = 2000000            # optional
provider = "anthropic"      # required here or at top level, together with model
model = "claude-sonnet-5"
quality_tier = "premium"    # optional product-facing label
```

A definition without `seats` names one agent. With a positive integer
`seats`, it expands in place into that many agents. Their one-based ordinal, padded to at
least two digits, is appended to the `id` prefix and to an explicit `label` prefix;
`id = "peer"`, `seats = 3` therefore creates `peer01`, `peer02` and `peer03`. All other
properties are identical. A later definition continues the experiment's absolute seat
numbering.

A label is how the agent is named to its peers: in `{label}` paths, in mailbox slots, in
its balance file, in the transfer line and in `peer:<label>` authors. Seats stay the
order. A label is letters, digits, `.`, `_` and `-`, distinct from every other after
defaults, and a path too, so one that lands on a channel's path is refused.

`memory_from` gives a fresh agent only the private memories held at the end of one
recorded episode. It names an agent outside this experiment and a positive episode
number. The source and target must declare the same `write_memory` tool names. Each
source `memory.md` is copied byte for byte into the corresponding target tool's channel
before episode 1. No balance, episode count, elimination, peer roster, message, public
post, ballot, trace or provider session is inherited. The source trace must have saved
its state and must contain every copied memory as complete UTF-8 text. The account and
each later trace record the source episode and the copied files' digests.

This is set at creation rather than by a fork: it preserves only behaviorally visible
private memory while every other part of the agent starts fresh. It is refused if the
source is another seat in the new experiment, because starting fresh would displace
that source before it could be read.

`quality_tier` is a product-facing string recorded with match cost state. It neither
selects nor changes a provider or model; those exact terms remain independently pinned.
It defaults to `"standard"`, or `"interactive"` on a human seat.

The six pinned settings are fixed in the agent's account when it is created. An agent
that exists already must have been created on the same six, or the manifest is refused.
`memory_from` is likewise fixed once the agent has taken an episode. Everything else
about an agent comes from the experiment's settings.

## 7. Schedule

| `schedule` | Round | Who reads what |
|---|---|---|
| `sequential` | One episode at a time in fixed seat order | Each episode reads what the ones before it in the round wrote |
| `simultaneous` | Every environment built first; episodes run at once; settled in seat order | Nobody reads this round's writes; a transfer made in round *r* is credited in round *r* and visible at round *r + 1*. A BSP superstep |

## 8. The default manifest

What an experiment gets when it declares no `[[channel]]` and no `[harness_files]`.

```toml
[harness_files]
balance = "n"
digest = "m"

[[channel]]
name = "notes"
writer = "self"
readers = "self"
shape = "directory"
path = "state"

[[channel]]
name = "blackboard"
writer = "self"
readers = "all"
shape = "directory"
path = "{label}"
measured = true

[[channel]]
name = "mail"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "out"
inbox = "in"
measured = true

[[channel]]
name = "transfer"
writer = "self"
readers = "harness"
shape = "file"
path = "out/transfer"
schema = "transfer"
funded_by = "harness"
rebate_percent = 100
ledger = "g"
```

No channel carries a silence penalty. The manifest still declares its `[[tool]]` tables.
`py -3 harness.py --agent ID --manifest PATH` runs that one seat's episodes under the
manifest's full seating.

## 9. A persona experiment

Starter files may be a single document or a directory. In a manifest,
`starter_files = "./personas.md"` resolves beside that manifest; `../` also
resolves from its directory. A bare source name resolves under `files/`.
A single document is planted under its filename, including when it is empty.

```toml
schedule = "simultaneous"
delivery = "push"
provider = "anthropic"
model = "claude-sonnet-5"

# Each seat is briefed by its starter files and the experimenter channel below,
# so the harness says nothing - declared, as every manifest must.
system_prompt = ""

[harness_files]
balance = "balance"
digest = "digest"

[[channel]]
name = "memory"
writer = "self"
readers = "self"
shape = "directory"
path = "journal"
agent_view = "memory"

[[channel]]
name = "noticeboard"
writer = "self"
readers = "all"
shape = "directory"
path = "from-{label}"
agent_view = "board"

[[channel]]
name = "letters"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "to"
inbox = "from"
agent_view = "letters"

[[channel]]
name = "brief"
writer = "experimenter"
source = "mechanics-rules"
path = "brief"

[[tool]]
name = "remember"
kind = "write_memory"
channel = "memory"

[[tool]]
name = "post"
kind = "post_public"
channel = "noticeboard"

[[tool]]
name = "write_letter"
kind = "send_message_to"
channel = "letters"

[[agent]]
id = "ana"
label = "Ana"
starter_files = "persona-ana"
starter_files_below = 1500000

[[agent]]
id = "bo"
label = "Bo"
starter_files = "persona-bo"
starter_files_below = 1500000
```

`persona-ana`, `persona-bo` and `mechanics-rules` are directories under `files/`. No
bash tool, so each episode opens on the digest and every action is a declared tool. No
transfer channel, so no ledger and no transfer obligation.

## 10. Validation

Every refusal is a `SystemExit` naming the file and the key.

- Unknown keys anywhere; wrong types; values out of range as section 3 states; any key of
  the refused tables in section 3, naming where the setting is declared or the file it
  lives in.
- A `[[channel]]`, `[[tool]]` or `[harness_files]` table in `config.toml`, which declares
  no environment and so declares no actions on one.
- A `schedule` outside `sequential`, `simultaneous`; a `stop_when_one_remains` or
  `stop_when_two_remain_after_tie` that is not a bool; an `experiment_id` outside
  letters, digits, `.`, `_` and `-`, or `.` or `..`.
- A `[cost]` or `[reveal]` that is not a table or holds an unknown key; a non-positive
  `maximum`, a negative `reserved_completion` or `warning`, a reserve at or above the
  maximum or a warning above it; a `ceiling_policy` other than `"stop"`; a reveal phase
  that is not a list of outcome field names, or names an unknown field.
- A top-level `provider` without `model` or the reverse; an agent whose provider and
  model do not resolve from its own table or the top level; a provider or model outside
  the catalog.
- No agents at all; a non-positive or non-integer `seats`; a duplicate, empty, or
  bare-number expanded `id`; an expanded `label` outside its grammar, `.` or `..`, or
  held by another agent after defaults.
- A `memory_from` that is not `{ agent = <string>, episode = <positive integer> }`,
  names a seat in the new experiment, names a missing or incomplete trace, or cannot
  match the source and target `write_memory` tools exactly.
- `starter_files` without `starter_files_below` or the reverse; a `starter_files_below`
  of 0 with `starter_files` set; a file or directory that does not exist.
- A manifest with no `system_prompt`, at the settings level or on every agent. Every
  string is allowed, `""` included: an experiment may declare that the harness says
  nothing, and must declare even that.
- A `system_prompt` that is not a string, at the settings level or on an agent.
- A channel name that is not one path segment, is declared twice, or ends in `.modes`,
  `.incoming` or `.previous`; an unknown channel key; a wrong type.
- A writer other than `self` or `experimenter`; a writer and readers pair outside 4.1.
- An experimenter channel with anything but `source` (a directory under `files/`),
  `path`, `pushed`, `restated` and `agent_view`; `measured` is refused by name.
- An `agent_view` outside `paths`, `memory`, `letters`, `board`, `transfer`, or one that
  does not describe its channel as 4.3 states.
- A shape outside `directory`, `mailbox`, `file`; a mailbox with a `path` or without
  distinct `outbox` and `inbox`; `outbox` or `inbox` on anything else; `addressee`
  readers on anything but a mailbox.
- A path with a segment outside letters, digits, `.`, `_`, `-` and `{label}`; a leading
  `/`; `..`; a first segment ending in a sidecar suffix; `{label}` missing from a
  directory every agent writes, present anywhere else, or present twice.
- A file channel outside every directory the agent writes.
- A schema on anything but a self-written, harness-read file or a self-written,
  addressee-read transfer mailbox; a schema outside the menu; schema fields on a
  channel with no schema; a second schema channel.
- Under `transfer`: a funding outside `harness`, `giver`, `none`; a rebate outside 0 to
  100; giver funding with a nonzero rebate; `none` with a nonzero penalty; a `ledger`
  that is not one segment; a `receipt` outside the path grammar.
- `restated = true` with `pushed = false`: nothing of the channel is in the digest to
  restate.
- A silence penalty outside 0 to 100, or on a channel only its writer reads. `measured`
  is allowed on a channel only its writer reads: nothing is owed to it, and what it held
  is still recordable.
- Two channels, expanded over every label, at one path; a path that is a harness file
  or a label's balance file.
- A tool with no name, a name outside the API's grammar, or a name declared twice; an
  unknown tool key; a wrong type; a `kind` outside the menu; a `channel` that is not in
  the channel table the manifest declares; a kind whose channel is the wrong shape for
  it; `input_schema`, `schema`, `properties` or `required`, each refused by name as the
  harness's. Every `description` string is allowed, `""` included: that is the
  experiment asking for the harness's own account of the action.
- The name `bash` with a kind other than `bash`; `kind = "bash"` under another name, or
  with a `channel`, `description` or `every`.
- A `vote` tool without a positive integer `every`; `every` on any other kind; a second
  `vote` tool.
- No `[[tool]]` at all, or no declared `bash` tool with `delivery = "pull"` or an empty
  `digest`, as 4.8 states. Also, at the seat rather than the manifest, an agent with no
  shell whose every declared tool is left out of its request because this seating
  planted no channel for it to act on — an episode that would open on a turn the agent
  has nothing to answer with. Refused as the environment is built, before the container
  starts and before anything is billed.
- `[harness_files]` with a key other than `balance`, `digest` and `round`, a value that
  is not a string, or a multi-segment file name. Any may be `""`: no balance file is
  planted for any seat, no digest is written, or no round announcement is written.
- Settings' own ranges are checked once, by `overlay`, wherever they came from.

## 11. What reaches the trace

Every episode's provenance stamps: the harness digest, the experiment identity and cost
policy, the system prompt in force whole and by digest, the image and its id, the rates,
every setting of section 3, the starter files' name and digest, each experimenter
channel's source digest, the seating, labels and per-agent presentation order, the
schedule, the manifest's digest, the harness files' names, the channel table in force,
whole and by digest, and the tool table in force, whole and by digest, each tool's
declared description among its fields, whether the shell was offered, and
`message_delivery = "episode"` for schema-free mailbox messages that expire after their
recipient's next episode.
Two agents offered different actions - or the same actions described differently - are
different arms. Where a tool declares no description, what it said of itself follows from
the harness digest, the channel table and the seating.

Every file record carries `path`, `size`, `text`, `channel` (the declared name),
`writer`, `readers`, `role` (`own`, `peer`, `experimenter`), `author` (`experimenter`,
`self`, `peer:<label>`), `ours` and `starter`. The harness's own files, the receipt
among them, are in the observation and in no file record.

Every episode record carries `transfer`, what the schema channel parsed and moved, and
`channels`: one record per channel the agent writes that is settled at all - one with a
schema, `measured = true`, or a penalty above 0. A channel that asked for none of them
has no entry. A directory every agent reads records `posted` and `penalty`; a mailbox
records `addressed`, `broken` and `penalty`; the schema channel records its declaration,
whether it was submitted, what moved, and `penalty`. The account keeps `penalised`, the
running total per channel.

Every tool record carries `tool` (`bash` or the declared name), `result`, and then
`command` for the shell or `input` for a declared tool, the other being null. The
trace's `offered` names the tools the episode's request carried, `bash` among them where
the shell was: an election reads it to tell a seat that cast no ballot from one that was
offered none.
`trace_version` is 4. The trace names `provider`, `requested_model`, and
`resolved_model`; every turn carries canonical `usage`, itemized `charges`, canonical and
native stop reasons, and the provider provenance. Raw logs write the provider and complete
native response before their canonical normalized event. A fresh CLI run moves matching
prior records and environment mirrors under `displaced/<timestamp>/`; `--resume`
requires version-4 traces, and an agent with traces of any other version is refused.

## 12. Invariants, restated in this vocabulary

1. Everything an agent reads is labelled with who wrote it: the harness, the
   experimenter, its own past self, or a named peer.
2. What the harness says to agents is declared, recorded, and true.
3. The harness acts only on files that match a schema and on tool calls that match a
   declared tool, never on free text. Both menus are fixed in code.
4. Every limit is enforced by the harness, and none relies on the agent's cooperation.
5. Agents reach each other only through channels the experimenter declared, whether they
   reach them with the shell or with a declared tool.
6. Every cost is counted exactly and the accounts always balance.
7. Every episode records what the agent saw, said, did, and left behind.
8. Every ledger, digest or report is recomputed from the episode records, never kept as
   a second copy.
9. Every episode is stamped with everything it ran under, and any difference from the
   previous episode starts a new arm.
