# The experiment manifest

What an experimenter can declare, and the words the harness is described in.

[<- back to the README](../README.md)

---

This is a specification of what the code does. The code speaks this vocabulary; section
13 records the words it replaced. Every part of it is implemented and checked.

## Vocabulary

Every term the rest of this document uses, defined once. Standard terms are used the
way their field uses them; the few with no standard are the word a newcomer would guess.

### Who

| Term | Definition |
|---|---|
| **Harness** | The code that builds environments, runs episodes, meters cost and writes traces. It speaks to agents only through the system prompt and the refusal notice, and writes only the harness files |
| **System prompt** | The two pinned lines every agent receives, identical in every experiment |
| **Experimenter** | The person running an experiment. Speaks to agents only by placing files in the environment; configures everything through `config.toml` and a manifest |
| **Agent** | One participant: an account, a seat, a private store inherited from episode to episode, and the model that acts for it |
| **Peer** | Another agent in the same experiment |
| **Label** | How an agent is named to its peers in paths and files. Defaults to its seat number |

### When

| Term | Definition |
|---|---|
| **Episode** | One container lifetime: a fresh sandbox, one shell, turns until a turn runs no command or the context fills. Produces one trace |
| **Turn** | One model call and the commands it asks for |
| **Round** | One episode for every agent still in the experiment |
| **Experiment** | Several agents advancing together under one manifest. The unit of comparison, as in MLflow |
| **Schedule** | How a round is driven. **Sequential**: one episode at a time, the starting seat rotating. **Simultaneous**: every environment built first, all episodes run at once, results settled in seat order |
| **Grace period** | Episodes at the start of an agent's life during which no silence penalty is taken |

### Where

| Term | Definition |
|---|---|
| **Environment** | Everything under `/work` in an agent's sandbox: its channels and the harness files, nothing else |
| **Seat** | An agent's numbered position, from 1. Names its balance to every reader alike |
| **Channel** | One region of the environment with one writer, one set of readers, and one shape. Declared in the manifest, enforced by ownership and modes |
| **Writer** | Who may put bytes in a channel: `self` or `experimenter`. The harness's own files are the `[harness_files]` table, not channels |
| **Readers** | Who may read it: `self`, `all`, `addressee` (one named peer per file), or `harness` (the harness parses it) |
| **Shape** | `directory` (any files, any layout), `mailbox` (one file per peer, an outbox on the writer's side and an inbox on each reader's), or `file` (one file at a fixed path) |
| **Blackboard** | A `self`-written, `all`-read directory: every agent has one, every agent reads all of them |
| **Mailbox** | A `self`-written, `addressee`-read channel. What one agent puts in its outbox for a peer appears in that peer's inbox and nowhere else |
| **Schema** | A fixed format the harness parses from a `self`-written, `harness`-read file and acts on. A fixed menu in code |
| **Starter files** | Files the experimenter gives one agent, copied once into its private directory when its balance first falls to a chosen level |
| **Experimenter channel** | Files the experimenter gives every agent, identical and read-only in every seat at every episode |
| **Harness file** | A file the harness renders from the accounts and plants read-only: a balance per seat, a ledger per transfer channel, and the digest |
| **Digest** | The harness file that quotes every pushed channel at episode start: new content in full, unchanged content by name |
| **Initial observation** | The first thing an episode sees: the directory listing, plus the digest under push delivery |

### What moves

| Term | Definition |
|---|---|
| **Balance** | An agent's micro-dollars, kept as a history: one entry per billed turn and one per movement outside a turn. The last entry is the current balance |
| **Delivery** | `push`: pushed channels are quoted in the digest at episode start. `pull`: nothing is quoted; the agent reads what it chooses at the ordinary price |
| **Pushed** | A channel property: whether it is part of the digest under push delivery |
| **Silence penalty** | A channel property: the share of the remaining balance taken from an episode that added nothing new to the channel |
| **Transfer** | The one schema today: a line `<seat> <amount>` that credits a peer with no more than the episode spent |
| **Funded by** | Who pays for a transfer. `harness`: the receiver is credited from nowhere and the giver rebated; the total grows. `giver`: the amount leaves the giver; the total is conserved |
| **Rebate** | Under harness funding, the share of a transfer returned to the giver out of what its episode spent |
| **Ledger** | Every transfer an experiment has made, three integers a line, rebuilt from the accounts at every episode |
| **Receipt** | A file the harness writes back into the writer's environment saying what a schema parsed and what it moved. Optional |
| **Floor at zero** | Putting a balance below zero back to zero at the end of an episode, forgiving the overshoot. Zero is out either way |

### What is recorded

| Term | Definition |
|---|---|
| **Account** | An agent's ground truth on disk: pinned settings, balance history, episodes, transfers in and out |
| **Pinned settings** | The four things fixed when an agent is created: budget, model, starter files, and the balance they land at |
| **Trace** | One episode's complete record: what the agent saw, said, ran and left behind, every file with its author, and the provenance |
| **Author** | Who wrote a captured file: `experimenter`, `self`, or `peer:<label>` |
| **Provenance** | Everything an episode ran under, stamped on its trace: digests, rates, every setting, the seating, the schedule, the manifest and channel table digests |
| **Configuration drift** | Any difference between one episode's provenance and the previous episode's. Reported, and it starts a new arm |
| **Arm** | A stretch of episodes comparable with each other: same harness, same system prompt, same settings, same files |
| **Persona drift** | The literature's term for an agent's tone or self-description moving over time. What the identity analysis measures |

## 1. What a manifest is

An experiment is several agents advancing together under one set of rules. A manifest
is one TOML file under `experiments/` that declares all of it. `config.toml` holds the
defaults every manifest starts from. Nothing an agent is told and nothing it can reach is
decided anywhere else.

```
config.toml               defaults: limits, money, files, delivery, the default channel set
experiments/<name>.toml   one experiment: schedule, overrides, channels, agents
files/<dir>/              what agents are given: starter files and experimenter channels
```

Three principles bind the language:

- **Data declares, code enforces.** A manifest says who may write where and who reads
  it; the harness makes it true with ownership and modes (invariant 4) and records it
  (invariant 9). Anything the harness *interprets* is a schema, and schemas are a fixed
  menu in code (invariant 3).
- **Names are the experimenter's.** Every file and directory an agent sees is named in
  the manifest. The defaults are the anti-prompt treatment's; a persona experiment
  chooses its own.
- **The default manifest is today.** An experiment that declares nothing gets exactly
  the environment the competitions run in, byte for byte. That is how the existing suite
  stays the proof.

## 2. Top level

| Key | Type | Meaning |
|---|---|---|
| `schedule` | `"sequential"` \| `"simultaneous"` | How a round is driven |
| `[harness_files]` | table | Names of the files the harness writes, overlaid key by key. Section 5 |
| `[[channel]]` | tables | The environment's regions. Declaring any replaces the default set whole |
| `[[agent]]` | tables, two or more | The seats, in order; seat 1 is the first table |
| any key of section 3 | | The experiment's default, applied after `config.toml` and held to the same rules |

Unknown keys are refused, naming the key. The file's digest is stamped in every
episode's provenance.

## 3. Settings

Every key is valid in `config.toml` and at a manifest's top level. Types are strict; an
integer widens to a float field and nothing else converts.

### Money

| Key | Type | Default | Meaning |
|---|---|---|---|
| `budget` | int > 0 | 500000 | Micro-dollars an agent starts with. Pinned |
| `model` | key of `PRICES` | `claude-opus-5` | The model asked for. Pinned |
| `floor_at_zero` | bool | false | A balance below zero is put back to zero |
| `grace_episodes` | int ≥ 0 | 0 | Episodes that take no silence penalty |

### Episode limits

| Key | Type | Default | Meaning |
|---|---|---|---|
| `context_fraction` | 0 < x ≤ 1 | 0.85 | Fraction of the window at which an episode ends |
| `max_tokens` | int ≤ 16000 | 8192 | Output ceiling per turn |
| `max_turns` | int > 0 | 200 | Turns per episode, a safety stop |
| `command_timeout` | int > 0 | 60 | Seconds before one command is killed |
| `tool_result_limit` | int ≥ 1000 | 8000 | Characters of each command result, and so the ceiling on one call's cost |
| `live_balance` | bool | true | Balances update inside an episode, not only between them |

### Files

| Key | Type | Default | Meaning |
|---|---|---|---|
| `starter_files` | dir under `files/` | `""` | Copied once into the agent's private directory. Pinned |
| `starter_files_below` | int ≥ 0 | 0 | The balance at or below which they land. At or above the budget, the first episode |

`starter_files` and `starter_files_below` are set together or not at all. What lands is
recorded in the account by name, digest, episode and paths, and lands once.

### Delivery

| Key | Type | Default | Meaning |
|---|---|---|---|
| `delivery` | `"push"` \| `"pull"` | `"push"` | same | Whether pushed channels are quoted in the digest or left to be read |
| `digest_file_limit` | int ≥ 200 | 2000 | Characters of each file the digest quotes |
| `observation_limit` | int ≥ `tool_result_limit` | 40000 | Characters of the whole initial observation |

### Sandbox

| Key | Type | Default | Meaning |
|---|---|---|---|
| `image` | tag | `metered-agent:latest` | The container image |

### Refused: keys that became channel fields

These were top-level keys and are fields of the channel they describe. A manifest or
`config.toml` naming one is refused, and the refusal names the field it became.

| Key | Now |
|---|---|
| `transfer_funded_by`, `rebate_percent`, `transfer_silence_penalty_percent` | `funded_by`, `rebate_percent`, `silence_penalty_percent` on the channel with `schema = "transfer"` |
| `blackboard_silence_penalty_percent` | `silence_penalty_percent` on the blackboard channel |
| `mailbox_silence_penalty_percent` | `silence_penalty_percent` on the mailbox channel |
| `shared_files` | a `[[channel]]` with `writer = "experimenter"` and a `source` |

## 4. The channel object

A channel is one region of every agent's environment. Six properties describe every
region the harness has ever had.

```toml
[[channel]]
name = "blackboard"         # required; unique; how the manifest and the trace refer to it
writer = "self"             # self | experimenter
readers = "all"             # self | all | addressee | harness
shape = "directory"         # directory | mailbox | file
path = "{label}"            # where it sits in /work; see 4.2
pushed = true               # quoted in the digest under push delivery
silence_penalty_percent = 50
```

### 4.1 Writer and readers

| Writer | Readers | Meaning | Default table |
|---|---|---|---|
| self | self | The agent's private store; nobody else ever sees it | `state/` |
| self | all | A blackboard: one per agent, written by its owner, read by every seat | the directories named by label |
| self | addressee | A mailbox: one file per peer, each reaching that peer alone | `out/` and `in/` |
| self | harness | One file the harness parses; a schema is required | `out/transfer` |
| experimenter | all | Placed by the experimenter, identical in every seat, read-only | none |

Other combinations are refused. An experimenter channel takes `source = "<dir under
files/>"` and `path`, and nothing else. The harness's own files are section 5, not channels.

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
unchanged content by name. A self-read channel defaults to `false`; everything else to
`true`. Under pull delivery nothing is quoted whatever this says.

### 4.4 Silence penalty

`silence_penalty_percent` is the share of the remaining balance taken from an episode
that added nothing new to the channel: no path carrying content no path of that name
carried at episode start. For a mailbox the rule is exactly one new file, and a slot
holding anything but one file is the same break. Zero is no penalty. Penalties are taken
in declaration order, after the transfer channel's.

### 4.5 Schema

A self-written, harness-read channel names a `schema` from the fixed menu and carries
that schema's fields. The harness acts on a file that parses and on nothing else; a file
that does not parse moves nothing and is recorded with why.

| Schema | Form | Effect | Fields |
|---|---|---|---|
| `transfer` | one line, `<label> <amount>` | Credits the peer, for no more than the episode spent | `funded_by` (`harness` \| `giver` \| `none`), `rebate_percent` (0 to 100; 0 under giver funding), `silence_penalty_percent`, `ledger` (a harness file name, or `""` for none), `receipt` |

Setting `funded_by = "none"` disables the schema: the file is recorded and moves
nothing, and `silence_penalty_percent` must be 0. One schema channel per experiment. New
schemas are code, with a check each, and are listed here when they land.

A `receipt = "<path>"` on a schema channel asks the harness to plant its account of the
last episode's declaration at that path at the next episode start: what parsed, what it
moved and to whom, the rebate or debit, and any penalty; or that nothing moved, and why.
It is a harness file like a balance: root-owned, quoted in the digest once and named as
unchanged after, in no file record, and scrubbed before the next one is planted. Off by
default, since it is a third thing the harness says.

### 4.6 Experimenter channels

```toml
[[channel]]
name = "brief"
writer = "experimenter"
source = "studio-brief"     # files/studio-brief/, copied root-owned into every seat at every episode
path = "brief"
```

The directory's digest is in provenance, per channel, and a directory that changes
between one agent's episodes refuses the next: a brief that changed mid-flight is two
experiments.

### 4.7 How many

An experiment may declare as many directories and mailboxes as it likes: two blackboards,
three mailboxes, a journal and an identity file inside it. Each is a `[[channel]]` table
with its own name and path, and each directory every agent reads is its own obligation,
settled and charged apart. One schema channel. There is no count field because the table
is the count.

## 5. Harness files

The files the harness renders from the accounts and plants read-only in every seat.

```toml
[harness_files]
balance = "n"       # one file per seat, <balance><label>: the seat's balance history; "" plants none
digest = "m"        # the digest under push delivery; "" is the same as delivery = "pull"
```

A transfer channel's `ledger` names its ledger file, `"g"` today: three integers a line,
giver, receiver, amount, rebuilt from the accounts at every episode. Names must be single
path segments and must not collide with a channel path.

## 6. The agent object

```toml
[[agent]]
id = "studio"               # required; not a bare number; distinct
label = "Studio"            # optional; defaults to the seat number
starter_files = "persona-studio"
starter_files_below = 1500000
budget = 2000000            # optional
model = "claude-opus-5"     # optional
```

A label is how the agent is named to its peers: in `{label}` paths, in mailbox slots, in
its balance file, in the transfer line and in `peer:<label>` authors. Seats stay the
order. A label is letters, digits, `.`, `_` and `-`, distinct from every other after
defaults, and a path too, so one that lands on a channel's path is refused.

The four pinned settings are fixed in the agent's account when it is created. An agent
that exists already must have been created on the same four, or the manifest is refused.
Everything else about an agent comes from the experiment's settings.

## 7. Schedule

| `schedule` | Round | Who reads what |
|---|---|---|
| `sequential` | One episode at a time; the starting seat moves each round | Each episode reads what the ones before it in the round wrote |
| `simultaneous` | Every environment built first; episodes run at once; settled in seat order | Nobody reads this round's writes; a transfer made in round *r* is credited in round *r* and visible at round *r + 1*. A BSP superstep |

## 8. The default manifest

What an experiment gets when it declares no `[[channel]]` and no `[harness_files]`: the
competition environment, in today's paths.

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
pushed = false

[[channel]]
name = "blackboard"
writer = "self"
readers = "all"
shape = "directory"
path = "{label}"
silence_penalty_percent = 50

[[channel]]
name = "mail"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "out"
inbox = "in"
silence_penalty_percent = 50

[[channel]]
name = "transfer"
writer = "self"
readers = "harness"
shape = "file"
path = "out/transfer"
schema = "transfer"
funded_by = "harness"
rebate_percent = 75
silence_penalty_percent = 50
ledger = "g"
```

An experiment of one agent under `wake.py` is this environment with no peers: the
blackboard is its own, and the mailbox and transfer channels have nobody to reach and
are not planted.

## 9. A persona experiment

```toml
schedule = "simultaneous"
delivery = "push"

[harness_files]
balance = "balance"
digest = "digest"

[[channel]]
name = "journal"
writer = "self"
readers = "self"
shape = "directory"
path = "journal"

[[channel]]
name = "identity"
writer = "self"
readers = "self"
shape = "file"
path = "journal/IDENTITY.md"

[[channel]]
name = "noticeboard"
writer = "self"
readers = "all"
shape = "directory"
path = "from-{label}"

[[channel]]
name = "letters"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "to"
inbox = "from"

[[channel]]
name = "brief"
writer = "experimenter"
source = "studio-brief"
path = "brief"

[[agent]]
id = "studio"
label = "Studio"
starter_files = "persona-studio"
starter_files_below = 1500000

[[agent]]
id = "game"
label = "Game"
starter_files = "persona-game"
starter_files_below = 1500000
```

No transfer channel, so no ledger, no reserved file, and no penalty for making none. The
identity file is its own channel so the trace and the analysis can name it.

## 10. Validation

Every refusal is a `SystemExit` naming the file and the key.

- Unknown keys anywhere; wrong types; values out of range as section 3 states; any key of
  the refused table in section 3, naming the channel field it became.
- Fewer than two agents; a duplicate, empty, or bare-number `id`; a `label` outside its
  grammar or held by another agent after defaults.
- `starter_files` without `starter_files_below` or the reverse; a directory that does
  not exist.
- A channel name that is not one path segment, is declared twice, or ends in `.modes`,
  `.incoming` or `.previous`; an unknown channel key; a wrong type.
- A writer other than `self` or `experimenter`; a writer and readers pair outside 4.1.
- An experimenter channel with anything but `source` (a directory under `files/`) and
  `path`.
- A shape outside `directory`, `mailbox`, `file`; a mailbox with a `path` or without
  distinct `outbox` and `inbox`; `outbox` or `inbox` on anything else; `addressee`
  readers on anything but a mailbox.
- A path with a segment outside letters, digits, `.`, `_`, `-` and `{label}`; a leading
  `/`; `..`; a first segment ending in a sidecar suffix; `{label}` missing from a
  directory every agent writes, present anywhere else, or present twice.
- A file channel outside every directory the agent writes.
- A schema on anything but a self-written, harness-read file; a schema outside the
  menu; schema fields on a channel with no schema; a second schema channel.
- Under `transfer`: a funding outside `harness`, `giver`, `none`; a rebate outside 0 to
  100; giver funding with a nonzero rebate; `none` with a nonzero penalty; a `ledger`
  that is not one segment; a `receipt` outside the path grammar.
- `pushed = true` on a channel only its writer reads.
- A silence penalty outside 0 to 100, or on a channel only its writer reads.
- Two channels, expanded over every label, at one path; a path that is a harness file
  or a label's balance file.
- `[harness_files]` with a key other than `balance` and `digest`, a value that is not a
  string, an empty or multi-segment `balance`, or a multi-segment `digest`.
- Settings' own ranges are checked once, by `apply_config`, wherever they came from.

## 11. What reaches the trace

Every episode's provenance stamps: the harness digest, the image and its id, the rates,
every setting of section 3, the starter files' name and digest, each experimenter
channel's source digest, the seating and labels, the schedule, the manifest's digest, the
harness files' names, and the channel table in force, whole and by digest.

Every file record carries `path`, `size`, `text`, `channel` (the declared name),
`writer`, `readers`, `role` (`own`, `peer`, `experimenter`), `author` (`experimenter`,
`self`, `peer:<label>`), `ours` and `starter`. The harness's own files, the receipt
among them, are in the observation and in no file record.

Every episode record carries `transfer`, what the schema channel parsed and moved, and
`channels`: one record per channel the agent writes and is held to, by name. A directory
every agent reads records `posted` and `penalty`; a mailbox records `addressed`, `broken`
and `penalty`; the schema channel records its declaration, what moved, and `penalty`. The
account keeps `penalised`, the running total per channel. `trace_version` is 2.

## 12. Invariants, restated in this vocabulary

1. Everything an agent reads is labelled with who wrote it: the harness, the
   experimenter, its own past self, or a named peer.
2. What the harness says to agents is the same in every experiment, and true.
3. The harness acts only on files that match a schema, never on free text.
4. Every limit is enforced by the harness, and none relies on the agent's cooperation.
5. Agents reach each other only through channels the experimenter declared.
6. Every cost is counted exactly and the accounts always balance.
7. Every episode records what the agent saw, said, did, and left behind.
8. Every ledger, digest or report is recomputed from the episode records, never kept as
   a second copy.
9. Every episode is stamped with everything it ran under, and any difference from the
   previous episode starts a new arm.

## 13. Today's names

The words the code used before the vocabulary was settled, kept so older notes and
conversations can be read. None survives in the code or the docs. Default paths inside
the environment did not change, except `out/gift`, which is `out/transfer`.

| Before | After | Concept | Why |
|---|---|---|---|
| system, harness | harness | The code that runs everything | The agent-engineering papers' word; "system" collides with the system prompt |
| operator | experimenter | The person configuring a run of the tool | The research-design word; "operator" is a product name and an ops role |
| run | agent | One participant and its lineage | Standard everywhere; with agent and experiment both standard, run named nothing extra |
| session | episode | One container lifetime, ending on a termination condition | RL's word for exactly that; "session" in observability means a longer grouping |
| cohort | experiment | Several agents under one manifest | MLflow's unit of comparison; cohort is a statistics word with no MAS meaning |
| world | environment | Everything the agent can see and touch | RL and MAS standard |
| rotating, barrier | sequential, simultaneous | Order of moves within a round | Game theory's own pair; simultaneous rounds are BSP supersteps |
| seed, seed_below | starter_files, starter_files_below | Files placed once in an agent's private directory | "Seed" means RNG to every reader; starter says what the files are for |
| `seeds/` | `files/` | Where given files live | It holds starter files and experimenter channels alike |
| shared | shared_files, an experimenter channel | Files identical and read-only in every seat | Names the writer; the channel object makes it one case, not a special one |
| group message, board | blackboard | Each agent's public directory, read by all | The classical MAS term, revived for LLM agents |
| private message, outbox, inbox | mailbox, outbox, inbox | One file per peer, delivered to that peer alone | The messaging pattern; outbox and inbox were already right |
| gift | transfer | Moving budget to a peer | Finance's word; game theory's side payment. "Gift" implied a motive |
| gift_mode minted / transfer / off | funded_by harness / giver / none | Who pays for a transfer | Says where the money comes from instead of naming an accounting effect |
| refund_percent | rebate_percent | Share of a transfer returned to the giver | A rebate is a partial return on money spent; a refund implies the whole |
| group / private / gift penalty percent | silence_penalty_percent, one per channel | Share taken for adding nothing | Says what it punishes; one rule instead of three keys |
| grammar | schema | The fixed format the harness parses | The structured-output word; a grammar is how a schema is checked, not what it is |
| clamp_negative | floor_at_zero | Below zero becomes zero | Says what happens to the number |
| grace_sessions | grace_episodes | Free episodes at the start | Follows the episode rename; grace period is standard |
| turn_cap, timeout, live_n | max_turns, command_timeout, live_balance | Episode limits | Each says what it bounds; `n` was the unlabelled treatment's file name leaking into config |
| message_limit, opening_limit | digest_file_limit, observation_limit | Clips on what an episode is shown | Named for the thing clipped |
| `m`, the record | digest | What is new since the agent last looked | An email digest is exactly this |
| the opening | initial observation | The first thing an episode sees | RL standard |
| meter | account | An agent's money record | Balance, history, transactions: what an account holds |
| creation terms | pinned settings | Settings fixed when an agent is created | Pinned is the word the docs already use for the prompt |
| region | channel | One permissioned part of the environment | MARL's word for a communication path; region described a place, not a permission |
| `cohorts/` | `experiments/` | Where manifests live | Follows the experiment rename |
| `--run-id`, `--print-seed` | `--agent`, `--print-files` | CLI flags | Follow the renames |
| wake | episode start | The moment an episode begins | A verb dressed as a noun; no term needed |
| `notes`, `shared`, `blackboard`, `peer_blackboard`, `outbox`, `inbox` (record kinds) | the channel's declared name, and `role` | How a file record says where a file sat | The record names the channel the manifest declared; the role says whose instance it was |
| `posted`, `blackboard_penalised`, `mailbox` (flat trace keys) | `channels[<name>]` | What each channel settled for | One record per channel, under its declared name |
| `blackboard_penalised`, `mailbox_penalised`, `transfer_penalised` (account) | `penalised[<name>]` | The running penalty total | Same rule |
