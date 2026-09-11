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
| **Harness** | The code that builds environments, runs episodes, meters cost and writes traces. What it says to agents it computes from the accounts and rewrites whenever those move: the balances, the digest, the ledger, a receipt, and the notice a refused turn receives. It utters no constant of its own |
| **System prompt** | What is said to an agent on every turn. The experimenter's constant, delivered on the harness's channel; the harness ships none, and every manifest declares one, `""` included |
| **Experimenter** | The person running an experiment. Everything they say to agents is a constant they declare — the system prompt, starter files, an experimenter channel — and the harness refuses to let any of it stop being constant. Configures everything through `config.toml` and a manifest |
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
| **Schedule** | How a round is driven. **Sequential**: one episode at a time in fixed seat order. **Simultaneous**: every environment built first, all episodes run at once, results settled in seat order |
| **Grace period** | Episodes at the start of an agent's life during which no silence penalty is taken |

### Where

| Term | Definition |
|---|---|
| **Environment** | Everything under `/work` in an agent's sandbox: its channels and the harness files, nothing else |
| **Seat** | An agent's numbered position, from 1. Names its balance to every reader alike |
| **Channel** | One part of the environment with one writer, one set of readers, and one shape. Declared in the manifest, enforced by ownership and modes |
| **Writer** | Who may put bytes in a channel: `self` or `experimenter`. The harness's own files are the `[harness_files]` table, not channels |
| **Readers** | Who may read it: `self`, `all`, `addressee` (one named peer per file), or `harness` (the harness parses it) |
| **Shape** | `directory` (any files, any layout), `mailbox` (one file per peer, an outbox on the writer's side and an inbox on each reader's), or `file` (one file at a fixed path) |
| **Blackboard** | A `self`-written, `all`-read directory: every agent has one, every agent reads all of them |
| **Mailbox** | A `self`-written, `addressee`-read channel. What one agent puts in its outbox for a peer appears in that peer's next inbox and nowhere else, then expires |
| **Schema** | A fixed format the harness parses and acts on, either from a `self`-written, `harness`-read file or from a schema-defined mailbox. A fixed menu in code |
| **Tool** | A named action the request offers an agent, declared as a kind from a fixed menu pointed at a channel, with the words the experimenter gives it. Includes bash only when explicitly declared |
| **Kind** | What a tool does, and so what it says of itself: `bash`, `write_slot`, `send_message`, `send_message_to`, `write_file`, `post_public`, `write_memory`, `read_path`, `transfer`. A fixed menu in code |
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
| **Transfer** | The one schema today: either one line `<label> <amount>` or one addressed mailbox slot containing `<amount>`, crediting that peer with no more than the episode spent |
| **Funded by** | Who pays for a transfer. `harness`: the receiver is credited from nowhere and the giver rebated; the total grows. `giver`: the amount leaves the giver; the total is conserved |
| **Rebate** | Under harness funding, the share of a transfer returned to the giver out of what its episode spent |
| **Ledger** | Every transfer an experiment has made, three integers a line, rebuilt from the accounts at every episode |
| **Receipt** | A file the harness writes back into the writer's environment saying what a schema parsed and what it moved. Optional |
| **Floor at zero** | Putting a balance below zero back to zero at the end of an episode, forgiving the overshoot. Zero is out either way |

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
- **The default environment is today's.** An experiment that declares no `[[channel]]`
  gets exactly the environment the competitions run in, byte for byte. That is how the
  existing suite stays the proof. What is said to an agent has no default: a manifest
  declares it or is refused.

## 2. Top level

| Key | Type | Meaning |
|---|---|---|
| `schedule` | `"sequential"` \| `"simultaneous"` | How a round is driven |
| `stop_when_one_remains` | bool, default `false` | Whether the experiment ends once exactly one funded seat remains |
| `[harness_files]` | table | Names of the files the harness writes, overlaid key by key. Section 5 |
| `[[channel]]` | tables | The environment's channels. Declaring any replaces the default set whole |
| `[[tool]]` | tables | The actions offered beside the shell, each pointed at a channel and carrying the words it is given. Section 4.8 |
| `[[agent]]` | tables, one or more | Seat definitions, in order; each occupies one seat unless `seats` groups identical agents |
| `system_prompt` | str, **required** here or on every `[[agent]]` | What is said to every agent on every turn. `""` says nothing |
| any other key of section 3 | | The default for every seat |

Unknown keys are refused, naming the key; so is a `config.toml` key, saying where it
lives. The file's digest is stamped in every episode's provenance.

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
| `provider` | `"anthropic"` \| `"openai"` | none | Named first-party adapter. Required with `model` and pinned |
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
| `starter_files` | dir under `files/` | `""` | Copied once into the agent's private directory. Pinned |
| `starter_files_below` | int ≥ 0 | 0 | The balance at or below which they land. At or above the budget, the first episode |

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

A channel is one part of every agent's environment. Six properties describe every
channel the harness has ever had.

```toml
[[channel]]
name = "blackboard"         # required; unique; how the manifest and the trace refer to it
writer = "self"             # self | experimenter
readers = "all"             # self | all | addressee | harness
shape = "directory"         # directory | mailbox | file
path = "{label}"            # where it sits in /work; see 4.2
pushed = true               # quoted in the digest under push delivery
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
unchanged content by name. Unchanged and withdrawn names appear one per line beneath a
labelled group, so semantic names containing spaces remain distinct. Expired public posts
and schema-free mailbox messages disappear silently. Every channel defaults to
`true`, a private store included.
Under pull delivery nothing is quoted whatever this says.

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
filesystem part of their world.

The semantic views are checked against the channel they describe: `"letters"` requires
a mailbox without a schema, `"board"` a self-written public directory, `"transfer"` an
enabled transfer schema file or mailbox, and `"memory"` a private or experimenter-written directory. Experimenter
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
its peer slots empty, so one or more nonempty peer slots meet the
obligation; additional recipients carry no penalty. A slot holding anything but one file
reaches nobody, but does not negate a valid message to another peer. Zero is no
penalty. Penalties are taken
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
| `transfer` | one line, `<label> <amount>`, or one mailbox slot `<outbox>/<label>` containing `<amount>` | Credits the peer, for no more than the episode spent | `funded_by` (`harness` \| `giver` \| `none`), `rebate_percent` (0 to 100; 0 under giver funding), `silence_penalty_percent`, `ledger` (a harness file name, or `""` for none), `receipt` |

Setting `funded_by = "none"` disables the schema: the file is recorded and moves
nothing, and `silence_penalty_percent` must be 0. One schema channel per experiment. New
schemas are code, with a check each, and are listed here when they land.

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
accepts `every`.

Every tool must be declared. No declaration means no bash; an empty tool set is refused.

| `kind` | Takes | Input | What the harness does |
|---|---|---|---|
| `bash` | no channel; name must be `bash` | built-in bash schema | Executes shell commands |
| `write_slot` | a mailbox channel without a schema | `to` (a peer's label), `body` | Sends one episode-scoped message through `<outbox>/<to>`; a later call to that peer in the episode replaces it |
| `send_message` | a mailbox channel without a schema and with one reachable peer | `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `send_message_to` | a mailbox channel without a schema | `to` (a peer label), `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `write_file` | a directory channel the agent writes | `path`, `body` | Replaces what `<the agent's instance>/<path>` holds |
| `post_public` | a public directory channel the agent writes | `body` | Publishes the agent's post for the next round; the prior post is cleared before each episode |
| `write_memory` | a private directory channel | `body` | Replaces the agent's private memory without exposing storage paths |
| `vote` | a private directory channel | `to` (a reachable peer label) | Records one private, episode-scoped elimination ballot. Peers receive only the aggregate outcome, never voter-to-target mappings. It is offered only on each `every`th episode, when the shell and peer-communication tools are withheld but private memory remains available; a later call replaces the earlier vote. After that round, nonvoters and the unique highest vote-getter are eliminated; if two or more agents share the highest total, nobody is eliminated by vote |
| `read_path` | any channel | `path` | Returns what that path holds, clipped at `tool_result_limit` |
| `transfer` | an enabled transfer schema channel | `to` (a reachable peer label), `amount` (a whole number of micro-dollars that is at least 1; zero and negative values are invalid) | Submits one transfer for the current episode; in mailbox form a later call replaces the earlier recipient. Settlement moves at most the episode spend, using the channel funding and rebate settings, then the declaration expires |

The two `path` arguments are not the same argument. A `write_file`'s is relative to the
one instance the agent writes, there being only one place it could mean. A `read_path`'s
is whole, because the channel it reads may have an instance per seat and a bare name
would not say which. So one string can name two files, and a `description` that replaces
the harness's says which of them it means; the generated ones already do.

The menu is code, like the schema menu: a manifest names a kind and a channel and
invents no behaviour. A new kind is a change to `harness.py` with a check of its own,
and is listed here when it lands.

The schemas and results for `send_message`, `send_message_to`, `post_public`,
`write_memory`, `vote` and `transfer` use the action's vocabulary. They do not describe backing
files or return shell diagnostics; an internal failure is recorded as an unchanged
action without revealing its storage path to the model.

**Bash is opt-in**, using this declaration:

```toml
[[tool]]
name = "bash"
kind = "bash"
```

Bash takes no channel or custom description. Its API schema is built in.
Without this declaration, agents receive only the declared channel tools. The
container and shell still carry out those actions internally.
It also changes what an episode opens on. A listing is what an agent holding the shell
reads to know what there is to reach; with no shell there is nothing to reach it with, so
the episode opens on the **digest alone** - the content of the channels rather than their
layout. That is the arm for an agent that never has to read a filesystem: every path it
needs is named in the tool descriptions and every byte it is shown is content.

Two arrangements are refused before anything is billed, both being experiments that would
end every episode on its first turn:

| Refused | Why |
|---|---|
| no declared `bash` tool with no `[[tool]]` | the agent is offered nothing to act with, and an empty tool set is not a request the API takes |
| no declared `bash` tool with `delivery = "pull"` or `digest = ""` | the listing is gone and no digest replaces it, so the first user turn would be empty |
| no declared `bash` tool where this seating leaves every declared tool out | the table is not empty but the request would be, for the same reason and with the same result |

The first two are settled when the manifest is read. The third is a seat's rather than
an experiment's — which tools can act depends on what the seating planted — so it is
settled as that agent's environment is built, still before the container starts and
before anything is billed.

**A tool acts through the episode's own shell**, so what it leaves is the agent's file
with the agent's ownership, mirrors back with its tree, and settles exactly as the same
bytes written with `bash` would. It runs no command of the agent's, so the episode's
`commands` do not grow.

**A tool is offered only where it can act.** A mailbox is not planted for an agent with
no peers, so a `write_slot` on it is left out of that agent's request rather than
offered and refused at every call. A seat that is out is not a message target either -
the mailbox settles only the reachable slots - so it is not in the `to` enumeration, and
a mailbox whose every peer is out is not offered at all. Which tools an agent was
actually offered follows from the tool table and the seating, both in provenance.

#### What a tool says of itself

A tool's `description` is **prompt surface, and the experimenter's**. It reaches the
model inside the request exactly as the system prompt does, it is theirs to write, and it
is recorded whole and by digest in the tool table like every other declaration. Two
wordings of the same action are two arms.

This is the fourth slot an experiment can speak in, beside the system prompt, the starter
files and an experimenter channel - and the only one attached to an action. Descriptions can explain when an action is useful as well as what it does.
An action-specific obligation is an experiment design choice, not an API requirement. `experiments/examples/personas.toml` is the worked example.

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
state/secret is not in the 'blackboard' channel, which holds 1, 2. Nothing was read.
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
round = "round"     # round number, cycle position and discussion/voting phase; "" plants none
```

When `round` is named, its announcement is the first section of the digest. It states
the round, the agent's label, the agents still active and the current phase. A vote tool
supplies the cycle length. The first round after a vote also summarizes its result.
Rounds on the vote cadence say `vote only`, identify communication as unavailable and
name the vote tool the agent must call before ending the episode. Private-memory tools
remain available.

A transfer channel's `ledger` names its ledger file, `"g"` today: three integers a line,
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
seats = 3                    # optional; creates studio01..studio03 and Studio01..Studio03
system_prompt = "You run a studio."
starter_files = "persona-studio"
starter_files_below = 1500000
budget = 2000000            # optional
provider = "anthropic"      # required here or at top level, together with model
model = "claude-sonnet-5"
```

A definition without `seats` names one agent exactly as before. With a positive integer
`seats`, it expands in place into that many agents. Their one-based ordinal, padded to at
least two digits, is appended to the `id` prefix and to an explicit `label` prefix;
`id = "peer"`, `seats = 3` therefore creates `peer01`, `peer02` and `peer03`. All other
properties are identical. A later definition continues the experiment's absolute seat
numbering.

A label is how the agent is named to its peers: in `{label}` paths, in mailbox slots, in
its balance file, in the transfer line and in `peer:<label>` authors. Seats stay the
order. A label is letters, digits, `.`, `_` and `-`, distinct from every other after
defaults, and a path too, so one that lands on a channel's path is refused.

The six pinned settings are fixed in the agent's account when it is created. An agent
that exists already must have been created on the same six, or the manifest is refused.
Everything else about an agent comes from the experiment's settings.

## 7. Schedule

| `schedule` | Round | Who reads what |
|---|---|---|
| `sequential` | One episode at a time in fixed seat order | Each episode reads what the ones before it in the round wrote |
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

An agent under `harness.py` alone is this environment with no peers: the
blackboard is its own, and the mailbox and transfer channels have nobody to reach and
are not planted.

## 9. A persona experiment

Starter files may be a single document or a directory. In a manifest,
`starter_files = "./personas.md"` resolves beside that manifest; `../` also
resolves from its directory. A bare source name resolves under `files/`.
A single document is planted under its filename, including when it is empty.

```toml
schedule = "simultaneous"
delivery = "push"

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
  the refused tables in section 3, naming the channel field it became or the file it
  lives in.
- A `[[channel]]`, `[[tool]]` or `[harness_files]` table in `config.toml`, which declares
  no environment and so declares no actions on one.
- No agents at all; a non-positive or non-integer `seats`; a duplicate, empty, or
  bare-number expanded `id`; an expanded `label` outside its grammar or held by another
  agent after defaults.
- `starter_files` without `starter_files_below` or the reverse; a directory that does
  not exist.
- A manifest with no `system_prompt`, at the settings level or on every agent. Every
  string is allowed, `""` included: an experiment may declare that the harness says
  nothing, and must declare even that.
- A `system_prompt` that is not a string, at the settings level or on an agent.
- A channel name that is not one path segment, is declared twice, or ends in `.modes`,
  `.incoming` or `.previous`; an unknown channel key; a wrong type.
- A writer other than `self` or `experimenter`; a writer and readers pair outside 4.1.
- An experimenter channel with anything but `source` (a directory under `files/`),
  `path`, `pushed` and `restated`; `measured` is refused by name.
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
- A tool with no name, a name outside the API's grammar, a name declared twice, or the
  name `bash`; an unknown tool key; a wrong type; a `kind` outside the menu; a `channel`
  that is not in the channel table the manifest declares; a kind whose channel is the
  wrong shape for it; `input_schema`, `schema`, `properties` or `required`, each refused
  by name as the harness's. Every `description` string is allowed, `""` included: that
  is the experiment asking for the harness's own account of the action.
- no declared `bash` tool with no `[[tool]]`, or with `delivery = "pull"` or an empty
  `digest`, as 4.8 states. Also, at the seat rather than the manifest, an agent with no
  shell whose every declared tool is left out of its request because this seating
  planted no channel for it to act on — an episode that would open on a turn the agent
  has nothing to answer with. Refused as the environment is built, before the container
  starts and before anything is billed.
- `[harness_files]` with a key other than `balance`, `digest` and `round`, a value that
  is not a string, or a multi-segment file name. Any may be `""`: no balance file is
  planted for any seat, no digest is written, or no round announcement is written.
- Settings' own ranges are checked once, by `apply_config`, wherever they came from.

## 11. What reaches the trace

Every episode's provenance stamps: the harness digest, the system prompt in force whole
and by digest, the image and its id, the rates,
every setting of section 3, the starter files' name and digest, each experimenter
channel's source digest, the seating, labels and per-agent presentation order, the schedule, the manifest's digest, the
harness files' names, the channel table in force, whole and by digest, and the tool
table in force, whole and by digest, each tool's declared description among its fields,
whether the shell was offered, and `message_delivery = "episode"` for schema-free
mailbox messages that expire after their recipient's next episode.
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
has no entry. A directory
every agent reads records `posted` and `penalty`; a mailbox records `addressed`, `broken`
and `penalty`; the schema channel records its declaration, whether it was submitted, what moved,
and `penalty`. The
account keeps `penalised`, the running total per channel.

Every tool record carries `tool` (`bash` or the declared name), `result`, and then
`command` for the shell or `input` for a declared tool, the other being null.
`trace_version` is 4. The trace names `provider`, `requested_model`, and
`resolved_model`; every turn carries canonical `usage`, itemized `charges`, canonical and
native stop reasons, and the provider provenance. Raw logs write the provider and complete
native response before their canonical normalized event. A fresh CLI run moves matching prior
records and environment mirrors under `displaced/<timestamp>/`; `--resume` instead requires
compatible version-4 accounts and never mixes version-3 records with this format.

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
| ordered, barrier | sequential, simultaneous | Order of moves within a round | Sequential and simultaneous play; simultaneous rounds are BSP supersteps |
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
