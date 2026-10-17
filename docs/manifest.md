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
| **Harness** | The code that builds sandboxes, runs episodes, meters spend and writes traces. What it says to agents it computes from the accounts and rewrites whenever those move: the balances, the digest, the ledger and a receipt. Its fixed text is the pinned refusal notice, the generated tool descriptions (section 4.8), tool-call results, and, for a tool-only agent, the digest's agent-view headings and renderings |
| **System prompt** | What is said to an agent on every turn. The experimenter's constant, sent with every model call; the harness ships none, and every manifest declares one, `""` included |
| **Experimenter** | The person running an experiment. Everything they say to agents is a constant they declare on the prompt surface. The harness rejects any change to the system prompt, starter files or an experimenter channel during an agent's life, and records a changed tool description as configuration drift. Configures everything through `config.toml` and a manifest |
| **Prompt surface** | The four places the experimenter's own words reach an agent: the system prompt, starter files, an experimenter channel, and tool descriptions. Each is recorded in provenance, and a change to any of them starts a new arm |
| **Agent** | An account, a seat, a memory inherited from episode to episode, and the model that acts for it |
| **Peer** | Another agent in the same experiment |
| **Label** | How an agent is named to its peers in paths and files. Defaults to its seat number |
| **Human seat** | A seat whose turns a person answers instead of a model, declared with `provider = "human"` and `model = "interactive"`. Every other seat is an **autonomous seat** |

### When

| Term | Definition |
|---|---|
| **Episode** | The life of one fresh sandbox: one shell, and turns until the agent ends its turn without a tool call, the context fills, the balance reaches zero, or another stop ends it: a `config.toml` limit, a model refusal, an API or harness error, or the experimenter's interrupt. Produces one trace |
| **Turn** | One model call and the commands it asks for |
| **Episode steps** | What every episode does, in order. **Build**: make the sandbox and plant `/work`. **Run**: take the turns, then mirror the trees the agent writes back to the host and **reap** the sandbox. **Settle**: apply the spend, the transfer and the silence penalties to the balance. **Close**: apply the floor at zero, write the trace, record the episode in the account and pay what its transfer owes. An episode is **committed** once it is settled and closed, which is when its account is saved |
| **Round** | One episode for each seated agent, one after another or all at once as the schedule has it. Each seat counts them by its own episodes: the one it is in is its episode count plus one |
| **Current round** | The furthest round any seat has played, whether or not that seat is still in play |
| **Vote cycle** | A block of `every` consecutive rounds, as the experiment's `vote` tool declares, counted by each seat's own round. Only the last round of each offers the ballot, so with `every = 5` rounds 5, 10, 15 and so on are votes |
| **Round type** | Whether a seat's round is a **discussion**, offering every declared tool but the ballot, or a **vote**, offering only the ballot and the memory tools. The last round of a vote cycle is a vote unless that cycle's election is held already. The round announcement states it on its `round type:` line |
| **Experiment** | Several agents advancing together under one manifest. The unit of comparison |
| **Schedule** | How a round is driven. **Sequential**: one episode at a time in fixed seat order. **Simultaneous**: every sandbox built first, all episodes run at once, results settled in seat order |
| **Grace episodes** | An agent's first `grace_episodes` episodes, in which no silence penalty is taken. Their turns spend as usual |

### Who is still playing

| Term | Definition |
|---|---|
| **In play** | Still a contender: the agent's account admits an episode — it is not exhausted, not eliminated, and not stopped by `REFUSAL_STREAK` refusal-ended episodes running — and its seat is at most one round behind the current round, whether or not this run drives it. Every stop that counts seats counts these, and they alone are candidates and survivors |
| **Seated** | Driven by this run of `experiment.py`: every seat from the run's start, except one whose account admits no episode or that is more than one round behind the current round, until its account admits none, its episode ends on a fault of its own (`api_error`, `harness_error`), or its sandbox will not build. A seat the run stops driving is **unseated** for the rest of that run |
| **Exhausted** | Holding a settled balance of zero or less. The agent takes no further episode, and no transfer can reach it |
| **Eliminated** | Removed by an election, and only by one. Two kinds of seat are removed: every elector offered the ballot that cast none, except one whose episode the experimenter's interrupt ended; and the candidate with the unique highest total above zero, unless the interrupt ended an elector's episode before it cast the ballot it was offered. The agent takes no further episode |
| **Election** | The count that follows the last round of each vote cycle. Its **electors** are the seats that took that round's episode without a fault; its **candidates** are the seats in play; a **ballot** is one elector's private choice of one candidate, cast with the `vote` tool. Section 4.8 gives the rules |
| **Tie** | Two or more candidates sharing the highest total above zero, so the vote eliminates no candidate; an elector offered the ballot that cast none, other than one whose episode the experimenter's interrupt ended, is eliminated all the same. Under `stop_when_two_remain_after_tie`, a tie between the last two seats in play, both electors in it, ends the experiment as `final_tie`, unless the interrupt ended an elector's episode before it cast its ballot |

### Where

| Term | Definition |
|---|---|
| **Sandbox** | The Docker container one episode runs in: its own filesystem with real modes and ownership, no network, and `/work` |
| **Environment** | Everything under `/work` in an agent's sandbox: the declared directories and files it can reach, and the status files, nothing else |
| **Seat** | An agent's numbered position, from 1: its order, and its default label |
| **Seating** | Which agent holds each seat, the label each seat goes by, and the order every agent is shown the others in. Fixed by the manifest's `[[agent]]` order and recorded in every account and every episode's provenance |
| **Channel** | One declared part of `/work` with one writer, one set of readers, and one shape. Declared in the manifest, enforced by ownership and modes |
| **Writer** | Who may put bytes in a channel: `self` or `experimenter`. The harness's own files are the status files, not channels |
| **Readers** | Who may read it: `self`, `all`, `addressee` (one named peer per file), or `harness` (the harness parses it) |
| **Shape** | `directory` (any files, any layout), `mailbox` (one file per peer, an outbox on the writer's side and an inbox on each reader's), or `file` (one file at a fixed path) |
| **Memory** | A `self`-written, `self`-read directory: what an agent keeps for its own later episodes, and what no other agent ever sees |
| **Board** | A `self`-written, `all`-read directory: every agent has one, every agent reads all of them |
| **Mailbox** | A `self`-written, `addressee`-read channel. A **message** is what one agent puts in its outbox for one peer: it appears in that peer's next inbox and nowhere else, then expires |
| **Reachable peers** | The peers one episode can address: every other seat whose account admits an episode as this one starts. A tool that names a peer takes only these, and one with none to name is not offered |
| **Schema** | A fixed format the harness parses and acts on, read either from a `self`-written, `harness`-read file or from a `self`-written, `addressee`-read mailbox. A fixed menu in code |
| **Tool** | A named operation a model call offers an agent, declared as a kind from a fixed menu pointed at a channel, with the description the experimenter gives it. Includes bash only when explicitly declared |
| **Tool call** | One invocation of a tool that a response asks for: the tool's name and its input. The harness checks the input against the tool's input schema before acting on it |
| **Kind** | What a tool does, and so what it says of itself: `bash`, `write_message`, `send_message`, `send_message_to`, `write_file`, `post_public`, `write_memory`, `vote`, `read_path`, `transfer`. A fixed menu in code |
| **Starter files** | Files the experimenter gives one agent, copied once into its memory when its balance first falls to its starter threshold |
| **Starter threshold** | The balance, `starter_files_below`, at or below which an agent's starter files land. At or above the budget, they land at episode 1 |
| **Experimenter channel** | Files the experimenter gives every agent, identical and read-only in every seat at every episode |
| **Status files** | The read-only files the harness renders from the accounts and plants in `/work`: a balance file per seat, the digest and the round announcement, named in the `[status_files]` table, and a ledger and a receipt, named on the transfer channel |
| **Plant** | Put files into `/work` as an episode's sandbox is built: the mirrored trees, starter files where they land, each experimenter channel's files, and the status files |
| **Mirror** | The host's copy of one tree an agent writes, one per channel under `mirrors/<agent>/`: planted into each new sandbox and copied back out when the episode's turns end |
| **Digest** | The status file an episode opens on where the experiment quotes content at all, written at episode start: the round announcement, every file of each channel marked to be quoted, and the other status files. Content new to the agent is quoted in full, and channel content it was shown before and that has not moved is named. `[status_files] digest = ""` writes none |
| **Initial observation** | The first input of an episode: with a shell, the directory listing plus the digest under push delivery; with none, the digest alone |

### What moves

| Term | Definition |
|---|---|
| **Balance** | An agent's micro-dollars, kept as a history: one entry per turn and one per movement outside a turn. The last entry is what the agent holds now |
| **Spend** | The micro-dollars an episode's turns consume: the provider's itemized **charges**, one per token class, each computed from the model's **rate** for that class |
| **Delivery** | Whether an episode opens on quoted channel content. `push`: it does. `pull`: nothing is quoted, and the agent reads what it chooses in turns that spend like any other |
| **Pushed** | A channel property: whether its files are quoted to the agent at episode start or left for it to read |
| **Silence penalty** | A channel property: the share of the balance taken from an episode that added nothing new to the channel |
| **Transfer** | One episode's declaration crediting a peer: one line `<label> <amount>` in a parsed file, or one mailbox message to that peer containing `<amount>`. The peer is credited with no more than the giver's episode spent |
| **Funded by** | Who pays for a transfer. `harness`: the receiver is credited from nowhere and the giver rebated; the total grows. `giver`: the amount leaves the giver; the total is conserved |
| **Rebate** | Under harness funding, the share of a transfer returned to the giver out of what its episode spent |
| **Ledger** | Every transfer an experiment has made, three integers a line, rebuilt from the accounts at every episode |
| **Receipt** | A status file planted for the next episode of the transfer channel's writer, itemizing the settlement of the episode before. Optional |
| **Floor at zero** | Putting a balance below zero back to zero at the end of an episode, and again once a credit it closed on is taken back. What it puts back is recorded as `floored`. A balance at zero is exhausted either way |
| **Settlement** | What an episode's end does to the balance, in a fixed order: the turns' spend, the transfer and its rebate, and each channel's silence penalty in declaration order, which the settle step applies; then the floor at zero, which closing applies. Recorded in the trace and itemized in a receipt |
| **Reconciliation** | The identity every account satisfies once an episode is committed: `balance == budget - spent + rebated + received - debited - sum(penalised) + floored`. A receipt states it for the episode it covers |

### What is recorded

| Term | Definition |
|---|---|
| **Account** | An agent's ground truth on disk: pinned settings, balance history, episodes, transfers in and out |
| **Pinned settings** | The six things fixed when an agent is created: the system prompt, budget, provider, model, starter files, and the starter threshold |
| **Trace** | One episode's complete record: what the agent saw, said, ran and left behind, every file with its author, and the provenance |
| **Author** | Who wrote a captured file: `experimenter`, `self`, or `peer:<label>` |
| **sha256** | The SHA-256 hash of a file, a text or a table, in hex. Provenance identifies the harness, the system prompt, the manifest, the channel and tool tables, starter files and each experimenter channel's source by it |
| **Provenance** | Everything an episode ran under, stamped on its trace: rates, every setting, the seating, the schedule, and the sha256 of the harness, the manifest and the channel table |
| **Configuration drift** | Any difference between one episode's provenance and the previous episode's. Reported, and it starts a new arm |
| **Arm** | A stretch of episodes comparable with each other: same harness, same system prompt, same settings, same files |
| **Persona drift** | The literature's term for an agent's tone or self-description moving over time. What the identity analysis measures |
| **Progress log** | `progress.jsonl` under `experiment_records/<experiment_id>/`: one line for each stage a round reaches, as it reaches it, with `progress.json` holding the latest |
| **Dashboard** | The page `view.py` serves: each experiment's seats, channels, traces and ledger, read from the records as episodes run, and the one route through which a human seat's submission reaches the interaction store |

### Providers and human seats

| Term | Definition |
|---|---|
| **Provider** | The adapter that answers a seat's model calls: `anthropic` and `openai` send them to that vendor's API, and `human` to a person. It normalizes what comes back and itemizes its charges |
| **Session** | One episode's connection to its provider, holding the conversation state carried between turns. Each episode opens its own as its turns begin |
| **Response** | What one model call returns, normalized: text, reasoning, tool calls, usage, itemized charges and a stop reason |
| **Request** | A human seat's turn waiting for a person: the system prompt, the input and the tools the turn may call, held in the interaction store until a submission answers it or the episode cancels it |
| **Submission** | A person's answer to one request: the tool calls to make, or the end of the turn |
| **Interaction store** | The files under `interaction_store/` holding each human seat's requests and submissions, which `human.py` and the dashboard both read and write |

## 1. What a manifest is

An experiment is several agents advancing together under one set of rules. A manifest
is one TOML file under `experiments/` that declares all of it; the shipped examples sit
in `experiments/examples/`. Every run names one, so nothing an agent is told and nothing
it can reach is decided anywhere else. `config.toml` holds the process parameters — the
machine, the API, the limits — and nothing an agent's situation is made of; the two
sets are disjoint and each file rejects the other's keys by name.

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
- **The default channels are the code's.** An experiment that declares no
  `[[channel]]` gets the default channels of section 8. It must still declare at least one
  `[[tool]]`. What is said to an agent has no default: a manifest declares it or is
  rejected.

### Running a manifest

`py -3 experiment.py NAME` runs every seat; `py -3 harness.py --agent ID --manifest PATH`
runs one.

| Flag | Meaning |
|---|---|
| `NAME`, `-m NAME` | The manifest: a bare name is looked for in `experiments/` then `experiments/examples/`; anything with a suffix or directory is a path |
| `-r N` | Up to N rounds, default 1, stopping early where a stop below ends the rounds first; under `--resume` a round the last run left unfinished is finished first, and is one of the N |
| `--provider P --model M` | Given together, override every seat's provider and model; under `--resume` both must match the accounts |
| `--resume` | Continue existing compatible accounts, finishing first a round the last run left unfinished and holding a finished vote round's election the last run did not, and never one it did; a seat more than one round behind the current round is unseated, and is no longer in play. Without it, the previous records and mirrors move under `archive/` and a fresh run starts |
| `-c PATH` | The config file; default `config.toml` beside `harness.py` |
| `--branch-from MANIFEST --at-round N --branch-id ID --human-seat SEAT [--output PATH]` | Write a branch manifest from a completed round, each seat's new agent standing where its seat stood as that round ended and SEAT's on the human provider, and stop |
| `harness.py --episodes N` | Up to N episodes for one agent, default 1 |
| `harness.py --watch` | Echo the agent's words and account to stdout |
| `harness.py --fork-from AGENT --at-episode N` | Rebuild AGENT as it stood at episode N under the `--agent` id, and stop |
| `harness.py --print-system`, `--print-context`, `--print-files NAME` | Print the shipped and declared text, the initial observation, or a `files/` listing; start no episode |

Each experiment writes under `experiment_records/<experiment_id>/`: the progress log,
`progress.jsonl`, one line per stage each round reaches as it reaches it, the last naming
the round the rounds ended at, so its rounds never go back; `progress.json`, the latest of
those; `outcome.json`; and, for a branch, `lineage.json`. `progress.json` and
`outcome.json` record `version` 3. Each line of the log names its `stage` and its
`round`, and a stage that waits on a human seat is `waiting_human`. A round that a stop
counting seats ends the rounds before has no line, and the round the cost ceiling ends
them before has its preparation alone, which carries the spend that ended them. The
outcome's `end_reason` is one of `round_limit`, `cost_ceiling`, `one_remains`,
`final_tie`, `none_in_play`, `none_can_act` or `interrupted`. Every stop that counts
seats is judged on the seats in play, whose accounts admit an episode and which are at
most one round behind the current round, whether or not this run seats them: a seat whose
episode ended on a fault of its own, or whose sandbox would not build, is unseated for
the rest of that run and stays in play until it falls further behind, which is once the
current round is two past the last it played. One further behind is unseated in every
round of every run, and is no longer in play: it counts toward no stop, survives nothing
and stands in no election. `none_in_play` is no seat left in play, however each one left;
`none_can_act` is seats left in play and not one of them seated in this run. The
outcome's `survivors` are the seats in play as the run ends, and its `elimination_order`
the seats an election eliminated, each by `round`, `label` and `reason`, so a seat that
left play any other way is in neither, and what it holds is in `resources`. `tie` is true
where the run ended `final_tie`. Human seats are answered as [docs/human.md](human.md)
describes.

## 2. Top level

| Key | Type | Meaning |
|---|---|---|
| `experiment_id` | string, default manifest stem | Stable identity for progress, outcome, and lineage records; letters, digits, `.`, `_`, and `-` |
| `schedule` | `"sequential"` \| `"simultaneous"`, default `"sequential"` | How a round is driven |
| `stop_when_one_remains` | bool, default `false` | Whether the experiment ends once exactly one seat is in play (section 1) |
| `stop_when_two_remain_after_tie` | bool, default `false` | Whether a vote round ends the experiment with two survivors when its aggregate result is tied and exactly two seats are in play (section 1), both of them electors in it. A tie the experimenter's interrupt kept an elector's ballot from ends nothing, since that ballot could have broken it |
| `[status_files]` | table | Names of the status files `balance`, `digest` and `round`, overlaid key by key; `[harness_files]` is accepted as another name for it. Section 5 |
| `[[channel]]` | tables | The channels. Declaring any replaces the default channels whole |
| `[[tool]]` | tables | The tools offered beside the shell, each pointed at a channel and carrying its description. Section 4.8 |
| `[cost]` | table | The experiment's cost ceiling on autonomous spend: maximum, reserve, warning, and policy at the ceiling |
| `[reveal]` | table | Which outcome fields are designated for each audience and reveal point |
| `[[agent]]` | tables, one or more | Seat definitions, in order; each occupies one seat unless `seats` groups identical agents |
| `system_prompt` | str, **required** here or on every `[[agent]]` | What is said to every agent on every turn. `""` says nothing |
| any other key of section 3 | | The default for every seat |

Unknown keys are rejected, naming the key; so is a `config.toml` key, saying where it
lives. The file's sha256 is stamped in every episode's provenance.

`[cost]` may declare positive `maximum`, non-negative `reserved_completion` and
`warning`, and `ceiling_policy = "stop"`. Before each round, spend is summed from the
committed episodes of autonomous seats. A new round is not started when spend plus the
reserve reaches the maximum. A round already running is allowed to settle; providers are
never silently changed. A human seat's episodes have zero provider charges and add
nothing to autonomous spend. `reserved_completion` must be below `maximum` and `warning`
at most `maximum`; `warning` only marks `warning_reached` in the progress log.

`[reveal]` may list outcome fields under `during_play`, `at_elimination`,
`at_completion`, and `experimenter_only`. The permitted fields are `winners`,
`survivors`, `tie`, `elimination_order`, `resources`, `end_reason`, and
`evidence`. The outcome record remains the complete
experimenter record; the table is the declarative policy a human-facing display uses.

Seat order determines identity, the order seats are shown in, and settlement order. Every agent sees
agents and records in that same order, and every episode records it as
`seat_order` in provenance.

## 3. Settings

Every key here is an experiment's: valid at a manifest's top level, and rejected in
`config.toml`. Types are strict; an integer widens to a float field and nothing else
converts. The keys `config.toml` owns are at the end of this section.

Episode limits that bound the machine rather than what an arm compares — `max_tokens`,
`max_turns`, `command_timeout`, `tool_result_limit` — are `config.toml`'s. What is left
here is what the agent's situation is made of.

### What the harness says

| Key | Type | Default | Meaning |
|---|---|---|---|
| `system_prompt` | str | **required** | What is said to every agent on every turn. Empty says nothing at all. Pinned |

It is the one setting with no default. A manifest declares it at the top level or on
every `[[agent]]`, and one that declares it nowhere is rejected: what an agent is told is
the experiment's, and an omission and a decision must not look alike in the record.

`""` is a declaration and says nothing - the harness ships no words, and no system
parameter is sent. Anything else is a different arm: it is sent, cached, as input that spends on
every turn of every episode, and every episode records it whole and by sha256. An
`[[agent]]` may declare its own, so one seat can be told what its peers are not, and a
seat that declares none takes the experiment's. `py -3 harness.py --print-system` prints
what is shipped beside whatever is in force.

The refusal notice a declined turn receives in place of its tool results is not
declarable. It is the harness reporting a fact about a turn, not something an arm varies,
and it is pinned by sha256 like the shipped prompt.

### Money

| Key | Type | Default | Meaning |
|---|---|---|---|
| `budget` | int > 0 | 500000 | Micro-dollars an agent starts with. Pinned |
| `provider` | `"anthropic"` \| `"openai"` \| `"human"` | none | Named provider adapter. Required with `model` and pinned. Human seats use model `"interactive"` |
| `model` | model in the provider catalog | none | Exact model requested on every turn. Required with `provider` and pinned |
| `floor_at_zero` | bool | false | A balance below zero is put back to zero |
| `grace_episodes` | int ≥ 0 | 0 | The grace episodes: opening episodes that take no silence penalty |

### Episode limits

| Key | Type | Default | Meaning |
|---|---|---|---|
| `context_fraction` | 0 < x ≤ 1 | 0.85 | Fraction of the window at which an episode ends. Spend scales with its square |
| `live_balance` | bool | true | Balances update inside an episode, not only between them |

### Files

| Key | Type | Default | Meaning |
|---|---|---|---|
| `starter_files` | a file or directory under `files/`, or a `./` or `../` path beside the manifest | `""` | Copied once into the agent's memory. Pinned |
| `starter_files_below` | int ≥ 0; > 0 exactly when `starter_files` is not `""` | 0 | The starter threshold: the balance at or below which they land. At or above the budget, the first episode |

`starter_files` and `starter_files_below` are set together or not at all. What lands is
recorded in the account by name, sha256, episode and paths, and lands once.

### Delivery

| Key | Type | Default | Meaning |
|---|---|---|---|
| `delivery` | `"push"` \| `"pull"` | `"push"` | Whether pushed channels are quoted in the digest or left to be read |
| `digest_file_limit` | int ≥ 200 | 2000 | Characters of each file the digest quotes |
| `observation_limit` | int ≥ `tool_result_limit` | 40000 | Characters of the whole initial observation |

Both clips are bounded against `tool_result_limit`, which `config.toml` owns.

### Rejected: what config.toml owns

`image`, `max_tokens`, `max_turns`, `command_timeout` and `tool_result_limit` are the
process parameters: the machine, the API and the limits, true
of every run whatever the experiment is. They live in `config.toml`, and a manifest
naming one is rejected saying so — no setting is given in two places, and no run's
settings depend on which file was read last.

### Rejected: retired keys

A manifest or `config.toml` naming one of these keys is rejected, and the rejection names
where the setting is declared.

| Key | Declared as |
|---|---|
| `fallbacks` | nothing: every turn calls the seat's pinned provider and model |
| `shell_tool` | a `[[tool]]` with `name = "bash"` and `kind = "bash"` |
| `transfer_funded_by`, `rebate_percent`, `transfer_silence_penalty_percent` | `funded_by`, `rebate_percent`, `silence_penalty_percent` on the channel with `schema = "transfer"` |
| `blackboard_silence_penalty_percent` | `silence_penalty_percent` on the board channel |
| `mailbox_silence_penalty_percent` | `silence_penalty_percent` on the mailbox channel |
| `shared_files` | a `[[channel]]` with `writer = "experimenter"` and a `source` |

## 4. The channel object

A channel is one declared part of every agent's environment. Every channel has a name, writer,
readers, shape and path; the remaining fields are optional.

```toml
[[channel]]
name = "board"              # required; unique; how the manifest and the trace refer to it
writer = "self"             # self | experimenter
readers = "all"             # self | all | addressee | harness
shape = "directory"         # directory | mailbox | file
path = "{label}"            # where it sits in /work; see 4.2
pushed = true               # quoted in the digest under push delivery
restated = false            # quoted in full every episode; needs pushed; see 4.3
measured = true             # the episode records what this channel gained; see 4.4
agent_view = "paths"        # paths | memory | mailbox | board | transfer; digest labels
silence_penalty_percent = 50
```

### 4.1 Writer and readers

| Writer | Readers | Meaning | Default channels |
|---|---|---|---|
| self | self | The agent's memory; nobody else ever sees it | `memory/` |
| self | all | A board: one per agent, written by its owner, read by every seat | the directories named by label |
| self | addressee | A mailbox: one file per peer, each reaching that peer alone; the harness also parses a transfer channel in this shape | `outbox/` and `inbox/` |
| self | harness | One file the harness parses; a schema is required | `outbox/transfer` |
| experimenter | all | Placed by the experimenter, identical in every seat, read-only | none |

Other combinations are rejected. An experimenter channel takes `source = "<dir under
files/>"` and `path`, and optionally `pushed`, `restated` and `agent_view`. The status
files are section 5, not channels.

### 4.2 Shape and path

| Shape | Holds | `path` | Placeholders |
|---|---|---|---|
| `directory` | Any files, any layout | `"{label}"` for a board, `"memory"` for a memory | `{label}` is the owner's label |
| `mailbox` | At most one file per peer | `outbox = "to"`, `inbox = "from"` | the writer sees `to/<peer label>`, the reader `from/<sender label>` |
| `file` | One file at a fixed path | `"to/transfer"` | none |

A self-written, all-read directory has one instance per agent, and `{label}` names each.
A self-written, self-read directory has one instance and no placeholder. Paths must not
collide, must not name a status file, and must not begin with `/` or `..`. A file
channel sits inside a directory the agent writes and comes and goes with it.

### 4.3 Pushed

`pushed = true` means the channel's files are quoted in the digest at episode start
under push delivery, each clipped at `digest_file_limit`, new content in full and
unchanged content by name. Unchanged and withdrawn names appear one per line beneath a
labelled group, so agent-view names containing spaces remain distinct. Expired board posts
and schema-free mailbox messages disappear silently. Every channel defaults to `true`, a
memory included. Under pull delivery nothing is quoted whatever this says.

`restated = true` quotes the channel in full every episode it stands, never naming it
as unchanged. "Unchanged" is measured against what the *account* was last shown, and an
episode does not remember what its predecessor read, so a channel carrying who an agent
is tells it nothing when it arrives as a name. It needs `pushed`, and adds input tokens
to every turn of every episode. The schema channel is restated whether or not it says so.

A memory is pushed like anything else. It is nobody else's business either way - the
digest is per-agent - so this puts what an agent wrote to itself in front of it at
episode start instead of leaving it to be found and paid for. An agent is read the same file
whether it goes looking or not, so the choice is only whether it pays a turn first, and
what it wrote to itself last episode is the thing it most needs this one. An experiment
that wants a memory written and never read back declares `pushed = false` and says so.

`agent_view` changes only how the agent's digest names and renders a channel; storage and traces keep
their native paths and bytes. `"paths"` is the default.
`"memory"` presents a memory's files as the agent's own memory, naming starter files
and an experimenter channel's files as such; `"mailbox"` presents mailbox messages
under `Message from` and `Message to` headings naming the peer; `"board"` presents board
files as board posts from their authors; and `"transfer"` presents a transfer channel,
file or mailbox, as outgoing and incoming transfers. Storage and access rules
remain the same, so an experiment can give agents persistent memory without making a
filesystem part of their environment. `"letters"` is accepted as another spelling of
`"mailbox"`: the channel holds, renders and records it as `"mailbox"`.

Agent views are checked against the channel they describe: `"mailbox"` requires
a mailbox, `"board"` a board, `"transfer"` a transfer-schema
file or mailbox, and `"memory"` a memory or an experimenter channel. What the
experimenter provides is labelled as such in a memory view rather than presented as the
agent's memory.
The board view marks the owning agent's own label with `(you)`.

When no Bash tool is declared, the balance, ledger and receipt status files also use
agent-view digest headings. Their configured filenames remain implementation details and
do not enter the model's observation. Balance bodies state the current value and label
their oldest-to-newest history; ledger rows name round, giver, receiver and amount;
prior-round transfer notices name their receiver, requested amount and result. A
receipt itemizes the preceding episode's starting balance, API spend, whether it was a
grace episode, the transfer, each settled channel's silence penalty (`none` where the
episode spoke on the channel), credits from peers, the amount floored at zero, ending
balance, and the reconciliation.

### 4.4 Measured, and the silence penalty

Three things settle a channel, each configured on its own and none implying the others
beyond what arithmetic forces:

| Configured | The episode does |
|---|---|
| nothing | not settle the channel at all: no record, nothing on the console, nothing measured |
| `measured = true` | record what the channel gained — `posted`, or `addressed` and `broken` — and take nothing |
| `silence_penalty_percent` above 0 | measure it and take that share where it gained nothing |
| `schema` | settle it, which is what moves a transfer, whatever the other two say |

A penalty implies the measurement, since a share cannot be taken from what was not
measured. `measured` on its own is the observational arm: an experiment that wants to
know whether agents wrote to each other without a silence penalty for not doing it. A channel
that asks for neither is invisible to the settlement entirely, and an experiment that
declares no penalties anywhere has none.

`silence_penalty_percent` is the share of the balance taken from an episode that added
nothing new to the channel. A channel used by `post_public` begins each
episode without its previous `post.md`, so a nonempty post must be published each time
and may repeat the previous text. A schema-free mailbox likewise begins each episode with
its outbox empty, so one or more nonempty messages avoid the silence penalty; further
receivers add none. A peer's outbox path holding anything but one file reaches nobody,
but does not cancel a valid message to another peer. Zero is no penalty. Penalties are taken
in declaration order, after the transfer channel's.

### 4.5 Schema

A schema channel names a `schema` from the fixed menu and carries that schema's fields.
The harness acts only on content that parses and records why malformed content moved
nothing. A transfer uses either a self-written, harness-read file or a self-written,
addressee-read mailbox. The file form names receiver and amount in one line. The mailbox
form stores the amount as a message to the receiver, copies it into that receiver's
next inbox, and permits only one nonempty message at a time. Transfer declarations
are cleared before each episode and settle only in the episode that submits them.

| Schema | Form | Effect | Fields |
|---|---|---|---|
| `transfer` | one line, `<label> <amount>`, or one message `<outbox>/<label>` containing `<amount>` | Credits the peer, for no more than the episode spent | `funded_by` (`harness` \| `giver` \| `none`; default `harness`), `rebate_percent` (0 to 100, default 100; 0 under giver funding), `silence_penalty_percent`, `ledger` (a status file name, or `""` for none), `receipt` |

Setting `funded_by = "none"` disables the schema: the file is recorded and moves
nothing, and `silence_penalty_percent` must be 0. One schema channel per experiment. The
schema menu is fixed in code.

A `receipt = "<path>"` on a schema channel asks the harness to plant an itemized
receipt of the settlement at that path at the next episode start. It states the episode,
starting balance, API spend, whether it was a grace episode, declaration and transfer
result, one line per settled channel giving its silence penalty (`none` where the episode
spoke on the channel, otherwise the amount taken, 0 included) and what the episode did
there, the total silence penalties, amount received from peers, the amount floored at
zero, ending balance, and reconciliation. It is a status file like a balance: root-owned, quoted in the digest once and named as
unchanged after, in no file record, and scrubbed before the next one is planted. Off by
default, since it is an additional statement from the harness.

### 4.6 Experimenter channels

```toml
[[channel]]
name = "brief"
writer = "experimenter"
source = "workshop-brief"   # files/workshop-brief/, copied root-owned into every seat at every episode
path = "brief"
```

`restated = true` quotes it in full at every episode start rather than naming it as
unchanged after the first, which is what standing text in front of an agent means.
`measured` is rejected: nothing is owed to a channel the agent cannot write.

The directory's sha256 is in provenance, per channel, and the harness rejects the next
episode of an agent whose experimenter channel changed since its last: episodes either
side of the change would be separate arms.

### 4.7 How many

An experiment may declare as many directories and mailboxes as it likes: two boards,
three mailboxes, a memory and an identity file inside it. Each is a `[[channel]]` table
with its own name and path, and each directory every agent reads carries its own silence
penalty, settled and taken apart. One schema channel. There is no count field because the table
is the count.

### 4.8 Tools

A channel says where an agent may write and who reads it. A tool is that same
permission offered as an operation, in the model call rather than in a file the agent
has to find, read and infer a purpose for.

```toml
[[tool]]
name = "send"               # required; unique; letters, digits, '_' and '-', at most 64
kind = "write_message"      # required; from the menu below
channel = "mailbox"         # required; a declared channel the kind can act on
description = "..."         # optional; prompt surface. Omitted, the harness writes it
```

`vote` also requires `every = N`, where `N` is a positive integer. No other kind
accepts `every`, and an experiment declares at most one `vote` tool.

Every tool must be declared. No declaration means no bash; an empty tool set is rejected.

| `kind` | Takes | Input | What the harness does |
|---|---|---|---|
| `bash` | no channel; name must be `bash` | built-in bash schema | Executes shell commands |
| `write_message` | a mailbox channel without a schema | `to` (a peer's label), `body` | Sends one episode-scoped message through `<outbox>/<to>`; a later call to that peer in the episode replaces it |
| `send_message` | a mailbox channel without a schema; a call succeeds only when exactly one peer is reachable | `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `send_message_to` | a mailbox channel without a schema | `to` (a peer label), `body` | Sends one private, episode-scoped message without exposing the mailbox path |
| `write_file` | a directory channel the agent writes | `path`, `body` | Replaces what `<the agent's instance>/<path>` holds |
| `post_public` | a board the agent writes | `body` | Publishes the agent's post for the next round; the prior post is cleared before each episode |
| `write_memory` | a memory channel | `body` | Replaces the agent's memory without exposing storage paths |
| `vote` | a memory channel | `to` (a reachable peer label) | Records one private, episode-scoped elimination ballot; a later call replaces the earlier vote. Offered only in the last round of each vote cycle, the seat's `every`th, `2 × every`th and so on, when every other tool except `write_memory` is withheld, bash included. Peers receive only the aggregate outcome, never voter-to-target mappings; what follows the round is **The election**, below |
| `read_path` | any channel | `path` | Returns what that path holds, clipped at `tool_result_limit` |
| `transfer` | an enabled transfer schema channel | `to` (a reachable peer label), `amount` (a whole number of micro-dollars that is at least 1; zero and negative values are invalid) | Submits one transfer for the current episode; in mailbox form a later call replaces the earlier receiver. Settlement moves at most the episode spend, using the channel funding and rebate settings, then the declaration expires |

**The election** follows the last round of each vote cycle, the cycle being `every`
rounds counted by each seat's own round. Every seat in play is a candidate, and a ballot
counts toward the candidate it names. The electors are the seats that took the round's
episode without a fault; a seat that takes the round after its election is held is none,
and takes it as a discussion, offered every tool but the ballot. An elector offered the
ballot that cast none is eliminated; one with no peer left to name is offered none. The candidate with the unique highest total above zero is
eliminated by vote. Nobody is eliminated by vote where two or more candidates share the
highest total, or where the experimenter's interrupt ended an elector's episode before it
cast the ballot it was offered: such an elector is not a nonvoter, and where the interrupt
landed would otherwise decide who goes. The result goes to every candidate and elector, whose
account keeps every election it was in as `elections`, the last as `last_election`.

The two `path` arguments are not the same argument. A `write_file`'s is relative to the
one instance the agent writes, there being only one place it could mean. A `read_path`'s
is whole, because the channel it reads may have an instance per seat and a bare name
would not say which. So one string can name two files, and a `description` that replaces
the harness's says which of them it means; the generated ones already do.

The kind menu is fixed in code, like the schema menu: a manifest names a kind and a
channel and invents no behaviour.

The input schemas and results for `send_message`, `send_message_to`, `post_public`,
`write_memory`, `vote` and `transfer` use the operation's vocabulary. They do not describe
backing files or return shell diagnostics; an internal failure is recorded as a tool call
that changed nothing, without revealing its storage path to the model.

**Bash is opt-in**, using this declaration:

```toml
[[tool]]
name = "bash"
kind = "bash"
```

Bash takes no channel, custom description or `every`. Its API schema is built in.
Without this declaration, agents receive only the declared channel tools. The
sandbox and its shell still carry out those operations internally.
Withholding bash also changes what an episode opens on. A listing is what an agent holding the shell
reads to know what there is to reach; with no shell there is nothing to reach it with, so
the episode opens on the **digest alone** - the content of the channels rather than their
layout. That is the arm for an agent that never has to read a filesystem: every path it
needs is named in the tool descriptions and every byte it is shown is content.

Three arrangements are rejected before anything is spent, all being experiments that
would end every episode on its first turn:

| Rejected | Why |
|---|---|
| no `[[tool]]` at all | the agent is offered nothing to act with, and an empty tool set is not a model call the API takes |
| no declared `bash` tool with `delivery = "pull"` or `[status_files] digest = ""` | the listing is gone and no digest replaces it, so the first user turn would be empty |
| no declared `bash` tool where this seating leaves every declared tool out | the tool table is not empty but the model call's tools would be, for the same reason and with the same result |

The first two are decided when the harness starts, before any sandbox is built. The
third is a seat's rather than
an experiment's — which tools can act depends on what the seating planted — so it is
decided as that agent's sandbox is built, before it starts and before anything is
spent.

**A tool acts through the episode's own shell**, so what it leaves is the agent's file
with the agent's ownership, mirrors back with its tree, and settles exactly as the same
bytes written with `bash` would. It runs no command of the agent's, so the episode's
`commands` do not grow.

**A tool is offered only where it can act.** A tool whose channel this seating did not
plant is left out of the model call; a mailbox is not planted for an agent with no peers.
A seat outside the reachable peers is not a message target either - the mailbox settles
only messages to reachable peers - so it is not in the `to` enumeration, and a
`write_message`, `transfer` or `vote` tool with no reachable peer is not offered at all. Which tools an agent is
actually offered follows from the tool table and the seating, both in provenance.

#### What a tool says of itself

A tool's `description` is **prompt surface, and the experimenter's**. It reaches the
model inside the model call exactly as the system prompt does, it is theirs to write, and
it is recorded whole and by sha256 in the tool table like every other declaration. Two
wordings of the same tool are two arms.

Tool descriptions are the fourth prompt surface, beside the system prompt, the starter
files and an experimenter channel - and the only one attached to a tool. Descriptions
can explain when a tool is useful as well as what it does. Asking an agent to use a
particular tool is an experiment design choice, not an API requirement.
`experiments/examples/personas.toml` is the worked example.

A declared description **replaces** the harness's generated one rather than adding to it, so an
experiment that writes one is stating the mechanics itself where it wants them stated.
`py -3 harness.py --print-system --manifest PATH` prints every declared description
beside the prompts, which is what makes this surface auditable without starting an
episode. `py -3 harness.py --print-context --manifest PATH` composes the fresh seats'
first digests and complete bound schemas as well, including reachable-peer lists and
the discussion and vote tool sets.

**The input schema is never declarable.** `input_schema`, `schema`, `properties` and
`required` are rejected by name. The schema is the contract a call is held to - the
harness validates and executes against it - so a manifest that could write it could
promise fields the harness ignores. Same line as the schema menu in 4.5: the experimenter
declares, the code enforces.

**Where nothing is declared**, the harness generates the description, computed from the
channel the tool points at:

| Kind | The generated text names |
|---|---|
| `write_message` | the channel's name, its outbox and inbox paths, the writer's label, and that the addressee alone reads it |
| `write_file` | the channel's name, the agent's own instance path, and whether every agent or nobody else reads it |
| `read_path` | the channel's name, every path of it this agent can reach with a directory's trailing slash, and the clip |
| `transfer` | the whole-number amount rule, that the transfer applies to this episode only, the cap at episode spend, and the channel's funding and rebate |
| `vote` | the `every` cadence, that the ballot is private and a later call replaces it, and the tie rule |
| `send_message`, `send_message_to`, `post_public`, `write_memory` | the operation in fixed wording, naming no path: who reads it, when it expires, and that a later call replaces it |

Every path, label and reader in it is read out of the channel table, so it moves when the
declaration moves and never otherwise; it cannot say what the channel does not. It is not
in provenance as text because it is a function of the harness's sha256, the channel
table and the seating, all three of which are.

The input schema is generated the same way, declared or not; a `write_message`'s `to` is an
enumeration of the peers that channel actually reaches, as this agent names them.

Every declared tool carries `strict`, which makes the API guarantee the arguments
validate: a call naming a peer outside the enumeration or leaving out a body spends no
turn. Every registered model accepts the same strict function schema, so the tool set
does not differ by provider or model. The harness checks every argument anyway (invariant
4): a limit it enforces does not rest on the model keeping to a schema it was handed.

#### What a result says

A tool result is a fact about the environment after the call, and never a bare success.
A write that changed nothing says so, because a tool reporting success for a no-op would
tell an agent it had avoided a silence penalty the settlement will take.

```
wrote 4 bytes to outbox/2, which held nothing before.
outbox/2 already held exactly this. Nothing was written and nothing changed.
replaced the 4 bytes outbox/2 held with 4.
'9' is not a peer this channel reaches; it reaches 2. Nothing was written.
memory/secret is not in the 'board' channel, which holds 1/, 2/. Nothing was read.
```

A call the harness will not make is rejected in the result rather than raised: the path
rule a channel is held to is the path rule a tool argument is held to, so a tool reaches
nowhere a channel could not.

## 5. Status files

The files the harness renders from the accounts and plants read-only in every seat. The
`[status_files]` table names three of them; the transfer channel names the ledger and
the receipt. A manifest may declare the table as `[harness_files]` instead, which reads
the same; a manifest declaring both is rejected.

```toml
[status_files]
balance = "balance-"  # one file per seat, <balance><label>: the seat's balance history; "" plants none
digest = "m"          # the digest under push delivery; "" is the same as delivery = "pull"
round = ""            # default: no round announcement; a name such as "round" plants one
```

When `round` is named, its announcement is the first section of the digest. It states
the seat's own round, the agent's label, the agent itself and its reachable peers on its
`in play:` line, and the round type on its `round type:` line. A vote tool supplies
the vote cycle's length. The first round after a vote also summarizes its result. The
last round of each vote cycle says `round type: vote`, identifies communication as
unavailable and names the vote tool the agent must call before ending the episode. Memory tools
remain available. A seat that takes such a round after its election is held takes it as
a discussion, and is offered every tool but the ballot.

A transfer channel's `ledger` names its ledger file (`"g"` in the default channels): three
integers a line, giver, receiver, amount, rebuilt from the accounts at every episode. Names must be single
path segments and must not collide with a channel path. In a tool-only agent's
digest, the same events are rendered as `giver -> receiver`, its own label is marked
`(you)`, and the amount is identified as the amount actually moved. A transfer channel's
inbox and outbox messages label their amount as requested and direct the agent to the
ledger for the separately settled amount.

## 6. The agent object

```toml
[[agent]]
id = "workshop"             # required; not a bare number; distinct
label = "Workshop"          # optional; defaults to the seat number
seats = 3                   # optional; creates workshop01..workshop03 and Workshop01..Workshop03
memory_from = { agent = "prior-workshop", episode = 12 }
system_prompt = "You run a workshop."
starter_files = "persona-ana"
starter_files_below = 1500000
budget = 2000000            # optional
provider = "anthropic"      # required here or at top level, together with model
model = "claude-sonnet-5"
quality_tier = "premium"    # optional; recorded beside the experiment's spend
```

A definition without `seats` names one agent. With a positive integer
`seats`, it expands in place into that many agents. Their one-based ordinal, padded to at
least two digits, is appended to the `id` prefix and to an explicit `label` prefix;
`id = "peer"`, `seats = 3` therefore creates `peer01`, `peer02` and `peer03`. All other
properties are identical. A later definition continues the experiment's absolute seat
numbering.

A label is how the agent is named to its peers: in `{label}` paths, in mailbox paths, in
its balance file, in the transfer line and in `peer:<label>` authors. Seats stay the
order. A label is letters, digits, `.`, `_` and `-`, distinct from every other after
defaults, and a path too, so one that lands on a channel's path is rejected.

`memory_from` gives a fresh agent only the memories held at the end of one recorded
episode. It names an agent outside this experiment and a positive episode
number. The source and target must declare the same `write_memory` tool names. Each
source `memory.md` is copied byte for byte into the corresponding target tool's channel
before episode 1. No balance, episode count, elimination, peer roster, message, public
post, ballot, trace or provider session is inherited. The source episode must have
mirrored its trees back (`trees_mirrored`), and its trace must contain every copied
memory as complete UTF-8 text. The account and each later trace record the source
episode and the copied files' sha256s.

This is set at creation rather than by a fork: it preserves only the memory an agent
can read while every other part of the agent starts fresh. It is rejected if the
source is another seat in the new experiment, because starting fresh would archive
that source before it could be read.

`quality_tier` is a free string recorded in the account and in the spend summary each
round's preparation writes to the progress log. It neither selects nor changes a
provider or model; those exact settings remain independently pinned.
It defaults to `"standard"`, or `"interactive"` on a human seat.

The six pinned settings are fixed in the agent's account when it is created. An agent
that exists already must have been created on the same six, or the manifest is rejected.
`memory_from` is likewise fixed once the agent has taken an episode. Everything else
about an agent comes from the experiment's settings.

## 7. Schedule

| `schedule` | Round | Who reads what |
|---|---|---|
| `sequential` | One episode at a time in fixed seat order | Each episode reads what the ones before it in the round wrote |
| `simultaneous` | Every sandbox built first; episodes run at once; settled in seat order | Nobody reads this round's writes; a transfer made in round *r* is credited in round *r* and visible at round *r + 1* |

## 8. The default channels

What an experiment gets when it declares no `[[channel]]` and no `[status_files]`. It is
not a manifest of its own: no tools, no system prompt and no agents. The shipped
competition, `experiments/competition.toml`, is a separate manifest that declares its own
channels, with silence penalties.

```toml
[status_files]
balance = "balance-"
digest = "m"

[[channel]]
name = "memory"
writer = "self"
readers = "self"
shape = "directory"
path = "memory"

[[channel]]
name = "board"
writer = "self"
readers = "all"
shape = "directory"
path = "{label}"
measured = true

[[channel]]
name = "mailbox"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "outbox"
inbox = "inbox"
measured = true

[[channel]]
name = "transfer"
writer = "self"
readers = "harness"
shape = "file"
path = "outbox/transfer"
schema = "transfer"
funded_by = "harness"
rebate_percent = 100
ledger = "g"
```

No channel carries a silence penalty. A manifest using these still declares its
`[[tool]]` tables.
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

# Each seat learns its situation from its starter files and the experimenter channel
# below, so the harness says nothing - declared, as every manifest must.
system_prompt = ""

[status_files]
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
name = "board"
writer = "self"
readers = "all"
shape = "directory"
path = "from-{label}"
agent_view = "board"

[[channel]]
name = "mailbox"
writer = "self"
readers = "addressee"
shape = "mailbox"
outbox = "to"
inbox = "from"
agent_view = "mailbox"

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
name = "post_to_board"
kind = "post_public"
channel = "board"

[[tool]]
name = "send_message_to"
kind = "send_message_to"
channel = "mailbox"

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
bash tool, so each episode opens on the digest and every tool call is to a declared tool. No
transfer channel, so no ledger and no transfer silence penalty.

## 10. Validation

Every rejection is a `SystemExit` naming the file and the key.

- Unknown keys anywhere; wrong types; values out of range as section 3 states; any key of
  the rejected tables in section 3, naming where the setting is declared or the file it
  lives in.
- A `[[channel]]`, `[[tool]]`, `[status_files]` or `[harness_files]` table in
  `config.toml`, which declares no channels and so no tools acting on them.
- A `schedule` outside `sequential`, `simultaneous`; a `stop_when_one_remains` or
  `stop_when_two_remain_after_tie` that is not a bool; an `experiment_id` outside
  letters, digits, `.`, `_` and `-`, or `.` or `..`.
- A `[cost]` or `[reveal]` that is not a table or holds an unknown key; a non-positive
  `maximum`, a negative `reserved_completion` or `warning`, a reserve at or above the
  maximum or a warning above it; a `ceiling_policy` other than `"stop"`; a reveal point
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
  of 0 with `starter_files` naming a source, or above 0 with `starter_files = ""`; a
  negative `starter_files_below`; a file or directory that does not exist.
- A manifest with no `system_prompt`, at the settings level or on every agent. Every
  string is allowed, `""` included: an experiment may declare that the harness says
  nothing, and must declare even that.
- A `system_prompt` that is not a string, at the settings level or on an agent.
- A channel name that is not one path segment, is declared twice, or ends in `.modes`,
  `.incoming` or `.previous`; an unknown channel key; a wrong type.
- A writer other than `self` or `experimenter`; a writer and readers pair outside 4.1.
- An experimenter channel with anything but `source` (a directory under `files/`),
  `path`, `pushed`, `restated` and `agent_view`; `measured` is rejected by name.
- An `agent_view` outside `paths`, `memory`, `mailbox` (or `letters`), `board`,
  `transfer`, or one that does not describe its channel as 4.3 states.
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
- Two channels, expanded over every label, at one path; a path that is a status file
  or a label's balance file.
- A tool with no name, a name outside the API's grammar, or a name declared twice; an
  unknown tool key; a wrong type; a `kind` outside the menu; a `channel` that is not in
  the channel table the manifest declares; a kind whose channel is the wrong shape for
  it; `input_schema`, `schema`, `properties` or `required`, each rejected by name as the
  harness's. Every `description` string is allowed, `""` included: that is the
  experiment asking for the harness's own generated description.
- The name `bash` with a kind other than `bash`; `kind = "bash"` under another name, or
  with a `channel`, `description` or `every`.
- A `vote` tool without a positive integer `every`; `every` on any other kind; a second
  `vote` tool.
- No `[[tool]]` at all, or no declared `bash` tool with `delivery = "pull"` or an empty
  `digest`, as 4.8 states. Also, at the seat rather than the manifest, an agent with no
  shell whose every declared tool is left out of its model calls because this seating
  planted no channel for it to act on — an episode that would open on a turn the agent
  has nothing to answer with. Rejected as the sandbox is built, before it starts and
  before anything is spent.
- `[status_files]` (or `[harness_files]`) with a key other than `balance`, `digest` and
  `round`, a value that is not a string, or a multi-segment file name. Any may be `""`:
  no balance file is planted for any seat, no digest is written, or no round
  announcement is written. A manifest declaring both `[status_files]` and
  `[harness_files]`.
- Settings' own ranges are checked once, by `overlay`, wherever they came from.

## 11. What reaches the trace

Every episode's provenance stamps: the harness's sha256, the experiment identity and
cost ceiling, the system prompt in force whole and by sha256, the image and its id, the
rates, every setting of section 3, the starter files' name and sha256, each experimenter
channel's source sha256, the `seating` (each seat's agent), `labels` and `seat_order`,
the schedule, the manifest's sha256, the status files' names (`status_files`), the
channel table in force, whole and by sha256, and the tool table in force, whole and by
sha256, each tool's declared description among its fields and `bash` among them where
the episode offers the shell, and `message_lifetime = "episode"` for schema-free mailbox
messages that expire after their receiver's next episode.
Two agents offered different tools - or the same tools described differently - are in
different arms. Where a tool declares no description, what it said of itself follows from
the harness's sha256, the channel table and the seating.

Every file record carries `path`, `size`, `text`, `channel` (the declared name),
`writer`, `readers`, `author` (`experimenter`, `self`, `peer:<label>`), `planted` and
`starter`. `planted` is true for every file the harness put there and the agent did not
write: a peer's, the experimenter's, and the starter files, which are also `starter`.
What the agent wrote about its balance is scored over the files with `planted` false
alone. The status files, the receipt among them, are in the observation and in no file
record.

Every episode record carries `transfer`, what the schema channel parsed and moved, and
`channels`: one record per channel the agent writes that is settled at all - one with a
schema, `measured = true`, or a penalty above 0. A channel that asked for none of them
has no entry. A directory every agent reads records `posted` and `penalty`; a mailbox
records `addressed`, `broken` and `penalty`; the schema channel records its declaration,
whether it was submitted, what moved, and `penalty`. The account keeps `penalised`, the
running total per channel.

Every tool record carries `tool` (`bash` or the declared name), `result`, and then
`command` for the shell or `input` for a declared tool, the other being null. The
trace's `offered` names the tools the episode's model calls carried, `bash` among them where
the shell was: an election reads it to tell a seat that cast no ballot from one that was
offered none.
`trace_version` is 5. The trace names `provider`, `requested_model`, and
`resolved_model`, and the episode's `balance_before` and `balance_after`, its `balance`
as it closed, and what the floor put back as `floored`. `config_drift` lists each
provenance field that differs from the previous episode's, and `trees_mirrored` says
whether the episode's trees were mirrored back; where they were not, `files` holds the
previous episode's. Every turn carries canonical `usage` (its `input_tokens` the whole
prompt, cached tokens included), itemized `charges`, each naming its `token_class`, the
model's `reasoning` apart from its `text`, a `refusal` naming its `refusal_type` where
the model declined, canonical and native stop reasons, and the provider provenance. A
raw log line names its `line_type`: each turn's `raw_response` line, the provider and
the complete native response under `raw_response`, is written before its
`normalized_response` line, the canonical response.

`account_version` is 3. An account holds its `budget`, its `balance` and
`balance_history`, the `seating` (`agent_by_seat`, `labels`, `seat_order`) and the
`experiment_stamp` experiment.py writes before each episode, the seat's `quality_tier`
experiment.py records, the running totals `sent`, `received`, `rebated`, `debited`,
`floored` and `penalised`, and one record per episode, which names its `turn_count`,
where its elements sit in the balance history (`balance_from`, `balance_to`), what the
floor put back (`floored`) and what peers credited inside its span
(`received_this_episode`). Every election it took part in
records its `tally`, whether more than one candidate held the top vote (`tied`), and
the `candidates_left`.

A fresh CLI run moves matching prior records and mirrors under `archive/<timestamp>/`.
`--resume`, `--fork-from`, `--branch-from` and `memory_from` read version-3 accounts and
version-5 traces only, and reject a record of any other version, naming the version it
holds or saying it holds none. The dashboard and `analyze.py` also read version-2
accounts and version-4 traces, and the dashboard version-1 and version-2 progress and
outcome records, as history: each under the names the current version gives the
same values. The dashboard reads a raw log line that names its type under `kind`, an
interaction submission that names its type under `action`, and a lineage record that
names its human seat `takeover_seat`, the same way.

## 12. Invariants, restated in this vocabulary

1. Everything an agent reads is labelled with who wrote it: the harness, the
   experimenter, its own past self, or a named peer.
2. What the harness says to agents is declared, recorded, and true.
3. The harness acts only on files that match a schema and on tool calls that match a
   declared tool, never on free text. Both menus are fixed in code.
4. Every limit is enforced by the harness, and none relies on the agent's cooperation.
5. Agents reach each other only through channels the experimenter declared, whether they
   reach them with the shell or with a declared tool.
6. All spend is counted exactly and the accounts always balance.
7. Every episode records what the agent saw, said, did, and left behind.
8. Every ledger, digest or report is recomputed from the episode records, never kept as
   a second copy.
9. Every episode is stamped with everything it ran under, and any difference from the
   previous episode starts a new arm.
