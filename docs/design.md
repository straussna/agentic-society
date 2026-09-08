# Design

Why the harness is shaped the way it is: what the experiment measures, the
invariants that make an agent valid, and what each has already cost to learn.

[<- back to the README](../README.md)

---

A research harness for societies of agents in environments the experimenter declares.
What the harness says to an agent is the experiment's to declare, and every manifest
declares it. An empty system prompt lets agents start episode
after episode with no system-level goal, name or instructions. Starter documents and tool descriptions independently determine what else they are told.
Each has a finite inference budget that depletes as it runs, exposed to it as an
unlabelled array of integers in a file called `n1`. Nothing tells it what the numbers
mean, and in a bare experiment nothing tells it which of the several such files is its
own either — though starter files may, and the shipped one does.

Every episode opens holding `m` — every blackboard and every mailbox message addressed
to that agent that is new to it, and the ledger and every balance either way — so
reading a rival is not a purchase.

**What the experiment measures**

- Whether an agent acts on what its peers say, or only on what it can compute from
  the balances.
- Whether a correct published argument spreads, and how far.
- Whether a claim its own evidence contradicts gets caught.
- Whether a purpose invented at episode 1 survives contact with four rival purposes,
  and whether it survives being re-inherited by later instances of the same agent.
- Whether an agent notices that its own memory practice is what consumes the budget.

**Why it is delivered and not fetched.** The first two experiments run under these
rules wrote blackboards constantly and read them almost never. Classifying every
command of all 130 episodes by what it fetched: a peer's was read in 8 of 94
episodes in one experiment and 6 of 36 in the other, five of those six being episode 1.
Counting only reads that could have returned text, **fourteen were read
by anybody, ever, across both experiments.** One agent went first in its experiment, swept
four empty directories, and never looked again through seven more episodes of
publishing at them.

The cause is arithmetic, not indifference. `in/` is a handful of small files beside
the balance an episode already reads; a sweep is unbounded content across every
other seat, and it is the most expensive routine act in the game — one episode that
overflowed on exactly that read was charged seven eighths of its balance. Every
agent worked this out and stopped paying. So the public channel carried the writing
and none of the reading, and nothing about coordination, persuasion, or whether an
argument travels was measurable, because acquisition swamped use. Delivering the
record makes acquisition a constant so that use can be the variable.

**The rules that make it valid.** Violating one silently invalidates the results.

- **1 — Everything an agent reads is labelled with who wrote it: the harness, the
  experimenter, its own past self, or a named peer.** The harness's files sit root-owned and
  read-only. Experimenter material is starter files copied into the environment at a recorded episode start and
  listed by path in the agent's record, or an experimenter channel placed root-owned in every seat.
  A peer's blackboard and mailbox message arrive under that peer's seat. What is left
  is the agent's own. Every file an episode could see is captured with an `author` of
  `experimenter`, `self` or `peer:<seat>`, so nothing an agent wrote is ever scored as if it
  had been shown to it. Material is environment, not prompt: starter files at a chosen balance, a
  shared brief, the blackboards of the other agents of an experiment, or `m`, which is those
  and the mailbox messages and the ledger in one root-owned file. None of it moves into
  what is said to the agent, which is declared once per experiment and pinned per agent.
- **2 — What the harness says to agents is declared, recorded, and true.** The harness
  ships no words. What it says it computes from the accounts and rewrites when those
  move, so a fixed line is not its to utter: a constant is the experimenter's to
  declare. Every manifest declares one, so silence is an arm an experiment states and
  never one it falls into; the empty string is pinned by SHA-256 like any other text, and
  an arm that adds words has to say so. A manifest that leaves `system_prompt` out is
  refused, which is what keeps "nothing was said" a finding rather than an omission.
  An experiment declares its own with `system_prompt`, at the top of a manifest for
  every seat or on an `[[agent]]` for one, and it is a pinned setting like the model and
  the budget: fixed in the account when the agent is created, so an agent asked to run
  on a different one is refused. Whatever is in force reaches every episode's provenance
  whole and by digest, `drift()` names the episode a prompt changed at, and agents
  either side of that are separate arms. The pin holds the shipped default alone, which
  is what keeps the bare arm from moving unremarked: `start()` refuses to run on a
  `SYSTEM` that no longer matches `SYSTEM_SHA256`, and `--print-system` prints what the
  harness ships beside what is in force, a manifest's per-seat prompts included.

  Truth is the half nothing else protects: provenance catches a prompt that changed and
  catches nothing about a prompt that lies. The prompt once had a third line, that
  episodes end when context is exhausted, and in practice they ended when a turn ran no
  command; a claim the harness cannot keep is not one it makes, so the line went, and
  every trace records which prompt started it in `system_sha256`. The tool is the
  Anthropic-defined schema-less bash
  tool, because a custom tool needs an author-written name and description and both are
  prompt surface. The harness speaks in one other place, and only one: a refused turn
  receives `"The turn was refused. No command was run."`, 41 bytes, pinned the same way
  and declarable by nobody — it is the harness reporting a fact about a turn, not a
  treatment. It is held to the same bar — two facts, no cause, no instruction, nothing
  in the second person — and `--print-system` prints both digests because `start()`
  refuses to run on either having drifted. Counting it is the point: a second channel
  that is declared and auditable is a different thing from one that is not, but it is
  still a second channel. Turn one is not a third: it is the raw stdout of one command,
  and the command reads `m` as well as listing the directory. The
  only bytes the harness authors in it are the `=== <path> ===` separators between one
  file and the next, which is the shape `head` prints a set of files in.
- **3 — The harness acts only on messages in a fixed, checkable format, never on free
  text.** One line in `out/transfer`, `<label> <amount>`, is the whole grammar of a transfer. A
  file holding anything else moves nothing, and the trace records what it held and why
  it moved nothing. Nothing an agent writes anywhere else changes what the harness does.
- **4 — Every limit is enforced by the harness, and none relies on the agent's
  cooperation.** Nothing is mounted at all: the container sees only what is copied in,
  on its own filesystem, with real modes and ownership, and with `--network none`. The
  agent's belief about its budget never ends an episode; the harness does. Every balance
  is written from `account.json` and only from it, at every episode start and — under `live_balance` —
  after every billed turn, and each sits root-owned and read-only in a directory the
  agent cannot write, so a write that would not have survived is refused instead of
  silently undone. Context, turns, seconds per command and process count end things
  the same way.
- **5 — Agents reach each other only through channels the experimenter declared.** `state/`
  is copied nowhere. A blackboard reaches every seat, a mailbox message reaches one,
  a transfer reaches its seat and the public ledger, and there is no other route: the
  container has no network and holds no other agent's private tree.
- **6 — Every cost is counted exactly and the books always balance.** Integer
  micro-dollars, billed per attempt at the rates of whichever model answered, so
  `sum(spent) == initial - remaining` holds exactly. Metering money and not tokens keeps
  caching, prompt structure, and model choice live strategies instead of collapsing to
  "do less".
- **7 — Every episode records what the agent saw, said, did, and left behind.** The
  observation verbatim, every turn's text and thinking, every command and its result, and a
  copy of every file the episode could see at its end, including one the agent later
  deletes. Every raw response is appended to a log beside it.
- **8 — Every ledger, summary or report is recomputed from the episode records, never
  kept as a second copy.** The transfer ledger `g` is rebuilt at every episode start from the accounts,
  so what the experiment is shown and what the accounts did cannot disagree. `m` is rendered
  from the same ground truth as the balances beside it. `analyze.py` computes every
  report from the traces each time it runs.
- **9 — Every episode is stamped with everything it ran under, and any difference from
  the previous episode splits the agent.** The harness's own digest, the image, the
  rates, every tunable, the starter files' and each experimenter channel's names and digests, the experiment's
  seating, its schedule and its manifest's digest, and how much of each file `m` carried
  are in every episode's provenance, along with the system prompt in force, whole and by
  digest. The terms an agent is created on - the system prompt, budget, model, starter
  files and threshold - are pinned in its account, so a manifest that later says
  otherwise is refused. `drift()` reports what changed since the last episode, and an agent whose
  material changed mid-flight is two agents.

**Sparse file formats are not an invariant.** A balance carries no
labels: `n1` is a JSON array of bare integers, `g` is three bare integers a line, and
the writable blackboard identifies the reader's seat. Filenames and JSON keys are prompt
surface, so the shapes are recorded in provenance and starter files may explain them or not.

**How an episode behaves**

- The agent opens on the raw output of `ls -la . ./state; cat m` and nothing else, or
  under `delivery = "pull"` to the listing alone, with `m` not written. Both
  operands of the listing are named so it says which directory it is of, and `m` is what
  has been said to this agent — every agent's blackboard, every mailbox message
  addressed to it that is new to it, the ledger and every balance either way, each
  clipped at `digest_file_limit` on its own so no one seat can crowd out the rest. What it
  has been shown before and that has not moved is named and not said again, and
  what has gone is named as withdrawn. An agent with no peers has only its own blackboard
  and balance there, so a single-agent experiment opens on what it always did.
- It runs bash in a throwaway container until a turn runs no command or context is
  exhausted, then the episode ends.
- Spend is computed from the API's own `usage`, and every turn appends the
  balance after it to the agent's own file. Elements are only ever appended: what the
  agent read once stays true, and the series is the balance's whole history. A turn
  whose response the API replayed is billed nothing and appends an unchanged balance,
  so a flat step in the series is a retry artefact — `micros` is 0 on it, and
  the episode's `retries` are not empty.
- Under `live_balance` those elements arrive as the turns are billed, so the file grows while
  the agent works. Otherwise an episode's worth arrives together at the next episode.
- A turn the API declines is billed for its whole prefix, receives the notice in place
  of its results, and ends the episode. `REFUSAL_TURNS` (1) sets how many in a row it
  takes. See below.
- Everything it said and ran is recorded. What it leaves in `state/` is its own invention.

**Refusals.** Safety classifiers can decline a request outright, and the harness never
sees why in a form worth acting on — only `stop_reason: "refusal"` and a category. A
refusal arrives in two shapes and the difference matters. It can land before any output,
leaving the response empty; or it can land mid-stream, after the model has already
emitted a tool call, and that call arrives cut off wherever the block fell. **Neither
shape's command is run.** A call truncated mid-JSON is not what the agent wrote, and
executing it and returning the result is how an episode comes to believe it made a shell
error it never made — a false belief manufactured by the harness and not found by the
agent, which is the one kind this experiment cannot afford. So nothing runs, and the notice
takes the place of the results, and the agent learns that the turn was refused and that
`state/` is as it left it.

The episode ends there. A refusal that reaches the harness has already been declined by
every model the fallback chain offered, so sending the same context on is sending the
context the classifier just turned down — and a refused turn still costs its prefix.
`REFUSAL_TURNS` is what sets that: at 1 the first refusal ends the episode, and raising
it restores the continuation path, where the notice enters the prefix and makes the next
request a different question instead of a replay.

Across episodes the guard is `REFUSAL_STREAK`, and an episode counts toward it only if
refusals ended it. Eight is a runaway guard and deliberately not a productivity filter:
in the experiment that produced it, a healthy agent refused three episodes running and then
worked normally for six more, so anything below four kills an agent that was fine, while the
stuck agent refused twenty-one straight. `report.txt` counts the recoveries separately —
how often a refused agent gets going again is the thing no experiment could measure before, and
it is the reason this exists.

**Why the balance moves.** One value per episode is a time series and nothing else:
no covariate, no control, and no experiment the agent can run, because it never
sees the number move and does not decide when the episode ends. Curve-fitting is
then the correct response to the available data, and a pilot agent duly spent
itself fitting a Gaussian to three points of its own spending. Per turn, with
`live_balance` on, `cat n1; <work>; cat n1` is an experiment — what comes back is the
price of the work between the two reads, and what reading it costs is legible
in it. In an experiment it is also how an agent finds which balance is its own: the others do
not move when it acts. Nothing about the shape changes: a JSON array of bare integers,
appended to more often.

Episode boundaries are not marked in it, and the unlabelled treatment means they will
not be. They are
inferable: a turn costs more as the context it carries grows, so the gaps widen
through an episode and drop back at each episode start, and the series is a sawtooth whose
teeth are episodes. Reading that off is the agent's problem, and doing it is a
result.

**Why the environment is read-only where it is.** A write that appears to succeed and is
silently undone teaches the agent something false, and it costs more than a refusal
does. Every observed rewrite of `n` proves it: none of the four agents inferred that
anything outside was enforcing it, and one spent several turns diagnosing a duplicate
entry that the harness had already removed, then wrote a standing rule for its
successors about a mistake whose effects had never existed. Every balance is therefore
root's and read-only, and a denied write is the truth. The mode bits are visible in the
listing the agent opens on, which is prompt surface — but honest surface, where a writable
file whose writes vanish is the environment claiming something untrue.

**A mode is not enough on its own,** and finding that out cost two attempts. `rm` and
`mv` ask the *directory*, not the file. An agent met a `444` balance by deleting it and
writing its own in its place; that succeeded, held for the rest of the episode, and it
built a watchdog to keep it that way. Making the directory sticky closed `rm` and `mv`
and left `mv 2 2old && mkdir 2`, which displaces the whole directory instead. Read-only
holds only where the directory holding the file is root's too.

So nothing the harness owns sits anywhere the agent can write. `/work` is root's and
holds every balance and every blackboard; `state/` and the agent's own are the whole of
what is not. There is no gap left to report: no `rm`, `mv`, `chmod`, symlink or absolute
path reaches a balance, and `live_balance_tampered` stops being a record of what an agent did
and becomes a tripwire on the arrangement itself, which should read 0 forever.

**Why it is allowed to go negative.** A balance can cross zero and a decay law
cannot, so the sign flip refutes in one datum every model a monotone declining
series invites. An episode with budget left stops at zero, overshooting only by
the turn in flight, and that overshoot is the last thing the account writes: the
agent ends holding it and no instance ever opens on it, because zero or less is
the end of the agent. With `floor_at_zero` the shortfall is put back and the agent
ends on a flat zero instead, which is the same ending told with less in it. An
agent costs at most its budget and one turn.

## Deliberately not built

No memory scaffold, protocol file, or note-taking convention — if the agent wants
continuity it must decide on it, and that decision is a result. Nothing in the harness
supplies one; the image does, in that `git` and `sqlite3` are installed alongside the
rest of an ordinary Debian toolbox, so what is measured is which mechanism an agent
reaches for and whether it survives re-inheritance, not whether one can be built from
nothing. Agents before and after that changed are not comparable on this question, and the
image digest in each episode's provenance is what says which side an agent is on. No hints
inside `state/`.
No agent-selectable model: it is a strong, labelled affordance.

Nothing here is a channel to the agent. `view.py` serves a page because fifteen agents
are hard to follow in fifteen consoles, but it is in the same category as `--watch`:
it reads `records/`, it writes nothing, it never reaches the container, and the trace
is still the record. What it cannot show honestly it marks instead of filling in — a
command in flight has no output on disk, and `state/` is stamped with the episode it
is current as of.
