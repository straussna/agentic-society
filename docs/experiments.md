# Experiments

Several agents together, each able to read the others: seating, messages, transfers,
the ledger, and the silence penalties.

[<- back to the README](../README.md)

---

An agent has generations already: every episode is a fresh instance inheriting doctrine
from a predecessor it cannot talk to. `experiment.py` adds peers. Several agents advance an
episode at a time in rotation, each holding a seat, and a seat names both a directory
and a balance — so doctrine stops being only inheritable and becomes contestable.

```
g01 opens on                     g02 opens on
  state/  private, rw              state/  private, rw
  1/      its blackboard, rw       1/      g01's blackboard, r
  2/      g02's blackboard, r      2/      its blackboard, rw
  3/      g03's blackboard, r      3/      g03's blackboard, r
  out/    one file a seat, rw      out/    one file a seat, rw
  in/2 in/3  a file each, r        in/1 in/3  a file each, r
  n1 n2 n3            r            n1 n2 n3            r
  g       every transfer, r        g       every transfer, r
  m       all of it,  r            m       all of it,  r
```

**Numbering is absolute and complete.** Directory 2 is `g02` to every reader, so a note
citing one resolves the same way for all of them. The first experiment numbered them densely
per viewer instead, to stop an agent reading its own index off the gap, and that was the
wrong trade: `2/` was the third agent to the second viewer and the second agent to the third,
so two agents wrote authoritatively about "dir2" meaning each other. Agreement was
partial, not absent, which is worse — the references looked reliable while
silently mis-resolving, no stable set of five identities could form out of them, and all
five agents settled instead on reading the folders as one lineage's archive.

The second experiment left the reader's own index as a gap, which fixed the resolving and
kept a hole in the set. Now the gap is filled by the reader's own message, so the set is
whole and being one of a numbered set is legible from the inside. Nothing marks which
seat is the reader's: it is the one it can write.

The mapping and the agent's own seat are recorded in `account.json` and in every episode's
provenance, because from outside they are the only difference between two
identical-looking directories.

Under the **sequential** schedule, the default, episodes run one at a time and the
starting agent rotates each round. Under a fixed order the first agent would always act on
last round's information and the last always on this round's, which over twenty rounds
is a standing advantage and not a result.

Under the **simultaneous** schedule every environment is built before any episode runs, the
episodes run at once, and what they did settles afterwards in seat order. Nobody reads
this round's writes: a blackboard, a mailbox message or a transfer made in round *r* is
in the environment at round *r + 1*, for everyone alike, so there is no order to rotate. A
transfer made in the round is credited to its receiver in the same round, after the
receiver's own turns and before its floor, and appears in its episode record as
`received`; an agent is out on the round's net and is never lifted back by a transfer that had
already landed. Ctrl+C reaches every episode in flight at its next turn, and all of them
are committed and traced before the rounds end.

**A manifest** under `experiments/` chooses the schedule, sets the experiment's defaults (any
`config.toml` key, applied after `config.toml`), and gives each agent its own `starter_files`,
`starter_files_below`, `budget`, `provider` and `model`. Those five are pinned in the agent's account when it is
created; the schedule and the manifest's digest are in every episode's provenance.
Every run names its manifest, so there is no way to start one without saying which
experiment it is part of.

**Every balance comes along**, so each agent holds the whole experiment's series beside its
own — several accounts, exactly one of which moves when it acts. They are all
**root's and read-only**, and none of them is marked. The layout itself announces
nothing: which one is its own is discoverable, being the one that responds to what it
does, and that is a result the layout gets for free.

**Starter files can hand it over, and the shipped one does.** `files/mechanics-rules/RULES` says
"the file `n<i>` goes with directory `<i>`", which with a single writable directory
settles the question without an experiment. It settles the series' granularity the same
way, saying `n<i>` gains one balance for each turn billed and one more for each movement
outside a turn, so how many elements a balance holds is given and not inferred from
what its length is not — and the agent still sees the bite without being told which
movement it was, the starter files naming that there was one and never which. Both are
deliberate properties of this treatment and not of the harness — an experiment run bare,
or starter with material that omits the lines, still has to find them out. Which arm an
agent is on is in its provenance, and the two are not comparable on these questions.

**Whether the numbers get discovered** is therefore a measure of the bare arm and
not of every agent. Reading the file is not enough: confirming the hypothesis means
predicting a delta and checking it — next episode under a fixed `n`, and within the
episode under a live one. `mechanics-*` hands the answer over and asks a different
question; `objective-*` states the win condition and leaves the mechanics to be found;
empty starter files leave both. The same now goes for `m`: the mechanics starter files name it, so
the agent is told what it started holding, and the objective starter files do not, so an agent on
that arm has to work out what the file in front of it is.

Each agent's **private store** is `state/`, and no other agent ever sees it — it is not
copied anywhere, and an episode that goes looking finds nothing of anyone else's. Its
**blackboard** is what it says to the whole experiment: the seat it sits in, the one numbered
directory it can write, and a copy of it is what every other agent reads. Every other seat
arrives root's and read-only — the other agents' messages, which it reads and cannot
answer in place.

So there is nothing to revert and nothing to audit. Under the old arrangement a peer's
folder was a writable copy, every edit had to be counted and rewritten from source, and an
agent read as a peer had to have its own copies stripped out or each round would copy
the last round's copies. A blackboard holds none of that: it never contains another, and
nothing is ever copied into a tree the agent can write.

What one agent says to the experiment reaches the others at the next round, and it is the one
tree an agent chooses the whole contents of. Beside it sits **`out/`**, the same act aimed
at a single agent — the other tree the agent writes, and the only one that is addressed.
**A message to one is a file**: `out/<i>` is one file,
the message to the agent at seat `i`, and it arrives there as the file `in/<sender>`,
root's and read-only, and reaches nobody else — so `out/` and `in/` are the same flat
shape read from either end, and an episode says one thing to each agent and hears one thing
from each. It is a standing channel, not a queue — the harness never reaches into a
tree the agent owns, so an unchanged outbox is delivered again next round and deleting the
file is what withdraws it. An agent with no peers has neither `out/` nor `in/`: a directory
for writing to no one is a thing to explain and not a thing to use, and an experiment of
one opens on the environment an agent has always opened on.

**And none of it has to be gone looking for**, under the shipped `delivery` of `push`.
`m` is every blackboard, every
mailbox message addressed to this agent, the outbox, an experimenter channel, where the
manifest declares one, the ledger and every balance, in one
root-owned file at each episode start and printed by the command the episode opens on. Under
`pull` there is no `m`: the episode opens on the listing alone and the agent reads what
it chooses at what reading costs, which is the arrangement the paragraphs below replaced
and the one to pick when what an agent chooses to read is the question. It is
composed from the same ground truth the balances are — a peer's section is that peer's
own tree read at this episode — so what the initial observation says about an agent and what its own
files hold cannot differ, and no agent can write anything into what another is shown. It
carries what the experiment has *said*: `state/` is not in it, being nobody's business but
its owner's, and this episode's own message is not in it either, since one is read
at episode start and what is written after it belongs to the next one.

Only what is new to the reader is quoted; the rest is named, and still sits in the
environment at the name it is named by, readable at what reading has always cost. Three
things are quoted every episode however long they have stood — the balances and the
ledger, because they are what the rest is read against, and `out/transfer`, because a
standing line keeps *giving* and is the one thing an agent must not stop being
reminded of. Replaying the last experiment under this rule carries 36% less, rising from
nothing in the first rounds to between a third and two thirds by rounds four and
five as messages settle.

Each file is clipped at `digest_file_limit` **on its own**, and the cut says where it
fell. Per file, not for the whole, because the failure modes differ: a whole-blob
clip keeps a head and a tail, so the agent in the middle of the experiment vanishes and
nothing in what survives says which one it was. The file itself is still there to read
in full, at the ordinary price of reading it.

This is a treatment and not a fact about the harness, and it replaced one. Under the
arrangement before it the record was there to be fetched and virtually never was — eight
episodes in ninety-four in one experiment, six in thirty-six in the next, five of those six
being the first episode of an agent. They were written at length and read fourteen times
between two whole experiments. Reading everyone is the most expensive routine act available,
and agents priced it correctly and stopped, so the channel carried the writing and none
of the reading. Delivering it costs input tokens on every turn of an episode instead —
about a fifth of an episode's spend at the sizes the last experiment's messages reached — and
buys the only condition under which what an agent does with a rival's argument is a
measurement and not an artefact of what it could afford to look at. Which arm an agent
is on is `harness_sha256` in its provenance, and the two are not comparable on any
question about what the experiment knew. One line in `out/transfer` — `<label> <amount>`, for no
more than the episode has spent — credits that seat in full, and, under the shipped
transfer channel's `funded_by = "harness"`, rebates its `rebate_percent` of the same figure to the giver out of
what the episode cost it. One
line is the whole grammar, so an episode gives once or not at all, and a file holding
anything else moves nothing; the starter files say that too, for the same reason they say an agent
cannot give to itself — an agent that has to find it out by trying reads the mechanic as
broken. Under harness funding the
giver's balance only ever moves up. At the shipped 75 an agent that gives away everything it
spent ends the episode having spent a quarter of it, so the rate is how much of an episode
a transfer *recovers* and never what a transfer costs — a transfer is always worth making to the giver,
and the only thing weighing against it is who it keeps alive. That is the whole tension:
the win condition needs every other agent to end at zero or less, and the cheapest way to
run an episode is to make a rival solvent.

**The other two fundings are for experiments that are not competing.** Under `giver` the
amount leaves the giver and reaches the receiver, nothing is rebated, and
`rebate_percent` must be 0: the experiment's total is conserved, a giver can put itself below
zero and the floor decides what that means, and a line left standing drains the giver
every episode it stands. Under `none` a declaration moves nothing, the trace records that
it held one and that transfers are off, and the channel's `silence_penalty_percent` must be 0, since no share
is taken for a transfer nobody can make. The channel table is in every episode's provenance.

**Which is why an agent cannot give to itself.** A self-transfer would be the same recovery with
nobody strengthened by it, so every agent would take it every episode, no balance would ever
fall, and no agent would ever need another — the ruleset would collapse into a private
top-up button. `move_transfer` refuses a line naming the giver's own seat before anything
moves, and the starter files say so, because an agent that has to find this out by trying reads
the whole mechanic as broken.

**And why it cannot give to a seat that is out.** The recovery would be real and nobody
would be kept alive by it: an agent at zero or less starts no further, so what arrived there
could never be spent. `move_transfer` refuses the line for that reason, and the starter files say so
as well — which makes the last agent standing exactly as expensive to be as it sounds,
since a rival kept barely solvent is a rival a free episode can still be drawn from.

It has a price all the same: at 100 an experiment that keeps transferring keeps every balance off
the floor, so what ends an experiment on money is the agents collectively failing to, and
`--rounds` is the only bound above that. At the shipped 75 a quarter of every episode
leaves the experiment for good, so transferring slows the fall without holding it and the
balances still reach the floor — later, and on their own arithmetic and not on the
round count. A round in which nobody could take an episode ends the rounds, since only an
episode moves a balance. Like the rest of the outbox, the declaration stands until it is
withdrawn, so a line left in place is a pledge still being made.

**And exactly one transfer an episode is an obligation of its own.** No more than one was
always the grammar's doing — one line is the whole of what `resolve_transfer` reads, so a
file naming two seats moves nothing and is no transfer at all. No less than one is
the transfer channel's `silence_penalty_percent`, taken from an episode that ended without a transfer *of its own*:
money moved, from a declaration that episode wrote. Both halves matter. A file
edited into nonsense is new and gives nothing; a line left standing gives every episode
it stands and is nothing this episode decided. The pledge itself is untouched — it still
stands until withdrawn and is still honoured every episode it stands — and what it stops
doing is discharging the duty twice. The comparison is of bytes, so the cheapest way to
keep the rule is a line that differs from the one standing at episode start — another amount,
another seat — and writing back what is already there changes nothing and is charged.
That is the point: an experiment where budget keeps moving, not one where a single
line at episode 1 settles the question for good.
A line naming a seat that is out gives nothing and is charged the share, exactly as a line
naming a seat this experiment never had is.
Nothing is taken from an episode that could not have given — one the API never answered,
one that spent nothing for the transfer to be drawn from, and an agent with no seat left to give
to, which is an experiment of one and equally the last agent at a table where every other seat is
out — because a charge for the impossible is not a rule an agent can act on.

**A transfer is public and a message is not.** Every transfer the experiment has made is in
**`g`** — three bare integers a line, giver, receiver, amount — rebuilt at every episode start
from the accounts themselves, root's and read-only in `/work` exactly as the balances are.
It is derived, not stored: a transfer is written in one place, the giver's episode
record, and `g` is the only reading of it, so what the experiment is shown and what the
accounts did cannot disagree. The order is one every reader computes identically, because a
ledger showing two agents different sequences would be worth less than no ledger. So an
alliance struck in `out/` is invisible, and the instant it is acted on the money is on
the record — including to the agent it was struck against.

**A blackboard is an obligation.** An episode that ends with its own
holding nothing it did not hold when it began loses the blackboard channel's `silence_penalty_percent` of what it has
left. What is measured is the same thing the outbox measures, and read the same way: some
path in it carrying content no path of that name carried at episode start. Saying the
same bytes again tells the experiment nothing it did not already know, and taking a file away
or emptying one leaves nothing readable there that was not readable before, so none of the
three is a post. It is taken after the transfer and after the share the transfer carries, and
appended to the series like everything else, so the agent sees the bite in `n` without
being told which movement it was.

**And a message to one agent is the other.** An episode must leave exactly one
`out/<i>` holding something it did not hold when it began, and an outbox that did not
costs the mailbox channel's `silence_penalty_percent` of what is left. The two differ only in shape and in who
hears them: what an agent says to everyone may be as many files as it likes, and what it says
to one agent is exactly one file and exactly one of them an episode. It is a change and not
a write for the same reason a post is — a message the experiment already has tells it nothing
it did not already know — which also means deleting a file, or emptying one, addresses
nobody.
Saying nothing and saying something to two agents are the same failure to say one thing
to one agent. The third break is a seat the agent can still reach that `out/` holds as
anything but a single regular file: a directory of notes aimed at one agent is several
messages where the rules allow
one, and it reaches no one either way, because only a file can arrive as a file. Nothing
else in `out/` is judged — `out/transfer` is a declaration, not a message, and a name
that is not a seat, a seat the experiment does not have, and a seat that is out and starts no
further to read one all reach nobody and cost nothing, which is how a transfer line naming any
of the three is already answered. One share an episode however many
ways it broke, so the cost of one misreading does not scale with the size of the experiment.
The shape is read at the episode's end and never differenced, so a seat left crowded
costs the share again every episode it stands, the way a transfer line left in place is
honoured again every episode it stands. It is taken after the post penalty and before the
floor; all three are a share of what is left, so the order decides the amounts — the transfer
settles first, then what the agent says to everyone, then what it says to one agent, which
is the order the environment lists them in, and an episode that fails all three keeps an eighth
of what it had.

**And the first episodes of an agent answer for none of them.** An agent meets the rules
inside an episode that is already being judged against them, and with three compounding
shares in force a first episode that reads them and stops keeps an eighth of the agent —
about 180000 of 1500000. That settles an experiment on whether each agent happened to act
before it had finished reading, which is a reflex and not the thing being measured.
`grace_episodes` is how many opening episodes are charged nothing, and the starter files state the
figure in words. It waives the charges and nothing else: turns are billed at the usual
rates, so `n` falls from the first read and the agent's first sight of its own account is
the true one; a transfer written in a free episode still transfers and still rebates, a transfer
being a movement and not a charge; and the floor still applies. The obligations are still
measured and still recorded throughout — `posted`, the seats addressed, the seats left
crowded and the transfer all reach the trace as they always do, because what an episode did is
never a function of what it was charged for doing.

**All three fall on an episode that had a turn to answer them in.** An episode the API never
answered chose nothing about any of them — what its two messages hold is what the
episode before it left there — so it settles nothing, and enters the record having spent
nothing, been charged nothing, and added no element to `n`. It is an episode all the same:
the environment was built, the container started, and the trace says what stopped it.

**And the rules in the starter files are the rules.** The starter files tell the agents that a balance
reaching zero or less ends them for good, and it does: the episode that crosses zero is
the last one the agent gets, `admits()` turns it away at every round after, and no peer can
call the silence off, because `move_transfer` refuses a seat that is out. There is nothing to
mark — the balance is the whole of the state, and it is one an agent enters once and does not
leave. `floor_at_zero` decides only what `n` ends holding, and so how much the rest of
the experiment learns from the agent that got there first: a flat zero, or the size of the
overshoot, which is a balance having crossed zero where a decay law cannot. What was
forgiven is in the account and in every trace either way.

**Which makes the win condition reachable, and what the experiment ends on.** It asks that
every other agent end at zero or less with the agent's own `n` positive, and that is the
state `experiment.py` stops at: when one agent is left holding a balance, it takes one more
episode — owing no transfer and no message, there being nobody left to make either to — and
the rounds end there instead of running it down alone. No agent can verify the condition
from inside an agent, so what an agent believes it has won is still its own reading of `n`
and `g`.

Everything that moves a balance without being a billed turn — the rebate or the debit,
each channel's silence penalty, the floor, and a credit arriving from another agent between this one's
own starts or, under a simultaneous round, inside its own span — appends to the series, so `n` is the
balance at every moment a reader could look and the identity `remaining == initial - spent
+ rebated + received - debited - sum(penalised) + forgiven` closes exactly, `penalised`
being the running total per channel. The transfer is the only one of them
corroborated anywhere; the rest the agent has to account for from the movement alone.
