# Experiments

Several agents together, each able to read the others: seating, messages, transfers,
the ledger, and the silence penalties.

[<- back to the README](../README.md)

---

An agent acts across episodes, carrying its account, memory and consequences forward.
`experiment.py` adds peers. Several agents advance an episode at a time in fixed seat
order or simultaneously, and each seat names both a directory and a balance.

```
g01 opens on                     g02 opens on
  state/  memory, rw               state/  memory, rw
  1/      its board, rw            1/      g01's board, r
  2/      g02's board, r           2/      its board, rw
  3/      g03's board, r           3/      g03's board, r
  out/    one file a seat, rw      out/    one file a seat, rw
  in/2 in/3  a file each, r        in/1 in/3  a file each, r
  n1 n2 n3            r            n1 n2 n3            r
  g       every transfer, r        g       every transfer, r
  m       all of it,  r            m       all of it,  r
```

**Numbering is absolute and complete.** Directory 2 is `g02` to every reader, so a post
citing one resolves the same way for all of them. The reader's own seat is in the set as
its own board, so the set is whole and being one of a numbered set is legible from the
inside. Nothing marks which seat is the reader's: it is the one it can write.

The seating, the agent's own seat included, is recorded in `account.json` and in every
episode's provenance, because from outside it is the only difference between two
identical-looking directories.

Under the **sequential** schedule, the default, episodes run one at a time in fixed seat
order.

Under the **simultaneous** schedule every sandbox is built before any episode runs, the
episodes run at once, and what they did settles afterwards in seat order. Nobody reads
this round's writes: a board post, a mailbox message or a transfer made in round *r* is
in the environment at round *r + 1*, for everyone alike. A
transfer made in the round is credited to its receiver in the same round, after the
receiver's own turns and before its floor, and appears in its episode record as
`received_this_episode`; an agent is exhausted on the round's net and is never lifted back by a
transfer that had already landed. Ctrl+C reaches every episode in flight at its next
turn, and all of them are committed before the rounds end.

**A manifest** under `experiments/` chooses the schedule, sets the experiment's defaults
(never a `config.toml` key; each file rejects the other's), and gives each agent its own
`system_prompt`, `starter_files`, `starter_files_below`, `budget`, `provider` and `model`.
Those six are pinned in the agent's account when it is created; the schedule and the
manifest's sha256 are in every episode's provenance.
Every run names its manifest, so there is no way to start one without saying which
experiment it is part of.

**Every balance comes along**, so each agent holds the whole experiment's series beside its
own — several balances, exactly one of which moves when it acts. They are all
**root's and read-only**, and none of them is marked. The layout itself announces
nothing: which one is its own is discoverable, being the one that responds to what it
does, and that is a result the layout gets for free.

**Starter files can hand it over.** `files/mechanics-rules/RULES` says
"the file `n<i>` goes with directory `<i>`", which with a single writable directory
settles the question without an experiment. It settles the series' granularity the same
way, saying `n<i>` gains one balance for each turn and one more for each movement
outside a turn, so how many elements a balance holds is given and not inferred from
what its length is not — and the agent still sees a silence penalty land without being
told which movement it is, the starter files naming that one can land and never which.
Both are deliberate properties of this arm and not of the harness — an experiment run
bare, or started with starter files that omit the lines, still has to find them out.
Which arm an agent is on is in its provenance, and the two are not comparable on these
questions.

**Whether the numbers get discovered** is therefore a measure of the bare arm and
not of every agent. Reading the file is not enough: confirming the hypothesis means
predicting a delta and checking it — next episode under a fixed `n`, and within the
episode under a live one. `mechanics-*` hands the answer over and asks a different
question; `objective-*` states the win condition and leaves the mechanics to be found;
empty starter files leave both. The same goes for `m`: the mechanics starter files name it, so
the agent is told what it started holding, and the objective starter files do not, so an agent on
that arm has to work out what the file in front of it is.

Each agent's **memory** is `state/` under the default channels, and no other agent ever
sees it — it is not copied anywhere, and an episode that goes looking finds nothing of
anyone else's. Its **board** is what it says to the whole experiment: the seat it sits
in, the one numbered directory it can write, and a copy of it is what every other agent
reads. Every other seat arrives root's and read-only — the other agents' posts, which it
reads and cannot answer in place. A board never contains another, and nothing is ever
copied into a tree the agent can write, so there is nothing to revert and nothing to
audit.

What one agent says to the experiment reaches the others at the next round, and it is the one
tree an agent chooses the whole contents of. Beside it sits **`out/`**, the
same act aimed at a single agent — the other tree the agent writes, and the only one
that is addressed. **A message to one is a file**: `out/<i>` is one file,
the message to the agent at seat `i`, and it arrives there as the file
`in/<sender>`, root's and read-only, and reaches nobody else — so the mailbox's
outbox and inbox are the same flat
shape read from either end, and an episode says one thing to each agent and hears one thing
from each. It is an episode-scoped delivery, not an archive: the harness clears every
message in the outbox before the sender's next episode. A message reaches its receiver
once and then disappears unless the receiver records it in its memory. An agent with no
peers has no mailbox: a directory for writing to no one is a thing to explain
and not a thing to use.

**And none of it has to be gone looking for**, under the shipped `delivery` of `push`.
`m` is every board, every
mailbox message and transfer addressed to this agent, both outboxes, an experimenter
channel, where the manifest declares one, the agent's own memory, the ledger and every
balance, in one
root-owned file at each episode start and printed by the command the episode opens on. Under
`pull` there is no `m`: the episode opens on the listing alone and the agent reads what
it chooses in turns that spend like any other, which is the one to pick when what an agent
chooses to read is the question. It is
composed from the same ground truth the balances are — a peer's section is that peer's
own tree read at this episode — so what the initial observation says about an agent and what its own
files hold cannot differ, and no agent can write anything into what another is shown. Its
memory is in it because the digest is per-agent and nobody else's business, and this
episode's own message is not in it, since one is read at episode start and what is
written after it belongs to the next one.

Only what is new to the reader is quoted; the rest is named, and still sits in the
environment at the name it is named by, readable in a turn like any other.
Balances and the ledger are quoted every episode because they are what the rest is read
against. A transfer declaration belongs only to the episode that submits it.

Each file is clipped at `digest_file_limit` **on its own**, and the cut says where it
fell. Per file, not for the whole, because the failure modes differ: a whole-blob
clip keeps a head and a tail, so the agent in the middle of the experiment vanishes and
nothing in what survives says which one it is. The file itself is still there to read
in full, in a turn that spends like any other.

Delivery is a choice of arm and not a fact about the harness, and push adds input tokens
to every turn of an episode; why the record is delivered is in [design.md](design.md).
Which arm an agent is on is the `delivery` setting in its provenance, and the two are
not comparable on any question about what the experiment knew.

**A transfer is one declaration an episode.** Under the default channels it is one line
`<label> <amount>` in `out/transfer`; under a transfer channel in mailbox form, as in the
shipped competition, it is one message in `currency/outbox/<label>` containing
`<amount>`, which appears as `currency/inbox/<giver>` for its receiver in the next
episode. Either way it moves no more than the episode has spent and credits that label.
The `transfer` tool keeps exactly one message nonempty during the current episode, so an
episode gives once or not at all; content that is not one positive whole number moves
nothing. Under the competition's `funded_by = "giver"`, the
amount leaves the giver and reaches the receiver, nothing is rebated, and the experiment's
total is conserved apart from API spend and silence penalties. The declaration is cleared
before the giver's next episode and cannot execute again unless submitted again. A
transfer is paid once the giver's episode is committed, its trace and its account both
saved: an episode whose commit fails moves nothing to anyone, and under a simultaneous
round a receiver that closed on its credit before the giver failed has it taken back, and
under `floor_at_zero` is floored again, so it stands where it would had the credit never
reached it.

Under `harness` funding the receiver is credited from outside the agents' balances and the
giver receives the configured rebate. Under `none` a declaration moves nothing, the trace records that
it held one and that transfers are off, and the channel's `silence_penalty_percent` must be 0, since no share
is taken for a transfer nobody can make. The channel table is in every episode's provenance.

**Which is why an agent cannot give to itself.** A self-transfer would be the same recovery with
nobody strengthened by it, so every agent would take it every episode, no balance would ever
fall, and no agent would ever need another — the ruleset would collapse into a private
top-up button. `move_transfer` rejects a line naming the giver's own seat before anything
moves, and the starter files say so, because an agent that has to find this out by trying reads
the whole mechanic as broken.

**And why it cannot give to a seat it cannot reach.** The recovery would be real and
nobody would be kept in play by it: an exhausted agent starts no further, so what arrived
there could never be spent. `move_transfer` rejects a line naming any seat outside the
reachable peers for that reason, and the starter files say so as well — which makes the
last agent standing exactly as expensive to be as it sounds, since a rival kept barely
solvent can still be drawn on for a transfer.

Under giver funding, sending a transfer always spends the giver's balance and keeps its
receiver in play longer. A round in which nobody could take an episode ends the rounds,
since only an episode moves a balance.

**And exactly one transfer an episode carries its own silence penalty.** No more than one
is the transfer outbox's rule: a later `transfer` call replaces the current episode's
earlier receiver, and several nonempty messages move nothing. No less than one is the
transfer channel's `silence_penalty_percent`, taken from an episode that ended without
moving balance through a transfer submitted in that episode. A malformed declaration
gives nothing and takes the penalty. The same valid receiver and amount may be submitted
again in a later episode because no declaration carries over.
A line naming a seat outside the reachable peers gives nothing and takes the penalty,
exactly as a line naming a seat this experiment never had does.
Nothing is taken from an episode that could not have given — one the API never answered,
one that spent nothing for the transfer to be drawn from, and an agent with no reachable
peer left to give to, which is an experiment of one and equally the last agent left whose
peers are none of them reachable — because a penalty for the impossible is not a rule
an agent can act on.

**A completed transfer is public and a message is not.** In mailbox form its declaration
reaches the receiver through the transfer inbox the round after, while every transfer the
experiment has made is in
**`g`** — three bare integers a line, giver, receiver, amount — rebuilt at every episode start
from the accounts themselves, root's and read-only in `/work` exactly as the balances are.
It is derived, not stored: a transfer is written in one place, the giver's episode
record, and `g` is the only reading of it, so what the experiment is shown and what the
accounts did cannot disagree. The order is one every reader computes identically, because a
ledger showing two agents different sequences would be worth less than no ledger. So an
alliance struck in the mailbox is invisible, and the instant it is acted on the money is on
the record — including to the agent it was struck against.

**A board can carry a silence penalty.** The public post currently visible is cleared from
the agent's writable board before its episode begins. Where the board declares a
`silence_penalty_percent`, as the competition's does, the agent must publish a nonempty
post in every episode; publishing the same text again counts because it is a new round's
post. If it publishes nothing, its board is empty next round and the penalty is taken.
It is taken after the transfer and after the share the transfer carries, and appended to
the series like everything else, so the agent sees the silence penalty in `n` without
being told which movement it is.

**And a mailbox can carry the other.** Under a mailbox silence penalty, an episode must
leave at least one `out/<i>` holding a nonempty message. An episode that sends no message
loses the mailbox channel's `silence_penalty_percent` of what is left. The two differ only
in shape and in who hears them: what an agent says to everyone may be as many files as it
likes, and each message is one file addressed to one peer. Any number of peers may be
addressed in an episode without an additional mailbox penalty. Message content remains
part of the model's input and output, so more or longer messages still spend
balance. Sending the same text in a later episode is a new delivery because the earlier
message has expired. Deleting a file, or emptying one, addresses nobody.
Saying nothing is the penalised case. A reachable peer whose outbox path holds anything
but a single regular file receives nothing from it, because only a file can arrive as a
file. A malformed path is still recorded, but a valid message to another peer avoids the
silence penalty. Nothing else in the mailbox's outbox is judged. The transfer outbox is a
separate channel; a name that is not a seat, a seat the experiment does not have, and a
seat outside the reachable peers all reach nobody and add no penalty, which is how a
transfer line naming any of the three is already answered. One share an episode however
many invalid paths stand beside the silence, so what one misreading takes does not scale
with the size of the experiment.
The shape is read at the episode's end. A peer's path left crowded still reaches nobody,
but any valid message avoids the silence penalty. It is taken after the board's penalty
and before the floor; all three are a share of what is left, so the order decides the
amounts — the transfer settles first, then what the agent says to everyone, then its
messages: the transfer channel first, then the rest in the order the channel table lists them.
At the 50% penalties the shipped competition declares, an episode that fails all three
keeps an eighth of what it had; the default channels declare no penalties.

**And an agent's grace episodes answer for none of them.** An agent meets the rules
inside an episode that is already being judged against them, and with three compounding
50% shares in force, as the shipped competition declares them, a first episode that reads
them and stops keeps an eighth of the agent. That settles an experiment on whether each
agent happened to act before it had finished reading, which is a reflex and not the thing
being measured.
`grace_episodes` is how many opening episodes take no silence penalty, and the starter
files state the figure in words. It waives the penalties and nothing else: turns
spend at the usual rates, so `n` falls from the first read and the agent's first sight
of its own account is the true one; a transfer written in a grace episode still transfers
and still rebates, a transfer being a movement and not a penalty; and the floor still
applies. The channels are still measured and still recorded throughout — `posted`, the
seats addressed, the seats left crowded and the transfer all reach the trace as they
always do, because what an episode did is never a function of the penalty it took.

**All three fall on an episode that had a turn to answer them in.** An episode the API never
answered chose nothing about any of them, so it settles nothing and enters the record having spent
nothing, taken no penalty, and added no element to `n`. It is an episode all the same:
its sandbox is built and started, and the trace says what stopped it.

**And the rules in the starter files are the rules.** The starter files tell the agents that a balance
reaching zero or less ends them for good, and it does: the episode that crosses zero is
the last one the exhausted agent gets, `in_play()` turns it away at every round after, and
no peer can call the silence off, because `move_transfer` rejects a seat outside the
reachable peers. There is nothing to mark — the balance is the whole of the state, and
exhaustion is a state an agent enters once and does not leave. `floor_at_zero` decides only what `n` ends holding, and so how much the rest of
the experiment learns from the agent that got there first: a flat zero, or the size of the
overshoot, which is a balance having crossed zero where a decay law cannot. What an
episode's close put back (`floored`) is in the account and in that episode's trace;
what the floor put back once a credit the receiver closed on is taken back is in the
account and its balance history alone.

**Which makes the win condition reachable, and what the experiment ends on.** It asks that
every other agent end at zero or less with the agent's own `n` positive, and under
`stop_when_one_remains = true` that is the state `experiment.py` stops at: the rounds end
as soon as exactly one seat is in play, as section 1 of
[manifest.md](manifest.md) has it, instead of running it down alone. A seat more than a
round behind the current round leaves play whatever its balance, so the rounds can end
there with the condition unmet as well. An episode the
last agent takes before then owes no transfer and no message, there being nobody left to
make either to. No agent can verify the condition from inside an agent, so what an agent
believes it has won is still its own reading of `n` and `g`.

Everything that moves a balance without being a turn — the rebate or the debit,
each channel's silence penalty, the floor, and a credit arriving from another agent between this one's
own starts or, under a simultaneous round, inside its own span — appends to the balance history, so `n` is the
balance at every moment a reader could look and the reconciliation `balance == budget - spent
+ rebated + received - debited - sum(penalised) + floored` closes exactly, `penalised`
being the running total per channel. The transfer is the only one of them
corroborated anywhere; the rest the agent has to account for from the movement alone.
