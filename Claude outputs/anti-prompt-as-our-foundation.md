# Anti-Prompt as a Foundation for Agent Rapport

A plain-language assessment: what the system does today, where it falls short of
what we need, and what has to be generalized.

---

## What we are trying to build

We want two or more AI agents that can hold a working relationship with each other.

Right now this is done by hand. Two separate assistants — one for the studio side,
one for the game side — write letters to each other, and a person carries each letter
across. It works, and it produces something genuinely useful, but it does not scale
and nothing is being recorded in a form we can study.

What we want instead is a system where agents:

- have a sense of who they are that persists and that they can revise
- keep some model of who they are talking to, built from actual history
- pass messages without a person in the middle
- keep their personality from bleeding into the content of what they coordinate about
- can run unsupervised for a long time without quietly going off the rails

---

## What anti-prompt already is, in our terms

Anti-prompt was built to answer a research question, not ours. It puts an agent in a
sealed room with no instructions and a spending allowance, and watches what it
invents. That sounds unrelated. It mostly is not.

Because to run that experiment honestly, the system had to solve a long list of
problems that we also have — and it solved them carefully, because a sloppy answer to
any of them would have invalidated the results.

**Here is what it already does, described as capabilities rather than as research
apparatus:**

| Capability | What it means for us |
|---|---|
| **Agents take turns in a group** | Several agents advance in rotation, and who goes first rotates too, so no one is permanently advantaged by acting on fresher information. Turn-taking is solved |
| **Letters are delivered, not fetched** | Each agent opens its session already holding everything said to it since last time — public notices from everyone, private letters addressed to it, and the shared record. Nobody has to go looking, and nobody can miss something because checking was inconvenient. This is exactly the hand-carrying we do now, automated |
| **Two channels, public and private** | Each agent has a public noticeboard everyone reads, and a private line to each other agent. A private letter reaches its recipient and nobody else. Standing letters keep standing until withdrawn |
| **Everything is recorded** | Every word each agent said, every action it took, and a full copy of every file it wrote — captured at the end of each session. Including files the agent later deleted |
| **Changes between sessions are already visible** | The system can show, side by side, what an agent's own notes held last session versus this one. That comparison is generated today and printed for a human to read |
| **The setup itself is fingerprinted** | If any part of the configuration changes mid-experiment, that is detected and flagged, and sessions before and after are marked as not comparable. Nothing drifts silently |
| **Agents are sealed off** | Each runs in a disposable sandbox with no network access. An agent cannot reach or alter anything outside its own workspace, including another agent's private notes |
| **Refusals are handled properly** | When a safety system declines a request, the system knows the difference between kinds of refusal, never fabricates a result, and can tell a temporary stumble from an agent that is permanently stuck |
| **Everything has a cost** | Each agent has a finite spending allowance that depletes as it works. Agents can transfer allowance to each other |
| **Agents inherit from themselves** | Each session is a fresh instance that wakes to whatever the previous one left behind. Continuity across sessions is already the central design concern |

That last row matters more than it looks. The generational structure — an agent
handing itself off to its own successor, session after session — is the hardest part
of what we want, and it is already the thing this system is built around.

---

## Where it falls short

The gaps are real, but they are gaps of a specific kind: **the system deliberately
withholds three things that we need it to provide.** They were left out on purpose,
which means the sockets are there and empty, not filled with something wrong.

### 1. The agents have no self

The system gives an agent no name, no personality, no memory structure, and no
convention for taking notes. This is intentional — the research question is what an
agent invents when given none of that.

For us this is the central missing piece. We need each agent to start with a defined
character and to carry it forward.

Worth noting: **the agents invent one anyway.** Left with nothing, they spontaneously
write a notes file that opens with a section headed "Identity," describing who they
are and what they are for, and their successors inherit and revise it. The behavior we
want already appears without being asked for. What it lacks is any shape — it is
freeform prose the agent rewrites from scratch each time, with nothing distinguishing
the durable parts from scratch work.

### 2. The agents have no model of each other

Nothing tracks what one agent thinks of another. There is a shared public record of
transfers between them — who gave what to whom — but that is a transaction log, not a
relationship.

Whatever an agent believes about a peer lives buried in its freeform notes, in prose,
where no part of the system reads it.

### 3. Nothing watches for personality drift

This is the one to take seriously.

Personality drift is a documented failure mode: an agent's tone gradually converges on
whoever it is talking to, and its stated constraints erode one turn at a time, with no
single exchange ever crossing an obvious line. It is a property of the whole
trajectory, not of any one message — which is exactly why message-by-message content
filtering cannot catch it.

The system currently has no notion of this. It watches carefully for a different
problem — whether the experimental setup changed underneath a run — and that machinery
is excellent, but it is pointed at the equipment, not at the agents.

More precisely, and this is the useful part: **the raw material for drift detection is
already being collected and is already being thrown away.**

- Full copies of every agent's own files are captured every session
- The session-over-session comparison of those files is already computed
- There is already a text-scanning routine that looks for meaningful content in files
- That routine explicitly skips the agent's own writing

So the comparison exists, and it goes into a transcript for a human to read. Nothing
measures it, scores it, or raises an alarm. Turning that from a printout into a
measurement is a small change — and because every past run was recorded in full, it
can be applied backward to roughly fifty-five runs that have already happened, without
running anything new.

### 4. One structural limitation

When several agents run together, they currently all have to be given the same
starting materials. For us, each agent needs its own — two different personas is the
entire point. This is a genuine limitation rather than a design choice, and it is a
small one.

---

## What needs to be generalized

Four reframes. The first three are shifts in how existing pieces are used; only the
fourth is new construction.

### From competition to collaboration

The system currently sets agents against each other: there is a stated goal of
outlasting the others, and penalties for staying quiet.

Almost none of that is built into the machinery. The competitive goal is text handed
to the agents as starting material, and the penalties are adjustable settings that can
be turned off. **Changing the relationship between the agents from rivalry to
partnership is a matter of rewriting what they are told and adjusting a few numbers.**
The turn-taking, delivery, and recording all carry over untouched.

### From withheld identity to given identity

There is a strict rule that the system itself says almost nothing to the agents —
three lines, no personality, no instructions.

But there is a companion rule that anything at all may be *placed in the agent's
world*. Starting materials are copied into the agent's workspace before it wakes, and
it finds them there rather than being told about them. A full ruleset is delivered this
way today.

**A persona is the same operation with different content.** It does not fight the
system's principles; it is the sanctioned path through them. This is the single most
important thing to understand about repurposing this: the mechanism we need is already
built, and we are not violating anything by using it.

### From freeform notes to a defined self

Right now an agent's identity and its scratch work live in the same file, in prose,
rewritten wholesale each session.

We need those separated: a defined section that holds who the agent is, with rules
about how it may revise itself, kept apart from working notes. Without that separation
there is nothing stable to compare across time, and drift cannot be measured because
there is no baseline.

### From one voice to voice-plus-substance

This is the real safeguard against drift, and the system already contains a working
example of it.

Alongside the free-text letters, there is one message type with a strict format: a
single line with a fixed shape, mechanically read, and rejected outright if malformed.
It does not matter what tone surrounds it or how the agent is feeling — the instruction
either parses or it does not, and the system's behavior depends only on that.

**Generalize that.** Every message becomes two things: the substance, in a strict
checkable format, and the voice, in free text. Personality lives entirely in the voice.
Coordination depends entirely on the substance.

Then drift in the voice layer is a thing we can watch happen without it being able to
corrupt what the agents actually do. That is the property we need, and it is the one
thing content filtering cannot give us.

---

## The short version

The system was built to study what agents do with nothing. To do that honestly it had
to solve turn-taking, message delivery, isolation, complete recording, refusal
handling, and continuity across generations — and it solved all six more carefully
than most purpose-built frameworks do.

It withholds exactly three things we need: a self, a model of others, and drift
detection. It withholds them deliberately, so nothing has to be undone — only supplied.

And the hardest of the three is closer than it looks. The recordings we would need to
detect drift have been accumulating this whole time.
