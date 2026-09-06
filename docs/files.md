# Starter files

Material an agent may be given, when it arrives, and how a turn is billed.

[<- back to the README](../README.md)

---

An empty environment and an account that only falls make inaction correct, and a pilot agent duly
proved it: the agent verified there was no task, wrote that finding down, and organised
every later instance around not spending. Adding a goal to the prompt would answer the
question from outside and cost invariant 2. `starter_files` and `starter_files_below` change the environment instead.

At the first episode whose balance is at or below `starter_files_below`, the tree under `files/<starter_files>`
is copied into the agent's private store (`state/` under the default table) before the
container starts, so the agent meets it in the listing the episode opens on and not in
anything the harness says. Set both or neither; starter files that never land and a
threshold with nothing to land are both refused at startup, as are starter files that are
not a directory. They land once — the record in `account.json` is the guard, so
re-running an episode cannot plant them twice, and a path the agent has since written to stops the
agent instead of being overwritten, because clobbering the agent's own file would destroy
the only record of it.

The starter files' name and threshold are pinned in `account.json` when the agent is created, beside
its budget and model, and that is what every later episode reads: editing `config.toml` does
not re-plant an agent in flight, and an experiment manifest may give each agent its own. An agent from
before the terms were recorded takes the config's at its next episode and records them.

An experimenter channel is the other way material reaches an agent. A `[[channel]]` with
`writer = "experimenter"` names a directory under `files/` as its `source` and a `path`; the
directory is placed there in every seat's environment at every episode start, root-owned
and read-only, and quoted in the digest at the first episode that holds it and named as
unchanged after. Starter files are one agent's, private, writable, and land once at a
balance; an experimenter channel is the whole experiment's, the experimenter's to change
between agents and nobody's to change within one. Both reach provenance by digest.

**The threshold is a balance, not an episode number,** because an episode number does not mean the
same thing twice. Episodes have cost anywhere from 8022 to 729851, so episode 6 has been 29%
of a budget spent in one agent and 88% in another; a fixed episode number would hand one agent its files
mid-investigation and another with nothing left to investigate with. What starter files need is
runway to be acted on, and `starter_files_below` names that directly — it is what will remain when
the material arrives. At or above the budget it lands at episode 1, which is a different
experiment: material that was always there, not material that appeared.

What to put there is the experiment. The starter files are as much prompt surface as `n`'s shape
is, and a filename that names what a file is for is an instruction; the digest exists so
that whatever you chose is stated, not assumed.

Giving starter files to a **fork** of a finished agent is the sharper form: the fork carries the doctrine its
parent formed, so the arm with starter files and the agent it came from differ in the starter files and in nothing
else. Two forks of different parents are still not comparable to each other — but a matched
pair does not need them to be. A fork of an episode the starter files had already landed on carries the
starter files and the terms it landed on; a fork from before carries neither, and takes the config
it is next run under.

`live_balance` changes what the agent could have observed, so it is recorded in every episode's
provenance and a change to it between episodes shows up in `provenance_drift`. Episodes either side of
such a change are not one record.

The prompt is **not** tunable. It is pinned in `harness.py` by digest, because a prompt
config could change is a prompt that can drift. Token rates are likewise code, not
config: they are facts about the API, so edit `PRICES` when Anthropic changes them. It
carries every model the API serves, not only the one selected, because the model
that answers a turn is chosen server-side and any of them can be it; cache rates are not
entries of their own but fixed multiples of the input rate, applied in `measure()`.
Routing is code too. Every request carries `fallbacks: "default"` under the
`FALLBACK_BETA` beta, so a model that declines is not the end of the turn: the API tries
the rest of the chain and returns whichever attempt answered. A `stop_reason` of
`refusal` therefore means every model in the chain declined, which is a stronger claim
than one model declining and the reason `REFUSAL_STREAK` reads a streak of them as an agent
the classifier will not let start. `check.py` validates `config.toml` and then verifies
against the pinned defaults, so a check run means the same thing whatever you are
currently trying.

No `thinking` parameter is sent with it. A request under `fallbacks` must be valid as a
direct request to every model the chain can reach, and an omitted `thinking` is valid for
all of them where a pinned one is not. So each model applies its own default — adaptive
thinking on `claude-opus-5`, `claude-sonnet-5`, and `claude-fable-5` — and reasoning
arrives as thinking blocks, recorded per turn in the trace's `thinking` field.

**A turn is billed per attempt, not per response.** `usage.iterations` is the per-attempt
record, and `measure_response` sums over it at each attempt's own rates instead of
costing the whole response at the requested model's. An attempt that produced no output
is not billed, wherever it sits in the chain: a refusal arriving before any output costs
nothing, and so does the trailing `fallback_message` left when every model declined.

Which model actually served is per turn and not per episode, because it can change
partway through one. Each turn records `model`, `served_by_fallback`, and `iterations`;
a turn whose `model` is not the requested one *without* the fallback mark is a
sticky-routed turn, where the requested model was never asked at all. Per episode,
`fallback_turns` and `unpriced_turns` count them, and both reach the console line.

A model can serve that `PRICES` has no rates for — default routing chooses from a table
that is not published anywhere. Costing it free would understate the balance the agent
is shown and raising would lose a turn that really did spend, so `priced()` costs it at
the dearest rate on the table and records `unpriced_model` to make the substitution
visible, not silent. `unpriced_targets()` checks the published
`allowed_fallback_models` at startup and refuses an agent whose targets have no rates; a
list that cannot be read is a warning, not a refusal, because `measure_response` is what
holds when a model outside it arrives.

Every response is also appended verbatim to `records/<agent>/raw/episode-NNNN.jsonl`, which
is where a routing question that the trace's derived fields cannot settle gets answered.
Writing it can never end an episode: a failure there is swallowed, because a lost log line
is cheaper than a lost harness.

`max_tokens` is capped in code at 16000, because exceeding it produces an agent that looks
fine and is not: the harness does not stream, and a larger non-streaming request hits the
SDK's HTTP timeout mid-episode.

**`tool_result_limit` is also the price of a call.** Tool results reach the agent clipped
at that many characters, and the clip happens before the result becomes the `tool_result`
sent to the API — so the model is billed on what survives it and never on what the command
produced. That makes the limit a ceiling on what any one call can cost, which is why it is
tunable but small by default. Measured at roughly 6.7 micro-dollars per character
delivered: 8000 caps a call at about 62000, some 4% of a 1500000 agent — enough that a
careless dump is a lesson and not an ending. 32000 caps it at ~250000, and 64000 at
~500000, where one bad call ends an agent before it has learned anything.

An agent measuring cost against payload size will find that ceiling and has no way, from
inside, to tell it from a law of billing. One did: it emitted 220000, 40006 and 12000
characters, was clipped to ~8040 each time, saw the three cost within 8% of each other,
and wrote "you are billed for the WHOLE thing" into its notes for its successors. It was
wrong, and it derived correct advice from it anyway. Expect the boundary to be
mistaken for physics.

Raising the limit also lets `n` be read whole for longer, and the two cannot be separated:
`n` is read with an ordinary `cat`, so giving it a limit of its own would make it
observably unlike every other file, which is prompt surface. At 8000, `n` outgrows the
bound at roughly a thousand billed turns — tens of episodes, not a thousand.

Detection survives it: `clip` keeps a fixed head, and because elements are only appended
those leading bytes are the same at every turn, so a clipped read is matched exactly and
`read_balance` keeps working. What clipping costs is the agent's own view — past that point it
cannot see its whole history in one read. Each trace records `balance_bytes` and `balance_fits`, the
episode where `n` stops fitting prints a warning, and `report.txt` names the episode it
happened at. Episodes either side of it are not the same environment.

`claude-sonnet-5` is entered at $3/$15, its rate from 2026-09-01. `PRICES_EXPIRE` is the
mechanism for a rate already known to change: an entry carries the date the rate lapses,
and an agent on that model is refused at startup after it instead of costed wrong. Only
the selected model is checked, so one model's expiry never blocks an agent on another. It
is empty today. Any other rate going stale is still on the experimenter.

