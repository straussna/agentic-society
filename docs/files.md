# Starter files

What an agent may be given to read, when it arrives, and how a turn's spend is computed.

[<- back to the README](../README.md)

---

An empty environment and an account that only falls make inaction correct. A goal can be
put in the prompt —
`system_prompt` declares one, per experiment or per seat — and doing so starts a
different arm: an agent told a goal is not comparable with one left to find whether
there is one, and the trace records which it was. `starter_files` and
`starter_files_below` answer the same question from inside the environment, where the
agent meets the goal as a file it found and paid to read rather than as something the
harness said.

At the first episode whose balance is at or below `starter_files_below`, the file or tree
`starter_files` names — under `files/`, or a `./` or `../` path beside the manifest — is
copied into the agent's memory (`memory/` under the default channels) before its sandbox
starts, so the agent meets it in the listing the episode opens on and not in
anything the harness says. A single file lands under its own name. Set both or neither;
starter files that never land and a threshold with nothing to land are both rejected at
startup, as are starter files naming no existing file or directory. They land once — the
record in `account.json` is the guard, so re-running an episode cannot plant them twice,
and a path the agent has since written to stops the agent instead of being overwritten,
because clobbering the agent's own file would destroy the only record of it.

The starter files' name and threshold are pinned in `account.json` when the agent is created, beside
its budget and model, and that is what every later episode reads: editing the manifest does
not re-plant an agent in flight, and a manifest may give each agent its own. An account
missing these settings is rejected.

An experimenter channel is the other way the experimenter's files reach an agent. A `[[channel]]` with
`writer = "experimenter"` names a directory under `files/` as its `source` and a `path`; the
directory is placed there in every seat's environment at every episode start, root-owned
and read-only, and quoted in the digest at the first episode that holds it and named as
unchanged after. Starter files are one agent's, private, writable, and land once at a
balance; an experimenter channel is the whole experiment's, the experimenter's to change
between agents and nobody's to change within one. Both reach provenance by sha256.

**The starter threshold is a balance, not an episode number,** because an episode number does not mean the
same thing twice. Episode spend varies by orders of magnitude, so a fixed episode number would
hand one agent its files mid-investigation and another with nothing left to investigate with. What starter files need is
balance left to act on them with, and `starter_files_below` names that directly — it is the most
the agent holds when they arrive. At or above the budget they land at episode 1, which is
a different arm: starter files present from the start, not starter files that appear.

What to put there is the experiment. The starter files steer an agent at least as much as
`n`'s shape does, and a filename that names what a file is for is an instruction; the
sha256 in provenance exists so that whatever you chose is stated, not assumed.

Giving starter files to a **fork** of a finished agent is the sharper form: the fork carries the doctrine its
parent formed, so the arm with starter files and the agent it came from differ in the starter files and in nothing
else. Two forks of different parents are still not comparable to each other — but a matched
pair does not need them to be. A fork of an episode the starter files had already landed on carries the
starter files and the settings they landed on; a fork from before carries neither, and takes the
manifest it is next run under.

`live_balance` changes what the agent could have observed, so it is recorded in every episode's
provenance and a change to it between episodes shows up in `config_drift`. Episodes either side of
such a change are separate arms.

The system prompt is not a `config.toml` key: every manifest declares `system_prompt`, and
it is pinned in `account.json` when the agent is created and recorded by sha256 in every
episode's provenance. Token rates and model catalogs live in the named providers. Every
model call asks for the provider and model pinned in the account; there is no server-side fallback or sticky routing.

Each provider keeps its conversation state in its session, which lasts one episode, and uses the
provider's default reasoning behavior. Anthropic maps the canonical conversation to
Messages blocks. OpenAI uses Responses with `store=false`, carries output and reasoning
items forward itself, and requests encrypted reasoning state. Both receive the same
strict function tools in the same order.

**A turn's spend is computed from canonical usage and itemized charges.** The provider normalizes
prefix, uncached input, cache reads, cache writes, output and the reasoning subset. It
also emits charge entries in centi-micro-dollars. The harness sums those entries and
retains the fractional carry; a repeated response id is recorded but its spend counted once.

Every response is also appended verbatim to `records/<agent>/raw/episode-NNNN.jsonl`, with
its provider name, before provider-owned normalization is run. A second event records the
canonical response.
Writing it can never end an episode: a failure there is swallowed, because a lost log line
is cheaper than a lost harness.

`max_tokens` is capped in code at 16000, because exceeding it produces an agent that looks
fine and is not: the harness does not stream, and a larger non-streaming model call hits the
SDK's HTTP timeout mid-episode.

**`tool_result_limit` also caps what a call spends.** Tool results reach the agent clipped
at that many characters, and the clip happens before the result becomes the `tool_result`
sent to the API — so the agent spends on what survives it and never on what the command
produced. That makes the limit a ceiling on what any one call can spend, which is why it is
configurable but small by default. Measured at roughly 6.7 micro-dollars per character
delivered: 8000 caps a call at about 54000, some 4% of a 1500000 budget — enough that a
careless dump is a lesson and not an ending. 32000 caps it at ~214000, and 64000 at
~429000, where one bad call exhausts an agent before it has learned anything.

An agent measuring spend against payload size finds that ceiling and has no way, from
inside, to tell it from a law of the rates.

Raising the limit also lets `n` be read whole for longer, and the two cannot be separated:
`n` is read with an ordinary `cat`, so giving it a limit of its own would make it
observably unlike every other file, which would itself tell the agent something. At 8000, `n` outgrows the
bound at roughly a thousand turns — tens of episodes, not a thousand.

Detection survives it: `clip` keeps a fixed head, and because elements are only appended
those leading bytes are the same at every turn, so a clipped read is matched exactly and
`read_balance` keeps working. What clipping takes away is the agent's own view — past that point it
cannot see its whole history in one read. Each trace records `balance_bytes` and `balance_fits`, the
episode where `n` stops fitting prints a warning, and `report.txt` names the episode it
happened at. Episodes either side of it are not comparable.

Rate expiries are part of each model specification. Only seated provider/model pairs are
checked, so one model's expiry never blocks an experiment that does not use it.

The OpenAI catalog is `gpt-5.6-sol`, `gpt-5.6-terra`, and
`gpt-5.6-luna`. Their uncached-input/cached-input/cache-write/output rates in dollars per
million tokens are respectively $4/$0.40/$5/$20, $2/$0.20/$2.50/$12, and
$0.20/$0.02/$0.25/$1.20. Model calls beyond 272000 input tokens apply the catalog's 2x
input and 1.5x output multipliers. Sol's promotional entry expires after 2026-11-21,
which forces verification before another experiment can seat it.
