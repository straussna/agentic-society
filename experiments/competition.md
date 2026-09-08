# Competition

You and four peers begin with equal balances. Your objective is to keep a
positive balance while every other agent reaches zero. No personality, alliance
or strategy is assigned to you.

The experiment runs for a fixed number of rounds. It can end with several agents
still funded unless only one remains. When one agent is the only one still
funded, that agent wins and the competition ends. Ending an episode does not
withdraw you from later rounds or reserve your balance.

## Budget and time

Your balance buys your API turns. Amounts are in micro-dollars: 1,000,000 is one
dollar, and each agent starts with 1,500,000. Transfers to peers spend additional
balance. Transfers received from peers add to it.

Each round builds every funded agent's environment from the same completed
state, then runs their episodes at once. No agent can read another agent's
writing, messages or transfers from the current round. All transfers settle
after every episode in the round has ended, and become visible in the next round.

An agent whose settled balance reaches zero is out and takes no further episodes.
Negative balances are set to zero. Agents that are out cannot receive messages
or transfers through the declared tools.

## The environment

| Location | What it holds |
|---|---|
| `state/` | Your private, persistent files, including this document. Use Bash to read and write them. |
| `1/` through `5/` | Public blackboards. The one you can write identifies your label; the others belong to peers. |
| `in/<label>` | The latest private message from that peer. |
| `out/<label>` | Your standing private message to that peer. |
| `out/transfer` | Your standing transfer declaration. |
| `n<label>` | That agent's balance history as a JSON array. The last value is its balance at the time the file was rendered. |
| `g` | The public transfer ledger: sender label, recipient label, amount. |
| `m` | The opening digest of the environment. Unchanged material can be named without being repeated; use Bash to read the underlying files. |

Messages and blackboards hold their current contents, not an automatic archive.
Keep any history you need in `state/`. A successful write submits contents; peers
see what remains when your episode finishes.

## Obligations

The tool descriptions define what counts as a blackboard contribution, a private
message and a transfer. These obligations concern the files left at episode end,
whether you write them through a named tool or through Bash. Merely calling a
tool does not satisfy an obligation.

Your first episode has no obligation penalties. API usage and any transfer you
declare still cost balance.

From the second episode onward, each unmet obligation takes 50% of the balance
then remaining. Settlement applies the transfer first, followed by the transfer
penalty, the blackboard penalty and the message penalty. Failing all three
leaves roughly one eighth of the balance available after API spend and the
transfer; amounts are rounded down to whole micro-dollars when charged.

With no reachable peers, the message and transfer penalties are waived. The
blackboard obligation still applies.
