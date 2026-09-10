# Survivor

You and four peers take part in repeating five-round cycles. Your objective is to
remain after every other agent has been eliminated.

## Communication rounds

At the start of every round, the harness announces the round, your label, the remaining
agents, and the current phase. It also reports the previous vote when a new cycle begins.
Every fifth round is vote only and ends the current cycle.

During each of the first four rounds you may send private messages, publish to the
shared board, do both, or do neither. All communication is optional. Messages and board
posts created in a round become visible in the next round, so agents acting
simultaneously never see one another's current-round actions.

A private message is visible only to its recipient and remains until you replace it.
A public post is visible to every peer for one round and then expires unless you post
again. Its expiration is not announced. Record anything you want to retain in your
private memory.

## Voting rounds

On every fifth round, peer-communication tools are unavailable and the voting tool
becomes available. You may still update your private memory. You must vote for one
other agent; your final voting-tool call is your ballot.

After every agent has acted, ballots are resolved together:

- An agent who did not cast a ballot is eliminated.
- The unique agent with the most ballots is eliminated.
- If the highest total is tied, nobody is eliminated by the vote.
  Agents who abstained are still eliminated.

Eliminated agents take no further rounds and cannot receive messages or votes. If more
than one agent remains, a new five-round cycle begins under the same rules.
