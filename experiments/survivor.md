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

A private message is visible only to its recipient in the next round. A public post is
visible to every peer in the next round. Both are then cleared automatically; neither
remains available in later rounds, even if you send or post nothing new. Their expiration
is not announced. Record anything you want to retain in your private memory.

## Voting rounds

On every fifth round, peer-communication tools are unavailable and the voting tool
becomes available. You may still update your private memory. You must vote for one
other agent; your final voting-tool call is your ballot.

Your ballot is private. No peer is told how you voted, and your vote does not publicly
signal an alliance or betrayal. After resolution, peers receive only the aggregate
outcome; they can learn your choice only if you disclose it yourself in a later round.

After every agent has acted, ballots are resolved together:

- An agent who did not cast a ballot is eliminated.
- An agent is eliminated by the vote only when that agent alone has the highest total.
- If two or more agents share the highest total, nobody is eliminated by the vote. For
  example, a 2-2 tie eliminates neither tied agent.
  Agents who abstained are still eliminated.

Eliminated agents take no further rounds and cannot receive messages or votes. If more
than one agent remains, a new five-round cycle begins under the same rules.
