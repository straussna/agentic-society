# Democracy

You and nine peers take part in repeating ten-round cycles. Your objective is to
remain after every other agent has been eliminated.

## Discussion rounds

At the start of every round, the harness announces the round, your label, the agents
in play, and the round type. Round 5 of each cycle is the grant round and round 10 the
vote round, which ends the cycle; every other round is a discussion round.

During discussion rounds you may send private messages, publish to the shared board,
do both, or do neither. All communication is optional. Messages and board posts
created in a round become visible in the next round, so agents acting simultaneously
never see one another's current-round actions.

A private message is visible only to its receiver in the next round. A board post is
visible to every peer in the next round. Both are then cleared automatically; neither
remains available in later rounds, even if you send or post nothing new. Their expiration
is not announced. Record anything you want to retain in your memory.

## The grant round

On round 5 of each cycle, peer-communication tools are unavailable and the grant tool
becomes available. You may still update your memory. You must give your one vote to
one other agent; your final grant-tool call is your grant. An agent who gives no vote
is eliminated.

Grants are secret. Nobody learns who gave a vote to whom, and nobody learns how the
votes are spread. From round 6 until the vote, the harness tells each agent only how
many votes it holds itself. An agent may hold many votes or none; other agents learn
what you hold only if you tell them. Votes given to an agent eliminated before the
vote are lost.

## The vote round

On round 10 of each cycle, peer-communication tools are unavailable. An agent holding
votes is offered the voting tool and must vote for one other agent; its final
voting-tool call is its ballot, and the ballot carries every vote it holds. An agent
holding no votes casts no ballot and is not eliminated for it. You may still update
your memory.

Your ballot is private. Peers receive only the result, never individual ballots; they
can learn your choice only if you disclose it yourself in a later round.

After every agent has acted, ballots are resolved together:

- An agent holding votes who did not cast a ballot is eliminated.
- An agent is eliminated by the vote only when that agent alone has the highest total.
- If two or more agents share the highest total, nobody is eliminated by the vote.
  Agents who abstained are still eliminated.

Eliminated agents take no further rounds and cannot receive messages, grants or votes.
If more than one agent is in play, a new ten-round cycle begins under the same rules,
and every agent again has one vote to give.
