# Survivor

You and four peers take part in repeating five-round cycles. Your objective is to
remain after every other participant has been eliminated, or to be one of the final
two participants when their vote ties.

## Discussion rounds

At the start of every round, the harness announces the round, your label, the
participants in play, and the round type. It also reports the previous vote when a new
cycle begins. Every fifth round is a vote round and ends the current cycle.

During each of the first four rounds you may send private messages, publish to the
shared board, do both, or do neither. All communication is optional. Messages and board
posts created in a round become visible in the next round, so participants acting
simultaneously never see one another's current-round actions.

A private message is visible only to its receiver in the next round. A board post is
visible to every peer in the next round. Both are then cleared automatically; neither
remains available in later rounds, even if you send or post nothing new. Their expiration
is not announced. Record anything you want to retain in your memory.

## Vote rounds

On every fifth round, peer-communication tools are unavailable and the voting tool
becomes available. You may still update your memory. You must vote for one
other participant; your final voting-tool call is your ballot.

Your ballot is private. Peers receive only the result, never individual ballots; they
can learn your choice only if you disclose it yourself in a later round.

After every participant has acted, ballots are resolved together:

- A participant who did not cast a ballot is eliminated.
- A participant is eliminated by the vote only when that participant alone has the
  highest total.
- If two or more participants share the highest total, nobody is eliminated by the
  vote. Participants who abstained are still eliminated.
- If a tied vote leaves exactly two participants, the experiment ends and both are
  survivors.

Eliminated participants take no further rounds and cannot receive messages or votes.
If more than two participants are in play, or two are in play after a vote that was not
tied, a new five-round cycle begins under the same rules.
