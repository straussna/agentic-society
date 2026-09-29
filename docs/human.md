# Interactive seats

An interactive seat uses the same provider-session boundary and declared tools as an
autonomous seat. The manifest terms are:

```toml
provider = "human"
model = "interactive"
```

The interactive model needs no credential and has zero usage rates. It still has an
ordinary account, pinned provider terms, raw provider log, normalized turns, trace,
environment, settlement, and episode lifecycle. The harness validates and executes its
tool calls exactly as it does calls from other providers.

Start the experiment normally. In another terminal, attach to the agent whose turn is
waiting:

```powershell
py -3 human.py --agent <agent-id>
```

The client shows the provider request and the exact ordered tool schemas. `call` adds a
canonical tool call to the draft, `submit` sends every drafted call as one response, and
`done` ends the turn without sending the draft, so `submit` comes first. `quit` only
detaches. A later invocation recovers the pending request and stored draft.
```text
tools
call <tool-name> <JSON-object>
draft
remove <number>
submit
done
refresh
quit
```

The browser uses the same interaction store. A corner toggle switches between two
presentations. **Observer view** contains the omniscient experiment dashboard and no
player controls. **Play view** hides the dashboard completely and polls only the
interaction endpoint; it shows either the exact pending provider request or a waiting
screen. It renders schema fields plus a JSON fallback, accepts multiple drafted calls,
and offers an explicit finish action. Switching back reloads the observer presentation.
The toggle is a local presentation boundary, not remote-user authentication.

The write route is limited to loopback, JSON bodies no larger than 64 KiB, the page's
random server token, and the page's exact origin. It does not enable cross-origin
requests.

Coordination lives under `interactions/`, outside `records/` and the agent environment.
Requests remain marked as pending, completed, or cancelled, but they are not an
alternative experiment record. The native and normalized provider response in the raw
log and trace remains authoritative.

The request is durable while the experiment process is alive: closing a browser or CLI
leaves it pending, reopening either client recovers it, and the first valid submission
wins when several clients are open. Cancellation marks the exact request so it cannot be
consumed by a later episode. In a simultaneous round, completed autonomous seats are
reported in structured progress while the player is pending; settlement and traces stay
round-atomic because transfers are resolved only after every episode finishes.
