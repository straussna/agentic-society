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
`done` explicitly ends the turn. `quit` only detaches. A later invocation recovers the
pending request and stored draft.

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

The dashboard uses the same interaction store. Its player panel appears only while an
interactive request is pending, renders schema fields plus a JSON fallback, accepts
multiple drafted calls, and offers an explicit finish action. The write route is limited
to loopback, JSON bodies no larger than 64 KiB, the page's random server token, and the
page's exact origin. It does not enable cross-origin requests.

Coordination lives under `interactions/`, outside `records/` and the agent environment.
Requests remain marked as pending, completed, or cancelled, but they are not an
alternative experiment record. The native and normalized provider response in the raw
log and trace remains authoritative.
