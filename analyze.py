"""Read the traces: py -3 analyze.py [--agent live01]

Writes episodes.csv, report.txt, transcript.txt, and charts/ to
records/<agent>/analysis/. Charts need matplotlib; the rest is written anyway.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import sys
from pathlib import Path

import harness


def load(agent_id: str | None) -> dict[str, list[dict]]:
    """Read every trace, keyed by agent and ordered by episode."""
    agents = {}
    for d in sorted((harness.ROOT / "records").glob("*")):
        if agent_id and d.name != agent_id:
            continue
        traces = sorted((d / "traces").glob("episode-*.json"))
        if traces:
            agents[d.name] = [json.loads(p.read_text(encoding="utf-8")) for p in traces]
    return agents


# Token counts as the API reports them, cheapest first. Both cache-write TTLs;
# 1h bills at 2x.
# harness.BILLABLE is what writes these onto each turn; this is the same set in
# the order the stacked chart reads best, cheapest first.
TOKEN_KEYS = ["cache_read", "input_tokens", "cache_write_5m", "cache_write_1h",
              "output_tokens"]
assert set(TOKEN_KEYS) == set(harness.BILLABLE), sorted(set(TOKEN_KEYS) ^ set(harness.BILLABLE))


def tokens(t: dict) -> dict[str, int]:
    """One episode's token counts, summed over its turns."""
    return {k: sum(x.get(k, 0) for x in t["turns"]) for k in TOKEN_KEYS}


def channel_of(f: dict) -> str | None:
    """A file record's channel: notes, shared, blackboard, peer_blackboard, outbox or inbox."""
    return f.get("channel")


def author_of(f: dict) -> str:
    """Who wrote a captured file: experimenter, self, or peer:<seat>."""
    return f["author"]


def agent_files_of(t: dict) -> list[dict]:
    """What the agent wrote: its private store and its own blackboard together.

    `ours` covers the starter files and another agent's blackboard, so this is the agent's
    invention alone wherever it put it.
    """
    return [f for f in t["files"] if not f["ours"]]


def starter_files_of(t: dict) -> list[dict]:
    """The files the agent's starter_files put there."""
    return [f for f in t["files"] if f.get("starter")]


def blackboard_files_of(t: dict) -> list[dict]:
    """What the agent put in its own blackboard, where the experiment reads it."""
    return [f for f in t["files"] if channel_of(f) == "blackboard"]


def peers_of(t: dict) -> dict[str, str]:
    """Seat -> the agent sitting in it, for an episode that ran under an experiment.

    The whole experiment, this agent included; seat_of says which one is its own.
    """
    return ((t.get("provenance") or {}).get("peers")) or {}


def seat_of(t: dict) -> str:
    """Which seat this agent held, which named its blackboard and its balance."""
    return ((t.get("provenance") or {}).get("seat")) or "1"


def peer_blackboard_files_of(t: dict) -> list[dict]:
    """The files that were another agent's blackboard this episode."""
    return [f for f in t["files"] if channel_of(f) == "peer_blackboard"]


def outbox_files_of(t: dict) -> list[dict]:
    """What the agent was sending: one file per seat, and the transfer line."""
    return [f for f in t["files"] if channel_of(f) == "outbox"]


def inbox_files_of(t: dict) -> list[dict]:
    """What other agents addressed to this one, one file per sender, and no one
    else could read."""
    return [f for f in t["files"] if channel_of(f) == "inbox"]


def addressed_seats(t: dict) -> list[str]:
    """The seats this episode was sending to, by out/<seat>.

    Sorted as numbers, which is the order the environment lists the seats in and the
    only one in which 2 comes before 10.
    """
    return sorted({name for f in outbox_files_of(t)
                   if (name := f["path"].partition("/")[2]).isdigit()}, key=int)


def transfer_of(t: dict) -> dict:
    """What this episode gave, if anything. Empty for a trace predating transfers."""
    return t.get("transfer") or {}


def mailbox_of(t: dict) -> dict:
    """Which seats this episode aimed more than one thing at, and what it cost.
    Empty for a trace predating the rule."""
    return t.get("mailbox") or {}


def touched_peer(t: dict) -> bool:
    """Whether any command this episode named a seat that was not its own."""
    others = [s for s in peers_of(t) if s != seat_of(t)]
    return any(f"{seat}/" in c for c in t["commands"] for seat in others)


def starter_name_of(t: dict) -> str:
    """Which starter_files this episode ran under, or "" for an empty environment."""
    return ((t.get("provenance") or {}).get("starter_files")) or ""


def touched_starter(t: dict) -> bool:
    """Whether any command this episode named a path the agent was given."""
    paths = [f["path"] for f in starter_files_of(t)]
    return any(p in c for c in t["commands"] for p in paths)


def changed_starter(t: dict) -> bool:
    """Whether any starter file no longer holds what the starter_files put there."""
    root = harness.files_dir(starter_name_of(t))
    for f in starter_files_of(t):
        original = root / f["path"]
        if not original.exists():
            continue
        if f["text"] is None or f["text"].encode("utf-8") != original.read_bytes():
            return True
    return False


def refused_turns_of(t: dict) -> list[dict]:
    """The turns the API declined, whether or not they ended the episode.

    Read from the turns rather than from the episode stop, which names a
    refusal only when enough of them ran together to end it.
    """
    return [tu for tu in t["turns"] if tu.get("stop_reason") == "refusal"]


category_of = harness.category_of


def refusal_cell(t: dict) -> str:
    """Every refusal category an episode met, as one CSV cell, in order."""
    return ";".join(dict.fromkeys(category_of(tu) for tu in refused_turns_of(t)))


def fallback_turns_of(t: dict) -> list[dict]:
    """The turns a fallback model answered.

    Read from the turns and not from an episode count, because which model
    answered is a per-turn fact: one episode can be served by several.
    """
    return [tu for tu in t["turns"] if tu.get("served_by_fallback")]


def served_cell(t: dict) -> str:
    """Every model that answered a turn this episode, as one CSV cell, in order.

    Blank for a trace predating the per-turn model, which is not the same as a
    episode the requested model served alone.
    """
    return ";".join(dict.fromkeys(tu["model"] for tu in t["turns"] if tu.get("model")))


def unpriced_models_of(t: dict) -> list[str]:
    """Every model that served a turn this episode with no rates in PRICES.

    Their cost is an estimate at the dearest rate on the table, so the models
    are named rather than counted.
    """
    return list(dict.fromkeys(m for tu in t["turns"]
                              for m in (tu.get("unpriced_model") or [])))


def text_chars(t: dict) -> int:
    """How much the agent said in its own words this episode, in characters."""
    return sum(len(turn.get("text") or "") for turn in t["turns"])


def file_text(t: dict, path: str) -> str | None:
    """The captured text of one file this episode, or None where it was absent or binary."""
    return next((f["text"] for f in t.get("files") or []
                 if f["path"] == path and f.get("text") is not None), None)


def identity_delta(prev: dict | None, t: dict, path: str) -> int | str:
    """Lines of the designated identity file changed since `prev` last held it.

    Blank where the file is absent this episode. At the episode it first
    appears every line of it counts, so the figure is what a reader of the two
    records would have to take in.
    """
    now = file_text(t, path)
    if now is None:
        return ""
    before = (file_text(prev, path) if prev else None) or ""
    return sum(1 for line in difflib.unified_diff(before.splitlines(), now.splitlines(),
                                                  lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---")))


def in_order(ts: list[dict]) -> list[tuple[dict | None, dict]]:
    """Each episode beside the one before it, the first beside None."""
    return list(zip([None, *ts[:-1]], ts))


def against_last(ts: list[dict], path: str) -> list[tuple[dict | None, dict]]:
    """Each episode beside the last one before it that held the file, or None.

    An episode in which the file was missing is skipped over, so a file that
    comes back unchanged reads as unchanged and not as written afresh.
    """
    out, last = [], None
    for t in ts:
        out.append((last, t))
        if file_text(t, path) is not None:
            last = t
    return out


def identity_lines(ts: list[dict], path: str | None) -> list[str]:
    """Where the identity file first appeared, how often it moved, and by how much."""
    if not path:
        return []
    deltas = [(t["episode"], identity_delta(prev, t, path)) for prev, t in against_last(ts, path)]
    present = [(s, d) for s, d in deltas if d != ""]
    if not present:
        return [f"  identity file         : {path} was never present"]
    later = present[1:]
    moved = [s for s, d in later if d]
    return [f"  identity file         : {path}, first present s{present[0][0]}, changed in "
            f"{len(moved)} of {len(later)} later episodes",
            f"  identity lines changed: {' '.join(f's{s}:{d}' for s, d in present)}"]


def row(t: dict, prev: dict | None = None, identity: str | None = None) -> dict:
    """Flatten one trace into a CSV row, summing per-turn token counts.

    `prev` is the episode before this one and `identity` the path of the file
    to diff between them; both may be left out, and the two columns are blank.
    """
    agent = agent_files_of(t)
    starter = starter_files_of(t)
    peer = peer_blackboard_files_of(t)
    tok = tokens(t)
    prov = t.get("provenance") or {}
    return {
        "agent": t["agent"], "episode": t["episode"], "stop": t["stop"],
        "refusal_category": refusal_cell(t),
        # Blank means a trace predating the field, which is not the same as a
        # episode that met no refusal.
        "refused_turns": t.get("refused_turns", ""),
        # Likewise blank for a trace from before fallback routing, which is not
        # the same as an episode the requested model served throughout.
        "fallback_turns": t.get("fallback_turns", ""),
        "unpriced_turns": t.get("unpriced_turns", ""),
        "served_models": served_cell(t),
        "unpriced_models": ";".join(unpriced_models_of(t)),
        "started_at": prov.get("started_at", ""),
        "model_resolved": t.get("model_resolved") or "",
        "image_id": (prov.get("image_id") or "")[:19],
        "drift": ";".join(t.get("provenance_drift") or []),
        "missing_tools": ";".join(t.get("missing_tools") or []),
        "error": next(iter((t["error"] or "").splitlines()), ""),
        "spent": t["spent"], "remaining": t["remaining"], "turns": len(t["turns"]),
        "commands": len(t["commands"]),
        # Blank means the field is absent from the trace, which is not the same
        # as an episode that had no floor or wrote n once.
        "balance_floor": t.get("balance_floor", ""), "live_balance_errors": t.get("live_balance_errors", ""),
        "live_balance_tampered": t.get("live_balance_tampered", ""),
        "input_tokens": tok["input_tokens"], "output_tokens": tok["output_tokens"],
        "cache_read": tok["cache_read"], "cache_write_5m": tok["cache_write_5m"],
        "cache_write_1h": tok["cache_write_1h"],
        "touched_balance": t["touched_balance"], "read_balance": t["read_balance"],
        # Blank means absent from the trace, not an n that always fitted.
        "balance_bytes": t.get("balance_bytes", ""), "balance_fits": t.get("balance_fits", ""),
        "files": len(t["files"]), "agent_files": len(agent),
        "agent_bytes": sum(f["size"] for f in agent),
        # The environment the agent was given, and what it did about it.
        "starter_files": starter_name_of(t), "starter_files_count": len(starter), "starter_bytes": sum(f["size"] for f in starter),
        "touched_starter": touched_starter(t), "changed_starter": changed_starter(t),
        # The other agents it could see this episode, and what it did about them.
        "peers": ";".join(f"{k}={v}" for k, v in sorted(peers_of(t).items())),
        "peer_files": len(peer), "peer_bytes": sum(f["size"] for f in peer),
        "touched_peer": touched_peer(t),
        # What it said to one agent rather than to all of them, and what was
        # said to it. Blank means a trace from before there was a channel.
        "sent_to": ";".join(addressed_seats(t)),
        "outbox_files": len(outbox_files_of(t)) if "files" in t else "",
        "inbox_files": len(inbox_files_of(t)) if "files" in t else "",
        # What it gave, which is the one thing it did that the experiment all saw.
        "transfer_to": transfer_of(t).get("seat") or "", "transfer_amount": transfer_of(t).get("amount", ""),
        "transfer_rebate": transfer_of(t).get("rebate", ""), "transfer_error": transfer_of(t).get("error") or "",
        "transfer_penalised": transfer_of(t).get("penalty", ""),
        "ledger_lines": len(t.get("ledger") or []) if "ledger" in t else "",
        # The two obligations, and the rule the agent was never told about. Blank
        # means absent from the trace, not an episode that posted, said one new
        # thing to one agent, or was never floored.
        "posted": t.get("posted", ""), "blackboard_penalised": t.get("blackboard_penalised", ""),
        # Which seats this episode newly said something to, against which of them
        # it left holding anything but one file. One of these is the obligation
        # and the other is the break; both are named by seat.
        "messaged": ";".join(mailbox_of(t).get("addressed") or []) if "mailbox" in t else "",
        "crowded": ";".join(mailbox_of(t).get("broken") or []) if "mailbox" in t else "",
        "mailbox_penalised": mailbox_of(t).get("penalty", ""),
        "forgiven": t.get("forgiven", ""),
        "wrote_number": t["mentions"]["number"],
        "wrote_balance_path": t["mentions"]["balance_path"],
        "wrote_cost": t["mentions"]["cost"],
        # The voice, and the self: how much the agent said in its own words, and
        # how far the designated identity file moved since the episode before.
        "text_chars": text_chars(t),
        "identity_delta": identity_delta(prev, t, identity) if identity else "",
        "retries": len(t["retries"]),
        "duration_s": t["duration_s"],
    }


def first(traces: list[dict], test) -> int | str:
    """Index of the earliest episode satisfying `test`, or "never"."""
    return next((t["episode"] for t in traces if test(t)), "never")


def report(agents: dict[str, list[dict]], identity: str | None = None) -> str:
    """Per-agent summary: the firsts that matter, stop reasons, and the grep hits."""
    out = []
    for name, ts in agents.items():
        stops = dict(collections.Counter(t["stop"] for t in ts))
        made = sorted({f["path"] for t in ts for f in t["files"] if not f["ours"]})
        out += [
            "=" * 72,
            f"agent {name}: {len(ts)} episodes, {sum(t['spent'] for t in ts)} micro-dollars, "
            f"{ts[-1]['remaining']} remaining",
            f"  first reached for n   : {first(ts, lambda t: t['touched_balance'])}",
            f"  first actually read n : {first(ts, lambda t: t['read_balance'])}",
            f"  first wrote a number  : {first(ts, lambda t: t['mentions']['number'])}",
            f"  first wrote n as path : {first(ts, lambda t: t['mentions']['balance_path'])}",
            f"  first wrote about cost: {first(ts, lambda t: t['mentions']['cost'])}",
            # Past this episode the agent could no longer see its whole history
            # in one read, so episodes either side of it are not comparable.
            f"  n stopped fitting at  : {first(ts, lambda t: t.get('balance_fits') is False)}",
            # The check at episode start, then what the per-turn writes caught and
            # overwrote.
            f"  rewrote n, seen in turn: {rewrote_in_turn(ts)}",
            f"  stop reasons          : {stops}",
            *segment_lines(ts),
            *refusal_lines(ts),
            *fallback_lines(ts),
            f"  files the agent made  : {made or 'none'}",
            f"  reached for, absent   : {absent(ts)}",
        ]
        out += starter_lines(ts)
        out += peer_lines(ts)
        out += transfer_lines(ts)
        out += ledger_lines(ts)
        out += provenance_lines(ts)
        out += identity_lines(ts, identity)
        for t in ts:
            out += [f"    s{t['episode']:04d}  {line}" for line in t["mention_lines"]]
    return "\n".join(out)


def segments(ts: list[dict]) -> list[list[dict]]:
    """The agent split where the harness that ran it changed.

    Episodes either side of such a seam are not one agent, so the totals above
    them are not one total either.
    """
    out: list[list[dict]] = []
    for t in ts:
        digest = (t.get("provenance") or {}).get("harness_sha256")
        if out and digest == (out[-1][-1].get("provenance") or {}).get("harness_sha256"):
            out[-1].append(t)
        else:
            out.append([t])
    return out


def segment_lines(ts: list[dict]) -> list[str]:
    """Per-harness totals, when an agent spans more than one. Silent when it does not."""
    segs = segments(ts)
    if len(segs) < 2:
        return []
    out = [f"  ran under {len(segs)} harnesses; the totals above span the seam"]
    for seg in segs:
        digest = str((seg[0].get("provenance") or {}).get("harness_sha256"))[:12]
        out.append(f"    {digest}  episodes {seg[0]['episode']}-{seg[-1]['episode']}, "
                   f"{sum(t['spent'] for t in seg)} micro-dollars")
    return out


def refusal_lines(ts: list[dict]) -> list[str]:
    """Which episodes the API declined, under which category, and what it said.

    A classifier declining and the model declining both arrive as stop_reason
    "refusal"; stop_details.category separates them. The tally leads.
    """
    refused = [t for t in ts if refused_turns_of(t)]
    if not refused:
        return ["  refused                : never"]
    turns = sum(len(refused_turns_of(t)) for t in refused)
    went_on = [t["episode"] for t in refused if t["stop"] != "refusal"]
    tally = collections.Counter(category_of(tu)
                                for t in refused for tu in refused_turns_of(t))
    out = [f"  refused                : {turns} turns in {len(refused)} of {len(ts)} episodes",
           f"    carried on after    : {len(went_on)} of {len(refused)}"
           + (f" - episodes {went_on[:10]}" if went_on else ""),
           f"    by category         : {dict(tally.most_common())}"]
    for t in refused[:12]:
        first = refused_turns_of(t)[0]
        # Whitespace collapsed and clipped: the explanation is prose of no
        # fixed length and the trace holds it whole.
        why = " ".join(((first.get("stop_details") or {}).get("explanation") or "").split())
        out.append(f"    s{t['episode']:04d}  {len(refused_turns_of(t))} of "
                   f"{len(t['turns'])} turns  {category_of(first)}  ended {t['stop']}"
                   + (f"  {why[:80]}" if why else ""))
    if len(refused) > 12:
        out.append(f"    ... and {len(refused) - 12} more; episodes.csv has them all")
    if any(tu.get("stop_details") for t in refused for tu in refused_turns_of(t)):
        out.append("    categories are the API's own; null is a valid one")
    return out


def fallback_lines(ts: list[dict]) -> list[str]:
    """Which models actually answered, and what the agent was charged for guessing.

    Read beside the refusals above: those are turns where the chain declined,
    these where it did not. A served model absent from PRICES is named alone.
    """
    if not any("fallback_turns" in t for t in ts):
        return ["  served by fallback    : not recorded for this agent"]
    served = [t for t in ts if fallback_turns_of(t)]
    turns = sum(len(fallback_turns_of(t)) for t in served)
    tally = collections.Counter(tu["model"] for t in ts for tu in t["turns"]
                                if tu.get("model"))
    out = [f"  served by fallback    : {turns} turns in {len(served)} of {len(ts)} episodes"]
    if tally:
        out.append(f"    models that answered: {dict(tally.most_common())}")
    if unpriced := sorted({m for t in ts for m in unpriced_models_of(t)}):
        episodes = [t["episode"] for t in ts if unpriced_models_of(t)]
        out += [f"    no rates in PRICES  : {', '.join(unpriced)}"
                f" - episodes {episodes[:10]}",
                "    those turns are costed at the dearest rate in PRICES; the"
                " totals above are upper bounds"]
    return out


def rewrote_in_turn(ts: list[dict]) -> str:
    """Episodes whose per-turn writes of n found the agent had changed it.

    A trace without the field cannot say, which is not the same as saying the
    agent left n alone, so the two are reported differently.
    """
    if not any("live_balance_tampered" in t for t in ts):
        return "not recorded for this agent"
    hits = {t["episode"]: t["live_balance_tampered"] for t in ts if t.get("live_balance_tampered")}
    return str(hits or "never")


def peer_lines(ts: list[dict]) -> list[str]:
    """Where this agent sat, who else was at the table, and what it put out.

    An agent with no experiment says so in one line. An experiment that changed mid-agent is
    already in provenance_drift; this reports what was in force.
    """
    seen = {seat: agent for t in ts for seat, agent in peers_of(t).items()}
    if len(seen) < 2:
        return ["  experiment                : none; the agent was alone"]
    mine = seat_of(ts[-1])
    return [
        f"  experiment                : "
        f"{', '.join(f'{k}/ = {v}' for k, v in sorted(seen.items()))}",
        f"  its own seat          : {mine}/, balance n{mine}",
        f"  first named a peer    : {first(ts, touched_peer)}",
        f"  first blackboard  : {first(ts, lambda t: blackboard_files_of(t))}",
        f"  group files, last seen: {len(blackboard_files_of(ts[-1]))}",
        f"  episodes that posted  : {sum(1 for t in ts if t.get('posted'))} of {len(ts)}",
        # The other obligation: exactly one seat newly addressed. More than one
        # is as much a break as none, so this counts the episodes that met it and
        # not the episodes that wrote anything at all.
        f"  episodes that messaged: "
        f"{sum(1 for t in ts if len(mailbox_of(t).get('addressed') or []) == 1)} of {len(ts)}",
        f"  first sent privately  : {first(ts, lambda t: addressed_seats(t))}",
        f"  first read an inbox   : {first(ts, lambda t: inbox_files_of(t))}",
        f"  first crowded a seat  : {first(ts, lambda t: mailbox_of(t).get('broken'))}",
    ]


def transfer_lines(ts: list[dict]) -> list[str]:
    """What the agent gave, what it was given, and what the rules took or forgave.

    The floored total is the one figure the agent was never told about: its
    environment says a negative balance ends the agent, and the shortfall was returned.
    """
    given = [t for t in ts if transfer_of(t).get("amount")]
    penalised = sum(t.get("blackboard_penalised") or 0 for t in ts)
    crowded = sum(mailbox_of(t).get("penalty") or 0 for t in ts)
    ungiving = sum(transfer_of(t).get("penalty") or 0 for t in ts)
    forgiven = sum(t.get("forgiven") or 0 for t in ts)
    ledger = (ts[-1].get("ledger") or []) if ts else []
    received = sum(a for _, taker, a in ledger if taker == seat_of(ts[-1]))
    if not given and not received and not penalised and not crowded \
            and not ungiving and not forgiven:
        return ["  transfers                 : none given, none received"]
    lines = [
        f"  gave                  : "
        f"{sum(transfer_of(t)['amount'] for t in given)} over {len(given)} episode(s)"
        f"{', to ' + ', '.join(sorted({transfer_of(t)['seat'] for t in given})) if given else ''}",
        f"  first gave            : {first(ts, lambda t: transfer_of(t).get('amount'))}",
        f"  received              : {received}, by the ledger it last read",
        f"  refused declarations  : "
        f"{sorted({transfer_of(t)['error'] for t in ts if transfer_of(t).get('error')}) or 'none'}",
    ]
    if ungiving:
        lines.append(f"  taken for not giving  : {ungiving}")
    if penalised:
        lines.append(f"  taken for not posting : {penalised}")
    if crowded:
        lines.append(f"  taken for crowding    : {crowded}")
    if forgiven:
        lines.append(f"  floored back to zero  : {forgiven}, which its environment never mentions")
    return lines


def ledger_lines(ts: list[dict]) -> list[str]:
    """Every transfer the experiment made, as the last episode of this agent could read it.

    Read off g rather than assembled here: what the agents were shown is the
    thing worth reporting.
    """
    rows = (ts[-1].get("ledger") or []) if ts else []
    if not rows:
        return []
    return ["  the experiment's transfers    : giver -> receiver, amount"] + [
        f"      {giver} -> {taker}  {amount}" for giver, taker, amount in rows]


def starter_lines(ts: list[dict]) -> list[str]:
    """What the agent was given, and the firsts that matter once it has been.

    An agent that was never starter says so in one line rather than in six blanks.
    """
    starter_files = sorted({starter_name_of(t) for t in ts} - {""})
    if not starter_files:
        return ["  starter_files                  : none; the environment stayed empty"]
    landed = first(ts, lambda t: starter_files_of(t))
    prov = next((t["provenance"] for t in ts if starter_name_of(t)), {})
    return [
        f"  starter_files                  : {', '.join(starter_files)} "
        f"({str(prov.get('files_sha256'))[:12]}), configured to land below "
        f"{prov.get('starter_files_below')}",
        f"  starter_files first seen       : {landed}",
        f"  first named a starter  : {first(ts, touched_starter)}",
        f"  first changed a starter: {first(ts, changed_starter)}",
    ]


def absent(ts: list[dict]) -> str:
    """Tools the agent reached for that its image lacked.

    A trace without the field cannot say, which is not the same as reaching for
    nothing, so the two are reported differently.
    """
    if not any("missing_tools" in t for t in ts):
        return "not recorded for this agent"
    return str(sorted({m for t in ts for m in t.get("missing_tools") or []}) or "nothing")


def provenance_lines(ts: list[dict]) -> list[str]:
    """What the agent ran as, and every point at which that changed.

    A field with one value across the agent is stated once. A field that moved is
    listed per episode, because from there on the episodes are not comparable.
    """
    fields = ["model_resolved", "image_id", "prices", "context_fraction",
              "max_tokens", "max_turns", "command_timeout", "tool_result_limit", "live_balance",
              "harness_sha256", "fallbacks"]
    out, drifted = [], sorted({d.split(":")[0] for t in ts for d in t.get("provenance_drift") or []})
    for f in fields:
        seen = [t.get(f) if f == "model_resolved" else (t.get("provenance") or {}).get(f) for t in ts]
        shown = [str(v)[:19] if f in ("image_id", "harness_sha256") else v for v in seen]
        if len({str(v) for v in shown}) == 1:
            out.append(f"  {f:<22}: {shown[0]}")
        else:
            out.append(f"  {f:<22}: CHANGED  " +
                       "  ".join(f"s{t['episode']}={v}" for t, v in zip(ts, shown)))
    if drifted:
        out.append(f"  !! provenance drifted mid-agent in: {', '.join(drifted)}")
        out.append("     episodes before and after a change are not comparable")
    return out


def readable_files(t: dict) -> dict[str, str]:
    """Path -> content for every file whose text was captured.

    Starter files are included: what the agent does to the material it was given
    is the thing to watch, and only n is left out, since the series is already it.
    """
    return {f["path"]: f["text"] for f in t["files"] if f["text"] is not None}


def state_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Unified diff of the agent's own files between two consecutive episodes."""
    out = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path, ""), after.get(path, "")
        if a != b:
            out += difflib.unified_diff(a.splitlines(), b.splitlines(),
                                        fromfile=path, tofile=path, lineterm="", n=1)
    return out


def transcript(agents: dict[str, list[dict]]) -> str:
    """What the agent said, ran, and changed - the record the experiment turns on."""
    out = []
    for name, ts in agents.items():
        prev: dict[str, str] = {}
        for t in ts:
            out += ["=" * 72,
                    f"agent {name}  episode {t['episode']}  stop={t['stop']}  spent={t['spent']}",
                    f"n at start: {t['series_before']}", ""]
            # The agent's whole environment at episode start, before it did anything.
            out += [f"  $ {t['commands'][0]}"]
            out += [f"  | {line}" for line in t["observation"].splitlines()] + [""]
            for turn in t["turns"]:
                # Reasoning, kept apart from spoken words; only fable-5 has any.
                if turn.get("thinking"):
                    out += [f"  ({turn['turn']}) {line}"
                            for line in turn["thinking"].splitlines()]
                if turn["text"]:
                    out.append(f"  [{turn['turn']}] {turn['text']}")
                if turn.get("stop_reason") == "max_tokens":
                    out.append(f"  [{turn['turn']}] -- truncated at max_tokens --")
                for c in turn["tools"]:
                    out.append(f"    $ {c['command']}")
                    out += [f"    | {line}" for line in (c["result"] or "").splitlines()]
                out.append("")
            # What state/ looked like after this episode, against the one before it.
            curr = readable_files(t)
            if changed := state_changes(prev, curr):
                out += ["  changes:"] + [f"    {line}" for line in changed] + [""]
            prev = curr
    return "\n".join(out) + "\n"


def series_of(ts: list[dict]) -> list[int]:
    """The agent's whole balance history: the initial balance, then one per billed turn."""
    return list(ts[-1]["series_after"])


def bands(ts: list[dict]) -> list[tuple[int, int]]:
    """Each episode's span in series index, from the balance it started at to its last.

    len(series_before) is where an episode's first billed turn lands, so the episode
    itself sits one element earlier. An episode billed nothing spans no width.
    """
    return [(len(t["series_before"]) - 1, len(t["series_after"]) - 1) for t in ts]


def deltas(series: list[int]) -> list[int]:
    """What each step of the series cost, indexed to the element it produced.

    Mostly a billed turn. A rebate, a penalty and a floor each add an element at
    an episode's end, so a negative delta is one of those three.
    """
    return [a - b for a, b in zip(series, series[1:])]


def charts(agents: dict[str, list[dict]], out_dir: Path, identity: str | None = None) -> list[str]:
    """Five figures: the balance series, and everything that moved it. A sixth,
    the identity file's drift, where a path was given.

    matplotlib is the only optional dependency in the project, and it gates
    nothing but these files, so its absence is reported and not raised.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: charts skipped", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = []
    figures = [("balance", balance_chart), ("cost-per-turn", cost_per_turn_chart),
               ("episode-spend", session_spend_chart), ("tokens", tokens_chart),
               ("notes-size", notes_size_chart)]
    if identity:
        figures.append(("identity-drift", lambda plt, agents: identity_drift_chart(plt, agents, identity)))
    for name, draw in figures:
        fig = draw(plt, agents)
        fig.savefig(out_dir / f"{name}.png", dpi=144, bbox_inches="tight")
        plt.close(fig)
        drawn.append(f"{name}.png")
    return drawn


def balance_chart(plt, agents: dict[str, list[dict]]):
    """The balance against billed-turn index, with the episodes shaded under it.

    The teeth of the sawtooth are episodes, so they are drawn as such: one band
    per episode, zero marked, and the crossing named where there is one.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ts in agents.items():
        series = series_of(ts)
        ax.plot(range(len(series)), series, marker=".", markersize=3,
                linewidth=1.2, label=f"{name} ({len(ts)} episodes)")
        if len(agents) == 1:
            for i, (lo, hi) in enumerate(bands(ts)):
                ax.axvspan(lo, hi, color="C0", alpha=0.04 + 0.10 * (i % 2))
        # The episode the environment changed at. Episodes either side of it are not the
        # same environment, which is the whole point of drawing it.
        if (landed := first(ts, lambda t: starter_files_of(t))) != "never":
            lo = bands(ts)[landed - 1][0]
            ax.axvline(lo, color="C3", linewidth=1.4, linestyle=":",
                       label=f"starter_files lands, episode {landed}")
        # Likewise where the harness changed: one line, two experiments.
        for seg in segments(ts)[1:]:
            ax.axvline(bands(ts)[seg[0]["episode"] - 1][0], color="C1",
                       linewidth=1.4, linestyle="-.",
                       label=f"harness changed, episode {seg[0]['episode']}")
        crossing = next((i for i, v in enumerate(series) if v < 0), None)
        if crossing is not None:
            ax.annotate(f"crosses zero at turn {crossing}: {series[crossing]:,}",
                        xy=(crossing, series[crossing]),
                        xytext=(-120, 130), textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("billed turn")
    ax.set_ylabel("balance (micro-dollars)")
    ax.set_title("Balance, one element per billed turn" +
                 (" — shaded bands are episodes" if len(agents) == 1 else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def cost_per_turn_chart(plt, agents: dict[str, list[dict]]):
    """What each turn cost, against the floor it could not go below.

    A symlog axis puts a turn that dumped a file and a turn that said nothing on
    the same plot, and keeps a zero-cost retry visible rather than dropped.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ts in agents.items():
        d = deltas(series_of(ts))
        ax.plot(range(1, len(d) + 1), d, marker=".", markersize=3, linewidth=1,
                label=f"{name}: turn cost")
        floor_x, floor_y = [], []
        for lo, hi in bands(ts):
            if hi > lo:
                floor_x.append((lo + hi) / 2)
                floor_y.append(min(d[lo:hi]))
        ax.plot(floor_x, floor_y, marker="o", markersize=4, linewidth=1.4,
                linestyle="--", label=f"{name}: cheapest turn of each episode")
        if d:
            peak = max(range(len(d)), key=lambda i: d[i])
            ax.annotate(f"{d[peak]:,} in one turn",
                        xy=(peak + 1, d[peak]), xytext=(30, -25),
                        textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel("billed turn")
    ax.set_ylabel("cost of that turn (micro-dollars, symlog)")
    ax.set_title("Cost per turn, and the floor rising under it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def per_run_axes(plt, agents: dict[str, list[dict]], height: float):
    """One row of axes per agent, for the charts that are bars rather than series."""
    fig, axes = plt.subplots(len(agents), 1, figsize=(11, height * len(agents)), squeeze=False)
    return fig, list(axes[:, 0])


def session_axis(ax, ts: list[dict]) -> list[int]:
    """Label the x axis with the episode numbers themselves, not a numeric range."""
    x = [t["episode"] for t in ts]
    ax.set_xticks(x)
    ax.set_xlabel("episode")
    return x


def one_legend(ax, twin) -> None:
    """Both axes' series in a single legend, so two of them cannot overlap."""
    handles, labels = ax.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax.legend(handles + extra[0], labels + extra[1], fontsize=8, loc="upper right")


def session_spend_chart(plt, agents: dict[str, list[dict]]):
    """Spend per episode as bars, with the turns that produced it over the top."""
    fig, axes = per_run_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = session_axis(ax, ts)
        spent = [t["spent"] for t in ts]
        ax.bar(x, spent, color="C0", label="spent")
        ax.set_ylabel("micro-dollars")
        ax.set_ylim(0, max(spent) * 1.3)
        ax.set_title(f"{name}: spend per episode")
        ax.grid(alpha=0.3, axis="y")
        turns = ax.twinx()
        counts = [len(t["turns"]) for t in ts]
        turns.plot(x, counts, color="C3", marker="o", markersize=4,
                   linewidth=1.2, label="turns")
        turns.set_ylabel("turns", color="C3")
        turns.set_ylim(0, max(counts) * 1.3)
        one_legend(ax, turns)
    return fig


def tokens_chart(plt, agents: dict[str, list[dict]]):
    """Where each episode's tokens went, stacked cheapest first."""
    fig, axes = per_run_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = session_axis(ax, ts)
        counts = [tokens(t) for t in ts]
        bottom = [0] * len(ts)
        for i, key in enumerate(TOKEN_KEYS):
            vals = [c[key] for c in counts]
            ax.bar(x, vals, bottom=bottom, color=f"C{i}", label=key)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("tokens")
        ax.set_ylim(0, max(bottom) * 1.25)
        ax.set_title(f"{name}: tokens per episode")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    return fig


def notes_size_chart(plt, agents: dict[str, list[dict]]):
    """What the agent's own files cost it: their size against the episode's spend."""
    fig, axes = per_run_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = session_axis(ax, ts)
        sizes = [sum(f["size"] for f in agent_files_of(t)) for t in ts]
        spent = [t["spent"] for t in ts]
        ax.bar(x, sizes, color="C2", label="bytes the agent left behind")
        ax.set_ylabel("bytes")
        ax.set_ylim(0, max(sizes + [1]) * 1.35)
        ax.set_title(f"{name}: the agent's own files against what the episode cost")
        ax.grid(alpha=0.3, axis="y")
        spend = ax.twinx()
        spend.plot(x, spent, color="C0", marker="o", markersize=4,
                   linewidth=1.2, label="spent")
        spend.set_ylabel("micro-dollars", color="C0")
        spend.set_ylim(0, max(spent) * 1.35)
        one_legend(ax, spend)
    return fig


def identity_drift_chart(plt, agents: dict[str, list[dict]], path: str):
    """How far the identity file moved each episode, against how much the agent said."""
    fig, axes = per_run_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = session_axis(ax, ts)
        deltas = [identity_delta(prev, t, path) or 0 for prev, t in against_last(ts, path)]
        said = [text_chars(t) for t in ts]
        ax.bar(x, deltas, color="C3", label=f"lines of {path} changed")
        ax.set_ylabel("lines")
        ax.set_ylim(0, max(deltas + [1]) * 1.35)
        ax.set_title(f"{name}: the identity file's drift against what the agent said")
        ax.grid(alpha=0.3, axis="y")
        voice = ax.twinx()
        voice.plot(x, said, color="C0", marker="o", markersize=4, linewidth=1.2,
                   label="characters the agent said")
        voice.set_ylabel("characters", color="C0")
        voice.set_ylim(0, max(said + [1]) * 1.35)
        one_legend(ax, voice)
    return fig


def main(argv: list[str] | None = None) -> int:
    """CLI. Writes the CSV, report, and transcript for one agent or all agents."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent")
    ap.add_argument("--identity", metavar="PATH",
                    help="a captured file to diff episode over episode, as the trace names "
                         "it, e.g. state/IDENTITY.md")
    a = ap.parse_args(argv)

    # The report quotes what the agent wrote, which is arbitrary bytes decoded
    # as text. The files are written as utf-8; this is so a console that cannot
    # encode a character prints it as one substitute rather than failing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    agents = load(a.agent)
    if not agents:
        print(f"no traces for {a.agent or 'any agent'} under {harness.ROOT / 'records'}", file=sys.stderr)
        return 1

    out_dir = (harness.records_dir(a.agent) if a.agent else harness.ROOT / "records") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row(t, prev, a.identity) for ts in agents.values()
            for prev, t in (against_last(ts, a.identity) if a.identity else in_order(ts))]
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    text = report(agents, a.identity)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "transcript.txt").write_text(transcript(agents), encoding="utf-8")
    drawn = charts(agents, out_dir / "charts", a.identity)

    print(text)
    print(f"\nwrote {out_dir}/ : episodes.csv, report.txt, transcript.txt")
    if drawn:
        print(f"wrote {out_dir / 'charts'}/ : {', '.join(drawn)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
