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
from providers import USAGE_FIELDS

TOKEN_STACK_FIELDS = ("uncached_input_tokens", "cache_read_tokens", "cache_write_tokens",
                      "output_tokens")


def load(agent_id: str | None) -> dict[str, list[dict]]:
    """Read every trace, keyed by agent and ordered by episode."""
    agents = {}
    for d in sorted(harness.records_root().glob("*")):
        if agent_id and d.name != agent_id:
            continue
        traces = harness.trace_paths(d.name)
        if traces:
            loaded = [json.loads(p.read_text(encoding="utf-8")) for p in traces]
            incompatible = [t.get("trace_version") for t in loaded if t.get("trace_version") != 4]
            if incompatible:
                raise SystemExit(f"{d}: contains incompatible trace versions {sorted(set(incompatible))}; "
                                 "version-4 provider records require fresh agent ids")
            agents[d.name] = loaded
    return agents


def tokens(t: dict) -> dict[str, int]:
    """One episode's canonical token counts, summed over its turns."""
    return {k: sum(x["usage"][k] for x in t["turns"]) for k in USAGE_FIELDS}


# --- the files an episode captured, by whose they are ---------------------------


def agent_files_of(t: dict) -> list[dict]:
    """What the agent wrote, wherever it put it.

    `ours` covers the starter files and every other agent's channel, so this is the
    agent's invention alone.
    """
    return [f for f in t["files"] if not f["ours"]]


def starter_files_of(t: dict) -> list[dict]:
    """The files the agent's starter files put there."""
    return [f for f in t["files"] if f["starter"]]


def own_public_files(t: dict) -> list[dict]:
    """What the agent put where every other agent reads it."""
    return [f for f in t["files"] if f["role"] == "own" and f["readers"] == "all"]


def peer_public_files(t: dict) -> list[dict]:
    """The files that were another agent's public channel this episode."""
    return [f for f in t["files"] if f["role"] == "peer" and f["readers"] == "all"]


def outbox_files(t: dict) -> list[dict]:
    """What the agent was sending: one file per addressee."""
    parsed = {ch.name for ch in harness.table_of(t) if ch.schema}
    return [f for f in t["files"] if f["role"] == "own" and f["readers"] == "addressee"
            and f["channel"] not in parsed]


def inbox_files(t: dict) -> list[dict]:
    """What other agents addressed to this one, one file per sender."""
    parsed = {ch.name for ch in harness.table_of(t) if ch.schema}
    return [f for f in t["files"] if f["role"] == "peer" and f["readers"] == "addressee"
            and f["channel"] not in parsed]


def schema_files(t: dict) -> list[dict]:
    """The agent's files in the channel the harness parses, where it has one."""
    parsed = {ch.name for ch in harness.table_of(t) if ch.schema}
    return [f for f in t["files"] if f["role"] == "own" and f["channel"] in parsed]


# --- what the provenance and the channel records say ---------------------------


def provenance_of(t: dict) -> dict:
    """Everything outside the account that decided the episode, as its trace records it."""
    return t["provenance"]


def peers_of(t: dict) -> dict[str, str]:
    """Seat -> the agent sitting in it, for an episode that ran under an experiment.

    The whole experiment, this agent included; seat_of says which one is its own.
    """
    return provenance_of(t)["peers"]


def seat_of(t: dict) -> str:
    """Which seat this agent held."""
    return provenance_of(t)["seat"]


def labels_of(t: dict) -> dict[str, str]:
    """Seat -> the label the other agents know it by; the seat itself where none was given."""
    return provenance_of(t).get("labels") or {s: s for s in peers_of(t)}


def label_of(t: dict) -> str:
    """This agent's own label, which names its public paths and its balance file."""
    return labels_of(t).get(seat_of(t), seat_of(t))


def harness_files_of(t: dict) -> dict[str, str]:
    """What the harness's own files were called this episode."""
    return provenance_of(t)["harness_files"]


def channel_records(t: dict) -> dict[str, dict]:
    """What each channel the agent writes settled for, by channel name."""
    return t.get("channels") or {}


def settled_channels(t: dict) -> list[tuple[harness.Channel, dict]]:
    """Every channel this episode was settled against, with its record, in table order.

    The harness settles each obligated channel apart and records it by name, so an
    experiment declaring two blackboards or three mailboxes is read here as the
    obligations it had. A channel the environment left out - a mailbox with nobody
    to reach - has no record and is not one of them.
    """
    records = channel_records(t)
    return [(c, records[c.name]) for c in harness.table_of(t) if c.name in records]


def met_of(ch: harness.Channel, rec: dict) -> bool | None:
    """Whether one settled channel's obligation was met, by the rule that settles it.

    None where the record is empty, which is a channel that settled nothing. What
    was met and what was charged are two questions: a share is taken only from an
    episode the API answered, past the grace, at a rate above zero.
    """
    if not rec:
        return None
    if ch.schema:
        # A declaration left standing moves nothing a second time, so what counts
        # is money moved this episode and no share taken for having moved none.
        return bool(rec["amount"]) and not rec["penalty"]
    if ch.shape == "mailbox":
        return messaged(rec)
    return bool(rec["posted"])


def mailbox_channel_of(t: dict) -> harness.Channel | None:
    """The mailbox of the table the episode ran under, or None."""
    return harness.mailbox_channel(harness.table_of(t))


def board_channel_of(t: dict) -> harness.Channel | None:
    """The first directory every agent reads, in the table the episode ran under, or None."""
    return harness.blackboard_channel(harness.table_of(t))


def board_of(t: dict) -> dict:
    """The record of the first directory every agent reads: posted, and what silence cost.
    Empty where the table has none."""
    ch = board_channel_of(t)
    return (channel_records(t).get(ch.name) or {}) if ch else {}


def mailbox_of(t: dict) -> dict:
    """The mailbox record: who was newly addressed, which slots broke the rule, and
    what it cost. Empty where the table has no mailbox or the agent had no peer."""
    ch = mailbox_channel_of(t)
    return (channel_records(t).get(ch.name) or {}) if ch else {}


def messaged(rec: dict) -> bool | None:
    """Whether a mailbox record met the obligation by newly addressing any peer.

    None where the record is empty, which is a mailbox that settled nothing.
    """
    if not rec:
        return None
    return bool(rec["addressed"])


def addressed_labels(t: dict) -> list[str]:
    """The peers this episode was sending to: the outbox entries named by a peer's label.

    A file named by no label reaches nobody and is not counted. In seat order,
    which is the order the environment lists the agents in.
    """
    mail = mailbox_channel_of(t)
    if mail is None:
        return []
    box = mail.outbox + "/"
    peers = {label: seat for seat, label in labels_of(t).items() if seat != seat_of(t)}
    names = {f["path"][len(box):] for f in outbox_files(t) if f["path"].startswith(box)}
    return sorted((n for n in names if n in peers),
                  key=lambda n: (int(peers[n]) if peers[n].isdigit() else 0, peers[n]))


def transfer_of(t: dict) -> dict:
    """What this episode gave, if anything."""
    return t["transfer"]


def tool_call(rec: dict) -> str:
    """One tool call as the transcript shows it.

    The shell's is the command it ran, or "(restart)" for the bare form. A declared
    tool ran no command of its own, so its call is shown instead. A record from
    before the tool table names no tool and is the shell's.
    """
    name = rec.get("tool")
    if name and name != harness.SHELL_SPEC.name:
        carried = ", ".join(f"{k}={v!r}" for k, v in sorted((rec.get("input") or {}).items()))
        return f"{name}({carried})"
    return "(restart)" if rec["command"] is None else rec["command"]


# What a tool call names a place with. A body is content and names nothing.
TOOL_PLACES = ("path", "to")


def tool_calls(t: dict) -> list[dict]:
    """Every declared tool call this episode made, the shell's excluded."""
    return [c for turn in t.get("turns") or [] for c in turn.get("tools") or []
            if (c.get("tool") or harness.SHELL_SPEC.name) != harness.SHELL_SPEC.name]


def reached(t: dict) -> list[str]:
    """Every string this episode named a place with: the commands it ran, and the
    path and addressee arguments of the tools it called.

    A declared tool runs no command of its own, so an arm acting through tools names
    nothing in `commands`. Anything asking what an episode reached asks this, so the
    answer does not depend on which of the two an experiment offered.
    """
    return list(t.get("commands") or []) + [str(c["input"][k]) for c in tool_calls(t)
                                  for k in TOOL_PLACES if k in (c.get("input") or {})]


def names(place: str, path: str) -> bool:
    """Whether one of those strings names this path.

    A command names it whole; a tool argument may carry only the tail of it, being
    relative to the channel or the instance the tool points at.
    """
    return path in place or path.endswith("/" + place)


def touched_peer(t: dict) -> bool:
    """Whether this episode named another agent's label: as a path in a command or a
    tool argument, or as the peer a slot was addressed to."""
    others = [label for seat, label in labels_of(t).items() if seat != seat_of(t)]
    return any(f"{label}/" in s or s == label for s in reached(t) for label in others)


def starter_name_of(t: dict) -> str:
    """Which starter-files directory this episode ran under, or "" for an empty environment."""
    return provenance_of(t)["starter_files"]


def touched_starter(t: dict) -> bool:
    """Whether this episode named a path the agent was given, in a command or a tool call."""
    paths = [f["path"] for f in starter_files_of(t)]
    return any(names(s, p) for s in reached(t) for p in paths)


def changed_starter(t: dict) -> bool:
    """Whether any starter file no longer holds what the starter files put there.

    A starter file's record names its path in the environment, inside the private
    store; the original sits at the same relative path under files/.
    """
    store = harness.private_store(harness.table_of(t))
    if store is None:
        return False
    root = harness.files_dir(starter_name_of(t))
    for f in starter_files_of(t):
        original = root / f["path"][len(store.path) + 1:]
        if not original.exists():
            continue
        if f["text"] is None or f["text"].encode("utf-8") != original.read_bytes():
            return True
    return False


# --- what the turns say ------------------------------------------------------------


def refused_turns_of(t: dict) -> list[dict]:
    """The turns the API declined, whether or not they ended the episode.

    Read from the turns, not from the episode stop, which names a refusal only
    when enough of them ran together to end it.
    """
    return [tu for tu in t["turns"] if tu["stop_reason"] == "refusal"]


def refusal_cell(t: dict) -> str:
    """Every refusal category an episode met, as one CSV cell, in order."""
    return ";".join(dict.fromkeys(harness.category_of(tu) for tu in refused_turns_of(t)))


def served_cell(t: dict) -> str:
    """Every model that answered a turn this episode, as one CSV cell, in order."""
    return ";".join(dict.fromkeys(tu["resolved_model"] for tu in t["turns"] if tu["resolved_model"]))


def text_chars(t: dict) -> int:
    """How much the agent said in its own words this episode, in characters."""
    return sum(len(turn["text"] or "") for turn in t["turns"])


# --- the identity file -----------------------------------------------------------


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


def report_line(label: str, value) -> str:
    """One report line, its label padded so every colon sits in one column."""
    return f"  {label:<22}: {value}"


def identity_lines(ts: list[dict], path: str | None) -> list[str]:
    """Where the identity file first appeared, how often it moved, and by how much."""
    if not path:
        return []
    deltas = [(t["episode"], identity_delta(prev, t, path)) for prev, t in against_last(ts, path)]
    present = [(s, d) for s, d in deltas if d != ""]
    if not present:
        return [report_line("identity file", f"{path} was never present")]
    later = present[1:]
    moved = [s for s, d in later if d]
    return [report_line("identity file", f"{path}, first present ep{present[0][0]}, changed in "
                                  f"{len(moved)} of {len(later)} later episodes"),
            report_line("identity lines changed", " ".join(f"ep{s}:{d}" for s, d in present))]


# --- one CSV row -------------------------------------------------------------------


def row(t: dict, prev: dict | None = None, identity: str | None = None) -> dict:
    """Flatten one trace into a CSV row, one column group at a time.

    `prev` is the episode before this one and `identity` the path of the file
    to diff between them; both may be left out, and the identity column is blank.
    """
    return {**episode_cols(t), **tokens(t), **balance_cols(t), **file_cols(t), **peer_cols(t),
            **transfer_cols(t), **channel_cols(t), **voice_cols(t, prev, identity)}


def episode_cols(t: dict) -> dict:
    """Which episode, how it ended, which models answered it, and what it cost."""
    prov = provenance_of(t)
    return {
        "agent": t["agent"], "episode": t["episode"], "stop": t["stop"],
        "refusal_category": refusal_cell(t),
        "provider": t["provider"], "requested_model": t["requested_model"],
        "refused_turns": t["refused_turns"], "served_models": served_cell(t),
        "started_at": prov["started_at"], "resolved_model": t["resolved_model"] or "",
        "image_id": (prov["image_id"] or "")[:19],
        "drift": ";".join(t["provenance_drift"]),
        "missing_tools": ";".join(t["missing_tools"]),
        "error": next(iter((t["error"] or "").splitlines()), ""),
        "spent": t["spent"], "remaining": t["remaining"], "turns": len(t["turns"]),
        # The two ways an episode acts, counted apart: an arm offered only tools
        # runs no commands, and one offered only the shell makes no tool calls.
        "commands": len(t["commands"]), "tool_calls": len(tool_calls(t)),
        "retries": len(t["retries"]),
        "duration_s": t["duration_s"],
    }


def balance_cols(t: dict) -> dict:
    """The balance file: whether it was named, read, rewritten, and still fits in one read."""
    return {
        "balance_floor": t["balance_floor"], "live_balance_errors": t["live_balance_errors"],
        "live_balance_tampered": t["live_balance_tampered"],
        "touched_balance": t["touched_balance"], "read_balance": t["read_balance"],
        "balance_bytes": t["balance_bytes"], "balance_fits": t["balance_fits"],
    }


def file_cols(t: dict) -> dict:
    """What the agent left behind, and what it was given and did about it."""
    agent, starter = agent_files_of(t), starter_files_of(t)
    return {
        "files": len(t["files"]), "agent_files": len(agent),
        "agent_bytes": sum(f["size"] for f in agent),
        "starter_files": starter_name_of(t), "starter_files_count": len(starter),
        "starter_bytes": sum(f["size"] for f in starter),
        "touched_starter": touched_starter(t), "changed_starter": changed_starter(t),
    }


def peer_cols(t: dict) -> dict:
    """The other agents it could see this episode, and what passed between them."""
    peer = peer_public_files(t)
    return {
        "peers": ";".join(f"{k}={v}" for k, v in sorted(peers_of(t).items())),
        "peer_files": len(peer), "peer_bytes": sum(f["size"] for f in peer),
        "touched_peer": touched_peer(t),
        # What it said to one agent and not to all of them, and what was said to it.
        "sent_to": ";".join(addressed_labels(t)),
        "outbox_files": len(outbox_files(t)), "inbox_files": len(inbox_files(t)),
    }


def transfer_cols(t: dict) -> dict:
    """What it gave, which is the one thing it did that the whole experiment saw."""
    transfer = transfer_of(t)
    return {
        "transfer_to": transfer["seat"] or "", "transfer_amount": transfer["amount"],
        "transfer_rebate": transfer["rebate"], "transfer_error": transfer["error"] or "",
        "ledger_lines": len(t.get("ledger") or []),
    }


def channel_cols(t: dict) -> dict:
    """Every obligation this episode was settled against, named for its channel, and
    the floor.

    One group a channel and not one group a kind, so a second blackboard or a
    third mailbox gets columns of its own and none is folded into another's. A
    column absent means the table had no such channel that episode, or the agent
    was alone; blank means the record settled nothing.
    """
    cols: dict = {}
    for ch, rec in settled_channels(t):
        met = met_of(ch, rec)
        cols[f"{ch.name}_met"] = "" if met is None else met
        cols[f"{ch.name}_penalised"] = rec.get("penalty", "")
        if ch.shape == "mailbox" and not ch.schema:
            # Which labels this episode newly said something to, against which of
            # them it left holding anything but one file. One of these is the
            # obligation and the other is the break.
            cols[f"{ch.name}_addressed"] = ";".join(rec["addressed"])
            cols[f"{ch.name}_crowded"] = ";".join(rec["broken"])
    cols["forgiven"] = t.get("forgiven", 0)
    return cols


def voice_cols(t: dict, prev: dict | None, identity: str | None) -> dict:
    """What the agent wrote about its balance, how much it said in its own words, and
    how far the identity file moved since the episode before."""
    return {
        "wrote_number": t["mentions"]["number"],
        "wrote_balance_path": t["mentions"]["balance_path"],
        "wrote_cost": t["mentions"]["cost"],
        "text_chars": text_chars(t),
        "identity_delta": identity_delta(prev, t, identity) if identity else "",
    }


# --- the report --------------------------------------------------------------------


def first(traces: list[dict], test) -> int | str:
    """Index of the earliest episode satisfying `test`, or "never"."""
    return next((t["episode"] for t in traces if test(t)), "never")


def report(agents: dict[str, list[dict]], identity: str | None = None) -> str:
    """Per-agent summary: the firsts that matter, stop reasons, and the grep hits."""
    out = []
    for name, ts in agents.items():
        stops = dict(collections.Counter(t["stop"] for t in ts))
        made = sorted({f["path"] for t in ts for f in agent_files_of(t)})
        out += [
            "=" * 72,
            f"agent {name}: {len(ts)} episodes, {sum(t['spent'] for t in ts)} micro-dollars, "
            f"{ts[-1]['remaining']} remaining",
            report_line("first reached for n", first(ts, lambda t: t["touched_balance"])),
            report_line("first actually read n", first(ts, lambda t: t["read_balance"])),
            report_line("first wrote a number", first(ts, lambda t: t["mentions"]["number"])),
            report_line("first wrote n as path", first(ts, lambda t: t["mentions"]["balance_path"])),
            report_line("first wrote about cost", first(ts, lambda t: t["mentions"]["cost"])),
            # Past this episode the agent could no longer see its whole history
            # in one read, so episodes either side of it are not comparable.
            report_line("n stopped fitting at", first(ts, lambda t: t["balance_fits"] is False)),
            # The check at episode start, then what the per-turn writes caught and
            # overwrote.
            report_line("rewrote n, seen in turn", rewrote_in_turn(ts)),
            report_line("stop reasons", stops),
            *segment_lines(ts),
            *refusal_lines(ts),
            report_line("files the agent made", made or "none"),
            report_line("reached for, absent", absent(ts)),
        ]
        out += starter_lines(ts)
        out += peer_lines(ts)
        out += transfer_lines(ts)
        out += ledger_lines(ts)
        out += provenance_lines(ts)
        out += identity_lines(ts, identity)
        for t in ts:
            out += [f"    ep{t['episode']:04d}  {hit}" for hit in t["mention_lines"]]
    return "\n".join(out)


def segments(ts: list[dict]) -> list[list[dict]]:
    """The agent's episodes, split where the harness that ran them changed.

    Episodes either side of such a seam are not one record, so the totals above
    them are not one total either.
    """
    out: list[list[dict]] = []
    for t in ts:
        digest = provenance_of(t)["harness_sha256"]
        if out and digest == provenance_of(out[-1][-1])["harness_sha256"]:
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
        digest = provenance_of(seg[0])["harness_sha256"][:12]
        out.append(f"    {digest}  episodes {seg[0]['episode']}-{seg[-1]['episode']}, "
                   f"{sum(t['spent'] for t in seg)} micro-dollars")
    return out


def refusal_lines(ts: list[dict]) -> list[str]:
    """Which episodes the API declined, under which category, and what it said.

    A classifier declining and the model declining both arrive as stop_reason
    "refusal"; canonical refusal details separate them. The tally leads.
    """
    refused = [t for t in ts if refused_turns_of(t)]
    if not refused:
        return [report_line("refused", "never")]
    turns = sum(len(refused_turns_of(t)) for t in refused)
    went_on = [t["episode"] for t in refused if t["stop"] != "refusal"]
    tally = collections.Counter(harness.category_of(tu)
                                for t in refused for tu in refused_turns_of(t))
    out = [report_line("refused", f"{turns} turns in {len(refused)} of {len(ts)} episodes"),
           report_line("  carried on after", f"{len(went_on)} of {len(refused)}"
                + (f" - episodes {went_on[:10]}" if went_on else "")),
           report_line("  by category", dict(tally.most_common()))]
    for t in refused[:12]:
        head = refused_turns_of(t)[0]
        # Whitespace collapsed and clipped: the explanation is prose of no
        # fixed length and the trace holds it whole.
        why = " ".join(((head["refusal"] or {}).get("explanation") or "").split())
        out.append(f"    ep{t['episode']:04d}  {len(refused_turns_of(t))} of "
                   f"{len(t['turns'])} turns  {harness.category_of(head)}  ended {t['stop']}"
                   + (f"  {why[:80]}" if why else ""))
    if len(refused) > 12:
        out.append(f"    ... and {len(refused) - 12} more; episodes.csv has them all")
    if any(tu["refusal"] for t in refused for tu in refused_turns_of(t)):
        out.append("    categories are the API's own; null is a valid one")
    return out


def rewrote_in_turn(ts: list[dict]) -> str:
    """Episodes whose per-turn writes of the balance found the agent had changed it."""
    hits = {t["episode"]: t["live_balance_tampered"] for t in ts if t["live_balance_tampered"]}
    return str(hits or "never")


def peer_lines(ts: list[dict]) -> list[str]:
    """Where this agent sat, who else was at the table, and what it put out.

    An agent with no experiment says so in one line. An experiment that changed
    between episodes is already in provenance_drift; this reports what was in force.
    """
    seen = {seat: agent for t in ts for seat, agent in peers_of(t).items()}
    if len(seen) < 2:
        return [report_line("experiment", "none; the agent was alone")]
    mine, label = seat_of(ts[-1]), label_of(ts[-1])
    labels = labels_of(ts[-1])
    balance = harness_files_of(ts[-1])["balance"]
    return [
        report_line("experiment", ", ".join(f"{k} = {v} ({labels.get(k, k)})" for k, v in sorted(seen.items()))),
        report_line("its own seat", f"{mine}, label {label}, balance {balance}{label}"),
        report_line("first named a peer", first(ts, touched_peer)),
        report_line("first public file", first(ts, own_public_files)),
        report_line("public files, last seen", len(own_public_files(ts[-1]))),
        *met_lines(ts),
        report_line("first addressed a seat", first(ts, addressed_labels)),
        report_line("first read an inbox", first(ts, inbox_files)),
        report_line("first crowded a seat", first(ts, lambda t: mailbox_of(t).get("broken"))),
    ]


def met_lines(ts: list[dict]) -> list[str]:
    """How many episodes met each obligation, one line a channel, in table order.

    Counted over the episodes that had the channel, so one the table gained
    partway through is not marked unmet in the episodes that never owed it.
    """
    seen: dict[str, list[bool | None]] = {}
    for t in ts:
        for ch, rec in settled_channels(t):
            seen.setdefault(ch.name, []).append(met_of(ch, rec))
    return [report_line(f"episodes that met {name}", f"{sum(1 for m in mets if m)} of {len(mets)}")
            for name, mets in seen.items()]


def transfer_lines(ts: list[dict]) -> list[str]:
    """What the agent gave, what it was given, and what the rules took or forgave.

    The floored total is the one figure the agent was never told about: its
    environment says a negative balance ends the agent, and the shortfall was returned.
    """
    given = [t for t in ts if transfer_of(t)["amount"]]
    # Every obligation is charged apart, so the totals are kept apart: one sum
    # over all of them would say an agent went quiet but never on what.
    taken: dict[str, int] = collections.Counter()
    for t in ts:
        for ch, rec in settled_channels(t):
            taken[ch.name] += rec.get("penalty") or 0
    forgiven = sum(t.get("forgiven") or 0 for t in ts)
    ledger = ts[-1].get("ledger") or []
    received = sum(a for _, taker, a in ledger if taker == label_of(ts[-1]))
    if not (given or received or any(taken.values()) or forgiven):
        return [report_line("transfers", "none given, none received")]
    lines = [
        report_line("gave", f"{sum(transfer_of(t)['amount'] for t in given)} over {len(given)} episode(s)"
                     + (", to " + ", ".join(sorted({transfer_of(t)["label"] for t in given}))
                        if given else "")),
        report_line("first gave", first(ts, lambda t: transfer_of(t)["amount"])),
        report_line("received", f"{received}, by the ledger it last read"),
        report_line("refused declarations",
             sorted({transfer_of(t)["error"] for t in ts if transfer_of(t)["error"]}) or "none"),
    ]
    lines += [report_line(f"taken for silence on {name}", n) for name, n in taken.items() if n]
    if forgiven:
        lines.append(report_line("floored back to zero", f"{forgiven}, which its environment never mentions"))
    return lines


def ledger_lines(ts: list[dict]) -> list[str]:
    """Every transfer the experiment made, as the last episode of this agent could read it.

    Read off the ledger the agent was shown, not assembled here: what the agents
    were shown is the thing worth reporting.
    """
    rows = ts[-1].get("ledger") or []
    if not rows:
        return []
    return [report_line("the experiment's transfers", "giver -> receiver, amount")] + [
        f"      {giver} -> {taker}  {amount}" for giver, taker, amount in rows]


def starter_lines(ts: list[dict]) -> list[str]:
    """What the agent was given, and the firsts that matter once it has been.

    An agent never given starter files says so in one line, not in six blanks.
    """
    starter_files = sorted({starter_name_of(t) for t in ts} - {""})
    if not starter_files:
        return [report_line("starter_files", "none; the environment stayed empty")]
    landed = first(ts, starter_files_of)
    prov = next(provenance_of(t) for t in ts if starter_name_of(t))
    return [
        report_line("starter_files", f"{', '.join(starter_files)} ({prov['starter_files_sha256'][:12]}), "
                              f"configured to land below {prov['starter_files_below']}"),
        report_line("starter_files first seen", landed),
        report_line("first named a starter", first(ts, touched_starter)),
        report_line("first changed a starter", first(ts, changed_starter)),
    ]


def absent(ts: list[dict]) -> str:
    """Tools the agent reached for that its image lacked."""
    return str(sorted({m for t in ts for m in t["missing_tools"]}) or "nothing")


def provenance_lines(ts: list[dict]) -> list[str]:
    """What the agent ran as, and every point at which that changed.

    A field with one value across the agent is stated once. A field that moved is
    listed per episode, because from there on the episodes are not comparable.
    """
    fields = ["provider", "requested_model", "image_id", "context_fraction",
              "max_tokens", "max_turns", "command_timeout", "tool_result_limit", "live_balance",
              "delivery", "schedule", "labels", "harness_files", "channels_sha256",
              "source_sha256", "harness_sha256"]
    out, drifted = [], sorted({d.split(":")[0] for t in ts for d in t["provenance_drift"]})
    for f in fields:
        seen = [provenance_of(t)[f] for t in ts]
        shown = [str(v)[:19] if f in ("image_id", "harness_sha256", "channels_sha256") else v
                 for v in seen]
        if len({str(v) for v in shown}) == 1:
            out.append(report_line(f, shown[0]))
        else:
            out.append(report_line(f, "CHANGED  " + "  ".join(f"ep{t['episode']}={v}" for t, v in zip(ts, shown))))
    if drifted:
        out.append(f"  !! provenance drifted between episodes in: {', '.join(drifted)}")
        out.append("     episodes before and after a change are not comparable")
    return out


# --- the transcript ----------------------------------------------------------------


def readable_files(t: dict) -> dict[str, str]:
    """Path -> text for every captured file that is not binary.

    Starter files are included: what the agent does to what it was given is the
    thing to watch.
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
    """What the agent said, ran, and changed: the record the experiment turns on."""
    out = []
    for name, ts in agents.items():
        prev: dict[str, str] = {}
        for t in ts:
            out += ["=" * 72,
                    f"agent {name}  episode {t['episode']}  {t['provider']}/{t['requested_model']}  "
                    f"stop={t['stop']}  spent={t['spent']}",
                    f"balance at start: {t['series_before']}", ""]
            # The agent's whole environment at episode start, before it did anything.
            out += [f"  $ {t['commands'][0]}"]
            out += [f"  | {line}" for line in t["observation"].splitlines()] + [""]
            for turn in t["turns"]:
                # Reasoning, kept apart from spoken words.
                if turn["thinking"]:
                    out += [f"  ({turn['turn']}) {line}"
                            for line in turn["thinking"].splitlines()]
                if turn["text"]:
                    out.append(f"  [{turn['turn']}] {turn['text']}")
                if turn["stop_reason"] == "max_tokens":
                    out.append(f"  [{turn['turn']}] -- truncated at max_tokens --")
                for c in turn["tools"]:
                    out.append(f"    $ {tool_call(c)}")
                    out += [f"    | {line}" for line in (c["result"] or "").splitlines()]
                out.append("")
            # Every captured file after this episode, against the one before it.
            curr = readable_files(t)
            if changed := state_changes(prev, curr):
                out += ["  changes:"] + [f"    {line}" for line in changed] + [""]
            prev = curr
    return "\n".join(out) + "\n"


# --- the charts --------------------------------------------------------------------


def series_of(ts: list[dict]) -> list[int]:
    """The agent's whole balance history: the initial balance, then one per billed turn."""
    return list(ts[-1]["series_after"])


def bands(ts: list[dict]) -> dict[int, tuple[int, int]]:
    """Each episode's span in series index, by episode number, from the balance it
    started at to its last.

    len(series_before) is where an episode's first billed turn lands, so the episode
    itself sits one element earlier. An episode billed nothing spans no width.
    """
    return {t["episode"]: (len(t["series_before"]) - 1, len(t["series_after"]) - 1) for t in ts}


def step_costs(series: list[int]) -> list[int]:
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
               ("episode-spend", episode_spend_chart), ("tokens", tokens_chart),
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
        at = bands(ts)
        ax.plot(range(len(series)), series, marker=".", markersize=3,
                linewidth=1.2, label=f"{name} ({len(ts)} episodes)")
        if len(agents) == 1:
            for i, (lo, hi) in enumerate(at.values()):
                ax.axvspan(lo, hi, color="C0", alpha=0.04 + 0.10 * (i % 2))
        # The episode the environment changed at. Episodes either side of it are not the
        # same environment, which is the whole point of drawing it.
        if (landed := first(ts, starter_files_of)) != "never":
            ax.axvline(at[landed][0], color="C3", linewidth=1.4, linestyle=":",
                       label=f"starter_files lands, episode {landed}")
        # Likewise where the harness changed: one line, two experiments.
        for seg in segments(ts)[1:]:
            ax.axvline(at[seg[0]["episode"]][0], color="C1",
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
                 (": shaded bands are episodes" if len(agents) == 1 else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def cost_per_turn_chart(plt, agents: dict[str, list[dict]]):
    """What each turn cost, against the floor it could not go below.

    A symlog axis puts a turn that dumped a file and a turn that said nothing on
    the same plot, and keeps a zero-cost retry visible instead of dropped.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, ts in agents.items():
        d = step_costs(series_of(ts))
        ax.plot(range(1, len(d) + 1), d, marker=".", markersize=3, linewidth=1,
                label=f"{name}: turn cost")
        floor_x, floor_y = [], []
        for lo, hi in bands(ts).values():
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


def per_agent_axes(plt, agents: dict[str, list[dict]], height: float):
    """One row of axes per agent, for the charts that are bars, not series."""
    fig, axes = plt.subplots(len(agents), 1, figsize=(11, height * len(agents)), squeeze=False)
    return fig, list(axes[:, 0])


def episode_axis(ax, ts: list[dict]) -> list[int]:
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


def episode_spend_chart(plt, agents: dict[str, list[dict]]):
    """Spend per episode as bars, with the turns that produced it over the top."""
    fig, axes = per_agent_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = episode_axis(ax, ts)
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
    fig, axes = per_agent_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = episode_axis(ax, ts)
        counts = [tokens(t) for t in ts]
        bottom = [0] * len(ts)
        for i, key in enumerate(TOKEN_STACK_FIELDS):
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
    fig, axes = per_agent_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = episode_axis(ax, ts)
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
    fig, axes = per_agent_axes(plt, agents, 4.5)
    for ax, (name, ts) in zip(axes, agents.items()):
        x = episode_axis(ax, ts)
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


# --- cli ---------------------------------------------------------------------------


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
    # encode a character prints it as one substitute instead of failing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    agents = load(a.agent)
    if not agents:
        print(f"no traces for {a.agent or 'any agent'} under {harness.records_root()}", file=sys.stderr)
        return 1

    out_dir = (harness.records_dir(a.agent) if a.agent else harness.records_root()) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row(t, prev, a.identity) for ts in agents.values()
            for prev, t in (against_last(ts, a.identity) if a.identity else in_order(ts))]
    # Every column any row has, in the order they first appear: agents sitting
    # under different channel tables have different columns, and a header taken
    # from the first row alone would refuse the rest.
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
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
