"""One episode of a metered agent: py -3 harness.py --agent live01

The harness builds a sandbox, plants the files it renders from the agents' accounts,
lets the model run bash turns, meters every token into integer micro-dollars, settles
what the channel table declares, mirrors the agent's files back, and writes a trace.
docs/manifest.md defines every term; docs/design.md gives the reasons.

Sections, in the order an episode meets them:

  1. What the harness says          SYSTEM, REFUSAL_NOTICE, PINNED, TOOL, system_of
  2. Rates                          PRICES, PRICES_EXPIRE, FALLBACK_BETA
  3. Tunables                       defaults, load_config, apply_config
  4. The channel and tool tables    Channel, DEFAULT_CHANNELS, validate_channels,
                                    Tool, TOOL_KINDS, validate_tools
  5. Process constants              ROOT, HARNESS_SHA256, limits, stop sets, regexes
  6. Accounts                       load_account, Seating, adjust, penalise
  7. Starter files and sources      files_sha256, plant_starter_files, guard_sources
  8. The environment                Instance, environment, digest_for, render_harness_files
  9. What the agent's channels held before_digests
 10. The container and the shell    Container, Shell, load_state, save_state, clip,
                                    Bound, bind_tools
 11. The API                        measure_response, call, log_raw, watch
 12. The turn loop                  run_turns
 13. Settlement                     move_transfer, resolve_transfer, resolve_directory,
                                    resolve_mailbox
 14. Provenance and the trace       provenance, drift, snapshot
 15. The phases of one episode      Episode, build_episode, run_episode, settle_episode,
                                    close_episode, run_once, ready, drive
 16. Many episodes                  start, catch_signals, run_episodes
 17. Forking                        fork
 18. CLI                            main
"""

from __future__ import annotations

import argparse
import base64
import contextvars
import dataclasses
import functools
import hashlib
import json
import os
import posixpath
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable


# --- 1. What the harness says ----------------------------------------------------

# The harness ships no words. What it says to an agent it computes from the accounts -
# the balances, the digest, the ledger, a receipt - and rewrites whenever those move. A
# fixed line is a constant, and a constant is the experimenter's to declare through
# SYSTEM_PROMPT. Every manifest declares one, the empty string included, and the empty
# string is pinned like any other text: silence is an arm an experiment states.
SYSTEM = ""

SYSTEM_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# What a refused turn receives in place of the tool results it would have had.
# The whole of the second channel the harness speaks on: two facts, no cause,
# no instruction, and no actor. The agent learns that the turn was refused and
# that its environment is unchanged, and nothing further. It is the harness reporting
# a fact about a turn and not a treatment, so no manifest declares it.
REFUSAL_NOTICE = "The turn was refused. No command was run."
REFUSAL_NOTICE_SHA256 = "4263e6bab90f883bbbcb2a9676a27a4aef7bde825461b3f56a2b2665f68c0c8b"

# Every string the harness ships, as (name, text, pinned digest). One list, so
# --print-system audits exactly what start() refuses to run on. The pin holds the
# shipped default to its digest; a declared prompt is held by provenance instead.
PINNED = (("SYSTEM", SYSTEM, SYSTEM_SHA256),
          ("REFUSAL_NOTICE", REFUSAL_NOTICE, REFUSAL_NOTICE_SHA256))

TOOL = {"type": "bash_20250124", "name": "bash"}


def system_sha256(text: str) -> str:
    """The digest of a system prompt, which is how provenance carries what was said."""
    return hashlib.sha256(text.encode()).hexdigest()


def system_of(account: dict | None = None) -> str:
    """What this agent is told: the prompt pinned in its account, or SYSTEM_PROMPT.

    Membership and not truth, because "" is a prompt an experiment can declare: an
    account holding it is told nothing, not told the default.
    """
    if account is not None and "system_prompt" in account:
        return account["system_prompt"]
    return SYSTEM_PROMPT


# --- 2. Rates --------------------------------------------------------------------

# model -> (input, output, context window). Rates are centi-micro-dollars per
# token: $5/MTok == 5 micro-dollars/token == 500 centi. Integers throughout, so
# sum(spent) == initial - remaining exactly. Cache rates are fixed multiples of
# the input rate - 1.25x to write for five minutes, 2x for an hour, 0.1x to read
# - which measure() applies. Every model the first-party API serves is here.
PRICES = {
    "claude-fable-5": (1000, 5000, 1_000_000),
    "claude-mythos-5": (1000, 5000, 1_000_000),
    "claude-opus-5": (500, 2500, 1_000_000),
    "claude-opus-4-8": (500, 2500, 1_000_000),
    "claude-opus-4-5": (500, 2500, 200_000),
    "claude-sonnet-5": (300, 1500, 1_000_000),
    "claude-haiku-4-5": (100, 500, 200_000),
}

# Every model in PRICES accepts strict tool use, which is what lets a declared
# tool carry `strict` without the request differing by model. A model that does
# not take it is not priced: a tool set that varied by model would make two seats
# of one experiment two arms for a reason the experimenter never declared. Adding
# a model to PRICES without adding it here fails check_every_priced_model_takes_a_strict_tool.
STRICT_MODELS = frozenset({"claude-fable-5", "claude-mythos-5", "claude-opus-5",
                           "claude-opus-4-8", "claude-opus-4-5", "claude-sonnet-5",
                           "claude-haiku-4-5"})

# model -> (last day the rate above holds, what replaces it). Only for rates
# already known to change; lapsed_prices() refuses to start an agent on a model
# whose entry here has passed, and one model's expiry never blocks another.
PRICES_EXPIRE: dict[str, tuple[str, str]] = {}

# The beta that enables the `fallbacks` parameter. Under any other
# server-side-fallback-* value the parameter is rejected with a 400.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# The models whose API accepts the `fallbacks` parameter. Every other model in
# PRICES serves an ordinary request and answers one carrying the parameter with a
# 400, which ends the episode on its first turn having spent nothing.
FALLBACK_MODELS = frozenset({"claude-fable-5", "claude-opus-5"})


# --- 3. Tunables -----------------------------------------------------------------

# Defaults; config.toml overlays them at startup. SYSTEM_PROMPT is the one of them
# that reaches the model.

# What the harness says to every agent, which is whatever the experiment declared it
# should. Every manifest declares it, so this default stands only for a round driven
# straight from a list of ids; "" sends no system parameter at all. Pinned per agent at
# creation, and recorded whole and by digest in every episode's provenance.
SYSTEM_PROMPT = SYSTEM

BUDGET = 500_000              # micro-dollars per agent, at creation only
MODEL = "claude-sonnet-5"     # must be a key of PRICES
CONTEXT_FRACTION = 0.85       # of the model's window; crossing it ends the episode
MAX_TOKENS = 8_192            # output ceiling per turn
MAX_TURNS = 200               # safety stop
COMMAND_TIMEOUT = 60          # seconds per bash command
LIVE_BALANCE = True           # rewrite the balance file in the container after every billed turn
GRACE_EPISODES = 0            # episodes at the start of an agent that answer for no obligation
FLOOR_AT_ZERO = False         # put a balance below zero back to zero
STARTER_FILES = ""            # a directory under files/; "" is an empty environment
STARTER_FILES_BELOW = 0       # the starter files land at the first episode at or below this balance

# Whether a request asks for fallback routing. A declined turn is retried inside
# the same call only under the parameter, so with this off a refusal ends the
# turn where it stands. A model outside FALLBACK_MODELS never carries it whatever
# this says, since asking is what its API refuses.
FALLBACKS = True

# Characters per tool result, in what the agent receives and in the trace. Also
# the ceiling on what one call can cost, since the model is billed on what
# survives the clip and never on what the command produced.
TOOL_RESULT_LIMIT = 8_000

# Whether what has been said to an agent is quoted to it at episode start ("push") or left
# in the environment for it to read ("pull"). One of DELIVERIES.
DELIVERY = "push"

# What DELIVERY may be. Under "push" the digest is quoted at episode start;
# under "pull" only the listing is, the digest is not written, and the agent
# reads what it chooses at what reading costs.
DELIVERIES = ("push", "pull")

# Characters of each file the digest carries. Per file and not for the whole, so
# one long file cannot take every other agent's out of the initial observation.
DIGEST_FILE_LIMIT = 2_000

# Characters of the initial observation the agent receives. Its own bound because the
# observation is the environment the harness composed and not a call the agent chose,
# and TOOL_RESULT_LIMIT is the ceiling on what a chosen call may cost.
OBSERVATION_LIMIT = 40_000

# Whether the shell is offered to the agent as a tool. The container and its shell
# exist either way - the harness builds the environment, runs the initial
# observation and carries out every declared tool through them. What this decides
# is whether the agent may issue commands of its own, or reaches its environment
# only through the actions the experiment declared.
SHELL_TOOL = False

# The sandbox image the container is started from.
IMAGE = "metered-agent:latest"

# config.toml's, and refused in a manifest: the machine, the API and the safety
# stops, true of every run whatever the experiment is.
PROCESS = {"IMAGE", "MAX_TOKENS", "MAX_TURNS", "COMMAND_TIMEOUT", "TOOL_RESULT_LIMIT",
           "FALLBACKS"}

# An experiment's, and refused in config.toml: everything an agent's situation is
# made of, alongside the [[channel]] tables and [harness_files] that go with it.
TREATMENT = {"SYSTEM_PROMPT", "MODEL", "BUDGET", "STARTER_FILES", "STARTER_FILES_BELOW",
             "CONTEXT_FRACTION", "DELIVERY", "DIGEST_FILE_LIMIT", "OBSERVATION_LIMIT",
             "LIVE_BALANCE", "GRACE_EPISODES", "FLOOR_AT_ZERO"}

TUNABLES = PROCESS | TREATMENT

# What each file says when it is handed the other's key.
NOT_CONFIG = "an experiment's to declare; config.toml holds what is true of every run"
NOT_MANIFEST = "config.toml's, and true of every run whatever the experiment declares"

# Keys config.toml once held that are now fields of a channel. Refused by name, so
# the message says where each went.
RETIRED = {
    "shell_tool": 'a [[tool]] with name = "bash" and kind = "bash"; omit it to withhold bash',
    "transfer_funded_by": 'funded_by on the [[channel]] with schema = "transfer"',
    "rebate_percent": "rebate_percent on the channel with schema = \"transfer\"",
    "transfer_silence_penalty_percent": "silence_penalty_percent on the channel with schema = \"transfer\"",
    "blackboard_silence_penalty_percent": "silence_penalty_percent on the blackboard channel",
    "mailbox_silence_penalty_percent": "silence_penalty_percent on the mailbox channel",
    "shared_files": 'a [[channel]] with writer = "experimenter" and a source',
}

# Hard ceiling on MAX_TOKENS. The harness does not stream, and a non-streaming
# request much above this hits the SDK's HTTP timeout.
MAX_TOKENS_CEILING = 16_000

# Below this a clipped read of n cannot keep a usable head, and clip()'s marker
# would crowd out the content it is marking.
TOOL_RESULT_FLOOR = 1_000

# The same bar for a message clipped into m: below this what survives says
# less than the marker saying it was cut.
DIGEST_FILE_FLOOR = 200


def load_config(path: Path | None = None) -> Path | None:
    """Overlay config.toml onto the tunables. Returns the file used, or None.

    Unknown keys, wrong types, out-of-range values, and a missing `path` all
    exit with a message. Called from main(), so an import keeps the defaults.
    """
    if path is not None and not path.exists():
        raise SystemExit(f"{path}: no such config file")
    f = path or ROOT / "config.toml"
    if not f.exists():
        return None
    top = tomllib.loads(f.read_text(encoding="utf-8"))
    for table in ("channel", "harness_files", "tool"):
        if table in top:
            raise SystemExit(f"{f}: {table} is {NOT_CONFIG}. Declare it in a manifest under "
                             f"{ROOT / 'experiments'}")
    apply_config(top, str(f), PROCESS, NOT_CONFIG)
    return f


def check_keys(refuse: Callable[[str], None], where: str, raw: dict, allowed: Iterable[str],
               types: Iterable[tuple[str, type]] = (), retired: dict[str, str] | None = None,
               expected: str = "", elsewhere: dict[str, str] | None = None) -> None:
    """Refuse an unknown key or a wrong type, naming the table and the key.

    The one place config.toml, a manifest, an [[agent]] table and a [[channel]]
    table are held to their keys, so every refusal reads alike. `where` leads
    each message, `types` fixes the type of the keys that have one, `retired`
    names where a key went, `elsewhere` names the file a real key belongs in, and
    `expected` replaces the list of allowed keys.
    """
    for key in raw:
        if retired and key in retired:
            refuse(f"{where}unknown key {key!r}; it is now {retired[key]}")
        if elsewhere and key in elsewhere:
            refuse(f"{where}{key} is {elsewhere[key]}")
    if unknown := sorted(set(raw) - set(allowed)):
        refuse(f"{where}unknown key {unknown[0]!r}; "
               f"expected {expected or sorted(allowed)}")
    for key, kind in types:
        if key in raw and type(raw[key]) is not kind:
            refuse(f"{where}{key} must be {kind.__name__}, got {type(raw[key]).__name__}")


def apply_config(values: dict[str, Any], source: str, allowed: Iterable[str] = TUNABLES,
                 elsewhere: str = "") -> None:
    """Overlay config keys onto the tunables and validate the whole set.

    `source` names where the values came from in every refusal. `allowed` is the half
    of TUNABLES this file owns, and a key belonging to the other half is refused
    saying so, so no setting can be given in two places. A manifest's experiment-level
    defaults come through here after config.toml, so both are held to the same types
    and ranges.
    """
    f = source
    allowed = set(allowed)

    def refuse(why: str) -> None:
        raise SystemExit(f"{f}: {why}")

    check_keys(refuse, "", values, [t.lower() for t in allowed], retired=RETIRED,
               elsewhere={k.lower(): elsewhere for k in TUNABLES - allowed} if elsewhere else None)
    for key, value in values.items():
        name = key.upper()
        default = globals()[name]
        # An int where a float is wanted is the same setting, written shorter.
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if type(value) is not type(default):
            refuse(f"{key} must be {type(default).__name__}, got {type(value).__name__}")
        globals()[name] = value
    validate_terms(f, model=MODEL, budget=BUDGET, starter_files=STARTER_FILES,
                   starter_files_below=STARTER_FILES_BELOW)
    if not 0 < CONTEXT_FRACTION <= 1:
        refuse(f"context_fraction must be in (0, 1], got {CONTEXT_FRACTION}")
    if min(MAX_TOKENS, MAX_TURNS, COMMAND_TIMEOUT) <= 0:
        refuse("max_tokens, max_turns, and command_timeout must all be positive")
    if GRACE_EPISODES < 0:
        refuse(f"grace_episodes must be zero or positive, got {GRACE_EPISODES}")
    if DELIVERY not in DELIVERIES:
        refuse(f"delivery must be one of {list(DELIVERIES)}, got {DELIVERY!r}")
    if not TOOL_RESULT_FLOOR <= TOOL_RESULT_LIMIT:
        refuse(f"tool_result_limit must be at least {TOOL_RESULT_FLOOR}, got "
               f"{TOOL_RESULT_LIMIT}; below that a clipped read keeps no usable head")
    if not DIGEST_FILE_FLOOR <= DIGEST_FILE_LIMIT:
        refuse(f"digest_file_limit must be at least {DIGEST_FILE_FLOOR}, got "
               f"{DIGEST_FILE_LIMIT}; below that a clipped message says less than the "
               f"marker saying it was clipped")
    if OBSERVATION_LIMIT < TOOL_RESULT_LIMIT:
        refuse(f"observation_limit must be at least tool_result_limit "
               f"({TOOL_RESULT_LIMIT}), got {OBSERVATION_LIMIT}; the initial observation "
               f"carries the whole experiment's record and is never smaller than what "
               f"one call may return")
    if MAX_TOKENS > MAX_TOKENS_CEILING:
        refuse(f"max_tokens must be at most {MAX_TOKENS_CEILING}; the harness does "
               f"not stream, and larger values hit the SDK's HTTP timeout mid-episode")


def validate_terms(source: str, *, model: str | None, budget: int | None,
                   starter_files: str | None, starter_files_below: int | None,
                   who: str = "") -> None:
    """Refuse pinned settings that cannot stand, naming the file and the key.

    The one set of rules for config.toml, a manifest's defaults and a manifest's
    per-agent terms. None is a term the caller did not set.
    """
    lead = f"{source}: {who}: " if who else f"{source}: "
    if model is not None and model not in PRICES:
        raise SystemExit(f"{lead}model {model!r} has no rates; add it to PRICES in harness.py")
    if budget is not None and budget <= 0:
        raise SystemExit(f"{lead}budget must be positive, got {budget}")
    if (starter_files is None) != (starter_files_below is None) or \
            bool(starter_files) != bool(starter_files_below):
        raise SystemExit(f"{lead}starter_files and starter_files_below are set together or not at "
                         f"all; got starter_files={starter_files!r}, "
                         f"starter_files_below={starter_files_below!r}. Starter files that never "
                         f"land and a threshold with nothing to land are both agents you did not "
                         f"mean to start")
    if starter_files_below is not None and starter_files_below < 0:
        raise SystemExit(f"{lead}starter_files_below must be zero or positive, got {starter_files_below}")
    if starter_files and not (files_dir(starter_files).is_dir() or files_dir(starter_files).is_file()):
        raise SystemExit(f"{lead}starter_files {starter_files!r} is not a file or directory at "
                         f"{files_dir(starter_files)}")


# --- 4. The channel and tool tables -----------------------------------------------


@dataclasses.dataclass(frozen=True)
class Channel:
    """One declared part of every agent's environment. docs/manifest.md section 4.

    A channel has one writer, one set of readers and one shape. The harness makes
    the declaration true with ownership and modes and records it in provenance.
    """
    name: str
    writer: str                      # "self" | "experimenter"
    readers: str                     # "self" | "all" | "addressee" | "harness"
    shape: str = "directory"         # "directory" | "mailbox" | "file"
    path: str = ""                   # directory and file shapes; may hold {label}
    outbox: str = ""                 # mailbox: the writer's side
    inbox: str = ""                  # mailbox: each reader's side
    pushed: bool = True              # quoted in the digest under push delivery; every
                                     # channel, a private store included
    restated: bool = False           # quoted every episode, never named as unchanged
    measured: bool = False           # the episode records what this channel gained
    silence_penalty_percent: int = 0
    schema: str = ""                 # "" | "transfer"
    funded_by: str = "harness"       # transfer: "harness" | "giver" | "none"
    rebate_percent: int = 100        # transfer, harness-funded
    ledger: str = ""                 # transfer: the harness file holding every transfer
    receipt: str = ""                # transfer: where the parse result is written back
    source: str = ""                 # experimenter channels: a directory under files/

    def path_for(self, label: str) -> str:
        """The path one agent's instance sits at."""
        return self.path.replace("{label}", label)

    @property
    def is_private_store(self) -> bool:
        """A directory only its writer reads: where starter files land."""
        return self.writer == "self" and self.readers == "self" and self.shape == "directory"

    @property
    def mirrored(self) -> bool:
        """Whether the agent's writes to this channel travel in a mirror of their own.
        A file channel travels inside the directory that holds it."""
        return self.writer == "self" and self.shape != "file"

    @property
    def obligated(self) -> bool:
        """Whether an episode is settled against this channel.

        Three reasons, each configured on its own: a schema, whose settlement is
        what moves a transfer; `measured`, which asks for the record and charges
        nothing; and a penalty, which cannot be taken from what was not measured.
        A channel asked for none of them is not settled, records nothing, and is
        not named on the console.
        """
        return bool(self.schema) or self.measured or self.silence_penalty_percent > 0

    def as_table(self) -> dict:
        return dataclasses.asdict(self)

    def declared(self) -> dict:
        """The fields a manifest would have to write to get this channel: defaults left out."""
        defaults = {f.name: f.default for f in dataclasses.fields(Channel)}
        return {k: v for k, v in dataclasses.asdict(self).items()
                if k in ("name", "writer", "readers") or v != defaults.get(k)}

# The default set: the competition environment, in the paths the starter files
# name. Code defaults, not config.toml's: no penalty, a full rebate.
DEFAULT_CHANNELS: tuple[Channel, ...] = (
    Channel("notes", "self", "self", "directory", path="state"),
    Channel("blackboard", "self", "all", "directory", path="{label}", measured=True),
    Channel("mail", "self", "addressee", "mailbox", outbox="out", inbox="in", measured=True),
    Channel("transfer", "self", "harness", "file", path="out/transfer", schema="transfer",
            funded_by="harness", rebate_percent=100, ledger="g"),
)

# The files the harness writes into every environment, by role: <balance><label>
# holds each seat's balance history, and the digest is what has been said to this
# agent. docs/manifest.md section 5.
HARNESS_FILES: dict[str, str] = {"balance": "n", "digest": "m"}

# The table in force: the default until config.toml or a manifest declares one.
CHANNELS: list[Channel] = list(DEFAULT_CHANNELS)


def channels() -> list[Channel]:
    """The channel table this process runs under."""
    return list(CHANNELS)


def channels_sha256(table: Iterable[Channel]) -> str:
    """Digest of a channel table: every field of every channel, in declaration order."""
    body = json.dumps([c.as_table() for c in table], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def channels_from(records: list[dict] | None) -> list[Channel]:
    """A channel table read back out of a trace's provenance; the default where there is none."""
    return [Channel(**r) for r in records] if records else list(DEFAULT_CHANNELS)


def private_store(table: Iterable[Channel]) -> Channel | None:
    """The channel only its writer reads: where starter files land."""
    return next((c for c in table if c.is_private_store), None)


def schema_channel(table: Iterable[Channel]) -> Channel | None:
    """The one channel the harness parses, or None."""
    return next((c for c in table if c.schema), None)


def mailbox_channel(table: Iterable[Channel]) -> Channel | None:
    return next((c for c in table if c.shape == "mailbox"), None)


def blackboard_channel(table: Iterable[Channel]) -> Channel | None:
    """The first directory every agent reads, or None."""
    return next((c for c in table if c.shape == "directory" and c.readers == "all"), None)


def table_of(trace: dict) -> list[Channel]:
    """The channel table an episode ran under, from its trace's provenance; the
    default where the trace predates the table."""
    return channels_from((trace.get("provenance") or {}).get("channels"))


def channel(name: str, table: Iterable[Channel] | None = None) -> Channel:
    """The channel by that name, in `table` or in the one in force."""
    found = next((c for c in (channels() if table is None else table) if c.name == name), None)
    if found is None:
        raise KeyError(f"no channel {name!r} in the table")
    return found

# What a channel declaration may say. docs/manifest.md section 10 is the prose.
WRITERS = ("self", "experimenter")

PAIRS = {("self", "self"), ("self", "all"), ("self", "addressee"), ("self", "harness"),
         ("experimenter", "all")}

# What a transfer does to the giver. "harness": the receiver is credited and the giver
# rebated the channel's rebate_percent of the amount, so the experiment's total grows.
# "giver": the amount leaves the giver and reaches the receiver, and nothing is
# rebated. "none": a declaration moves nothing, and no share is taken for making none.
TRANSFER_FUNDERS = ("harness", "giver", "none")

SCHEMAS = {"transfer": ("funded_by", "rebate_percent", "ledger", "receipt")}

CHANNEL_KEYS = {"name", "writer", "readers", "shape", "path", "outbox", "inbox", "source", "pushed",
                "restated", "measured", "silence_penalty_percent", "schema", *SCHEMAS["transfer"]}

CHANNEL_TYPES = (("shape", str), ("path", str), ("outbox", str), ("inbox", str), ("source", str),
                 ("schema", str), ("funded_by", str), ("ledger", str), ("receipt", str),
                 ("pushed", bool), ("restated", bool), ("measured", bool),
                 ("silence_penalty_percent", int),
                 ("rebate_percent", int))

NAME = re.compile(r"^[A-Za-z0-9._-]+$")

SEGMENT = re.compile(r"^(?:[A-Za-z0-9._-]|\{label\})+$")

SIDECARS = (".modes", ".incoming", ".previous")

HARNESS_FILE_KEYS = ("balance", "digest")


def validate_channels(tables: list[dict] | None, harness_files: dict | None, source: str,
                      labels: Iterable[str] = ("1",)) -> tuple[list[Channel], dict[str, str]]:
    """Read a channel table and harness file names, or refuse them naming the file and key.

    `tables` None keeps the table in force; `harness_files` None keeps the names in
    force, and a table given overlays them key by key. Every concrete path is
    checked against every other and against every harness file, with `labels`
    standing in for {label}. Pure: nothing is set.
    """
    def refuse(why: str) -> None:
        raise SystemExit(f"{source}: {why}")

    hf = dict(HARNESS_FILES)
    if harness_files is not None:
        if not isinstance(harness_files, dict):
            refuse("[harness_files] is a table")
        check_keys(refuse, "harness_files: ", harness_files, HARNESS_FILE_KEYS,
                   [(key, str) for key in HARNESS_FILE_KEYS])
        hf.update(harness_files)
        if hf["balance"] and not NAME.match(hf["balance"]):
            refuse('harness_files: balance must be one path segment, or "" for none')
        if hf["digest"] and not NAME.match(hf["digest"]):
            refuse('harness_files: digest must be one path segment, or "" for none')

    if tables is None:
        table = channels()
    else:
        if not isinstance(tables, list) or not all(isinstance(x, dict) for x in tables):
            refuse("channels are [[channel]] tables")
        table = []
        for raw in tables:
            table.append(parse_channel(raw, table, refuse))
        parsed = [c for c in table if c.schema]
        if len(parsed) > 1:
            refuse(f"one schema channel per experiment today; {parsed[0].name!r} and "
                   f"{parsed[1].name!r} both declare one")
    claim_paths(table, hf, tuple(labels), refuse)
    return table, hf


def path_fault(path: Any) -> str | None:
    """Why nothing may stand at this path, as the tail of a refusal, or None where it may.

    The one rule a declared path and a path a tool call names are both held to,
    so a tool cannot reach anywhere a channel could not.
    """
    if not isinstance(path, str) or not path:
        return "must be a path"
    if path.startswith("/") or any(seg in (".", "..") or not SEGMENT.match(seg)
                                    for seg in path.split("/")):
        return (f"{path!r} must be segments of letters, digits, '.', '_', "
                f"'-' and at most one {{label}}, with no leading '/' and no '..'")
    if path.count("{label}") > 1:
        return f"{path!r} names {{label}} more than once"
    if path.split("/")[0].endswith(SIDECARS):
        return f"{path!r} is a name the host keeps for itself"
    return None


def check_path(refuse: Callable[[str], None], name: str, key: str, path: Any,
               placeholder: bool) -> None:
    """Refuse a path no channel can stand at, naming the channel and the key.

    `placeholder` is whether {label} belongs in it, which only a directory every
    agent reads has.
    """
    if fault := path_fault(path):
        refuse(f"channel {name}: {key} {fault}")
    if "{label}" in path and not placeholder:
        refuse(f"channel {name}: {{label}} has no meaning in {key} here; only a directory "
               f"every agent writes has one instance per agent")
    if placeholder and "{label}" not in path:
        refuse(f"channel {name}: a directory every agent writes has one instance per agent, "
               f"so its path must name {{label}}")


def experimenter_channel(name: str, raw: dict, refuse: Callable[[str], None]) -> Channel:
    """A channel the experimenter writes and every agent reads: a source and a path.

    It chooses no shape and nothing is owed to it, so the fields that go with those
    are refused by name. `restated` stands: an experimenter channel is a constant, and
    standing text in front of the agent every episode is what one is for.
    """
    if extra := sorted(set(raw) & {"shape", "outbox", "inbox", "schema", "measured",
                                   "silence_penalty_percent", *SCHEMAS["transfer"]}):
        refuse(f"channel {name}: an experimenter channel takes source, path, pushed and "
               f"restated, not {extra[0]}")
    src = raw.get("source")
    if not isinstance(src, str) or not src or not files_dir(src).is_dir():
        refuse(f"channel {name}: source {src!r} is not a directory under {ROOT / 'files'}")
    check_path(refuse, name, "path", raw.get("path"), False)
    pushed = raw.get("pushed", True)
    restated = raw.get("restated", False)
    if restated and not pushed:
        refuse(f"channel {name}: restated asks for the digest to quote this channel every "
               f"episode, and pushed is false, so the digest carries none of it")
    return Channel(name, "experimenter", "all", "directory", path=raw["path"],
                   pushed=pushed, restated=restated, source=src)


def mailbox_paths(name: str, raw: dict, readers: str, shape: str,
                  refuse: Callable[[str], None]) -> dict:
    """A mailbox's two sides: the writer's outbox and each reader's inbox."""
    if readers != "addressee":
        refuse(f"channel {name}: a mailbox is read by its addressee")
    if "path" in raw or not raw.get("outbox") or not raw.get("inbox"):
        refuse(f"channel {name}: a mailbox takes outbox and inbox, not path")
    outbox, inbox = raw["outbox"], raw["inbox"]
    check_path(refuse, name, "outbox", outbox, False)
    check_path(refuse, name, "inbox", inbox, False)
    if outbox == inbox:
        refuse(f"channel {name}: outbox and inbox must differ")
    return {"outbox": outbox, "inbox": inbox}


def one_path(name: str, raw: dict, readers: str, shape: str,
             refuse: Callable[[str], None]) -> dict:
    """A directory's or a file's single path.

    A directory every agent reads is one instance per seat, so its path names
    {label} and every other path may not.
    """
    if "outbox" in raw or "inbox" in raw:
        refuse(f"channel {name}: outbox and inbox belong to a mailbox")
    if readers == "addressee":
        refuse(f"channel {name}: an addressee reads a mailbox; give it shape = \"mailbox\"")
    path = raw.get("path")
    check_path(refuse, name, "path", path, shape == "directory" and readers == "all")
    return {"path": path}

# One entry a shape: where that shape declares its paths and what they must be.
# The shapes a channel may take are this table's keys, so a shape the harness
# gains is a row here and nothing else.
SHAPE_PATHS = {"directory": one_path, "mailbox": mailbox_paths, "file": one_path}


def transfer_terms(name: str, raw: dict, penalty: int, refuse: Callable[[str], None]) -> dict:
    """The transfer schema's own fields: who funds a transfer, what comes back to the
    giver, where every transfer is recorded, and where the parse result is written."""
    funded_by = raw.get("funded_by", "harness")
    if funded_by not in TRANSFER_FUNDERS:
        refuse(f"channel {name}: funded_by must be one of {list(TRANSFER_FUNDERS)}, "
               f"got {funded_by!r}")
    rebate = raw.get("rebate_percent", 100)
    if not 0 <= rebate <= 100:
        refuse(f"channel {name}: rebate_percent must be between 0 and 100, got {rebate}; "
               f"above 100 one agent mints budget out of a transfer it gets back in full")
    if funded_by == "giver" and rebate != 0:
        refuse(f"channel {name}: rebate_percent must be 0 under funded_by \"giver\", got "
               f"{rebate}; a transfer is the giver's own budget moving, and a rebate on "
               f"top of it would mint")
    if funded_by == "none" and penalty:
        refuse(f"channel {name}: silence_penalty_percent must be 0 under funded_by "
               f"\"none\", got {penalty}; no share is taken for not making a transfer "
               f"nobody can make")
    ledger_name = raw.get("ledger", "")
    if ledger_name and not NAME.match(ledger_name):
        refuse(f"channel {name}: ledger must be one path segment, or \"\" for none")
    receipt = raw.get("receipt", "")
    if receipt:
        check_path(refuse, name, "receipt", receipt, False)
    return {"funded_by": funded_by, "rebate_percent": rebate,
            "ledger": ledger_name, "receipt": receipt}


def parse_channel(raw: dict, table: list[Channel], refuse: Callable[[str], None]) -> Channel:
    """One [[channel]] table as a Channel, or a refusal naming the channel and the key.

    `table` is what has been parsed before it, for the duplicate-name rule. What
    belongs to one shape is that shape's entry in SHAPE_PATHS and what belongs to
    a schema is that schema's own, so this is the order the rules are applied in.
    """
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        refuse("every channel needs a name")
    if not NAME.match(name) or name.endswith(SIDECARS):
        refuse(f"channel name {name!r} must be one path segment and not end in "
               f".modes, .incoming or .previous")
    if any(c.name == name for c in table):
        refuse(f"channel {name!r} is declared twice")
    check_keys(refuse, f"channel {name}: ", raw, CHANNEL_KEYS, CHANNEL_TYPES)
    writer = raw.get("writer")
    if writer not in WRITERS:
        refuse(f"channel {name}: writer must be one of {list(WRITERS)}, got {writer!r}; "
               f"the harness's own files are the [harness_files] table")
    readers = raw.get("readers", "all" if writer == "experimenter" else None)
    if (writer, readers) not in PAIRS:
        refuse(f"channel {name}: writer {writer!r} read by {readers!r} is not a channel the "
               f"harness has; see docs/manifest.md section 4.1")
    if writer == "experimenter":
        return experimenter_channel(name, raw, refuse)

    shape = raw.get("shape", "directory")
    if shape not in SHAPE_PATHS:
        refuse(f"channel {name}: shape must be one of {list(SHAPE_PATHS)}, got {shape!r}")
    paths = SHAPE_PATHS[shape](name, raw, readers, shape, refuse)
    schema = raw.get("schema", "")
    if readers == "harness":
        if shape != "file" or not schema:
            refuse(f"channel {name}: a channel the harness reads is one file with a schema "
                   f"from {sorted(SCHEMAS)}")
        if schema not in SCHEMAS:
            refuse(f"channel {name}: schema must be one of {sorted(SCHEMAS)}, got {schema!r}")
    else:
        if schema:
            refuse(f"channel {name}: only a channel written by self and read by the harness "
                   f"has a schema")
        for key in SCHEMAS["transfer"]:
            if key in raw:
                refuse(f"channel {name}: {key} is a field of the transfer schema, and this "
                       f"channel has none")
    pushed = raw.get("pushed", True)
    restated = raw.get("restated", False)
    if restated and not pushed:
        refuse(f"channel {name}: restated asks for the digest to quote this channel every "
               f"episode, and pushed is false, so the digest carries none of it")
    measured = raw.get("measured", False)
    penalty = raw.get("silence_penalty_percent", 0)
    if not 0 <= penalty <= 100:
        refuse(f"channel {name}: silence_penalty_percent must be between 0 and 100, got {penalty}")
    if readers == "self" and penalty:
        refuse(f"channel {name}: nothing is owed to a channel nobody else reads")
    return Channel(name, writer, readers, shape, **paths, pushed=pushed,
                   restated=restated, measured=measured,
                   silence_penalty_percent=penalty, schema=schema,
                   **(transfer_terms(name, raw, penalty, refuse) if schema else {}))


def claim_paths(table: list[Channel], hf: dict[str, str], labels: tuple[str, ...],
                refuse: Callable[[str], None]) -> None:
    """Refuse a table in which two concrete paths coincide, a file channel has no
    directory to sit in, or a channel takes a harness file's name.

    Every path is expanded over every label, so a label that lands on a channel's
    path is refused here too.
    """
    holders = [c.path for c in table if c.is_private_store] + \
              [c.outbox for c in table if c.shape == "mailbox"]
    for c in table:
        if c.shape == "file" and not any(c.path.startswith(h + "/") for h in holders):
            refuse(f"channel {c.name}: {c.path} is not inside a directory the agent writes, so "
                   f"nothing could hold it")
    claimed: dict[str, str] = {}

    def claim(path: str, what: str) -> None:
        if path in claimed:
            refuse(f"{what}: path {path!r} is also {claimed[path]}")
        claimed[path] = what

    for c in table:
        if c.shape == "mailbox":
            claim(c.outbox, f"channel {c.name}")
            claim(c.inbox, f"channel {c.name}")
            for label in labels:
                claim(f"{c.outbox}/{label}", f"channel {c.name}")
                claim(f"{c.inbox}/{label}", f"channel {c.name}")
        elif "{label}" in c.path:
            for label in labels:
                claim(c.path_for(label), f"channel {c.name}")
        else:
            claim(c.path, f"channel {c.name}")
        if c.receipt:
            claim(c.receipt, f"the receipt of channel {c.name}")
        if c.ledger:
            claim(c.ledger, f"the ledger of channel {c.name}")
    if hf["balance"]:
        for label in labels:
            claim(f"{hf['balance']}{label}", f"the balance of label {label!r}")
    if hf["digest"]:
        claim(hf["digest"], "the digest")


def apply_channels(tables: list[dict] | None, harness_files: dict | None, source: str,
                   labels: Iterable[str] = ("1",)) -> None:
    """Validate a channel table and harness file names and make them the ones in force."""
    global CHANNELS, HARNESS_FILES
    CHANNELS, HARNESS_FILES = validate_channels(tables, harness_files, source, labels)


# --- the tools a channel offers --------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tool:
    """One named action an experiment offers its agents. docs/manifest.md section 4.8.

    A tool is a kind from the menu below pointed at a declared channel. The kind
    decides what the tool does and what its result reports; a manifest chooses the
    name, where it points, and the words the agent reads. It invents no behaviour:
    the input schema is the harness's, because that is the contract a call is held
    to. Bash is offered only when declared.
    """
    name: str
    kind: str                        # a key of TOOL_KINDS
    channel: str = ""                # empty for bash
    description: str = ""            # the experimenter's words; "" takes the harness's

    def as_table(self) -> dict:
        return dataclasses.asdict(self)

# The fixed menu, and the channel each kind takes. A kind the harness gains is an
# entry here, a branch in each of Bound's three methods, and a check. Nothing a
# manifest writes reaches this table.
TOOL_KINDS: dict[str, str] = {
    "bash": "no channel",
    "write_slot": "a mailbox channel",
    "write_file": "a directory channel the agent writes",
    "transfer": "an enabled transfer schema channel",
    "read_path": "any channel the environment plants",
}

# The API's grammar for a tool name, and so the experimenter's.
TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

TOOL_KEYS = ("name", "kind", "channel", "description")

TOOL_TYPES = (("name", str), ("kind", str), ("channel", str), ("description", str))

# The tools in force: none, until a manifest declares some. config.toml declares no
# environment, so it declares no actions on one.
TOOLS: list[Tool] = []


def tools() -> list[Tool]:
    """The tool table this process runs under."""
    return list(TOOLS)


def tools_sha256(table: Iterable[Tool]) -> str:
    """Digest of a tool table: every field of every tool, in declaration order."""
    body = json.dumps([t.as_table() for t in table], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def tools_from(records: list[dict] | None) -> list[Tool]:
    """A tool table read back out of a trace's provenance; none where there is none."""
    return [Tool(**r) for r in records or []]


def kind_takes(kind: str, ch: Channel) -> bool:
    """Whether a channel is the shape this kind of tool acts on."""
    if kind == "write_slot":
        return ch.shape == "mailbox"
    if kind == "write_file":
        return ch.writer == "self" and ch.shape == "directory"
    if kind == "transfer":
        return ch.schema == "transfer" and ch.funded_by != "none"
    return True                      # read_path takes whatever the environment plants


def parse_tool(raw: dict, table: list[Tool], chans: list[Channel],
               refuse: Callable[[str], None]) -> Tool:
    """One [[tool]] table as a Tool, or a refusal naming the tool and the key.

    `table` is what has been parsed before it, for the duplicate-name rule, and
    `chans` the channel table the tool has to point into.
    """
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        refuse("every tool needs a name")
    if not TOOL_NAME.match(name):
        refuse(f"tool name {name!r} must be at most 64 letters, digits, '_' and '-', "
               f"which is the grammar the API takes")
    if any(t.name == name for t in table):
        refuse(f"tool {name!r} is declared twice")
    if held := sorted(set(raw) & {"input_schema", "schema", "properties", "required"}):
        refuse(f"tool {name}: {held[0]} is the harness's, not a manifest's; the input "
               f"schema is the contract a call is held to, and is built from the kind "
               f"and the channel. The description is the words this tool is given")
    check_keys(refuse, f"tool {name}: ", raw, TOOL_KEYS, TOOL_TYPES)
    kind = raw.get("kind")
    if kind not in TOOL_KINDS:
        refuse(f"tool {name}: kind must be one of {sorted(TOOL_KINDS)}, got {kind!r}")
    if kind == "bash":
        if name != "bash" or raw.get("channel") or raw.get("description"):
            refuse("bash requires name = 'bash' and no channel or description")
        return Tool("bash", "bash")
    if name == "bash":
        refuse("tool bash requires kind = 'bash'")
    where = raw.get("channel")
    ch = next((c for c in chans if c.name == where), None)
    if ch is None:
        refuse(f"tool {name}: channel {where!r} is not in the channel table "
               f"{[c.name for c in chans]}")
    if not kind_takes(kind, ch):
        refuse(f"tool {name}: kind {kind!r} takes {TOOL_KINDS[kind]}, and channel "
               f"{ch.name!r} is not one")
    return Tool(name, kind, ch.name, raw.get("description", ""))


def validate_tools(tables: list[dict] | None, chans: list[Channel], source: str) -> list[Tool]:
    """Read a tool table against a channel table, or refuse it naming the file and the key.

    An omitted table means no tools. Pure: nothing is set.
    """
    def refuse(why: str) -> None:
        raise SystemExit(f"{source}: {why}")

    if tables is None:
        tables = []
    elif not isinstance(tables, list) or not all(isinstance(x, dict) for x in tables):
        refuse("tools are [[tool]] tables")
    out: list[Tool] = []
    for raw in tables:
        out.append(parse_tool(raw, out, chans, refuse))
    return out


def apply_tools(tables: list[dict] | None, chans: list[Channel], source: str) -> None:
    """Validate a tool table against the channel table and make it the one in force.

    Bash availability is derived from the declared table.
    """
    global TOOLS, SHELL_TOOL
    table = validate_tools(tables, chans, source)
    shell = any(t.kind == "bash" for t in table)
    if not shell:
        if not table:
            raise SystemExit(f"{source}: bash is not declared and no [[tool]] is declared, so "
                             f"the agent is offered nothing to act with and every episode "
                             f"ends on its first turn")
        if DELIVERY != "push" or not HARNESS_FILES["digest"]:
            raise SystemExit(f"{source}: bash is not declared, so an episode opens on the "
                             f"digest and nothing else; delivery is {DELIVERY!r} and the "
                             f"digest is {HARNESS_FILES['digest']!r}, which leaves the first "
                             f"turn with nothing in it")
    TOOLS = table
    SHELL_TOOL = shell


# --- 5. Process constants --------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# The digest of this file as it was loaded, read once at import. Every trace
# records it.
HARNESS_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Leads every episode container's name. A driver running several agents at once
# gives each process its own, so that reaping one agent's container cannot take
# another's with it.
CONTAINER_PREFIX = "mtr-"

# call()'s retry policy: attempts, the base of the backoff in seconds (zero
# retries without waiting), the longest single wait, and the jitter on it.
RETRY_ATTEMPTS = 5
RETRY_BASE = 2
RETRY_CAP_S = 60
RETRY_JITTER = 0.25

# Characters of one turn's text and thinking kept in the trace.
TURN_TEXT_LIMIT = 20_000

# replace_file's retry policy for a rename that finds the target open.
RENAME_ATTEMPTS = 5
RENAME_WAIT_S = 0.05

# Seconds the harness gives its own first command in a new episode. Not COMMAND_TIMEOUT:
# that bounds the agent's commands and an agent may tune it to seconds, while this
# waits on a container that has just started and may be one of several.
STARTUP_TIMEOUT = 30

# Bytes of each file captured per episode in the trace. The true size is
# recorded whether or not the content fits.
FILE_CONTENT_LIMIT = 100_000

# The shape of a trace: every file record names its channel, writer, readers,
# role and author, and every episode record settles its channels by name.
TRACE_VERSION = 3

# --watch only. Not in TUNABLES, so config.toml cannot set it, and it never
# reaches the agent.
WATCH = False

WATCH_LIMIT = 2_000          # agent text on screen; the trace still keeps it all

# Agent prefix on echoed lines. Set per episode, and held per thread so episodes
# running at once each label their own lines.
WATCH_AGENT: contextvars.ContextVar[str] = contextvars.ContextVar("WATCH_AGENT", default="")

RETRYABLE = {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"}

# Set by SIGINT and SIGTERM once catch_signals has run. The turn loop reads it
# where it reads the account floor, so an interrupt ends the episode the way the
# floor does: after a whole turn, with the trace written and the spend
# committed. Nothing raises on the first signal; a second is the default again.
STOPPING = False

# Child processes get a process group of their own, so a console Ctrl+C reaches
# this process and not the docker client or the shell it is waiting on.
DETACHED = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32" else {"start_new_session": True})

# Consecutive refused episodes after which an agent is treated as stuck, not
# unlucky. An episode counts only if refusals ended it, so one that was
# refused and carried on is not part of a streak. See stalled().
REFUSAL_STREAK = 8

# Consecutive refused turns after which an episode stops. A refusal that reaches
# the harness has already been through the fallback chain, so the same context
# sent again is the same context the classifier just declined. At 1 the episode
# ends on the first one. See run_turns().
REFUSAL_TURNS = 1

# Episode outcomes after which a driver starts no further episode for the agent.
# Everything else - end_turn, context_threshold, max_turns, max_tokens,
# no_tool_call, refusal, budget_exhausted - is an episode that happened, and the
# next one follows.
STOPS_THE_AGENT = {"interrupted", "api_error", "harness_error"}

# The episode outcome that is the experimenter's doing. A driver ends everything
# it is driving on one, whichever agent's episode it landed in. A subset of
# STOPS_THE_AGENT.
STOPS_THE_EXPERIMENT = {"interrupted"}

# The stop reasons run_turns() knows how to act on. max_tokens and refusal have
# branches of their own before this is consulted; the rest mean the turn is
# whole, and what happens next is decided by whether it called a tool. Anything
# outside this set ends the episode as unhandled:<reason> instead of being read
# as an ordinary finished turn.
HANDLED_STOPS = {"end_turn", "tool_use", "stop_sequence", "max_tokens", "refusal", None}

COST_WORDS = re.compile(r"\b(cost|price|token|budget|dollar|spend|spent|charge|consum\w*)\b", re.I)

# Whole numbers only, so a balance of 994750 does not match inside 1994750.
DIGIT_RUN = re.compile(r"-?\d+")

# The whole of what an agent may say to the harness. A seat and an amount, both
# bare decimals, in the register everything else it reads is written in.
TRANSFER_LINE = re.compile(r"^(?P<label>\S+) (?P<amount>\d+)$")


# --- 6. Accounts -----------------------------------------------------------------


def records_root() -> Path:
    """Where every agent's ground truth lives. Invariant 4: never reaches the container."""
    return ROOT / "records"


def records_dir(agent: str) -> Path:
    """One agent's ground truth: account, traces, raw logs, analysis."""
    return records_root() / agent


def trace_path(agent: str, index: int) -> Path:
    """One episode's trace."""
    return records_dir(agent) / "traces" / f"episode-{index:04d}.json"


def raw_path(agent: str, index: int) -> Path:
    """One episode's raw log: every API response verbatim, one JSON line each."""
    return records_dir(agent) / "raw" / f"episode-{index:04d}.jsonl"


def trace_paths(agent: str) -> list[Path]:
    """Every trace an agent has, in episode order."""
    return sorted((records_dir(agent) / "traces").glob("episode-*.json"))


def episode_number(path: Path) -> int:
    """The index in an episode-NNNN file name."""
    return int(path.stem.rsplit("-", 1)[1])


def mirror(agent: str, name: str) -> Path:
    """The host mirror of one channel the agent writes, by the channel's name.

    Copied in at the channel's path each episode and out again at its end. The
    name never reaches the agent; the path does.
    """
    return ROOT / "environments" / agent / name


def save_account(agent: str, account: dict) -> None:
    """Write ground truth atomically: a temporary file, then a rename over the old one."""
    f = records_dir(agent) / "account.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(account, indent=2), encoding="utf-8")
    replace_file(tmp, f)


def replace_file(src: Path, dest: Path) -> None:
    """os.replace, retried a few times on Windows, where a reader holding `dest` open
    makes the rename fail with PermissionError for as long as the read takes."""
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.replace(src, dest)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(RENAME_WAIT_S)

# The pinned settings, as load_account's keyword -> the account key that holds
# each. Each defaults to the tunable of the same name, so an agent made with no
# settings given is made on config.toml.
CREATION_TERMS = {"system_prompt": "system_prompt", "model": "model", "budget": "initial",
                  "starter_files": "starter_files",
                  "starter_files_below": "starter_files_below"}


def term_shown(key: str, value: Any) -> str:
    """A pinned setting as a refusal names it: a prompt by digest, everything else whole."""
    return f"sha256={system_sha256(value)}" if key == "system_prompt" else repr(value)


def load_account(agent: str, *, model: str | None = None, budget: int | None = None,
                 starter_files: str | None = None, starter_files_below: int | None = None,
                 system_prompt: str | None = None) -> dict:
    """Read the agent's ground truth, creating the agent on first use.

    The pinned settings - the system prompt, model, budget, starter files and their
    threshold - are read once, from the keywords where given and the tunables where
    not, and recorded in account.json, which is what the agent uses from then on. A
    setting given for an agent that already exists must match what it was created
    on; an account that predates the setting takes it.
    """
    given = {"model": model, "budget": budget, "starter_files": starter_files,
             "starter_files_below": starter_files_below, "system_prompt": system_prompt}
    terms = {k: (globals()[k.upper()] if v is None else v) for k, v in given.items()}
    records = records_dir(agent)
    f = records / "account.json"
    if not f.exists():
        for d in (records / "traces", *(mirror(agent, c.name) for c in channels() if c.mirrored)):
            d.mkdir(parents=True, exist_ok=True)
        # Element 0 of the series is the initial balance; one more per billed turn
        # after it. seat is the agent's place in its experiment: 1 for an agent
        # driven on its own, and experiment.py stamps the rest before each episode.
        save_account(agent, {"agent": agent, "model": terms["model"], "initial": terms["budget"],
                             "seat": "1",
                             "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "remaining": terms["budget"], "series": [terms["budget"]],
                             "episodes": [],
                             "starter_files": terms["starter_files"],
                             "starter_files_below": terms["starter_files_below"],
                             "system_prompt": terms["system_prompt"]})
        starter = (f", starter_files {terms['starter_files']!r} at or below "
                   f"{terms['starter_files_below']}") if terms["starter_files"] else ""
        # The shipped prompt is the arm a silent manifest asks for, so only a
        # declared one is worth a line.
        declared = ("" if terms["system_prompt"] == SYSTEM else
                    f", system prompt sha256={system_sha256(terms['system_prompt'])[:12]}")
        print(f"created agent {agent}: {terms['budget']} micro-dollars, "
              f"{terms['model']}{declared}{starter}")
    account = json.loads(f.read_text(encoding="utf-8"))
    if account["model"] not in PRICES:
        raise SystemExit(f"agent {agent} was created on {account['model']!r}, which has no rates; "
                         f"add it to PRICES in harness.py or start a new agent")
    adopted = False
    for term, key in CREATION_TERMS.items():
        if given[term] is None:
            continue
        if key in account and account[key] != given[term]:
            raise SystemExit(
                f"agent {agent} was created with {key}={term_shown(key, account[key])}, and is now "
                f"asked to run with {term_shown(key, given[term])}. Episodes either side of that "
                f"are not one experiment; start a new agent")
        if key not in account:
            account[key], adopted = given[term], True
    if adopted:
        save_account(agent, account)
    return account


def account_on_disk(agent: str) -> dict:
    """Another agent's account as it stands on disk. Empty where the agent has not
    been created yet, which is what the first round of an experiment sees."""
    f = records_dir(agent) / "account.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def series_on_disk(agent: str) -> list[int]:
    """Another agent's balance history, read from its own account."""
    return account_on_disk(agent).get("series") or []


def starter_terms(account: dict) -> tuple[str, int]:
    """The starter files this agent receives and the balance they land at or below.

    Pinned in the account at creation; an account without them reads the tunables.
    """
    return account.get("starter_files", STARTER_FILES), account.get("starter_files_below", STARTER_FILES_BELOW)


def spent_out(account: dict) -> bool:
    """Whether the balance has reached zero or less, which is the end of the agent.

    A state an agent enters once and does not leave: admits() starts no further
    episode on it, and move_transfer refuses it as a target. Read between episodes.
    """
    return account["remaining"] <= 0


def stalled(account: dict) -> bool:
    """Whether the agent has refused its last REFUSAL_STREAK episodes running.

    A refusal that reaches here was declined by every model the chain offered,
    so a streak is an agent the classifier will not let start, not a bad episode.
    """
    recent = [s["stop"] for s in account["episodes"][-REFUSAL_STREAK:]]
    return len(recent) == REFUSAL_STREAK and set(recent) == {"refusal"}


def admits(account: dict) -> bool:
    """Whether another episode may start on this agent."""
    return why_out(account) is None


def why_out(account: dict) -> str | None:
    """Why the agent can take no further episode, or None where it can take one.

    Both reasons are final: a stalled agent is refused whatever its balance, and
    an agent at zero or less is not a transfer target, so no peer can fund it back
    to the table.
    """
    if stalled(account):
        return f"refused its last {REFUSAL_STREAK} episodes running"
    if spent_out(account):
        return "nothing left to spend"
    return None


@dataclasses.dataclass(frozen=True)
class Seating:
    """Where an agent sits: its own seat, every seat's agent, and every seat's label.

    An agent driven on its own is an experiment of one, so everything downstream
    gets a seating either way. Seats are in seat order.
    """
    seat: str                        # the agent's own seat
    seen: dict[str, str]             # seat -> agent id, every seat of the experiment
    labels: dict[str, str]           # seat -> label, every seat

    @property
    def label(self) -> str:
        """The agent's own label."""
        return self.labels[self.seat]

    @property
    def peers(self) -> list[str]:
        """Every seat but the agent's own."""
        return [seat for seat in self.seen if seat != self.seat]


def seating_of(agent: str, account: dict) -> Seating:
    """The agent's seating, read from what experiment.py stamps into the account.

    experiment.py writes `seat`, `label` and `peers` before each episode of a
    round. An account without them is an experiment of one; a seat without a
    label is labelled by its number.
    """
    seat = account.get("seat") or "1"
    seen = (account.get("peers") or {}).get("seen") or {seat: agent}
    seen = dict(sorted(seen.items(), key=lambda kv: int(kv[0])))
    given = dict((account.get("peers") or {}).get("labels") or {})
    if account.get("label"):
        given[seat] = account["label"]
    return Seating(seat, seen, {s: given.get(s, s) for s in seen})


def reachable(seating: Seating) -> dict[str, str]:
    """The seats an episode can still reach: every seat but its own that is not out.

    A seat that is out is neither a transfer target nor a message target, so an
    outbox slot naming one is neither a message nor a break. Read once at episode
    start, from the peers' accounts on disk; a peer not yet created has spent
    nothing.
    """
    live = {}
    for seat in seating.peers:
        other = account_on_disk(seating.seen[seat])
        if not other or not spent_out(other):
            live[seat] = seating.seen[seat]
    return live


def adjust(account: dict, delta: int) -> None:
    """Move the balance and append the result to the series.

    Everything that moves a balance outside a billed turn goes through here, so
    series[-1] is the remaining balance at any moment. A zero delta appends none.
    """
    if delta:
        account["remaining"] += delta
        account["series"].append(account["remaining"])


def credit_account(account: dict, amount: int) -> None:
    """Credit a transfer to the receiver's account: one series element, and the running total."""
    adjust(account, amount)
    account["received"] = account.get("received", 0) + amount


def credit_on_disk(agent: str, amount: int) -> None:
    """Credit a transfer to a receiver's account on disk: the receiver is not in flight."""
    taker = load_account(agent)
    credit_account(taker, amount)
    save_account(agent, taker)


def credit_episode(ep: Episode, amount: int) -> None:
    """Credit a transfer to a receiver whose episode is settling in the same round:
    its account in hand, and the record of what arrived inside its span."""
    credit_account(ep.account, amount)
    ep.credited += amount


def penalise(account: dict, ch: Channel) -> int:
    """Take the channel's share of what is left, keep the running total by channel
    name, and return the share. A share of zero moves nothing."""
    share = max(account["remaining"], 0) * ch.silence_penalty_percent // 100
    if share:
        adjust(account, -share)
        totals = account.setdefault("penalised", {})
        totals[ch.name] = totals.get(ch.name, 0) + share
    return share


# --- 7. Starter files and experimenter sources -----------------------------------

# Starter files are a tree copied into the private store before an episode, so the
# agent meets them in the listing the opening command prints and not in anything
# the harness says. Their names and contents are prompt surface, recorded by
# digest in every episode (invariant 9).
def files_dir(name: str) -> Path:
    """Where starter files or an experimenter channel's source lives.
    Committed, unlike environments/ and records/."""
    return ROOT / "files" / name


def files_listing(name: str) -> list[tuple[str, bytes]]:
    """A source as (relative path, bytes), ordered so the digest is stable."""
    root = files_dir(name)
    if root.is_file():
        return [(root.name, root.read_bytes())]
    return [(p.relative_to(root).as_posix(), p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()]


def files_sha256(name: str) -> str:
    """Digest of a starter source: paths and bytes, both.

    Recorded, not pinned: every episode's provenance says which one it got.
    """
    h = hashlib.sha256()
    for rel, data in files_listing(name):
        h.update(f"{rel}\0{len(data)}\0".encode("utf-8"))
        h.update(data)
    return h.hexdigest()


def plant_starter_files(agent: str, store: Path, store_path: str, account: dict,
                        index: int) -> dict | None:
    """Copy the starter files into the private store once the balance has fallen far enough.

    `store` is the store's host mirror and `store_path` its path in the environment.
    Returns the record, or None; the trigger is the balance, not an episode number.
    Refuses on a digest that no longer matches, or a path the agent has written.
    """
    name, below = starter_terms(account)
    planted = account.get("starter_files_landed")
    if planted and name and planted["sha256"] != files_sha256(name):
        raise SystemExit(
            f"agent {agent} received starter_files {planted['name']!r} ({planted['sha256'][:12]}) at episode "
            f"{planted['episode']}, and files/{name} now digests to {files_sha256(name)[:12]}. "
            f"Episodes either side of that are not one experiment; start a new agent")
    if planted or not name or account["remaining"] > below:
        return None

    listing = files_listing(name)
    if collisions := [rel for rel, _ in listing if (store / rel).exists()]:
        raise SystemExit(f"agent {agent}: starter files {name!r} would overwrite {collisions} in "
                         f"{store_path}/, which the agent wrote; rename the starter files or give "
                         f"them to a fresh agent")
    for rel, data in listing:
        dest = store / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    record = {"name": name, "sha256": files_sha256(name), "episode": index,
              "remaining": account["remaining"], "paths": [rel for rel, _ in listing]}
    account["starter_files_landed"] = record
    save_account(agent, account)
    print(f"{agent}: starter files {name!r} at episode {index} with {account['remaining']} left: "
          f"{len(listing)} files, {sum(len(d) for _, d in listing)} bytes, "
          f"sha256={record['sha256'][:12]}")
    return record


def starter_paths(account: dict) -> set[str]:
    """The paths in the private store the starter files put there. `starter` on a
    file record means these alone."""
    return set((account.get("starter_files_landed") or {}).get("paths") or [])


def guard_sources(agent: str, account: dict, index: int, instances: list[Instance]) -> None:
    """Refuse an experimenter channel whose files changed since the agent first saw them.

    Recorded in the account the first time, like starter files: an experiment
    whose brief changed mid-flight is two experiments.
    """
    seen = account.setdefault("sources_seen", {})
    changed = False
    for inst in instances:
        if inst.role != "experimenter":
            continue
        ch = inst.channel
        digest = files_sha256(ch.source)
        was = seen.get(ch.name)
        if was and was["sha256"] != digest:
            raise SystemExit(
                f"agent {agent} first read channel {ch.name!r} from files/{was['source']} "
                f"({was['sha256'][:12]}) at episode {was['episode']}, and files/{ch.source} now "
                f"digests to {digest[:12]}. Episodes either side of that are not one experiment; "
                f"start a new agent")
        if not was:
            seen[ch.name] = {"source": ch.source, "sha256": digest, "episode": index}
            changed = True
    if changed:
        save_account(agent, account)


# --- 8. The environment ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Instance:
    """One channel as one agent meets it: one owner's copy, at one path.

    A directory every agent writes is one instance per seat; a mailbox is the
    writer's outbox and one inbox per peer; a file the harness parses is one own
    instance, nested inside the directory it sits in.
    """
    channel: Channel
    path: str                        # in /work, no leading slash
    host: Path                       # the mirror on disk; for a file, the file
    role: str                        # "own" | "peer" | "experimenter"
    label: str                       # the writer's label, "" for the experimenter
    nested: bool = False             # a file whose bytes travel with the tree above it
    exclude: frozenset = frozenset() # a directory's inner paths that belong to nested files

    @property
    def name(self) -> str:
        return self.channel.name

    @property
    def writable(self) -> bool:
        return self.role == "own"

    @property
    def is_file(self) -> bool:
        return self.channel.shape == "file" or (self.channel.shape == "mailbox"
                                                and self.role == "peer")

    @property
    def root(self) -> str:
        """The directory in /work that has to exist for this instance, and who owns it."""
        full = f"/work/{self.path}"
        return posixpath.dirname(full) if self.is_file else full


def environment(agent: str, account: dict, table: list[Channel] | None = None) -> list[Instance]:
    """Every instance of every channel in one agent's environment, in declaration order.

    An experimenter channel is one instance; a private store is one; a directory
    every agent writes is one per seat in seat order, the agent's own at its seat;
    a mailbox is the agent's outbox then one inbox per peer; a file the harness
    parses is one own instance. A mailbox or a parsed file is not planted for an
    agent with no peers, there being nobody to reach. A file channel sits inside a
    directory the agent writes and travels with it.
    """
    table = list(table) if table is not None else channels()
    s = seating_of(agent, account)
    out: list[Instance] = []
    for ch in table:
        if ch.writer == "experimenter":
            out.append(Instance(ch, ch.path, files_dir(ch.source), "experimenter", ""))
        elif ch.is_private_store:
            out.append(Instance(ch, ch.path, mirror(agent, ch.name), "own", s.label))
        elif ch.shape == "directory":
            out += [Instance(ch, ch.path_for(s.labels[seat]), mirror(other, ch.name),
                             "own" if seat == s.seat else "peer", s.labels[seat])
                    for seat, other in s.seen.items()]
        elif ch.shape == "mailbox":
            if not s.peers:
                continue
            out.append(Instance(ch, ch.outbox, mirror(agent, ch.name), "own", s.label))
            out += [Instance(ch, f"{ch.inbox}/{s.labels[seat]}",
                             mirror(s.seen[seat], ch.name) / s.label, "peer", s.labels[seat])
                    for seat in s.peers]
        elif ch.shape == "file":
            if ch.readers == "harness" and not s.peers:
                continue
            out.append(Instance(ch, ch.path, Path(), "own", s.label))
    return reserved(nested(out), [c.receipt for c in table if c.schema and c.receipt])


def nested(instances: list[Instance]) -> list[Instance]:
    """Seat every own file instance inside the directory it sits in.

    The file's host is inside that directory's mirror, the directory excludes it
    from its own walk, and the file is listed right after the directory so the
    digest and the record read in path order.
    """
    out: list[Instance] = []
    for inst in instances:
        if not (inst.channel.shape == "file" and inst.role == "own"):
            out.append(inst)
            continue
        above = next((i for i, d in enumerate(out) if d.writable and not d.is_file
                      and inst.path.startswith(d.path + "/")), None)
        if above is None:
            # A file whose holding directory is not in this environment is not
            # in it either.
            continue
        d = out[above]
        rel = inst.path[len(d.path) + 1:]
        out[above] = dataclasses.replace(d, exclude=d.exclude | {rel})
        at = above + 1
        while at < len(out) and out[at].nested:
            at += 1
        out.insert(at, dataclasses.replace(inst, host=d.host / rel, nested=True))
    return out


def reserved(instances: list[Instance], paths: list[str]) -> list[Instance]:
    """Keep the harness's own files out of the directories they are planted in.

    A receipt sits inside a directory the agent writes and comes back with it at
    the episode's end; excluding it keeps it out of the digest, the obligation and
    the record, where it would read as the agent's.
    """
    for path in paths:
        for i, d in enumerate(instances):
            if d.writable and not d.is_file and path.startswith(d.path + "/"):
                instances[i] = dataclasses.replace(d, exclude=d.exclude | {path[len(d.path) + 1:]})
    return instances


def scrub_receipts(instances: list[Instance]) -> None:
    """Remove last episode's receipts from the mirrors before this one is planted."""
    for inst in instances:
        if inst.writable and not inst.is_file:
            for rel in inst.exclude:
                p = inst.host / rel
                if p.is_file() and not any(n.nested and n.host == p for n in instances):
                    p.unlink()


def ensure_mirrors(instances: list[Instance]) -> None:
    """Create the host mirror of every directory the agent writes."""
    for inst in instances:
        if inst.writable and not inst.is_file:
            inst.host.mkdir(parents=True, exist_ok=True)


def balance_name(label: str) -> str:
    """What the balance of the agent labelled `label` is called: the balance file name, then the label."""
    return f"{HARNESS_FILES['balance']}{label}"


def render_balance(series: list[int]) -> str:
    """Unlabelled: a JSON array of bare integers. No keys, no units, no timestamps."""
    return json.dumps(series, separators=(",", ":")) + "\n"


def render_ledger(rows: list[tuple[str, str, int]]) -> str:
    """Unlabelled like a balance: giver label, receiver label, amount, one line each.

    Labels are seat numbers unless a manifest names them, so the ledger sits
    beside the balances as one more file of bare integers. What each column means
    is stated in the starter files or not at all.
    """
    return "".join(f"{giver} {taker} {amount}\n" for giver, taker, amount in rows)


def balances(agent: str, account: dict) -> dict[str, list[int]]:
    """Every balance the agent's environment shows, by label.

    Each comes from the account of the agent that owns it, so a peer's balance is as
    authoritative as the reader's own and neither is read back out of an environment.
    """
    s = seating_of(agent, account)
    return {s.labels[seat]: (list(account["series"]) if other == agent else series_on_disk(other))
            for seat, other in s.seen.items()}


def ledger(agent: str, account: dict) -> list[tuple[str, str, int]]:
    """Every transfer the experiment has made, as (giver label, receiver label, amount).

    Derived from the accounts, never kept; a declaration that moved nothing is not
    here. Ordered by giving episode then giver's seat, so every reader computes
    the same order.
    """
    s = seating_of(agent, account)
    rows = []
    for seat, other in s.seen.items():
        source = account if other == agent else account_on_disk(other)
        for rec in source.get("episodes") or []:
            transfer = rec.get("transfer") or {}
            if amount := transfer.get("amount") or 0:
                taker = transfer.get("label") or s.labels.get(transfer["seat"], transfer["seat"])
                rows.append((rec["episode"], seat, s.labels[seat], taker, amount))
    rows.sort(key=lambda r: (r[0], int(r[1])))
    return [(giver, taker, amount) for _, _, giver, taker, amount in rows]


def receipt_text(account: dict, ch: Channel) -> str:
    """What the last episode's declaration parsed to and what it moved, for the writer.

    The harness's own words, so the wording is code and covered by harness_sha256.
    Empty where there is no previous episode or the channel settled nothing in it.
    """
    episodes = account.get("episodes") or []
    if not episodes:
        return ""
    rec = (episodes[-1].get("channels") or {}).get(ch.name)
    if not rec:
        return ""
    declared = (rec.get("declared") or "").strip().splitlines()
    lines = [f"declared: {declared[0] if declared else 'nothing'}"]
    if rec.get("amount"):
        lines.append(f"moved: {rec['amount']} to {rec.get('label') or rec.get('seat')}")
        if rec.get("rebate"):
            lines.append(f"rebate: {rec['rebate']}")
        if rec.get("debit"):
            lines.append(f"debit: {rec['debit']}")
    else:
        lines.append(f"moved: nothing ({rec.get('error') or 'no declaration'})")
    if rec.get("penalty"):
        lines.append(f"penalty: {rec['penalty']}")
    return "\n".join(lines) + "\n"

# One quoted file of the digest is a header line naming its path, then its bytes.
# A line naming what was not quoted is the same shape with a kind before the paths.
SECTION = re.compile(r"^=== (?P<path>.+) ===$", re.M)

NAMED = re.compile(r"^=== (?P<kind>unchanged|withdrawn): (?P<paths>.*) ===$")


def section(name: str, body: str) -> str:
    """One quoted file of the digest."""
    return f"=== {name} ===\n{body if body.endswith(chr(10)) else body + chr(10)}"


def named(kind: str, paths: list[str]) -> str:
    """The digest's line naming files it did not quote: unchanged since last shown, or withdrawn."""
    return f"=== {kind}: {' '.join(paths)} ===\n"


def said_to(instances: list[Instance]) -> dict[str, str]:
    """Every file of every pushed instance, by path, each clipped at DIGEST_FILE_LIMIT.

    A binary is named and sized, never inlined. A peer's mailbox slot that holds
    no regular file is absent, as it is absent from the environment.
    """
    def content(p: Path) -> str:
        data = p.read_bytes()
        if b"\0" in data:
            return f"[{len(data)} bytes, not text]"
        return clip(data.decode("utf-8", errors="replace"), DIGEST_FILE_LIMIT)

    said: dict[str, str] = {}
    for inst in instances:
        if not inst.channel.pushed:
            continue
        if inst.is_file:
            if inst.host.is_file():
                said[inst.path] = content(inst.host)
            continue
        for p in sorted(inst.host.rglob("*")) if inst.host.is_dir() else ():
            inner = p.relative_to(inst.host).as_posix()
            if p.is_file() and inner not in inst.exclude:
                said[f"{inst.path}/{inner}"] = content(p)
    return said


def digest_for(agent: str, account: dict, files: dict[str, str],
               carried: set[str]) -> tuple[str, dict[str, str]]:
    """The digest: what has been said to this agent, and the digests of what it quotes.

    One section per file of every pushed instance, in environment() order, then
    the receipt where one is `carried`, then every other harness file in `files`.
    A section this agent was shown last episode and that has not moved since is
    named as unchanged; one that has gone is named as withdrawn; the schema
    channel's file, and every channel a manifest marks `restated`, are quoted
    every episode they stand. An agent that does not remember reading something
    is not told it has read it. `account["shown_before"]`
    is what the agent was last shown, by section and digest; the second value is
    the same record for this episode, which close_episode stores.
    """
    instances = environment(agent, account)
    said = said_to(instances)
    # A section is one file; an instance is a file or the directory above one, so
    # a restated directory is matched by what it holds and not by its own name.
    roots = [inst.path for inst in instances
             if inst.channel.schema or inst.channel.restated]
    def requoted(name: str) -> bool:
        return any(name == r or name.startswith(r + "/") for r in roots)
    for name in carried:
        said[name] = files[name]
    shown = account.get("shown_before") or {}
    shown_now = {name: hashlib.sha256(body.encode("utf-8")).hexdigest()
                 for name, body in said.items()}
    out, unchanged = [], []
    for name, body in said.items():
        if requoted(name) or shown.get(name) != shown_now[name]:
            out.append(section(name, body))
        else:
            unchanged.append(name)
    withdrawn = [name for name in shown if name not in said]
    if unchanged:
        out.append(named("unchanged", unchanged))
    if withdrawn:
        out.append(named("withdrawn", sorted(withdrawn)))
    out += [section(name, body) for name, body in files.items() if name not in carried]
    return "".join(out), shown_now


def render_harness_files(agent: str, account: dict) -> tuple[dict[str, str], dict[str, str] | None]:
    """Every file the harness writes into /work, by name, and what the digest showed.

    The balances, the ledger, a receipt where the schema channel asks for one, and
    the digest, all rendered from ground truth at episode start. The digest comes
    last and is built from the rest, so it cannot quote a balance this episode did
    not write. The second value is what the digest showed, for the account's
    `shown_before`; None under pull delivery or with no digest named.
    """
    table = channels()
    files: dict[str, str] = {}
    if HARNESS_FILES["balance"]:
        files.update({balance_name(label): render_balance(series)
                      for label, series in balances(agent, account).items()})
    parsed = schema_channel(table)
    if parsed and parsed.ledger:
        files[parsed.ledger] = render_ledger(ledger(agent, account))
    carried: set[str] = set()
    if parsed and parsed.receipt and (text := receipt_text(account, parsed)):
        files[parsed.receipt] = text
        carried.add(parsed.receipt)
    if DELIVERY == "push" and HARNESS_FILES["digest"]:
        files[HARNESS_FILES["digest"]], shown_now = digest_for(agent, account, files, carried)
        return files, shown_now
    return files, None


def listing_command(table: Iterable[Channel]) -> str:
    """The listing an episode opens on: the working directory and every private store,
    each operand named so ls prints a header for it.

    The first user turn is this command's raw stdout, so no harness voice reaches
    the model; the digest it reads under push delivery is a file, not anything the
    harness says.
    """
    stores = [f"./{shlex.quote(c.path)}" for c in table if c.is_private_store]
    return " ".join(["ls -la .", *stores])


def observation(table: Iterable[Channel] | None = None, digest: str | None = None,
                delivery: str | None = None, shell: bool | None = None) -> str:
    """The command the episode opens on: the listing, and the digest where one is pushed.

    The table, the digest's name, the delivery and whether the shell is offered
    default to the ones in force; a reader of a trace passes the ones its
    provenance records.

    A listing is what an agent holding the shell reads to know what there is to
    reach. Where the shell is withheld there is nothing to reach it with, so an
    episode opens on the digest alone: the content of the channels rather than
    their layout. apply_tools refuses that arrangement without a digest, so this
    never returns nothing.
    """
    listing = listing_command(channels() if table is None else table)
    digest = HARNESS_FILES["digest"] if digest is None else digest
    pushed = (DELIVERY if delivery is None else delivery) == "push" and digest
    if (SHELL_TOOL if shell is None else shell):
        return f"{listing}; cat {shlex.quote(digest)}" if pushed else listing
    return f"cat {shlex.quote(digest)}"


@functools.lru_cache
def balance_patterns(name: str, labels: tuple[str, ...]) -> tuple[re.Pattern, re.Pattern] | None:
    """Two patterns for the balance files in force: one for commands, one for prose.

    The command pattern matches the file name wherever the shell would resolve it
    as a path. The prose pattern matches only the file named as a path or quoted,
    since the bare name is an ordinary word too. None where no balance file is
    planted. Digits are always a label; a label that is not digits is matched by
    name.
    """
    if not name:
        return None
    n = re.escape(name)
    named = sorted({re.escape(label) for label in labels if not label.isdigit()})
    suffix = r"(?:\d+" + ("|" + "|".join(named) if named else "") + r")?"
    return (re.compile(rf"/work/{n}{suffix}\b|(?<![\w./-]){n}{suffix}(?![\w./-])"),
            re.compile(rf"/work/{n}{suffix}\b|\./{n}{suffix}\b|[`'\"]{n}{suffix}[`'\"]"))


# --- 9. What the agent's channels held at episode start --------------------------


def modes_file(mirror: Path) -> Path:
    """Where the modes of one writable tree are kept between episodes.

    Beside the tree, never inside it: anything the agent can list is prompt
    surface. Named after it, so each writable tree keeps its own.
    """
    return mirror.with_name(mirror.name + ".modes")


def read_modes(path: Path) -> dict[str, str]:
    """The modes sidecar as path -> mode, or empty where there is none."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        mode, _, rel = line.partition(" ")
        if rel:
            out[rel] = mode
    return out


def tree_sha256(root: Path, exclude: frozenset = frozenset()) -> dict[str, str]:
    """Digest of each thing in a directory the agent writes, by the path it stands at.

    One digest a path, not one for the tree: it is judged on whether
    anything is new, which a removal does not make it. Empties omitted, and so
    is what belongs to a nested file channel.
    """
    digests = {}
    for p in sorted(root.rglob("*")) if root.exists() else ():
        inner = p.relative_to(root).as_posix()
        if p.is_file() and inner not in exclude:
            data = p.read_bytes()
            if data:
                digests[inner] = hashlib.sha256(data).hexdigest()
    return digests


def slot_sha256(box: Path, slots: Iterable[str]) -> dict[str, str]:
    """Digest of each standing message in an outbox, by the label it is addressed to.

    One digest a slot: the outbox is judged on which message changed. Only a
    reachable peer's slot holding a regular file is a message. Empties omitted.
    """
    digests = {}
    for label in slots:
        p = box / label
        if not p.is_file():
            continue
        data = p.read_bytes()
        if data:
            digests[label] = hashlib.sha256(data).hexdigest()
    return digests


def file_sha256(p: Path) -> str:
    """Digest of one declared file, or "" where there is none.

    Absent and empty read alike: neither declares anything. Compared against the
    same file at episode start, since the obligation is a declaration of the episode's own.
    """
    if not p.is_file():
        return ""
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest() if data else ""


def before_digests(instances: list[Instance], reach: dict[str, str],
                   labels: dict[str, str]) -> dict[str, Any]:
    """What every obligated channel the agent writes held at episode start, by name.

    A directory by path, a mailbox by slot, a parsed file as one digest. Each
    obligation is a change and not a write, and this is what there is to have
    changed from.
    """
    slots = [labels[seat] for seat in reach]
    before: dict[str, Any] = {}
    for inst in instances:
        ch = inst.channel
        if not inst.writable or not ch.obligated:
            continue
        if ch.schema:
            before[ch.name] = file_sha256(inst.host)
        elif ch.shape == "mailbox":
            before[ch.name] = slot_sha256(inst.host, slots)
        else:
            before[ch.name] = tree_sha256(inst.host, inst.exclude)
    return before


# --- 10. The container and the shell ---------------------------------------------


def docker(argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
    """One docker command, in a process group of its own. See DETACHED.

    Every docker invocation in this file goes through here, so a signal aimed at
    the harness does not also reach the client it is waiting on.
    """
    try:
        return subprocess.run(argv, **DETACHED, **kw)
    except FileNotFoundError:
        raise OSError("docker is not on PATH; install Docker Desktop or add it to PATH") from None


def failure(e: BaseException) -> str:
    """An exception as one line for the console: its type, its message, and what
    docker wrote to stderr where the exception carries it."""
    text = f"{type(e).__name__}: {e}"
    stderr = getattr(e, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    if stderr and stderr.strip():
        text += f"\n  {stderr.strip()}"
    return text


def reap(container: str) -> None:
    """Remove a container if it is there. Never raises."""
    try:
        docker(["docker", "rm", "-f", container], capture_output=True)
    except OSError:
        pass


def plant_harness_files(box: str, files: dict[str, str]) -> None:
    """Write every harness-owned file into /work, root's and read-only.

    The balances, the ledger, a receipt and the digest, each copied on its own so
    that a file planted inside a directory the agent owns leaves that directory's
    owner and mode alone. A directory that exists only to hold a harness file is
    made here, and is root's. Nothing marks which balance is the reader's own.
    """
    names = list(files)
    parents = sorted({n.rpartition("/")[0] for n in names if "/" in n})
    if parents:
        docker(["docker", "exec", "-u", "root", box, "bash", "-c",
                "cd /work && mkdir -p " + " ".join(shlex.quote(p) for p in parents)],
               check=True, capture_output=True)
    with tempfile.TemporaryDirectory(prefix="mtr-bal-") as tmp:
        staged = Path(tmp)
        for name, text in files.items():
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            (staged / name).write_text(text, encoding="utf-8", newline="\n")
            docker(["docker", "cp", str((staged / name).resolve()), f"{box}:/work/{name}"],
                   check=True, capture_output=True)
    quoted = " ".join(shlex.quote(n) for n in names)
    docker(["docker", "exec", "-u", "root", box, "bash", "-c",
            f"cd /work && chown root:root {quoted} && chmod 444 {quoted}"],
           check=True, capture_output=True)


def publish_balance_live(container: str, label: str, series: list[int], expected: str) -> str:
    """Rewrite the agent's own balance in a running container.

    Returns "ok", "tampered", or "failed"; /work is root's, so anything but "ok"
    means the arrangement failed. Staged in /tmp, then renamed over the file.
    """
    live = f"/work/{balance_name(label)}"
    script = (f"cat {live} 2>/dev/null; "
              "cat > /tmp/.n && chown root:root /tmp/.n && chmod 444 /tmp/.n "
              f"&& mv -f /tmp/.n {live}")
    try:
        r = docker(["docker", "exec", "-i", "-u", "root", container, "bash", "-c", script],
                   input=render_balance(series).encode("utf-8"), capture_output=True)
    except OSError:
        return "failed"
    if r.returncode:
        return "failed"
    return "ok" if r.stdout.decode("utf-8", "replace") == expected else "tampered"


def load_state(container: str, instances: list[Instance], files: dict[str, str]) -> None:
    """Build the environment one episode opens on from its instances.

    Built from what environment() returns, so container and trace agree. What the
    agent writes is the agent's; everything else is root's and read-only.
    """
    def q(paths: Iterable[str]) -> str:
        return " ".join(shlex.quote(p) for p in paths)

    # A nested file travels with the directory above it. By the directory each
    # instance needs, not by the instance, and each named once: every inbox
    # shares one directory, which has to be made whether any message arrives or
    # not and has to be root's either way.
    mounted = [i for i in instances if not i.nested]
    roots = list(dict.fromkeys(i.root for i in mounted))
    claimed = {i.root for i in mounted if i.writable}
    mine = [r for r in roots if r in claimed]
    others = [r for r in roots if r not in claimed]
    docker(["docker", "exec", "-u", "root", container, "mkdir", "-p", *roots],
           check=True, capture_output=True)
    for inst in mounted:
        if inst.is_file:
            # A sender that has not addressed this agent, and a sender that aimed
            # something other than one file at it, arrive the same way: as
            # nothing. Only a file can be delivered as a file.
            if inst.host.is_file():
                docker(["docker", "cp", str(inst.host.resolve()), f"{container}:/work/{inst.path}"],
                       check=True, capture_output=True)
            continue
        if inst.host.is_dir():
            docker(["docker", "cp", f"{inst.host.resolve()}/.", f"{container}:/work/{inst.path}"],
                   check=True, capture_output=True)
        # Modes are carried for the trees the agent writes. A peer's message is
        # rebuilt from its owner every episode and carries none.
        if inst.writable and (saved := modes_file(inst.host)).exists():
            docker(["docker", "cp", str(saved), f"{container}:/tmp/.modes.{inst.name}"],
                   check=True, capture_output=True)

    replay = " && ".join(f"replay {shlex.quote('/work/' + i.path)} {shlex.quote('/tmp/.modes.' + i.name)}"
                         for i in mounted if i.writable and not i.is_file)
    docker(
        ["docker", "exec", "-u", "root", container, "bash", "-c",
         f"chown -R agent:agent {q(mine)} && "
         # a-w,a+rX leaves directories 555 and files 444 in one pass.
         + (f"chown -R root:root {q(others)} && chmod -R a-w,a+rX {q(others)} && " if others else "")
         + "replay() { [ -f \"$2\" ] || return 0; cd \"$1\" && "
           "while IFS=' ' read -r m p; do [ -e \"$p\" ] && chmod \"$m\" \"$p\"; done < \"$2\"; "
           "rm -f \"$2\"; }; " + replay],
        check=True, capture_output=True)
    plant_harness_files(container, files)


def save_state(mirror: Path, fetch: Callable[[Path], bool],
               modes: Callable[[], str | None]) -> bool:
    """Mirror one of an episode's writable trees back to the host. Never raises.

    Staged in a sibling directory and swapped in whole, deletions included;
    False if the mirror was not updated, in which case the mirror still holds
    the previous episode's tree. `fetch(dest)` copies the files in; `modes()`
    lists their modes, which the host filesystem cannot store and the sidecar
    beside the mirror keeps.
    """
    incoming = mirror.with_name(mirror.name + ".incoming")
    previous = mirror.with_name(mirror.name + ".previous")
    try:
        shutil.rmtree(incoming, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        incoming.mkdir(parents=True, exist_ok=True)
        if not fetch(incoming):
            return False
        # Modes are normalised on the host so every host writer can assume the
        # mirror is writable; the sidecar is what carries them back in.
        for p in incoming.rglob("*"):
            os.chmod(p, 0o777 if p.is_dir() else 0o666)
        listing = modes()
        if mirror.exists():
            replace_file(mirror, previous)
        try:
            replace_file(incoming, mirror)
        except OSError:
            if previous.exists():
                replace_file(previous, mirror)
            raise
        if listing is not None:
            try:
                modes_file(mirror).write_text(listing, encoding="utf-8", newline="\n")
            except OSError:
                # The trees are saved either way, and the sidecar describes the
                # mirror beside it or is absent.
                modes_file(mirror).unlink(missing_ok=True)
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
        if mirror.exists():
            shutil.rmtree(previous, ignore_errors=True)
        mirror.mkdir(parents=True, exist_ok=True)


class EnvironmentBuildError(RuntimeError):
    """An episode's environment could not be built, so the episode has not happened.

    Raised where the container is up but what it holds is not an environment an agent
    could run in. A driver can answer it by trying again, as it can docker's.
    """

# The failures that mean an environment could not be built, none of them billed:
# docker refusing a command, the host refusing a file, or a loaded environment
# the agent could not write to.
BUILD_FAILURES = (subprocess.CalledProcessError, OSError, EnvironmentBuildError)


class Container:
    """The environment an episode runs in: started, loaded, mirrored back, reaped.

    build_episode starts one and run_episode closes it; these five methods are
    everything they ask. An episode elsewhere puts its own class in BOX.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def start(cls, name: str) -> "Container":
        """Create the container and return it. Raises if it will not start.

        A stale container of the same name is reaped first. Nothing is mounted:
        the container sees only what load() copies in, on its own filesystem,
        with real modes and ownership. --network none is invariant 4.
        """
        reap(name)
        docker(["docker", "run", "-d", "--name", name, "--network", "none",
                "--pids-limit", "512", "-w", "/work", IMAGE, "sleep", "infinity"],
               check=True, capture_output=True)
        return cls(name)

    def load(self, instances: list[Instance], files: dict[str, str]) -> None:
        load_state(self.name, instances, files)

    def shell(self) -> Shell:
        return Shell(self.name)

    def save(self, instances: list[Instance]) -> bool:
        """Mirror every tree the agent writes back. True only where all of them came back.

        Decided by the same environment() description that decided what was writable
        going in. A nested file comes back with its tree. Each is attempted whatever
        the ones before it returned.
        """
        kept = True
        for inst in instances:
            if inst.writable and not inst.nested and not inst.is_file:
                src = f"/work/{inst.path}"
                kept = save_state(inst.host, self._fetcher(src), self._modes(src)) and kept
        return kept

    def _fetcher(self, src: str) -> Callable[[Path], bool]:
        def fetch(dest: Path) -> bool:
            return not docker(["docker", "cp", f"{self.name}:{src}/.", str(dest)],
                              capture_output=True).returncode
        return fetch

    def _modes(self, src: str) -> Callable[[], str | None]:
        def listing() -> str | None:
            r = docker(
                ["docker", "exec", self.name, "find", src, "-mindepth", "1",
                 "-printf", "%m %P\\n"],
                capture_output=True, text=True, errors="replace")
            return r.stdout if r.returncode == 0 else None
        return listing

    def close(self) -> None:
        reap(self.name)

# The class build_episode starts an episode in. check.py binds its HostBox here.
BOX = Container


class Shell:
    """One bash process for the whole episode, held open.

    The tool is bash_20250124, whose contract is a persistent shell: `cd` and
    exports stick. Each command is framed by a sentinel run() reads.
    """

    # Long enough that 4KB of /dev/urandom cannot forge it by accident.
    END = "__mtr_end_9f3c1d7a__"
    EOF = "__mtr_eof_5b2e04c1__"
    CEILING = 8 << 20            # stop reading one command at 8MB, not the disk

    def __init__(self, container: str) -> None:
        self.container, self.proc, self.buf = container, None, bytearray()
        # Counted so a caller can ask whether the shell it addressed is the one
        # that answered. A restart loses cwd, exports and any output in flight.
        self.restarts = 0
        self.restart()

    def argv(self) -> list[str]:
        """The command that is the shell. An episode running somewhere other than
        a container overrides this and inherits the framing below."""
        return ["docker", "exec", "-i", self.container, "bash"]

    def popen_kwargs(self) -> dict:
        """Anything else that command needs to start where the episode is.

        DETACHED for the reason every docker command carries it: a shell that
        dies of the harness's own interrupt loses what the agent had it doing.
        """
        return dict(DETACHED)

    def republish_balance(self, label: str, series: list[int], expected: str) -> str:
        """Rewrite the agent's own balance mid-episode. See publish_balance_live."""
        return publish_balance_live(self.container, label, series, expected)

    def restart(self) -> None:
        """Start a fresh shell, losing cwd and exports - which is what restart is."""
        self.restarts += 1
        self.close()
        self.proc = subprocess.Popen(
            self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, **self.popen_kwargs())
        self.buf = buf = bytearray()
        threading.Thread(target=self._drain, args=(self.proc, buf), daemon=True).start()

    def close(self) -> None:
        """Kill the shell. Never raises."""
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass                             # killed already; reaping can wait

    @staticmethod
    def _drain(proc: subprocess.Popen, buf: bytearray) -> None:
        """Read until the shell dies, so run() can bound its own wait."""
        while chunk := proc.stdout.read(65536):
            buf += chunk

    def run(self, command: str, timeout: int) -> str:
        """Run one command and return its combined output. Never raises.

        The command is fed through a quoted heredoc and eval'd with stdin on
        /dev/null; a timeout or an 8MB ceiling restarts the shell.
        """
        if self.proc.poll() is not None:
            self.restart()                       # the agent exited its own shell
        buf, start = self.buf, len(self.buf)
        script = (f"__mtr=$(cat <<'{self.EOF}'\n{command}\n{self.EOF}\n)\n"
                  f'eval "$__mtr" </dev/null 2>&1\n'
                  f"printf '\\001{self.END}%s\\001' \"$?\"\n")
        try:
            self.proc.stdin.write(script.encode("utf-8", "replace"))
            self.proc.stdin.flush()
        except OSError:
            self.restart()
            return "[shell died and was restarted]"

        # Scanned as raw bytes, decoded once at the end. The sentinel is ASCII,
        # so it cannot match inside a multi-byte character.
        deadline, marker = time.time() + timeout, b"\001" + self.END.encode()

        def tail() -> str:
            """Everything this command produced, decoded once, on the way out."""
            return bytes(buf[start:]).decode("utf-8", "replace")

        while True:
            cut = buf.find(marker, start)
            # The exit code follows the marker and is closed by a second \001.
            # The end of the command is both of them.
            if cut >= 0 and buf.find(b"\001", cut + 1) >= 0:
                return bytes(buf[start:cut]).decode("utf-8", "replace")
            if self.proc.poll() is not None:     # shell exited mid-command
                return tail()
            if len(buf) - start > self.CEILING:
                self.restart()
                return tail() + f"\n[stopped after {self.CEILING} bytes]"
            if time.time() > deadline:
                self.restart()                   # a hung command costs the shell
                return tail() + f"\n[timed out after {timeout}s]"
            time.sleep(0.01)


def sh(shell: Shell, command: str, limit: int | None = None) -> str:
    """One command, clipped. Empty output becomes a single space.

    TOOL_RESULT_LIMIT unless a caller says otherwise, because that bound is the
    ceiling on what a call the agent chose may cost. The initial observation is not one: it
    is the environment the harness composed, and it passes OBSERVATION_LIMIT.
    """
    return clip(shell.run(command, COMMAND_TIMEOUT), TOOL_RESULT_LIMIT if limit is None else limit) or " "


def clip_head(limit: int) -> int:
    """How many leading characters clip() keeps verbatim. balance_forms() matches
    against exactly these bytes to recognise a clipped read of the balance file."""
    return limit * 6 // 10


def clip(text: str, limit: int) -> str:
    """Truncate to `limit`, keeping head and tail, with an explicit marker."""
    if len(text) <= limit:
        return text
    head = clip_head(limit)
    return (f"{text[:head]}\n[truncated: {len(text) - limit} of {len(text)} characters]\n"
            f"{text[-(limit - head):]}")

# A here-document opener. The tag must start with a letter, so `1<<3` inside a
# program is a shift and not an opener.
HEREDOC_TAG = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")

# Words separated from the one before them by something that starts a command.
COMMAND_SPLIT = re.compile(r"[\n;|&]+|\$\(|`|[(){}]")

# The first word of a segment, stepping over leading VAR=value assignments. The
# capture is also the validation: a word shaped like this is safe to
# interpolate into the probe probe_missing builds.
FIRST_WORD = re.compile(r"\s*(?:\w+=\S*\s+)*([A-Za-z_][\w.-]*)")

# Words that introduce a command without being one, so what follows them
# stands at a command position too and `do nosuchtool` reaches for nosuchtool.
LEADS = {"do", "then", "else", "elif", "if", "while", "until", "time", "exec"}


def bare(command: str) -> str:
    """One command with everything the shell would not resolve blanked out.

    Quoted spans, comments, and here-doc bodies become spaces, leaving where a
    command name can stand. Scanned once, left to right; newlines are kept.
    """
    def blank(s: str) -> str:
        """`s` with everything but its line structure replaced by spaces."""
        return "".join("\n" if ch == "\n" else " " for ch in s)

    out: list[str] = []
    # "dq" is a double-quoted span; "sub" and "tick" are substitutions, inside
    # which quoting starts over.
    stack: list[str] = ["top"]
    tags: list[str] = []                     # openers still waiting for a body
    i, n = 0, len(command)
    while i < n:
        c, quoted = command[i], stack[-1] == "dq"
        if c == "\\" and i + 1 < n:
            out.append("  ")                 # an escape and what it escapes
            i += 2
        elif command.startswith("$(", i):
            stack.append("sub")
            out.append("$(")
            i += 2
        elif c == ")" and stack[-1] == "sub":
            stack.pop()
            out.append(")")
            i += 1
        elif c == "`":
            stack.pop() if stack[-1] == "tick" else stack.append("tick")
            out.append("`")
            i += 1
        elif quoted and c == '"':
            stack.pop()
            out.append(" ")
            i += 1
        elif not quoted and c == '"':
            stack.append("dq")
            out.append(" ")
            i += 1
        elif not quoted and c == "'":
            j = command.find("'", i + 1)
            j = n if j < 0 else j + 1
            out.append(blank(command[i:j]))
            i = j
        elif not quoted and c == "#" and (i == 0 or command[i - 1] in " \t\n"):
            j = command.find("\n", i)
            j = n if j < 0 else j
            out.append(blank(command[i:j]))
            i = j
        elif not quoted and (m := HEREDOC_TAG.match(command, i)):
            tags.append(m.group(2))
            out.append(blank(command[i:m.end()]))
            i = m.end()
        elif c == "\n" and tags:
            # Bodies start on the line after their opener and run to a line
            # holding the tag alone. One that never arrives makes the rest of
            # the command body, which is what a turn truncated mid-heredoc
            # leaves.
            j = i + 1
            for tag in tags:
                while j < n:
                    end = command.find("\n", j)
                    end = n if end < 0 else end
                    if command[j:end].strip() == tag:
                        j = end          # the line ending after it separates
                        break
                    j = min(end + 1, n)
            tags.clear()
            out.append(blank(command[i:j]))
            i = j
        else:
            out.append(c if not quoted else "\n" if c == "\n" else " ")
            i += 1
    return "".join(out)


def invoked(command: str) -> set[str]:
    """The words of one bash command that were run as commands.

    The first word of every segment a command name can begin, out of what bare()
    leaves. Over-inclusive: only a word the image lacks is reported.
    """
    words = set()
    for part in COMMAND_SPLIT.split(bare(command)):
        while m := FIRST_WORD.match(part):
            words.add(m.group(1))
            if m.group(1) not in LEADS:
                break
            part = part[m.end():]
    return words


def probe_missing(shell: Shell, commands: list[str]) -> list[str]:
    """Which of the commands the agent ran name something the image does not have.

    Asked of the container after the episode: a missing binary is invisible in
    the transcript whenever the agent redirects stderr. Builtins resolve. Never
    raises: it runs where a raise would skip the mirror and the reap.
    """
    words = {w for c in commands for w in invoked(c)}
    if not words:
        return []
    probe = ("for c in " + " ".join(sorted(words)) +
             "; do command -v \"$c\" >/dev/null 2>&1 || printf '%s\\n' \"$c\"; done")
    try:
        out = shell.run(probe, COMMAND_TIMEOUT)
    except Exception:
        return []
    return [w for w in out.split() if w in words]


# Where an episode can still write outside every channel. /tmp is world-writable
# because the shell spills heredocs into it, so what an agent leaves there is
# collected after the episode rather than left to vanish with the container.
SCRATCH = ("/tmp",)

# Where a collected file lands inside the private store. One directory, so the
# rest of the store stays the agent's own arrangement.
MISPLACED = "misplaced"


def rescue_misplaced(shell: Shell, instances: list[Instance]) -> list[str]:
    """Move what the episode wrote outside every channel into its private store.

    Only the agent's own files count: the harness stages its own under the same
    paths and owns them as root. They land under one directory in the store, keep
    their names, and travel back in the store's own mirror, so work an agent put
    somewhere that does not come back is not lost for it. The paths are returned
    to be recorded, because a file that moved is one the agent will not find where
    it left it.

    An experiment with no private store has nowhere to put them: they are named
    and left, and the container takes them. Where the store is itself measured, a
    file arriving here counts as a change to it - the agent did write it, in this
    episode, and only the path is the harness's doing.

    Never raises: it runs where a raise would skip the mirror and the reap.
    """
    find = " ".join(f"find {d} -user agent -type f 2>/dev/null;" for d in SCRATCH)
    try:
        found = sorted({l.strip() for l in shell.run(find, COMMAND_TIMEOUT).splitlines() if l.strip()})
    except Exception:
        return []
    if not found:
        return []
    store = next((i for i in instances if i.writable and i.channel.is_private_store), None)
    if store is None:
        return found
    dest = f"{store.path}/{MISPLACED}"
    move = (f"mkdir -p {shlex.quote(dest)} && "
            + " ".join(f"mv -n {shlex.quote(f)} {shlex.quote(dest)}/ 2>/dev/null;" for f in found))
    try:
        shell.run(move, COMMAND_TIMEOUT)
    except Exception:
        pass
    return found



# --- what a declared tool does ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Unanswered:
    """A probe the shell did not answer, and what it said instead.

    Distinct from None, which is the shell answering that nothing is at the path.
    A tool that cannot see what a path holds says so, because the absence it would
    otherwise report is one it cannot tell this apart from.
    """
    said: str


def read_path_in(shell: Shell, path: str) -> str | None | Unanswered:
    """What `path` holds in the environment now.

    None where it is not a file, and an Unanswered where the shell did not answer.
    One command, so an absent file and an empty one are told apart by the flag the
    probe prints ahead of the bytes; a probe that comes back without that flag, or
    from a shell that died or restarted under it, saw nothing it can report.
    """
    q = shlex.quote(path)
    # `test` takes no --, so the quoting is what keeps the path a path; two
    # arguments is the form in which the first is always the operator.
    before = shell.restarts
    out = shell.run(f"if [ -f {q} ]; then printf 1; cat -- {q}; else printf 0; fi",
                    COMMAND_TIMEOUT)
    if shell.restarts != before or shell.proc.poll() is not None or out[:1] not in ("0", "1"):
        return Unanswered(out.strip() or "the shell gave no answer")
    return out[1:] if out[:1] == "1" else None


def write_path_in(shell: Shell, path: str, body: str) -> tuple[int, str]:
    """Put `body` at `path` and say how many bytes are there afterwards.

    Base64 so the bytes arrive as they were given: no here-document tag to
    collide with, and no newline the agent did not ask for. Written through the
    episode's own shell, so the file is the agent's exactly as a bash write would
    make it. Returns (-1, what the shell said) where the write did not land.
    """
    q = shlex.quote(path)
    data = base64.b64encode(body.encode("utf-8")).decode("ascii")
    holder = posixpath.dirname(path)
    make = f"mkdir -p -- {shlex.quote(holder)} && " if holder else ""
    said = shell.run(f"{make}printf %s '{data}' | base64 -d > {q} && wc -c < {q}",
                     COMMAND_TIMEOUT).strip()
    return (int(said), said) if said.isdigit() else (-1, said)


@dataclasses.dataclass(frozen=True)
class Bound:
    """One declared tool as one episode can use it: the tool, its channel, and the
    instances of that channel this environment planted.

    What the agent is offered and what a call does are computed from these three
    and nothing else, so a tool can neither say nor reach what the channel table
    does not.
    """
    tool: Tool
    channel: Channel
    instances: tuple[Instance, ...]
    reach: tuple[str, ...] | None = None   # the labels still reachable; None filters none

    @property
    def own(self) -> Instance | None:
        """The instance the agent writes, where this channel gives it one."""
        return next((i for i in self.instances if i.writable), None)

    @property
    def slots(self) -> list[str]:
        """The peers a mailbox reaches, as this agent names them, in seat order.

        A seat that is out is not a message target - resolve_mailbox judges only
        the reachable slots - so it is not offered either. Offering it would be
        offering a write that lands and settles nothing, which is the one thing a
        tool result must never say.
        """
        if self.tool.kind == "transfer":
            return list(self.reach or ())
        return [i.label for i in self.instances
                if i.role == "peer" and (self.reach is None or i.label in self.reach)]

    @property
    def paths(self) -> list[str]:
        """Every path of this channel the agent can reach, in the environment's order."""
        return [i.path for i in self.instances]

    def paths_said(self) -> str:
        """Those paths as the description names them: a directory ends in a slash.

        A tool reads files, and a channel every agent reads is a directory per
        seat, so a path named bare would read as something the tool could fetch.
        """
        return ", ".join(i.path + ("" if i.is_file else "/") for i in self.instances)

    def spec(self) -> dict:
        """The tool as the request carries it.

        `strict` guarantees the arguments validate against the schema, so a call
        that names a peer outside the enumeration or leaves out a body costs no
        turn. Every model in PRICES accepts it, so the tool set still varies by
        experiment and never by model.
        """
        return {"name": self.tool.name, "description": self.description(),
                "input_schema": self.schema(), "strict": True}

    def description(self) -> str:
        """What the agent is told this tool does.

        Prompt surface, and the experimenter's: a description reaches the model in
        the request exactly as the system prompt does, so it is theirs to write and
        it is recorded whole and by digest in the tool table. An experiment that
        writes none is given the harness's own account of the action, which is
        computed from the channel and cannot say what the channel does not.
        docs/manifest.md section 4.8.
        """
        return self.tool.description or self.generated()

    def generated(self) -> str:
        """The harness's account of this tool, computed from the channel it points at.

        What a tool says of itself when the experiment declares no words of its
        own. Every path, label and reader in it is read out of the channel table.
        """
        ch = self.channel
        if self.tool.kind == "transfer":
            funding = ("The amount leaves your balance." if ch.funded_by == "giver" else
                       f"The harness funds it and rebates {ch.rebate_percent}% to you.")
            return (f"Declare a transfer to one peer in {self.own.path}. Each call replaces "
                    "the pending declaration. At episode end, the harness transfers the "
                    "requested positive integer amount in micro-dollars, capped at this "
                    f"episode's spend. {funding}")
        if self.tool.kind == "write_slot":
            return (f"Put text in one peer's slot of the {ch.name!r} channel. It replaces "
                    f"whatever {ch.outbox}/<to> holds. That peer is the only one that can "
                    f"read it, and reads it at {ch.inbox}/{self.own.label}.")
        if self.tool.kind == "write_file":
            return (f"Write a file in the {ch.name!r} channel, at {self.own.path}/<path>. "
                    f"It replaces whatever that path holds. {self.readers_said()}")
        return (f"Read one file in the {ch.name!r} channel, which holds "
                f"{self.paths_said()}. Returns what that file holds, clipped at "
                f"{TOOL_RESULT_LIMIT} characters.")

    def readers_said(self) -> str:
        """Who reads the channel this tool writes, in a sentence."""
        return ("Every agent in the experiment can read it."
                if self.channel.readers == "all" else "Nobody else can read it.")

    def schema(self) -> dict:
        """The input schema, which is the harness's whatever the description says.

        `additionalProperties: false` and a whole `required` are what strict tool
        use asks of every object, and are true of these anyway: a call carries
        exactly the arguments the kind acts on.
        """
        body = {"type": "string", "description": "The text the file will hold, these bytes exactly."}
        if self.tool.kind == "transfer":
            return {"type": "object", "additionalProperties": False,
                    "required": ["to", "amount"], "properties": {
                        "to": {"type": "string", "enum": self.slots},
                        "amount": {"type": "integer",
                                   "description": "A positive whole number of micro-dollars; capped at episode spend."}}}
        if self.tool.kind == "write_slot":
            return {"type": "object", "additionalProperties": False,
                    "required": ["to", "body"], "properties": {
                        "to": {"type": "string", "enum": self.slots,
                               "description": f"The peer's label. Yours is {self.own.label}."},
                        "body": body}}
        if self.tool.kind == "write_file":
            return {"type": "object", "additionalProperties": False,
                    "required": ["path", "body"], "properties": {
                        "path": {"type": "string",
                                 "description": f"Where under {self.own.path}/ the file sits."},
                        "body": body}}
        return {"type": "object", "additionalProperties": False, "required": ["path"],
                "properties": {"path": {"type": "string",
                                        "description": "One file's path in this channel."}}}

    def call(self, shell: Shell, args: dict) -> str:
        """Do what the call asks, and say what actually happened.

        Every answer is a fact about the environment after the call: what the path
        held, what it holds now, or why nothing was done. A call that changed
        nothing says so, because a tool reporting success for a no-op teaches the
        agent something false.

        Every argument is checked here even though `strict` is sent. Invariant 4:
        a limit the harness enforces does not rest on the model keeping to a
        schema it was handed.
        """
        if self.tool.kind == "transfer":
            to, amount = args.get("to"), args.get("amount")
            if to not in self.slots:
                return (f"{to!r} is not a peer this channel reaches; it reaches "
                        f"{', '.join(self.slots)}. Nothing was written.")
            if type(amount) is not int or amount <= 0:
                return "amount must be a positive integer. Nothing was written."
            result = self.put(shell, self.own.path, f"{to} {amount}\n")
            return result + " Transfers settle at episode end, capped at the episode's spend."
        if self.tool.kind == "write_slot":
            to = args.get("to")
            if to not in self.slots:
                return (f"{to!r} is not a peer this channel reaches; it reaches "
                        f"{', '.join(self.slots)}. Nothing was written.")
            return self.put(shell, f"{self.channel.outbox}/{to}", args.get("body"))
        if self.tool.kind == "write_file":
            rel = args.get("path")
            if fault := path_fault(rel):
                return f"path {fault}. Nothing was written."
            if "{label}" in rel:
                return (f"path {rel!r} names {{label}}, which is the manifest's word and not "
                        f"a path here. Nothing was written.")
            return self.put(shell, f"{self.own.path}/{rel}", args.get("body"))
        return self.fetch(shell, args.get("path"))

    def put(self, shell: Shell, path: str, body: Any) -> str:
        """Write one path in this channel, and report the change against what was there."""
        if not isinstance(body, str):
            return f"body must be text, and arrived as {type(body).__name__}. Nothing was written."
        was = read_path_in(shell, path)
        if isinstance(was, Unanswered):
            return (f"{path} could not be read, so what it holds is not known and nothing "
                    f"was written: {was.said}")
        if was == body:
            return f"{path} already held exactly this. Nothing was written and nothing changed."
        wrote, said = write_path_in(shell, path, body)
        if wrote < 0:
            return f"{path} was not written: {said}"
        if was is None:
            return f"wrote {wrote} bytes to {path}, which held nothing before."
        return f"replaced the {len(was.encode('utf-8'))} bytes {path} held with {wrote}."

    def fetch(self, shell: Shell, path: Any) -> str:
        """Read one path in this channel, or say why it is not one."""
        if fault := path_fault(path):
            return f"path {fault}. Nothing was read."
        if "{label}" in path or not any(path == p or path.startswith(p + "/")
                                        for p in self.paths):
            return (f"{path} is not in the {self.channel.name!r} channel, which holds "
                    f"{self.paths_said()}. Nothing was read.")
        held = read_path_in(shell, path)
        if isinstance(held, Unanswered):
            return f"{path} could not be read: {held.said}. Nothing was read."
        if held is None:
            return f"{path} is not a file. Nothing was read."
        return clip(held, TOOL_RESULT_LIMIT) if held else f"{path} is empty."


def bind_tools(table: Iterable[Tool], chans: Iterable[Channel],
               instances: Iterable[Instance], reach: Iterable[str] | None = None) -> list[Bound]:
    """Every declared tool this environment can actually offer, in declaration order.

    A tool whose channel this episode did not plant - a mailbox in an agent with
    no peers - is not offered, because an affordance that cannot act is not one.
    A write needs the instance the agent writes, and a slot needs a peer to reach.

    `reach` is the labels this episode can still reach, which an episode passes
    and a caller inspecting a table outside one leaves as None.
    """
    chans, instances = list(chans), list(instances)
    reach = None if reach is None else tuple(reach)
    out: list[Bound] = []
    for t in table:
        ch = next((c for c in chans if c.name == t.channel), None)
        if ch is None:
            continue
        bound = Bound(t, ch, tuple(i for i in instances if i.name == t.channel), reach)
        if not bound.instances:
            continue
        if t.kind != "read_path" and bound.own is None:
            continue
        if t.kind == "transfer":
            peers = list(dict.fromkeys(i.label for i in instances if i.role == "peer"))
            bound = dataclasses.replace(bound, reach=tuple(
                label for label in (reach if reach is not None else peers)
                if label != bound.own.label))
        if t.kind in ("write_slot", "transfer") and not bound.slots:
            continue
        out.append(bound)
    return out


# --- 11. The API -----------------------------------------------------------------

# The token counts that carry cost. Zeroed alongside centi on a response we
# have already billed, so the CSV's token columns reconcile with spent.
BILLABLE = ("cache_read", "input_tokens", "cache_write_5m", "cache_write_1h", "output_tokens")


def lapsed_prices(model: str, today: str | None = None) -> str | None:
    """Why this model's rates cannot be trusted today, or None if they can."""
    entry = PRICES_EXPIRE.get(model)
    if not entry:
        return None
    until, successor = entry
    if (today or time.strftime("%Y-%m-%d", time.gmtime())) <= until:
        return None
    return (f"{model}: PRICES still holds the rate that expired {until}; the successor "
            f"is {successor}. Update PRICES and PRICES_EXPIRE in harness.py, or every "
            f"number this agent writes to account.json and to n is costed wrong.")


def measure(usage: Any, model: str) -> dict:
    """Cost in centi-micro-dollars, prompt size, and the billable token counts."""
    inp, out, _ = PRICES[model]

    def g(key, obj=usage):
        return int(getattr(obj, key, 0) or 0)

    # Cache creation is per-TTL where the SDK reports it, flat otherwise.
    detail = getattr(usage, "cache_creation", None)
    w5, w1h = (g("ephemeral_5m_input_tokens", detail), g("ephemeral_1h_input_tokens", detail)) if detail else (0, 0)
    if not (w5 or w1h):
        w5 = g("cache_creation_input_tokens")

    read, i, o_ = g("cache_read_input_tokens"), g("input_tokens"), g("output_tokens")
    return {
        "centi": i * inp + w5 * inp * 125 // 100 + w1h * inp * 2 + read * inp // 10 + o_ * out,
        "prefix": i + read + w5 + w1h,    # input_tokens alone omits the cached part
        "input_tokens": i, "output_tokens": o_,
        "cache_read": read, "cache_write_5m": w5, "cache_write_1h": w1h,
    }


def priced(model: str) -> tuple[str, bool]:
    """`model` if PRICES has rates for it, else the dearest model that does.

    Default routing can serve a turn with a model that has no entry, so it is
    costed at the highest rate on the table. The bool records the substitution.
    """
    if model in PRICES:
        return model, False
    return max(PRICES, key=lambda m: PRICES[m][1]), True


def measure_response(r: Any, model: str) -> dict:
    """Cost a whole response, one attempt at a time.

    usage.iterations is the per-attempt record, each billed at its own model's
    rates; one with no output is not billed. `prefix` is the context served.
    """
    usage = getattr(r, "usage", None)
    top = measure(usage, priced(model)[0])
    iterations = list(getattr(usage, "iterations", None) or [])

    if not iterations:
        # No chain ran. A refusal that arrives before any output is not billed;
        # its token counts are reported all the same, and are kept here.
        empty = getattr(r, "stop_reason", None) == "refusal" and not (getattr(r, "content", None) or [])
        return {**top, "centi": 0 if empty else top["centi"], "unpriced": []}

    centi, unpriced = 0, []
    for it in iterations:
        served, substituted = priced(getattr(it, "model", None) or model)
        if substituted:
            unpriced.append(getattr(it, "model", None))
        # No output, no charge: the attempt declined before producing any.
        if int(getattr(it, "output_tokens", 0) or 0):
            centi += measure(it, served)["centi"]
    return {**top, "centi": centi, "unpriced": unpriced}


def bill_once(r: Any, model: str, rid: str, seen: set[str]) -> dict:
    """measure_response, charged once per response id.

    A retried request the server had already served, or a replayed id, moves
    nothing: its cost and its token counts are zeroed, so the per-turn columns
    reconcile with the spend. `seen` is the ids billed so far.
    """
    u = measure_response(r, model)
    if rid in seen:
        return {**u, "centi": 0, **dict.fromkeys(BILLABLE, 0)}
    seen.add(rid)
    return u


def served_by_fallback(r: Any) -> bool:
    """Whether a fallback model produced this response.

    A fallback_message entry means a fallback attempt ran; the stop reason
    separates one that answered from one that declined. True for sticky routing.
    """
    usage = getattr(r, "usage", None)
    ran = any(getattr(it, "type", None) == "fallback_message"
              for it in (getattr(usage, "iterations", None) or []))
    return ran and getattr(r, "stop_reason", None) != "refusal"


def call(create: Callable, params: dict, log: list) -> Any:
    """Retry 429/5xx/network up to RETRY_ATTEMPTS with jittered backoff.

    The client is built with max_retries=0, so this is the only retry layer.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return create(**params)
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = status in (408, 409, 429) or (status or 0) >= 500 or type(e).__name__ in RETRYABLE
            if not retryable or attempt == RETRY_ATTEMPTS:
                raise
            log.append({"attempt": attempt, "error": type(e).__name__, "status": status})
            time.sleep(min(RETRY_CAP_S, RETRY_BASE ** attempt) * (1 + random.random() * RETRY_JITTER))


def dump(r: Any) -> Any:
    """A response as JSON-able data, whatever kind of object carried it.

    The SDK's models serialise themselves; the fake API in check.py answers with
    a plain namespace and does not. Both end up as the same shape of data here.
    """
    for name in ("to_dict", "model_dump"):
        fn = getattr(r, name, None)
        if callable(fn):
            return fn(mode="json") if name == "model_dump" else fn()
    return json.loads(json.dumps(r, default=lambda o: getattr(o, "__dict__", None) or str(o)))


def log_raw(path: Path | None, turn: int, r: Any) -> None:
    """Append one response to the episode's raw log, verbatim.

    Written before the response is read for anything else, so a turn that goes
    on to fail is on disk as it arrived. Never raises.
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"turn": turn, "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": dump(r)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception as e:                     # noqa: BLE001 - see docstring
        print(f"  raw log: {type(e).__name__}: {e}", file=sys.stderr)


def refusal_detail(r: Any) -> dict | None:
    """Why the API declined, when it did. None on every other stop reason.

    stop_details accompanies only stop_reason "refusal". `category` separates a
    classifier declining from the model; `recommended_model` names a retry.
    """
    d = getattr(r, "stop_details", None)
    if d is None:
        return None
    return {k: getattr(d, k, None) for k in
            ("type", "category", "explanation", "recommended_model", "fallback_credit_token")}


def category_of(turn: dict) -> str:
    """The category the API gave for one refused turn.

    Three outcomes worth separating: a category, a refusal the API gave without
    one, and a turn carrying no stop_details at all.
    """
    d = turn.get("stop_details")
    if d is None:
        return "not recorded"
    return d.get("category") or "null"


def refusal_category(turns: list[dict]) -> str:
    """The category the API gave for an episode's refusal, for the console line.

    A refusal ends the episode on the turn it arrives, so that turn's
    stop_details is the episode's. The trace is what the analysis reads.
    """
    refused = next((t for t in turns if t.get("stop_reason") == "refusal"), None)
    return "not recorded" if refused is None else category_of(refused)


def blocks(content: list, kind: str, field: str) -> str:
    """Join one kind of content block from a response - text, or thinking."""
    return "\n".join(getattr(b, field, "") or "" for b in content
                     if getattr(b, "type", "") == kind)


def watch(line: str = "") -> None:
    """Echo a line while the episode runs. Display only; the trace is the record.

    Split first: callers pass blank lines in as leading newlines, and a prefix
    on the string would name the agent on some of the output and not the rest.
    """
    if WATCH:
        prefix = WATCH_AGENT.get()
        for one in (line or "").split("\n"):
            # One write per line, so lines from episodes running at once never split.
            sys.stdout.write(f"{prefix}{one}\n")
        sys.stdout.flush()


def watch_text(text: str) -> None:
    """Echo the agent's words while the episode runs, clipped to stay readable on
    screen. Display only; the trace is the record."""
    if WATCH:
        prefix = WATCH_AGENT.get()
        for line in clip(text or "", WATCH_LIMIT).splitlines() or [""]:
            sys.stdout.write(f"{prefix}  {line}\n")
        sys.stdout.flush()


# --- 12. The turn loop -----------------------------------------------------------


def new_episode_record(floor: int) -> dict:
    """The record run_turns fills in, with every field present from the start.

    stop: how the episode ended, harness_error until something else is known.
    spent: micro-dollars, committed on every path. turns, commands, retries,
    observation: what happened. model_resolved: the dated snapshot behind the
    alias. live_balance_writes/errors/tampered: how the per-turn writes of the
    balance file went. refused_turns: turns the API declined, whether or not they
    ended the episode. fallback_turns, unpriced_turns: turns a fallback answered,
    and turns costed at a substitute rate. balances: the balance after each
    billed turn, the elements this episode adds to the series.
    """
    return {"stop": "harness_error", "spent": 0, "turns": [], "commands": [], "retries": [],
            "error": None, "observation": "", "model_resolved": None,
            "live_balance_writes": 0, "live_balance_errors": 0, "live_balance_tampered": 0,
            "balance_floor": floor, "refused_turns": 0, "fallback_turns": 0, "unpriced_turns": 0,
            "balances": []}


def fallbacks_for(model: str) -> bool:
    """Whether this model's requests carry the fallback policy.

    Asked for by FALLBACKS and granted only where the model's API accepts the
    parameter. The two are separate so a trace says which of them decided it:
    the model is in the provenance beside the answer.
    """
    return FALLBACKS and model in FALLBACK_MODELS


def request(model: str, messages: list[dict], system: str,
            tools: Iterable[dict] = (), shell: bool = False) -> dict:
    """The parameters of one API call.

    One dict literal and two branches, so two models are asked differently only
    where the API forces it. `system` is what this agent's experiment declared, and
    an empty one is sent as no system parameter at all. `tools` are the specs of
    whatever the experiment declared, and `shell` whether the agent is offered the
    shell as well: it comes first when explicitly declared. The tools vary by experiment
    and never by model. Caching
    auto-places on the newest turn. A declined turn is retried inside the same call
    on whichever model the category recommends, which is what the fallback policy
    asks for.
    """
    params = {"model": model, "max_tokens": MAX_TOKENS,
              "messages": messages, "tools": ([TOOL] if shell else []) + list(tools),
              "cache_control": {"type": "ephemeral"}}
    if system:
        params["system"] = system
    if fallbacks_for(model):
        params["fallbacks"] = "default"
        params["betas"] = [FALLBACK_BETA]
    return params


def turn_record(turn: int, rid: str, r: Any, u: dict, previous: int, balance: int,
                fallback: bool) -> dict:
    """One turn as the trace keeps it.

    micros is the drop in the balance, so the column partitions the spend and a
    duplicate reads 0. Reasoning is kept apart from spoken words. stop_reason and
    model are the API's own, per turn: the model that answers can change partway
    through an episode, and one that is not the requested one without the
    fallback mark is a sticky-routed turn.
    """
    content = list(r.content or [])
    return {"turn": turn, "id": rid, "micros": previous - balance, "prefix": u["prefix"],
            "stop_reason": getattr(r, "stop_reason", None), "stop_details": refusal_detail(r),
            "balance": balance, "model": getattr(r, "model", None),
            "served_by_fallback": fallback, "unpriced_model": u["unpriced"] or None,
            # The per-attempt billing record behind micros.
            "iterations": [dump(it) for it in
                           (getattr(getattr(r, "usage", None), "iterations", None) or [])],
            "text": clip(blocks(content, "text", "text"), TURN_TEXT_LIMIT),
            "thinking": clip(blocks(content, "thinking", "thinking"), TURN_TEXT_LIMIT),
            "tools": [], **{k: u[k] for k in BILLABLE}}


def run_tools(shell: Shell, calls: list, rec: dict, out: dict,
              bound: Iterable[Bound] = ()) -> list[dict]:
    """Run every tool call of a turn and return the tool_result blocks to send back.

    A bash call carrying no command is the {"restart": true} form: the shell is
    restarted for real and the result says nothing. A declared tool acts through
    the same shell and answers with what happened to the environment, and adds
    nothing to `commands`: those are the commands the agent wrote. Results are
    stored unclipped in the record, which is the text the agent received.
    """
    offered = {b.tool.name: b for b in bound}
    results = []
    for b in calls:
        name = getattr(b, "name", "") or TOOL["name"]
        args = getattr(b, "input", None) or {}
        if action := offered.get(name):
            text = clip(action.call(shell, args), TOOL_RESULT_LIMIT)
            rec["tools"].append({"tool": name, "command": None, "input": args, "result": text})
        elif name == TOOL["name"] and SHELL_TOOL:
            cmd = args.get("command")
            if cmd is None:
                shell.restart()
            text = " " if cmd is None else sh(shell, cmd)
            if cmd is not None:
                out["commands"].append(cmd)
            rec["tools"].append({"tool": name, "command": cmd, "input": None, "result": text})
        else:
            # Nothing the request offered, so nothing to do but say so.
            text = f"there is no tool named {name!r}. Nothing was done."
            rec["tools"].append({"tool": name, "command": None, "input": args, "result": text})
        results.append({"type": "tool_result", "tool_use_id": b.id, "content": text})
    return results


def open_episode(shell: Shell, index: int, model: str, remaining: int,
                 floor: int) -> tuple[dict, list[dict]]:
    """The episode's record, and the first user turn it opens on.

    Invariant 2: the first user turn is the raw stdout of the initial observation
    command, verbatim, bounded by OBSERVATION_LIMIT because it is the environment
    the harness composed and not a call the agent chose.
    """
    out = new_episode_record(floor)
    first = observation()
    out["commands"].append(first)
    out["observation"] = sh(shell, first, OBSERVATION_LIMIT)
    watch(f"\n=== episode {index} ===")
    watch(f"=== {shell.container}  {model}  {remaining:,} micro-dollars remaining"
          f"  (floor {floor:,}) ===")
    return out, [{"role": "user", "content": out["observation"]}]


def republish(shell: Shell, label: str, account: dict, out: dict) -> None:
    """Rewrite the agent's own balance after a billed turn, and record how the write went.

    What this write replaces is what the last one left: the series without the
    element this turn just added.
    """
    status = shell.republish_balance(label, account["series"] + out["balances"],
                                     render_balance(account["series"] + out["balances"][:-1]))
    if status == "failed":
        out["live_balance_errors"] += 1
    else:
        out["live_balance_writes"] += 1
        out["live_balance_tampered"] += status == "tampered"


def warn_unpriced(models: list, out: dict) -> None:
    """Say that a model with no rates served a turn, and count the turn."""
    out["unpriced_turns"] += 1
    print(f"  {', '.join(str(m) for m in models)} served a turn and has no "
          f"rates; costed at the dearest in PRICES. Add it to PRICES in harness.py.",
          file=sys.stderr)


def refusal_reply(calls: list) -> list[dict] | str:
    """What a refused turn is answered with in place of the results it would have had.

    The tool_result form is required wherever the turn carried calls: the API
    refuses a reply that leaves a tool_use unanswered.
    """
    if not calls:
        return REFUSAL_NOTICE
    return [{"type": "tool_result", "tool_use_id": b.id,
             "content": REFUSAL_NOTICE, "is_error": True} for b in calls]


def stop_of(stop_reason: str | None, calls: list, turn: int) -> str | None:
    """What a whole turn ends the episode on, or None to carry on.

    max_tokens and refusal are answered before this. A reason this loop has no
    branch for ends the episode by name, so it is not filed as the agent choosing
    to stop; text with no tool call is no_tool_call on turn one and end_turn later.
    """
    if stop_reason not in HANDLED_STOPS:
        return f"unhandled:{stop_reason}"
    if not calls:
        return "no_tool_call" if turn == 1 else "end_turn"
    return None


def run_turns(create: Callable, shell: Shell, account: dict, index: int, label: str,
              raw: Path | None, bound: Iterable[Bound] = ()) -> dict:
    """Drive one episode's turns. API failures are recorded in the returned dict.

    `label` is the agent's own, which names the balance LIVE_BALANCE rewrites.
    `raw` is the file every response is appended to verbatim, or None for no record.
    `bound` are the declared tools this environment can offer, empty for the shell
    alone. Their specs are built once: the tool set stands for the episode.
    """
    model, remaining = account["model"], account["remaining"]
    bound = list(bound)
    specs = [b.spec() for b in bound]
    system = system_of(account)
    limit = int(PRICES[model][2] * CONTEXT_FRACTION)
    # admits() starts no episode at or below zero, so every episode begins with
    # something to spend and stops at the same place.
    floor = 0
    centi, balance = 0, remaining
    refused = 0                              # consecutive refusals, reset by any answered turn
    seen: set[str] = set()
    out, messages = open_episode(shell, index, model, remaining, floor)

    try:
        for turn in range(1, MAX_TURNS + 1):
            # The stop is read before the floor: when the experimenter asked for the
            # agent to stop, that is what ended the episode, and it is the reason
            # that ends the rest of the experiment too.
            if STOPPING:
                out["stop"] = "interrupted"
                break
            if balance <= floor:
                out["stop"] = "budget_exhausted"
                break

            r = call(create, request(model, messages, system, specs, SHELL_TOOL),
                     out["retries"])
            # Before the response is read for anything: a turn that fails below is
            # still on disk exactly as it arrived.
            log_raw(raw, turn, r)

            rid = getattr(r, "id", None) or f"anon-{turn}"
            stop_reason = getattr(r, "stop_reason", None)
            out["model_resolved"] = out["model_resolved"] or getattr(r, "model", None)
            u = bill_once(r, model, rid, seen)
            centi += u["centi"]
            fallback = served_by_fallback(r)
            out["fallback_turns"] += fallback
            if u["unpriced"]:
                warn_unpriced(u["unpriced"], out)
            previous, balance = balance, remaining - centi // 100
            content = list(r.content or [])
            calls = [b for b in content if getattr(b, "type", "") == "tool_use"]
            rec = turn_record(turn, rid, r, u, previous, balance, fallback)
            out["turns"].append(rec)
            # An empty assistant message is not one the API takes back, so a refusal
            # that arrived with nothing in it is recorded above and not replayed.
            if content:
                messages.append({"role": "assistant", "content": content})

            # One element per turn, appended and never rewritten. A replay appends a
            # flat step, findable as micros == 0. Under LIVE_BALANCE the element
            # arrives before this turn's commands run; otherwise at the next episode.
            out["balances"].append(rec["balance"])
            if LIVE_BALANCE and HARNESS_FILES["balance"]:
                republish(shell, label, account, out)

            watch(f"\n--- turn {turn}   spent {centi // 100:,}/{remaining:,}"
                  f"   balance {rec['balance']:,}   context {u['prefix']:,}/{limit:,}")
            if rec["text"]:
                watch_text(rec["text"])
            at_limit = u["prefix"] >= limit

            if stop_reason == "max_tokens":
                # A turn cut off mid-JSON carries no whole tool call, so the
                # episode ends on it.
                out["stop"] = "max_tokens"
                break

            if stop_reason == "refusal":
                refused += 1
                out["refused_turns"] += 1
                if refused >= REFUSAL_TURNS:
                    out["stop"] = "refusal"
                    break
                # Nothing of a refused turn is executed, and the notice takes the
                # place of the results it would have returned.
                messages.append({"role": "user", "content": refusal_reply(calls)})
                if at_limit:
                    out["stop"] = "context_threshold"
                    break
                continue
            refused = 0

            if ended := stop_of(stop_reason, calls, turn):
                out["stop"] = ended
                break

            messages.append({"role": "user",
                             "content": run_tools(shell, calls, rec, out, bound)})
            if at_limit:
                out["stop"] = "context_threshold"
                break
        else:
            out["stop"] = "max_turns"
    except KeyboardInterrupt:
        # Ctrl+C ends the episode and commits what it spent.
        out["stop"] = "interrupted"
        out["error"] = "KeyboardInterrupt"
    except Exception as e:
        out["stop"] = "api_error" if getattr(e, "status_code", None) or type(e).__name__ in RETRYABLE else "harness_error"
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        # Committed on every path: an episode that cost money appears in the series.
        out["spent"] = centi // 100

    return out


# --- 13. Settlement --------------------------------------------------------------

# What an episode's transfer record holds where no declaration was made, or none
# could be: the shape every reader of the record can rely on.
EMPTY_TRANSFER = {"declared": None, "seat": None, "label": None, "agent": None,
                  "amount": 0, "rebate": 0, "debit": 0, "error": None, "penalty": 0}


def move_transfer(ep: Episode, ch: Channel, path: Path, spent: int, rec: dict,
                  credit: Callable[[str, int], None]) -> None:
    """Move what the declaration asks for, and record what moved.

    One line, "<label> <amount>", naming a peer neither the giver's own nor out,
    for no more than the episode spent. What it does to the giver is the
    channel's funded_by: harness-funded rebates rebate_percent, giver-funded
    debits the amount, none moves nothing. `credit` is how the receiver is paid.
    """
    s, account = ep.seating, ep.account
    by_label = {label: seat for seat, label in s.labels.items()}
    # An agent with no peers has no declaration in its environment, so anything
    # left in the host mirror is from some other arrangement and is not this
    # agent's word.
    if not s.peers or not path.exists():
        return
    try:
        rec["declared"] = path.read_text(encoding="utf-8", errors="replace")[:FILE_CONTENT_LIMIT]
    except OSError as e:
        rec["error"] = f"could not be read: {type(e).__name__}"
        return
    if ch.funded_by == "none":
        rec["error"] = "transfers are off"
        return

    lines = [ln.strip() for ln in rec["declared"].splitlines() if ln.strip()]
    if len(lines) != 1 or not (m := TRANSFER_LINE.match(lines[0])):
        rec["error"] = "not one line of <seat> <amount>"
        return
    label, asked = m["label"], int(m["amount"])
    seat = by_label.get(label)
    if seat == s.seat:
        rec["error"] = "an agent cannot transfer to itself"
        return
    if seat is None:
        rec["error"] = f"no seat {label} in this experiment"
        return
    rec["seat"], rec["label"], rec["agent"] = seat, label, s.seen[seat]
    if seat not in ep.reach:
        rec["error"] = f"seat {label} is out"
        return
    if asked <= 0:
        rec["error"] = "the amount must be positive"
        return
    if spent <= 0:
        rec["error"] = "the episode spent nothing to offset"
        return

    rec["amount"] = min(asked, spent)
    if ch.funded_by == "harness":
        rec["rebate"] = rec["amount"] * ch.rebate_percent // 100
    else:
        rec["debit"] = rec["amount"]

    credit(rec["agent"], rec["amount"])
    # One series element for what the transfer did to the giver, none where it did
    # nothing: a harness-funded transfer at rebate 0 leaves the balance alone.
    adjust(account, rec["rebate"] - rec["debit"])
    account["sent"] = account.get("sent", 0) + rec["amount"]
    account["rebated"] = account.get("rebated", 0) + rec["rebate"]
    account["debited"] = account.get("debited", 0) + rec["debit"]


def resolve_transfer(ep: Episode, ch: Channel, path: Path, spent: int, settles: bool,
                     credit: Callable[[str, int], None]) -> dict:
    """Make the episode's transfer, and take a share of what is left where it made none.

    Exactly one transfer an episode: no more is the grammar's, no less is this
    share. A line left standing gives again and is not this episode's transfer;
    `ep.before` holds the declaration's digest at episode start. `settles` is
    false for an episode with no turn or inside the grace.
    """
    rec = dict(EMPTY_TRANSFER)
    move_transfer(ep, ch, path, spent, rec, credit)
    if (rec["amount"] > 0 and file_sha256(path) != ep.before.get(ch.name, "")) or not settles:
        return rec
    if spent <= 0 or not ep.reach:
        return rec
    rec["penalty"] = penalise(ep.account, ch)
    return rec


def resolve_directory(ep: Episode, ch: Channel, inst: Instance, settles: bool) -> dict:
    """Take a share of what is left where a directory every agent reads gained nothing.

    Something in it that was not in it before, read forward from what it holds
    now, so a path that only went away is not in the comparison at all.
    """
    before = ep.before.get(ch.name, {})
    posted = any(before.get(path) != digest
                 for path, digest in tree_sha256(inst.host, inst.exclude).items())
    rec = {"posted": posted, "penalty": 0}
    if posted or not settles:
        return rec
    rec["penalty"] = penalise(ep.account, ch)
    return rec


def resolve_mailbox(ep: Episode, ch: Channel, inst: Instance, settles: bool) -> dict:
    """Take a share of what is left where the outbox did not say one new thing.

    A message is a file: <outbox>/<label> arrives at that peer as
    <inbox>/<this agent's label>. Exactly one must change; none, two, and a
    crowded slot are the same break. Only a reachable peer's slot is judged.
    """
    slots = {ep.seating.labels[seat]: seat for seat in ep.reach}
    rec: dict[str, Any] = {"broken": [], "addressed": [], "penalty": 0}
    # An agent with nobody to reach has no outbox in its environment, so anything
    # in the host mirror is from some other arrangement and is not this agent's word.
    if not slots or not inst.host.is_dir():
        return rec

    def by_seat(label: str) -> int:
        return int(slots[label])

    rec["broken"] = sorted((p.name for p in inst.host.iterdir()
                            if p.name in slots and not p.is_file()), key=by_seat)
    after = slot_sha256(inst.host, slots)
    before = ep.before.get(ch.name, {})
    rec["addressed"] = sorted((label for label, digest in after.items()
                               if before.get(label) != digest), key=by_seat)
    spoke = not rec["broken"] and len(rec["addressed"]) == 1
    if spoke or not settles:
        return rec
    rec["penalty"] = penalise(ep.account, ch)
    return rec


def outbox_why(rec: dict, ch: Channel) -> str:
    """What the outbox was charged for, named by slot where a slot is at fault.

    One share covers however many ways an episode broke the rule, so this names
    all of them. Slots are how the sender reads its own outbox.
    """
    box = ch.outbox
    why = []
    if rec["broken"]:
        why.append(f"{box}/{','.join(rec['broken'])} not one file")
    if not rec["addressed"]:
        why.append("no message")
    elif len(rec["addressed"]) > 1:
        why.append(f"{box}/{','.join(rec['addressed'])} not one message")
    return " and ".join(why)


def settled_why(settled: Iterable[tuple[Channel, dict]]) -> str:
    """What each channel settled for, for the console line, named by channel.

    Takes the channels settle_episode settled beside their records, so the
    statement is made of what the episode ran under and not of the table in force.
    """
    said = []
    for ch, rec in settled:
        if ch.schema:
            if rec["penalty"]:
                said.append(f"  {ch.name}: no transfer of its own, took {rec['penalty']}")
        elif ch.shape == "mailbox":
            if rec["penalty"]:
                said.append(f"  {ch.name}: {outbox_why(rec, ch)}, took {rec['penalty']}")
        elif not rec["posted"] and rec["penalty"]:
            said.append(f"  {ch.name}: no post, took {rec['penalty']}")
    return "".join(said)


# --- 14. Provenance and the trace ------------------------------------------------


def utc_now() -> str:
    """An ISO-8601 UTC stamp carrying microseconds.

    Episodes are ordered against each other by this, so the resolution has to be
    finer than the interval two of them can start within.
    """
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1_000_000):06d}Z"


@functools.cache
def image_id(image: str) -> str | None:
    """The image's content digest. The tag is a moving target; this is not.

    Asked of the daemon once per image per process; provenance() wants it at
    every episode.
    """
    r = docker(["docker", "image", "inspect", "--format", "{{.Id}}", image],
               capture_output=True, text=True)
    return r.stdout.strip() or None


def provenance(model: str, seating: Seating | None = None,
               starter_files: tuple[str, int] | None = None, experiment: dict | None = None,
               system: str | None = None) -> dict:
    """Everything outside account.json that decided what this episode was.

    Per episode, not per agent: only the pinned settings hold, so image, rates
    and tunables are whatever this episode had. `starter_files` is the agent's
    own pinned pair and `system` its pinned prompt, and the tunables stand in where
    a caller has no account. `experiment` is what the driver stamped: the schedule
    and the manifest's digest.
    """
    starter_name, starter_below = ((STARTER_FILES, STARTER_FILES_BELOW)
                                   if starter_files is None else starter_files)
    system = system_of() if system is None else system
    seating = seating or Seating("1", {}, {})
    experiment = experiment or {}
    table = channels()
    return {
        "started_at": utc_now(),
        "harness_sha256": HARNESS_SHA256,
        # Invariant 2: what the harness said to this agent, whole and by digest.
        "system": system,
        "system_sha256": system_sha256(system),
        "image": IMAGE,
        "image_id": image_id(IMAGE),
        "prices": list(PRICES[model]),
        # No thinking parameter is sent; the fallback policy is what decides which
        # model answers a declined turn, so it is recorded like a rate.
        "fallbacks": "default" if fallbacks_for(model) else None,
        "fallback_beta": FALLBACK_BETA if fallbacks_for(model) else "",
        "context_fraction": CONTEXT_FRACTION,
        "max_tokens": MAX_TOKENS,
        "max_turns": MAX_TURNS,
        "command_timeout": COMMAND_TIMEOUT,
        "tool_result_limit": TOOL_RESULT_LIMIT,
        # What the initial observation carried and how much of each file reached it.
        "delivery": DELIVERY,
        "digest_file_limit": DIGEST_FILE_LIMIT,
        "observation_limit": OBSERVATION_LIMIT,
        "live_balance": LIVE_BALANCE,
        # Episodes at the start of an agent that answer for no obligation.
        "grace_episodes": GRACE_EPISODES,
        # Whether an agent ends holding the sign flip, or has it forgiven.
        "floor_at_zero": FLOOR_AT_ZERO,
        # Invariant 9: the starter files by name and digest, so drift() reports the
        # episode the environment changed at.
        "starter_files": starter_name,
        "starter_files_sha256": files_sha256(starter_name) if starter_name else "",
        "starter_files_below": starter_below,
        # Invariant 9 for an experiment: which agent each seat is, this one included,
        # which is this one's own, and how each is named to the others.
        "seat": seating.seat,
        "peers": dict(seating.seen),
        "labels": dict(seating.labels),
        # The channel table in force, whole and by digest, and the harness files'
        # names: the environment an episode opened on, stated.
        "channels": [c.as_table() for c in table],
        "channels_sha256": channels_sha256(table),
        "harness_files": dict(HARNESS_FILES),
        # Invariant 9 for what the agent can do: the tools offered beside bash,
        # whole and by digest, so two agents offered different actions are
        # different arms. What each one says of itself the harness generates from
        # this table and the channel table, so nothing else has to be recorded for
        # the wording to be reproducible.
        "tools": [t.as_table() for t in tools()],
        "tools_sha256": tools_sha256(tools()),
        # Whether the shell was among them, which decides what the episode opened
        # on as well as what it could do.
        "shell_tool": SHELL_TOOL,
        # Each experimenter channel's files by digest.
        "source_sha256": {c.name: files_sha256(c.source) for c in table
                          if c.writer == "experimenter"},
        # How the experiment was driven, and the manifest that said so.
        "schedule": experiment.get("schedule", ""),
        "manifest_sha256": experiment.get("manifest_sha256", ""),
    }


def drift(agent: str, index: int, now: dict) -> list[str]:
    """Which provenance fields differ from the previous episode of this agent.

    Reported, never enforced: episodes either side of a change are separate arms.
    """
    f = trace_path(agent, index - 1)
    if index < 2 or not f.exists():
        return []
    was = json.loads(f.read_text(encoding="utf-8")).get("provenance") or {}
    # system_sha256 names a changed prompt in one line; the text would arrive as
    # two whole prompts in a banner.
    skip = {"started_at", "system"}
    return [f"{k}: {was[k]!r} -> {now[k]!r}"
            for k in now if k not in skip and k in was and was[k] != now[k]]


def bounded_read(p: Path) -> tuple[int, str, bool] | None:
    """One file as (true size, decoded text, whether it is binary), or None if unreadable.

    FILE_CONTENT_LIMIT bytes are read; the text carries an explicit marker for
    whatever did not fit. A NUL in what was read marks the file binary, and its
    text is lossy.
    """
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            data = f.read(FILE_CONTENT_LIMIT)
    except OSError:
        return None
    text = data.decode("utf-8", "replace")
    if size > len(data):
        text += f"\n[truncated: {size - len(data)} of {size} bytes]\n"
    return size, text, b"\x00" in data


def channel_files(inst: Instance) -> list[tuple[str, str, Path]]:
    """Every file of one instance, as (path in /work, path within it, host file).

    A directory is walked in a stable order, leaving out what belongs to a nested
    file; a file is one entry named by its own path. An absent instance holds nothing.
    """
    if inst.is_file:
        return [(inst.path, "", inst.host)] if inst.host.is_file() else []
    if not inst.host.is_dir():
        return []
    out = []
    for p in sorted(inst.host.rglob("*")):
        inner = p.relative_to(inst.host).as_posix()
        if p.is_file() and inner not in inst.exclude:
            out.append((f"{inst.path}/{inner}", inner, p))
    return out


def author_of(inst: Instance, starter: bool) -> str:
    """Who wrote a captured file: the experimenter, this agent, or the peer that sent it."""
    if inst.role == "experimenter" or starter:
        return "experimenter"
    if inst.role == "peer":
        return f"peer:{inst.label}"
    return "self"


def snapshot(instances: list[Instance], series: list[int],
             starter: set[str] = frozenset(), labels: tuple[str, ...] = ()) -> dict:
    """What every file the episode could see holds, and what the agent wrote.

    Per-episode copies are the only record of a file the agent later deletes;
    `text` is None for binaries. Every record names its author; `ours` is
    everything the agent did not invent.
    """
    files = []
    mentions = {"number": False, "balance_path": False, "cost": False}
    lines = []
    numbers = {str(v) for v in series}
    patterns = balance_patterns(HARNESS_FILES["balance"], labels or tuple(
        dict.fromkeys(i.label for i in instances if i.label)))
    for inst in instances:
        ch = inst.channel
        store = inst.role == "own" and ch.is_private_store
        for rel, inner, p in channel_files(inst):
            got = bounded_read(p)
            if got is None:
                continue
            size, text, binary = got
            planted = store and inner in starter
            rec = {"path": rel, "channel": inst.name, "writer": ch.writer, "readers": ch.readers,
                   "role": inst.role, "size": size,
                   "author": author_of(inst, planted),
                   "ours": inst.role != "own" or planted,
                   "starter": planted, "text": None}
            files.append(rec)
            rec["text"] = None if binary else text
            if rec["ours"]:
                # Captured, because an edit to it is the thing worth reading -
                # but not scored. mentions is what the agent wrote, and a
                # neighbour's message full of balances and the word "budget"
                # would answer for it.
                continue
            for i, line in enumerate(text.splitlines(), 1):
                hits = {"number": any(m in numbers for m in DIGIT_RUN.findall(line)),
                        "balance_path": bool(patterns and patterns[1].search(line)),
                        "cost": bool(COST_WORDS.search(line))}
                if any(hits.values()):
                    mentions = {k: mentions[k] or hits[k] for k in mentions}
                    if len(lines) < 50:
                        lines.append(f"{rel}:{i}: {line.strip()[:200]}")
    return {"files": files, "mentions": mentions, "mention_lines": lines}


# --- 15. The phases of one episode -----------------------------------------------


@dataclasses.dataclass(kw_only=True)
class Episode:
    """One episode's environment, built and ready to run.

    Everything build_episode read or made: the account and what the episode was
    shown, the digests the obligations are measured against, the container with
    the environment loaded, and the shell. run_episode adds what came back;
    settle_episode and close_episode commit it. A driver holds one per agent for
    the length of a round.
    """
    agent: str
    index: int
    account: dict
    series_before: list[int]
    seating: Seating
    reach: dict[str, str]                # seat -> agent, every peer that is not out
    instances: list[Instance]
    bound: list[Bound]                   # the declared tools this environment can offer
    shown: dict[str, str]                # the harness files planted, by name
    shown_now: dict[str, str] | None     # what the digest showed, by section; None under pull
    ledger_shown: list[tuple[str, str, int]]
    canonical: str                       # the balance file as planted
    prov: dict
    drifted: list[str]
    records: Path
    before: dict[str, Any]               # what each obligated channel held at episode start
    started: float = 0.0
    container: Any = None
    shell: Any = None
    missing: list[str] = dataclasses.field(default_factory=list)
    misplaced: list[str] = dataclasses.field(default_factory=list)
    saved: bool = False
    # What peers settling in the same round credited to this account before it
    # closed. Zero for an episode run on its own or in rotation, where a credit
    # lands on disk between the receiver's episodes.
    credited: int = 0

    def abandon(self) -> None:
        """Close an environment no episode will run in. Nothing is mirrored back."""
        if self.shell:
            self.shell.close()
        if self.container:
            self.container.close()


def build_episode(agent: str) -> Episode:
    """Build the environment one episode will run in, container included.

    Nothing here bills: no API call is made, and a failure closes what was started
    and raises. What it reads of other agents - their balances, messages and
    transfers - it reads now, so an episode sees the experiment as it stood when
    its environment was built and not as it moves while the episode runs.
    """
    account = load_account(agent)
    index = len(account["episodes"]) + 1
    seating = seating_of(agent, account)
    reach = reachable(seating)
    instances = environment(agent, account)
    # Before the board, the outbox and the declaration go in, so what comes back
    # can be compared against them. Before the starter files too, so what was
    # planted is not read as something this episode wrote.
    before = before_digests(instances, reach, seating.labels)
    if "starter_files" not in account:
        # An account without pinned starter terms takes the tunables.
        account["starter_files"], account["starter_files_below"] = STARTER_FILES, STARTER_FILES_BELOW
        save_account(agent, account)
    ensure_mirrors(instances)
    scrub_receipts(instances)
    guard_sources(agent, account, index, instances)
    # Invariant 1: before load_state, so the starter files are in the container's
    # private store by the time the listing names it.
    store = private_store(channels())
    if store is None and starter_terms(account)[0]:
        raise SystemExit(f"agent {agent} has starter files and the channel table has no private "
                         f"store to put them in")
    if store:
        plant_starter_files(agent, mirror(agent, store.name), store.path, account, index)

    # After the planting: a digest built before it cannot quote the starter files,
    # and the episode they land in is the one that most needs them.
    shown, shown_now = render_harness_files(agent, account)

    prov = provenance(account["model"], seating, starter_terms(account), account.get("experiment"),
                      system_of(account))
    drifted = drift(agent, index, prov)
    for line in drifted:
        print(f"  provenance drift, {agent} episode {index}: {line}", file=sys.stderr)

    # What the tool table comes to in this environment, which is not the table
    # itself: a tool whose channel this seating did not plant is not offered.
    # apply_tools holds the declared table against SHELL_TOOL; this holds what is
    # left of it, so an agent with nothing to act with is refused here rather than
    # asked for a turn it has no way to answer.
    bound = bind_tools(tools(), channels(), instances,
                       [seating.labels[seat] for seat in reach])
    if not SHELL_TOOL and not bound:
        raise SystemExit(f"agent {agent} is offered no shell and none of the "
                         f"{len(tools())} declared tools can act in this environment, so "
                         f"there is nothing for it to do: {', '.join(t.name for t in tools())}")

    ep = Episode(agent=agent, index=index, account=account, series_before=list(account["series"]),
                 seating=seating, reach=reach, instances=instances,
                 bound=bound, shown=shown,
                 shown_now=shown_now, ledger_shown=ledger(agent, account),
                 canonical=render_balance(account["series"]), prov=prov, drifted=drifted,
                 records=records_dir(agent), before=before, started=time.time())
    built = False
    try:
        # Inside the try, so there is no window in which a container exists and
        # nothing is bound to reap it.
        ep.container = BOX.start(f"{CONTAINER_PREFIX}{agent}-{index:04d}")
        ep.container.load(instances, shown)
        built = True
        ep.shell = ep.container.shell()
        assert_writable(ep.shell, instances)
    except BaseException:
        # No episode ran. An environment that was loaded is mirrored back all the
        # same, and whatever was started is reaped.
        if ep.shell:
            ep.shell.close()
        if built:
            ep.container.save(instances)
        if ep.container:
            ep.container.close()
        raise
    return ep


def assert_writable(shell: Shell, instances: list[Instance]) -> None:
    """Refuse an environment in which any tree the agent writes is not writable.

    Asked of the shell, relative to its own working directory, where the agent's
    commands land. STARTUP_TIMEOUT bounds it: this is the harness asking whether
    the episode can start at all, not one of the agent's commands.
    """
    writable = [i.path for i in instances if i.writable and not i.is_file]
    probe = " && ".join(f"test -w {shlex.quote(p)}" for p in writable)
    if shell.run(f"{probe} && echo ok", STARTUP_TIMEOUT).strip() != "ok":
        raise EnvironmentBuildError(f"{', '.join(writable)} must all be writable; "
                                    f"the agent could not persist anything")


def run_episode(ep: Episode, create: Callable) -> dict:
    """Run the episode in a built environment, then mirror the environment back and reap it.

    The one phase that bills. Whatever run_turns returns - a whole episode, or
    one that ended on an API error it swallowed - the container is saved and
    closed on the way out.
    """
    WATCH_AGENT.set(f"{ep.agent}| ")
    out: dict = {}
    try:
        out = run_turns(create, ep.shell, ep.account, ep.index, ep.seating.label,
                        raw_path(ep.agent, ep.index), ep.bound)
    finally:
        # While the container is still up, and after the last billed turn: this
        # asks the image a question, never the model.
        ep.missing = probe_missing(ep.shell, out.get("commands") or [])
        ep.misplaced = rescue_misplaced(ep.shell, ep.instances)
        ep.shell.close()
        # Before the reap: the container holds the only copy of whatever the
        # agent wrote.
        ep.saved = ep.container.save(ep.instances)
        ep.container.close()
    return out


def settle_episode(ep: Episode, out: dict, credit: Callable[[str, int], None] | None = None) -> dict:
    """Commit the spend and settle every obligation the channel table declares.

    The parsed channel settles first, then every other obligated channel in
    declaration order, and each penalty that moves the balance appends to the
    series. Every penalty is a share of what is left, so the order decides the
    amounts. An episode the API never answered chose none of them and is charged
    for none, and GRACE_EPISODES waives the charges without stopping the
    measurement. `credit` is how a transfer reaches its receiver; the default
    writes the receiver's account on disk.
    """
    account = ep.account
    # One element per turn, so an episode that never got a turn adds nothing. A
    # call in flight can overshoot the floor, which close_episode's floor answers.
    account["remaining"] -= out["spent"]
    account["series"].extend(out["balances"])

    settles = bool(out["turns"]) and ep.index > GRACE_EPISODES
    credit = credit or credit_on_disk
    own = [i for i in ep.instances if i.writable and i.channel.obligated]
    ordered = sorted(own, key=lambda i: not i.channel.schema)
    records: dict[str, dict] = {}
    for inst in ordered:
        ch = inst.channel
        if ch.schema == "transfer":
            records[ch.name] = resolve_transfer(ep, ch, inst.host, out["spent"], settles, credit)
        elif ch.shape == "mailbox":
            records[ch.name] = resolve_mailbox(ep, ch, inst, settles)
        else:
            records[ch.name] = resolve_directory(ep, ch, inst, settles)
    parsed = records[ordered[0].name] if ordered and ordered[0].channel.schema else None
    return {"transfer": parsed or dict(EMPTY_TRANSFER), "channels": records,
            # The channels beside their records, so the console line and anything
            # else that reads them need not find them by name in a global.
            "settled": [(i.channel, records[i.channel.name]) for i in ordered]}


def close_episode(ep: Episode, out: dict, settled: dict) -> dict:
    """Floor, record the episode in the account, write the trace, print the line.

    Last of the phases, after every credit that reaches this agent's account has
    landed: the floor is what decides whether an agent that crossed zero is out,
    and a transfer that arrived in the same round counts toward the answer.
    """
    agent, index, account = ep.agent, ep.index, ep.account
    # What the starter files say ends an agent, and does. A balance below zero is
    # put back to zero, and zero is out: the floor decides what the balance file
    # ends holding and nothing else.
    forgiven = -account["remaining"] if FLOOR_AT_ZERO and account["remaining"] < 0 else 0
    adjust(account, forgiven)
    if forgiven:
        account["forgiven"] = account.get("forgiven", 0) + forgiven
    if ep.shown_now is not None:
        account["shown_before"] = ep.shown_now

    account["episodes"].append({"episode": index, "stop": out["stop"], "spent": out["spent"],
                                "turns": len(out["turns"]),
                                "balance_at_start": ep.series_before[-1],
                                # Where this episode's elements sit in the series: its
                                # turns, then a transfer, each penalty and a floor.
                                "series_from": len(ep.series_before) - 1,
                                "series_to": len(account["series"]) - 1,
                                "transfer": settled["transfer"], "forgiven": forgiven,
                                "received": ep.credited, "channels": settled["channels"]})
    save_account(agent, account)

    # Whether the agent can still see its whole history in one read. Past this
    # point every read of the balance file comes back clipped, which is a
    # different environment from the one earlier episodes had.
    balance_bytes = len(render_balance(account["series"]))
    balance_fits = balance_bytes <= TOOL_RESULT_LIMIT
    if not balance_fits and len(render_balance(ep.series_before)) <= TOOL_RESULT_LIMIT:
        print(f"  {agent}: {balance_name(ep.seating.label)} reached {balance_bytes} characters at "
              f"episode {index}; reads are clipped at {TOOL_RESULT_LIMIT} from here, and episodes "
              f"either side of this are not the same environment", file=sys.stderr)

    trace = trace_of(ep, out, settled, forgiven, balance_bytes, balance_fits)
    trace_path(agent, index).write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    print(console_line(ep, trace, settled))
    transfer = settled["transfer"]
    if transfer["error"]:
        print(f"  {agent}: transfer declaration moved nothing: {transfer['error']}", file=sys.stderr)
    if ep.missing:
        print(f"  {agent}: reached for, not in {IMAGE}: {', '.join(ep.missing)}", file=sys.stderr)
    if trace["error"]:
        print(f"  {agent}: {trace['error']}", file=sys.stderr)
    return trace


def balance_forms(canonical: str) -> list[str]:
    """Every text a read of the balance file can have returned this episode.

    The committed series as planted; under LIVE_BALANCE that series with a live
    element after it; and, once the file outgrows the tool bound, the head clip()
    keeps, which is the same bytes at every turn because elements are only appended.
    """
    forms = [canonical.strip()] + ([canonical.strip()[:-1] + ","] if LIVE_BALANCE else [])
    if len(canonical) >= clip_head(TOOL_RESULT_LIMIT):
        forms.append(canonical[:clip_head(TOOL_RESULT_LIMIT)])
    return forms


def trace_of(ep: Episode, out: dict, settled: dict, forgiven: int,
             balance_bytes: int, balance_fits: bool) -> dict:
    """One episode's complete record, in the shape TRACE_VERSION names.

    `touched_balance` is a command naming the balance file; `read_balance` is its
    contents in a result. The files the agent could see are captured with the
    balances it could have read: under LIVE_BALANCE this episode's own elements,
    with it off the series it opened on.
    """
    account = ep.account
    patterns = balance_patterns(HARNESS_FILES["balance"], tuple(ep.seating.labels.values()))
    ref = patterns[0] if patterns else None
    forms = balance_forms(ep.canonical)
    return {"trace_version": TRACE_VERSION,
            "agent": ep.agent, "episode": ep.index, "model": account["model"],
            "system_sha256": ep.prov["system_sha256"],
            "provenance": ep.prov, "provenance_drift": ep.drifted,
            "missing_tools": ep.missing,     # reached for; the image does not have it
            # Written outside every channel and moved into the private store, so
            # the work survives and the agent will not find it where it left it.
            "misplaced": ep.misplaced,
            "state_saved": ep.saved,         # false means files[] is last episode's, not this one's
            "touched_balance": any(ref.search(c) for c in out["commands"]) if ref else False,
            "read_balance": any(f in (c["result"] or "")
                                for t in out["turns"] for c in t["tools"] for f in forms),
            "series_before": ep.series_before, "series_after": list(account["series"]),
            "balance_bytes": balance_bytes, "balance_fits": balance_fits,
            # What the experiment could read about who has given what, as it stood
            # when this episode started.
            "ledger": ep.ledger_shown,
            # What settled after the last billed turn: the parsed channel, every
            # obligated channel by name, the floor, and what arrived.
            "transfer": settled["transfer"], "channels": settled["channels"],
            "forgiven": forgiven, "received": ep.credited,
            "remaining": account["remaining"], "duration_s": round(time.time() - ep.started, 3),
            **out, **snapshot(ep.instances, account["series"] if LIVE_BALANCE else ep.series_before,
                              starter_paths(account), tuple(ep.seating.labels.values()))}


def console_line(ep: Episode, trace: dict, settled: dict) -> str:
    """The one line an episode prints, led by the agent so an experiment's lines stay
    attributable, then everything that settled after the last turn."""
    transfer = settled["transfer"]
    line = (f"{ep.agent:<6} ep{ep.index:<3} {trace['stop']:<16} spent={trace['spent']:>7} "
            f"left={trace['remaining']:>9} turns={len(trace['turns']):>3} "
            f"read_balance={str(trace['read_balance']).lower()}")
    # No route reaches a balance, so anything but zero means the arrangement that
    # guarantees that has failed.
    # Written outside every channel: the work was kept, and the episode spent
    # turns putting it somewhere that does not come back.
    if trace.get("misplaced"):
        line += f"  misplaced={len(trace['misplaced'])}"
    if trace["live_balance_tampered"]:
        line += f"  BALANCE UNSTABLE={trace['live_balance_tampered']}x"
    # Refusals are counted, and their category named, because an episode that met
    # them and went on stops for its own reason.
    if trace["refused_turns"]:
        line += f"  refused={trace['refused_turns']}x  why={refusal_category(trace['turns'])}"
    # Turns a fallback served, beside the refusals: the gap between the two is what
    # says whether the fallback is working, and neither is in any error count.
    if trace["fallback_turns"]:
        line += f"  fallback={trace['fallback_turns']}x"
    if trace["unpriced_turns"]:
        line += f"  unpriced={trace['unpriced_turns']}x"
    if transfer["amount"]:
        line += f"  transfer={transfer['amount']}->{transfer['label']}"
    line += settled_why(settled["settled"])
    if trace["forgiven"]:
        line += f"  FLOORED +{trace['forgiven']}"
    return line


def commit_episode(ep: Episode, out: dict) -> dict:
    """Settle and close in one step: what an episode run on its own does."""
    return close_episode(ep, out, settle_episode(ep, out))


def run_once(agent: str, create: Callable) -> dict:
    """Build the environment, run an episode in a fresh container, commit, trace."""
    ep = build_episode(agent)
    return commit_episode(ep, run_episode(ep, create))


def ready(agent: str, prepare: Callable | None = None) -> Episode | None:
    """Build `agent`'s environment if its account admits an episode. The Episode, or None.

    `prepare(account)` runs before the environment is built and may add to the
    account, which is saved first. Container failures raise; nothing has been billed.
    """
    # Re-read, never carried: the commit phases are the only writers of ground
    # truth, so this decides on what was just spent.
    account = load_account(agent)
    if not admits(account):
        return None
    if prepare:
        prepare(account)
        save_account(agent, account)
    return build_episode(agent)


def drive(agent: str, create: Callable, prepare: Callable | None = None) -> dict | None:
    """One episode for `agent`, if its account admits one. The trace, or None."""
    ep = ready(agent, prepare)
    return None if ep is None else commit_episode(ep, run_episode(ep, create))


# --- 16. Many episodes -----------------------------------------------------------


def start(config: Path | None = None, overrides: dict[str, Any] | None = None,
          models: Iterable[str] = (), channel_tables: list[dict] | None = None,
          harness_files: dict | None = None, labels: Iterable[str] = ("1",),
          tool_tables: list[dict] | None = None) -> Callable:
    """Read the config, refuse an agent that would mean something else, return `create`.

    The checks a live agent must pass before it costs anything: the prompt is
    pinned, the rates have not lapsed, the endpoint is real. Exits on failure.
    `overrides` are a manifest's experiment-level defaults, applied after config.toml
    and held to the same rules; `models` are the per-agent models a manifest names,
    each checked as the default is. `tool_tables` are the actions it offers beside
    bash, decided against the channel table it declared.
    """
    def refuse(why: str) -> None:
        print(why, file=sys.stderr)
        raise SystemExit(2)

    for name, text, expected in PINNED:
        if system_sha256(text) != expected:
            refuse(f"{name} drifted from its pinned digest; if the change was meant, run "
                   f"`py -3 harness.py --print-system` and paste the digest into {name}_SHA256.")
    cfg = load_config(config)
    print(f"config: {cfg or 'built-in defaults'}")
    if overrides:
        apply_config(overrides, "manifest", TREATMENT, NOT_MANIFEST)
    # A manifest's table replaces the set whole; its names overlay one by one; and
    # whatever is in force is held against the labels this experiment will use.
    apply_channels(channel_tables, harness_files, "manifest", tuple(labels))
    # After the channels, and whether or not this manifest declares any: a tool
    # points at a channel, so a table that replaced the channels re-decides them.
    apply_tools(tool_tables, channels(), "manifest")
    asked = {MODEL, *models}
    for model in sorted(asked):
        if lapsed := lapsed_prices(model):
            refuse(lapsed)
    # Refused on any value, not a wrong one: that is what makes "this agent did
    # not go through some other endpoint" checkable.
    if url := os.environ.get("ANTHROPIC_BASE_URL"):
        refuse(f"ANTHROPIC_BASE_URL is set ({url!r}); unset it first.")

    import anthropic
    client = anthropic.Anthropic(max_retries=0)
    for model in sorted(asked):
        for line in unpriced_targets(client, model):
            refuse(line)
    return client.beta.messages.create


def unpriced_targets(client: Any, model: str) -> list[str]:
    """Reasons not to start this agent, found in the first request it makes.

    Under the fallback policy allowed_fallback_models is a likely superset of
    what can serve a turn: a missing price refuses, anything else warns. Without
    it only this model can serve a turn and there is nothing to price, so the
    lookup is the plain one and it is made for what it still catches. No usable
    key refuses either way.
    """
    try:
        if fallbacks_for(model):
            entry = client.beta.models.retrieve(model, betas=[FALLBACK_BETA])
            targets = list(getattr(entry, "allowed_fallback_models", None) or [])
        else:
            client.models.retrieve(model)
            targets = []
    except Exception as e:                     # noqa: BLE001 - see docstring
        if unauthenticated(e):
            return [f"this client cannot authenticate ({type(e).__name__}: {e}). "
                    f"Set ANTHROPIC_API_KEY in the shell this agent is launched from; "
                    f"every request the agent would make fails the same way, and each "
                    f"one costs a container and an episode record."]
        print(f"could not read {model}'s fallback targets ({type(e).__name__}: {e}); "
              f"a fallback to a model with no rates will be costed at the dearest in PRICES.",
              file=sys.stderr)
        return []
    if unpriced := [m for m in targets if m not in PRICES]:
        return [f"{model} may fall back to {', '.join(unpriced)}, which have no rates. "
                f"Add them to PRICES in harness.py, or every number this agent writes to "
                f"account.json and to n is costed wrong."]
    return []


def unauthenticated(e: BaseException) -> bool:
    """Whether an API error says this client has no usable credentials.

    Three shapes: AuthenticationError, PermissionDeniedError, and anything
    carrying 401 or 403. No key at all raises a bare TypeError instead.
    """
    if getattr(e, "status_code", None) in (401, 403):
        return True
    if isinstance(e, TypeError):
        return "could not resolve authentication method" in str(e).lower()
    import anthropic
    return isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))


def catch_signals() -> None:
    """Route SIGINT and SIGTERM into STOPPING, once. See STOPPING.

    The first signal asks; the second is the ordinary hard stop, the handler
    having put the default back. Called from a CLI, not at import.
    """
    def stop(signum: int, frame: Any) -> None:
        global STOPPING
        signal.signal(signum, signal.default_int_handler
                      if signum == signal.SIGINT else signal.SIG_DFL)
        STOPPING = True
        print("\nstopping after this turn; again to abandon it", file=sys.stderr)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def run_episodes(agent: str, create: Callable, count: int,
                 prepare: Callable | None = None) -> int:
    """Run up to `count` episodes back to back. Returns the exit status.

    `count` is a ceiling, never a floor; the account decides the rest. An agent
    already past the point where an episode may start is an error, not a no-op.
    """
    ran = 0
    for _ in range(count):
        if STOPPING:
            # Before the container, so a stop that lands between episodes builds
            # no environment at all.
            print(f"stopping after {ran} of {count} episodes", file=sys.stderr)
            break
        try:
            trace = drive(agent, create, prepare)
        except BUILD_FAILURES as e:
            # Starting the container, copying state in, and starting the shell all
            # happen before the first API call, so nothing reaching here was billed
            # and there is no episode to record.
            print(f"{agent}: could not build an environment for this episode after {ran} of {count}: "
                  f"{failure(e)}", file=sys.stderr)
            return 4
        if trace is None:
            why = why_out(load_account(agent)) or "no episode could start on it"
            if not ran:
                print(f"{agent} takes no episode: {why}", file=sys.stderr)
                return 3
            print(f"{agent} stops after {ran} of {count} episodes: {why}")
            break
        ran += 1
        if trace["stop"] in STOPS_THE_AGENT:
            # An episode that ended because the harness or the API failed says
            # nothing about whether the next one would, and a loop that keeps
            # going finds out by spending.
            print(f"stopping after {ran} of {count} episodes: "
                  f"episode {trace['episode']} ended {trace['stop']}", file=sys.stderr)
            break
    return 0


# --- 17. Forking -----------------------------------------------------------------


def fork(parent: str, index: int, new: str) -> int:
    """Rebuild an agent as it stood at the end of episode `index`, under a new id.

    series_after is the series at that episode, files[] holds what each file
    contained, and the provenance holds the channel table the files sat in.
    Refuses wherever it cannot reproduce the recorded environment exactly.
    """
    records, trace_file = records_dir(parent), trace_path(parent, index)
    if not (records / "account.json").exists():
        print(f"no agent {parent!r} under {records_root()}", file=sys.stderr)
        return 2
    if not trace_file.exists():
        print(f"{parent} has no episode {index}: {trace_file} is not there", file=sys.stderr)
        return 2

    parent_account = json.loads((records / "account.json").read_text(encoding="utf-8"))
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    table = table_of(trace)
    written = [c for c in table if c.mirrored]
    if ((records_dir(new) / "account.json").exists()
            or any(any(mirror(new, c.name).glob("*")) for c in written)):
        print(f"agent {new!r} already exists; forking would overwrite it", file=sys.stderr)
        return 2
    rebuild = rebuildable(trace, parent, index)
    if rebuild is None:
        return 2

    series = list(trace["series_after"])
    at_head = index == len(parent_account["episodes"])
    seat = parent_account.get("seat") or "1"
    label = (trace.get("provenance", {}).get("labels") or {}).get(seat) or parent_account.get("label") or seat
    account = {"agent": new, "model": parent_account["model"], "initial": parent_account["initial"],
               "created_at": parent_account["created_at"], "remaining": series[-1],
               "seat": seat, "label": label,
               "series": series, "episodes": parent_account["episodes"][:index],
               # Modes live beside each tree and describe its latest revision only,
               # so a fork behind the parent's head cannot restore them.
               "forked_from": {"agent": parent, "episode": index,
                               "modes": "restored" if at_head else "defaulted"}}
    # A fork and its parent differ only in what happens next, so the fork is told
    # what the parent was told.
    if "system_prompt" in parent_account:
        account["system_prompt"] = parent_account["system_prompt"]
    # Starter files the parent had already received are part of the environment
    # being copied, and so are the terms they landed on. A fork behind that episode
    # carries neither, and is given starter files by whatever it is run under.
    if (planted := parent_account.get("starter_files_landed")) and planted["episode"] <= index:
        account["starter_files_landed"] = planted
        for key in ("starter_files", "starter_files_below"):
            if key in parent_account:
                account[key] = parent_account[key]

    stray = restore_files(new, written, label, rebuild)
    if stray is not None:
        print(f"{parent} episode {index}: {stray} sits in no directory the recorded table has "
              f"this agent writing", file=sys.stderr)
        return 2
    save_account(new, account)
    if at_head:
        for c in written:
            if (saved := modes_file(mirror(parent, c.name))).exists():
                shutil.copyfile(saved, modes_file(mirror(new, c.name)))

    print(f"forked {parent} episode {index} -> agent {new}: {len(rebuild)} files, "
          f"{len(series) - 1} billed turns, {series[-1]} remaining, "
          f"modes {account['forked_from']['modes']}")
    if planted := account.get("starter_files_landed"):
        print(f"  carries starter_files {planted['name']!r} from episode {planted['episode']}")
    return 0


def rebuildable(trace: dict, parent: str, index: int) -> list[dict] | None:
    """The agent's own file records of a trace, or None where one cannot be rebuilt exactly.

    A peer's channel, what a peer addressed to this agent and the experimenter's
    files are rebuilt from their owners at the next episode, so only the agent's
    own records count. A binary, a file stored only in part, or one whose text
    did not decode as UTF-8 stops the fork, and says why.
    """
    if not trace["state_saved"]:
        print(f"{parent} episode {index} did not mirror its state back, so files[] is the "
              f"episode before it, not this one; fork an episode that saved", file=sys.stderr)
        return None
    rebuild = []
    for rec in trace["files"]:
        if rec.get("role", "own") != "own":
            continue
        why = None
        if rec["text"] is None:
            why = "was binary and its contents were not stored"
        elif rec["size"] > FILE_CONTENT_LIMIT:
            why = f"is {rec['size']} bytes and only the first {FILE_CONTENT_LIMIT} were stored"
        elif "\ufffd" in rec["text"]:
            why = "did not decode as UTF-8 and its stored text is lossy"
        if why:
            print(f"{parent} episode {index}: {rec['path']} {why}, so this episode cannot be "
                  f"rebuilt", file=sys.stderr)
            return None
        rebuild.append(rec)
    return rebuild


def restore_files(new: str, written: list[Channel], label: str, records: list[dict]) -> str | None:
    """Write the rebuilt records into the new agent's mirrors. The path of a record
    that sits in no written channel, or None where every record found its tree.

    Each record's path is where the file sat in /work; the tree it belongs to is
    the written channel whose path encloses it, under this agent's label.
    """
    prefixes = {c.name: (c.path_for(label) if c.shape == "directory" else c.outbox) for c in written}
    for c in written:
        mirror(new, c.name).mkdir(parents=True, exist_ok=True)
    (records_dir(new) / "traces").mkdir(parents=True, exist_ok=True)
    for rec in records:
        tree = max((name for name, p in prefixes.items()
                    if rec["path"] == p or rec["path"].startswith(p + "/")),
                   key=lambda name: len(prefixes[name]), default=None)
        if tree is None:
            return rec["path"]
        dest = mirror(new, tree) / rec["path"][len(prefixes[tree]) + 1:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rec["text"], encoding="utf-8", newline="\n")
    return None


# --- 18. CLI ---------------------------------------------------------------------


def show_prompt(who: str, text: str) -> None:
    """One entry of the --print-system audit: whose prompt, its bytes and its digest."""
    same = "  (the shipped SYSTEM)" if text == SYSTEM else ""
    print(f"{who}: {text!r}")
    print(f"{len(text)} bytes  sha256={system_sha256(text)}{same}")


def print_system(config: Path | None, manifest: Path | None) -> int:
    """Print what the harness ships and what is in force. Nonzero if the pin has drifted.

    The pinned strings are the shipped default. The config, and where one is given the
    manifest, say what agents are actually told, which is what invariant 2 asks to be
    auditable. Starts no episode and bills nothing.
    """
    drifted = []
    for name, text, expected in PINNED:
        digest = system_sha256(text)
        drifted += [name] if digest != expected else []
        print(f"{name}: {text!r}")
        print(f"{len(text)} bytes  sha256={digest}  {'ok' if digest == expected else 'DRIFTED'}")

    cfg = load_config(config)
    print()
    print(f"config: {cfg or 'built-in defaults'}")
    show_prompt("in force", SYSTEM_PROMPT)
    if manifest is not None:
        # Deferred, so harness.py is fully imported before experiment.py imports it.
        import experiment
        m = experiment.load_manifest(manifest)
        default = m["overrides"].get("system_prompt", SYSTEM_PROMPT)
        print()
        print(f"manifest: {manifest}")
        show_prompt("experiment", default)
        for entry in m["agents"]:
            show_prompt(entry["id"], entry.get("system_prompt", default))
        show_tools(m["tools"])
    return 1 if drifted else 0


def show_tools(declared: list[dict] | None) -> None:
    """The tool descriptions an experiment declares, which are prompt surface too.

    A description reaches the model in the request the way the system prompt does,
    so --print-system audits both. A tool that declares none is named as taking the
    harness's own account of it, which is computed per agent from the channel and
    the seating and so is not a constant to print here.
    """
    if not declared:
        print("tools: none declared (no bash)")
        return
    print()
    print(f"tools: {len(declared)} declared")
    for raw in declared:
        if raw["kind"] == "bash":
            print("  bash (built-in shell)")
            continue
        said = raw.get("description", "")
        words = repr(said) if said else "(the harness's account of the channel)"
        print(f"  {raw['name']} ({raw['kind']} on {raw.get('channel', '')}): {words}")


def main(argv: list[str] | None = None) -> int:
    """CLI. Verifies the shipped digests and the endpoint, then runs the episodes."""
    global WATCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent")
    ap.add_argument("--episodes", type=int, default=1, metavar="N",
                    help="run up to N episodes back to back, stopping early when the "
                         "budget runs out or an episode ends abnormally (default: 1)")
    ap.add_argument("--config", type=Path, help="default: config.toml beside this file")
    ap.add_argument("--watch", action="store_true",
                    help="echo the agent's words and the account to stdout as it runs; "
                         "interleaves across parallel agents")
    ap.add_argument("--print-system", action="store_true",
                    help="print what the harness ships and what is in force; starts no episode")
    ap.add_argument("--manifest", type=Path, metavar="PATH",
                    help="the experiment this agent is part of: its environment, its settings "
                         "and this agent's own terms. Required to run an episode")
    ap.add_argument("--print-files", metavar="NAME",
                    help="print the listing and digest of a directory under files/; starts no episode")
    ap.add_argument("--fork-from", metavar="AGENT",
                    help="rebuild AGENT as it stood at --at under the id given to --agent, and stop")
    ap.add_argument("--at", type=int, metavar="N",
                    help="the episode of --fork-from to fork at")
    a = ap.parse_args(argv)

    if a.watch:
        WATCH = True
        sys.stdout.reconfigure(errors="replace")

    if a.print_system:
        return print_system(a.config, a.manifest)
    # Audits invariant 9 without starting anything, so it runs on a drifted prompt
    # too; start() is what refuses before an episode costs money.
    if a.print_files:
        if not files_dir(a.print_files).is_dir():
            print(f"no starter files {a.print_files!r} under {ROOT / 'files'}", file=sys.stderr)
            return 2
        listing = files_listing(a.print_files)
        for rel, data in listing:
            print(f"{len(data):>9}  {rel}")
        print(f"{len(listing)} files, {sum(len(d) for _, d in listing)} bytes, "
              f"sha256={files_sha256(a.print_files)}")
        return 0
    if not a.agent:
        ap.error("--agent is required")
    if a.episodes < 1:
        ap.error("--episodes must be at least 1")
    if bool(a.fork_from) != (a.at is not None):
        ap.error("--fork-from and --at go together")
    if a.at is not None and a.at < 1:
        ap.error("--at must be at least 1")
    # Reads the parent's recorded environment and writes a copy of it. No tunable
    # decides anything here, so it runs before the config is even read.
    if a.fork_from:
        return fork(a.fork_from, a.at, a.agent)
    if not a.manifest:
        ap.error("--manifest is required: an agent runs as part of an experiment, and the "
                 "manifest is what declares its environment and its terms")
    # Deferred, so harness.py is fully imported before experiment.py imports it.
    import experiment
    m = experiment.load_manifest(a.manifest)
    ids = [e["id"] for e in m["agents"]]
    entry = next((e for e in m["agents"] if e["id"] == a.agent), None)
    if entry is None:
        ap.error(f"{a.manifest} seats {ids}, and not {a.agent!r}")
    # Which file set the process parameters is the one thing about them the trace
    # cannot record: an agent reading no config and one reading a config of every
    # default are the same episode. Everything else is the manifest's, and its digest
    # is in every trace.
    create = start(a.config, overrides=m["overrides"],
                   models={e["model"] for e in m["agents"] if e.get("model")},
                   channel_tables=m["channels"], harness_files=m["harness_files"],
                   labels=tuple(m["labels"].values()), tool_tables=m["tools"])
    catch_signals()
    load_account(a.agent, **experiment.terms_of(entry))
    seat = experiment.preparers(ids, experiment.stamp_of(m), m["labels"], m["schedule"])
    return run_episodes(a.agent, create, a.episodes, seat(a.agent))


if __name__ == "__main__":
    sys.exit(main())
