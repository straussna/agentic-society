"""One episode of a metered agent.

    .venv/Scripts/python.exe harness.py --agent live01

The harness builds a sandbox, plants the files it renders from the agents' accounts,
lets the model run bash turns, meters every token into integer micro-dollars, settles
what the channel table declares, mirrors the agent's files back, and writes a trace.
docs/manifest.md defines every term; docs/design.md gives the reasons.

Sections, in the order an episode meets them:

  1. What the harness says          SYSTEM, REFUSAL_NOTICE, SHIPPED_TEXT, SHELL_SPEC,
                                    system_of
  2. Providers                      providers package catalogs and adapters
  3. Settings                       PROCESS, MANIFEST_SETTINGS, load_config, rejecter,
                                    overlay
  4. The channel and tool tables    Channel, DEFAULT_CHANNELS, validate_channels,
                                    Tool, validate_tools, Settings, SETTINGS
  5. Process constants              HARNESS_SHA256, limits, stop sets, regexes
  6. Accounts                       Account, load_account, Seating, adjust, penalise
  7. Starter files and sources      files_sha256, plant_starter_files, guard_sources
  8. The environment                Instance, instances, digest_for,
                                    render_status_files
  9. What the agent's channels held before_sha256
 10. The sandbox and the shell      Sandbox, Container, Shell, plant_trees,
                                    mirror_trees, clip, Bound, TOOL_KINDS, bind_tools,
                                    episode_specs
 11. The API                        with_retries, log_raw, watch
 12. The turn loop                  run_turns
 13. Settlement                     make_transfer, resolve_transfer, resolve_directory,
                                    resolve_mailbox, SETTLEMENTS
 14. Provenance and the trace       provenance, config_drift, snapshot
 15. The steps of one episode       Episode, build_episode, run_episode, settle_episode,
                                    close_episode, commit_episode, run_once, ready,
                                    drive
 16. Many episodes                  start, catch_signals, run_episodes
 17. Forking                        fork
 18. CLI                            main
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import io
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
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import MappingProxyType
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NoReturn,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
    get_args,
)

import progress
import providers
from providers import (
    NormalizedResponse,
    ProviderError,
    ProviderRouter,
    SessionInfo,
    ToolCall,
    ToolResult,
    ToolSpec,
)


# --- 1. What the harness says ----------------------------------------------------

# The harness ships no words. What it says to an agent it computes from the accounts -
# the balances, the digest, the ledger, a receipt - and rewrites whenever those move. A
# fixed line is a constant, and a constant is the experimenter's to declare through
# system_prompt. Every manifest declares one, the empty string included, and the empty
# string is pinned like any other text: silence is an arm an experiment states.
SYSTEM = ""

SYSTEM_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# What a refused turn receives in place of the tool results it would have had.
# The whole of the second channel the harness speaks on: two facts, no cause,
# no instruction, and no actor. The agent learns that the turn was refused and
# that its environment is unchanged, and nothing further. It is the harness reporting
# a fact about a turn and not part of what an arm compares, so no manifest declares it.
REFUSAL_NOTICE = "The turn was refused. No command was run."
REFUSAL_NOTICE_SHA256 = (
    "4263e6bab90f883bbbcb2a9676a27a4aef7bde825461b3f56a2b2665f68c0c8b"
)

# Every string the harness ships, as (name, text, pinned sha256). One list, so
# --print-system audits exactly what start() rejects a run on. The pin holds the
# shipped default to its sha256; a declared prompt is held by provenance instead.
SHIPPED_TEXT = (
    ("SYSTEM", SYSTEM, SYSTEM_SHA256),
    ("REFUSAL_NOTICE", REFUSAL_NOTICE, REFUSAL_NOTICE_SHA256),
)

SHELL_SPEC = ToolSpec(
    "bash",
    "Run a command in the episode shell. Set command to null to restart the shell.",
    {
        "type": "object",
        "properties": {"command": {"type": ["string", "null"]}},
        "required": ["command"],
        "additionalProperties": False,
    },
)


def system_sha256(text: str) -> str:
    """The sha256 of a system prompt, which is how provenance carries what was said."""
    return hashlib.sha256(text.encode()).hexdigest()


def system_of(account: Account | None = None) -> str:
    """What this agent is told: the prompt pinned in its account, or the system_prompt in force.

    Membership and not truth, because "" is a prompt an experiment can declare: an
    account holding it is told nothing, not told the default.
    """
    if account is not None and "system_prompt" in account:
        return account["system_prompt"]
    return SETTINGS.system_prompt


# --- 3. Settings -----------------------------------------------------------------

# Every setting is a field of Settings, at the end of section 4, beside its default;
# config.toml and then a manifest overlay them at startup. system_prompt is the one of
# them that reaches the model. This section says which file owns which, and holds
# them to one set of rules wherever they came from.

# What delivery may be. Under "push" the digest is quoted at episode start;
# under "pull" only the listing is, the digest is not written, and the agent
# reads what it chooses at what reading costs.
DELIVERIES = ("push", "pull")

# config.toml's, and rejected in a manifest: the process parameters, which are the
# machine, the API and the limits, true of every run whatever the experiment is.
PROCESS = {"IMAGE", "MAX_TOKENS", "MAX_TURNS", "COMMAND_TIMEOUT", "TOOL_RESULT_LIMIT"}

# An experiment's, and rejected in config.toml: everything an agent's situation is
# made of, alongside the [[channel]] tables and [status_files] that go with it.
MANIFEST_SETTINGS = {
    "SYSTEM_PROMPT",
    "BUDGET",
    "STARTER_FILES",
    "STARTER_FILES_BELOW",
    "CONTEXT_FRACTION",
    "DELIVERY",
    "DIGEST_FILE_LIMIT",
    "OBSERVATION_LIMIT",
    "LIVE_BALANCE",
    "GRACE_EPISODES",
    "FLOOR_AT_ZERO",
}

SETTING_KEYS = PROCESS | MANIFEST_SETTINGS

# What each file says when it is handed the other's key.
NOT_CONFIG = "an experiment's to declare; config.toml holds what is true of every run"
NOT_MANIFEST = "config.toml's, and true of every run whatever the experiment declares"

# Keys rejected by name in config.toml and a manifest, each mapped to where that
# setting is declared, so the rejection names the place. docs/manifest.md lists them.
RETIRED = {
    "fallbacks": "nothing: every turn calls the provider and model pinned to its seat",
    "shell_tool": 'a [[tool]] with name = "bash" and kind = "bash"; omit it to withhold bash',
    "transfer_funded_by": 'funded_by on the [[channel]] with schema = "transfer"',
    "rebate_percent": 'rebate_percent on the channel with schema = "transfer"',
    "transfer_silence_penalty_percent": 'silence_penalty_percent on the channel with schema = "transfer"',
    "blackboard_silence_penalty_percent": (
        "silence_penalty_percent on the board channel"
    ),
    "mailbox_silence_penalty_percent": "silence_penalty_percent on the mailbox channel",
    "shared_files": 'a [[channel]] with writer = "experimenter" and a source',
}

# Hard ceiling on max_tokens. The harness does not stream, and a non-streaming
# model call much above this hits the SDK's HTTP timeout.
MAX_TOKENS_CEILING = 16_000

# What config.toml may set the two limits to. A limit is true of every run and no
# part of an agent's situation: lower, it can end an episode the agent had not
# finished or kill a command that would have, which makes it part of the agent's
# situation, which config.toml cannot declare; higher, it no longer stops a runaway.
MAX_TURNS_FLOOR = 100
MAX_TURNS_CEILING = 500
COMMAND_TIMEOUT_FLOOR = 30
COMMAND_TIMEOUT_CEILING = 120

# Below this a clipped read of n cannot keep a usable head, and clip()'s marker
# would crowd out the content it is marking.
TOOL_RESULT_FLOOR = 1_000

# The same bar for a message clipped into m: below this what survives says
# less than the marker saying it was cut.
DIGEST_FILE_FLOOR = 200


def load_config(
    base: Settings, path: Path | None = None
) -> tuple[Settings, Path | None]:
    """The settings config.toml makes of `base`, and the file used, or None.

    The file is `path`, or config.toml under `base`'s root. Unknown keys, wrong types,
    out-of-range values, and a missing `path` all exit with a message. Pure: nothing
    is installed, and start() installs what it builds on the result. Nothing reads
    the file at import, so an import keeps the defaults.
    """
    if path is not None and not path.exists():
        raise SystemExit(f"{path}: no such config file")
    f = path or base.root / "config.toml"
    if not f.exists():
        return base, None
    top = tomllib.loads(f.read_text(encoding="utf-8"))
    for table in ("channel", *STATUS_FILES_TABLES, "tool"):
        if table in top:
            raise SystemExit(
                f"{f}: {table} is {NOT_CONFIG}. Declare it in a manifest under {base.root / 'experiments'}"
            )
    return overlay(base, top, str(f), PROCESS, NOT_CONFIG), f


# How a validator rejects: it ends the run with SystemExit and never returns, so
# every rule after a rejection may take the one before it as holding.
Reject = Callable[[str], NoReturn]


def rejecter(source: str | Path) -> Reject:
    """How a file's validator rejects: every message led by the file it came from."""

    def reject(why: str) -> NoReturn:
        raise SystemExit(f"{source}: {why}")

    return reject


def check_keys(
    reject: Reject,
    where: str,
    raw: dict,
    allowed: Iterable[str],
    types: Iterable[tuple[str, type]] = (),
    retired: dict[str, str] | None = None,
    expected: str = "",
    elsewhere: dict[str, str] | None = None,
) -> None:
    """Reject an unknown key or a wrong type, naming the table and the key.

    The one place config.toml, a manifest, an [[agent]] table and a [[channel]]
    table are held to their keys, so every rejection reads alike. `where` leads
    each message, `types` fixes the type of the keys that have one, `retired`
    names where each rejected key's setting is declared, `elsewhere` names the file
    a real key belongs in, and `expected` replaces the list of allowed keys.
    """
    for key in raw:
        if retired and key in retired:
            reject(f"{where}unknown key {key!r}; it is now {retired[key]}")
        if elsewhere and key in elsewhere:
            reject(f"{where}{key} is {elsewhere[key]}")
    if unknown := sorted(set(raw) - set(allowed)):
        reject(
            f"{where}unknown key {unknown[0]!r}; expected {expected or sorted(allowed)}"
        )
    for key, expected_type in types:
        if key in raw and type(raw[key]) is not expected_type:
            reject(
                f"{where}{key} must be {expected_type.__name__}, "
                f"got {type(raw[key]).__name__}"
            )


def overlay(
    base: Settings,
    values: dict[str, Any],
    source: str,
    allowed: Iterable[str] = SETTING_KEYS,
    elsewhere: str = "",
) -> Settings:
    """The settings config keys make of `base`, validated as a whole.

    `source` names where the values came from in every rejection. `allowed` is the half
    of SETTING_KEYS this file owns, and a key belonging to the other half is rejected
    saying so, so no setting can be given in two places. A manifest's experiment-level
    defaults come through here after config.toml, so both are held to the same types
    and ranges. The limits are held to their ranges where `values` gives them,
    and not where a check has amended them directly. Pure: nothing is installed. The
    starter files are found under the root in force, so `base` holds that root (see
    Settings).
    """
    reject = rejecter(source)
    allowed = set(allowed)
    check_keys(
        reject,
        "",
        values,
        [t.lower() for t in allowed],
        retired=RETIRED,
        elsewhere={k.lower(): elsewhere for k in SETTING_KEYS - allowed}
        if elsewhere
        else None,
    )
    changes: dict[str, Any] = {}
    for key, value in values.items():
        default = getattr(base, key)
        # An int where a float is wanted is the same setting, written shorter.
        if (
            isinstance(default, float)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = float(value)
        if type(value) is not type(default):
            reject(
                f"{key} must be {type(default).__name__}, got {type(value).__name__}"
            )
        changes[key] = value
    s = dataclasses.replace(base, **changes)
    validate_settings(
        source,
        provider=None,
        model=None,
        budget=s.budget,
        starter_files=s.starter_files,
        starter_files_below=s.starter_files_below,
    )
    if not 0 < s.context_fraction <= 1:
        reject(f"context_fraction must be in (0, 1], got {s.context_fraction}")
    if min(s.max_tokens, s.max_turns, s.command_timeout) <= 0:
        reject("max_tokens, max_turns, and command_timeout must all be positive")
    if (
        "max_turns" in values
        and not MAX_TURNS_FLOOR <= s.max_turns <= MAX_TURNS_CEILING
    ):
        reject(
            f"max_turns must be from {MAX_TURNS_FLOOR} to {MAX_TURNS_CEILING}, got "
            f"{s.max_turns}; it is a limit, true of every run: lower, it can end an "
            f"episode the agent had not finished, and higher, it no longer stops a runaway"
        )
    if (
        "command_timeout" in values
        and not COMMAND_TIMEOUT_FLOOR <= s.command_timeout <= COMMAND_TIMEOUT_CEILING
    ):
        reject(
            f"command_timeout must be from {COMMAND_TIMEOUT_FLOOR} to "
            f"{COMMAND_TIMEOUT_CEILING} seconds, got {s.command_timeout}; it is a safety "
            f"stop, true of every run: lower, it can kill a command that would have "
            f"finished, and higher, it no longer stops a hung one"
        )
    if s.grace_episodes < 0:
        reject(f"grace_episodes must be zero or positive, got {s.grace_episodes}")
    if s.delivery not in DELIVERIES:
        reject(f"delivery must be one of {list(DELIVERIES)}, got {s.delivery!r}")
    if not TOOL_RESULT_FLOOR <= s.tool_result_limit:
        reject(
            f"tool_result_limit must be at least {TOOL_RESULT_FLOOR}, got "
            f"{s.tool_result_limit}; below that a clipped read keeps no usable head"
        )
    if not DIGEST_FILE_FLOOR <= s.digest_file_limit:
        reject(
            f"digest_file_limit must be at least {DIGEST_FILE_FLOOR}, got "
            f"{s.digest_file_limit}; below that a clipped message says less than the "
            f"marker saying it was clipped"
        )
    if s.observation_limit < s.tool_result_limit:
        reject(
            f"observation_limit must be at least tool_result_limit "
            f"({s.tool_result_limit}), got {s.observation_limit}; the initial observation "
            f"carries the whole experiment's record and is never smaller than what "
            f"one call may return"
        )
    if s.max_tokens > MAX_TOKENS_CEILING:
        reject(
            f"max_tokens must be at most {MAX_TOKENS_CEILING}; the harness does "
            f"not stream, and larger values hit the SDK's HTTP timeout mid-episode"
        )
    return s


def apply_config(
    values: dict[str, Any],
    source: str,
    allowed: Iterable[str] = SETTING_KEYS,
    elsewhere: str = "",
) -> None:
    """Overlay config keys onto the settings in force: overlay, installed.

    A rejected key installs nothing, the keys before it included.
    """
    global SETTINGS
    SETTINGS = overlay(SETTINGS, values, source, allowed, elsewhere)


def validate_settings(
    source: str,
    *,
    provider: str | None,
    model: str | None,
    budget: int | None,
    starter_files: str | None,
    starter_files_below: int | None,
    who: str = "",
) -> None:
    """Reject pinned settings that cannot stand, naming the file and the key.

    The one set of rules for config.toml, a manifest's defaults and a manifest's
    per-agent settings. None is a setting the caller did not set.
    """
    lead = f"{source}: {who}: " if who else f"{source}: "
    if (provider is None) != (model is None):
        raise SystemExit(f"{lead}provider and model must be set together")
    if provider is not None:
        try:
            providers.model_spec(provider, model or "")
        except providers.ProviderConfigurationError as error:
            raise SystemExit(f"{lead}{error}") from None
    if budget is not None and budget <= 0:
        raise SystemExit(f"{lead}budget must be positive, got {budget}")
    if (starter_files is None) != (starter_files_below is None) or bool(
        starter_files
    ) != bool(starter_files_below):
        raise SystemExit(
            f"{lead}starter_files and starter_files_below are set together or not at "
            f"all; got starter_files={starter_files!r}, "
            f"starter_files_below={starter_files_below!r}. Starter files that never "
            f"land and a threshold with nothing to land are both agents you did not "
            f"mean to start"
        )
    if starter_files_below is not None and starter_files_below < 0:
        raise SystemExit(
            f"{lead}starter_files_below must be zero or positive, got {starter_files_below}"
        )
    if starter_files and not (
        files_dir(starter_files).is_dir() or files_dir(starter_files).is_file()
    ):
        raise SystemExit(
            f"{lead}starter_files {starter_files!r} is not a file or directory at {files_dir(starter_files)}"
        )


# --- 4. The channel and tool tables -----------------------------------------------

# What each enumerated field of a channel may hold. WRITERS, SHAPE_PATHS,
# TRANSFER_FUNDERS and AGENT_VIEWS, which a declaration is checked against, are
# read out of these, so the check and the field's type cannot disagree.
Writer = Literal["self", "experimenter"]
Readers = Literal["self", "all", "addressee", "harness"]
Shape = Literal["directory", "mailbox", "file"]
Schema = Literal["", "transfer"]
Funder = Literal["harness", "giver", "none"]
AgentView = Literal["paths", "memory", "mailbox", "board", "transfer"]

# Another spelling an agent_view declaration may use, and the view it reads as.
# A channel holds the view it reads as, so a declaration under either spelling
# runs and is recorded the same.
AGENT_VIEW_ALIASES: Mapping[str, AgentView] = MappingProxyType({"letters": "mailbox"})

# The rule an episode is settled against a channel by. Channel.settles_as picks it
# and SETTLEMENTS holds it.
SettlesAs = Literal["transfer", "mailbox", "directory"]


@dataclasses.dataclass(frozen=True)
class Channel:
    """One declared part of every agent's environment. docs/manifest.md section 4.

    A channel has one writer, one set of readers and one shape. The harness makes
    the declaration true with ownership and modes and records it in provenance.
    """

    name: str
    writer: Writer
    readers: Readers
    shape: Shape = "directory"
    path: str = ""  # directory and file shapes; may hold {label}
    outbox: str = ""  # mailbox: the writer's side
    inbox: str = ""  # mailbox: each reader's side
    pushed: bool = True  # quoted in the digest under push delivery; every
    # channel, memory included
    restated: bool = False  # quoted every episode, never named as unchanged
    measured: bool = False  # the episode records what this channel gained
    silence_penalty_percent: int = 0
    schema: Schema = ""
    funded_by: Funder = "harness"  # transfer: who pays for one
    rebate_percent: int = 100  # transfer, harness-funded
    ledger: str = ""  # transfer: the status file holding every transfer
    receipt: str = ""  # transfer: where the parse result is written back
    source: str = ""  # experimenter channels: a directory under files/
    agent_view: AgentView = "paths"  # how the digest names what the channel holds

    def __post_init__(self) -> None:
        """Hold an agent_view declared under its alias as the view it reads as."""
        view = AGENT_VIEW_ALIASES.get(self.agent_view, self.agent_view)
        object.__setattr__(self, "agent_view", view)

    def path_for(self, label: str) -> str:
        """The path one agent's instance sits at."""
        return self.path.replace("{label}", label)

    @property
    def is_memory(self) -> bool:
        """A directory only its writer reads: a memory channel, where starter files
        land."""
        return (
            self.writer == "self"
            and self.readers == "self"
            and self.shape == "directory"
        )

    @property
    def mirrored(self) -> bool:
        """Whether the agent's writes to this channel travel in a mirror of their own.
        A file channel travels inside the directory that holds it."""
        return self.writer == "self" and self.shape != "file"

    @property
    def is_settled(self) -> bool:
        """Whether an episode is settled against this channel.

        Three reasons, each configured on its own: a schema, whose settlement is
        what moves a transfer; `measured`, which asks for the record and charges
        nothing; and a penalty, which cannot be taken from what was not measured.
        A channel asked for none of them is not settled, records nothing, and is
        not named on the console.
        """
        return bool(self.schema) or self.measured or self.silence_penalty_percent > 0

    @property
    def settles_as(self) -> SettlesAs:
        """The rule a settled channel is settled by: its schema's where it has one,
        a mailbox's where it is one, and otherwise a directory's, which asks what the
        channel gained."""
        if self.schema:
            return "transfer"
        return "mailbox" if self.shape == "mailbox" else "directory"

    def as_table(self) -> dict:
        return dataclasses.asdict(self)

    def declared(self) -> dict:
        """The fields a manifest would have to write to get this channel: defaults left out."""
        defaults = {f.name: f.default for f in dataclasses.fields(Channel)}
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k in ("name", "writer", "readers") or v != defaults.get(k)
        }


# The default channels, in the paths the starter files name. Code defaults, not
# config.toml's: no penalty, a full rebate.
DEFAULT_CHANNELS: tuple[Channel, ...] = (
    Channel("memory", "self", "self", "directory", path="memory"),
    Channel("board", "self", "all", "directory", path="{label}", measured=True),
    Channel(
        "mailbox",
        "self",
        "addressee",
        "mailbox",
        outbox="outbox",
        inbox="inbox",
        measured=True,
    ),
    Channel(
        "transfer",
        "self",
        "harness",
        "file",
        path="outbox/transfer",
        schema="transfer",
        funded_by="harness",
        rebate_percent=100,
        ledger="g",
    ),
)

# The files the harness writes into every environment, by role: <balance><label>
# holds each seat's balance history, and the digest quotes every pushed channel to
# this agent. docs/manifest.md section 5. The code's default names, as
# DEFAULT_CHANNELS are its default channels: what a trace that records none is read
# under.
DEFAULT_STATUS_FILES: Mapping[str, str] = MappingProxyType(
    {"balance": "n", "digest": "m"}
)


def channels() -> list[Channel]:
    """The channel table this process runs under."""
    return list(SETTINGS.channels)


def channels_sha256(table: Iterable[Channel]) -> str:
    """The sha256 of a channel table: every field of every channel, in declaration
    order."""
    body = json.dumps(
        [c.as_table() for c in table], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def channels_from(records: list[dict] | None) -> list[Channel]:
    """A channel table read back out of a trace's provenance; the default where there is none."""
    return [Channel(**r) for r in records] if records else list(DEFAULT_CHANNELS)


def memory_channel(table: Iterable[Channel]) -> Channel | None:
    """The memory channel: the directory only its writer reads, where starter files
    land."""
    return next((c for c in table if c.is_memory), None)


def schema_channel(table: Iterable[Channel]) -> Channel | None:
    """The one channel the harness parses, or None."""
    return next((c for c in table if c.schema), None)


def mailbox_channel(table: Iterable[Channel]) -> Channel | None:
    """The first unparsed mailbox, used for private messages."""
    return next((c for c in table if c.shape == "mailbox" and not c.schema), None)


def table_of(trace: dict) -> list[Channel]:
    """The channel table an episode ran under, from its trace's provenance; the
    default where the trace predates the table."""
    return channels_from((trace.get("provenance") or {}).get("channels"))


def channel(name: str, table: Iterable[Channel] | None = None) -> Channel:
    """The channel by that name, in `table` or in the one in force."""
    found = next(
        (c for c in (channels() if table is None else table) if c.name == name), None
    )
    if found is None:
        raise KeyError(f"no channel {name!r} in the table")
    return found


# What a channel declaration may say. docs/manifest.md section 10 is the prose.
WRITERS: tuple[Writer, ...] = get_args(Writer)

PAIRS = {
    ("self", "self"),
    ("self", "all"),
    ("self", "addressee"),
    ("self", "harness"),
    ("experimenter", "all"),
}

# What a transfer does to the giver. "harness": the receiver is credited and the giver
# rebated the channel's rebate_percent of the amount, so the experiment's total grows.
# "giver": the amount leaves the giver and reaches the receiver, and nothing is
# rebated. "none": a declaration moves nothing, and no share is taken for making none.
TRANSFER_FUNDERS: tuple[Funder, ...] = get_args(Funder)

SCHEMAS = {"transfer": ("funded_by", "rebate_percent", "ledger", "receipt")}

CHANNEL_KEYS = {
    "name",
    "writer",
    "readers",
    "shape",
    "path",
    "outbox",
    "inbox",
    "source",
    "pushed",
    "restated",
    "measured",
    "silence_penalty_percent",
    "schema",
    "agent_view",
    *SCHEMAS["transfer"],
}

# How the digest may present a channel. Which of them fits which channel is
# validate_agent_view's to decide.
AGENT_VIEWS: tuple[AgentView, ...] = get_args(AgentView)

CHANNEL_TYPES = (
    ("shape", str),
    ("path", str),
    ("outbox", str),
    ("inbox", str),
    ("source", str),
    ("schema", str),
    ("funded_by", str),
    ("ledger", str),
    ("receipt", str),
    ("pushed", bool),
    ("restated", bool),
    ("measured", bool),
    ("silence_penalty_percent", int),
    ("rebate_percent", int),
    ("agent_view", str),
)

NAME = re.compile(r"^[A-Za-z0-9._-]+$")

SEGMENT = re.compile(r"^(?:[A-Za-z0-9._-]|\{label\})+$")

SIDECARS = (".modes", ".incoming", ".previous")

STATUS_FILE_KEYS = ("balance", "digest", "round")

# The manifest table that names the status files, then the other name it is
# accepted under. A manifest declares one of them; config.toml declares neither.
STATUS_FILES_TABLES = ("status_files", "harness_files")


def validate_channels(
    tables: list[dict] | None,
    status_files: dict | None,
    source: str,
    labels: Iterable[str] = ("1",),
    *,
    base: Settings | None = None,
    table_name: str = "status_files",
) -> tuple[list[Channel], dict[str, str]]:
    """Read a channel table and status file names, or reject them naming the file
    and key.

    `tables` None keeps the table `base` holds; `status_files` None keeps the names
    it holds, and a table given overlays them key by key. `base` is the settings in
    force where none is given, and `table_name` is the one of STATUS_FILES_TABLES
    the names were declared under, which a rejection names. Every concrete path is
    checked against every other and against every status file, with `labels`
    standing in for {label}. Pure: nothing is set.
    """
    reject = rejecter(source)
    base = SETTINGS if base is None else base
    hf = dict(base.status_files)
    if status_files is not None:
        if not isinstance(status_files, dict):
            reject(f"[{table_name}] is a table")
        check_keys(
            reject,
            f"{table_name}: ",
            status_files,
            STATUS_FILE_KEYS,
            [(key, str) for key in STATUS_FILE_KEYS],
        )
        hf.update(status_files)
        for key in STATUS_FILE_KEYS:
            if hf.get(key) and not NAME.match(hf[key]):
                reject(
                    f'{table_name}: {key} must be one path segment, or "" for none'
                )

    if tables is None:
        table = list(base.channels)
    else:
        if not isinstance(tables, list) or not all(isinstance(x, dict) for x in tables):
            reject("channels are [[channel]] tables")
        table = []
        for raw in tables:
            table.append(parse_channel(raw, table, reject))
        parsed = [c for c in table if c.schema]
        if len(parsed) > 1:
            reject(
                f"one schema channel per experiment today; {parsed[0].name!r} and {parsed[1].name!r} both declare one"
            )
    claim_paths(table, hf, tuple(labels), reject)
    return table, hf


def path_fault(path: Any) -> str | None:
    """Why nothing may stand at this path, as the tail of a rejection, or None where
    it may.

    The one rule a declared path and a path a tool call names are both held to,
    so a tool cannot reach anywhere a channel could not.
    """
    if not isinstance(path, str) or not path:
        return "must be a path"
    if path.startswith("/") or any(
        seg in (".", "..") or not SEGMENT.match(seg) for seg in path.split("/")
    ):
        return (
            f"{path!r} must be segments of letters, digits, '.', '_', "
            f"'-' and at most one {{label}}, with no leading '/' and no '..'"
        )
    if path.count("{label}") > 1:
        return f"{path!r} names {{label}} more than once"
    if path.split("/")[0].endswith(SIDECARS):
        return f"{path!r} is a name the host keeps for itself"
    return None


def check_path(
    reject: Reject, name: str, key: str, path: Any, placeholder: bool
) -> None:
    """Reject a path no channel can stand at, naming the channel and the key.

    `placeholder` is whether {label} belongs in it, which only a directory every
    agent reads has.
    """
    if fault := path_fault(path):
        reject(f"channel {name}: {key} {fault}")
    if "{label}" in path and not placeholder:
        reject(
            f"channel {name}: {{label}} has no meaning in {key} here; only a directory "
            f"every agent writes has one instance per agent"
        )
    if placeholder and "{label}" not in path:
        reject(
            f"channel {name}: a directory every agent writes has one instance per agent, "
            f"so its path must name {{label}}"
        )


def validate_agent_view(
    channel: Channel, reject: Reject, declared: str | None = None
) -> None:
    """Require a view from AGENT_VIEWS, and one that describes the channel shape it
    names.

    `declared` is the spelling the manifest used, which a rejection names beside the
    view it reads as when the two differ.
    """
    view = channel.agent_view
    named = repr(view)
    if declared not in (None, view):
        named = f"{declared!r} (read as {view!r})"
    if view not in AGENT_VIEWS:
        reject(
            f"channel {channel.name}: agent_view must be one of {list(AGENT_VIEWS)}, got {view!r}"
        )
    valid = (
        view == "paths"
        or (
            view == "memory"
            and channel.shape == "directory"
            and (
                (channel.writer == "self" and channel.readers == "self")
                or channel.writer == "experimenter"
            )
        )
        or (view == "mailbox" and channel.shape == "mailbox")
        or (
            view == "board"
            and channel.shape == "directory"
            and channel.writer == "self"
            and channel.readers == "all"
        )
        or (
            view == "transfer"
            and channel.schema == "transfer"
            and channel.shape in ("file", "mailbox")
        )
    )
    if not valid:
        reject(
            f"channel {channel.name}: agent_view {named} does not describe its "
            f"{channel.writer}/{channel.readers}/{channel.shape} channel"
        )


def experimenter_channel(name: str, raw: dict, reject: Reject) -> Channel:
    """A channel the experimenter writes and every agent reads: a source and a path.

    It chooses no shape and nothing is owed to it, so the fields that go with those
    are rejected by name. `restated` stands: an experimenter channel is a constant, and
    standing text in front of the agent every episode is what one is for.
    """
    if extra := sorted(
        set(raw)
        & {
            "shape",
            "outbox",
            "inbox",
            "schema",
            "measured",
            "silence_penalty_percent",
            *SCHEMAS["transfer"],
        }
    ):
        reject(
            f"channel {name}: an experimenter channel takes source, path, pushed and restated, not {extra[0]}"
        )
    src = raw.get("source")
    if not isinstance(src, str) or not src or not files_dir(src).is_dir():
        reject(
            f"channel {name}: source {src!r} is not a directory under {SETTINGS.root / 'files'}"
        )
    check_path(reject, name, "path", raw.get("path"), False)
    pushed = raw.get("pushed", True)
    restated = raw.get("restated", False)
    if restated and not pushed:
        reject(
            f"channel {name}: restated asks for the digest to quote this channel every "
            f"episode, and pushed is false, so the digest carries none of it"
        )
    channel = Channel(
        name,
        "experimenter",
        "all",
        "directory",
        path=raw["path"],
        pushed=pushed,
        restated=restated,
        source=src,
        agent_view=raw.get("agent_view", "paths"),
    )
    validate_agent_view(channel, reject, raw.get("agent_view"))
    return channel


def mailbox_paths(
    name: str, raw: dict, readers: Readers, shape: Shape, reject: Reject
) -> dict:
    """A mailbox's two sides: the writer's outbox and each reader's inbox."""
    if readers != "addressee":
        reject(f"channel {name}: a mailbox is read by its addressee")
    if "path" in raw or not raw.get("outbox") or not raw.get("inbox"):
        reject(f"channel {name}: a mailbox takes outbox and inbox, not path")
    outbox, inbox = raw["outbox"], raw["inbox"]
    check_path(reject, name, "outbox", outbox, False)
    check_path(reject, name, "inbox", inbox, False)
    if outbox == inbox:
        reject(f"channel {name}: outbox and inbox must differ")
    return {"outbox": outbox, "inbox": inbox}


def one_path(
    name: str, raw: dict, readers: Readers, shape: Shape, reject: Reject
) -> dict:
    """A directory's or a file's single path.

    A directory every agent reads is one instance per seat, so its path names
    {label} and every other path may not.
    """
    if "outbox" in raw or "inbox" in raw:
        reject(f"channel {name}: outbox and inbox belong to a mailbox")
    if readers == "addressee":
        reject(
            f'channel {name}: an addressee reads a mailbox; give it shape = "mailbox"'
        )
    path = raw.get("path")
    check_path(reject, name, "path", path, shape == "directory" and readers == "all")
    return {"path": path}


# One entry a shape: where that shape declares its paths and what they must be. The
# keys are Shape's, in its order: a mailbox declares its two sides and every other
# shape its one path.
SHAPE_PATHS: dict[Shape, Callable[[str, dict, Readers, Shape, Reject], dict]] = {
    shape: mailbox_paths if shape == "mailbox" else one_path
    for shape in get_args(Shape)
}


def transfer_terms(name: str, raw: dict, penalty: int, reject: Reject) -> dict:
    """The transfer schema's own fields: who funds a transfer, what comes back to the
    giver, where every transfer is recorded, and where the parse result is written."""
    funded_by = raw.get("funded_by", "harness")
    if funded_by not in TRANSFER_FUNDERS:
        reject(
            f"channel {name}: funded_by must be one of {list(TRANSFER_FUNDERS)}, got {funded_by!r}"
        )
    rebate = raw.get("rebate_percent", 100)
    if not 0 <= rebate <= 100:
        reject(
            f"channel {name}: rebate_percent must be between 0 and 100, got {rebate}; "
            f"above 100 one agent mints budget out of a transfer it gets back in full"
        )
    if funded_by == "giver" and rebate != 0:
        reject(
            f'channel {name}: rebate_percent must be 0 under funded_by "giver", got '
            f"{rebate}; a transfer is the giver's own budget moving, and a rebate on "
            f"top of it would mint"
        )
    if funded_by == "none" and penalty:
        reject(
            f"channel {name}: silence_penalty_percent must be 0 under funded_by "
            f'"none", got {penalty}; no share is taken for not making a transfer '
            f"nobody can make"
        )
    ledger_name = raw.get("ledger", "")
    if ledger_name and not NAME.match(ledger_name):
        reject(f'channel {name}: ledger must be one path segment, or "" for none')
    receipt = raw.get("receipt", "")
    if receipt:
        check_path(reject, name, "receipt", receipt, False)
    return {
        "funded_by": funded_by,
        "rebate_percent": rebate,
        "ledger": ledger_name,
        "receipt": receipt,
    }


def parse_channel(raw: dict, table: list[Channel], reject: Reject) -> Channel:
    """One [[channel]] table as a Channel, or a rejection naming the channel and the
    key.

    `table` is what has been parsed before it, for the duplicate-name rule. What
    belongs to one shape is that shape's entry in SHAPE_PATHS and what belongs to
    a schema is that schema's own, so this is the order the rules are applied in.
    """
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        reject("every channel needs a name")
    if not NAME.match(name) or name.endswith(SIDECARS):
        reject(
            f"channel name {name!r} must be one path segment and not end in .modes, .incoming or .previous"
        )
    if any(c.name == name for c in table):
        reject(f"channel {name!r} is declared twice")
    check_keys(reject, f"channel {name}: ", raw, CHANNEL_KEYS, CHANNEL_TYPES)
    writer = raw.get("writer")
    if writer not in WRITERS:
        reject(
            f"channel {name}: writer must be one of {list(WRITERS)}, got {writer!r}; "
            f"the harness's own files are the status files"
        )
    readers = raw.get("readers", "all" if writer == "experimenter" else None)
    if (writer, readers) not in PAIRS:
        reject(
            f"channel {name}: writer {writer!r} read by {readers!r} is not a channel the "
            f"harness has; see docs/manifest.md section 4.1"
        )
    if writer == "experimenter":
        return experimenter_channel(name, raw, reject)

    shape = raw.get("shape", "directory")
    if shape not in SHAPE_PATHS:
        reject(
            f"channel {name}: shape must be one of {list(SHAPE_PATHS)}, got {shape!r}"
        )
    paths = SHAPE_PATHS[shape](name, raw, readers, shape, reject)
    schema = raw.get("schema", "")
    if readers == "harness":
        if shape != "file" or not schema:
            reject(
                f"channel {name}: a channel the harness reads is one file with a schema from {sorted(SCHEMAS)}"
            )
        if schema not in SCHEMAS:
            reject(
                f"channel {name}: schema must be one of {sorted(SCHEMAS)}, got {schema!r}"
            )
    else:
        if schema and not (
            schema == "transfer" and readers == "addressee" and shape == "mailbox"
        ):
            reject(
                f"channel {name}: a schema is either one file read by the harness or "
                f"a transfer mailbox read by its addressee"
            )
        for key in SCHEMAS["transfer"]:
            if not schema and key in raw:
                reject(
                    f"channel {name}: {key} is a field of the transfer schema, and this channel has none"
                )
    pushed = raw.get("pushed", True)
    restated = raw.get("restated", False)
    if restated and not pushed:
        reject(
            f"channel {name}: restated asks for the digest to quote this channel every "
            f"episode, and pushed is false, so the digest carries none of it"
        )
    measured = raw.get("measured", False)
    penalty = raw.get("silence_penalty_percent", 0)
    if not 0 <= penalty <= 100:
        reject(
            f"channel {name}: silence_penalty_percent must be between 0 and 100, got {penalty}"
        )
    if readers == "self" and penalty:
        reject(f"channel {name}: nothing is owed to a channel nobody else reads")
    channel = Channel(
        name,
        writer,
        readers,
        shape,
        **paths,
        pushed=pushed,
        restated=restated,
        measured=measured,
        silence_penalty_percent=penalty,
        schema=schema,
        agent_view=raw.get("agent_view", "paths"),
        **(transfer_terms(name, raw, penalty, reject) if schema else {}),
    )
    validate_agent_view(channel, reject, raw.get("agent_view"))
    return channel


def claim_paths(
    table: list[Channel], hf: dict[str, str], labels: tuple[str, ...], reject: Reject
) -> None:
    """Reject a table in which two concrete paths coincide, a file channel has no
    directory to sit in, or a channel takes a status file's name.

    Every path is expanded over every label, so a label that lands on a channel's
    path is rejected here too.
    """
    holders = [c.path for c in table if c.is_memory] + [
        c.outbox for c in table if c.shape == "mailbox"
    ]
    for c in table:
        if c.shape == "file" and not any(c.path.startswith(h + "/") for h in holders):
            reject(
                f"channel {c.name}: {c.path} is not inside a directory the agent writes, so nothing could hold it"
            )
    claimed: dict[str, str] = {}

    def claim(path: str, what: str) -> None:
        if path in claimed:
            reject(f"{what}: path {path!r} is also {claimed[path]}")
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
    if hf.get("round"):
        claim(hf["round"], "the round status")


def with_channels(
    base: Settings,
    tables: list[dict] | None,
    status_files: dict | None,
    source: str,
    labels: Iterable[str] = ("1",),
) -> Settings:
    """`base` with a channel table and status file names validated into it. Pure.

    An experimenter channel's source is found under the root in force, so `base`
    holds that root (see Settings).
    """
    table, hf = validate_channels(tables, status_files, source, labels, base=base)
    return dataclasses.replace(
        base, channels=tuple(table), status_files=MappingProxyType(hf)
    )


def apply_channels(
    tables: list[dict] | None,
    status_files: dict | None,
    source: str,
    labels: Iterable[str] = ("1",),
) -> None:
    """Validate a channel table and status file names and put them in force."""
    global SETTINGS
    SETTINGS = with_channels(SETTINGS, tables, status_files, source, labels)


# --- the tools a channel offers --------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tool:
    """One tool an experiment declares for its agents. docs/manifest.md section 4.8.

    A tool is a kind from the menu, TOOL_KINDS, pointed at a declared channel. The kind
    decides what the tool does and what its result reports; a manifest chooses the
    name, where it points, and the words the agent reads. It invents no behaviour:
    the input schema is the harness's, because that is the contract a call is held
    to. Bash is offered only when declared.
    """

    name: str
    kind: str  # a key of TOOL_KINDS
    channel: str = ""  # empty for bash
    description: str = ""  # the experimenter's words; "" takes the harness's
    every: int = 0  # a ballot: offered on each Nth round

    def as_table(self) -> dict:
        table = dataclasses.asdict(self)
        if not self.every:
            table.pop("every")
        return table


# The API's grammar for a tool name, and so the experimenter's.
TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

TOOL_KEYS = ("name", "kind", "channel", "description", "every")

TOOL_TYPES = (
    ("name", str),
    ("kind", str),
    ("channel", str),
    ("description", str),
    ("every", int),
)


def tools() -> list[Tool]:
    """The tool table this process runs under."""
    return list(SETTINGS.tools)


def tools_sha256(table: Iterable[Tool]) -> str:
    """The sha256 of a tool table: every field of every tool, in declaration order."""
    body = json.dumps(
        [t.as_table() for t in table], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def tools_from(records: list[dict] | None) -> list[Tool]:
    """A tool table read back out of a trace's provenance; none where there is none."""
    return [Tool(**r) for r in records or []]


def parse_tool(
    raw: dict, table: list[Tool], chans: list[Channel], reject: Reject
) -> Tool:
    """One [[tool]] table as a Tool, or a rejection naming the tool and the key.

    `table` is what has been parsed before it, for the duplicate-name rule, and
    `chans` the channel table the tool has to point into.
    """
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        reject("every tool needs a name")
    if not TOOL_NAME.match(name):
        reject(
            f"tool name {name!r} must be at most 64 letters, digits, '_' and '-', which is the grammar the API takes"
        )
    if any(t.name == name for t in table):
        reject(f"tool {name!r} is declared twice")
    if held := sorted(set(raw) & {"input_schema", "schema", "properties", "required"}):
        reject(
            f"tool {name}: {held[0]} is the harness's, not a manifest's; the input "
            f"schema is the contract a call is held to, and is built from the kind "
            f"and the channel. The description is the words this tool is given"
        )
    check_keys(reject, f"tool {name}: ", raw, TOOL_KEYS, TOOL_TYPES)
    kind = raw.get("kind")
    if kind not in TOOL_KINDS:
        reject(f"tool {name}: kind must be one of {sorted(TOOL_KINDS)}, got {kind!r}")
    if kind == "bash":
        if (
            name != "bash"
            or raw.get("channel")
            or raw.get("description")
            or "every" in raw
        ):
            reject("bash requires name = 'bash' and no channel, description or every")
        return Tool("bash", "bash")
    if name == "bash":
        reject("tool bash requires kind = 'bash'")
    where = raw.get("channel")
    ch = next((c for c in chans if c.name == where), None)
    if ch is None:
        reject(
            f"tool {name}: channel {where!r} is not in the channel table {[c.name for c in chans]}"
        )
    spec = TOOL_KINDS[kind]
    if not spec.fits(ch):
        reject(
            f"tool {name}: kind {kind!r} takes {spec.takes}, and channel {ch.name!r} is not one"
        )
    every = raw.get("every", 0)
    if spec.ballot:
        if every < 1:
            reject(
                f"tool {name}: kind {kind!r} requires every to be a positive integer, got {every!r}"
            )
    elif "every" in raw:
        reject(
            f"tool {name}: every belongs to kind "
            + " or ".join(repr(k) for k, v in TOOL_KINDS.items() if v.ballot)
        )
    return Tool(name, kind, ch.name, raw.get("description", ""), every)


def validate_tools(
    tables: list[dict] | None, chans: list[Channel], source: str
) -> list[Tool]:
    """Read a tool table against a channel table, or reject it naming the file and
    the key.

    An omitted table means no tools. Pure: nothing is set.
    """
    reject = rejecter(source)
    if tables is None:
        tables = []
    elif not isinstance(tables, list) or not all(isinstance(x, dict) for x in tables):
        reject("tools are [[tool]] tables")
    out: list[Tool] = []
    for raw in tables:
        out.append(parse_tool(raw, out, chans, reject))
    if len([tool for tool in out if TOOL_KINDS[tool.kind].ballot]) > 1:
        reject("an experiment declares at most one vote tool")
    return out


def with_tools(
    base: Settings, tables: list[dict] | None, chans: list[Channel], source: str
) -> Settings:
    """`base` with a tool table validated against `chans` into it. Pure.

    Bash is offered where the table declares it. A table that declares nothing, or
    withholds bash where `base` pushes no digest for an episode to open on, is rejected.
    """
    s = dataclasses.replace(base, tools=tuple(validate_tools(tables, chans, source)))
    if not s.bash_declared:
        if not s.tools:
            raise SystemExit(
                f"{source}: bash is not declared and no [[tool]] is declared, so "
                f"the agent is offered nothing to act with and every episode "
                f"ends on its first turn"
            )
        if s.delivery != "push" or not s.status_files["digest"]:
            raise SystemExit(
                f"{source}: bash is not declared, so an episode opens on the "
                f"digest and nothing else; delivery is {s.delivery!r} and the "
                f"digest is {s.status_files['digest']!r}, which leaves the first "
                f"turn with nothing in it"
            )
    return s


def apply_tools(tables: list[dict] | None, chans: list[Channel], source: str) -> None:
    """Validate a tool table against the channel table and make it the one in force."""
    global SETTINGS
    SETTINGS = with_tools(SETTINGS, tables, chans, source)


# --- the settings in force -------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Settings:
    """Everything a run is configured with: the settings, the channel and tool tables,
    the status files' names, and the directory every path resolves against.

    Frozen, and replaced whole: config.toml, a manifest and a check each make a new one
    of the one before. A setting's field is its lowercased name, so a config key, a
    manifest key and the field holding it are spelled alike.

    No builder moves root, and each is handed settings whose root is the one in force:
    load_config finds config.toml under the root it is handed, while overlay and
    with_channels find starter files and an experimenter channel's source under the
    root in force, through files_dir. Composing under another root builds first and
    moves root after, as print_context does.
    """

    # Where records/, mirrors/, interaction_store/, archive/, files/ and config.toml
    # are: this file's directory, until a check points it at a throwaway one.
    root: Path = Path(__file__).resolve().parent

    # config.toml's half: PROCESS.
    # The sandbox image the container is started from.
    image: str = "metered-agent:latest"
    max_tokens: int = 8_192  # output ceiling per turn
    max_turns: int = 200  # a limit
    command_timeout: int = 60  # seconds per bash command
    # Characters per tool result, in what the agent receives and in the trace. Also
    # the ceiling on what one call can cost, since the model is charged on what
    # survives the clip and never on what the command produced.
    tool_result_limit: int = 8_000

    # An experiment's half: MANIFEST_SETTINGS.
    # What the harness says to every agent, which is whatever the experiment declared it
    # should. Every manifest declares it, so this default stands only for a round driven
    # straight from a list of ids; "" sends no system parameter at all. Pinned per agent
    # at creation, and recorded whole and by sha256 in every episode's provenance.
    system_prompt: str = SYSTEM
    budget: int = 500_000  # micro-dollars per agent, at creation only
    context_fraction: float = (
        0.85  # of the model's window; crossing it ends the episode
    )
    # Rewrite the balance file in the container after every turn.
    live_balance: bool = True
    # Episodes at the start of an agent that take no silence penalty.
    grace_episodes: int = 0
    floor_at_zero: bool = False  # put a balance below zero back to zero
    # A file or directory under files/, or a path beside the manifest; "" is an empty
    # environment.
    starter_files: str = ""
    # The starter files land at the first episode at or below this balance.
    starter_files_below: int = 0
    # Whether the pushed channels are quoted to an agent at episode start ("push")
    # or left in the environment for it to read ("pull"). One of DELIVERIES.
    delivery: str = "push"
    # Characters of each file the digest carries. Per file and not for the whole, so
    # one long file cannot take every other agent's out of the initial observation.
    digest_file_limit: int = 2_000
    # Characters of the initial observation the agent receives. Its own bound because
    # the observation is the environment the harness composed and not a call the agent
    # chose, and tool_result_limit is the ceiling on what a chosen call may cost.
    observation_limit: int = 40_000

    # A manifest's tables. The channel table: the default until a manifest declares one.
    channels: tuple[Channel, ...] = DEFAULT_CHANNELS
    # The status files' names: the default until a manifest declares its own.
    status_files: Mapping[str, str] = DEFAULT_STATUS_FILES
    # The tools: none, until a manifest declares some. config.toml declares no
    # environment, so it declares no tools on one.
    tools: tuple[Tool, ...] = ()

    @property
    def bash_declared(self) -> bool:
        """Whether the shell is offered to the agent as a tool: whether bash is declared.

        The sandbox and its shell exist either way - the harness builds the
        environment, runs the initial observation and carries out every declared tool
        through them. What this decides is whether the agent may issue commands of its
        own, or reaches its environment only through the tools the experiment declared.
        """
        return any(t.kind == "bash" for t in self.tools)


# The settings in force. Replaced whole and never edited, and only by configuration
# code on the main thread while no episode runs: start(), print_context() and a
# check's fixtures. The turn loop, a round's episode threads and view.py's request
# threads only read it. Read as harness.SETTINGS, never imported by name: a name
# imported holds the settings in force at the import and misses every one after.
SETTINGS = Settings()


@contextlib.contextmanager
def using(settings: Settings) -> Iterator[Settings]:
    """`settings` in force for the length of a block, and the ones before it after.

    For configuration code on the main thread, as SETTINGS is: an audit composing under
    settings of its own, and a check.
    """
    global SETTINGS
    saved, SETTINGS = SETTINGS, settings
    try:
        yield settings
    finally:
        SETTINGS = saved


# --- 5. Process constants --------------------------------------------------------

# The sha256 of this file as it was loaded, read once at import. Every trace
# records it.
HARNESS_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Leads every episode container's name. A driver running several agents at once
# gives each process its own, so that reaping one agent's container cannot take
# another's with it.
CONTAINER_PREFIX = "mtr-"

# with_retries()'s retry policy: attempts, the base of the backoff in seconds (zero
# retries without waiting), the longest single wait, and the jitter on it.
RETRY_ATTEMPTS = 5
RETRY_BASE = 2
RETRY_CAP_S = 60
RETRY_JITTER = 0.25

# Characters of one turn's text and reasoning kept in the trace.
TURN_TEXT_LIMIT = 20_000

# Seconds the harness gives its own first command in a new episode. Not command_timeout:
# that bounds the agent's commands and an agent may tune it to seconds, while this
# waits on a container that has just started and may be one of several.
STARTUP_TIMEOUT = 30

# Bytes of each file captured per episode in the trace. The true size is
# recorded whether or not the content fits.
FILE_CONTENT_LIMIT = 100_000

# The shape of a trace: every file record names its channel, writer, readers and
# author and whether the harness planted it, and every episode record settles its
# channels by name.
TRACE_VERSION = 5

# The shape of an account: provider-neutral, its pinned settings beside the balance
# history.
ACCOUNT_VERSION = 3

# --watch only. Not in SETTING_KEYS, so config.toml cannot set it, and it never
# reaches the agent.
WATCH = False

WATCH_LIMIT = 2_000  # agent text on screen; the trace still keeps it all

# Agent prefix on echoed lines. Set per episode, and held per thread so episodes
# running at once each label their own lines.
WATCH_AGENT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "WATCH_AGENT", default=""
)

# Set by SIGINT and SIGTERM once catch_signals has run. The turn loop reads it
# just before it checks whether the balance is exhausted, so an interrupt ends the
# episode the way an exhausted balance does: after a whole turn, with the spend
# settled and the trace written. Nothing raises on the first signal; a second is
# the default again.
# A flag and not a field of SETTINGS: the signal handler interrupt sets it, and so
# does a check's fake router on a round's episode thread, and neither may replace
# the settings.
INTERRUPTING = False

# Child processes get a process group of their own, so a console Ctrl+C reaches
# this process and not the docker client or the shell it is waiting on.
DETACHED = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32"
    else {"start_new_session": True}
)

# Consecutive refused episodes after which an agent is treated as stuck, not
# unlucky. An episode counts only if refusals ended it, so one that was
# refused and carried on is not part of a streak. See stuck().
REFUSAL_STREAK = 8

# Consecutive refused turns after which an episode stops. The same context sent
# again is the same context the configured model just declined. At 1 the episode
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

# The stop reasons stop_of() reads as a whole turn, whose next step is decided by
# whether it called a tool. run_turns() answers max_tokens, refusal and other
# before it asks; any other reason reaching stop_of() ends the episode as
# unhandled:<reason> instead of being read as an ordinary finished turn.
HANDLED_STOPS = {"end_turn", "tool_use"}

COST_WORDS = re.compile(
    r"\b(cost|price|token|budget|dollar|spend|spent|charge|consum\w*)\b", re.I
)

# Whole numbers only, so a balance of 994750 does not match inside 1994750.
DIGIT_RUN = re.compile(r"-?\d+")

# The whole of what an agent may say to the harness. A label and an amount, the
# amount a bare decimal, in the register everything else it reads is written in.
TRANSFER_LINE = re.compile(r"^(?P<label>\S+) (?P<amount>\d+)$")


# --- 6. Accounts -----------------------------------------------------------------


def records_root() -> Path:
    """Where every agent's ground truth lives. Invariant 4: never reaches the container."""
    return SETTINGS.root / "records"


def interaction_store_root() -> Path:
    """Where a human seat's requests wait on a person: the human provider publishes
    them there, and view.py and human.py write the submissions."""
    return SETTINGS.root / "interaction_store"


def records_dir(agent: str) -> Path:
    """One agent's ground truth: account, traces, raw logs, analysis."""
    return records_root() / agent


# What a run started without --resume says of the records it moved aside.
FRESH_START = (
    "warning: starting fresh; archived the previous records and mirrors of {names} "
    "under {bundle} "
    "(use --resume to continue existing compatible records)"
)


def archive_agents(agents: Iterable[str], notice: str) -> Path | None:
    """Move existing records and mirrors into one preserved run bundle under archive/.

    `notice` is the line that says why, printed with {names} and {bundle} filled in
    where anything was moved.
    """
    agents = tuple(agents)
    sources = [
        (subdir, place(agent))
        for subdir, place in (("records", records_dir), ("mirrors", mirror_dir))
        for agent in agents
        if place(agent).exists()
    ]
    if not sources:
        return None

    archive = SETTINGS.root / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    moment = time.time_ns()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(moment / 1_000_000_000))
    base = f"{stamp}-{moment % 1_000_000_000:09d}Z"
    bundle = archive / base
    suffix = 1
    while bundle.exists():
        bundle = archive / f"{base}-{suffix}"
        suffix += 1
    bundle.mkdir()

    moved: list[tuple[Path, Path]] = []
    try:
        for subdir, source in sources:
            destination = bundle / subdir / source.name
            destination.parent.mkdir(exist_ok=True)
            replace_file(source, destination)
            moved.append((source, destination))
    except BaseException:
        for source, destination in reversed(moved):
            replace_file(destination, source)
        for child in bundle.iterdir():
            child.rmdir()
        bundle.rmdir()
        raise

    names = sorted({source.name for _, source in sources})
    print(notice.format(names=", ".join(names), bundle=bundle), file=sys.stderr)
    return bundle


def account_path(agent: str) -> Path:
    """One agent's account, which is what makes its records directory an agent's."""
    return records_dir(agent) / "account.json"


def traces_dir(agent: str) -> Path:
    """Where one agent's traces are kept, an episode-NNNN.json each."""
    return records_dir(agent) / "traces"


def trace_path(agent: str, episode: int) -> Path:
    """One episode's trace."""
    return traces_dir(agent) / f"episode-{episode:04d}.json"


def raw_path(agent: str, episode: int) -> Path:
    """One episode's raw log: a native event then canonical event for each response."""
    return records_dir(agent) / "raw" / f"episode-{episode:04d}.jsonl"


def trace_paths(agent: str) -> list[Path]:
    """Every trace an agent has, in episode order."""
    return sorted(traces_dir(agent).glob("episode-*.json"))


def raw_paths(agent: str) -> list[Path]:
    """Every raw log an agent has, in episode order."""
    return sorted((records_dir(agent) / "raw").glob("episode-*.jsonl"))


def episode_number(path: Path) -> int:
    """The episode an episode-NNNN file name holds."""
    return int(path.stem.rsplit("-", 1)[1])


def committed_traces(agent: str, account: Mapping[str, Any]) -> list[Path]:
    """The traces of the episodes `account` lists, in episode order.

    close_episode writes the trace before the account, so a trace past the episodes
    the account lists is one whose commit stopped between the two, and no episode of
    the agent's. An account that lists none, or is not there, has no trace.
    """
    count = len(account.get("episodes") or [])
    return [p for p in trace_paths(agent) if episode_number(p) <= count]


def mirror_dir(agent: str) -> Path:
    """Where the host mirrors of every channel one agent writes are kept."""
    return SETTINGS.root / "mirrors" / agent


def mirror(agent: str, name: str) -> Path:
    """The host mirror of one channel the agent writes, by the channel's name.

    Copied in at the channel's path each episode and out again at its end. The
    name never reaches the agent; the path does.
    """
    return mirror_dir(agent) / name


class Elimination(TypedDict):
    """Why an election took an agent out: the round it followed, the reason in words,
    and the votes the agent received in it."""

    round: int
    reason: str
    votes: int


class Election(TypedDict):
    """One election, as every account it is written to keeps it, each seat by label.

    tally: each candidate's votes. electors: who played the round it follows.
    abstainers: electors offered a ballot who cast none. interrupted: electors the
    experimenter's interrupt ended before they cast one. voted_out: the one the vote
    took out, "" for none. top_votes, tied: the highest total, and whether more than one
    candidate held it. candidates_left: the candidates it left in.
    """

    round: int
    tally: dict[str, int]
    electors: list[str]
    abstainers: list[str]
    interrupted: list[str]
    voted_out: str
    top_votes: int
    tied: bool
    candidates_left: list[str]


class AccountEpisode(TypedDict):
    """One committed episode, as close_episode lists it in the account.

    turn_count: how many turns it took. balance_from, balance_to: where in the
    balance history it starts and ends. transfer, channels: what settle_episode
    recorded. floored: what the floor put back. received_this_episode: what peers
    settling in the same round credited inside its span. receipts: what was planted
    at each receipt path, by sha256.
    """

    episode: int
    stop: str
    spent: int
    turn_count: int
    balance_at_start: int
    balance_from: int
    balance_to: int
    transfer: TransferRecord
    floored: int
    received_this_episode: int
    channels: dict[str, Mapping[str, Any]]
    receipts: dict[str, str]


class Account(TypedDict):
    """One agent's account.json, which save_account writes whole.

    load_account and fork create it with every required key. load_account pins the
    starter settings and the system prompt beside them; fork carries the starter
    settings only where the parent's starter files had landed by the episode it forks,
    and the prompt only where the parent holds one, so build_episode pins the settings
    in force for starter settings an account lacks and system_of tells one without a
    prompt the one in force. The rest are added as they first happen. account_on_disk
    and the viewer read an account as it stands on disk, {} where there is none, so
    what they hand one to takes a Mapping.
    """

    account_version: int
    agent: str
    provider: str
    model: str
    budget: int  # micro-dollars
    created_at: str
    seat: str
    balance: int
    balance_history: list[int]
    episodes: list[AccountEpisode]
    starter_files: NotRequired[str]
    starter_files_below: NotRequired[int]
    system_prompt: NotRequired[str]
    forked_from: NotRequired[dict[str, Any]]  # agent, episode, modes
    # What experiment.py stamps before each episode.
    label: NotRequired[str]
    seating: NotRequired[dict[str, Any]]  # agent_by_seat, labels, seat_order
    experiment_stamp: NotRequired[dict[str, Any]]  # the stamp, and memory_from
    # The starter files once they land, each experimenter source as first read, and
    # the digest's sections as last shown.
    starter_files_landed: NotRequired[dict[str, Any]]
    sources_seen: NotRequired[dict[str, dict[str, Any]]]
    shown_before: NotRequired[dict[str, str]]
    # Running totals: what transfers sent, received, rebated and debited, what the
    # floor put back, and what each channel's penalty took.
    sent: NotRequired[int]
    received: NotRequired[int]
    rebated: NotRequired[int]
    debited: NotRequired[int]
    floored: NotRequired[int]
    penalised: NotRequired[dict[str, int]]  # by channel name
    # experiment.py's memory, election, branch and product records.
    memory_from: NotRequired[dict[str, Any] | None]
    memory_inherited: NotRequired[dict[str, Any]]
    elections: NotRequired[list[Election]]
    last_election: NotRequired[Election]
    eliminated: NotRequired[Elimination]
    branch: NotRequired[dict[str, Any]]
    product: NotRequired[dict[str, str]]  # quality_tier


def save_account(agent: str, account: Account) -> None:
    """Write ground truth atomically: a temporary file, then a rename over the old one."""
    f = account_path(agent)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(account, indent=2), encoding="utf-8")
    replace_file(tmp, f)


def save_trace(agent: str, episode: int, trace: dict) -> None:
    """Write one episode's trace the way save_account writes an account, so a reader
    finds it whole or not at all."""
    f = trace_path(agent, episode)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    replace_file(tmp, f)


def replace_file(src: Path, dest: Path) -> None:
    """os.replace, retried while a Windows reader holds `dest` open:
    progress.replace."""
    progress.replace(src, dest)


# The pinned settings, as load_account's keyword -> the account key that holds each.
PINNED_SETTINGS = {
    "system_prompt": "system_prompt",
    "provider": "provider",
    "model": "model",
    "budget": "budget",
    "starter_files": "starter_files",
    "starter_files_below": "starter_files_below",
}


# The pinned settings every account holds, whether or not load_account is given them:
# the provider and model it runs on.
MODEL_SETTINGS = ("provider", "model")


def setting_shown(key: str, value: Any) -> str:
    """A pinned setting as a rejection names it: a prompt by sha256, everything else
    whole."""
    return f"sha256={system_sha256(value)}" if key == "system_prompt" else repr(value)


def missing_setting(agent: str, key: str) -> str:
    """The rejection of an account that holds no `key`."""
    return (
        f"agent {agent} has an incompatible account missing {key!r}; "
        "start a fresh agent id"
    )


# The key each record names its version under, and the one version harness.py reads.
RECORD_VERSIONS = {
    "account": ("account_version", ACCOUNT_VERSION),
    "trace": ("trace_version", TRACE_VERSION),
}


def stale(record: Mapping[str, Any], record_type: str) -> str | None:
    """Why `record`, an account or a trace as `record_type` names it, is not one
    harness.py reads: the version it records, or that it records none, beside the
    version read. None for a record of that version."""
    key, current = RECORD_VERSIONS[record_type]
    found = record.get(key)
    if found == current:
        return None
    article = "an" if record_type == "account" else "a"
    held = (
        f"a version-{found} {record_type}"
        if found is not None
        else f"{article} {record_type} that records no version"
    )
    return f"{held}, and harness.py reads version-{current} {record_type}s only"


# The keys a version-2 account and a version-4 trace hold under other names than
# the current versions do, by where each key sits, each with its current name. The
# dashboard and analyze.py read those records as history through
# account_as_current and trace_as_current; nothing reads them to run an episode.
ACCOUNT_2_KEYS = {
    "initial": "budget",
    "remaining": "balance",
    "series": "balance_history",
    "peers": "seating",
    "forgiven": "floored",
    "experiment": "experiment_stamp",
}
SEATING_2_KEYS = {"seen": "agent_by_seat", "presentation": "seat_order"}
EPISODE_2_KEYS = {
    "turns": "turn_count",
    "series_from": "balance_from",
    "series_to": "balance_to",
    "forgiven": "floored",
    "received": "received_this_episode",
}
ELECTION_2_KEYS = {"remaining": "candidates_left", "top_tied": "tied"}
STARTER_2_KEYS = {"remaining": "balance"}
TRACE_4_KEYS = {
    "series_before": "balance_before",
    "series_after": "balance_after",
    "remaining": "balance",
    "forgiven": "floored",
    "provenance_drift": "config_drift",
    "state_saved": "trees_mirrored",
}
PROVENANCE_4_KEYS = {
    "peers": "seating",
    "presentation_order": "seat_order",
    "message_delivery": "message_lifetime",
    "harness_files": "status_files",
}
TURN_4_KEYS = {"thinking": "reasoning"}
CHARGE_4_KEYS = {"kind": "token_class"}
REFUSAL_4_KEYS = {"kind": "refusal_type"}
FILE_4_KEYS = {"ours": "planted"}


def rekeyed(record: Mapping[str, Any], keys: Mapping[str, str]) -> dict[str, Any]:
    """A copy of `record` with every key `keys` names under the name it maps to."""
    return {keys.get(key, key): value for key, value in record.items()}


def account_as_current(account: Mapping[str, Any]) -> Mapping[str, Any]:
    """A version-2 account in the shape of the current version, its recorded
    account_version kept; any other account as it is."""
    if account.get("account_version") != 2:
        return account
    out = rekeyed(account, ACCOUNT_2_KEYS)
    if "seating" in out:
        out["seating"] = rekeyed(out["seating"], SEATING_2_KEYS)
    out["episodes"] = [
        rekeyed(episode, EPISODE_2_KEYS) for episode in out.get("episodes") or []
    ]
    if "elections" in out:
        out["elections"] = [
            rekeyed(election, ELECTION_2_KEYS) for election in out["elections"]
        ]
    if "last_election" in out:
        out["last_election"] = rekeyed(out["last_election"], ELECTION_2_KEYS)
    if "starter_files_landed" in out:
        out["starter_files_landed"] = rekeyed(
            out["starter_files_landed"], STARTER_2_KEYS
        )
    return out


def trace_as_current(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """A version-4 trace in the shape of the current version, its recorded
    trace_version kept; any other trace as it is.

    shell_tool goes from the provenance, whose tool table says the same: whether it
    declares bash. A file record's role goes, its author and starter flag saying
    what it says, and ours is planted.
    """
    if trace.get("trace_version") != 4:
        return trace
    out = rekeyed(trace, TRACE_4_KEYS)
    provenance = rekeyed(out.get("provenance") or {}, PROVENANCE_4_KEYS)
    provenance.pop("shell_tool", None)
    out["provenance"] = provenance
    turns = []
    for turn in out.get("turns") or []:
        turn = rekeyed(turn, TURN_4_KEYS)
        turn["charges"] = [
            rekeyed(charge, CHARGE_4_KEYS) for charge in turn.get("charges") or []
        ]
        if turn.get("refusal"):
            turn["refusal"] = rekeyed(turn["refusal"], REFUSAL_4_KEYS)
        turns.append(turn)
    out["turns"] = turns
    files = [rekeyed(file, FILE_4_KEYS) for file in out.get("files") or []]
    out["files"] = [
        {key: value for key, value in file.items() if key != "role"} for file in files
    ]
    return out


def load_account(
    agent: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    budget: int | None = None,
    starter_files: str | None = None,
    starter_files_below: int | None = None,
    system_prompt: str | None = None,
) -> Account:
    """Read the agent's ground truth, creating the agent on first use.

    The pinned settings - the system prompt, provider, model, budget, starter files and
    their threshold - are read once and recorded in account.json, which is what the agent
    uses from then on. Provider and model are always explicit; other omitted settings
    use the experiment's settings. A setting given for an existing agent must match what
    it was created on. An account that holds no provider, no model, or no setting it is
    given is rejected by the key it lacks.
    """
    s = SETTINGS
    given = {
        "provider": provider,
        "model": model,
        "budget": budget,
        "starter_files": starter_files,
        "starter_files_below": starter_files_below,
        "system_prompt": system_prompt,
    }
    f = account_path(agent)
    if not f.exists():
        if provider is None or model is None:
            raise SystemExit(
                f"agent {agent} needs an explicit provider and model before it can be created"
            )
        budget = s.budget if budget is None else budget
        name = s.starter_files if starter_files is None else starter_files
        below = (
            s.starter_files_below
            if starter_files_below is None
            else starter_files_below
        )
        prompt = s.system_prompt if system_prompt is None else system_prompt
        validate_settings(
            "account",
            provider=provider,
            model=model,
            budget=budget,
            starter_files=name,
            starter_files_below=below,
            who=agent,
        )
        for d in (
            traces_dir(agent),
            *(mirror(agent, c.name) for c in s.channels if c.mirrored),
        ):
            d.mkdir(parents=True, exist_ok=True)
        # Element 0 of the balance history is the budget; one more per turn
        # after it. seat is the agent's place in its experiment: 1 for an agent
        # driven on its own, and experiment.py stamps the rest before each episode.
        save_account(
            agent,
            {
                "account_version": ACCOUNT_VERSION,
                "agent": agent,
                "provider": provider,
                "model": model,
                "budget": budget,
                "seat": "1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "balance": budget,
                "balance_history": [budget],
                "episodes": [],
                "starter_files": name,
                "starter_files_below": below,
                "system_prompt": prompt,
            },
        )
        starter = f", starter_files {name!r} at or below {below}" if name else ""
        # The shipped prompt is the arm a silent manifest asks for, so only a
        # declared one is worth a line.
        declared = (
            ""
            if prompt == SYSTEM
            else f", system prompt sha256={system_sha256(prompt)[:12]}"
        )
        print(
            f"created agent {agent}: {budget} micro-dollars, "
            f"{provider}/{model}{declared}{starter}"
        )
    account = json.loads(f.read_text(encoding="utf-8"))
    if why := stale(account, "account"):
        raise SystemExit(f"agent {agent} has {why}; start a fresh agent id")
    for key in MODEL_SETTINGS:
        if key not in account:
            raise SystemExit(missing_setting(agent, key))
    try:
        providers.model_spec(account["provider"], account["model"])
    except providers.ProviderConfigurationError as error:
        raise SystemExit(f"agent {agent}: {error}; start a fresh agent id") from None
    for setting, key in PINNED_SETTINGS.items():
        if given[setting] is None:
            continue
        if key in account and account[key] != given[setting]:
            raise SystemExit(
                f"agent {agent} was created with {key}="
                f"{setting_shown(key, account[key])}, and is now asked to run with "
                f"{setting_shown(key, given[setting])}. Episodes either side of that "
                f"are not one experiment; start a new agent"
            )
        if key not in account:
            raise SystemExit(missing_setting(agent, key))
    return account


def account_on_disk(agent: str) -> Mapping[str, Any]:
    """Another agent's account as it stands on disk. Empty where the agent has not
    been created yet, which is what the first round of an experiment sees."""
    f = account_path(agent)
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def balance_history_on_disk(agent: str) -> list[int]:
    """Another agent's balance history, read from its own account."""
    return account_on_disk(agent).get("balance_history") or []


def starter_settings(account: Account) -> tuple[str, int]:
    """The starter files this agent receives and the balance they land at or below.

    Pinned in the account at creation; an account without them reads the settings in
    force.
    """
    return (
        account.get("starter_files", SETTINGS.starter_files),
        account.get("starter_files_below", SETTINGS.starter_files_below),
    )


def exhausted(account: Mapping[str, Any]) -> bool:
    """Whether the balance has reached zero or less: the agent is exhausted.

    A state an agent enters once and does not leave: in_play() starts no further
    episode on it, and make_transfer rejects it as a target. Read between episodes.
    """
    return account["balance"] <= 0


def stuck(account: Mapping[str, Any]) -> bool:
    """Whether the agent has refused its last REFUSAL_STREAK episodes running.

    Every turn calls the provider and model pinned to the seat, so a streak of
    REFUSAL_STREAK refusal-ended episodes means that model will not let the agent
    start, not that an episode went badly.
    """
    recent = [s["stop"] for s in account["episodes"][-REFUSAL_STREAK:]]
    return len(recent) == REFUSAL_STREAK and set(recent) == {"refusal"}


def in_play(account: Account) -> bool:
    """Whether the account is in play: another episode may start on this agent.

    This judges the account alone. An experiment also takes a seat more than a round
    behind the current round out of play: experiment.in_play.
    """
    return why_not_in_play(account) is None


def why_not_in_play(account: Mapping[str, Any]) -> str | None:
    """Why the account is not in play and the agent can take no further episode, or
    None where it can take one.

    Every reason is final: an eliminated agent cannot return, a stuck agent is out of
    play whatever its balance, and an exhausted agent is not a transfer target, so no
    peer can fund it back into play.
    """
    if eliminated := account.get("eliminated"):
        return eliminated.get("reason", "eliminated")
    if stuck(account):
        return f"refused its last {REFUSAL_STREAK} episodes running"
    if exhausted(account):
        return "nothing left to spend"
    return None


@dataclasses.dataclass(frozen=True)
class Seating:
    """Where an agent sits: its own seat, every seat's agent, every seat's label, and
    the order every seat is shown in.

    An agent driven on its own is an experiment of one, so everything downstream
    gets a seating either way. Both maps and seat_order run in seat number order.
    """

    seat: str  # the agent's own seat
    agent_by_seat: dict[str, str]  # seat -> agent id, every seat of the experiment
    labels: dict[str, str]  # seat -> label, every seat
    seat_order: tuple[str, ...]  # every seat in fixed display order

    @property
    def label(self) -> str:
        """The agent's own label."""
        return self.labels[self.seat]

    @property
    def peers(self) -> list[str]:
        """Every seat but the agent's own."""
        return [seat for seat in self.seat_order if seat != self.seat]


def seating_of(agent: str, account: Mapping[str, Any]) -> Seating:
    """The agent's seating, read from what experiment.py stamps into the account.

    experiment.py writes `seat`, `label` and `seating` before each episode of a
    round. An account without them is an experiment of one; a seat without a
    label is labelled by its number.
    """
    seat = account.get("seat") or "1"
    stamped = account.get("seating") or {}
    by_seat = stamped.get("agent_by_seat") or {seat: agent}
    by_seat = dict(sorted(by_seat.items(), key=lambda kv: int(kv[0])))
    given = dict(stamped.get("labels") or {})
    if account.get("label"):
        given[seat] = account["label"]
    raw_order = stamped.get("seat_order")
    order = (
        tuple(str(s) for s in raw_order if str(s) in by_seat)
        if raw_order
        else tuple(by_seat)
    )
    if raw_order and (len(order) != len(by_seat) or set(order) != set(by_seat)):
        order = tuple(by_seat)
    return Seating(seat, by_seat, {s: given.get(s, s) for s in by_seat}, order)


def reachable(seating: Seating) -> dict[str, str]:
    """The seats an episode can still reach: every seat but its own whose agent is in
    play.

    A seat whose agent is not in play is neither a transfer target nor a message
    target, so an outbox file naming one is neither a message nor a break. Read once
    at episode start, from the peers' accounts on disk; a peer not yet created has
    spent nothing.
    """
    peers_in_play = {}
    for seat in seating.peers:
        other = account_on_disk(seating.agent_by_seat[seat])
        if not other or why_not_in_play(other) is None:
            peers_in_play[seat] = seating.agent_by_seat[seat]
    return peers_in_play


def adjust(account: Account, delta: int) -> None:
    """Move the balance and append the result to the balance history.

    Everything that moves a balance outside a turn goes through here, so
    balance_history[-1] is the balance at any moment. A zero delta appends none.
    """
    if delta:
        account["balance"] += delta
        account["balance_history"].append(account["balance"])


def credit_account(account: Account, amount: int) -> None:
    """Credit a transfer to the receiver's account: one balance-history element, and
    the running total."""
    adjust(account, amount)
    account["received"] = account.get("received", 0) + amount


def credit_on_disk(agent: str, amount: int) -> None:
    """Credit a transfer to a receiver's account on disk, or take one back with a negative
    amount: the receiver is not in flight."""
    receiver = load_account(agent)
    credit_account(receiver, amount)
    save_account(agent, receiver)


def credit_episode(ep: Episode, amount: int) -> None:
    """Credit a transfer to a receiver whose episode is settling in the same round:
    its account in hand, and the record of what arrived inside its span."""
    credit_account(ep.account, amount)
    ep.credited += amount


def penalise(account: Account, ch: Channel) -> int:
    """Take the channel's share of what is left, keep the running total by channel
    name, and return the share. A share of zero moves nothing."""
    share = max(account["balance"], 0) * ch.silence_penalty_percent // 100
    if share:
        adjust(account, -share)
        totals = account.setdefault("penalised", {})
        totals[ch.name] = totals.get(ch.name, 0) + share
    return share


# --- 7. Starter files and experimenter sources -----------------------------------


# Starter files are a file or tree copied into the memory channel before an episode,
# so the agent meets them in the listing the opening command prints and not in
# anything the harness says. Their names and contents are prompt surface, recorded
# by sha256 in every episode (invariant 9).
def files_dir(name: str) -> Path:
    """Where starter files or an experimenter channel's source lives.
    Committed, unlike mirrors/ and records/."""
    return SETTINGS.root / "files" / name


def files_listing(name: str) -> list[tuple[str, bytes]]:
    """A source as (relative path, bytes), ordered so its sha256 is stable."""
    root = files_dir(name)
    if root.is_file():
        return [(root.name, root.read_bytes())]
    return [
        (p.relative_to(root).as_posix(), p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def files_sha256(name: str) -> str:
    """The sha256 of a starter source: paths and bytes, both.

    Recorded, not pinned: every episode's provenance says which one it got.
    """
    h = hashlib.sha256()
    for rel, data in files_listing(name):
        h.update(f"{rel}\0{len(data)}\0".encode("utf-8"))
        h.update(data)
    return h.hexdigest()


def plant_starter_files(
    agent: str, memory: Path, memory_path: str, account: Account, episode: int
) -> dict | None:
    """Copy the starter files into memory once the balance has fallen far enough.

    `memory` is the memory channel's host mirror and `memory_path` its path in the
    environment.
    Returns the record, or None; the trigger is the balance, not an episode number.
    Rejects starter files whose sha256 no longer matches, or that would overwrite a
    path the agent has written.
    """
    name, below = starter_settings(account)
    planted = account.get("starter_files_landed")
    if planted and name and planted["sha256"] != files_sha256(name):
        raise SystemExit(
            f"agent {agent} received starter_files {planted['name']!r} "
            f"({planted['sha256'][:12]}) at episode {planted['episode']}, and "
            f"files/{name} now has sha256 {files_sha256(name)[:12]}. Episodes either "
            f"side of that are not one experiment; start a new agent"
        )
    if planted or not name or account["balance"] > below:
        return None

    listing = files_listing(name)
    if collisions := [rel for rel, _ in listing if (memory / rel).exists()]:
        raise SystemExit(
            f"agent {agent}: starter files {name!r} would overwrite {collisions} in "
            f"{memory_path}/, which the agent wrote; rename the starter files or give "
            f"them to a fresh agent"
        )
    for rel, data in listing:
        dest = memory / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    record = {
        "name": name,
        "sha256": files_sha256(name),
        "episode": episode,
        "balance": account["balance"],
        "paths": [rel for rel, _ in listing],
    }
    account["starter_files_landed"] = record
    save_account(agent, account)
    print(
        f"{agent}: starter files {name!r} at episode {episode} "
        f"with {account['balance']} left: "
        f"{len(listing)} files, {sum(len(d) for _, d in listing)} bytes, "
        f"sha256={record['sha256'][:12]}"
    )
    return record


def starter_paths(account: Mapping[str, Any]) -> set[str]:
    """The paths in memory the starter files put there. `starter` on a file record
    means these alone."""
    return set((account.get("starter_files_landed") or {}).get("paths") or [])


def guard_sources(
    agent: str, account: Account, episode: int, instances: list[Instance]
) -> None:
    """Reject an experimenter channel whose files changed since the agent first saw
    them.

    Recorded in the account the first time, like starter files: an experiment
    whose source changed mid-flight spans two arms.
    """
    seen = account.setdefault("sources_seen", {})
    changed = False
    for inst in instances:
        if inst.role != "experimenter":
            continue
        ch = inst.channel
        sha256 = files_sha256(ch.source)
        was = seen.get(ch.name)
        if was and was["sha256"] != sha256:
            raise SystemExit(
                f"agent {agent} first read channel {ch.name!r} from "
                f"files/{was['source']} ({was['sha256'][:12]}) at episode "
                f"{was['episode']}, and files/{ch.source} now has sha256 "
                f"{sha256[:12]}. Episodes either side of that are not one "
                f"experiment; start a new agent"
            )
        if not was:
            seen[ch.name] = {"source": ch.source, "sha256": sha256, "episode": episode}
            changed = True
    if changed:
        save_account(agent, account)


# --- 8. The environment ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Instance:
    """One channel as one agent meets it: one owner's copy, at one path.

    A directory every agent writes is one instance per seat; a mailbox, including
    a transfer mailbox, is the writer's outbox and one inbox per peer; a parsed
    file is one own instance, nested inside the directory it sits in.
    """

    channel: Channel
    path: str  # in /work, no leading slash
    host: Path  # the mirror on disk; for a file, the file
    role: str  # "self" | "peer" | "experimenter"
    label: str  # the writer's label, "" for the experimenter
    nested: bool = False  # a file whose bytes travel with the tree above it
    exclude: frozenset = (
        frozenset()
    )  # a directory's inner paths that belong to nested files

    @property
    def name(self) -> str:
        return self.channel.name

    @property
    def writable(self) -> bool:
        return self.role == "self"

    @property
    def is_file(self) -> bool:
        return self.channel.shape == "file" or (
            self.channel.shape == "mailbox" and self.role == "peer"
        )

    @property
    def root(self) -> str:
        """The directory in /work that has to exist for this instance, and who owns it."""
        full = f"/work/{self.path}"
        return posixpath.dirname(full) if self.is_file else full


def instances(
    agent: str, account: Mapping[str, Any], table: list[Channel] | None = None
) -> list[Instance]:
    """Every instance of every channel in one agent's environment, in declaration order.

    An experimenter channel is one instance; a memory channel is one; a directory
    every agent writes is one per seat in seat order, the agent's own at its seat;
    a mailbox is the agent's outbox then one inbox per peer; a file the harness
    parses is one own instance. A mailbox or parsed file is not planted for an
    agent with no peers, there being nobody to reach. A file channel sits inside a
    directory the agent writes and travels with it.
    """
    table = list(table) if table is not None else channels()
    s = seating_of(agent, account)
    out: list[Instance] = []
    for ch in table:
        if ch.writer == "experimenter":
            out.append(Instance(ch, ch.path, files_dir(ch.source), "experimenter", ""))
        elif ch.is_memory:
            out.append(Instance(ch, ch.path, mirror(agent, ch.name), "self", s.label))
        elif ch.shape == "directory":
            out += [
                Instance(
                    ch,
                    ch.path_for(s.labels[seat]),
                    mirror(s.agent_by_seat[seat], ch.name),
                    "self" if seat == s.seat else "peer",
                    s.labels[seat],
                )
                for seat in s.seat_order
            ]
        elif ch.shape == "mailbox":
            if not s.peers:
                continue
            out.append(Instance(ch, ch.outbox, mirror(agent, ch.name), "self", s.label))
            out += [
                Instance(
                    ch,
                    f"{ch.inbox}/{s.labels[seat]}",
                    mirror(s.agent_by_seat[seat], ch.name) / s.label,
                    "peer",
                    s.labels[seat],
                )
                for seat in s.peers
            ]
        elif ch.shape == "file":
            if ch.readers == "harness" and not s.peers:
                continue
            out.append(Instance(ch, ch.path, Path(), "self", s.label))
    return reserved(nested(out), [c.receipt for c in table if c.schema and c.receipt])


def nested(instances: list[Instance]) -> list[Instance]:
    """Seat every own file instance inside the directory it sits in.

    The file's host is inside that directory's mirror, the directory excludes it
    from its own walk, and the file is listed right after the directory so the
    digest and the record read in path order.
    """
    out: list[Instance] = []
    for inst in instances:
        if not (inst.channel.shape == "file" and inst.role == "self"):
            out.append(inst)
            continue
        above = next(
            (
                i
                for i, d in enumerate(out)
                if d.writable and not d.is_file and inst.path.startswith(d.path + "/")
            ),
            None,
        )
        if above is None:
            # A file whose holding directory is not in this environment is not
            # in it either.
            continue
        d = out[above]
        rel = inst.path[len(d.path) + 1 :]
        out[above] = dataclasses.replace(d, exclude=d.exclude | {rel})
        at = above + 1
        while at < len(out) and out[at].nested:
            at += 1
        out.insert(at, dataclasses.replace(inst, host=d.host / rel, nested=True))
    return out


def reserved(instances: list[Instance], paths: list[str]) -> list[Instance]:
    """Keep the status files out of the directories they are planted in.

    A receipt sits inside a directory the agent writes and comes back with it at
    the episode's end; excluding it keeps it out of the digest, the settlement and
    the record, where it would read as the agent's. A file the agent puts at that
    path in its place is its own: snapshot records it and scrub_receipts leaves it.
    """
    for path in paths:
        for i, d in enumerate(instances):
            if d.writable and not d.is_file and path.startswith(d.path + "/"):
                instances[i] = dataclasses.replace(
                    d, exclude=d.exclude | {path[len(d.path) + 1 :]}
                )
    return instances


def receipts_planted(files: Mapping[str, str]) -> dict[str, str]:
    """The receipts among the status files an episode planted, as path -> sha256 of the
    bytes planted. Kept in the episode's record, it is what tells a receipt from a file
    the agent put in its place."""
    return {
        c.receipt: hashlib.sha256(files[c.receipt].encode("utf-8")).hexdigest()
        for c in channels()
        if c.schema and c.receipt and c.receipt in files
    }


def receipt_files(
    instances: list[Instance], planted: Mapping[str, str]
) -> dict[Path, bool]:
    """Every file at a receipt path inside a directory the agent writes, as the file in
    the mirror -> whether it holds the bytes `planted` names for its path in /work.

    The directory is the agent's, so it can remove a receipt and write a file of its
    own there; only one holding what the harness planted is the harness's.
    """
    claimed = {n.host for n in instances if n.nested}
    out: dict[Path, bool] = {}
    for inst in instances:
        if not inst.writable or inst.is_file:
            continue
        for rel in sorted(inst.exclude):
            p = inst.host / rel
            if p.is_file() and p not in claimed:
                out[p] = file_sha256(p) == planted.get(f"{inst.path}/{rel}")
    return out


def scrub_receipts(instances: list[Instance], planted: Mapping[str, str]) -> None:
    """Remove last episode's receipts from the mirrors before this one is planted.

    `planted` is what the last episode's record says was planted. A file there holding
    anything else is the agent's, recorded by the trace of the episode that left it,
    and stays.
    """
    for p, receipt in receipt_files(instances, planted).items():
        if receipt:
            p.unlink()


def ensure_mirrors(instances: list[Instance]) -> None:
    """Create the host mirror of every directory the agent writes, putting back first a
    tree a failed swap left aside. Raises OSError where that tree cannot be put back."""
    for inst in instances:
        if inst.writable and not inst.is_file:
            restore_previous(inst.host)
            inst.host.mkdir(parents=True, exist_ok=True)


def balance_name(label: str) -> str:
    """What the balance of the agent labelled `label` is called: the balance file name, then the label."""
    return f"{SETTINGS.status_files['balance']}{label}"


def render_balance(balance_history: list[int]) -> str:
    """Unlabelled: a JSON array of bare integers. No keys, no units, no timestamps."""
    return json.dumps(balance_history, separators=(",", ":")) + "\n"


def render_ledger(rows: list[tuple[str, str, int]]) -> str:
    """Unlabelled like a balance: giver label, receiver label, amount, one line each.

    Labels are seat numbers unless a manifest names them, so the ledger sits
    beside the balances as one more file of bare integers. What each column means
    is stated in the starter files or not at all.
    """
    return "".join(f"{giver} {receiver} {amount}\n" for giver, receiver, amount in rows)


def balances(agent: str, account: Account) -> dict[str, list[int]]:
    """Every balance the agent's environment shows, by label.

    Each comes from the account of the agent that owns it, so a peer's balance is as
    authoritative as the reader's own and neither is read back out of an environment.
    """
    s = seating_of(agent, account)
    return {
        s.labels[seat]: (
            list(account["balance_history"])
            if s.agent_by_seat[seat] == agent
            else balance_history_on_disk(s.agent_by_seat[seat])
        )
        for seat in s.seat_order
    }


def ledger_events(
    agent: str, account: Mapping[str, Any]
) -> list[tuple[int, str, str, int]]:
    """Every completed transfer as (round, giver label, receiver label, amount)."""
    s = seating_of(agent, account)
    rows = []
    for seat, other in s.agent_by_seat.items():
        source = account if other == agent else account_on_disk(other)
        for episode in source.get("episodes") or []:
            transfer = episode.get("transfer") or {}
            if amount := transfer.get("amount") or 0:
                receiver = transfer.get("label") or s.labels.get(
                    transfer["seat"], transfer["seat"]
                )
                rows.append(
                    (episode["episode"], seat, s.labels[seat], receiver, amount)
                )
    rows.sort(key=lambda row: (row[0], int(row[1])))
    return [
        (episode, giver, receiver, amount)
        for episode, _, giver, receiver, amount in rows
    ]


def ledger(agent: str, account: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Every transfer the experiment has made, as (giver label, receiver label, amount).

    Derived from the accounts, never kept; a declaration that moved nothing is not
    here. Ordered by giving episode then giver's seat, so every reader computes
    the same order.
    """
    return [
        (giver, receiver, amount)
        for _, giver, receiver, amount in ledger_events(agent, account)
    ]


def receipt_text(account: Account, ch: Channel) -> str:
    """The writer's itemized settlement for its most recently completed episode."""
    episodes = account.get("episodes") or []
    if not episodes:
        return ""
    episode = episodes[-1]
    records = episode.get("channels") or {}
    transfer = records.get(ch.name)
    if not transfer:
        return ""
    declared = (transfer.get("declared") or "").strip().splitlines()
    start = episode.get("balance_at_start", 0)
    history = account.get("balance_history") or []
    at = episode.get("balance_to", len(history) - 1)
    ending = history[at] if 0 <= at < len(history) else account.get("balance", 0)
    penalties = sum(record.get("penalty", 0) for record in records.values())
    received = episode.get("received_this_episode", 0)
    floored = episode.get("floored", 0)
    transfer_made = bool(transfer.get("amount"))
    grace = episode.get("episode", 0) <= SETTINGS.grace_episodes
    giver = seating_of(account["agent"], account).label
    receiver = transfer.get("label") or transfer.get("seat")
    moved = (
        f"{giver} (you) -> {receiver}; "
        f"actual amount moved: {transfer['amount']} micro-dollars"
        if transfer.get("amount")
        else f"nothing ({transfer.get('error') or 'no declaration'})"
    )
    lines = [
        f"episode: {episode.get('episode', '?')}",
        f"starting balance: {start}",
        f"API spend: {episode.get('spent', 0)}",
        f"grace episode: {'yes (silence penalties waived)' if grace else 'no'}",
        f"transfer declaration: {declared[0] if declared else 'nothing'}",
        f"transfer made: {'yes' if transfer_made else 'no'}",
        f"transfer moved: {moved}",
        f"transfer debit: {transfer.get('debit', 0)}",
        f"transfer rebate: {transfer.get('rebate', 0)}",
    ]
    for channel in channels():
        if not channel.is_settled or channel.name not in records:
            continue
        record = records[channel.name]
        settlement = SETTLEMENTS[channel.settles_as]
        spoke = settlement.spoke(record)
        # A channel the episode spoke on takes no silence penalty. A silent one states
        # what was taken, which is 0 inside the grace, at a rate of 0, and wherever its
        # rule takes nothing.
        penalty = "none" if spoke else record.get("penalty", 0)
        lines.append(
            f"{channel.name} silence penalty: {penalty} "
            f"({settlement.detail(record, spoke)})"
        )
    lines += [
        f"total silence penalties: {penalties}",
        f"received from peers: {received}",
        f"floored at zero: {floored}",
        f"ending balance: {ending}",
        (
            "reconciliation: starting balance - API spend - transfer debit "
            "+ transfer rebate - total silence penalties + received from peers "
            "+ floored at zero = ending balance"
        ),
    ]
    return "\n".join(lines) + "\n"


# One quoted item of the digest is a header line naming it, then its content.
# An unchanged or withdrawn group is a header naming the group, followed by one display
# name per line.
SECTION = re.compile(r"^=== (?P<path>.+) ===$", re.M)

NAMED = re.compile(r"^=== (?P<group>unchanged|withdrawn)(?:: (?P<paths>.*))? ===$")


def section(name: str, body: str) -> str:
    """One quoted file of the digest."""
    return f"=== {name} ===\n{body if body.endswith(chr(10)) else body + chr(10)}"


def named(group: str, paths: list[str]) -> str:
    """A digest group naming items it did not quote: unchanged or withdrawn."""
    return f"=== {group} ===\n" + "".join(f"- {path}\n" for path in paths)


def said_to(instances: list[Instance]) -> dict[str, str]:
    """Every file of every pushed instance, by path, each clipped at digest_file_limit.

    A binary is named and sized, never inlined. A peer's message that is not a
    regular file is absent, as it is absent from the environment.
    """

    def content(p: Path) -> str:
        data = p.read_bytes()
        if b"\0" in data:
            return f"[{len(data)} bytes, not text]"
        return clip(data.decode("utf-8", errors="replace"), SETTINGS.digest_file_limit)

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


def owning_instance(path: str, instances: Iterable[Instance]) -> Instance | None:
    """The instance a digest path belongs to: the deepest one that is the path or
    holds it, so a file nested in a directory is its own. None where none does."""
    holders = [i for i in instances if path == i.path or path.startswith(i.path + "/")]
    return max(holders, key=lambda i: len(i.path)) if holders else None


def digest_name(
    path: str,
    instances: list[Instance],
    experimenter_paths: frozenset[str] = frozenset(),
) -> str:
    """The agent-facing name of an internal item in a digest."""
    inst = owning_instance(path, instances)
    if inst is None or inst.channel.agent_view == "paths":
        return path
    if inst.channel.agent_view == "mailbox":
        incoming = f"{inst.channel.inbox}/"
        outgoing = f"{inst.channel.outbox}/"
        if path.startswith(incoming):
            return f"Message from {path.removeprefix(incoming)}"
        if path.startswith(outgoing):
            return f"Message to {path.removeprefix(outgoing)}"
    if inst.channel.agent_view == "board":
        suffix = " (you)" if inst.role == "self" else ""
        return f"Board post from {inst.label}{suffix}"
    if inst.channel.agent_view == "memory":
        name = path[len(inst.path) :].lstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
        if inst.role == "experimenter":
            source = "Experimenter channel"
        elif path in experimenter_paths:
            source = "Starter files"
        else:
            source = "Memory"
        if name == "personas" and source == "Starter files":
            return source
        if name == "memory" and source == "Memory":
            return source
        return f"{source}: {name.replace('_', ' ')}"
    if inst.channel.agent_view == "transfer":
        if inst.channel.shape == "mailbox":
            incoming = f"{inst.channel.inbox}/"
            outgoing = f"{inst.channel.outbox}/"
            if path.startswith(incoming):
                return (
                    f"Transfer received from {path.removeprefix(incoming)} last round"
                )
            if path.startswith(outgoing):
                return f"Transfer sent to {path.removeprefix(outgoing)} last round"
        return "Transfer submitted last round"
    return path


def experimenter_digest_paths(
    account: Mapping[str, Any], instances: list[Instance]
) -> frozenset[str]:
    """Starter-file paths, which the digest presents as starter files."""
    memory = next((inst for inst in instances if inst.channel.is_memory), None)
    return (
        frozenset(f"{memory.path}/{path}" for path in starter_paths(account))
        if memory
        else frozenset()
    )


def harness_digest_name(name: str, agent: str, account: Account) -> str:
    """A status file's agent-view heading in a tool-only observation."""
    s = SETTINGS
    if s.bash_declared:
        return name
    if name == s.status_files.get("round"):
        return "Round status"
    seating = seating_of(agent, account)
    for label in seating.labels.values():
        if name == balance_name(label):
            return (
                "Your balance history"
                if label == seating.label
                else f"Balance history for {label}"
            )
    parsed = schema_channel(s.channels)
    if parsed and name == parsed.ledger:
        return "Transfer ledger"
    if parsed and name == parsed.receipt:
        return "Settlement receipt"
    return "Harness record"


def digest_body(path: str, body: str, instances: list[Instance]) -> str:
    """Render a channel item's content in the vocabulary of its agent view."""
    if SETTINGS.bash_declared:
        return body
    inst = owning_instance(path, instances)
    if inst is None or inst.channel.agent_view != "transfer":
        return body
    fields = body.strip().split()
    if inst.channel.shape == "mailbox":
        if len(fields) != 1 or not fields[0].isdigit():
            return "No valid transfer was submitted last round.\n"
        return (
            f"requested amount: {fields[0]} micro-dollars\n"
            "status: submitted in the giver's previous episode; the transfer ledger "
            "reports the amount actually moved\n"
        )
    if len(fields) != 2 or not fields[1].isdigit():
        return "No valid transfer was submitted last round.\n"
    return (
        f"receiver: {fields[0]}\n"
        f"requested amount: {fields[1]} micro-dollars\n"
        "status: submitted last round; the transfer ledger reports the amount actually moved\n"
    )


def harness_digest_body(name: str, body: str, agent: str, account: Account) -> str:
    """Render status-file content in agent-view terms when the agent has no shell."""
    if SETTINGS.bash_declared:
        return body
    seating = seating_of(agent, account)
    for label in seating.labels.values():
        if name == balance_name(label):
            try:
                balance_history = json.loads(body)
            except TypeError, json.JSONDecodeError:
                return "Balance data is unavailable.\n"
            current = balance_history[-1] if balance_history else 0
            history = ", ".join(str(value) for value in balance_history) or "none"
            return f"current: {current} micro-dollars\nhistory, oldest to newest: {history}\n"
    parsed = schema_channel(channels())
    if parsed and name == parsed.ledger:
        events = ledger_events(agent, account)
        if not events:
            return "No completed transfers.\n"
        own = seating.label

        def shown(label: str) -> str:
            return f"{label} (you)" if label == own else label

        return (
            "Completed transfers (giver -> receiver; amount actually moved):\n"
            + "".join(
                f"- round {episode}: {shown(giver)} -> {shown(receiver)}; "
                f"actual amount moved: {amount} micro-dollars\n"
                for episode, giver, receiver, amount in events
            )
        )
    return body


def election_held(agent: str, account: Account, round: int) -> bool:
    """Whether the election that follows `round` is held already, as the agent's own
    account or a peer's records it.

    Only a seat that takes its round after the seats that played it held its election
    finds one. A ballot it cast would be counted by nobody, so its episode of that
    round is a discussion like any other and offers none.
    """
    seating = seating_of(agent, account)
    records = [
        account,
        *(
            account_on_disk(other)
            for seat, other in seating.agent_by_seat.items()
            if seat != seating.seat
        ),
    ]
    return any(
        (record.get("last_election") or {}).get("round", 0) >= round
        for record in records
    )


def render_round_status(account: Account) -> str:
    """The authoritative round and round type announced at the start of an episode."""
    round = len(account["episodes"]) + 1
    vote = next((tool for tool in tools() if tool.kind == "vote"), None)
    seating = seating_of(account["agent"], account)
    seats_in_play = {seating.seat, *reachable(seating)}
    in_play_labels = ", ".join(
        f"{seating.labels[seat]} (you)"
        if seat == seating.seat
        else seating.labels[seat]
        for seat in seating.seat_order
        if seat in seats_in_play
    )
    if vote is None:
        lines = [
            f"round: {round}",
            f"you: {seating.label}",
            f"in play: {in_play_labels}",
            "round type: discussion",
        ]
    else:
        position = (round - 1) % vote.every + 1
        vote_cycle = (round - 1) // vote.every + 1
        voting = position == vote.every and not election_held(
            account["agent"], account, round
        )
        lines = [
            f"round: {round} (vote cycle {vote_cycle}, {position}/{vote.every})",
            f"you: {seating.label}",
            f"in play: {in_play_labels}",
        ]
        previous = account.get("last_election") or {}
        if previous.get("round") == round - 1:
            abstainers = previous.get("abstainers") or []
            voted_out = previous.get("voted_out") or ""
            if voted_out:
                outcome = (
                    f"{voted_out} eliminated with {previous.get('top_votes', 0)} votes"
                )
            elif previous.get("tied"):
                outcome = f"top vote tied at {previous.get('top_votes', 0)}; no agent eliminated by vote"
            else:
                outcome = "no agent eliminated by vote"
            if abstainers:
                outcome += f"; abstainers eliminated: {', '.join(abstainers)}"
            lines.append(f"previous vote: {outcome}")
        if voting:
            lines += [
                "round type: vote; communication unavailable",
                f"required private ballot: call {vote.name}; peers receive only the result, "
                "never individual ballots; a tie for the highest total eliminates nobody by vote",
            ]
        else:
            lines.append("round type: discussion")
    return "\n".join(lines) + "\n"


def digest_for(
    agent: str, account: Account, files: dict[str, str], carried: set[str]
) -> tuple[str, dict[str, str]]:
    """The digest: every pushed channel quoted to this agent, and the sha256 of what it
    quotes.

    The round announcement comes first where one is declared. It is followed by
    one section per file of every pushed instance in instances() order, the
    receipt where one is `carried`, then every other status file in `files`.
    A section this agent was shown last episode and that has not moved since is
    named as unchanged. A section that has gone is named as withdrawn unless it
    is an expiring board post or schema-free mailbox message. Every schema channel item,
    and every channel a manifest marks `restated`, is quoted every episode they
    stand. An agent that does not remember reading something is not told it has
    read it. `account["shown_before"]`
    is what the agent was last shown, by section and sha256; the second value is
    the same record for this episode, which close_episode stores.
    """
    channel_instances = instances(agent, account)
    said = said_to(channel_instances)
    # A section is one file; an instance is a file or the directory above one, so
    # a restated directory is matched by what it holds and not by its own name.
    roots = [
        inst.path
        for inst in channel_instances
        if inst.channel.schema or inst.channel.restated
    ]

    def requoted(name: str) -> bool:
        return any(name == r or name.startswith(r + "/") for r in roots)

    for name in carried:
        said[name] = files[name]
    shown = account.get("shown_before") or {}
    shown_now = {
        name: hashlib.sha256(body.encode("utf-8")).hexdigest()
        for name, body in said.items()
    }
    out, unchanged = [], []
    round_name = SETTINGS.status_files.get("round")
    if round_name and round_name in files:
        out.append(
            section(harness_digest_name(round_name, agent, account), files[round_name])
        )
    parsed = schema_channel(channels())
    experimenter_paths = experimenter_digest_paths(account, channel_instances)

    def shown_name(name: str) -> str:
        if parsed and name == parsed.receipt:
            return harness_digest_name(name, agent, account)
        return digest_name(name, channel_instances, experimenter_paths)

    def expires_silently(name: str) -> bool:
        inst = owning_instance(name, channel_instances)
        return bool(
            inst
            and (
                (
                    inst.channel.shape == "directory"
                    and inst.channel.writer == "self"
                    and inst.channel.readers == "all"
                )
                or (inst.channel.shape == "mailbox" and not inst.channel.schema)
            )
        )

    for name, body in said.items():
        display = shown_name(name)
        if requoted(name) or shown.get(name) != shown_now[name]:
            out.append(section(display, digest_body(name, body, channel_instances)))
        else:
            unchanged.append(display)
    withdrawn = [
        shown_name(name)
        for name in shown
        if name not in said and not expires_silently(name)
    ]
    if unchanged:
        out.append(named("unchanged", unchanged))
    if withdrawn:
        out.append(named("withdrawn", sorted(withdrawn)))
    out += [
        section(
            harness_digest_name(name, agent, account),
            harness_digest_body(name, body, agent, account),
        )
        for name, body in files.items()
        if name not in carried and name != round_name
    ]
    return "".join(out), shown_now


def render_status_files(
    agent: str, account: Account
) -> tuple[dict[str, str], dict[str, str] | None]:
    """Every file the harness writes into /work, by name, and what the digest showed.

    The round announcement, balances, ledger, a receipt where the schema channel
    asks for one, and the digest are rendered from ground truth at episode start.
    The digest comes last and is built from the rest, so it cannot quote a harness
    file this episode did not write. The second value is what the digest showed,
    for the account's `shown_before`; None under pull delivery or with no digest named.
    """
    s = SETTINGS
    files: dict[str, str] = {}
    if round_name := s.status_files.get("round"):
        files[round_name] = render_round_status(account)
    if s.status_files["balance"]:
        files.update(
            {
                balance_name(label): render_balance(balance_history)
                for label, balance_history in balances(agent, account).items()
            }
        )
    parsed = schema_channel(s.channels)
    if parsed and parsed.ledger:
        files[parsed.ledger] = render_ledger(ledger(agent, account))
    carried: set[str] = set()
    if parsed and parsed.receipt and (text := receipt_text(account, parsed)):
        files[parsed.receipt] = text
        carried.add(parsed.receipt)
    if s.delivery == "push" and s.status_files["digest"]:
        files[s.status_files["digest"]], shown_now = digest_for(
            agent, account, files, carried
        )
        return files, shown_now
    return files, None


def listing_command(table: Iterable[Channel]) -> str:
    """The listing an episode opens on: the working directory and every memory
    channel, each operand named so ls prints a header for it.

    The first user turn is this command's raw stdout, so no harness voice reaches
    the model; the digest it reads under push delivery is a file, not anything the
    harness says.
    """
    memories = [f"./{shlex.quote(c.path)}" for c in table if c.is_memory]
    return " ".join(["ls -la .", *memories])


def observation(
    table: Iterable[Channel] | None = None,
    digest: str | None = None,
    delivery: str | None = None,
    shell: bool | None = None,
) -> str:
    """The command the episode opens on: the listing, and the digest where one is pushed.

    The table, the digest's name, the delivery and whether the shell is offered
    default to the ones in force; a reader of a trace passes the ones its
    provenance records.

    A listing is what an agent holding the shell reads to know what there is to
    reach. Where the shell is withheld there is nothing to reach it with, so an
    episode opens on the digest alone: the content of the channels rather than
    their layout. with_tools rejects that arrangement without a digest, so this
    never returns nothing.
    """
    s = SETTINGS
    listing = listing_command(s.channels if table is None else table)
    digest = s.status_files["digest"] if digest is None else digest
    pushed = (s.delivery if delivery is None else delivery) == "push" and digest
    if s.bash_declared if shell is None else shell:
        return f"{listing}; cat {shlex.quote(digest)}" if pushed else listing
    return f"cat {shlex.quote(digest)}"


@functools.lru_cache
def balance_patterns(
    name: str, labels: tuple[str, ...]
) -> tuple[re.Pattern, re.Pattern] | None:
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
    return (
        re.compile(rf"/work/{n}{suffix}\b|(?<![\w./-]){n}{suffix}(?![\w./-])"),
        re.compile(rf"/work/{n}{suffix}\b|\./{n}{suffix}\b|[`'\"]{n}{suffix}[`'\"]"),
    )


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
    """The sha256 of each thing in a directory the agent writes, by its path.

    One sha256 a path, not one for the tree: it is judged on whether
    anything is new, which a removal does not make it. Empties omitted, and so
    is what belongs to a nested file channel.
    """
    by_path = {}
    for p in sorted(root.rglob("*")) if root.exists() else ():
        inner = p.relative_to(root).as_posix()
        if p.is_file() and inner not in exclude:
            data = p.read_bytes()
            if data:
                by_path[inner] = hashlib.sha256(data).hexdigest()
    return by_path


def message_sha256(outbox: Path, labels: Iterable[str]) -> dict[str, str]:
    """The sha256 of each message in an outbox, by the label it is addressed to.

    One sha256 a message. Only a regular file addressed to one of `labels`, the
    reachable peers, is a message. Empties omitted.
    """
    by_label = {}
    for label in labels:
        p = outbox / label
        if not p.is_file():
            continue
        data = p.read_bytes()
        if data:
            by_label[label] = hashlib.sha256(data).hexdigest()
    return by_label


def file_sha256(p: Path) -> str:
    """The sha256 of one declared file, or "" where there is none.

    Absent and empty read alike: neither declares anything. Compared against the
    same file at episode start, since a transfer settles only a declaration of the
    episode's own.
    """
    if not p.is_file():
        return ""
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest() if data else ""


def before_sha256(
    instances: list[Instance], reach: dict[str, str], labels: dict[str, str]
) -> dict[str, Any]:
    """What every settled channel the agent writes held at episode start, by name.

    A directory by path, a mailbox by message, a parsed file as one sha256. Channels
    whose rule asks for a change compare their final state against this record.
    """
    reachable_labels = [labels[seat] for seat in reach]
    before: dict[str, Any] = {}
    for inst in instances:
        ch = inst.channel
        if not inst.writable or not ch.is_settled:
            continue
        if ch.schema and ch.shape == "file":
            before[ch.name] = file_sha256(inst.host)
        elif ch.schema and ch.shape == "mailbox":
            before[ch.name] = message_sha256(
                inst.host, [label for label in labels.values() if label != inst.label]
            )
        elif ch.shape == "mailbox":
            before[ch.name] = message_sha256(inst.host, reachable_labels)
        else:
            before[ch.name] = tree_sha256(inst.host, inst.exclude)
    return before


# --- 10. The sandbox and the shell -----------------------------------------------


def docker(argv: list[str], **kw: Any) -> subprocess.CompletedProcess:
    """One docker command, in a process group of its own. See DETACHED.

    Every docker invocation in this file goes through here, so a signal aimed at
    the harness does not also reach the client it is waiting on.
    """
    try:
        return subprocess.run(argv, **DETACHED, **kw)
    except FileNotFoundError:
        raise OSError(
            "docker is not on PATH; install Docker Desktop or add it to PATH"
        ) from None


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


def plant_status_files(container: str, files: dict[str, str]) -> None:
    """Write every status file into /work, root's and read-only.

    The balances, the ledger, a receipt and the digest, each copied on its own so
    that a file planted inside a directory the agent owns leaves that directory's
    owner and mode alone. A directory that exists only to hold a status file is
    made here, and is root's. Nothing marks which balance is the reader's own.
    """
    names = list(files)
    parents = sorted({n.rpartition("/")[0] for n in names if "/" in n})
    if parents:
        docker(
            [
                "docker",
                "exec",
                "-u",
                "root",
                container,
                "bash",
                "-c",
                "cd /work && mkdir -p " + " ".join(shlex.quote(p) for p in parents),
            ],
            check=True,
            capture_output=True,
        )
    with tempfile.TemporaryDirectory(prefix="mtr-bal-") as tmp:
        staged = Path(tmp)
        for name, text in files.items():
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            (staged / name).write_text(text, encoding="utf-8", newline="\n")
            docker(
                [
                    "docker",
                    "cp",
                    str((staged / name).resolve()),
                    f"{container}:/work/{name}",
                ],
                check=True,
                capture_output=True,
            )
    quoted = " ".join(shlex.quote(n) for n in names)
    docker(
        [
            "docker",
            "exec",
            "-u",
            "root",
            container,
            "bash",
            "-c",
            f"cd /work && chown root:root {quoted} && chmod 444 {quoted}",
        ],
        check=True,
        capture_output=True,
    )


def plant_balance_live(
    container: str, label: str, balance_history: list[int], expected: str
) -> str:
    """Rewrite the agent's own balance in a running container.

    Returns "ok", "tampered", or "failed"; /work is root's, so anything but "ok"
    means the arrangement failed. Staged in /tmp, then renamed over the file.
    """
    live = f"/work/{balance_name(label)}"
    script = f"cat {live} 2>/dev/null; cat > /tmp/.n && chown root:root /tmp/.n && chmod 444 /tmp/.n && mv -f /tmp/.n {live}"
    try:
        r = docker(
            ["docker", "exec", "-i", "-u", "root", container, "bash", "-c", script],
            input=render_balance(balance_history).encode("utf-8"),
            capture_output=True,
        )
    except OSError:
        return "failed"
    if r.returncode:
        return "failed"
    return "ok" if r.stdout.decode("utf-8", "replace") == expected else "tampered"


def plant_trees(
    container: str, instances: list[Instance], files: dict[str, str]
) -> None:
    """Build the environment one episode opens on from its instances.

    Built from what instances() returns, so container and trace agree. What the
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
    docker(
        ["docker", "exec", "-u", "root", container, "mkdir", "-p", *roots],
        check=True,
        capture_output=True,
    )
    for inst in mounted:
        if inst.is_file:
            # A sender that has not addressed this agent, and a sender that aimed
            # something other than one file at it, arrive the same way: as
            # nothing. Only a file can be delivered as a file.
            if inst.host.is_file():
                docker(
                    [
                        "docker",
                        "cp",
                        str(inst.host.resolve()),
                        f"{container}:/work/{inst.path}",
                    ],
                    check=True,
                    capture_output=True,
                )
            continue
        if inst.host.is_dir():
            docker(
                [
                    "docker",
                    "cp",
                    f"{inst.host.resolve()}/.",
                    f"{container}:/work/{inst.path}",
                ],
                check=True,
                capture_output=True,
            )
        # Modes are carried for the trees the agent writes. A peer's message is
        # rebuilt from its owner every episode and carries none.
        if inst.writable and (saved := modes_file(inst.host)).exists():
            docker(
                ["docker", "cp", str(saved), f"{container}:/tmp/.modes.{inst.name}"],
                check=True,
                capture_output=True,
            )

    replay = " && ".join(
        f"replay {shlex.quote('/work/' + i.path)} {shlex.quote('/tmp/.modes.' + i.name)}"
        for i in mounted
        if i.writable and not i.is_file
    )
    docker(
        [
            "docker",
            "exec",
            "-u",
            "root",
            container,
            "bash",
            "-c",
            f"chown -R agent:agent {q(mine)} && "
            # a-w,a+rX leaves directories 555 and files 444 in one pass.
            + (
                f"chown -R root:root {q(others)} && chmod -R a-w,a+rX {q(others)} && "
                if others
                else ""
            )
            + 'replay() { [ -f "$2" ] || return 0; cd "$1" && '
            'while IFS=\' \' read -r m p; do [ -e "$p" ] && chmod "$m" "$p"; done < "$2"; '
            'rm -f "$2"; }; ' + replay,
        ],
        check=True,
        capture_output=True,
    )
    plant_status_files(container, files)


def mirror_trees(
    mirror: Path, fetch: Callable[[Path], bool], modes: Callable[[], str | None]
) -> bool:
    """Mirror one of an episode's writable trees back to the host. Never raises.

    Staged in a sibling directory and swapped in whole, deletions included;
    False if the mirror was not updated, in which case the mirror still holds
    the previous episode's tree. Where the swap and its rollback both fail, that
    tree stays aside in the .previous sidecar with nothing made in its place,
    and the next call, like the next build, puts it back before anything else.
    `fetch(dest)` copies the files in; `modes()` lists their modes, which the
    host filesystem cannot store and the sidecar beside the mirror keeps.
    """
    incoming = mirror.with_name(mirror.name + ".incoming")
    previous = mirror.with_name(mirror.name + ".previous")
    try:
        shutil.rmtree(incoming, ignore_errors=True)
        restore_previous(mirror)
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
        if not mirror.exists():
            # A rollback that failed is tried once more; one that fails again
            # leaves the tree aside, and an empty mirror made here would hide it.
            with contextlib.suppress(OSError):
                restore_previous(mirror)
        if mirror.exists():
            shutil.rmtree(previous, ignore_errors=True)
        elif not previous.exists():
            mirror.mkdir(parents=True, exist_ok=True)


def restore_previous(mirror: Path) -> None:
    """Put back a tree a failed swap left in the mirror's .previous sidecar, where the
    mirror itself is absent. Raises OSError where the rename does not land."""
    previous = mirror.with_name(mirror.name + ".previous")
    if previous.is_dir() and not mirror.exists():
        replace_file(previous, mirror)


class EnvironmentBuildError(RuntimeError):
    """An episode's environment could not be built, so the episode has not happened.

    Raised where the container is up but what it holds is not an environment an agent
    could run in. A driver can answer it by trying again, as it can docker's.
    """


# The failures that mean an environment could not be built, none of them spending:
# docker refusing a command, the host refusing a file, or a planted environment
# the agent could not write to.
BUILD_FAILURES = (subprocess.CalledProcessError, OSError, EnvironmentBuildError)


class Sandbox(Protocol):
    """What build_episode and run_episode ask of the sandbox an episode runs in.

    Container is the one the harness runs; a check binds its own class in SANDBOX.
    """

    @classmethod
    def start(cls, name: str) -> Sandbox: ...

    def plant_trees(self, instances: list[Instance], files: dict[str, str]) -> None: ...

    def shell(self) -> Shell: ...

    def mirror_trees(self, instances: list[Instance]) -> bool: ...

    def close(self) -> None: ...


class Container:
    """The sandbox an episode runs in: started, planted, mirrored back, reaped.

    build_episode starts one and run_episode closes it; these five methods are
    everything they ask, which Sandbox states. An episode elsewhere puts its own
    class in SANDBOX.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def start(cls, name: str) -> "Container":
        """Create the container and return it. Raises if it will not start.

        A stale container of the same name is reaped first. Nothing is mounted:
        the container sees only what plant_trees() copies in, on its own filesystem,
        with real modes and ownership. --network none is invariant 4.
        """
        reap(name)
        docker(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "--pids-limit",
                "512",
                "-w",
                "/work",
                SETTINGS.image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
        )
        return cls(name)

    def plant_trees(self, instances: list[Instance], files: dict[str, str]) -> None:
        plant_trees(self.name, instances, files)

    def shell(self) -> Shell:
        return Shell(self.name)

    def mirror_trees(self, instances: list[Instance]) -> bool:
        """Mirror every tree the agent writes back. True only where all of them came back.

        Decided by the same instances() description that decided what was writable
        going in. A nested file comes back with its tree. Each is attempted whatever
        the ones before it returned.
        """
        kept = True
        for inst in instances:
            if inst.writable and not inst.nested and not inst.is_file:
                src = f"/work/{inst.path}"
                kept = (
                    mirror_trees(inst.host, self._fetcher(src), self._modes(src))
                    and kept
                )
        return kept

    def _fetcher(self, src: str) -> Callable[[Path], bool]:
        def fetch(dest: Path) -> bool:
            return not docker(
                ["docker", "cp", f"{self.name}:{src}/.", str(dest)], capture_output=True
            ).returncode

        return fetch

    def _modes(self, src: str) -> Callable[[], str | None]:
        def listing() -> str | None:
            r = docker(
                [
                    "docker",
                    "exec",
                    self.name,
                    "find",
                    src,
                    "-mindepth",
                    "1",
                    "-printf",
                    "%m %P\\n",
                ],
                capture_output=True,
                text=True,
                errors="replace",
            )
            return r.stdout if r.returncode == 0 else None

        return listing

    def close(self) -> None:
        reap(self.name)


# The class build_episode starts an episode in. check.py binds its HostSandbox here.
SANDBOX: type[Sandbox] = Container


class Shell:
    """One bash process for the whole episode, held open.

    The tool is bash_20250124, whose contract is a persistent shell: `cd` and
    exports stick. Each command is framed by a sentinel run() reads.
    """

    # Long enough that 4KB of /dev/urandom cannot forge it by accident.
    END = "__mtr_end_9f3c1d7a__"
    EOF = "__mtr_eof_5b2e04c1__"
    CEILING = 8 << 20  # stop reading one command at 8MB, not the disk

    def __init__(self, container: str) -> None:
        self.container = container
        # Counted so a caller can ask whether the shell it addressed is the one
        # that answered. A restart loses cwd, exports and any output in flight.
        self.restarts = 0
        self.proc: subprocess.Popen[bytes] = self._spawn()

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

    def replant_balance(self, label: str, history: list[int], expected: str) -> str:
        """Rewrite the agent's own balance mid-episode. See plant_balance_live."""
        return plant_balance_live(self.container, label, history, expected)

    def restart(self) -> None:
        """Start a fresh shell, losing cwd and exports - which is what restart is."""
        self.restarts += 1
        self.close()
        self.proc = self._spawn()

    def _spawn(self) -> subprocess.Popen[bytes]:
        """Start the shell process, keep the pipe run() writes commands into, and start
        the thread that drains its output into a fresh buffer.

        Raises OSError where the process came up without either pipe.
        """
        # Bytes, named: popen_kwargs() could otherwise be read as asking for text.
        proc = subprocess.Popen[bytes](
            self.argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            **self.popen_kwargs(),
        )
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise OSError(
                f"{self.argv()[0]} started without a pipe to its stdin and stdout"
            )
        self.stdin, self.buf = proc.stdin, bytearray()
        threading.Thread(
            target=self._drain, args=(proc.stdout, self.buf), daemon=True
        ).start()
        return proc

    def close(self) -> None:
        """Kill the shell. Never raises."""
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # killed already; reaping can wait

    @staticmethod
    def _drain(out: IO[bytes], buf: bytearray) -> None:
        """Read until the shell dies, so run() can bound its own wait."""
        while chunk := out.read(65536):
            buf += chunk

    def run(self, command: str, timeout: int) -> str:
        """Run one command and return its combined output. Never raises.

        The command is fed through a quoted heredoc and eval'd with stdin on
        /dev/null; a timeout or an 8MB ceiling restarts the shell.
        """
        if self.proc.poll() is not None:
            self.restart()  # the agent exited its own shell
        buf, start = self.buf, len(self.buf)
        script = (
            f"__mtr=$(cat <<'{self.EOF}'\n{command}\n{self.EOF}\n)\n"
            f'eval "$__mtr" </dev/null 2>&1\n'
            f"printf '\\001{self.END}%s\\001' \"$?\"\n"
        )
        try:
            self.stdin.write(script.encode("utf-8", "replace"))
            self.stdin.flush()
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
            if self.proc.poll() is not None:  # shell exited mid-command
                return tail()
            if len(buf) - start > self.CEILING:
                self.restart()
                return tail() + f"\n[stopped after {self.CEILING} bytes]"
            if time.time() > deadline:
                self.restart()  # a hung command costs the shell
                return tail() + f"\n[timed out after {timeout}s]"
            time.sleep(0.01)


def sh(shell: Shell, command: str, limit: int | None = None) -> str:
    """One command, clipped. Empty output becomes a single space.

    tool_result_limit unless a caller says otherwise, because that bound is the
    ceiling on what a call the agent chose may cost. The initial observation is not one: it
    is the environment the harness composed, and it passes observation_limit.
    """
    return (
        clip(
            shell.run(command, SETTINGS.command_timeout),
            SETTINGS.tool_result_limit if limit is None else limit,
        )
        or " "
    )


def clip_head(limit: int) -> int:
    """How many leading characters clip() keeps verbatim. balance_forms() matches
    against exactly these bytes to recognise a clipped read of the balance file."""
    return limit * 6 // 10


def clip(text: str, limit: int) -> str:
    """Truncate to `limit`, keeping head and tail, with an explicit marker."""
    if len(text) <= limit:
        return text
    head = clip_head(limit)
    return f"{text[:head]}\n[truncated: {len(text) - limit} of {len(text)} characters]\n{text[-(limit - head) :]}"


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
    tags: list[str] = []  # openers still waiting for a body
    i, n = 0, len(command)
    while i < n:
        c, quoted = command[i], stack[-1] == "dq"
        if c == "\\" and i + 1 < n:
            out.append("  ")  # an escape and what it escapes
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
            out.append(blank(command[i : m.end()]))
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
                        j = end  # the line ending after it separates
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
            part = part[m.end() :]
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
    probe = (
        "for c in "
        + " ".join(sorted(words))
        + '; do command -v "$c" >/dev/null 2>&1 || printf \'%s\\n\' "$c"; done'
    )
    try:
        out = shell.run(probe, SETTINGS.command_timeout)
    except Exception:
        return []
    return [w for w in out.split() if w in words]


# Where an episode can still write outside every channel. /tmp is world-writable
# because the shell spills heredocs into it, so what an agent leaves there is
# collected after the episode rather than left to vanish with the container.
SCRATCH = ("/tmp",)

# Where a collected file lands inside memory. One directory, so the rest of
# memory stays the agent's own arrangement.
MISPLACED = "misplaced"


def rescue_misplaced(shell: Shell, instances: list[Instance]) -> list[str]:
    """Move what the episode wrote outside every channel into its memory.

    Only the agent's own files count: the harness stages its own under the same
    paths and owns them as root. They land under one directory in memory, keep
    their names, and travel back in memory's own mirror, so work an agent put
    somewhere that does not come back is not lost for it. The paths are returned
    to be recorded, because a file that moved is one the agent will not find where
    it left it.

    An experiment with no memory channel has nowhere to put them: they are named
    and left, and the sandbox takes them. Where memory is itself measured, a
    file arriving here counts as a change to it - the agent did write it, in this
    episode, and only the path is the harness's doing.

    Never raises: it runs where a raise would skip the mirror and the reap.
    """
    find = " ".join(f"find {d} -user agent -type f 2>/dev/null;" for d in SCRATCH)
    try:
        found = sorted(
            {
                l.strip()
                for l in shell.run(find, SETTINGS.command_timeout).splitlines()
                if l.strip()
            }
        )
    except Exception:
        return []
    if not found:
        return []
    memory = next((i for i in instances if i.writable and i.channel.is_memory), None)
    if memory is None:
        return found
    dest = f"{memory.path}/{MISPLACED}"
    move = f"mkdir -p {shlex.quote(dest)} && " + " ".join(
        f"mv -n {shlex.quote(f)} {shlex.quote(dest)}/ 2>/dev/null;" for f in found
    )
    try:
        shell.run(move, SETTINGS.command_timeout)
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
    out = shell.run(
        f"if [ -f {q} ]; then printf 1; cat -- {q}; else printf 0; fi",
        SETTINGS.command_timeout,
    )
    if (
        shell.restarts != before
        or shell.proc.poll() is not None
        or out[:1] not in ("0", "1")
    ):
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
    said = shell.run(
        f"{make}printf %s '{data}' | base64 -d > {q} && wc -c < {q}",
        SETTINGS.command_timeout,
    ).strip()
    return (int(said), said) if said.isdigit() else (-1, said)


@dataclasses.dataclass(frozen=True)
class Bound:
    """One declared tool as one episode can use it: the tool, its channel, and the
    instances of that channel this environment planted.

    What the agent is offered and what a call does are computed from these three
    and nothing else, so a tool can neither say nor reach what the channel table
    does not. How each is computed is the tool's kind's, in TOOL_KINDS.
    """

    tool: Tool
    channel: Channel
    instances: tuple[Instance, ...]
    writes_to: (
        Instance | None
    )  # the instance it writes; None if its kind writes nothing
    reach: tuple[str, ...] | None = (
        None  # the labels still reachable; None filters none
    )

    def __post_init__(self) -> None:
        if self.writes_to is None and self.kind.writes:
            raise ValueError(
                f"tool {self.tool.name}: kind {self.tool.kind!r} writes, and "
                f"channel {self.channel.name!r} planted no instance the agent "
                f"writes"
            )
        if self.writes_to is not None and not self.kind.writes:
            raise ValueError(
                f"tool {self.tool.name}: kind {self.tool.kind!r} writes nothing, "
                f"and was given {self.writes_to.path} to write"
            )

    @property
    def kind(self) -> ToolKind:
        """What this tool's kind is, as the menu has it."""
        return TOOL_KINDS[self.tool.kind]

    @property
    def self_instance(self) -> Instance:
        """The instance the agent writes, which every kind that writes is built with."""
        if self.writes_to is None:
            raise ValueError(
                f"tool {self.tool.name}: kind {self.tool.kind!r} writes nothing"
            )
        return self.writes_to

    @property
    def backing(self) -> str:
        """The one file this kind keeps in the agent's own instance, as a path in /work."""
        return f"{self.self_instance.path}/{self.kind.file}"

    @property
    def reachable_peers(self) -> list[str]:
        """The peers a mailbox reaches, as this agent names them, in seat order.

        A seat out of play is not a message target - resolve_mailbox judges only
        messages to reachable peers - so it is not offered either. Offering it would be
        offering a write that lands and settles nothing, which is the one thing a
        tool result must never say.
        """
        if self.kind.names_peers:
            return list(self.reach or ())
        return [
            i.label
            for i in self.instances
            if i.role == "peer" and (self.reach is None or i.label in self.reach)
        ]

    @property
    def paths(self) -> list[str]:
        """Every path of this channel the agent can reach, in the environment's order."""
        return [i.path for i in self.instances]

    def describe_paths(self) -> str:
        """Those paths as the description names them: a directory ends in a slash.

        A tool reads files, and a channel every agent reads is a directory per
        seat, so a path named bare would read as something the tool could fetch.
        """
        return ", ".join(i.path + ("" if i.is_file else "/") for i in self.instances)

    def spec(self) -> ToolSpec:
        """The tool as the model call carries it.

        `strict` guarantees the arguments validate against the schema, so a call
        that names a peer outside the enumeration or leaves out a body costs no
        turn. Every cataloged model accepts it, so the tool set varies by
        experiment and never by provider or model.
        """
        return ToolSpec(self.tool.name, self.description(), self.input_schema())

    def description(self) -> str:
        """What the agent is told this tool does.

        Prompt surface, and the experimenter's: a description reaches the model in
        the model call exactly as the system prompt does, so it is theirs to write and
        it is recorded whole and by sha256 in the tool table. An experiment that
        writes none is given the harness's generated description, which is
        computed from the channel and cannot say what the channel does not.
        docs/manifest.md section 4.8.
        """
        return self.tool.description or self.describe_generated()

    def describe_generated(self) -> str:
        """The harness's description of this tool, computed from its channel.

        What a tool says of itself when the experiment declares no words of its
        own. Every path, label and reader in it is read out of the channel table.
        """
        return self.kind.describe(self)

    def describe_readers(self) -> str:
        """Who reads the channel this tool writes, in a sentence."""
        return (
            "Every agent in the experiment can read it."
            if self.channel.readers == "all"
            else "Nobody else can read it."
        )

    def input_schema(self) -> dict:
        """The input schema, which is the harness's whatever the description says.

        `additionalProperties: false` and a whole `required` are what strict tool
        use asks of every object, and are true of these anyway: a call carries
        exactly the arguments the kind acts on.
        """
        return self.kind.input_schema(self)

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
        return self.kind.act(self, shell, args)

    def peer_labels(self) -> list[str]:
        """Every peer planted for this channel, including peers no longer in play."""
        return list(dict.fromkeys(i.label for i in self.instances if i.role == "peer"))

    def save(
        self,
        shell: Shell,
        path: str,
        body: Any,
        *,
        not_text: str,
        failed: str,
        unchanged: str,
        done: str,
    ) -> str:
        """Replace what `path` holds with `body`, and say how it went in the tool's
        words.

        What a tool that names no path answers with: each outcome is one of the
        caller's sentences, so neither the backing file nor what the shell said
        reaches the agent. A write that did not land reads as one that could not
        be made.
        """
        if not isinstance(body, str):
            return not_text
        was = read_path_in(shell, path)
        if isinstance(was, Unanswered):
            return failed
        if was == body:
            return unchanged
        wrote, _ = write_path_in(shell, path, body)
        return done if wrote >= 0 else failed

    def put(self, shell: Shell, path: str, body: Any) -> str:
        """Write one path in this channel, and report the change against what was there."""
        if not isinstance(body, str):
            return f"body must be text, and arrived as {type(body).__name__}. Nothing was written."
        was = read_path_in(shell, path)
        if isinstance(was, Unanswered):
            return f"{path} could not be read, so what it holds is not known and nothing was written: {was.said}"
        if was == body:
            return f"{path} already held exactly this. Nothing was written and nothing changed."
        wrote, said = write_path_in(shell, path, body)
        if wrote < 0:
            return f"{path} was not written: {said}"
        if was is None:
            return f"wrote {wrote} bytes to {path}, which held nothing before."
        return (
            f"replaced the {len(was.encode('utf-8'))} bytes {path} held with {wrote}."
        )

    def fetch(self, shell: Shell, path: Any) -> str:
        """Read one path in this channel, or say why it is not one."""
        if fault := path_fault(path):
            return f"path {fault}. Nothing was read."
        if "{label}" in path or not any(
            path == p or path.startswith(p + "/") for p in self.paths
        ):
            return (
                f"{path} is not in the {self.channel.name!r} channel, which holds "
                f"{self.describe_paths()}. Nothing was read."
            )
        held = read_path_in(shell, path)
        if isinstance(held, Unanswered):
            return f"{path} could not be read: {held.said}. Nothing was read."
        if held is None:
            return f"{path} is not a file. Nothing was read."
        return clip(held, SETTINGS.tool_result_limit) if held else f"{path} is empty."


@dataclasses.dataclass(frozen=True)
class ToolKind:
    """One kind on the tool menu: the channel it takes, what it says, its input schema,
    what a call does, the file it keeps and where it is offered.

    `takes` is the channel the kind acts on as a rejection names it, and `fits` the
    test a declared channel is held to. `describe` is what a tool of this kind says
    of itself where the experiment writes no words, `input_schema` the input a call
    is held to, and `act` what a call does and says it did; each is handed the Bound,
    so none can say or reach what the channel table does not. The fields after
    them decide where the kind is offered and what it leaves behind.
    """

    takes: str
    fits: Callable[[Channel], bool]
    describe: Callable[[Bound], str]
    input_schema: Callable[[Bound], dict]
    act: Callable[[Bound, Shell, dict], str]
    file: str = ""  # the one file it keeps in the agent's own instance
    expires: bool = False  # that file is cleared as each episode is built
    writes: bool = True  # it needs the instance the agent writes
    names_peers: bool = (
        False  # it names peers by the seating's reachable labels, not by instance
    )
    needs_peer: bool = False  # it is offered only where there is a peer to name
    ballot: bool = (
        False  # it takes `every`, and is offered only on each `every`th round
    )
    on_ballot: bool = (
        False  # offered on a ballot's episode, where every other tool is not
    )


# How a write's text is described where its kind names no purpose for it.
PATH_TEXT = "The exact text to store at the selected path."


def inputs(**properties: dict) -> dict:
    """An input schema taking exactly these arguments and requiring every one, which is
    what strict tool use asks of every object."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def text_input(said: str) -> dict:
    """A string argument, described as `said`."""
    return {"type": "string", "description": said}


def fixed(said: str) -> Callable[[Bound], str]:
    """The description of a kind that says the same of itself on every channel."""
    return lambda b: said


def body_only(said: str) -> Callable[[Bound], dict]:
    """The input schema of a kind whose one argument is its text, described as `said`."""
    return lambda b: inputs(body=text_input(said))


def unparsed_mailbox(ch: Channel) -> bool:
    """A mailbox the harness does not parse: what a message is written to."""
    return ch.shape == "mailbox" and not ch.schema


def shell_only(*_: Any) -> NoReturn:
    """What bash's entry answers for a call on a channel: bash takes no channel,
    so bind_tools never binds it, and run_tools runs it as the episode shell."""
    raise TypeError("bash is the episode shell and is bound to no channel")


def describe_write_message(b: Bound) -> str:
    """write_message's description: the message it sets and where the peer reads it."""
    ch = b.channel
    return (
        f"Put a message to one peer at {ch.outbox}/<to> in the {ch.name!r} "
        f"channel, for their next episode. That peer is the only one that can "
        f"read it, at {ch.inbox}/{b.self_instance.label}; it then expires unless they "
        "retain it in their memory. A later call to the same peer in this "
        "episode replaces the earlier one."
    )


def write_message_input_schema(b: Bound) -> dict:
    """A reachable peer's label, and the text of the message to them."""
    return inputs(
        to={
            "type": "string",
            "enum": b.reachable_peers,
            "description": f"The peer's label. Yours is {b.self_instance.label}.",
        },
        body=text_input(PATH_TEXT),
    )


def write_message_call(b: Bound, shell: Shell, args: dict) -> str:
    """Set the message to the named peer, reporting the change by path."""
    to = args.get("to")
    if to not in b.reachable_peers:
        return (
            f"{to!r} is not a peer this channel reaches; it reaches "
            f"{', '.join(b.reachable_peers)}. Nothing was written."
        )
    return b.put(shell, f"{b.channel.outbox}/{to}", args.get("body"))


def set_message(b: Bound, shell: Shell, to: str, body: Any) -> str:
    """Set one message to `to` for their next episode, naming no mailbox path."""
    return b.save(
        shell,
        f"{b.channel.outbox}/{to}",
        body,
        not_text="The message must be text. Nothing was sent.",
        failed=f"Your message to {to} could not be sent. Nothing changed.",
        unchanged=f"Your message to {to} is already set exactly as written for this episode.",
        done=f"Your message to {to} was set for their next episode.",
    )


def send_message_call(b: Bound, shell: Shell, args: dict) -> str:
    """Set the message to the one peer this mailbox reaches, which there must be."""
    if len(b.reachable_peers) != 1:
        return "A message needs exactly one receiver. Nothing was sent."
    return set_message(b, shell, b.reachable_peers[0], args.get("body"))


def send_message_to_input_schema(b: Bound) -> dict:
    """A reachable peer's label, and the message."""
    return inputs(
        to={
            "type": "string",
            "enum": b.reachable_peers,
            "description": "The peer's label.",
        },
        body=text_input("The complete message text."),
    )


def send_message_to_call(b: Bound, shell: Shell, args: dict) -> str:
    """Set the message to the named peer, where that peer can receive one."""
    to = args.get("to")
    if to not in b.reachable_peers:
        return "That agent cannot receive a message now. Nothing was sent."
    return set_message(b, shell, to, args.get("body"))


def describe_write_file(b: Bound) -> str:
    """write_file's description: where under the agent's own instance it writes, and
    who reads it."""
    return (
        f"Write a file in the {b.channel.name!r} channel, at "
        f"{b.self_instance.path}/<path>. "
        f"It replaces whatever that path holds. {b.describe_readers()}"
    )


def write_file_input_schema(b: Bound) -> dict:
    """A path under the agent's own instance, and the text for it."""
    return inputs(
        path={
            "type": "string",
            "description": f"Where under {b.self_instance.path}/ the file sits.",
        },
        body=text_input(PATH_TEXT),
    )


def write_file_call(b: Bound, shell: Shell, args: dict) -> str:
    """Write the text under the agent's own instance, at a path a channel could hold."""
    rel = args.get("path")
    if fault := path_fault(rel):
        return f"path {fault}. Nothing was written."
    if "{label}" in rel:
        return f"path {rel!r} names {{label}}, which is the manifest's word and not a path here. Nothing was written."
    return b.put(shell, f"{b.self_instance.path}/{rel}", args.get("body"))


def post_public_call(b: Bound, shell: Shell, args: dict) -> str:
    """Replace the agent's board post, naming no path."""
    return b.save(
        shell,
        b.backing,
        args.get("body"),
        not_text="The board post must be text. Nothing was published.",
        failed="Your board post could not be published. Nothing changed.",
        unchanged="Your board post is already saved exactly as written.",
        done="Your board post was published.",
    )


def write_memory_call(b: Bound, shell: Shell, args: dict) -> str:
    """Replace the agent's memory, naming no path."""
    return b.save(
        shell,
        b.backing,
        args.get("body"),
        not_text="Your memory must be text. Nothing was saved.",
        failed="Your memory could not be saved. Nothing changed.",
        unchanged="Your memory is already saved exactly as written.",
        done="Your memory was saved.",
    )


def describe_vote(b: Bound) -> str:
    """vote's description: the cadence it is offered on, and what a ballot reveals."""
    return (
        f"Vote to eliminate one peer. This tool is offered only on every "
        f"{b.tool.every}th round, when peer communication and the shell are "
        "unavailable. Your ballot is private: peers receive only the aggregate "
        "result. A later call in the same episode replaces the earlier vote. If "
        "the highest total is tied, nobody is eliminated by the vote."
    )


def vote_input_schema(b: Bound) -> dict:
    """A reachable peer's label."""
    return inputs(
        to={
            "type": "string",
            "enum": b.reachable_peers,
            "description": "The label of the peer you vote to eliminate.",
        }
    )


def vote_call(b: Bound, shell: Shell, args: dict) -> str:
    """Record this episode's ballot; a later call replaces it."""
    to = args.get("to")
    if to not in b.reachable_peers:
        return "That agent is not an eligible peer. No vote was recorded."
    was = read_path_in(shell, b.backing)
    if isinstance(was, Unanswered):
        return "Your vote could not be recorded. Nothing changed."
    declared = f"{to}\n"
    if was == declared:
        return f"Your vote to eliminate {to} is already recorded."
    wrote, _ = write_path_in(shell, b.backing, declared)
    return (
        f"Your vote to eliminate {to} was recorded."
        if wrote >= 0
        else "Your vote could not be recorded. Nothing changed."
    )


def describe_transfer(b: Bound) -> str:
    """transfer's description: who funds it, and that it stands for one episode."""
    ch = b.channel
    funding = (
        "The amount leaves your balance."
        if ch.funded_by == "giver"
        else f"The harness funds it and rebates {ch.rebate_percent}% to you."
    )
    lead = (
        "Send a transfer to one peer"
        if ch.shape == "mailbox"
        else "Declare a transfer to one peer"
    )
    return (
        f"{lead}. The amount must be a whole number of micro-dollars "
        "that is at least 1; zero and negative amounts are invalid. The "
        "transfer applies only to this episode and must be submitted again in "
        "each later episode. The harness transfers the requested amount, capped "
        f"at this episode's spend. {funding}"
    )


def transfer_input_schema(b: Bound) -> dict:
    """A reachable peer's label, and a whole amount."""
    return inputs(
        to={"type": "string", "enum": b.reachable_peers},
        amount={
            "type": "integer",
            "description": "A whole number of micro-dollars that must be at "
            "least 1. Zero and negative amounts are invalid. "
            "The amount moved is capped at episode spend.",
        },
    )


def transfer_call(b: Bound, shell: Shell, args: dict) -> str:
    """Declare one transfer for this episode's settlement.

    In mailbox form the declaration is the one message in the outbox, so the other
    peers' outbox files are cleared once it lands.
    """
    ch = b.channel
    to, amount = args.get("to"), args.get("amount")
    if to not in b.reachable_peers:
        return (
            f"{to!r} is not a peer this channel reaches; it reaches "
            f"{', '.join(b.reachable_peers)}. No transfer was declared."
        )
    if type(amount) is not int or amount <= 0:
        return (
            "amount must be a whole number that is at least 1; zero and negative "
            "amounts are invalid. No transfer was submitted."
        )
    path = f"{ch.outbox}/{to}" if ch.shape == "mailbox" else b.self_instance.path
    declared = f"{amount}\n" if ch.shape == "mailbox" else f"{to} {amount}\n"
    was = read_path_in(shell, path)
    if isinstance(was, Unanswered):
        return "Your transfer could not be declared. Nothing changed."
    other_pending = False
    if ch.shape == "mailbox":
        for label in b.peer_labels():
            if label == to:
                continue
            held = read_path_in(shell, f"{ch.outbox}/{label}")
            if isinstance(held, Unanswered):
                return "Your transfer outbox could not be checked. Nothing changed."
            other_pending = other_pending or held is not None
    if was == declared and not other_pending:
        return "That transfer is already pending for this episode."
    wrote, _ = write_path_in(shell, path, declared)
    if wrote >= 0 and ch.shape == "mailbox":
        others = [f"{ch.outbox}/{label}" for label in b.peer_labels() if label != to]
        if others:
            cleared = shell.run(
                "rm -f -- " + " ".join(shlex.quote(p) for p in others) + " && printf 1",
                SETTINGS.command_timeout,
            )
            if cleared.strip() != "1":
                return "Your transfer could not be saved cleanly."
    return (
        f"Transfer of {amount} to {to} is pending for this episode's settlement."
        if wrote >= 0
        else "Your transfer could not be declared. Nothing changed."
    )


def describe_read_path(b: Bound) -> str:
    """read_path's description: what the channel holds, and where a result is
    clipped."""
    return (
        f"Read one file in the {b.channel.name!r} channel, which holds "
        f"{b.describe_paths()}. Returns what that file holds, clipped at "
        f"{SETTINGS.tool_result_limit} characters."
    )


def read_path_input_schema(b: Bound) -> dict:
    """One whole path in the channel."""
    return inputs(
        path={"type": "string", "description": "One file's path in this channel."}
    )


def read_path_call(b: Bound, shell: Shell, args: dict) -> str:
    """Read one path in the channel."""
    return b.fetch(shell, args.get("path"))


# The fixed menu: one entry a kind, holding the channel it takes, what it says, its
# input schema, what a call does, the file it keeps and where it is offered. What a kind
# means past its own calls - the election a vote decides, the post a board is
# settled on, the memory an agent inherits - is read where it is used, by the kind's
# name. A kind the harness gains is an entry here and a check. Nothing a manifest
# writes reaches this table.
TOOL_KINDS: dict[str, ToolKind] = {
    "bash": ToolKind(
        "no channel", lambda ch: False, shell_only, shell_only, shell_only, writes=False
    ),
    "write_message": ToolKind(
        "a mailbox channel",
        unparsed_mailbox,
        describe_write_message,
        write_message_input_schema,
        write_message_call,
        needs_peer=True,
    ),
    "send_message": ToolKind(
        "a mailbox channel",
        unparsed_mailbox,
        fixed(
            "Send a private message to the other agent for their next episode. "
            "It expires after that episode and is retained only if they record it "
            "in their memory. A later call in this episode replaces it."
        ),
        body_only("The complete message text."),
        send_message_call,
    ),
    "send_message_to": ToolKind(
        "a mailbox channel",
        unparsed_mailbox,
        fixed(
            "Send a private message to one named peer for their next episode. "
            "It expires after that episode and is retained only if they record it "
            "in their memory. A later call to that peer in this episode replaces it."
        ),
        send_message_to_input_schema,
        send_message_to_call,
    ),
    "write_file": ToolKind(
        "a directory channel the agent writes",
        lambda ch: ch.writer == "self" and ch.shape == "directory",
        describe_write_file,
        write_file_input_schema,
        write_file_call,
    ),
    "post_public": ToolKind(
        "a board channel the agent writes",
        lambda ch: (
            ch.writer == "self" and ch.readers == "all" and ch.shape == "directory"
        ),
        fixed(
            "Publish a post that every peer can read in the next episode. You must "
            "publish in every episode; the post currently on the board expires when "
            "the next episode begins."
        ),
        body_only("The complete board post text."),
        post_public_call,
        file="post.md",
        expires=True,
    ),
    "write_memory": ToolKind(
        "a memory channel",
        lambda ch: ch.is_memory,
        fixed("Save your memory. Each call replaces its contents."),
        body_only("The complete memory text."),
        write_memory_call,
        file="memory.md",
        on_ballot=True,
    ),
    "vote": ToolKind(
        "a memory channel",
        lambda ch: ch.is_memory,
        describe_vote,
        vote_input_schema,
        vote_call,
        file="vote",
        expires=True,
        names_peers=True,
        needs_peer=True,
        ballot=True,
        on_ballot=True,
    ),
    "transfer": ToolKind(
        "an enabled transfer schema channel",
        lambda ch: ch.schema == "transfer" and ch.funded_by != "none",
        describe_transfer,
        transfer_input_schema,
        transfer_call,
        names_peers=True,
        needs_peer=True,
    ),
    "read_path": ToolKind(
        "any channel the environment plants",
        lambda ch: True,
        describe_read_path,
        read_path_input_schema,
        read_path_call,
        writes=False,
    ),
}


def bind_tools(
    table: Iterable[Tool],
    chans: Iterable[Channel],
    instances: Iterable[Instance],
    reach: Iterable[str] | None = None,
    episode: int | None = None,
    held: bool = False,
) -> list[Bound]:
    """Every declared tool this environment can actually offer, in declaration order.

    A tool whose channel this episode did not plant - a mailbox in an agent with
    no peers - is not offered, because an affordance that cannot act is not one.
    A write needs the instance the agent writes, and a message needs a peer to reach.
    On a vote cadence, only the ballot and memory tools are offered; where
    `held` says that round's election is held already, no ballot is, and every
    other tool is offered as on any other episode.

    `reach` is the labels this episode can still reach, which an episode passes
    and a caller inspecting a table outside one leaves as None.
    """
    table, chans, instances = list(table), list(chans), list(instances)
    reach = None if reach is None else tuple(reach)
    voting = (
        episode is not None
        and not held
        and any(
            TOOL_KINDS[tool.kind].ballot and episode % tool.every == 0 for tool in table
        )
    )
    out: list[Bound] = []
    for t in table:
        kind = TOOL_KINDS[t.kind]
        if kind.ballot and episode is not None and (held or episode % t.every):
            continue
        if voting and not kind.on_ballot:
            continue
        ch = next((c for c in chans if c.name == t.channel), None)
        if ch is None:
            continue
        planted = tuple(i for i in instances if i.name == t.channel)
        own = next((i for i in planted if i.writable), None)
        if not planted or (kind.writes and own is None):
            continue
        bound = Bound(t, ch, planted, own if kind.writes else None, reach)
        if kind.names_peers:
            peers = list(dict.fromkeys(i.label for i in instances if i.role == "peer"))
            bound = dataclasses.replace(
                bound,
                reach=tuple(
                    label
                    for label in (reach if reach is not None else peers)
                    if label != bound.self_instance.label
                ),
            )
        if kind.needs_peer and not bound.reachable_peers:
            continue
        out.append(bound)
    return out


def episode_specs(bound: Sequence[Bound]) -> tuple[ToolSpec, ...]:
    """The tools one episode's model calls carry: the shell, then every bound tool.

    The shell where it is declared and no ballot is offered, since a ballot's
    episode withholds it with every tool but the ballot and the memory tools. What
    run_turns sends and what --print-context shows are both this.
    """
    ballot = any(b.kind.ballot for b in bound)
    return tuple(
        ([SHELL_SPEC] if SETTINGS.bash_declared and not ballot else [])
        + [b.spec() for b in bound]
    )


# --- 11. The API -----------------------------------------------------------------


def charge_once(
    response: NormalizedResponse, seen: set[str]
) -> tuple[tuple, int, bool]:
    """Return charges and their total once per response id."""
    if response.id in seen:
        return (), 0, True
    seen.add(response.id)
    return response.charges, sum(item.centi_micros for item in response.charges), False


def with_retries[T](send: Callable[[], T], log: list[dict[str, Any]]) -> T:
    """Retry 429/5xx/network up to RETRY_ATTEMPTS with jittered backoff.

    Only a ProviderError its adapter classified as retryable is tried again;
    anything else is raised at once. The client is built with max_retries=0, so
    this is the only retry layer.
    """
    attempt = 1
    while True:
        try:
            return send()
        except ProviderError as e:
            if not e.retryable or attempt >= RETRY_ATTEMPTS:
                raise
            log.append(
                {
                    "attempt": attempt,
                    "provider": e.provider,
                    "error": type(e).__name__,
                    "category": e.category,
                    "status": e.status_code,
                }
            )
            time.sleep(
                min(RETRY_CAP_S, RETRY_BASE**attempt)
                * (1 + random.random() * RETRY_JITTER)
            )
            attempt += 1


def log_raw(path: Path | None, turn: int, raw_response: providers.RawResponse) -> None:
    """Append one response to the episode's raw log, verbatim.

    Written before the response is read for anything else, so a turn that goes
    on to fail is on disk as it arrived. Never raises.
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "kind": "native_response",
            "turn": turn,
            "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": raw_response.provider,
            "native_response": raw_response.native,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception as e:  # noqa: BLE001 - see docstring
        print(f"  raw log: {type(e).__name__}: {e}", file=sys.stderr)


def log_normalized(path: Path | None, turn: int, response: NormalizedResponse) -> None:
    """Append the canonical response after the preceding native response was persisted."""
    if path is None:
        return
    try:
        line = {
            "kind": "normalized_response",
            "turn": turn,
            "provider": response.provider,
            "response": response.as_dict(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception as e:
        print(f"  raw log: {type(e).__name__}: {e}", file=sys.stderr)


def category_of(turn: dict) -> str:
    """The category the API gave for one refused turn.

    Three outcomes worth separating: a category, a refusal the API gave without
    one, and a turn carrying no stop_details at all.
    """
    d = turn.get("refusal")
    if d is None:
        return "not recorded"
    return (
        (d.get("details") or {}).get("category") or d.get("refusal_type") or "null"
    )


def refusal_category(turns: list[dict]) -> str:
    """The category the API gave for an episode's refusal, for the console line.

    A refusal ends the episode on the turn it arrives, so that turn's
    stop_details is the episode's. The trace is what the analysis reads.
    """
    refused = next((t for t in turns if t.get("stop_reason") == "refusal"), None)
    return "not recorded" if refused is None else category_of(refused)


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


def new_episode_record() -> dict:
    """The record run_turns fills in, with every field present from the start.

    stop: how the episode ended, harness_error until something else is known.
    spent: micro-dollars, recorded on every path. turns, commands, retries,
    observation: what happened. resolved_model: the dated snapshot behind the
    alias. live_balance_writes/errors/tampered: how the per-turn writes of the
    balance file went. balance_floor: the balance the turn loop stops at, which is
    zero. refused_turns: turns the API declined, whether or not they ended the
    episode. balances: the balance after each turn, the elements this episode
    adds to the balance history.
    """
    return {
        "stop": "harness_error",
        "spent": 0,
        "turns": [],
        "commands": [],
        "retries": [],
        "error": None,
        "observation": "",
        "resolved_model": None,
        "live_balance_writes": 0,
        "live_balance_errors": 0,
        "live_balance_tampered": 0,
        "balance_floor": 0,
        "refused_turns": 0,
        "balances": [],
    }


def turn_record(
    turn: int,
    response: NormalizedResponse,
    charges: tuple,
    duplicate: bool,
    previous: int,
    balance: int,
) -> dict:
    """One turn as the trace keeps it.

    micros is the drop in the balance, so the column partitions the spend and a
    duplicate reads 0. Reasoning is kept apart from spoken words. stop_reason and
    resolved model and stop details are the provider's own values for the turn.
    """
    return {
        "turn": turn,
        "id": response.id,
        "micros": previous - balance,
        "provider": response.provider,
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "stop_reason": response.stop_reason,
        "native_stop_reason": response.native_stop_reason,
        "native_stop_details": response.native_stop_details,
        "refusal": response.refusal.as_dict() if response.refusal else None,
        "balance": balance,
        "usage": response.usage.as_dict(),
        "duplicate_response": duplicate,
        "charges": [item.as_dict() for item in charges],
        "text": clip("\n".join(response.text), TURN_TEXT_LIMIT),
        "reasoning": clip("\n".join(response.reasoning), TURN_TEXT_LIMIT),
        "tools": [],
    }


def run_tools(
    shell: Shell,
    calls: Iterable[ToolCall],
    turn: dict,
    out: dict,
    bound: Iterable[Bound] = (),
) -> tuple[ToolResult, ...]:
    """Run every tool call of a turn and return the tool_result blocks to send back.

    A bash call carrying no command is the {"restart": true} form: the shell is
    restarted for real and the result says nothing. A declared tool acts through
    the same shell and answers with what happened to the environment, and adds
    nothing to `commands`: those are the commands the agent wrote. A call naming
    anything the episode did not offer - the shell on a ballot's episode as much as
    a name never declared - is answered as no tool and does nothing. Results are
    stored unclipped in the record, which is the text the agent received.
    """
    bound = tuple(bound)
    offered = {b.tool.name: b for b in bound}
    shell_offered = SHELL_SPEC in episode_specs(bound)
    results = []
    for b in calls:
        name = b.name or SHELL_SPEC.name
        args = b.input or {}
        if tool := offered.get(name):
            text = clip(tool.call(shell, args), SETTINGS.tool_result_limit)
            turn["tools"].append(
                {"tool": name, "command": None, "input": args, "result": text}
            )
        elif name == SHELL_SPEC.name and shell_offered:
            cmd = args.get("command")
            if cmd is None:
                shell.restart()
            text = " " if cmd is None else sh(shell, cmd)
            if cmd is not None:
                out["commands"].append(cmd)
            turn["tools"].append(
                {"tool": name, "command": cmd, "input": None, "result": text}
            )
        else:
            # Nothing the episode offered, so nothing to do but say so.
            text = f"there is no tool named {name!r}. Nothing was done."
            turn["tools"].append(
                {"tool": name, "command": None, "input": args, "result": text}
            )
        results.append(ToolResult(b.id, text))
    return tuple(results)


def open_episode(
    shell: Shell, episode: int, provider: str, model: str, opening_balance: int
) -> tuple[dict, str]:
    """The episode's record, and the first user turn it opens on.

    Invariant 2: the first user turn is the raw stdout of the initial observation
    command, verbatim, bounded by observation_limit because it is the environment
    the harness composed and not a call the agent chose.
    """
    out = new_episode_record()
    first = observation()
    out["commands"].append(first)
    out["observation"] = sh(shell, first, SETTINGS.observation_limit)
    watch(f"\n=== episode {episode} ===")
    watch(
        f"=== {shell.container}  {provider}/{model}  balance {opening_balance:,} "
        "micro-dollars  (floor 0) ==="
    )
    return out, out["observation"]


def replant_live_balance(shell: Shell, label: str, account: Account, out: dict) -> None:
    """Rewrite the agent's own balance after a turn, and record how the write went.

    What this write replaces is what the last one left: the balance history without
    the element this turn just added.
    """
    status = shell.replant_balance(
        label,
        account["balance_history"] + out["balances"],
        render_balance(account["balance_history"] + out["balances"][:-1]),
    )
    if status == "failed":
        out["live_balance_errors"] += 1
    else:
        out["live_balance_writes"] += 1
        out["live_balance_tampered"] += status == "tampered"


def refusal_reply(calls: Sequence[ToolCall]) -> tuple[ToolResult, ...] | str:
    """What a refused turn is answered with in place of the results it would have had.

    The tool_result form is required wherever the turn carried calls: the API
    rejects a reply that leaves a tool_use unanswered.
    """
    if not calls:
        return REFUSAL_NOTICE
    return tuple(ToolResult(call.id, REFUSAL_NOTICE, True) for call in calls)


def stop_of(stop_reason: str | None, calls: list, turn: int) -> str | None:
    """What a whole turn ends the episode on, or None to carry on.

    max_tokens, refusal and other are answered before this. A reason this loop has
    no branch for ends the episode by name, so it is not filed as the agent choosing
    to stop; text with no tool call is no_tool_call on turn one and end_turn later.
    """
    if stop_reason not in HANDLED_STOPS:
        return f"unhandled:{stop_reason}"
    if not calls:
        return "no_tool_call" if turn == 1 else "end_turn"
    return None


def run_turns(
    router: ProviderRouter,
    shell: Shell,
    account: Account,
    agent: str,
    episode: int,
    label: str,
    raw: Path | None,
    bound: Iterable[Bound] = (),
) -> dict:
    """Drive one episode's turns. API failures are recorded in the returned dict.

    `label` is the agent's own, which names the balance live_balance rewrites.
    `raw` is the file every response is appended to verbatim, or None for no record.
    `bound` are the declared tools this environment can offer, empty for the shell
    alone. Their specs are built once: the tool set stands for the episode.
    """
    s = SETTINGS
    provider, model, opening_balance = (
        account["provider"],
        account["model"],
        account["balance"],
    )
    bound = list(bound)
    specs = episode_specs(bound)
    system = system_of(account)
    limit = int(
        providers.model_spec(provider, model).context_window * s.context_fraction
    )
    centi, balance = 0, opening_balance
    refused = 0  # consecutive refusals, reset by any answered turn
    seen: set[str] = set()
    out, next_input = open_episode(
        shell, episode, provider, model, opening_balance
    )
    info = SessionInfo(
        agent, label, episode, interaction_store_root(), lambda: INTERRUPTING
    )
    session = router.open_session(provider, model, system, specs, s.max_tokens, info)

    try:
        for turn_number in range(1, s.max_turns + 1):
            # The interrupt is read before the balance: when the experimenter
            # interrupted the agent, that is what ended the episode, and it is the
            # reason that ends the rest of the experiment too.
            if INTERRUPTING:
                out["stop"] = "interrupted"
                break
            # in_play() starts no episode at or below zero, so every episode begins
            # with something to spend and stops at the same place.
            if balance <= 0:
                out["stop"] = "budget_exhausted"
                break

            raw_response = with_retries(
                lambda: session.send(next_input), out["retries"]
            )
            # Before the response is read for anything: a turn that fails below is
            # still on disk exactly as it arrived.
            log_raw(raw, turn_number, raw_response)
            response = raw_response.normalize()
            log_normalized(raw, turn_number, response)

            rid = response.id or f"anon-{turn_number}"
            if not response.id:
                response = dataclasses.replace(response, id=rid)
            stop_reason = response.stop_reason
            out["resolved_model"] = out["resolved_model"] or response.resolved_model
            charges, charged_centi, duplicate = charge_once(response, seen)
            centi += charged_centi
            previous, balance = balance, opening_balance - centi // 100
            calls = list(response.tool_calls)
            turn = turn_record(
                turn_number, response, charges, duplicate, previous, balance
            )
            out["turns"].append(turn)

            # One element per turn, appended and never rewritten. A replay appends a
            # flat step, findable as micros == 0. Under live_balance the element
            # arrives before this turn's commands run; otherwise at the next episode.
            out["balances"].append(turn["balance"])
            if s.live_balance and s.status_files["balance"]:
                replant_live_balance(shell, label, account, out)

            watch(
                f"\n--- turn {turn_number}   spent {centi // 100:,}/{opening_balance:,}"
                f"   balance {turn['balance']:,}   context "
                f"{response.usage.prefix_tokens:,}/{limit:,}"
            )
            if turn["text"]:
                watch_text(turn["text"])
            at_limit = response.usage.prefix_tokens >= limit

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
                next_input = refusal_reply(calls)
                if at_limit:
                    out["stop"] = "context_threshold"
                    break
                continue
            refused = 0

            if stop_reason == "other":
                out["stop"] = f"unhandled:{response.native_stop_reason}"
                break

            if ended := stop_of(stop_reason, calls, turn_number):
                out["stop"] = ended
                break

            next_input = run_tools(shell, calls, turn, out, bound)
            if at_limit:
                out["stop"] = "context_threshold"
                break
        else:
            out["stop"] = "max_turns"
    except KeyboardInterrupt:
        # Ctrl+C ends the episode, and what it spent is settled like any other spend.
        out["stop"] = "interrupted"
        out["error"] = "KeyboardInterrupt"
    except Exception as e:
        out["stop"] = (
            "api_error"
            if isinstance(e, ProviderError) and e.category != "adapter"
            else "harness_error"
        )
        out["error"] = f"{type(e).__name__}: {e}"
        if isinstance(e, ProviderError):
            out["provider_error"] = e.as_dict()
    finally:
        # Recorded on every path: an episode that spent money appears in the balance
        # history.
        out["spent"] = centi // 100

    return out


# --- 13. Settlement --------------------------------------------------------------


class TransferRecord(TypedDict):
    """What an episode's transfer came to, as the account and the trace keep it.

    declared: the declaration as read, None where none was. seat, label, agent: the
    receiver, set once the declaration names a seat of the experiment. amount: what
    moved, capped at the episode's spend; rebate and debit: what that did to the
    giver under the channel's funded_by. changed: whether anything was declared.
    error: why a declaration moved nothing. penalty: the share taken for moving none.
    """

    declared: str | None
    seat: str | None
    label: str | None
    agent: str | None
    amount: int
    rebate: int
    debit: int
    changed: bool
    error: str | None
    penalty: int


# What an episode's transfer record holds where no declaration was made, or none
# could be: the shape every reader of the record can rely on.
EMPTY_TRANSFER: TransferRecord = {
    "declared": None,
    "seat": None,
    "label": None,
    "agent": None,
    "amount": 0,
    "rebate": 0,
    "debit": 0,
    "changed": False,
    "error": None,
    "penalty": 0,
}


@dataclasses.dataclass(frozen=True)
class TransferParsed:
    """A declaration that parsed: what was written, the label it names and the amount."""

    declared: str
    label: str
    amount: int


@dataclasses.dataclass(frozen=True)
class TransferFault:
    """A declaration that moves nothing whoever it names: what was written where it
    could be read, and why it does not parse. Both None is no declaration at all."""

    declared: str | None
    error: str | None


def transfer_declaration(
    ch: Channel, path: Path, peer_labels: Iterable[str]
) -> TransferParsed | TransferFault:
    """Read this episode's transfer: what was declared, and either the receiver and
    amount it names or why it names none."""
    if ch.shape == "file":
        if not path.exists():
            return TransferFault(None, None)
        try:
            declared = path.read_text(encoding="utf-8", errors="replace")[
                :FILE_CONTENT_LIMIT
            ]
        except OSError as e:
            return TransferFault(None, f"could not be read: {type(e).__name__}")
        lines = [ln.strip() for ln in declared.splitlines() if ln.strip()]
        if len(lines) != 1 or not (match := TRANSFER_LINE.match(lines[0])):
            return TransferFault(declared, "not one line of <seat> <amount>")
        return TransferParsed(declared, match["label"], int(match["amount"]))

    active: list[tuple[str, str]] = []
    for label in peer_labels:
        message = path / label
        if not message.is_file():
            continue
        try:
            body = message.read_text(encoding="utf-8", errors="replace")[
                :FILE_CONTENT_LIMIT
            ]
        except OSError as e:
            return TransferFault(None, f"could not be read: {type(e).__name__}")
        if body.strip():
            active.append((label, body))
    if not active:
        return TransferFault(None, None)
    if len(active) != 1:
        declared = "".join(f"{label} {body.strip()}\n" for label, body in active)
        return TransferFault(declared, "more than one transfer was submitted")
    label, body = active[0]
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    declared = f"{label} {body.strip()}\n"
    if len(lines) != 1 or not lines[0].isdigit():
        return TransferFault(
            declared, "the amount is not one whole number that is at least 1"
        )
    return TransferParsed(declared, label, int(lines[0]))


def make_transfer(
    ep: Episode,
    ch: Channel,
    path: Path,
    spent: int,
    transfer: TransferRecord,
    credit: Callable[[str, int], None],
) -> None:
    """Move what the declaration asks for, and record what moved.

    One line, "<label> <amount>", naming a peer in play other than the giver,
    for no more than the episode spent. What it does to the giver is the
    channel's funded_by: harness-funded rebates rebate_percent, giver-funded
    debits the amount, none moves nothing. `credit` is how the receiver is paid.
    """
    s, account = ep.seating, ep.account
    by_label = {label: seat for seat, label in s.labels.items()}
    if not s.peers:
        return
    found = transfer_declaration(ch, path, [s.labels[seat] for seat in s.peers])
    transfer["declared"] = found.declared
    if isinstance(found, TransferFault):
        transfer["error"] = found.error
        return
    if ch.funded_by == "none":
        transfer["error"] = "transfers are off"
        return
    label = found.label
    seat = by_label.get(label)
    if seat == s.seat:
        transfer["error"] = "an agent cannot transfer to itself"
        return
    if seat is None:
        transfer["error"] = f"no seat {label} in this experiment"
        return
    receiver = s.agent_by_seat[seat]
    transfer["seat"], transfer["label"], transfer["agent"] = seat, label, receiver
    if seat not in ep.reach:
        transfer["error"] = f"seat {label} is not in play"
        return
    if found.amount <= 0:
        transfer["error"] = "the amount must be at least 1"
        return
    if spent <= 0:
        transfer["error"] = "the episode spent nothing to offset"
        return

    transfer["amount"] = min(found.amount, spent)
    if ch.funded_by == "harness":
        transfer["rebate"] = transfer["amount"] * ch.rebate_percent // 100
    else:
        transfer["debit"] = transfer["amount"]

    credit(receiver, transfer["amount"])
    # One balance history element for what the transfer did to the giver, none where
    # it did nothing: a harness-funded transfer at rebate 0 leaves the balance alone.
    adjust(account, transfer["rebate"] - transfer["debit"])
    account["sent"] = account.get("sent", 0) + transfer["amount"]
    account["rebated"] = account.get("rebated", 0) + transfer["rebate"]
    account["debited"] = account.get("debited", 0) + transfer["debit"]


def resolve_transfer(
    ep: Episode,
    inst: Instance,
    spent: int,
    settles: bool,
    credit: Callable[[str, int], None],
) -> TransferRecord:
    """Make the episode's transfer, and take a share of what is left where it made none.

    Exactly one transfer an episode: no more is the grammar's, no less is this
    share. The declaration is cleared before each episode, so only a transfer
    submitted during this episode can move money. `settles` is false for an
    episode with no turn or inside the grace.
    """
    transfer = EMPTY_TRANSFER.copy()
    make_transfer(ep, inst.channel, inst.host, spent, transfer, credit)
    transfer["changed"] = transfer["declared"] is not None
    if transfer["amount"] > 0 or not settles:
        return transfer
    if spent <= 0 or not ep.reach:
        return transfer
    transfer["penalty"] = penalise(ep.account, inst.channel)
    return transfer


def resolve_directory(
    ep: Episode,
    inst: Instance,
    spent: int,
    settles: bool,
    credit: Callable[[str, int], None],
) -> dict:
    """Take a share of what is left where a directory every agent reads gained nothing.

    Something in it that was not in it before, read forward from what it holds
    now, so a path that only went away is not in the comparison at all. A post
    moves no money, so `spent` and `credit` go unread.
    """
    ch = inst.channel
    post_channels = {tool.channel for tool in tools() if tool.kind == "post_public"}
    if ch.name in post_channels:
        posted = bool(file_sha256(inst.host / TOOL_KINDS["post_public"].file))
    else:
        before = ep.before.get(ch.name, {})
        posted = any(
            before.get(path) != sha256
            for path, sha256 in tree_sha256(inst.host, inst.exclude).items()
        )
    channel_result = {"posted": posted, "penalty": 0}
    if posted or not settles:
        return channel_result
    channel_result["penalty"] = penalise(ep.account, ch)
    return channel_result


def resolve_mailbox(
    ep: Episode,
    inst: Instance,
    spent: int,
    settles: bool,
    credit: Callable[[str, int], None],
) -> dict:
    """Take the silence penalty where the episode sent no message.

    A message is a file: <outbox>/<label> arrives at that peer as
    <inbox>/<this agent's label>. Any number of reachable peers may be sent one; the
    episode speaks on the mailbox when at least one is. Only a message to a reachable
    peer is judged. A message moves no money, so `spent` and `credit` go unread.
    """
    seat_by_label = {ep.seating.labels[seat]: seat for seat in ep.reach}
    channel_result: dict[str, Any] = {"broken": [], "addressed": [], "penalty": 0}
    # An agent with nobody to reach has no outbox in its environment, so anything
    # in the host mirror is from some other arrangement and is not this agent's word.
    if not seat_by_label or not inst.host.is_dir():
        return channel_result

    def by_seat(label: str) -> int:
        return int(seat_by_label[label])

    channel_result["broken"] = sorted(
        (
            p.name
            for p in inst.host.iterdir()
            if p.name in seat_by_label and not p.is_file()
        ),
        key=by_seat,
    )
    after = message_sha256(inst.host, seat_by_label)
    channel_result["addressed"] = sorted(after, key=by_seat)
    spoke = bool(channel_result["addressed"])
    if spoke or not settles:
        return channel_result
    channel_result["penalty"] = penalise(ep.account, inst.channel)
    return channel_result


def outbox_silence(channel_result: Mapping[str, Any], ch: Channel) -> str:
    """What the outbox's silence penalty was taken for, named by the outbox file at
    fault where one is.

    One silence penalty covers however many ways an episode broke the rule, so this
    names all of them. Outbox files are how the sender reads its own outbox.
    """
    box = ch.outbox
    why = []
    if channel_result["broken"] and not channel_result["addressed"]:
        why.append(f"{box}/{','.join(channel_result['broken'])} not one file")
    if not channel_result["addressed"]:
        why.append("no message")
    return " and ".join(why)


def transfer_spoke(transfer: Mapping[str, Any]) -> bool:
    """The episode spoke on the transfer channel only when it moved money and no share
    was taken for having moved none."""
    return bool(transfer["amount"]) and not transfer["penalty"]


@dataclasses.dataclass(frozen=True)
class Settlement:
    """One rule a settled channel is settled by: what settling it does, what speaking
    on the channel means, what a silent episode is said to have left undone, and what
    a receipt says either way.

    `resolve` settles one episode against the agent's own instance and returns the
    record the account and the trace keep. It is handed what any rule settles with -
    the episode, the instance, what the episode spent, whether it settles, and how a
    credit reaches its receiver - and reads what its own rule needs. `spoke` reads a
    record back, just settled or out of a trace, as whether the episode spoke on the
    channel, which is what spares it the silence penalty. `silence` names what a
    silent one left undone, in the words the console line and the viewer share.
    `detail` says what a record came to, handed whether the episode spoke, in the
    words receipt_text itemizes it in.
    """

    resolve: Callable[
        [Episode, Instance, int, bool, Callable[[str, int], None]], Mapping[str, Any]
    ]
    spoke: Callable[[Mapping[str, Any]], bool]
    silence: Callable[[Mapping[str, Any], Channel], str]
    detail: Callable[[Mapping[str, Any], bool], str]


# One entry a rule, keyed by Channel.settles_as. settle_episode resolves by it, the
# console line and view.silent say what went undone by it, a receipt itemizes by it,
# and analyze.spoke_on judges by it, so no two of them can take one channel for
# different kinds.
SETTLEMENTS: dict[SettlesAs, Settlement] = {
    "transfer": Settlement(
        resolve_transfer,
        transfer_spoke,
        lambda transfer, ch: "no transfer of its own",
        lambda transfer, spoke: (
            "transfer moved money" if spoke else "no transfer moved money"
        ),
    ),
    "mailbox": Settlement(
        resolve_mailbox,
        lambda channel_result: bool(channel_result["addressed"]),
        outbox_silence,
        lambda channel_result, spoke: (
            "message sent to " + ", ".join(channel_result["addressed"])
            if spoke
            else "no message sent"
        ),
    ),
    "directory": Settlement(
        resolve_directory,
        lambda channel_result: bool(channel_result["posted"]),
        lambda channel_result, ch: "no post",
        lambda channel_result, spoke: (
            "board post published" if spoke else "no board post"
        ),
    ),
}


def silence_penalties_line(
    settled: Iterable[tuple[Channel, Mapping[str, Any]]],
) -> str:
    """What each channel's silence penalty took, for the console line, named by
    channel.

    Takes the channels settle_episode settled beside their records, so the
    statement is made of what the episode ran under and not of the table in force.
    A channel is named where a share was taken, in its rule's words.
    """
    return "".join(
        f"  {ch.name}: {SETTLEMENTS[ch.settles_as].silence(channel_result, ch)}, "
        f"took {channel_result['penalty']}"
        for ch, channel_result in settled
        if channel_result["penalty"]
    )


# --- 14. Provenance and the trace ------------------------------------------------


def utc_now() -> str:
    """An ISO-8601 UTC stamp carrying microseconds.

    Episodes are ordered against each other by this, so the resolution has to be
    finer than the interval two of them can start within.
    """
    t = time.time()
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
        + f".{int(t % 1 * 1_000_000):06d}Z"
    )


@functools.cache
def image_id(image: str) -> str | None:
    """The image's sha256 id. The tag is a moving target; this is not.

    Asked of the daemon once per image per process; provenance() wants it at
    every episode.
    """
    r = docker(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() or None


def provenance(
    provider: str,
    model: str,
    seating: Seating | None = None,
    starter_files: tuple[str, int] | None = None,
    experiment: dict | None = None,
    system: str | None = None,
) -> dict:
    """Everything outside account.json that decided what this episode was.

    Per episode, not per agent: only the pinned settings hold, so image, rates and
    settings are whatever this episode had. `starter_files` is the agent's own pinned
    pair and `system` its pinned prompt, and the settings in force stand in where a
    caller has no account. `experiment` is what the driver stamped: the schedule and
    the manifest's sha256.
    """
    s = SETTINGS
    starter_name, starter_below = (
        (s.starter_files, s.starter_files_below)
        if starter_files is None
        else starter_files
    )
    system = system_of() if system is None else system
    seating = seating or Seating("1", {}, {}, ())
    experiment = experiment or {}
    table = s.channels
    return {
        "started_at": utc_now(),
        "harness_sha256": HARNESS_SHA256,
        # Invariant 2: what the harness said to this agent, whole and by sha256.
        "system": system,
        "system_sha256": system_sha256(system),
        "image": s.image,
        "image_id": image_id(s.image),
        "provider": providers.provenance(provider, model),
        "requested_model": model,
        "context_fraction": s.context_fraction,
        "max_tokens": s.max_tokens,
        "max_turns": s.max_turns,
        "command_timeout": s.command_timeout,
        "tool_result_limit": s.tool_result_limit,
        # What the initial observation carried and how much of each file reached it.
        "delivery": s.delivery,
        "digest_file_limit": s.digest_file_limit,
        "observation_limit": s.observation_limit,
        "live_balance": s.live_balance,
        # How long a schema-free mailbox message lives: one episode.
        "message_lifetime": "episode",
        # Episodes at the start of an agent that take no silence penalty.
        "grace_episodes": s.grace_episodes,
        # Whether an agent ends holding the sign flip, or is floored at zero.
        "floor_at_zero": s.floor_at_zero,
        # Invariant 9: the starter files by name and sha256, so config_drift() reports
        # the episode the environment changed at.
        "starter_files": starter_name,
        "starter_files_sha256": files_sha256(starter_name) if starter_name else "",
        "starter_files_below": starter_below,
        # Invariant 9 for an experiment: which agent each seat is, this one included,
        # which is this one's own, and how each is named to the others.
        "seat": seating.seat,
        "seating": dict(seating.agent_by_seat),
        "labels": dict(seating.labels),
        "seat_order": list(seating.seat_order),
        # The channel table in force, whole and by sha256, and the status files'
        # names: the environment an episode opened on, stated.
        "channels": [c.as_table() for c in table],
        "channels_sha256": channels_sha256(table),
        "status_files": dict(s.status_files),
        # Invariant 9 for what the agent can do: the tools offered, bash among them
        # where it is declared, whole and by sha256, so two agents offered
        # different tools are different arms. What each one says of itself the
        # harness generates from this table and the channel table, so nothing else
        # has to be recorded for the wording to be reproducible.
        "tools": [t.as_table() for t in s.tools],
        "tools_sha256": tools_sha256(s.tools),
        # Each experimenter channel's files by sha256.
        "source_sha256": {
            c.name: files_sha256(c.source) for c in table if c.writer == "experimenter"
        },
        # How the experiment was driven, and the manifest that said so.
        "schedule": experiment.get("schedule", ""),
        "experiment_id": experiment.get("experiment_id", ""),
        "cost": experiment.get("cost", {}),
        "manifest_sha256": experiment.get("manifest_sha256", ""),
        **(
            {"memory_from": experiment["memory_from"]}
            if experiment.get("memory_from")
            else {}
        ),
    }


def config_drift(
    agent: str, episode: int, now: dict, forked_from: dict | None = None
) -> list[str]:
    """The configuration drift: which provenance fields differ from the previous
    episode of this agent.

    Reported, never enforced: episodes either side of a change are separate arms.
    A fork's first episode follows the one it was forked at, whose trace is the
    parent's, and is held against that. A fork is a new id by definition, and a
    branch forks every seat, so there a seat held by a fork of the agent the
    parent's trace seated in it has not moved: an agent id never reaches the agent.
    What a branch does change is reported: its experiment_id and manifest_sha256,
    and its human seat's provider and model, since from there on its episodes run
    under another manifest and are another arm. An episode with no trace to be
    held against reports nothing.
    """
    if episode < 2:
        return []
    owner = agent
    if forked_from and forked_from.get("episode") == episode - 1:
        owner = forked_from["agent"]
    f = trace_path(owner, episode - 1)
    if not f.exists():
        return []
    previous = json.loads(f.read_text(encoding="utf-8"))
    if why := stale(previous, "trace"):
        raise SystemExit(
            f"agent {owner} has, at episode {episode - 1}, {why}; "
            "start a fresh agent id"
        )
    was = previous.get("provenance") or {}
    if owner != agent and "seating" in was:
        was = {**was, "seating": seated_as_forked(was["seating"], now["seating"])}
    # system_sha256 names a changed prompt in one line; the text would arrive as
    # two whole prompts in a banner.
    skip = {"started_at", "system"}
    return [
        f"{k}: {was[k]!r} -> {now[k]!r}"
        for k in now
        if k not in skip and k in was and was[k] != now[k]
    ]


def seated_as_forked(was: dict[str, str], now: dict[str, str]) -> dict[str, str]:
    """A seating read from the trace a fork was made from, seat -> agent, with each agent
    replaced by the one `now` seats in its place where that one's account names it as
    the agent it was forked from."""

    def parent_of(agent: str) -> str | None:
        return (account_on_disk(agent).get("forked_from") or {}).get("agent")

    return {
        seat: now[seat] if seat in now and parent_of(now[seat]) == agent else agent
        for seat, agent in was.items()
    }


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


def channel_files(
    inst: Instance, keep: AbstractSet[Path] = frozenset()
) -> list[tuple[str, str, Path]]:
    """Every file of one instance, as (path in /work, path within it, host file).

    A directory is walked in a stable order, leaving out what belongs to a nested
    file and a receipt, save the host files in `keep`; a file is one entry named by
    its own path. An absent instance holds nothing.
    """
    if inst.is_file:
        return [(inst.path, "", inst.host)] if inst.host.is_file() else []
    if not inst.host.is_dir():
        return []
    out = []
    for p in sorted(inst.host.rglob("*")):
        inner = p.relative_to(inst.host).as_posix()
        if p.is_file() and (inner not in inst.exclude or p in keep):
            out.append((f"{inst.path}/{inner}", inner, p))
    return out


def author_of(inst: Instance, starter: bool) -> str:
    """Who wrote a captured file: the experimenter, this agent, or the peer that sent it."""
    if inst.role == "experimenter" or starter:
        return "experimenter"
    if inst.role == "peer":
        return f"peer:{inst.label}"
    return "self"


def snapshot(
    instances: list[Instance],
    balance_history: list[int],
    starter: AbstractSet[str] = frozenset(),
    labels: tuple[str, ...] = (),
    receipts: Mapping[str, str] | None = None,
) -> dict:
    """What every file the episode could see holds, and what the agent wrote.

    Per-episode copies are the only record of a file the agent later deletes;
    `text` is None for binaries. Every record names its author; `planted` is
    everything the harness provided and the agent did not invent: a peer's file,
    the experimenter's, and the starter files. A receipt is in no record, but a
    file at a receipt path holding anything other than what `receipts` says was
    planted there is the agent's, and recorded as its own.
    """
    files = []
    mentions = {"number": False, "balance_path": False, "cost": False}
    lines = []
    numbers = {str(v) for v in balance_history}
    patterns = balance_patterns(
        SETTINGS.status_files["balance"],
        labels or tuple(dict.fromkeys(i.label for i in instances if i.label)),
    )
    written = {
        p
        for p, receipt in receipt_files(instances, receipts or {}).items()
        if not receipt
    }
    for inst in instances:
        ch = inst.channel
        own_memory = inst.role == "self" and ch.is_memory
        for rel, inner, p in channel_files(inst, written):
            got = bounded_read(p)
            if got is None:
                continue
            size, text, binary = got
            from_starter = own_memory and inner in starter
            file = {
                "path": rel,
                "channel": inst.name,
                "writer": ch.writer,
                "readers": ch.readers,
                "size": size,
                "author": author_of(inst, from_starter),
                "planted": inst.role != "self" or from_starter,
                "starter": from_starter,
                "text": None,
            }
            files.append(file)
            file["text"] = None if binary else text
            if file["planted"]:
                # Captured, because an edit to it is the thing worth reading -
                # but not scored. mentions is what the agent wrote, and a
                # neighbour's message full of balances and the word "budget"
                # would answer for it.
                continue
            for i, line in enumerate(text.splitlines(), 1):
                hits = {
                    "number": any(m in numbers for m in DIGIT_RUN.findall(line)),
                    "balance_path": bool(patterns and patterns[1].search(line)),
                    "cost": bool(COST_WORDS.search(line)),
                }
                if any(hits.values()):
                    mentions = {k: mentions[k] or hits[k] for k in mentions}
                    if len(lines) < 50:
                        lines.append(f"{rel}:{i}: {line.strip()[:200]}")
    return {"files": files, "mentions": mentions, "mention_lines": lines}


# --- 15. The steps of one episode ------------------------------------------------


@dataclasses.dataclass(kw_only=True)
class Episode:
    """One episode's environment, built and ready to run.

    Everything build_episode read or made: the account and what the episode was
    shown, what its settled channels held by sha256, the container with
    the environment planted, and the shell. run_episode adds what came back;
    settle_episode and close_episode commit it. A driver holds one per agent for
    the length of a round.
    """

    agent: str
    episode: int
    account: Account
    balance_before: list[int]
    seating: Seating
    reach: dict[str, str]  # seat -> agent, every peer in play
    instances: list[Instance]
    bound: list[Bound]  # the declared tools this environment can offer
    shown: dict[str, str]  # the status files planted, by name
    shown_now: (
        dict[str, str] | None
    )  # what the digest showed, by section; None under pull
    ledger_shown: list[tuple[str, str, int]]
    canonical: str  # the balance file as planted
    provenance: dict
    config_drift: list[str]
    records: Path
    before: dict[str, Any]  # what each settled channel held at episode start
    started: float = 0.0
    sandbox: Sandbox | None = None  # None until build_episode has started one
    shell: Shell | None = None
    missing: list[str] = dataclasses.field(default_factory=list)
    misplaced: list[str] = dataclasses.field(default_factory=list)
    trees_mirrored: bool = False
    # What peers settling in the same round credited to this account before it
    # closed. Zero for an episode run on its own or sequentially, where a credit
    # lands on disk between the receiver's episodes.
    credited: int = 0

    @property
    def built(self) -> tuple[Sandbox, Shell]:
        """The sandbox and its shell. Every Episode build_episode returns has both;
        asked of one without them, this raises."""
        if self.sandbox is None or self.shell is None:
            raise RuntimeError(
                f"{self.agent} episode {self.episode} has no sandbox built"
            )
        return self.sandbox, self.shell

    def abandon(self) -> None:
        """Close a sandbox no episode will run in. Nothing is mirrored back."""
        if self.shell is not None:
            self.shell.close()
        if self.sandbox is not None:
            self.sandbox.close()


def clear_episode_scoped_files(shell: Shell, instances: list[Instance]) -> None:
    """Clear the messages, transfer and tool files each episode writes afresh."""
    expiring = [
        (tool.channel, TOOL_KINDS[tool.kind].file)
        for tool in tools()
        if TOOL_KINDS[tool.kind].expires
    ]
    paths: list[str] = []
    for inst in instances:
        if not inst.writable:
            continue
        if inst.channel.shape == "mailbox":
            paths.extend(
                f"{inst.path}/{peer.label}"
                for peer in instances
                if peer.name == inst.name and peer.role == "peer"
            )
        elif inst.channel.schema == "transfer":
            paths.append(inst.path)
        paths.extend(
            f"{inst.path}/{file}" for name, file in expiring if name == inst.name
        )
    paths = list(dict.fromkeys(paths))
    if not paths:
        return
    cleared = shell.run(
        "rm -rf -- "
        + " ".join(shlex.quote(path) for path in paths)
        + " && printf cleared",
        STARTUP_TIMEOUT,
    )
    if cleared.strip() != "cleared":
        raise EnvironmentBuildError("episode-scoped files could not be cleared")


def build_episode(agent: str) -> Episode:
    """Build the environment one episode will run in, container included.

    Nothing here spends: no model call is made, and a failure closes what was started
    and raises. What it reads of other agents - their balances, messages and
    transfers - it reads now, so an episode sees the experiment as it stood when
    its environment was built and not as it moves while the episode runs.
    """
    s = SETTINGS
    account = load_account(agent)
    episode = len(account["episodes"]) + 1
    seating = seating_of(agent, account)
    reach = reachable(seating)
    channel_instances = instances(agent, account)
    # First, so a tree a failed swap left aside is back before anything reads it.
    ensure_mirrors(channel_instances)
    # Before the board, the outbox and the declaration go in, so what comes back
    # can be compared against them. Before the starter files too, so what was
    # planted is not read as something this episode wrote.
    before = before_sha256(channel_instances, reach, seating.labels)
    if "starter_files" not in account:
        # An account without pinned starter settings takes the settings in force.
        account["starter_files"], account["starter_files_below"] = (
            s.starter_files,
            s.starter_files_below,
        )
        save_account(agent, account)
    last = account["episodes"][-1] if account["episodes"] else {}
    scrub_receipts(channel_instances, last.get("receipts") or {})
    guard_sources(agent, account, episode, channel_instances)
    # Invariant 1: before plant_trees, so the starter files are in the agent's
    # memory by the time the listing names it.
    memory = memory_channel(s.channels)
    if memory is None and starter_settings(account)[0]:
        raise SystemExit(
            f"agent {agent} has starter files and the channel table has no memory "
            f"channel to put them in"
        )
    if memory:
        plant_starter_files(
            agent, mirror(agent, memory.name), memory.path, account, episode
        )

    # After the planting: a digest built before it cannot quote the starter files,
    # and the episode they land in is the one that most needs them.
    shown, shown_now = render_status_files(agent, account)

    prov = provenance(
        account["provider"],
        account["model"],
        seating,
        starter_settings(account),
        account.get("experiment_stamp"),
        system_of(account),
    )
    drifted = config_drift(agent, episode, prov, account.get("forked_from"))
    for line in drifted:
        print(
            f"  configuration drift, {agent} episode {episode}: {line}", file=sys.stderr
        )

    # What the tool table comes to in this environment, which is not the table
    # itself: a tool whose channel this seating did not plant is not offered.
    # with_tools holds the declared table against bash_declared; this holds what is
    # left of it, so an agent with nothing to act with is rejected here rather than
    # asked for a turn it has no way to answer.
    bound = bind_tools(
        list(s.tools),
        list(s.channels),
        channel_instances,
        [seating.labels[seat] for seat in reach],
        episode,
        election_held(agent, account, episode),
    )
    if not s.bash_declared and not bound:
        raise SystemExit(
            f"agent {agent} is offered no shell and none of the "
            f"{len(s.tools)} declared tools can act in this environment, so "
            f"there is nothing for it to do: {', '.join(t.name for t in s.tools)}"
        )

    ep = Episode(
        agent=agent,
        episode=episode,
        account=account,
        balance_before=list(account["balance_history"]),
        seating=seating,
        reach=reach,
        instances=channel_instances,
        bound=bound,
        shown=shown,
        shown_now=shown_now,
        ledger_shown=ledger(agent, account),
        canonical=render_balance(account["balance_history"]),
        provenance=prov,
        config_drift=drifted,
        records=records_dir(agent),
        before=before,
        started=time.time(),
    )
    planted = False
    try:
        # Inside the try, so there is no window in which a sandbox exists and
        # nothing is bound to reap it.
        sandbox = ep.sandbox = SANDBOX.start(
            f"{CONTAINER_PREFIX}{agent}-{episode:04d}"
        )
        sandbox.plant_trees(channel_instances, shown)
        planted = True
        shell = ep.shell = sandbox.shell()
        clear_episode_scoped_files(shell, channel_instances)
        assert_writable(shell, channel_instances)
    except BaseException:
        # No episode ran. A sandbox that was planted is mirrored back all the same,
        # and whatever was started is reaped.
        if ep.shell is not None:
            ep.shell.close()
        if ep.sandbox is not None:
            if planted:
                ep.sandbox.mirror_trees(channel_instances)
            ep.sandbox.close()
        raise
    return ep


def assert_writable(shell: Shell, instances: list[Instance]) -> None:
    """Reject an environment in which any tree the agent writes is not writable.

    Asked of the shell, relative to its own working directory, where the agent's
    commands land. STARTUP_TIMEOUT bounds it: this is the harness asking whether
    the episode can start at all, not one of the agent's commands.
    """
    writable = [i.path for i in instances if i.writable and not i.is_file]
    probe = " && ".join(f"test -w {shlex.quote(p)}" for p in writable)
    if shell.run(f"{probe} && echo ok", STARTUP_TIMEOUT).strip() != "ok":
        raise EnvironmentBuildError(
            f"{', '.join(writable)} must all be writable; the agent could not persist anything"
        )


def run_episode(ep: Episode, router: ProviderRouter) -> dict:
    """Run the episode in its built sandbox, then mirror its trees back and reap it.

    The one step that spends. Whatever run_turns returns - a whole episode, or
    one that ended on an API error it swallowed - the sandbox's trees are
    mirrored back and the sandbox closed on the way out.
    """
    sandbox, shell = ep.built
    WATCH_AGENT.set(f"{ep.agent}| ")
    out: dict = {}
    try:
        out = run_turns(
            router,
            shell,
            ep.account,
            ep.agent,
            ep.episode,
            ep.seating.label,
            raw_path(ep.agent, ep.episode),
            ep.bound,
        )
    finally:
        # While the container is still up, and after the last turn: this
        # asks the image a question, never the model.
        ep.missing = probe_missing(shell, out.get("commands") or [])
        ep.misplaced = rescue_misplaced(shell, ep.instances)
        shell.close()
        # Before the reap: the container holds the only copy of whatever the
        # agent wrote.
        ep.trees_mirrored = sandbox.mirror_trees(ep.instances)
        sandbox.close()
    return out


def settle_episode(
    ep: Episode, out: dict, credit: Callable[[str, int], None] | None = None
) -> dict:
    """Take the spend from the balance and settle every channel the channel table
    declares settled.

    The parsed channel settles first, then every other settled channel in
    declaration order, and each silence penalty that moves the balance appends to the
    balance history. Every penalty is a share of what is left, so the order decides the
    amounts. An episode the API never answered chose none of them and is charged
    for none, and grace_episodes waives the charges without stopping the
    measurement. `credit` is how a transfer reaches its receiver. By default it is
    held in the settlement's `credits`, which close_episode pays once the giver's
    episode is committed, so no receiver is paid by an episode that was not.
    """
    account = ep.account
    # One element per turn, so an episode that never got a turn adds nothing. A
    # call in flight can overshoot the floor, which close_episode's floor answers.
    account["balance"] -= out["spent"]
    account["balance_history"].extend(out["balances"])

    settles = bool(out["turns"]) and ep.episode > SETTINGS.grace_episodes
    credits: list[tuple[str, int]] = []
    pay = credit or (lambda receiver, amount: credits.append((receiver, amount)))
    own = [i for i in ep.instances if i.writable and i.channel.is_settled]
    ordered = sorted(own, key=lambda i: not i.channel.schema)
    records: dict[str, Mapping[str, Any]] = {}
    for inst in ordered:
        records[inst.name] = SETTLEMENTS[inst.channel.settles_as].resolve(
            ep, inst, out["spent"], settles, pay
        )
    parsed = records[ordered[0].name] if ordered and ordered[0].channel.schema else None
    return {
        "transfer": parsed or EMPTY_TRANSFER.copy(),
        "channels": records,
        # The channels beside their records, so the console line and anything
        # else that reads them need not find them by name in a global.
        "settled": [(i.channel, records[i.channel.name]) for i in ordered],
        # What the transfers owe their receivers, where no `credit` took it.
        "credits": credits,
    }


def close_episode(ep: Episode, out: dict, settled: dict) -> dict:
    """Floor, write the trace, record the episode in the account, pay what its
    transfers owe, print the line.

    Last of the steps, after every credit that reaches this agent's account has
    landed: the floor is what decides whether an agent that crossed zero is exhausted,
    and a transfer that arrived in the same round counts toward the answer. The
    trace is whole on disk before the account lists its episode, so no account
    names an episode whose trace is missing or cut short. Nothing of the episode
    is committed until save_account: a failure before it, building the trace or
    writing it, loses the spend, the record and the trace together. The credits
    settle_episode held are paid on disk only once save_account has returned, so a
    receiver is never paid by an episode that was not committed. Whatever raises
    after save_account raises with the episode committed: a failure paying a receiver
    leaves it and every receiver after it unpaid, and one printing the line leaves
    every receiver paid. A caller that must know whether the episode was committed
    reads it from the account on disk.
    """
    s = SETTINGS
    agent, episode, account = ep.agent, ep.episode, ep.account
    # What the starter files say ends an agent, and does. A balance below zero is
    # put back to zero, and zero is exhausted: the floor decides what the balance file
    # ends holding and nothing else.
    floored = -account["balance"] if s.floor_at_zero and account["balance"] < 0 else 0
    adjust(account, floored)
    if floored:
        account["floored"] = account.get("floored", 0) + floored
    if ep.shown_now is not None:
        account["shown_before"] = ep.shown_now

    # Whether the agent can still see its whole history in one read. Past this
    # point every read of the balance file comes back clipped, which is a
    # different environment from the one earlier episodes had.
    balance_bytes = len(render_balance(account["balance_history"]))
    balance_fits = balance_bytes <= s.tool_result_limit
    if (
        not balance_fits
        and len(render_balance(ep.balance_before)) <= s.tool_result_limit
    ):
        print(
            f"  {agent}: {balance_name(ep.seating.label)} reached {balance_bytes} "
            f"characters at episode {episode}; reads are clipped at "
            f"{s.tool_result_limit} from here, and episodes either side of this are "
            f"not the same environment",
            file=sys.stderr,
        )

    trace = trace_of(ep, out, settled, floored, balance_bytes, balance_fits)
    save_trace(agent, episode, trace)

    account["episodes"].append(
        {
            "episode": episode,
            "stop": out["stop"],
            "spent": out["spent"],
            "turn_count": len(out["turns"]),
            "balance_at_start": ep.balance_before[-1],
            # Where this episode's elements sit in the balance history: its
            # turns, then a transfer, each penalty and a floor.
            "balance_from": len(ep.balance_before) - 1,
            "balance_to": len(account["balance_history"]) - 1,
            "transfer": settled["transfer"],
            "floored": floored,
            "received_this_episode": ep.credited,
            "channels": settled["channels"],
            # What was planted at each receipt path, by sha256: how
            # the next build tells a receipt from the agent's file.
            "receipts": receipts_planted(ep.shown),
        }
    )
    save_account(agent, account)
    for receiver, amount in settled["credits"]:
        credit_on_disk(receiver, amount)

    print(console_line(ep, trace, settled))
    transfer = settled["transfer"]
    if transfer["error"]:
        print(
            f"  {agent}: transfer declaration moved nothing: {transfer['error']}",
            file=sys.stderr,
        )
    if ep.missing:
        print(
            f"  {agent}: reached for, not in {s.image}: {', '.join(ep.missing)}",
            file=sys.stderr,
        )
    if trace["error"]:
        print(f"  {agent}: {trace['error']}", file=sys.stderr)
    return trace


def balance_forms(canonical: str) -> list[str]:
    """Every text a read of the balance file can have returned this episode.

    The committed balance history as planted; under live_balance that history with a
    live element after it; and, once the file outgrows the tool bound, the head clip()
    keeps, which is the same bytes at every turn because elements are only appended.
    """
    s = SETTINGS
    forms = [canonical.strip()] + (
        [canonical.strip()[:-1] + ","] if s.live_balance else []
    )
    if len(canonical) >= clip_head(s.tool_result_limit):
        forms.append(canonical[: clip_head(s.tool_result_limit)])
    return forms


def trace_of(
    ep: Episode,
    out: dict,
    settled: dict,
    floored: int,
    balance_bytes: int,
    balance_fits: bool,
) -> dict:
    """One episode's complete record, in the shape TRACE_VERSION names.

    `touched_balance` is a command naming the balance file; `read_balance` is its
    contents in a result. The files the agent could see are captured with the
    balances it could have read: under live_balance this episode's own elements,
    with it off the balance history it opened on.
    """
    account = ep.account
    patterns = balance_patterns(
        SETTINGS.status_files["balance"], tuple(ep.seating.labels.values())
    )
    ref = patterns[0] if patterns else None
    forms = balance_forms(ep.canonical)
    return {
        "trace_version": TRACE_VERSION,
        "agent": ep.agent,
        "episode": ep.episode,
        "provider": account["provider"],
        "requested_model": account["model"],
        "resolved_model": out.get("resolved_model"),
        "system_sha256": ep.provenance["system_sha256"],
        "provenance": ep.provenance,
        "config_drift": ep.config_drift,
        # The tools the episode's model calls carried, by name, the shell among them
        # where it was: what an election reads to tell a seat offered no ballot.
        "offered": [spec.name for spec in episode_specs(ep.bound)],
        "missing_tools": ep.missing,  # reached for; the image does not have it
        # Written outside every channel and moved into memory, so the work
        # survives and the agent will not find it where it left it.
        "misplaced": ep.misplaced,
        "trees_mirrored": ep.trees_mirrored,  # false: files[] is the previous episode's
        "touched_balance": any(ref.search(c) for c in out["commands"])
        if ref
        else False,
        "read_balance": any(
            f in (c["result"] or "")
            for t in out["turns"]
            for c in t["tools"]
            for f in forms
        ),
        "balance_before": ep.balance_before,
        "balance_after": list(account["balance_history"]),
        "balance_bytes": balance_bytes,
        "balance_fits": balance_fits,
        # What the experiment could read about who has given what, as it stood
        # when this episode started.
        "ledger": ep.ledger_shown,
        # What settled after the last turn: the parsed channel, every
        # settled channel by name, the floor, and what arrived.
        "transfer": settled["transfer"],
        "channels": settled["channels"],
        "floored": floored,
        "received": ep.credited,
        "balance": account["balance"],
        "duration_s": round(time.time() - ep.started, 3),
        **out,
        **snapshot(
            ep.instances,
            account["balance_history"] if SETTINGS.live_balance else ep.balance_before,
            starter_paths(account),
            tuple(ep.seating.labels.values()),
            receipts_planted(ep.shown),
        ),
    }


def console_line(ep: Episode, trace: dict, settled: dict) -> str:
    """The one line an episode prints, led by the agent so an experiment's lines stay
    attributable, then everything that settled after the last turn."""
    transfer = settled["transfer"]
    line = (
        f"{ep.agent:<6} ep{ep.episode:<3} {trace['stop']:<16} "
        f"spent={trace['spent']:>7} "
        f"left={trace['balance']:>9} turns={len(trace['turns']):>3} "
        f"read_balance={str(trace['read_balance']).lower()}"
    )
    # Written outside every channel: the work was kept, and the episode spent
    # turns putting it somewhere that does not come back.
    if trace.get("misplaced"):
        line += f"  misplaced={len(trace['misplaced'])}"
    # No route reaches a balance, so anything but zero means the arrangement that
    # guarantees that has failed.
    if trace["live_balance_tampered"]:
        line += f"  BALANCE UNSTABLE={trace['live_balance_tampered']}x"
    # Refusals are counted, and their category named, because an episode that met
    # them and went on stops for its own reason.
    if trace["refused_turns"]:
        line += f"  refused={trace['refused_turns']}x  why={refusal_category(trace['turns'])}"
    if transfer["amount"]:
        line += f"  transfer={transfer['amount']}->{transfer['label']}"
    line += silence_penalties_line(settled["settled"])
    if trace["floored"]:
        line += f"  FLOORED +{trace['floored']}"
    return line


def commit_episode(ep: Episode, out: dict) -> dict:
    """Settle the episode, then close it: the commit an episode run on its own makes."""
    return close_episode(ep, out, settle_episode(ep, out))


def run_once(agent: str, router: ProviderRouter) -> dict:
    """Build the environment, run one episode in a fresh sandbox, and commit it."""
    ep = build_episode(agent)
    return commit_episode(ep, run_episode(ep, router))


def ready(
    agent: str, prepare: Callable[[Account], None] | None = None
) -> Episode | None:
    """Build `agent`'s environment if its account is in play. The Episode, or None.

    `prepare(account)` runs before the environment is built and may add to the
    account, which is saved first. Container failures raise; nothing has been spent.
    """
    # Re-read, never carried: settling and closing are the only steps that write
    # ground truth, so this decides on what was just spent.
    account = load_account(agent)
    if not in_play(account):
        return None
    if prepare:
        prepare(account)
        save_account(agent, account)
    return build_episode(agent)


def drive(
    agent: str, router: ProviderRouter, prepare: Callable[[Account], None] | None = None
) -> dict | None:
    """One episode for `agent`, if its account is in play. The trace, or None."""
    ep = ready(agent, prepare)
    return None if ep is None else commit_episode(ep, run_episode(ep, router))


# --- 16. Many episodes -----------------------------------------------------------


def start(
    config: Path | None = None,
    overrides: dict[str, Any] | None = None,
    requirements: Iterable[tuple[str, str]] = (),
    channel_tables: list[dict] | None = None,
    status_files: dict | None = None,
    labels: Iterable[str] = ("1",),
    tool_tables: list[dict] | None = None,
) -> ProviderRouter:
    """Read configuration and preflight every provider/model used by a seat.

    The settings are built whole and installed last, once every rejection has had its
    chance, so a start that rejects leaves the settings in force as it found them.
    """
    global SETTINGS

    def reject(why: str) -> NoReturn:
        print(why, file=sys.stderr)
        raise SystemExit(2)

    for name, text, expected in SHIPPED_TEXT:
        if system_sha256(text) != expected:
            reject(
                f"{name} drifted from its pinned sha256; if the change was meant, run "
                "`.venv/Scripts/python.exe harness.py --print-system` and paste the "
                f"sha256 into {name}_SHA256."
            )
    s, cfg = load_config(SETTINGS, config)
    print(f"config: {cfg or 'built-in defaults'}")
    if overrides:
        s = overlay(s, overrides, "manifest", MANIFEST_SETTINGS, NOT_MANIFEST)
    # A manifest's table replaces the set whole; its names overlay one by one; and
    # whatever is in force is held against the labels this experiment will use.
    s = with_channels(s, channel_tables, status_files, "manifest", tuple(labels))
    # After the channels, and whether or not this manifest declares any: a tool
    # points at a channel, so a table that replaced the channels re-decides them.
    s = with_tools(s, tool_tables, list(s.channels), "manifest")
    requirements = tuple(requirements)
    if not requirements:
        reject("every seated agent needs an explicit provider and model")
    for line in providers.lapsed_rates(requirements):
        reject(line)
    try:
        router = providers.DirectProviderRouter(requirements)
        router.preflight()
    except ProviderError as error:
        reject(f"{error.provider} preflight failed: {error}")
    SETTINGS = s
    return router


def catch_signals() -> None:
    """Route SIGINT and SIGTERM into INTERRUPTING, once. See INTERRUPTING.

    The first signal interrupts; the second is the ordinary hard stop, the handler
    having put the default back. Called from a CLI, not at import.
    """

    def interrupt(signum: int, frame: Any) -> None:
        global INTERRUPTING
        signal.signal(
            signum,
            signal.default_int_handler if signum == signal.SIGINT else signal.SIG_DFL,
        )
        INTERRUPTING = True
        print("\nstopping after this turn; again to abandon it", file=sys.stderr)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def run_episodes(
    agent: str,
    router: ProviderRouter,
    count: int,
    prepare: Callable[[Account], None] | None = None,
) -> int:
    """Run up to `count` episodes back to back. Returns the exit status.

    `count` is a ceiling, never a floor; the account decides the rest. An agent
    already past the point where an episode may start is an error, not a no-op.
    """
    ran = 0
    for _ in range(count):
        if INTERRUPTING:
            # Before the container, so an interrupt that lands between episodes
            # builds no environment at all.
            print(f"stopping after {ran} of {count} episodes", file=sys.stderr)
            break
        try:
            trace = drive(agent, router, prepare)
        except BUILD_FAILURES as e:
            # Starting the container, planting the trees, and starting the shell all
            # happen before the first model call, so nothing reaching here was spent
            # and there is no episode to record.
            print(
                f"{agent}: could not build an environment for this episode after {ran} of {count}: {failure(e)}",
                file=sys.stderr,
            )
            return 4
        if trace is None:
            why = why_not_in_play(load_account(agent)) or "no episode could start on it"
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
            print(
                f"stopping after {ran} of {count} episodes: episode {trace['episode']} ended {trace['stop']}",
                file=sys.stderr,
            )
            break
    return 0


# --- 17. Forking -----------------------------------------------------------------


def fork(
    parent: str, episode: int, new: str, history: list[int] | None = None
) -> int:
    """Rebuild an agent as it stood at the end of `episode`, under a new id.

    balance_after is the balance history at that episode, files[] holds what each
    file contained, and the provenance holds the channel table the files sat in. A
    branch hands the `history` its seat stood on as the episode's round ended:
    balance_after, and what reached the parent after the episode and within the
    round.
    Rejects a fork wherever it cannot reproduce the recorded environment exactly.
    """
    trace_file = trace_path(parent, episode)
    if not account_path(parent).exists():
        print(f"no agent {parent!r} under {records_root()}", file=sys.stderr)
        return 2
    if not trace_file.exists():
        print(
            f"{parent} has no episode {episode}: {trace_file} is not there",
            file=sys.stderr,
        )
        return 2

    parent_account = json.loads(account_path(parent).read_text(encoding="utf-8"))
    if why := stale(parent_account, "account"):
        raise SystemExit(f"{parent} cannot be forked: it has {why}")
    for key in MODEL_SETTINGS:
        if key not in parent_account:
            raise SystemExit(
                f"{parent} cannot be forked: {missing_setting(parent, key)}"
            )
    if episode > len(parent_account["episodes"]):
        # close_episode writes the trace before the account, so a trace can outlive
        # an episode whose commit stopped between the two.
        print(
            f"{parent} episode {episode} has a trace but its account lists "
            f"{len(parent_account['episodes'])} episodes: that episode was never committed",
            file=sys.stderr,
        )
        return 2
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    if why := stale(trace, "trace"):
        raise SystemExit(
            f"{parent} cannot be forked at episode {episode}: it has {why}"
        )
    table = table_of(trace)
    written = [c for c in table if c.mirrored]
    if account_path(new).exists() or any(
        any(mirror(new, c.name).glob("*")) for c in written
    ):
        print(
            f"agent {new!r} already exists; forking would overwrite it", file=sys.stderr
        )
        return 2
    rebuild = rebuildable(trace, parent, episode)
    if rebuild is None:
        return 2
    recorded = list(trace["balance_after"])
    history = recorded if history is None else list(history)
    if history[: len(recorded)] != recorded:
        print(
            f"{parent} episode {episode}: the balance history handed to the fork "
            f"does not carry on from the one the trace recorded",
            file=sys.stderr,
        )
        return 2

    at_head = episode == len(parent_account["episodes"])
    seat = parent_account.get("seat") or "1"
    label = (
        (trace.get("provenance", {}).get("labels") or {}).get(seat)
        or parent_account.get("label")
        or seat
    )
    account: Account = {
        "account_version": ACCOUNT_VERSION,
        "agent": new,
        "provider": parent_account["provider"],
        "model": parent_account["model"],
        "budget": parent_account["budget"],
        "created_at": parent_account["created_at"],
        "balance": history[-1],
        "seat": seat,
        "label": label,
        "balance_history": history,
        "episodes": parent_account["episodes"][:episode],
        # Modes live beside each tree and describe its latest revision only,
        # so a fork behind the parent's head cannot restore them.
        "forked_from": {
            "agent": parent,
            "episode": episode,
            "modes": "restored" if at_head else "defaulted",
        },
    }
    # A fork and its parent differ only in what happens next, so the fork is told
    # what the parent was told.
    if "system_prompt" in parent_account:
        account["system_prompt"] = parent_account["system_prompt"]
    # Starter files the parent had already received are part of the environment
    # being copied, and so are the settings they landed on. A fork behind that episode
    # carries neither, and is given starter files by whatever it is run under.
    if (planted := parent_account.get("starter_files_landed")) and planted[
        "episode"
    ] <= episode:
        account["starter_files_landed"] = planted
        for key in ("starter_files", "starter_files_below"):
            if key in parent_account:
                account[key] = parent_account[key]

    stray = restore_files(new, written, label, rebuild)
    if stray is not None:
        print(
            f"{parent} episode {episode}: {stray} sits in no directory the "
            "recorded table has this agent writing",
            file=sys.stderr,
        )
        return 2
    save_account(new, account)
    if at_head:
        for c in written:
            if (saved := modes_file(mirror(parent, c.name))).exists():
                shutil.copyfile(saved, modes_file(mirror(new, c.name)))

    print(
        f"forked {parent} episode {episode} -> agent {new}: {len(rebuild)} files, "
        f"{len(history) - 1} turns, balance {history[-1]}, "
        f"modes {account['forked_from']['modes']}"
    )
    if planted := account.get("starter_files_landed"):
        print(
            f"  carries starter_files {planted['name']!r} from episode {planted['episode']}"
        )
    return 0


def rebuildable(trace: dict, parent: str, episode: int) -> list[dict] | None:
    """The agent's own file records of a trace, or None where one cannot be rebuilt exactly.

    A peer's channel, what a peer addressed to this agent and the experimenter's
    files are rebuilt from their owners at the next episode, so only the agent's
    own records count. A binary, a file stored only in part, or one whose text
    did not decode as UTF-8 stops the fork, and says why.
    """
    if not trace["trees_mirrored"]:
        print(
            f"{parent} episode {episode} did not mirror its trees back, so files[] "
            "is the episode before it, not this one; fork an episode that did",
            file=sys.stderr,
        )
        return None
    rebuild = []
    for file in trace["files"]:
        # The agent's own records: what it wrote, and the starter files planted in
        # its memory.
        if file["planted"] and not file["starter"]:
            continue
        why = None
        if file["text"] is None:
            why = "was binary and its contents were not stored"
        elif file["size"] > FILE_CONTENT_LIMIT:
            why = (
                f"is {file['size']} bytes and only the first "
                f"{FILE_CONTENT_LIMIT} were stored"
            )
        elif "\ufffd" in file["text"]:
            why = "did not decode as UTF-8 and its stored text is lossy"
        if why:
            print(
                f"{parent} episode {episode}: {file['path']} {why}, "
                "so this episode cannot be rebuilt",
                file=sys.stderr,
            )
            return None
        rebuild.append(file)
    return rebuild


def restore_files(
    new: str, written: list[Channel], label: str, records: list[dict]
) -> str | None:
    """Write the rebuilt records into the new agent's mirrors. The path of a record
    that sits in no written channel, or None where every record found its tree.

    Each record's path is where the file sat in /work; the tree it belongs to is
    the written channel whose path encloses it, under this agent's label. Every
    record is placed before any is written, so a stray leaves the new id nothing.
    """
    prefixes = {
        c.name: (c.path_for(label) if c.shape == "directory" else c.outbox)
        for c in written
    }
    placed: list[tuple[Path, str]] = []
    for file in records:
        tree = max(
            (
                name
                for name, p in prefixes.items()
                if file["path"] == p or file["path"].startswith(p + "/")
            ),
            key=lambda name: len(prefixes[name]),
            default=None,
        )
        if tree is None:
            return file["path"]
        placed.append(
            (mirror(new, tree) / file["path"][len(prefixes[tree]) + 1 :], file["text"])
        )
    for c in written:
        mirror(new, c.name).mkdir(parents=True, exist_ok=True)
    traces_dir(new).mkdir(parents=True, exist_ok=True)
    for dest, text in placed:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
    return None


# --- 18. CLI ---------------------------------------------------------------------


def show_prompt(who: str, text: str) -> None:
    """One entry of the --print-system audit: whose prompt, its bytes and its sha256."""
    same = "  (the shipped SYSTEM)" if text == SYSTEM else ""
    print(f"{who}: {text!r}")
    print(f"{len(text)} bytes  sha256={system_sha256(text)}{same}")


def print_system(config: Path | None, manifest: Path | None) -> int:
    """Print what the harness ships and what is in force. Nonzero if the pin has drifted.

    The pinned strings are the shipped default. The config, and where one is given the
    manifest, say what agents are actually told, which is what invariant 2 asks to be
    auditable. Starts no episode, spends nothing, and installs no setting.
    """
    drifted = []
    for name, text, expected in SHIPPED_TEXT:
        sha256 = system_sha256(text)
        drifted += [name] if sha256 != expected else []
        print(f"{name}: {text!r}")
        verdict = "ok" if sha256 == expected else "DRIFTED"
        print(f"{len(text)} bytes  sha256={sha256}  {verdict}")

    s, cfg = load_config(SETTINGS, config)
    print()
    print(f"config: {cfg or 'built-in defaults'}")
    show_prompt("in force", s.system_prompt)
    if manifest is not None:
        # Deferred, so harness.py is fully imported before experiment.py imports it.
        import experiment

        m = experiment.load_manifest(manifest)
        default = m["overrides"].get("system_prompt", s.system_prompt)
        print()
        print(f"manifest: {manifest}")
        show_prompt("experiment", default)
        for entry in m["agents"]:
            show_prompt(entry["id"], entry.get("system_prompt", default))
        chans = validate_channels(
            m["channels"],
            m["status_files"],
            str(manifest),
            tuple(m["labels"].values()),
            base=s,
        )[0]
        declared = validate_tools(m["tools"], chans, str(manifest))
        show_tools(m["tools"], declared)
    return 1 if drifted else 0


def print_context(
    config: Path | None, manifest: Path, selected: str | None = None
) -> int:
    """Render the complete context of fresh agents' first discussion and vote episodes.

    Uses a temporary root, so starter files and synthetic accounts can be composed by
    the same functions that build an episode without touching records, mirrors,
    containers or providers. A shell experiment's opening listing depends on the
    container and is therefore shown as its command beside the rendered digest. The
    manifest's settings are in force only while it composes, under that root.
    """
    # Deferred for the same reason as print_system(): experiment imports this module.
    import experiment

    m = experiment.load_manifest(manifest)
    agents = m["agents"]
    if selected is not None and selected not in {entry["id"] for entry in agents}:
        raise SystemExit(f"{manifest}: no agent {selected!r}")

    s, cfg = load_config(SETTINGS, config)
    if m["overrides"]:
        s = overlay(s, m["overrides"], str(manifest), MANIFEST_SETTINGS, NOT_MANIFEST)
    s = with_channels(
        s, m["channels"], m["status_files"], str(manifest), tuple(m["labels"].values())
    )
    s = with_tools(s, m["tools"], list(s.channels), str(manifest))
    source_root = s.root

    def stage_source(name: str, audit_root: Path) -> None:
        if not name or Path(name).is_absolute():
            return
        source = source_root / "files" / name
        destination = audit_root / "files" / name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    print(f"config: {cfg or 'built-in defaults'}")
    print(f"manifest: {manifest}")
    with (
        tempfile.TemporaryDirectory(prefix="agent-context-") as temporary,
        using(dataclasses.replace(s, root=Path(temporary))),
    ):
        audit_root = Path(temporary)
        sources = [c.source for c in s.channels if c.writer == "experimenter"]
        sources += [entry.get("starter_files", s.starter_files) for entry in agents]
        for name in dict.fromkeys(filter(None, sources)):
            stage_source(name, audit_root)
        for entry in agents:
            if inherited := entry.get("memory_from"):
                # Where inherit_memory reads it under this root, from the same place
                # under the root the source agent ran in.
                destination = trace_path(inherited["agent"], inherited["episode"])
                source = source_root / destination.relative_to(audit_root)
                if source.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)

        seats = {str(i): entry["id"] for i, entry in enumerate(agents, 1)}
        stamp = experiment.stamp_of(m)
        with contextlib.redirect_stdout(io.StringIO()):
            for entry in agents:
                account = load_account(
                    entry["id"], **experiment.pinned_settings_of(entry)
                )
                experiment.inherit_memory(entry, account)
                experiment.preparer(entry["id"], seats, stamp, m["labels"])(account)
                save_account(entry["id"], account)

        vote = next((tool for tool in s.tools if tool.kind == "vote"), None)
        episodes = [1] if vote is None or vote.every == 1 else [1, vote.every]
        for entry in agents:
            if selected is not None and entry["id"] != selected:
                continue
            account = load_account(entry["id"])
            channel_instances = instances(entry["id"], account)
            ensure_mirrors(channel_instances)
            memory = memory_channel(s.channels)
            if memory is None and starter_settings(account)[0]:
                raise SystemExit(
                    f"agent {entry['id']} has starter files and the channel table "
                    f"has no memory channel to put them in"
                )
            if memory is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    plant_starter_files(
                        entry["id"],
                        mirror(entry["id"], memory.name),
                        memory.path,
                        account,
                        1,
                    )

            print(f"\n=== agent {entry['id']} ===")
            system = system_of(account)
            print(
                f"system ({len(system.encode('utf-8'))} bytes, sha256={system_sha256(system)}):"
            )
            print(system, end="" if system.endswith("\n") else "\n")
            for episode in episodes:
                view = json.loads(json.dumps(account))
                view["episodes"] = [
                    {"episode": earlier, "stop": "end_turn"}
                    for earlier in range(1, episode)
                ]
                channel_instances = instances(entry["id"], view)
                rendered, _ = render_status_files(entry["id"], view)
                opening = rendered.get(s.status_files["digest"], "")
                reach = [
                    view["seating"]["labels"][seat]
                    for seat in seating_of(entry["id"], view).peers
                ]
                bound = bind_tools(
                    list(s.tools), list(s.channels), channel_instances, reach, episode
                )
                specs = episode_specs(bound)

                print(f"\n--- episode {episode} opening ---")
                if s.bash_declared:
                    print(f"opening command: {observation()}")
                    print("rendered digest (the container listing precedes it):")
                else:
                    print(f"opening input ({len(opening)} characters):")
                print(opening, end="" if opening.endswith("\n") else "\n")
                print("tool specs:")
                print(
                    json.dumps(
                        [spec.as_dict() for spec in specs], indent=2, ensure_ascii=False
                    )
                )
    return 0


def show_tools(
    declared: list[dict] | None, validated: list[Tool] | None = None
) -> None:
    """The declared tool table's sha256 and descriptions, which are prompt surface too.

    A description reaches the model in the model call the way the system prompt does,
    so --print-system audits both. A tool that declares none is named as taking the
    harness's generated description, which is computed per agent from the channel and
    seating and so is not a constant to print here.
    """
    if not declared:
        print("tools: none declared (no bash)")
        return
    print()
    stamp = f", sha256={tools_sha256(validated)}" if validated is not None else ""
    print(f"tools: {len(declared)} declared{stamp}")
    for raw in declared:
        if raw["kind"] == "bash":
            print("  bash (built-in shell)")
            continue
        said = raw.get("description", "")
        words = repr(said) if said else "(the harness's generated description)"
        print(f"  {raw['name']} ({raw['kind']} on {raw.get('channel', '')}): {words}")


def main(argv: list[str] | None = None) -> int:
    """CLI. Checks the shipped sha256 pins and the endpoint, then runs the episodes."""
    global WATCH
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--agent")
    ap.add_argument(
        "--episodes",
        type=int,
        default=1,
        metavar="N",
        help="run up to N episodes back to back, stopping early when the "
        "budget runs out or an episode ends abnormally (default: 1)",
    )
    ap.add_argument("--config", type=Path, help="default: config.toml beside this file")
    ap.add_argument(
        "--watch",
        action="store_true",
        help="echo the agent's words and the account to stdout as it runs; interleaves across parallel agents",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue the existing compatible agent account; without this flag, "
        "previous records and mirrors are preserved under archive/ and a fresh "
        "run starts",
    )
    ap.add_argument(
        "--print-system",
        action="store_true",
        help="print what the harness ships and what is in force; starts no episode",
    )
    ap.add_argument(
        "--print-context",
        action="store_true",
        help="render fresh agents' opening input and complete tool schemas; requires --manifest and starts no episode",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        metavar="PATH",
        help="the experiment this agent is part of: its environment, its settings "
        "and this agent's own pinned settings. Required to run an episode",
    )
    ap.add_argument(
        "--print-files",
        metavar="NAME",
        help=(
            "print the listing and sha256 of a directory under files/; "
            "starts no episode"
        ),
    )
    ap.add_argument(
        "--fork-from",
        metavar="AGENT",
        help="rebuild AGENT as it stood at --at-episode under the id given to --agent, "
        "and stop",
    )
    ap.add_argument(
        "--at-episode",
        type=int,
        metavar="N",
        help="the episode of --fork-from to fork at",
    )
    a = ap.parse_args(argv)

    if a.watch:
        WATCH = True
        sys.stdout.reconfigure(errors="replace")

    if a.print_system:
        return print_system(a.config, a.manifest)
    if a.print_context:
        if a.manifest is None:
            ap.error("--print-context requires --manifest")
        return print_context(a.config, a.manifest, a.agent)
    # Audits invariant 9 without starting anything, so it runs on a drifted prompt
    # too; start() is what rejects a run before an episode costs money.
    if a.print_files:
        if not files_dir(a.print_files).is_dir():
            print(
                f"no starter files {a.print_files!r} under {SETTINGS.root / 'files'}",
                file=sys.stderr,
            )
            return 2
        listing = files_listing(a.print_files)
        for rel, data in listing:
            print(f"{len(data):>9}  {rel}")
        print(
            f"{len(listing)} files, {sum(len(d) for _, d in listing)} bytes, sha256={files_sha256(a.print_files)}"
        )
        return 0
    if not a.agent:
        ap.error("--agent is required")
    if a.episodes < 1:
        ap.error("--episodes must be at least 1")
    if bool(a.fork_from) != (a.at_episode is not None):
        ap.error("--fork-from and --at-episode go together")
    if a.at_episode is not None and a.at_episode < 1:
        ap.error("--at-episode must be at least 1")
    # Reads the parent's recorded environment and writes a copy of it. No setting
    # decides anything here, so it runs before the config is even read.
    if a.fork_from:
        return fork(a.fork_from, a.at_episode, a.agent)
    if not a.manifest:
        ap.error(
            "--manifest is required: an agent runs as part of an experiment, and the "
            "manifest is what declares its environment and its settings"
        )
    # Deferred, so harness.py is fully imported before experiment.py imports it.
    import experiment

    m = experiment.load_manifest(a.manifest)
    ids = [e["id"] for e in m["agents"]]
    entry = next((e for e in m["agents"] if e["id"] == a.agent), None)
    if entry is None:
        ap.error(f"{a.manifest} seats {ids}, and not {a.agent!r}")
    # Which file set the process parameters is the one thing about them the trace
    # cannot record: an agent reading no config and one reading a config of every
    # default are the same episode. Everything else is the manifest's, and its sha256
    # is in every trace.
    router = start(
        a.config,
        overrides=m["overrides"],
        requirements={(e["provider"], e["model"]) for e in m["agents"]},
        channel_tables=m["channels"],
        status_files=m["status_files"],
        labels=tuple(m["labels"].values()),
        tool_tables=m["tools"],
    )
    catch_signals()
    if not a.resume:
        archive_agents([a.agent], FRESH_START)
    account = load_account(a.agent, **experiment.pinned_settings_of(entry))
    experiment.inherit_memory(entry, account)
    seat = experiment.preparers(ids, experiment.stamp_of(m), m["labels"], m["schedule"])
    return run_episodes(a.agent, router, a.episodes, seat(a.agent))


if __name__ == "__main__":
    # experiment.py imports this file as `harness`, so the CLI runs in that module and
    # the settings it installs are the ones experiment.py reads.
    import harness

    sys.exit(harness.main())
