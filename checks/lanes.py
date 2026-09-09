"""The two lanes an episode can run in, and the fixtures every check shares.

The boxes (a directory on this machine, a recording one, one that refuses to
start), the throwaway roots, the fake experiment, the persona table, and the
helpers that read the record back."""

from __future__ import annotations

from typing import Callable
from pathlib import Path
import contextlib
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import experiment
import harness
import providers
import view

from checks.fake import fake, usage

# The API errors here are scripted, so the wait between retries is time spent
# proving nothing. What is retried and how often still is.
harness.RETRY_BASE = 0


class Skip(Exception):
    """This check needs something this machine cannot give it."""

# Set by --real: every check takes a container, including the ones that would
# otherwise run on the host box. What proves the two lanes still agree.
REAL_ONLY = False

# The answer to docker_ready(), once some process has paid for it. Carried into
# workers and not asked again in each of them.
_DOCKER: bool | None = None

# The pid of the process running this suite, carried into every worker so that
# each one's containers say which suite they belong to. What lets the sweep find
# its own and nothing else: another suite's live containers, or a real agent's,
# are not this one's to remove.
SUITE = os.getpid()

# The name every suite's worker containers share: two numbers and two dashes. A
# real agent's container, mtr-w01-0001, has one number and cannot match.
WIDE_SWEEP = r"mtr-w[0-9]+-[0-9]+-"


def configure(real: bool, docker: bool, suite: int) -> None:
    """Set up a process to run checks in. Called in the parent and every worker.

    Each worker gets its own container prefix and carries `suite`, so no worker
    reaps another's. The Docker answer is carried in and not asked again.
    """
    global REAL_ONLY, _DOCKER, SUITE
    REAL_ONLY, _DOCKER, SUITE = real, docker, suite
    harness.CONTAINER_PREFIX = f"mtr-w{suite}-{os.getpid()}-"


def sweep_filter() -> str:
    """The name a container has to contain to be this suite's to remove.

    A docker name filter matches anywhere, so this has to be a string nothing
    else can contain. The suite's pid is what makes it one.
    """
    return f"mtr-w{SUITE}-"


def docker_ready() -> bool:
    """True if the daemon is up and the image is built. Asked once per process."""
    global _DOCKER
    if _DOCKER is None:
        _DOCKER = _ask_docker()
    return _DOCKER


def _ask_docker() -> bool:
    """Put the question to the daemon. Says so once if the image is not built."""
    if not shutil.which("docker") or subprocess.run(["docker", "info"], capture_output=True).returncode:
        return False
    if subprocess.run(["docker", "image", "inspect", harness.IMAGE], capture_output=True).returncode:
        print(f"image {harness.IMAGE} not built\n")
        return False
    return True

#
# Arithmetic checks - what a turn cost, what reached the series, which stop an
# episode ended on - run against a directory and a bash process on this machine.
# What only a container can show stays on docker_root below.
def host_bash() -> str | None:
    """The bash to run the host box's episodes in, as an absolute path.

    An absolute path: on Windows PATH can resolve `bash` to Git's or WSL's, which
    disagree about what a path is and what /tmp means.
    """
    return shutil.which("bash")


class HostShell(harness.Shell):
    """The episode shell, as a bash process on this machine.

    Inherits the sentinel framing, timeout, and output ceiling from harness.Shell;
    only where the process runs and how a balance is rewritten differ.
    """

    def __init__(self, box: "HostBox") -> None:
        self.box = box
        super().__init__(box.name)

    def argv(self) -> list[str]:
        # No profile: what the agent's shell is must not depend on this account.
        return [host_bash(), "--norc", "--noprofile"]

    def popen_kwargs(self) -> dict:
        # DETACHED from the base class, so the two lanes agree about which
        # processes a signal aimed at the harness reaches.
        return {**super().popen_kwargs(),
                "cwd": str(self.box.work),
                # Stop MSYS rewriting paths inside the agent's own commands.
                "env": {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}}

    def republish_balance(self, label: str, series: list[int], expected: str) -> str:
        n = self.box.work / harness.balance_name(label)
        was = n.read_text(encoding="utf-8") if n.exists() else ""
        n.write_text(harness.render_balance(series), encoding="utf-8", newline="\n")
        return "ok" if was == expected else "tampered"


class HostBox:
    """An episode's environment as a directory on this machine, in place of a container.

    Same five methods run_once asks of harness.Container, same instances. No
    ownership and no locking: a check turning on either belongs on docker_root.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = tempfile.mkdtemp(prefix="mtr-host-")
        self.work = Path(self.dir)

    @classmethod
    def start(cls, name: str) -> "HostBox":
        return cls(name)

    def load(self, instances: list[harness.Instance], files: dict[str, str]) -> None:
        for inst in instances:
            if inst.nested:
                continue                    # rides in with the tree above it
            dest = self.work / inst.path
            if inst.is_file:
                # A sender that has not addressed this agent, and a sender that
                # aimed something other than one file at it, arrive the same way:
                # as nothing.
                dest.parent.mkdir(exist_ok=True, parents=True)
                if inst.host.is_file():
                    shutil.copyfile(inst.host, dest)
                continue
            dest.mkdir(exist_ok=True, parents=True)
            if inst.host.is_dir():
                shutil.copytree(inst.host, dest, dirs_exist_ok=True)
        for name, text in files.items():
            (self.work / name).parent.mkdir(exist_ok=True, parents=True)
            (self.work / name).write_text(text, encoding="utf-8", newline="\n")

    def shell(self) -> HostShell:
        return HostShell(self)

    def save(self, instances: list[harness.Instance]) -> bool:
        kept = True
        for inst in instances:
            if inst.writable and not inst.nested and not inst.is_file:
                kept = harness.save_state(
                    inst.host, self._fetcher(self.work / inst.path), lambda: None) and kept
        return kept

    def _fetcher(self, src: Path) -> Callable[[Path], bool]:
        def fetch(dest: Path) -> bool:
            if not src.is_dir():
                return False
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return True
        return fetch

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class RecordingBox(HostBox):
    """A HostBox that writes down every start, load and close, for a check on ordering.

    `events` is bound by recording(), which a check holds for its duration.
    """
    events: list[tuple[str, str]] = []
    lock = threading.Lock()

    @classmethod
    def note(cls, what: str, name: str) -> None:
        with cls.lock:
            cls.events.append((what, name))

    @classmethod
    def start(cls, name: str) -> "RecordingBox":
        cls.note("start", name)
        return cls(name)

    def load(self, instances: list[harness.Instance], files: dict[str, str]) -> None:
        super().load(instances, files)
        self.note("load", self.name)

    def close(self) -> None:
        self.note("close", self.name)
        super().close()


@contextlib.contextmanager
def recording():
    """A fresh RecordingBox.events for one check, cleared again on exit."""
    RecordingBox.events = []
    try:
        yield RecordingBox.events
    finally:
        RecordingBox.events = []


class NoBox:
    """A box no episode may be built in: start() fails the check that reached it."""

    @classmethod
    def start(cls, name: str) -> "NoBox":
        raise AssertionError(f"an environment was built for {name}, and none may be")


def never_start(*args, **kwargs) -> Callable:
    """Stands in for harness.start where nothing under test may reach the API."""
    raise AssertionError("harness.start was reached, and nothing here may start an agent")


def agent_of(container: str) -> str:
    """The agent a container name belongs to: the prefix and the episode index removed."""
    return container[len(harness.CONTAINER_PREFIX):].rsplit("-", 1)[0]


def leaked_containers() -> str:
    """The names of every container this worker's episodes of agent t left behind."""
    return subprocess.run(["docker", "ps", "-a", "--filter", f"name={harness.CONTAINER_PREFIX}t-",
                           "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.strip()


def manifest_file(root: Path, text: str, name: str = "c.toml") -> Path:
    """Write an experiment manifest under a temporary ROOT, for the manifest checks to use."""
    head = text.split("[[agent]]", 1)[0]
    prefix = ""
    if not re.search(r"(?m)^provider\s*=", head):
        prefix += 'provider = "anthropic"\n'
    if not re.search(r"(?m)^model\s*=", head):
        prefix += 'model = "claude-sonnet-5"\n'
    text = prefix + text
    p = root / "experiments" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def seats_manifest(root: Path, ids: Iterable[str], name: str = "seats.toml") -> Path:
    """A manifest seating these agents and declaring silence.

    Every run names its experiment, so a round driven from a list of ids needs a file
    to name; the settings come from whatever the check pinned. A manifest states what
    its agents are told, so the empty prompt is written out.
    """
    seat = "[[agent]]" + chr(10) + 'id = "%s"' + chr(10)
    return manifest_file(root, 'system_prompt = ""' + chr(10) + "".join(seat % i for i in ids), name)


def tables(*extra: dict, **per_name: dict) -> list[dict]:
    """The default channel table as [[channel]] tables, with fields changed by name.

    tables(transfer={"rebate_percent": 50}) is the default table with one field moved;
    positional dicts are whole extra channels. For temp_root(channels=...).
    """
    declared = [c.declared() for c in harness.DEFAULT_CHANNELS]
    by = {t["name"]: t for t in declared}
    for name, fields in per_name.items():
        by[name].update(fields)
    return declared + list(extra)

# The obligation at half of what is left, on one channel or on all three, and
# the transfer channel at its full rebate.
HALF = {"silence_penalty_percent": 50}

ALL_OWED = tables(blackboard=HALF, mail=HALF, transfer=HALF)

FULL_REBATE = tables(transfer={"rebate_percent": 100})


def offers(*declared: dict) -> list[dict]:
    """[[tool]] tables, one per (kind, channel) pair given as "kind:channel".

    offers("write_slot:mail") is one tool named after its kind; a dict is a whole
    tool table, for a check that wants its own name. For temp_root(tools=...).
    """
    out = []
    for d in declared:
        out.append(d if isinstance(d, dict) else
                   dict(zip(("kind", "channel"), d.split(":")), name=d.split(":")[0]))
    return out


def digest_name() -> str:
    """What the digest is called under the table in force."""
    return harness.HARNESS_FILES["digest"]


def ledger_name() -> str:
    """What the ledger is called under the table in force."""
    parsed = harness.schema_channel(harness.channels())
    return parsed.ledger if parsed else ""


def channel_toml(declared: list[dict], harness_files: dict | None = None) -> str:
    """Render channel tables (and harness file names) as the TOML a config file holds."""
    def value(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return repr(v) if isinstance(v, int) else '"' + v.replace('"', '\\"') + '"'
    out = []
    if harness_files:
        out.append("[harness_files]\n" + "".join(f"{k} = {value(v)}\n" for k, v in harness_files.items()))
    for table in declared:
        out.append("[[channel]]\n" + "".join(f"{k} = {value(v)}\n" for k, v in table.items()))
    return "\n".join(out)


def shared(root: Path, name: str = "brief", path: str = "shared", **files: str) -> Path:
    """Plant an experimenter channel's files under a temporary ROOT and declare the channel.

    Inside a root's block, since pinned() puts the table back when the block ends.
    """
    planted = plant(root, name, **files)
    harness.apply_channels(tables({"name": path, "writer": "experimenter", "source": name,
                                   "path": path}), None, "check")
    return planted

# Every harness global a check is allowed to move, and therefore every one pinned()
# puts back. temp_root refuses any name outside this set.
RESTORED = harness.TUNABLES | {"ROOT", "WATCH", "REFUSAL_TURNS", "BOX", "drive", "ready", "start",
                                "CHANNELS", "HARNESS_FILES", "TOOLS", "SHELL_TOOL", "PINNED", "load_account",
                                "replace_file",
                            # Set per check and put back by pinned(), so no check
                            # carries into the next in the same worker.
                            "STOPPING", "catch_signals"}


@contextlib.contextmanager
def pinned():
    """Restore every episode global a check may move, on exit.

    catch_signals is stubbed and not restored: a handler installed by a check
    driving harness.main would outlive it and answer the suite's own Ctrl+C.
    """
    saved = {k: getattr(harness, k) for k in RESTORED}
    harness.catch_signals = lambda: None
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(harness, k, v)


@contextlib.contextmanager
def rooted(box, channels=None, harness_files=None, tools=None, **overrides):
    """Point harness at a throwaway directory, with episodes running in `box`.

    `overrides` set harness module globals (MAX_TURNS=1, COMMAND_TIMEOUT=2) for the
    duration, and pinned() puts every one of them back. `tools` are [[tool]] tables,
    decided against whatever channel table is then in force.
    """
    provider = overrides.pop("PROVIDER", "anthropic")
    model = overrides.pop("MODEL", "claude-sonnet-5")
    unknown = set(overrides) - RESTORED
    assert not unknown, f"a root cannot restore {sorted(unknown)}"
    with pinned(), tempfile.TemporaryDirectory(
            prefix="mtr-check-", ignore_cleanup_errors=True) as d:
        harness.ROOT = Path(d)
        harness.BOX = box
        load_account = harness.load_account

        def load_test_account(agent, **terms):
            existing = harness.account_on_disk(agent)
            terms.setdefault("provider", existing.get("provider", provider))
            terms.setdefault("model", existing.get("model", model))
            return load_account(agent, **terms)

        harness.load_account = load_test_account
        for k, v in overrides.items():
            setattr(harness, k, v)
        if channels is not None or harness_files is not None:
            harness.apply_channels(channels, harness_files, "check")
        declared = [{"name": "bash", "kind": "bash"}] if tools is None else tools
        harness.apply_tools(declared, harness.channels(), "check")
        yield Path(d)


@contextlib.contextmanager
def host_root(channels=None, harness_files=None, tools=None, **overrides):
    """A throwaway agent pinned to this machine, whatever --real says.

    For the few checks about what is *sent* and not about the episode it drives.
    Everything else wants temp_root, which --real does promote.
    """
    if not host_bash():
        raise Skip
    with rooted(HostBox, channels=channels, harness_files=harness_files, tools=tools, **overrides) as d:
        yield d


@contextlib.contextmanager
def temp_root(channels=None, harness_files=None, tools=None, **overrides):
    """A throwaway agent whose episodes are a directory and a shell on this machine.

    What most checks want: the pipeline end to end - account, turns, series,
    trace - without paying for a container that proves nothing they assert.
    """
    if REAL_ONLY:
        with docker_root(channels=channels, harness_files=harness_files, tools=tools, **overrides) as d:
            yield d
        return
    with host_root(channels=channels, harness_files=harness_files, tools=tools, **overrides) as d:
        yield d


@contextlib.contextmanager
def docker_root(channels=None, harness_files=None, tools=None, **overrides):
    """A throwaway agent whose episodes are real containers.

    For the checks that turn on something only a container has. Skips when
    Docker is unavailable, which is the one reason a check here cannot run.
    """
    if not docker_ready():
        raise Skip
    with rooted(harness.Container, channels=channels, harness_files=harness_files, tools=tools, **overrides) as d:
        yield d


@contextlib.contextmanager
def quiet():
    """Swallow episode output so the check list stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


@contextlib.contextmanager
def serving():
    """view.py's server bound to a free port on a thread; yields the base URL."""
    httpd = view.serve(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def got(base: str, path: str) -> tuple[int, dict]:
    """One GET against the served API: the status and the JSON body."""
    with urllib.request.urlopen(base + path) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def episode_once(*steps, seen=None):
    """One episode against a fresh agent, output suppressed."""
    with quiet():
        return harness.run_once("t", fake(*steps, seen=seen))


def ground_truth(agent="t") -> dict:
    """account.json as it stands on disk, read back and not carried."""
    return json.loads((harness.records_dir(agent) / "account.json").read_text(encoding="utf-8"))


def episodes_taken(ids: list[str]) -> dict[str, int]:
    """How many episodes each agent's account records."""
    return {agent: len(ground_truth(agent)["episodes"]) for agent in ids}


def trace_on_disk(agent: str, index: int) -> dict:
    """One episode's trace, read back from where close_episode wrote it."""
    return json.loads(harness.trace_path(agent, index).read_text(encoding="utf-8"))


def files_by_path(t: dict) -> dict[str, dict]:
    """A trace's file records, by the path each stood at."""
    return {f["path"]: f for f in t["files"]}


def reconciled(account: dict, spent: int) -> int:
    """What an account must hold: every term of the identity, each one series element."""
    return (account["initial"] - spent + account.get("rebated", 0) + account.get("received", 0)
            - account.get("debited", 0) - sum((account.get("penalised") or {}).values())
            + account.get("forgiven", 0))


def span_of(series: list[int], s: dict) -> list[int]:
    """The elements of the series one episode's record spans, harness included."""
    return series[s["series_from"]:s["series_to"] + 1]


def elements_of(s: dict) -> int:
    """How many series elements an episode's record says it appended, plus the entry at its start."""
    return (s["turns"] + 1 + bool(s["transfer"]["rebate"] or s["transfer"].get("debit"))
            + bool(s["transfer"]["penalty"]) + bool(s["channels"]["blackboard"]["penalty"])
            + bool(s["channels"]["mail"]["penalty"]) + bool(s["forgiven"])
            # A credit from a peer settling in the same simultaneous round lands
            # inside the receiver's span; in rotation it lands between spans.
            + bool(s.get("received")))


def without_listing_times(x):
    """The same value with `ls -la` mtimes flattened.

    A listing renders its times to the minute, so two environments built either side
    of one differ there. The minute a directory was made is not something
    watching can reach.
    """
    if isinstance(x, str):
        return re.sub(r"[A-Z][a-z]{2} [ \d]?\d \d{2}:\d{2}", "<mtime>", x)
    if isinstance(x, list):
        return [without_listing_times(v) for v in x]
    if isinstance(x, dict):
        return {k: without_listing_times(v) for k, v in x.items()}
    return x


def differs(a, b) -> str:
    """The first few lines on which two request records disagree."""
    def lines(x) -> list[str]:
        return json.dumps(x, indent=1, default=str, sort_keys=True).splitlines()
    delta = [d for d in difflib.unified_diff(lines(a), lines(b), lineterm='')
             if d.startswith(('+', '-')) and not d.startswith(('+++', '---'))]
    return ' | '.join(d[:300] for d in delta[:6]) or '(equal)'


def plant(root: Path, name: str = "s", **files: str) -> Path:
    """Write a starter-files tree under a temporary ROOT, for the starter-files checks to use."""
    d = root / "files" / name
    for rel, text in (files or {"m1": "alpha\n", "d/m2": "beta\n"}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8", newline="\n")
    return d


def lay_out(root: Path, **agents: dict[str, str]) -> list[str]:
    """Lay out an experiment's directories under the ROOT in force, for the experiment checks to use.

    "group/" goes on that agent's blackboard, "out/" in its outbox where only the seat
    it names reads it, and anything else in its private store.
    """
    trees = {"group/": "blackboard", "out/": "mail"}
    for agent, files in agents.items():
        for where in ("notes", "blackboard", "mail"):
            harness.mirror(agent, where).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            prefix = next((k for k in trees if name.startswith(k)), None)
            tree = harness.mirror(agent, trees.get(prefix, "notes"))
            p = tree / name.removeprefix(prefix or "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
    return list(agents)


def environment_of(agent: str, ids: list[str]) -> list[harness.Instance]:
    """One agent's environment, built the way the harness builds it.

    Through harness.environment and not beside it: a check that assembled the
    instances itself would agree with a copy of the layout and not with the layout.
    """
    seats = experiment.seats_of(ids)
    seat = next(i for i, r in seats.items() if r == agent)
    return harness.environment(agent, {"seat": seat, "peers": {"seen": seats}})


def seated(root: Path, agent: str = "t", labels: dict[str, str] | None = None,
           **agents: dict[str, str]) -> list[str]:
    """Lay out an experiment, create every account, and seat every agent in it.

    What experiment.py's prepare() does before each episode of a round: a seat, the
    whole seating, and neighbours that exist and are seated themselves.
    """
    # An agent that is not in its own experiment is not a seating at all, so `agent` is
    # added if the caller left it out - at the front, since the seat a check does
    # not name is the one it does not care about. A caller that does name it
    # keeps it where it put it, which is how a check reaches a seat other than 1.
    ids = lay_out(root, **(agents if agent in agents else {agent: {}} | agents))
    seats = experiment.seats_of(ids)
    for seat, other in seats.items():
        with quiet():
            account = harness.load_account(other)
        named = {s: (labels or {}).get(s, s) for s in seats}
        account["seat"], account["label"] = seat, named[seat]
        account["peers"] = {"seen": seats, "labels": named}
        harness.save_account(other, account)
    return ids


def turn_cost() -> int:
    """What one scripted turn costs, in micro-dollars."""
    raw = usage()
    spec = providers.model_spec("anthropic", "claude-sonnet-5")
    return (raw.input_tokens * spec.rate("uncached_input") +
            raw.output_tokens * spec.rate("output")) // 100


def put_out(agent: str) -> None:
    """Leave an agent flat on zero, where an episode that spent past its budget leaves it.

    Written into the account and not spent down to, so a check about what
    happens to a seat that is out does not also depend on how it got there.
    """
    account = harness.load_account(agent)
    account["remaining"] = 0
    account["series"].append(0)
    harness.save_account(agent, account)


def unfinished() -> None:
    """Take away the first episode's trace, leaving the raw log a running one leaves."""
    harness.trace_path("t", 1).unlink()


@contextlib.contextmanager
def two_seats():
    """An experiment of two, laid out and seated, with a blackboard and a store each."""
    with rooted(HostBox) as root:
        ids = seated(root, "g01",
                     g01={"NOTES.md": "given\n", "secret.md": "mine\n", "group/msg": "hello 2\n"},
                     g02={"secret.md": "theirs\n", "group/msg": "hello 1\n"})
        for agent, series in (("g01", [1000, 900]), ("g02", [1000, 800])):
            account = harness.load_account(agent)
            account["series"], account["remaining"] = series, series[-1]
            account["starter_files_landed"] = {"name": "objective-notes", "paths": ["NOTES.md"]}
            harness.save_account(agent, account)
        yield view.experiment_of("g01"), experiment.seats_of(ids)


def fake_experiment(root: Path, acted: list[tuple], series: tuple[int, ...] = (1000,),
                    agents: tuple[str, ...] = ("g01", "g02", "g03"), **trace_fields) -> dict:
    """An experiment written straight to disk: one trace per episode taken, one account per seat.

    `acted` is the episodes in the order they started, each `(agent, "hh:mm")` or
    `(agent, "hh:mm", {fields})` for a trace with more in it; `trace_fields` go
    into every trace. Returns the experiment as view.py reads it.
    """
    seats = experiment.seats_of(list(agents))
    taken: dict[str, int] = {}
    for agent, at, *more in acted:
        taken[agent] = taken.get(agent, 0) + 1
        p = harness.trace_path(agent, taken[agent])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "agent": agent, "episode": taken[agent], "stop": "end_turn", "spent": 1,
            "turns": [], "remaining": 0, "files": [], "state_saved": True,
            "provenance": {"started_at": f"2026-01-01T{at}:00Z", "peers": seats},
            **trace_fields, **(more[0] if more else {})}), encoding="utf-8")
    for seat, agent in seats.items():
        harness.records_dir(agent).mkdir(parents=True, exist_ok=True)
        (harness.records_dir(agent) / "account.json").write_text(json.dumps({
            "agent": agent, "seat": seat, "peers": {"seen": seats},
            "series": list(series), "remaining": series[-1], "initial": 1000, "episodes": [],
        }), encoding="utf-8")
    return view.experiment_named("g")

# The spec's persona experiment (docs/manifest.md section 9), without its brief:
# a journal with an identity file inside it, a noticeboard per label, letters, and
# no transfer channel.
PERSONA = [
    {"name": "journal", "writer": "self", "readers": "self", "shape": "directory", "path": "journal"},
    {"name": "identity", "writer": "self", "readers": "self", "shape": "file",
     "path": "journal/IDENTITY.md"},
    {"name": "noticeboard", "writer": "self", "readers": "all", "shape": "directory",
     "path": "from-{label}", "measured": True},
    {"name": "letters", "writer": "self", "readers": "addressee", "shape": "mailbox",
     "outbox": "to", "inbox": "from", "measured": True},
]

PERSONA_FILES = {"balance": "balance", "digest": "digest"}

PERSONA_LABELS = {"1": "Studio", "2": "Game"}


def refused(call, *words: str, because: str = "accepted what should have been refused",
            code: int | None = None) -> None:
    """Run `call`, which must exit naming every word given, and with `code` where one is given."""
    try:
        call()
    except SystemExit as e:
        if code is not None:
            assert e.code == code, (code, e.code)
        for word in words:
            assert word in str(e), (word, str(e))
    else:
        raise AssertionError(because)
