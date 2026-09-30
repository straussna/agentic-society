"""Provider contracts, wire shapes, pricing, and mixed-seat configuration."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS

import experiment
import harness
import providers
from providers import SessionContext, ToolResult, ToolSpec
from providers.anthropic import AnthropicProvider
from providers.base import classify_error
from providers.openai import OpenAIProvider, normalize as normalize_openai
from checks.fake import DEFAULT, per_agent, say
from checks.lanes import amend, episode_once, pinned, quiet, temp_root


TOOLS = (ToolSpec("bash", "run", {"type": "object", "properties": {"command": {"type": "string"}},
                                  "required": ["command"], "additionalProperties": False}),)
CONTEXT = SessionContext("a", "1", 1, Path("interactions"))


class Messages:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def create(self, **params):
        self.sent.append(params)
        return self.response


class Responses(Messages):
    pass


class Raising:
    """An SDK resource whose every call fails with `error`."""

    def __init__(self, error):
        self.error = error

    def create(self, *args, **params):
        raise self.error

    retrieve = create


# Stand-ins for the SDKs' exception classes, named and nested as both SDKs declare
# them; the checks import neither SDK. classify_error reads a class's name, the
# names in its MRO, and status_code.
class APIError(Exception):
    pass


class APIConnectionError(APIError):
    def __init__(self, message="Connection error."):
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    def __init__(self):
        super().__init__("Request timed out.")


class APIStatusError(APIError):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


class AuthenticationError(APIStatusError):
    def __init__(self):
        super().__init__(401)


class PermissionDeniedError(APIStatusError):
    def __init__(self):
        super().__init__(403)


def statusless(name):
    """An exception of a class called `name` with no status code, which only its name classifies."""
    return type(name, (APIError,), {})()


@contextlib.contextmanager
def swapped(table, **values):
    """Set each key of `table` to its value, or remove it for None; put every one back on exit."""
    def put(key, value):
        if value is None:
            table.pop(key, None)
        else:
            table[key] = value
    old = {key: table.get(key) for key in values}
    try:
        for key, value in values.items():
            put(key, value)
        yield
    finally:
        for key, value in old.items():
            put(key, value)


def unbuildable(name):
    """An SDK module whose client constructors fail the check that reaches them."""
    module = types.ModuleType(name)

    def build(*args, **kwargs):
        raise AssertionError(f"the {name} SDK built a client")
    module.Anthropic = module.OpenAI = build
    return module


def check_anthropic_messages_wire_shape_and_state():
    response = NS(id="a1", model="claude-sonnet-5-20260901", stop_reason="tool_use",
                  stop_details=None, usage=NS(input_tokens=10, output_tokens=2,
                                              cache_read_input_tokens=3,
                                              cache_creation_input_tokens=4,
                                              cache_creation=None),
                  content=[NS(type="tool_use", id="call1", name="bash",
                              input={"command": "pwd"})],
                  model_dump=lambda: {"id": "a1", "model": "claude-sonnet-5-20260901",
                                      "stop_reason": "tool_use", "content": [], "usage": {}})
    messages = Messages(response)
    session = AnthropicProvider(NS(messages=messages)).open_session(
        "claude-sonnet-5", "system", TOOLS, 123, CONTEXT)
    first = session.request("hello")
    turn = first.normalize()
    session.request((ToolResult("call1", "ok"),))
    sent = messages.sent
    assert set(sent[0]) == {"model", "max_tokens", "messages", "tools", "cache_control", "system"}
    assert sent[0]["model"] == "claude-sonnet-5" and sent[0]["max_tokens"] == 123
    assert sent[0]["tools"] == [{"name": "bash", "description": "run",
                                  "input_schema": TOOLS[0].input_schema, "strict": True}]
    assert sent[1]["messages"][-1]["content"][0] == {
        "type": "tool_result", "tool_use_id": "call1", "content": "ok", "is_error": False}
    assert turn.provider == "anthropic" and turn.tool_calls[0].input == {"command": "pwd"}


def check_openai_responses_wire_shape_and_manual_state():
    output = [NS(type="reasoning", id="rs1", summary=[NS(text="thought")], content=[],
                 encrypted_content="cipher", status=None),
               NS(type="function_call", id="fc1", call_id="call1", name="bash",
                  arguments='{"command":"pwd"}', status="completed", async_=None)]
    response = NS(id="r1", model="gpt-5.6-terra-20260908", status="completed",
                  incomplete_details=None, output=output,
                  usage=NS(input_tokens=20, output_tokens=5,
                           input_tokens_details=NS(cached_tokens=7, cache_write_tokens=2),
                           output_tokens_details=NS(reasoning_tokens=3)),
                  model_dump=lambda: {"id": "r1", "model": "gpt-5.6-terra-20260908",
                                      "status": "completed", "output": [], "usage": {}})
    responses = Responses(response)
    session = OpenAIProvider(NS(responses=responses)).open_session(
        "gpt-5.6-terra", "system", TOOLS, 321, CONTEXT)
    turn = session.request("hello").normalize()
    session.request((ToolResult("call1", "ok"),))
    sent = responses.sent
    assert sent[0]["store"] is False
    assert sent[0]["include"] == ["reasoning.encrypted_content"]
    assert sent[0]["tools"] == [{"type": "function", "name": "bash", "description": "run",
                                  "parameters": TOOLS[0].input_schema, "strict": True}]
    assert sent[1]["input"][-1] == {"type": "function_call_output", "call_id": "call1",
                                     "output": "ok"}
    reasoning = next(item for item in sent[1]["input"] if item.get("type") == "reasoning")
    call = next(item for item in sent[1]["input"] if item.get("type") == "function_call")
    assert reasoning == {"type": "reasoning", "id": "rs1", "summary": [{"text": "thought"}],
                         "content": [], "encrypted_content": "cipher"}
    assert call == {"type": "function_call", "call_id": "call1", "name": "bash",
                    "arguments": '{"command":"pwd"}'}
    assert turn.provider == "openai" and turn.usage.reasoning_tokens == 3


def check_provider_tool_schemas_are_identical():
    anthropic_response = NS(id="a", model="claude-sonnet-5", stop_reason="end_turn",
                            stop_details=None, content=[], usage=NS(input_tokens=0, output_tokens=0,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0,
                            cache_creation=None), model_dump=lambda: {})
    openai_response = NS(id="o", model="gpt-5.6-terra", status="completed",
                         incomplete_details=None, output=[], usage=NS(input_tokens=0, output_tokens=0,
                         input_tokens_details=NS(cached_tokens=0, cache_write_tokens=0),
                         output_tokens_details=NS(reasoning_tokens=0)), model_dump=lambda: {})
    am, om = Messages(anthropic_response), Responses(openai_response)
    AnthropicProvider(NS(messages=am)).open_session("claude-sonnet-5", "", TOOLS, 1, CONTEXT).request("x")
    OpenAIProvider(NS(responses=om)).open_session("gpt-5.6-terra", "", TOOLS, 1, CONTEXT).request("x")
    a = am.sent[0]["tools"][0]
    o = om.sent[0]["tools"][0]
    assert (a["name"], a["description"], a["input_schema"], a["strict"]) == \
           (o["name"], o["description"], o["parameters"], o["strict"])


def check_openai_usage_prices_cache_reasoning_and_long_context():
    def response(prefix):
        return NS(id=str(prefix), model="gpt-5.6-terra", status="completed",
                  incomplete_details=None, output=[], usage=NS(
                      input_tokens=prefix, output_tokens=100,
                      input_tokens_details=NS(cached_tokens=1000, cache_write_tokens=500),
                      output_tokens_details=NS(reasoning_tokens=40)))
    short = normalize_openai(response(10_000), "gpt-5.6-terra")
    long = normalize_openai(response(300_000), "gpt-5.6-terra")
    assert short.usage.uncached_input_tokens == 8_500
    assert short.usage.reasoning_tokens == 40 and short.usage.output_tokens == 100
    assert sum(c.centi_micros for c in long.charges) > 2 * sum(c.centi_micros for c in short.charges)
    assert next(c for c in long.charges if c.kind == "output").centi_micros == 100 * 1200 * 3 // 2


def check_promotional_price_expiry_is_scoped_to_seated_models():
    assert not providers.lapsed_prices([("openai", "gpt-5.6-sol")], dt.date(2026, 11, 21))
    assert providers.lapsed_prices([("openai", "gpt-5.6-sol")], dt.date(2026, 11, 22))
    assert not providers.lapsed_prices([("openai", "gpt-5.6-terra")], dt.date(2099, 1, 1))


def check_manifest_resolves_provider_model_per_seat():
    text = '''provider = "anthropic"\nmodel = "claude-sonnet-5"\nsystem_prompt = ""\n\n[[agent]]\nid = "a"\n\n[[agent]]\nid = "b"\nprovider = "openai"\nmodel = "gpt-5.6-terra"\n'''
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "mixed.toml"
        path.write_text(text, encoding="utf-8")
        manifest = experiment.load_manifest(path)
    assert [(a["provider"], a["model"]) for a in manifest["agents"]] == [
        ("anthropic", "claude-sonnet-5"), ("openai", "gpt-5.6-terra")]


def check_mixed_provider_simultaneous_round():
    with temp_root(), quiet():
        harness = __import__("harness")
        harness.load_account("claude", provider="anthropic", model="claude-sonnet-5")
        harness.load_account("gpt", provider="openai", model="gpt-5.6-terra")
        live = {"claude", "gpt"}
        ran = experiment.simultaneous_round(
            ["claude", "gpt"], live, 0,
            per_agent(claude=(say("a"),), gpt=(say("o"),)))
        traces = [json.loads(harness.trace_path(agent, 1).read_text(encoding="utf-8"))
                  for agent in ("claude", "gpt")]
    assert ran and [(t["provider"], t["requested_model"]) for t in traces] == [
        ("anthropic", "claude-sonnet-5"), ("openai", "gpt-5.6-terra")]


def check_unknown_providers_and_models_are_refused():
    for provider, model in (("gateway", "x"), ("anthropic", "gpt-5.6-terra"),
                            ("openai", "claude-sonnet-5")):
        try:
            providers.model_spec(provider, model)
        except providers.ProviderConfigurationError:
            continue
        raise AssertionError(f"accepted {provider}/{model}")


def check_cli_provider_and_model_overrides_are_paired():
    manifest = Path(__file__).parents[1] / "experiments" / "examples" / "sandbox.toml"
    for lone in (("--provider", "openai"), ("--model", "gpt-5.6-terra")):
        try:
            experiment.main([str(manifest), *lone])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"accepted lone override {lone}")


def check_provider_preflight_requires_only_its_own_key():
    """A provider without its key, or whose key is refused, says where the key goes.

    A preflight that got no answer is not about the key, and does not say so; nor is
    one the key was accepted for and not permitted, whose key is already set.
    """
    class Models:
        def retrieve(self, *args, **kwargs):
            return NS(id=args[0] if args else kwargs.get("model_id"))
    with swapped(os.environ, ANTHROPIC_API_KEY=None, OPENAI_API_KEY="test"):
        OpenAIProvider(NS(models=Models())).preflight(["gpt-5.6-terra"])
        try:
            AnthropicProvider(NS(models=Models())).preflight(["claude-sonnet-5"])
        except providers.ProviderError as error:
            assert error.category == "authentication" and error.provider == "anthropic"
            assert str(error).endswith(
                "Set ANTHROPIC_API_KEY in the shell this experiment is launched from."), str(error)
        else:
            raise AssertionError("Anthropic started without its key")
        os.environ.pop("OPENAI_API_KEY")
        try:
            OpenAIProvider(NS(models=Models())).preflight(["gpt-5.6-terra"])
        except providers.ProviderError as error:
            assert "Set OPENAI_API_KEY in the shell" in str(error), str(error)
        else:
            raise AssertionError("OpenAI started without its key")

    with swapped(os.environ, ANTHROPIC_API_KEY="test", OPENAI_API_KEY="test"):
        for build, variable, model in ((AnthropicProvider, "ANTHROPIC_API_KEY", "claude-sonnet-5"),
                                       (OpenAIProvider, "OPENAI_API_KEY", "gpt-5.6-terra")):
            for error, category, hinted in ((AuthenticationError(), "authentication", True),
                                            (PermissionDeniedError(), "authentication", False),
                                            (APIConnectionError(), "retryable_api", False)):
                try:
                    build(NS(models=Raising(error))).preflight([model])
                except providers.ProviderError as failure:
                    assert failure.category == category, (variable, failure.as_dict())
                    assert failure.status_code == getattr(error, "status_code", None)
                    hint = f"Set {variable} in the shell this experiment is launched from."
                    assert str(failure).endswith(hint) == hinted, str(failure)
                    assert hinted or str(failure) == f"{type(error).__name__}: {error}", \
                        str(failure)
                else:
                    raise AssertionError(f"{variable}: preflight passed {error!r}")


def check_start_refuses_a_missing_key_before_any_client_is_built():
    """An experiment launched without a seat's key is told which key and where it goes.

    The refusal comes before either SDK builds a client, since one may refuse a
    missing key itself, in words that say nothing of the shell. A start that refuses
    installs none of the settings it read: the tool table it was given included.
    """
    unset = dict.fromkeys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL",
                           "OPENAI_BASE_URL"))
    sdks = {name: unbuildable(name) for name in ("anthropic", "openai")}
    for provider, model in (("anthropic", "claude-sonnet-5"), ("openai", "gpt-5.6-terra")):
        variable = f"{provider.upper()}_API_KEY"
        with swapped(os.environ, **unset), swapped(sys.modules, **sdks), pinned(), \
                tempfile.TemporaryDirectory() as folder, quiet() as out:
            amend(root=Path(folder))
            before = harness.SETTINGS
            try:
                harness.start(requirements=[(provider, model)],
                              tool_tables=[{"name": "bash", "kind": "bash"}])
            except SystemExit as error:
                assert error.code == 2, error.code
            else:
                raise AssertionError(f"{provider} started without {variable}")
            assert harness.SETTINGS is before, "a refused start leaves the settings as it found them"
        refusal = out.getvalue().strip().splitlines()[-1]
        assert refusal == (f"{provider} preflight failed: {variable} is not set. Set {variable} "
                           "in the shell this experiment is launched from."), refusal


def check_custom_provider_endpoints_are_refused():
    for variable, build in (("ANTHROPIC_BASE_URL", lambda: AnthropicProvider(NS())),
                            ("OPENAI_BASE_URL", lambda: OpenAIProvider(NS()))):
        with swapped(os.environ, **{variable: "https://proxy.invalid"}):
            try:
                build()
            except providers.ProviderConfigurationError:
                pass
            else:
                raise AssertionError(f"accepted {variable}")


def check_version_three_accounts_are_refused():
    with temp_root() as root:
        path = root / "records" / "old" / "account.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"agent": "old", "model": "claude-sonnet-5"}), encoding="utf-8")
        try:
            __import__("harness").load_account("old", provider="anthropic", model="claude-sonnet-5")
        except SystemExit as error:
            assert "version-3" in str(error) and "fresh agent id" in str(error)
        else:
            raise AssertionError("accepted a version-3 account")


def check_each_adapter_declares_the_provenance_its_traces_record():
    """A trace's provider record is the name, the adapter's own facts, and the model spec.

    The facts are declared once, on the adapter class, and pinned here as a reader of
    earlier traces knows them: a change to an adapter's is a change to every trace.
    """
    declared = {
        "anthropic": {"adapter": "anthropic-messages-v1", "endpoint": "first-party"},
        "openai": {"adapter": "openai-responses-v1", "endpoint": "first-party", "store": False,
                   "reasoning_state": "encrypted"},
        "human": {"adapter": "interaction-store-v1", "endpoint": "local"},
    }
    assert list(providers.FACTORIES) == list(declared), list(providers.FACTORIES)
    for name, factory in providers.FACTORIES.items():
        assert providers.CATALOGS[name] is factory.models, f"{name}: the catalog is the adapter's"
        for model, spec in factory.models.items():
            record = providers.provenance(name, model)
            assert list(record) == ["name", *declared[name], "model_spec"], (name, list(record))
            assert record == {"name": name, **declared[name], "model_spec": spec.as_dict()}, record
            assert (record["model_spec"]["provider"], record["model_spec"]["name"]) == (name, model)
    assert [name for name in providers.FACTORIES if providers.is_interactive(name)] == ["human"], \
        "a person answers the human seat's turns, and no other's"


def check_both_adapters_classify_a_failure_alike_and_retry_a_lost_connection():
    """A request that got no answer is retried like a 429, whichever adapter sent it.

    The SDKs raise APIConnectionError and APITimeoutError with no status code. Read
    by status alone they are the adapter's fault, and one dropped connection ends a
    billed episode as harness_error on its first attempt.
    """
    cases = ((APIConnectionError(), "retryable_api"), (APITimeoutError(), "retryable_api"),
             (ConnectionResetError(), "retryable_api"), (TimeoutError(), "retryable_api"),
             (APIStatusError(429), "retryable_api"), (APIStatusError(503), "retryable_api"),
             (APIStatusError(400), "permanent_api"), (APIStatusError(404), "permanent_api"),
             (APIStatusError(401), "authentication"), (APIStatusError(403), "authentication"),
             (AuthenticationError(), "authentication"), (APIError("unreadable"), "adapter"),
             (ValueError("unreadable"), "adapter"),
             (statusless("AuthenticationError"), "authentication"),
             (statusless("PermissionDeniedError"), "authentication"))
    sessions = {
        "anthropic": lambda raising: AnthropicProvider(NS(messages=raising)).open_session(
            "claude-sonnet-5", "", TOOLS, 1, CONTEXT),
        "openai": lambda raising: OpenAIProvider(NS(responses=raising)).open_session(
            "gpt-5.6-terra", "", TOOLS, 1, CONTEXT),
    }
    for error, category in cases:
        for name, session in sessions.items():
            try:
                session(Raising(error)).request("x")
            except providers.ProviderError as failure:
                assert (failure.provider, failure.category) == (name, category), \
                    (name, type(error).__name__, failure.as_dict())
                assert failure.status_code == getattr(error, "status_code", None)
                assert failure.native_type == type(error).__name__ and failure.__cause__ is error
                # What a trace's provider_error says of a failure mid-episode: the
                # SDK's words and no more, the key having been accepted to get there.
                assert str(failure) == f"{type(error).__name__}: {error}", str(failure)
            else:
                raise AssertionError(f"{name} returned a response for {error!r}")

    # The retry is the harness's: a lost connection is tried again, and one that
    # outlasts every attempt ends the episode as the API's failure.
    with temp_root():
        t = episode_once(classify_error(APIConnectionError(), "anthropic"), *DEFAULT)
    assert t["stop"] == "end_turn", t["stop"]
    assert t["retries"] == [{"attempt": 1, "provider": "anthropic", "error": "ProviderError",
                             "category": "retryable_api", "status": None}], t["retries"]
    with temp_root():
        t = episode_once(*(classify_error(APITimeoutError(), "anthropic")
                           for _ in range(harness.RETRY_ATTEMPTS)))
    assert t["stop"] == "api_error", t["stop"]
    assert len(t["retries"]) == harness.RETRY_ATTEMPTS - 1, t["retries"]
    assert t["provider_error"]["native_type"] == "APITimeoutError", t["provider_error"]


def check_openai_normalize_names_a_refusal_a_truncation_and_an_unknown_stop():
    """Every way a Responses turn can end reaches its canonical stop.

    A refusal part is a refusal, a response cut at max_output_tokens is max_tokens,
    one neither completed nor cut there is other, and function arguments that are
    not a JSON object are the adapter's failure and never a tool call.
    """
    def response(output, status="completed", incomplete=None):
        return NS(id="r", model="gpt-5.6-terra", status=status, incomplete_details=incomplete,
                  output=output, usage=NS(input_tokens=10, output_tokens=5,
                                          input_tokens_details=NS(cached_tokens=0,
                                                                  cache_write_tokens=0),
                                          output_tokens_details=NS(reasoning_tokens=0)))

    def message(*parts):
        return NS(type="message", content=list(parts))

    refused = normalize_openai(response([message(
        NS(type="output_text", text="partial"),
        NS(type="refusal", refusal="I can't help with that."))]), "gpt-5.6-terra")
    assert refused.stop_reason == "refusal", refused.stop_reason
    assert refused.refusal and refused.refusal.explanation == "I can't help with that.", refused.refusal
    assert refused.text == ("partial",) and refused.charges, "a refusal that wrote text is billed"

    cut = normalize_openai(response([message(NS(type="output_text", text="half"))], "incomplete",
                                    NS(reason="max_output_tokens")), "gpt-5.6-terra")
    assert (cut.stop_reason, cut.native_stop_reason) == ("max_tokens", "max_output_tokens"), cut
    assert cut.native_stop_details == {"reason": "max_output_tokens"}, cut.native_stop_details
    assert cut.refusal is None

    filtered = normalize_openai(response([], "incomplete", NS(reason="content_filter")),
                                "gpt-5.6-terra")
    assert (filtered.stop_reason, filtered.native_stop_reason) == ("other", "content_filter")
    pending = normalize_openai(response([], "in_progress"), "gpt-5.6-terra")
    assert (pending.stop_reason, pending.native_stop_reason) == ("other", "in_progress"), pending

    for arguments in ("[1]", "{"):
        call = NS(type="function_call", call_id="c", name="bash", arguments=arguments)
        try:
            normalize_openai(response([call]), "gpt-5.6-terra")
        except providers.ProviderError as error:
            assert (error.provider, error.category) == ("openai", "adapter"), error.as_dict()
        else:
            raise AssertionError(f"arguments {arguments!r} became a tool call")
