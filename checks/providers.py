"""Provider contracts, wire shapes, pricing, and mixed-seat configuration."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import experiment
import providers
from providers import ToolResult, ToolSpec
from providers.anthropic import AnthropicProvider, normalize as normalize_anthropic
from providers.openai import OpenAIProvider, normalize as normalize_openai
from checks.fake import per_agent, say
from checks.lanes import quiet, temp_root


TOOLS = (ToolSpec("bash", "run", {"type": "object", "properties": {"command": {"type": "string"}},
                                  "required": ["command"], "additionalProperties": False}),)


class Messages:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def create(self, **params):
        self.sent.append(params)
        return self.response


class Responses(Messages):
    pass


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
        "claude-sonnet-5", "system", TOOLS, 123)
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
    output = [NS(type="reasoning", summary=[NS(text="thought")], encrypted_content="cipher"),
              NS(type="function_call", id="fc1", call_id="call1", name="bash",
                 arguments='{"command":"pwd"}')]
    response = NS(id="r1", model="gpt-5.6-terra-20260908", status="completed",
                  incomplete_details=None, output=output,
                  usage=NS(input_tokens=20, output_tokens=5,
                           input_tokens_details=NS(cached_tokens=7, cache_write_tokens=2),
                           output_tokens_details=NS(reasoning_tokens=3)),
                  model_dump=lambda: {"id": "r1", "model": "gpt-5.6-terra-20260908",
                                      "status": "completed", "output": [], "usage": {}})
    responses = Responses(response)
    session = OpenAIProvider(NS(responses=responses)).open_session(
        "gpt-5.6-terra", "system", TOOLS, 321)
    turn = session.request("hello").normalize()
    session.request((ToolResult("call1", "ok"),))
    sent = responses.sent
    assert sent[0]["store"] is False
    assert sent[0]["include"] == ["reasoning.encrypted_content"]
    assert sent[0]["tools"] == [{"type": "function", "name": "bash", "description": "run",
                                  "parameters": TOOLS[0].input_schema, "strict": True}]
    assert sent[1]["input"][-1] == {"type": "function_call_output", "call_id": "call1",
                                     "output": "ok"}
    assert any(item.get("type") == "reasoning" for item in sent[1]["input"])
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
    AnthropicProvider(NS(messages=am)).open_session("claude-sonnet-5", "", TOOLS, 1).request("x")
    OpenAIProvider(NS(responses=om)).open_session("gpt-5.6-terra", "", TOOLS, 1).request("x")
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
    class Models:
        def retrieve(self, *args, **kwargs):
            return NS(id=args[0] if args else kwargs.get("model_id"))
    old = {name: os.environ.get(name) for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    try:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["OPENAI_API_KEY"] = "test"
        OpenAIProvider(NS(models=Models())).preflight(["gpt-5.6-terra"])
        try:
            AnthropicProvider(NS(models=Models())).preflight(["claude-sonnet-5"])
        except providers.ProviderError as error:
            assert error.category == "authentication"
        else:
            raise AssertionError("Anthropic started without its key")
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_custom_provider_endpoints_are_refused():
    for variable, build in (("ANTHROPIC_BASE_URL", lambda: AnthropicProvider(NS())),
                            ("OPENAI_BASE_URL", lambda: OpenAIProvider(NS()))):
        old = os.environ.get(variable)
        os.environ[variable] = "https://proxy.invalid"
        try:
            try:
                build()
            except providers.ProviderConfigurationError:
                pass
            else:
                raise AssertionError(f"accepted {variable}")
        finally:
            if old is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = old


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
