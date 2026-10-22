from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import telemetry
from benchmark.metrics import ModelMetrics
from catalog import grouping
from catalog.loader import load_catalog, resolve_model, routable_models, slug, unique_prefix
from catalog.rate_limits import (
    effective_rpm,
    parse_rate_limit,
    parse_tokens,
    quota_cost,
    rpm_from_tpm,
)
from catalog.schema import ModelEntry, Provider
from catalog.task_types import normalize_modality
from providers.cloudflare import CloudflareAdapter, normalize_cf
from providers.huggingface import HF_BASE, HuggingFaceAdapter
from providers.ollama import OllamaAdapter, normalize_ollama_response, parse_ollama_body, to_ollama_payload
from providers.openai_adapter import OpenAIAdapter, normalize_openai_response
from providers.registry import ADAPTERS, adapter_for
from routing.budget import Budget, QuotaLedger
from routing.constraints import ANONYMOUS_PROVIDERS, apply_constraints, is_local, model_capabilities, provider_index
from routing.decision import RouteDecision
from routing.health import HealthTracker, percentile
from routing.policies import POLICIES, RoutingContext, classify
from routing.request import RoutingRequest
from routing.scoring import DEFAULT_WEIGHTS, normalize, normalize_weights, rank, weighted_score

CATALOG = load_catalog()
MODEL = {m.key: m for m in routable_models(CATALOG)}

# Catalog models whose modality and rate limit make the fixture deterministic.
TEXT_A = "aion_labs/aion-labs/aion-rp-llama-3.1-8b"
TEXT_B = "cloudflare_workers_ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CODE_C = "mistral_ai/codestral-2508"
POOL = [MODEL[TEXT_A], MODEL[TEXT_B], MODEL[CODE_C]]

# Every rate-limit string in catalog/data.json, split into what parses and what is opaque.
PARSED_RATE_LIMITS = {
    "1 concurrent request": {"concurrent": 1},
    "1,000 RPM, 50,000 TPM": {"rpm": 1000, "tpm": 50000},
    "10K neurons/day (shared)": {"neurons_per_day": 10000},
    "15 RPM, 1,500 RPD": {"rpm": 15, "rpd": 1500},
    "15 RPM, 20K TPD": {"rpm": 15, "tpd": 20000},
    "2 RPM (anonymous)": {"rpm": 2},
    "2,000 RPD total; <=500 RPD/model (dynamic)": {"rpd": 2000},
    "20 RPM": {"rpm": 20},
    "20 RPM, 50 RPD": {"rpm": 20, "rpd": 50},
    "200 req/hr": {"rph": 200},
    "30 RPM, 1,000 RPD": {"rpm": 30, "rpd": 1000},
    "30 RPM, 1,500 RPD": {"rpm": 30, "rpd": 1500},
    "40 RPM, 10,000 RPD": {"rpm": 40, "rpd": 10000},
    "5 RPM, 50 RPD": {"rpm": 5, "rpd": 50},
    "60 RPM, 250 req/hr (free token)": {"rpm": 60, "rph": 250},
    "~1 RPS, 500K TPM": {"rps": 1, "tpm": 500000},
}
OPAQUE_RATE_LIMITS = {
    "—",
    "Credit-metered",
    "Dynamic quotas + dynamic concurrency",
    "Session/weekly limits (unpublished)",
}
EXPECTED_MODALITY = {
    "Code": {"code"},
    "LLM, MLLM": {"text"},
    "Multimodal": {"vision"},
    "Text": {"text"},
    "Text (code)": {"code"},
    "Text (reasoning)": {"reasoning"},
    "Text + Image": {"vision"},
    "Text + Image + Audio": {"audio", "vision"},
    "Text + Image + Audio + Video": {"audio", "vision"},
    "Text + Image + Code": {"code", "vision"},
    "Text + Image + Video": {"vision"},
    "Text + Vision": {"vision"},
    "Text / Image": {"vision"},
    "Text, Image": {"vision"},
    "Text, Image, Audio, Embeddings": {"audio", "embeddings", "vision"},
    "Text, Image, Video, Speech, Embeddings": {"audio", "embeddings", "vision"},
}


def catalog_values(field: str) -> set:
    return {getattr(m, field) for p in CATALOG.providers for m in p.models if getattr(m, field)}


def fields(limits) -> dict:
    """Non-None parsed fields, so the assertion only covers what was actually read."""
    return {k: v for k, v in limits.__dict__.items() if k not in {"parsed", "raw"} and v is not None}


def entry(provider: str, name: str, modality: str = "Text", rate_limit: str = "",
          context: str = "32K", max_output: str = "32K") -> ModelEntry:
    model = ModelEntry(
        id=name, name=name, modality=modality, rateLimit=rate_limit,
        context=context, maxOutput=max_output, provider=provider,
    )
    model.key = f"{provider}/{name}"
    return model


def metrics(key: str, pass_rate: float, scored_n: int, p50: float, error_rate: float) -> ModelMetrics:
    provider, model = key.split("/", 1)
    return ModelMetrics(
        key=key, provider=provider, model=model, pass_rate=pass_rate,
        scored_n=scored_n, attempts=scored_n, p50_ms=p50, error_rate=error_rate,
    )


def fixture_context() -> RoutingContext:
    return RoutingContext(
        keys={"aion_labs": "k", "cloudflare_workers_ai": "k", "mistral_ai": "k"},
        metrics={
            TEXT_A: metrics(TEXT_A, 0.60, 6, 1200, 0.20),
            TEXT_B: metrics(TEXT_B, 0.80, 8, 400, 0.00),
            CODE_C: metrics(CODE_C, 0.90, 5, 300, 0.00),
        },
    )


# ---------------------------------------------------------------- rate limits

def test_parse_rate_limit_against_every_real_catalog_string():
    values = catalog_values("rateLimit")
    assert values == set(PARSED_RATE_LIMITS) | set(OPAQUE_RATE_LIMITS)

    for raw, expected in PARSED_RATE_LIMITS.items():
        limits = parse_rate_limit(raw)
        assert limits.parsed is True, raw
        assert limits.raw == raw
        assert fields(limits) == expected, raw

    for raw in OPAQUE_RATE_LIMITS:
        limits = parse_rate_limit(raw)
        assert limits.parsed is False, raw
        assert fields(limits) == {}, raw


def test_parse_rate_limit_edge_cases():
    assert parse_rate_limit(None).raw == ""
    assert parse_rate_limit(None).parsed is False
    assert parse_rate_limit("").parsed is False
    # The first value wins per unit, so '<=500 RPD/model' does not overwrite the total.
    assert parse_rate_limit("2,000 RPD total; <=500 RPD/model (dynamic)").rpd == 2000
    assert parse_rate_limit("60 RPM, 250 req/hr").rph == 250
    assert parse_rate_limit("10K neurons/day").neurons_per_day == 10000
    assert parse_rate_limit("1 concurrent request").concurrent == 1
    assert parse_rate_limit("nonsense").parsed is False


def test_parse_tokens_and_effective_rpm():
    assert parse_tokens("128K") == 128_000
    assert parse_tokens("1M") == 1_000_000
    assert parse_tokens("131072") == 131072
    assert parse_tokens("Shared w/ context") is None
    assert parse_tokens(None) is None

    # The runner caps at 5 rpm even when the published limit is higher.
    assert effective_rpm(parse_rate_limit("1,000 RPM, 50,000 TPM")) == 5
    assert effective_rpm(parse_rate_limit("2 RPM (anonymous)")) == 2
    assert effective_rpm(parse_rate_limit("~1 RPS, 500K TPM")) == 5
    assert effective_rpm(parse_rate_limit("200 req/hr")) == 3
    assert effective_rpm(parse_rate_limit("1 concurrent request")) == 1
    assert effective_rpm(parse_rate_limit("Credit-metered")) == 5
    assert effective_rpm(parse_rate_limit("60 RPM, 250 req/hr"), cap=2) == 2

    assert rpm_from_tpm(500_000, 512) == 976
    assert rpm_from_tpm(None, 512) is None
    assert rpm_from_tpm(500, 0) is None


def test_quota_cost_is_quota_consumption_and_none_when_unmeasurable():
    assert quota_cost(parse_rate_limit("15 RPM, 20K TPD"), 1, 512) == 512 / 20000
    assert quota_cost(parse_rate_limit("10K neurons/day (shared)"), 1, 512) == 512 / 10000
    assert quota_cost(parse_rate_limit("~1 RPS, 500K TPM"), 1, 512) == 512 / (500000 * 1440)
    # req/hr is a rate, so the daily share is the hourly rate times 24.
    assert quota_cost(parse_rate_limit("200 req/hr"), 1, 512) == 1 / (200 * 24)
    # Nothing measurable -> None, never a fabricated cost.
    assert quota_cost(parse_rate_limit("Credit-metered"), 1, 512) is None
    assert quota_cost(parse_rate_limit("—"), 1, 512) is None
    # A request larger than the daily budget is capped at the whole budget.
    assert quota_cost(parse_rate_limit("20 RPM, 50 RPD"), 60, 512) == 1.0


# -------------------------------------------------------------- task types

def test_normalize_modality_against_every_catalog_value():
    values = catalog_values("modality")
    assert values == set(EXPECTED_MODALITY)
    for raw, expected in EXPECTED_MODALITY.items():
        assert normalize_modality(raw) == expected, raw
    # No modality means the text group, not an empty group.
    assert normalize_modality(None) == {"text"}
    assert normalize_modality("") == {"text"}
    # Rules are additive: a model can belong to several groups.
    assert normalize_modality("Text + Image + Audio") == {"audio", "vision"}
    assert normalize_modality("Code") == {"code"}


def test_classify_intent():
    assert classify("Why does this fail? Analyze step by step") == "reasoning"
    assert classify("Write a python function to parse a regex") == "code"
    assert classify("Embed this sentence into a vector") == "embeddings"
    assert classify("Rank the passages by relevance") == "rerank"
    assert classify("What is in this screenshot?") == "vision"
    assert classify("Transcribe the audio") == "audio"
    assert classify("Stay in character as the wizard") == "roleplay"
    assert classify("Summarize this paragraph") == "text"
    assert classify("") == "text"
    # First match wins, so the earlier rule takes precedence.
    assert classify("Analyze the code") == "reasoning"


# -------------------------------------------------------------- catalog

def test_resolve_model_accepts_the_documented_forms():
    ovh = MODEL["ovhcloud_ai_endpoints/gpt-oss-20b"]
    assert resolve_model(CATALOG, "ovhcloud_ai_endpoints/gpt-oss-20b") == [ovh]
    # Short provider names resolve when the prefix is unique.
    assert resolve_model(CATALOG, "ovhcloud/gpt-oss-20b") == [ovh]
    assert resolve_model(CATALOG, "gpt-oss-20b") == [ovh]
    assert resolve_model(CATALOG, ovh.name.lower()) == [ovh]
    # Model ids may contain slashes: only a known provider slug is treated as a prefix.
    assert resolve_model(CATALOG, "aion-labs/aion-rp-llama-3.1-8b") == [MODEL[TEXT_A]]
    assert resolve_model(CATALOG, "auto") == []
    assert resolve_model(CATALOG, "auto:text") == []
    assert resolve_model(CATALOG, "") == []
    assert resolve_model(CATALOG, "nope/nope") == []
    assert unique_prefix("ovhcloud", {slug(p.name) for p in CATALOG.providers}) == "ovhcloud_ai_endpoints"


def test_group_and_counts_match_the_catalog():
    grouped = grouping.group(CATALOG)
    assert set(grouped) == {"provider_api", "inference_provider"}

    total = 0
    for category, inner in grouped.items():
        for task_type, models in inner.items():
            assert task_type in grouping.task_type_order()
            assert models == sorted(models, key=lambda m: m.key)
            for model in models:
                assert not model.placeholder
                assert model.provider in {slug(p.name) for p in CATALOG.providers}
                assert task_type in normalize_modality(model.modality)
            total += len(models)

    # A model with several task types appears in each matching group.
    multi = MODEL["ovhcloud_ai_endpoints/Qwen3.5-397B-A17B"]
    assert multi in grouped["inference_provider"]["vision"]
    counts = grouping.group_counts(grouped)
    assert sum(sum(inner.values()) for inner in counts.values()) == total
    assert grouping.render_table(grouped).startswith("## inference_provider")

    with pytest.raises(ValueError):
        grouping.group(CATALOG, primary="provider")


# -------------------------------------------------------------- constraints

def test_apply_constraints_records_a_reason_for_every_exclusion():
    ctx = fixture_context()
    unknown = entry("not_a_provider", "ghost")
    placeholder = ModelEntry(name="+ 3 more models", modality="Text")
    placeholder.key = "groq/+ 3 more models"

    candidates = [
        placeholder,
        unknown,
        entry("groq", "no-key-model", modality="Text + Code"),
        entry("mistral_ai", "text-only", modality="Text"),
        entry("mistral_ai", "plain code", modality="Text + Code"),
        entry("mistral_ai", "json mode small context", modality="Text + Code", context="8K"),
        entry("mistral_ai", "json mode small output", modality="Text + Code", max_output="1K"),
        entry("mistral_ai", "json mode", modality="Text + Code"),
    ]
    req = RoutingRequest(
        task_type="code",
        require=frozenset({"json"}),
        input_tokens=10_000,
        output_tokens=2_000,
    )
    kept, excluded = apply_constraints(candidates, req, {"mistral_ai": "k"}, HealthTracker(), None)

    assert kept == [entry("mistral_ai", "json mode", modality="Text + Code")]
    assert excluded == {
        "groq/+ 3 more models": "unknown",
        "not_a_provider/ghost": "unknown",
        "groq/no-key-model": "no_key",
        "mistral_ai/text-only": "task_type",
        "mistral_ai/plain code": "capability",
        "mistral_ai/json mode small context": "context",
        "mistral_ai/json mode small output": "capability",
    }


def test_constraint_order_and_availability():
    ctx = fixture_context()
    model = entry("mistral_ai", "code-model", modality="Code")

    # The key check runs before the capability check, so the reported reason is no_key.
    kept, excluded = apply_constraints([model], RoutingRequest(task_type="code", require=frozenset({"json"})),
                                       {}, HealthTracker(), None)
    assert kept == [] and excluded[model.key] == "no_key"

    # Anonymous tiers are never excluded for a missing key.
    anonymous = entry("ovhcloud_ai_endpoints", "anonymous-model", modality="Text")
    kept, excluded = apply_constraints([anonymous], RoutingRequest(task_type="text"), {}, HealthTracker(), None)
    assert kept == [anonymous] and excluded == {}
    assert "ovhcloud_ai_endpoints" in ANONYMOUS_PROVIDERS

    # Cooldown is a hard constraint.
    health = HealthTracker()
    health.record("mistral_ai", "error", 100, retry_after=30)
    kept, excluded = apply_constraints([model], RoutingRequest(task_type="code"), {"mistral_ai": "k"}, health, None)
    assert kept == [] and excluded[model.key] == "health"

    # Per-provider request budget.
    ledger = QuotaLedger(Budget(per_provider_max_requests=1))
    ledger.charge("mistral_ai", model.id, 1, 10)
    kept, excluded = apply_constraints([model], RoutingRequest(task_type="code"), {"mistral_ai": "k"},
                                       HealthTracker(), ledger)
    assert kept == [] and excluded[model.key] == "budget"
    assert ledger.remaining("mistral_ai")["requests"] == 0

    # Explicitly unavailable provider.
    kept, excluded = apply_constraints([model], RoutingRequest(task_type="code", unavailable=frozenset({"mistral_ai"})),
                                       {"mistral_ai": "k"}, HealthTracker(), None)
    assert kept == [] and excluded[model.key] == "unavailable"

    # local_only is a hard constraint; the catalog has no self-hosted provider.
    kept, excluded = apply_constraints([model], RoutingRequest(task_type="code", local_only=True),
                                       {"mistral_ai": "k"}, HealthTracker(), None)
    assert kept == [] and excluded[model.key] == "unavailable"


def test_is_local_and_capabilities():
    assert is_local(Provider(name="Local", category="inference_provider", baseUrl="http://localhost:11434/v1"))
    assert is_local(Provider(name="LAN", category="inference_provider", baseUrl="https://192.168.1.5/v1"))
    assert not is_local(Provider(name="Remote", category="inference_provider", baseUrl="https://api.groq.com/openai/v1"))

    code = MODEL[CODE_C]
    assert "code" in model_capabilities(code)
    assert "json" not in model_capabilities(code)
    assert "json" in model_capabilities(entry("groq", "json mode", modality="Text"))
    assert "tools" in model_capabilities(entry("groq", "tool calling", modality="Text"))


# -------------------------------------------------------------- policies

def test_each_policy_on_the_fixture_pool():
    ctx = fixture_context()
    req = RoutingRequest(task_type="text", messages=({"role": "user", "content": "summarize this"},))

    expected = {
        "explicit": (TEXT_B, "explicit"),
        "semantic": (TEXT_B, "quality"),
        "quality": (TEXT_B, "quality"),
        "cheapest": (TEXT_A, "cost"),
        "weighted": (TEXT_B, "quality"),
        "hybrid": (TEXT_B, "quality"),
        "cascade": (TEXT_A, "cascade"),
    }
    for name, router in POLICIES.items():
        req = RoutingRequest(
            task_type="text",
            messages=({"role": "user", "content": "summarize this"},),
            policy=name,
        )
        decision = router.route(req, POOL, ctx)
        assert decision.policy == name
        assert decision.key == expected[name][0], name
        assert decision.reason == expected[name][1], name
        # The code-only model is excluded before scoring, on every policy.
        assert decision.excluded == {CODE_C: "task_type"}, name

    # The cheapest axis is quota consumption, so the score is 1 - cost.
    decision = POLICIES["cheapest"].route(req, POOL, ctx)
    assert decision.score == pytest.approx(1.0 - 512 / 20000, abs=1e-4)


def test_policy_reasons_and_fallbacks():
    ctx = fixture_context()

    # Semantic routing infers the task type from the prompt. The task type is unknown at
    # constraint time, so nothing is recorded as excluded; the group filter decides.
    req = RoutingRequest(messages=({"role": "user", "content": "write a python function"},))
    decision = POLICIES["semantic"].route(req, POOL, ctx)
    assert decision.reason == "quality" and decision.key == CODE_C
    assert decision.excluded == {}
    # Once the task type is known, the same pool excludes the models outside the group.
    decision = POLICIES["semantic"].route(req.with_task_type("code"), POOL, ctx)
    assert decision.key == CODE_C and decision.excluded == {TEXT_A: "task_type", TEXT_B: "task_type"}

    # Nothing measured -> the policy says so instead of inventing a score.
    empty = RoutingContext(keys={"aion_labs": "k", "cloudflare_workers_ai": "k", "mistral_ai": "k"})
    decision = POLICIES["quality"].route(RoutingRequest(task_type="text"), POOL, empty)
    assert decision.reason == "unknown" and decision.score == 0.0
    assert decision.key == TEXT_A

    # Hybrid keeps the cheap model when its measured pass rate clears the task threshold.
    ctx.pass_rate_by_task = {TEXT_A: {"text": 0.75}}
    decision = POLICIES["hybrid"].route(RoutingRequest(task_type="text"), POOL, ctx)
    assert decision.reason == "threshold" and decision.key == TEXT_A
    assert decision.score == 0.75
    # Below the text threshold (0.70) the strong model wins.
    ctx.pass_rate_by_task = {TEXT_A: {"text": 0.65}}
    decision = POLICIES["hybrid"].route(RoutingRequest(task_type="text"), POOL, ctx)
    assert decision.reason == "quality" and decision.key == TEXT_B

    # Cascade orders attempts cheapest-first over the constrained pool.
    decision = POLICIES["cascade"].route(RoutingRequest(task_type="text"), POOL, ctx)
    assert [a.key for a in decision.attempts] == [TEXT_A, TEXT_B]
    assert all(a.reason == "cascade" for a in decision.attempts)

    # A provider whose request budget is spent is dropped before scoring, so the chain stops.
    ledger = QuotaLedger(Budget(per_provider_max_requests=1))
    ledger.charge("cloudflare_workers_ai", "x", 1, 10)
    ctx.ledger = ledger
    decision = POLICIES["cascade"].route(RoutingRequest(task_type="text"), POOL, ctx)
    assert [a.key for a in decision.attempts] == [TEXT_A]
    assert decision.excluded == {CODE_C: "task_type", TEXT_B: "budget"}

    # Every candidate excluded -> no route, and the reasons are reported.
    # no_key is checked before the task type, so the code model is dropped for the key first.
    decision = POLICIES["weighted"].route(RoutingRequest(task_type="text"), POOL, RoutingContext())
    assert decision.provider == "" and decision.reason == "unknown"
    assert decision.excluded == {TEXT_A: "no_key", TEXT_B: "no_key", CODE_C: "no_key"}


def test_scoring_normalisation_and_min_n_gate():
    ctx = fixture_context()
    weights = normalize_weights({"quality": 2, "latency": 1, "bogus": 5, "reliability": -1})
    assert weights == {"quality": 2 / 3, "latency": 1 / 3, "reliability": 0.0}
    assert normalize_weights({}) == DEFAULT_WEIGHTS

    pool = [MODEL[TEXT_A], MODEL[TEXT_B]]
    # Single-axis weights: quality is higher-is-better, cost and latency are inverted.
    assert weighted_score(MODEL[TEXT_A], ctx.metrics, {"quality": 1}, ctx.reliability, pool) == 0.0
    assert weighted_score(MODEL[TEXT_B], ctx.metrics, {"quality": 1}, ctx.reliability, pool) == 1.0
    assert weighted_score(MODEL[TEXT_A], ctx.metrics, {"quota_cost": 1}, ctx.reliability, pool) == 1.0
    assert weighted_score(MODEL[TEXT_B], ctx.metrics, {"quota_cost": 1}, ctx.reliability, pool) == 0.0

    # Axes with no measurement are dropped and their weight redistributed, not scored as 0.
    quiet_a = entry("groq", "quiet a", rate_limit="Credit-metered")
    quiet_b = entry("groq", "quiet b", rate_limit="Credit-metered")
    quiet = {
        quiet_a.key: ModelMetrics(key=quiet_a.key, provider="groq", model="quiet a", pass_rate=0.5, scored_n=5),
        quiet_b.key: ModelMetrics(key=quiet_b.key, provider="groq", model="quiet b", pass_rate=1.0, scored_n=5),
    }
    assert weighted_score(quiet_a, quiet, DEFAULT_WEIGHTS, None, [quiet_a, quiet_b]) == 0.0
    assert weighted_score(quiet_b, quiet, DEFAULT_WEIGHTS, None, [quiet_a, quiet_b]) == 1.0

    # Below the minimum-N gate a model is coverage-only and never ranked.
    ranked = rank(POOL, ctx.metrics, DEFAULT_WEIGHTS, ctx.reliability, min_n=5)
    assert [m.key for m, _ in ranked] == [CODE_C, TEXT_B, TEXT_A]
    assert rank(POOL, ctx.metrics, DEFAULT_WEIGHTS, ctx.reliability, min_n=9) == []

    # Equal values normalise to 1.0 rather than dividing by zero.
    assert normalize({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert normalize({}) == {}


def test_health_percentile_includes_failures():
    assert percentile([100, 200, 300], 0.50) == 200.0
    assert percentile([100], 0.95) == 100.0
    assert percentile([], 0.50) is None

    health = HealthTracker()
    health.record("groq", "pass", 100)
    health.record("groq", "pass", 300)
    health.record("groq", "error", 900)
    assert health.success_rate("groq") == pytest.approx(2 / 3)
    assert health.latencies("groq") == [100, 300, 900]
    assert health.is_available("groq") is True

    snapshot = health.snapshot()
    assert snapshot["groq"]["n"] == 3
    assert snapshot["groq"]["p50_ms"] == 300.0
    assert snapshot["groq"]["last_outcome"] == "error"

    # A 429 with Retry-After puts the provider into cooldown, so it is not routable.
    health.record("groq", "error", 900, retry_after=30, now=0.0)
    assert health.is_available("groq", now=10.0) is False
    assert health.is_available("groq", now=10 ** 12) is True
    assert health.success_rate("unknown-provider") is None

    # Window caps the retained events.
    small = HealthTracker(window=2)
    for outcome in ("pass", "pass", "error"):
        small.record("groq", outcome, 100)
    assert small.success_rate("groq") == 0.5

    with pytest.raises(ValueError):
        health.record("groq", "wrong-outcome", 100)


# -------------------------------------------------------------- telemetry

def test_telemetry_rejects_unknown_values_and_keeps_four_fields(tmp_path):
    log = tmp_path / "routes.jsonl"
    decision = type("D", (), {"key": "groq/model", "reason": "quality"})()
    telemetry.log_route(decision, 120, "pass", path=log)

    row = json.loads(log.read_text(encoding="utf-8"))
    assert row == {"route": "groq/model", "reason": "quality", "latency_ms": 120, "outcome": "pass"}
    assert set(row) == set(telemetry.FIELDS)

    with pytest.raises(ValueError):
        telemetry.log_route(type("D", (), {"key": "x", "reason": "llm-judge"})(), 10, "pass", path=log)
    with pytest.raises(ValueError):
        telemetry.log_route(decision, 10, "hallucinated", path=log)

    log.write_text(log.read_text(encoding="utf-8") + json.dumps(
        {"route": "groq/model", "reason": "vibes", "latency_ms": 10, "outcome": "pass"}
    ) + "\n" + json.dumps({"route": "groq/model", "latency_ms": 10, "outcome": "pass"}) + "\n")

    summary = telemetry.summarize(log, {})
    assert summary["n"] == 1
    assert summary["rejected"] == 2
    assert summary["rejected_rows"][0]["unknown"] == ["vibes", "pass"]
    assert summary["rejected_rows"][1]["missing"] == ["reason"]
    assert summary["by_reason"]["quality"]["n"] == 1
    assert summary["by_reason"]["quality"]["p50_ms"] == 120.0
    assert summary["outcomes"] == {"pass": 1}

    # The decision object enforces the same allowlist, so a bad reason never reaches the log.
    RouteDecision(provider="groq", model_id="m", key="groq/m", reason="quality")
    with pytest.raises(ValueError):
        RouteDecision(provider="groq", model_id="m", key="groq/m", reason="llm-judge")


# -------------------------------------------------------------- adapters

def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_openai_adapter_chat_and_normalise():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "cmpl-1",
            "model": "model-x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
        })

    adapter = OpenAIAdapter("groq", "secret", mock_client(handler))
    result = await adapter.chat("model-x", {"messages": [{"role": "user", "content": "hi"}], "temperature": 1})

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["temperature"] == 1
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "PONG"
    # usage is absent upstream but the shape is still guaranteed.
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert adapter.supports_stream() is True


@pytest.mark.asyncio
async def test_openai_adapter_retries_once_and_classifies_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAIAdapter("groq", "k", mock_client(handler))
    result = await adapter.chat("m", {})
    assert calls["n"] == 2 and result["choices"][0]["message"]["content"] == "ok"

    def always_429(request):
        return httpx.Response(429, json={"error": "quota"})

    with pytest.raises(Exception) as exc:
        await OpenAIAdapter("groq", "k", mock_client(always_429)).chat("m", {})
    assert exc.value.error_class == "429"

    with pytest.raises(Exception) as exc:
        await OpenAIAdapter("groq", "k", mock_client(lambda request: httpx.Response(401, json={"error": "bad key"}))).chat("m", {})
    assert exc.value.error_class == "auth"

    with pytest.raises(Exception) as exc:
        await OpenAIAdapter("groq", "k", mock_client(lambda request: httpx.Response(503, json={"error": "down"}))).chat("m", {})
    assert exc.value.error_class == "server"

    def timeout(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(Exception) as exc:
        await OpenAIAdapter("groq", "k", mock_client(timeout)).chat("m", {})
    assert exc.value.error_class == "timeout" and exc.value.status is None


@pytest.mark.asyncio
async def test_openai_adapter_streaming_passes_reasoning_content_through():
    sse = (
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]}) + "\n\n"
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "PONG"}}]}) + "\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request):
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, content=sse.encode())

    adapter = OpenAIAdapter("groq", "k", mock_client(handler))
    chunks = [chunk async for chunk in await adapter.chat("m", {}, stream=True)]
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "thinking"

    def error(request):
        return httpx.Response(429, content=b"rate limited")

    with pytest.raises(Exception) as exc:
        async for _ in await OpenAIAdapter("groq", "k", mock_client(error)).chat("m", {}, stream=True):
            pass
    assert exc.value.error_class == "429"


@pytest.mark.asyncio
async def test_openai_adapter_embeddings_and_list_models():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": {"total_tokens": 3}})

    adapter = OpenAIAdapter("groq", "k", mock_client(handler))
    embeddings = await adapter.embeddings("m1", {"input": "text"})
    assert embeddings["data"][0]["embedding"] == [0.1, 0.2]
    assert embeddings["usage"] == {"total_tokens": 3}
    assert await adapter.list_models() == ["m1", "m2"]


@pytest.mark.asyncio
async def test_cloudflare_adapter_url_and_normalise(monkeypatch):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    adapter = CloudflareAdapter("cloudflare_workers_ai", "k", mock_client(lambda request: httpx.Response(200, json={})))
    with pytest.raises(Exception) as exc:
        adapter.url("chat/completions")
    assert exc.value.error_class == "auth"

    monkeypatch.setenv("CF_ACCOUNT_ID", "acct-1")

    def handler(request):
        assert "accounts/acct-1/ai/run" in str(request.url)
        return httpx.Response(200, json={"result": {"model": "cf-model", "choices": [{"message": {"content": "ok"}}]}})

    adapter = CloudflareAdapter("cloudflare_workers_ai", "k", mock_client(handler))
    result = await adapter.chat("cf-model", {})
    assert result["choices"][0]["message"]["content"] == "ok"
    assert result["model"] == "cf-model"

    wrapped = normalize_cf({"id": "x", "result": {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}})
    assert wrapped["usage"]["total_tokens"] == 5
    assert wrapped["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_ollama_adapter_merges_ndjson_chunks():
    body = "\n".join([
        json.dumps({"model": "llama", "message": {"role": "assistant", "content": "PO"}}),
        json.dumps({"model": "llama", "message": {"role": "assistant", "content": "NG"}}),
        json.dumps({"model": "llama", "message": {"role": "assistant", "content": ""},
                    "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 5}),
    ])
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=body)

    adapter = OllamaAdapter("ollama_cloud", "k", mock_client(handler))
    result = await adapter.chat(
        "llama", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20, "temperature": 0.5}
    )

    assert adapter.supports_stream() is False
    assert seen["url"] == "https://ollama.com/api/chat"
    assert seen["body"]["options"] == {"num_predict": 20, "temperature": 0.5}
    assert result["choices"][0]["message"]["content"] == "PONG"
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}

    # A single JSON body still parses through the same hook.
    assert parse_ollama_body(json.dumps({"message": {"content": "ok"}, "eval_count": 1}))["message"]["content"] == "ok"
    assert parse_ollama_body("") == {}
    assert to_ollama_payload({"messages": [{"role": "user", "content": "hi"}]}) == {"messages": [{"role": "user", "content": "hi"}]}

    def tags(request):
        return httpx.Response(200, json={"models": [{"model": "beam"}, {"model": "gemma4:31b"}]})

    assert await OllamaAdapter("ollama_cloud", "k", mock_client(tags)).list_models() == ["beam", "gemma4:31b"]


@pytest.mark.asyncio
async def test_huggingface_base_and_registry_dispatch():
    client = mock_client(lambda request: httpx.Response(200, json={}))
    assert adapter_for("hugging_face", "k", client).base == HF_BASE
    assert isinstance(adapter_for("hugging_face", "k", client), HuggingFaceAdapter)
    assert isinstance(adapter_for("ollama_cloud", "k", client), OllamaAdapter)
    assert isinstance(adapter_for("cloudflare_workers_ai", "k", client), CloudflareAdapter)
    # Everything not in the registry is OpenAI-compatible.
    assert isinstance(adapter_for("groq", "k", client), OpenAIAdapter)
    # Per-model host override wins over the catalog baseUrl.
    assert adapter_for("groq", "k", client, base="https://mirror.example/v1/").base == "https://mirror.example/v1"
    assert set(ADAPTERS) == {"cloudflare_workers_ai", "ollama_cloud", "hugging_face"}


def test_normalize_openai_response_shape():
    raw = {"id": "1", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                 "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    out = normalize_openai_response(raw)
    assert out["usage"]["total_tokens"] == 15
    # Providers that answer with a plain 'text' field instead of a message.
    assert normalize_openai_response({"choices": [{"text": "hi"}]})["choices"][0]["message"]["content"] == "hi"
    assert normalize_openai_response({})["choices"] == []
