from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from benchmark import report
from benchmark.checks import PROBE_PROMPT, Task, load_tasks
from benchmark.metrics import aggregate, pareto_frontier
from benchmark.runner import read_rows, run_one
from catalog.loader import load_catalog, routable_models, resolve_model, slug
from config import env_keys, load_config, save_config
from providers.base import ProviderError
from providers.registry import adapter_for
from request_log import MAX_TEXT, LogEntry, RequestLog, derive_metrics, now_iso
from routing.constraints import apply_constraints, provider_index
from routing.decision import RouteDecision
from routing.health import HealthTracker
from routing.policies import POLICIES, RoutingContext, classify, decide
from routing.request import RoutingRequest
from telemetry import log_route

ROOT = Path(__file__).parent
TEMPLATES_DIR = ROOT / "web" / "templates"
STATIC_DIR = ROOT / "web" / "static"
RESULTS_DIR = ROOT / "results"
TASKS_PATH = ROOT / "benchmark" / "tasks.json"

PROBE_MAX_TOKENS = 16
PROBE_TIMEOUT = 30.0

CONFIG = load_config()
LOG = RequestLog()
ROUTE_LOG = Path("routes.jsonl")
CTX = RoutingContext(keys=env_keys(), health=HealthTracker())
CLIENT = httpx.AsyncClient(timeout=60.0)

app = FastAPI(title="LLM Router")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def fmt(value) -> str:
    """Unmeasured values render as 'unknown', never as a guess."""
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def pct(value) -> str:
    return "unknown" if value is None else f"{value * 100:.1f}%"


templates.env.globals["fmt"] = fmt
templates.env.globals["pct"] = pct


def load_scores() -> dict:
    path = RESULTS_DIR / "scores.json"
    return report.load_scores(path) if path.exists() else {}


CTX.metrics = load_scores()


def config_keys() -> dict[str, str]:
    """Env keys are the default; runtime keys set from the UI override them."""
    keys = dict(env_keys())
    keys.update({p: v for p, v in CONFIG.provider_keys.items() if v})
    for model in CONFIG.models:
        if model.api_key:
            keys[model.provider] = model.api_key
    if CONFIG.api_key:
        for model in CONFIG.models:
            keys.setdefault(model.provider, CONFIG.api_key)
    return keys


def error_json(status: int, message: str, excluded: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"message": message, "type": "router_error"}}
    if excluded:
        body["error"]["excluded"] = excluded
    return JSONResponse(body, status_code=status)


def usage_of(usage: dict) -> tuple[int, int, int | None]:
    usage = usage or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return prompt, completion, int(cached) if cached else None


def prompt_text(messages: list) -> str:
    """Text parts of the messages; multimodal content is a list of typed parts."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts.append(" ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"))
        else:
            parts.append(str(content or ""))
    return " ".join(p for p in parts if p)


def answer_of(raw: dict) -> str:
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or choices[0].get("text") or "")


def candidates_for(requested: str) -> list:
    catalog = load_catalog()
    if requested and requested != "auto":
        found = resolve_model(catalog, requested)
        if found:
            return found
    flat = [entry for model in CONFIG.selected_models() for entry in resolve_model(catalog, model.model_key)]
    return flat or routable_models(catalog)


def route(prompt: str, requested: str, policy: str = "weighted") -> RouteDecision:
    req = RoutingRequest(model=requested, task_type=classify(prompt), messages=({"role": "user", "content": prompt},), policy=policy)
    candidates = candidates_for(requested)
    ctx = RoutingContext(keys=config_keys(), health=CTX.health, ledger=CTX.ledger, metrics=CTX.metrics, weights=CTX.weights)
    kept, excluded = apply_constraints(candidates, req, ctx.keys, ctx.health, ctx.ledger)

    hit = CONFIG.keyword_match(prompt)
    if hit:
        target = [m for m in kept if m.key == hit.model_key]
        if target:
            return decide(target[0], req, ctx, "explicit", 0.0, excluded)

    router = POLICIES.get(policy, POLICIES["weighted"])
    return router.route(req, candidates, ctx)


def finalize(entry: LogEntry, decision: RouteDecision, usage: dict, ttft_ms: int | None = None) -> LogEntry:
    prompt_tokens, completion_tokens, cached = usage_of(usage)
    entry.reused = cached
    entry.prefill_tps, entry.decode_tps, entry.cache_hit_rate = derive_metrics(
        prompt_tokens, completion_tokens, ttft_ms, entry.duration_ms, cached
    )
    entry.ttft_ms = ttft_ms
    LOG.record(entry)
    log_route(decision, entry.duration_ms or 0, entry.outcome, ROUTE_LOG)
    return entry


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages") or []
    prompt = prompt_text(messages)
    requested = str(payload.get("model", "auto") or "auto")
    decision = route(prompt, requested, str(payload.get("policy", "weighted")))
    entry = LogEntry(
        time=now_iso(),
        model=decision.key,
        status="",
        prompt=prompt[:MAX_TEXT],
        output="",
        route_reason=decision.reason,
    )

    if not decision.key:
        entry.status, entry.outcome, entry.duration_ms = "schema", "policy_blocked", 0
        LOG.record(entry)
        log_route(decision, 0, "policy_blocked", ROUTE_LOG)
        return error_json(400, "no routable model", decision.excluded)

    adapter = adapter_for(decision.provider, CONFIG.key_for(decision.key), CLIENT, base=CONFIG.host_for(decision.key))
    # Everything except the gateway-controlled fields is passed through, so optional
    # SDK params (temperature, top_p, seed, chat_template_kwargs, ...) reach the upstream.
    body = {k: v for k, v in payload.items() if k not in {"model", "messages", "stream", "policy"}}
    body["messages"] = messages
    started = time.perf_counter()

    if payload.get("stream"):
        return StreamingResponse(stream_and_log(adapter, decision, body, entry, started), media_type="text/event-stream")

    try:
        raw = await adapter.chat(decision.model_id, body, stream=False)
    except ProviderError as exc:
        entry.status, entry.outcome = exc.error_class, "error"
        entry.duration_ms = int((time.perf_counter() - started) * 1000)
        LOG.record(entry)
        log_route(decision, entry.duration_ms, "error", ROUTE_LOG)
        CTX.health.record(decision.provider, "error", entry.duration_ms, exc.retry_after)
        return error_json(exc.status or 500, exc.message, decision.excluded)

    entry.status, entry.outcome = "ok", "pass"
    entry.output = answer_of(raw)[:MAX_TEXT]
    entry.duration_ms = int((time.perf_counter() - started) * 1000)
    CTX.health.record(decision.provider, "pass", entry.duration_ms)
    finalize(entry, decision, raw.get("usage") or {})
    return raw


async def stream_and_log(adapter, decision, body: dict, entry: LogEntry, started: float):
    usage: dict = {}
    parts: list[str] = []
    try:
        async for chunk in await adapter.chat(decision.model_id, body, stream=True):
            if entry.ttft_ms is None:
                entry.ttft_ms = int((time.perf_counter() - started) * 1000)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or choice.get("message") or {}
                content = delta.get("content") or choice.get("text") or ""
                if content:
                    parts.append(content)
            yield f"data: {json.dumps(chunk)}\n\n"
        entry.status, entry.outcome = "ok", "pass"
    except ProviderError as exc:
        entry.status, entry.outcome = exc.error_class, "error"
        yield f"data: {json.dumps({'error': {'message': exc.message}})}\n\n"

    entry.output = "".join(parts)[:MAX_TEXT]
    entry.duration_ms = int((time.perf_counter() - started) * 1000)
    CTX.health.record(decision.provider, entry.outcome, entry.duration_ms)
    finalize(entry, decision, usage, entry.ttft_ms)
    yield "data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, limit: int = 100):
    limit = min(max(1, limit), LOG.max_entries)
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "entries": LOG.recent(limit),
            "limit": limit,
            "stats": LOG.stats(),
            "config": CONFIG,
            "selected": CONFIG.selected_models(),
            "key_source": "env" if env_keys() else "runtime",
            "active": "home",
        },
    )


def last_results() -> dict[str, dict]:
    """Most recent attempt per model from the accumulated results history."""
    last: dict[str, dict] = {}
    for row in benchmark_rows():
        last[row["model_key"]] = row
    return last


def outcome_counts(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        label = row["outcome"] if row["outcome"] != "error" else (row["error_class"] or "error")
        counts[label] = counts.get(label, 0) + 1
    return " ".join(f"{label}={n}" for label, n in sorted(counts.items()))


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    env = env_keys()
    last = last_results()
    grouped: dict[str, list] = {}
    for model in CONFIG.models:
        provider = provider_index().get(model.provider)
        key = CONFIG.key_for(model.model_key)
        source = (
            "runtime"
            if model.api_key
            else "provider"
            if CONFIG.provider_keys.get(model.provider)
            else "env"
            if env.get(model.provider)
            else "global"
            if CONFIG.api_key
            else "none"
        )
        grouped.setdefault(model.provider, []).append(
            {
                "model": model,
                "masked": mask(key),
                "source": source,
                "default_host": provider.baseUrl if provider else "",
                "last": last.get(model.model_key),
            }
        )

    groups = [
        {
            "provider": provider,
            "category": report.category_of(provider),
            "tpm": CONFIG.tpm_for(provider),
            "masked": mask(CONFIG.provider_keys.get(provider) or env.get(provider) or CONFIG.api_key),
            "summary": outcome_counts([row["last"] for row in grouped[provider] if row["last"]]),
            "rows": grouped[provider],
        }
        for provider in sorted(grouped)
    ]
    return templates.TemplateResponse(
        request,
        "config.html",
        {
            "config": CONFIG,
            "groups": groups,
            "tested": sum(1 for row in CONFIG.models if last.get(row.model_key)),
            "probe_prompt": PROBE_PROMPT,
            "active": "config",
        },
    )


@app.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page(request: Request):
    scores = load_scores()
    frontier = set(pareto_frontier(scores))
    view = [
        {
            "key": m.key,
            "provider": m.provider,
            "category": report.category_of(m.provider),
            "pass_rate": m.pass_rate,
            "scored_n": m.scored_n,
            "p50_ms": m.p50_ms,
            "p95_ms": m.p95_ms,
            "tokens_per_sec": m.tokens_per_sec,
            "error_rate": m.error_rate,
            "coverage": m.coverage,
            "quota_cost": m.quota_cost,
            "frontier": m.key in frontier,
        }
        for m in sorted(scores.values(), key=lambda m: (report.category_of(m.provider), m.provider, m.key))
    ]
    rows = benchmark_rows()
    return templates.TemplateResponse(
        request,
        "benchmark.html",
        {
            "scores": view,
            "rows": rows,
            "last_run": last_run(rows),
            "min_n": 5,
            "probe_prompt": PROBE_PROMPT,
            "tasks": load_tasks(TASKS_PATH),
            "active": "benchmark",
        },
    )


def probe_task() -> Task:
    """The Retest task: reachability only, no checks, so it never scores."""
    return Task(
        id="probe",
        task_type="text",
        prompt=PROBE_PROMPT,
        checks=[],
        timeout_s=int(PROBE_TIMEOUT),
        scored=False,
    )


def last_run(rows: list[dict]) -> str:
    """Newest attempt time; rows recorded before the run_at column fall back to the file mtime."""
    times = [row.get("run_at") for row in rows if row.get("run_at")]
    if times:
        return max(times)
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def benchmark_rows() -> list[dict]:
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mask(value: str) -> str:
    if not value:
        return "none"
    return value if len(value) <= 8 else f"{value[:4]}…{value[-4:]}"


@app.get("/api/log")
async def api_log(limit: int = 100):
    return {"stats": LOG.stats(), "entries": [e.as_dict() for e in LOG.recent(min(max(1, limit), LOG.max_entries))]}


@app.get("/api/config")
async def api_config():
    return CONFIG.as_dict()


@app.post("/api/config")
async def api_update_config(request: Request):
    data = await request.json()
    if data.get("api_host"):
        CONFIG.api_host = str(data["api_host"])

    for entry in data.get("providers", []):
        provider = str(entry.get("provider", ""))
        if not provider:
            continue
        CONFIG.provider_keys[provider] = str(entry.get("api_key", ""))
        if entry.get("tpm") is not None:
            try:
                tpm = int(entry["tpm"])
            except (TypeError, ValueError):
                continue
            if tpm > 0:
                CONFIG.provider_tpm[provider] = tpm

    for row in data.get("models", []):
        model = CONFIG.by_key(str(row.get("model_key", "")))
        if model is None:
            continue
        model.host = str(row.get("host", ""))
        model.keyword = str(row.get("keyword", ""))
        model.api_key = str(row.get("api_key", ""))
        if "selected" in row:
            model.selected = bool(row["selected"])
    save_config(CONFIG)
    return {"saved": True, "models": len(CONFIG.models)}


@app.get("/api/benchmark")
async def api_benchmark():
    scores = load_scores()
    return {"scores": {k: v.as_dict() for k, v in scores.items()}, "rows": benchmark_rows()}


@app.post("/api/retest")
async def api_retest(request: Request):
    """One probe attempt for a single model, appended to the recorded history."""
    data = await request.json()
    model_key = str(data.get("model_key", "")).strip()
    found = resolve_model(load_catalog(), model_key)
    if not found:
        return error_json(404, f"unknown model {model_key or 'none'}")

    model = found[0]
    row = await run_one(
        CLIENT,
        model.provider,
        model,
        probe_task(),
        CONFIG.key_for(model.key),
        max_tokens=PROBE_MAX_TOKENS,
        timeout=PROBE_TIMEOUT,
        base=CONFIG.host_for(model.key),
    )

    rows = read_rows(RESULTS_DIR / "results.csv") + [row]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.write_csv(rows, RESULTS_DIR / "results.csv")
    report.write_scores(aggregate(rows), RESULTS_DIR / "scores.json")
    CTX.metrics = load_scores()
    return {"attempt": row.as_dict(), "recorded": len(rows)}


@app.get("/providers")
async def providers():
    catalog = load_catalog()
    keys = config_keys()
    return {
        "categories": {
            category: [
                {
                    "name": p.name,
                    "slug": slug(p.name),
                    "baseUrl": p.baseUrl,
                    "models": len([m for m in p.models if m.id]),
                    "key": bool(keys.get(slug(p.name))),
                }
                for p in catalog.providers
                if p.category == category
            ]
            for category in ("provider_api", "inference_provider")
        }
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m.key, "provider": m.provider} for m in routable_models(load_catalog())]}


@app.get("/health")
async def health():
    return {"status": "ok", "log": LOG.stats(), "health": CTX.health.snapshot(), "selected": len(CONFIG.selected_models())}


def main() -> None:
    LOG.load()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
