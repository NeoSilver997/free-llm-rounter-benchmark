from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import typer

from benchmark.checks import PROBE_PROMPT, Task, load_tasks
from benchmark.metrics import aggregate, win_rate_at_budget
from benchmark.report import load_scores, write_csv, write_html, write_json, write_scores
from benchmark.router_eval import cost_quality_curve, load_recorded, render_plot, replay
from benchmark.runner import BenchOptions, Row, plan, read_rows, run, run_one
from catalog.grouping import group, group_counts, render_table
from catalog.loader import load_catalog, resolve_model, routable_models, slug, unique_prefix
from catalog.rate_limits import effective_rpm, parse_rate_limit, rpm_from_tpm
from catalog.schema import Catalog
from config import DEFAULT_TPM, env_keys, load_config
from routing.budget import Budget
from routing.constraints import ANONYMOUS_PROVIDERS
from routing.policies import POLICIES, RoutingContext, classify
from routing.request import RoutingRequest
from telemetry import ALLOWED_OUTCOMES, ALLOWED_REASONS, summarize

app = typer.Typer(help="LLM API router: grouped catalog, deterministic benchmark, policy replay.", no_args_is_help=True)

CATALOG_URL = "https://raw.githubusercontent.com/mnfst/awesome-free-llm-apis/main/data.json"
CATALOG_PATH = Path("catalog/data.json")
TASKS_PATH = Path("benchmark/tasks.json")
RESULTS_DIR = Path("results")
ROUTE_LOG = Path("routes.jsonl")


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _fmt(value) -> str:
    return "unknown" if value is None else f"{value:.3f}"


@app.command()
def refresh() -> None:
    """Download data.json into catalog/data.json; the snapshot keeps runs reproducible."""
    raw = httpx.get(CATALOG_URL, timeout=30.0).raise_for_status().json()
    catalog = Catalog.model_validate(raw)  # reject a malformed snapshot before writing it
    CATALOG_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    typer.echo(f"{len(catalog.providers)} providers, {len(routable_models(catalog))} routable models -> {CATALOG_PATH}")


@app.command()
def groups(task_type: str = "", format: str = typer.Option("table", "--format")) -> None:
    """Category -> task type -> models (the 'group by type' deliverable)."""
    grouped = group(load_catalog())
    if task_type:
        grouped = {category: {task_type: models} for category, inner in grouped.items() if (models := inner.get(task_type))}

    if format == "json":
        typer.echo(json.dumps(group_counts(grouped), indent=2))
    else:
        typer.echo(render_table(grouped))


@app.command()
def route(model: str, prompt: str, policy: str = "weighted", require: str = "", local_only: bool = False) -> None:
    """Pick a route for a prompt without calling the provider."""
    catalog = load_catalog()
    scores = RESULTS_DIR / "scores.json"
    ctx = RoutingContext(keys=env_keys(), metrics=load_scores(scores) if scores.exists() else {})
    req = RoutingRequest(
        model=model,
        task_type=classify(prompt),
        messages=({"role": "user", "content": prompt},),
        require=frozenset(_csv_values(require)),
        local_only=local_only,
        policy=policy,
    )
    candidates = resolve_model(catalog, model) or routable_models(catalog)
    decision = POLICIES.get(policy, POLICIES["weighted"]).route(req, candidates, ctx)

    typer.echo(f"policy={decision.policy} task_type={req.task_type} route={decision.key or 'none'}")
    typer.echo(f"reason={decision.reason} score={decision.score}")
    for key, why in sorted(decision.excluded.items()):
        typer.echo(f"  excluded {key}: {why}")


@app.command()
def probe(
    concurrency: int = 4,
    timeout: int = 30,
    max_tokens: int = 16,
    per_provider: int = 0,
    tpm: int = 0,
    providers: str = "",
    skip_tested: bool = False,
) -> None:
    """One tiny chat request per routable model; results.csv keeps one row per attempt across runs."""
    catalog = load_catalog()
    keys = env_keys()
    cfg = load_config()
    task = Task(id="probe", task_type="text", prompt=PROBE_PROMPT, checks=[], timeout_s=timeout, scored=False)

    history = read_rows(RESULTS_DIR / "results.csv")
    wanted = set(_csv_values(providers))
    tested = {row.model_key for row in history}
    by_provider: dict[str, list] = {}
    for model in routable_models(catalog):
        if wanted and model.provider not in wanted:
            continue
        if skip_tested and model.key in tested:
            continue
        if per_provider and len(by_provider.get(model.provider, [])) >= per_provider:
            continue
        by_provider.setdefault(model.provider, []).append(model)

    async def sweep() -> list[Row]:
        rows: list[Row] = []
        semaphore = asyncio.Semaphore(max(1, concurrency))
        total = sum(len(g) for g in by_provider.values())
        async with httpx.AsyncClient(timeout=httpx.Timeout(float(timeout), connect=10.0)) as client:
            async def worker(provider: str, group: list) -> None:
                async with semaphore:
                    # Pacing only where quota is actually consumed; keyless calls fail fast.
                    budget = rpm_from_tpm(tpm or cfg.tpm_for(provider), max_tokens + 20)
                    rpm = min(effective_rpm(parse_rate_limit(group[0].rateLimit), cap=200), budget or 200)
                    delay = min(60.0 / rpm, 30.0) if keys.get(provider) or provider in ANONYMOUS_PROVIDERS else 0.0
                    print(
                        f"start {provider}: {len(group)} models, tpm={tpm or cfg.tpm_for(provider)} -> {rpm} rpm -> {delay:.1f}s pacing",
                        flush=True,
                    )
                    for model in group:
                        row = await run_one(
                            client, provider, model, task, keys.get(provider, ""), max_tokens=max_tokens, timeout=float(timeout)
                        )
                        rows.append(row)
                        print(f"  [{len(rows)}/{total}] {row.model_key} {row.outcome} {row.error_class} {row.latency_ms} ms", flush=True)
                        await asyncio.sleep(delay or 0.1)
            await asyncio.gather(*(worker(p, g) for p, g in sorted(by_provider.items())))
        return rows

    new_rows = asyncio.run(sweep())
    rows = history + new_rows
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(rows, RESULTS_DIR / "results.json")
    write_csv(rows, RESULTS_DIR / "results.csv")
    write_scores(aggregate(rows), RESULTS_DIR / "scores.json")

    per_provider: dict[str, dict[str, int]] = {}
    for row in new_rows:
        counts = per_provider.setdefault(row.provider, {})
        label = row.outcome if row.outcome != "error" else row.error_class
        counts[label] = counts.get(label, 0) + 1

    typer.echo(f"probe: {len(new_rows)} new attempts, {len(rows)} recorded in total")
    for provider in sorted(per_provider):
        detail = " ".join(f"{label}={n}" for label, n in sorted(per_provider[provider].items()))
        typer.echo(f"  {provider:<24} {detail}")
    typer.echo(f"results written to {RESULTS_DIR}")


@app.command()
def bench(
    providers: str = "",
    task_type: str = "",
    models: str = "",
    max_requests: int = 15,
    concurrency: int = 4,
    attempt_all: bool = False,
    dry_run: bool = False,
    tpm: int = 0,
) -> None:
    """Run the deterministic task matrix and write results.json / results.csv / scores.json / report.html."""
    catalog = load_catalog()
    tasks = load_tasks(TASKS_PATH)
    known = {slug(p.name) for p in catalog.providers}
    cfg = load_config()
    opts = BenchOptions(
        providers=tuple(unique_prefix(p, known) or p for p in _csv_values(providers)),
        task_types=_csv_values(task_type),
        models=tuple(m.strip() for m in models.split(",") if m.strip()),
        max_requests=max_requests,
        concurrency=concurrency,
        keys=env_keys(),
        dry_run=dry_run,
        out_dir=RESULTS_DIR,
        attempt_all=attempt_all,
        tpm={p: tpm or cfg.tpm_for(p) for p in cfg.providers()},
    )

    calls = plan(catalog, tasks, opts)
    reasons: dict[str, int] = {}
    for call in calls:
        reasons[call.reason] = reasons.get(call.reason, 0) + 1
    typer.echo(f"plan: {len(calls)} attempts over {len(set(c.model for c in calls))} models")
    for reason, n in sorted(reasons.items()):
        typer.echo(f"  {reason}: {n}")

    if dry_run:
        return

    history = read_rows(RESULTS_DIR / "results.csv")
    new_rows = asyncio.run(run(catalog, tasks, opts))
    rows = history + new_rows
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(rows, RESULTS_DIR / "results.json")
    write_csv(rows, RESULTS_DIR / "results.csv")

    metrics = aggregate(rows)
    write_scores(metrics, RESULTS_DIR / "scores.json")
    policy_metrics = replay(POLICIES.values(), tasks, load_recorded(RESULTS_DIR / "results.json"), catalog=catalog)
    write_html(metrics, policy_metrics, RESULTS_DIR / "report.html", opts.min_n)

    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row.outcome] = outcomes.get(row.outcome, 0) + 1
    typer.echo(f"run: {len(new_rows)} new attempts, {len(rows)} recorded in total -> {outcomes}")
    typer.echo(f"ranked (n >= {opts.min_n}): {sum(1 for m in metrics.values() if m.scored_n >= opts.min_n)} of {len(metrics)}")
    typer.echo(f"results written to {RESULTS_DIR}")


@app.command("eval")
def evaluate(budget: int = 20, policy: str = "") -> None:
    """Offline policy replay on recorded benchmark results (no extra API calls)."""
    recorded_path = RESULTS_DIR / "results.json"
    if not recorded_path.exists():
        raise typer.Exit(f"no recorded results: run 'python cli.py bench' first ({recorded_path})")

    recorded = load_recorded(recorded_path)
    tasks = load_tasks(TASKS_PATH)
    policies = [POLICIES[policy]] if policy in POLICIES else list(POLICIES.values())
    policy_metrics = replay(policies, tasks, recorded)

    typer.echo(f"policy                quality  cost      latency   escal  budget  win rate at budget={budget}")
    for pm in sorted(policy_metrics.values(), key=lambda p: p.policy):
        wins = ", ".join(f"{t}={v:.2f}" for t, v in pm.win_rate.items()) or "-"
        typer.echo(
            f"{pm.policy:<20} {_fmt(pm.quality)}  {_fmt(pm.mean_cost)}  "
            f"{_fmt(pm.mean_latency_ms)}  {_fmt(pm.escalation_rate)}  {pm.budget_used:.6f}  {wins}"
        )

    typer.echo("\ncost-vs-quality curve (policies):")
    typer.echo(render_plot(cost_quality_curve(policy_metrics)))

    metrics = aggregate([row for per_task in recorded.values() for row in per_task.values()])
    win_rate = win_rate_at_budget(metrics, Budget(max_requests=budget))
    typer.echo("\nwin rate at budget (per task type):")
    for task_type, value in sorted(win_rate.items()):
        typer.echo(f"  {task_type}: {value:.2f}")


@app.command()
def summary() -> None:
    """Summarize the content-free route telemetry JSONL."""
    if not ROUTE_LOG.exists():
        raise typer.Exit(f"no telemetry log ({ROUTE_LOG})")
    report = summarize(ROUTE_LOG, {"reasons": ALLOWED_REASONS, "outcomes": ALLOWED_OUTCOMES})
    typer.echo(json.dumps(report, indent=2))


@app.command()
def serve(port: int = 8000) -> None:
    """Serve the gateway and the UI."""
    import server

    server.LOG.load()
    import uvicorn

    uvicorn.run(server.app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
