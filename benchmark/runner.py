from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

import httpx

from benchmark.checks import Task, run_check
from catalog.loader import load_catalog, routable_models
from catalog.rate_limits import effective_rpm, parse_rate_limit, rpm_from_tpm
from catalog.schema import Catalog, ModelEntry
from catalog.task_types import model_task_types
from providers.registry import adapter_for
from routing.constraints import ANONYMOUS_PROVIDERS

OUTCOMES = ("pass", "fail", "error", "skipped")
ERROR_CLASSES = ("ok", "429", "timeout", "auth", "schema", "unsupported", "server")


def attempt_time() -> str:
    """Attempt timestamp as plain local text, so results.csv stays readable by hand."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class BenchOptions:
    providers: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    max_requests: int = 15
    concurrency: int = 4
    keys: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    min_n: int = 5
    out_dir: Path = Path("results")
    attempt_all: bool = False  # keyless models are attempted anyway, so coverage records the auth failure
    tpm: dict[str, int] = field(default_factory=dict)  # per-provider token budget
    tokens_per_request: int = 600


@dataclass
class PlannedCall:
    provider: str
    model: str
    task_id: str
    task_type: str
    reason: str  # "run" or "skipped: no key" etc.


@dataclass
class Row:
    provider: str
    model: str
    task_id: str
    task_type: str
    outcome: str
    latency_ms: int | None = None
    ttft_ms: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float | None = None
    http_status: int | None = None
    error_class: str = "ok"
    answer_hash: str = ""
    rate_limit: str = ""
    run_at: str = ""

    @property
    def model_key(self) -> str:
        return f"{self.provider}/{self.model}"

    def as_dict(self) -> dict:
        return {
            "model_key": self.model_key,
            "provider": self.provider,
            "model": self.model,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_sec": self.tokens_per_sec,
            "http_status": self.http_status,
            "error_class": self.error_class,
            "answer_hash": self.answer_hash,
            "rate_limit": self.rate_limit,
            "run_at": self.run_at,
        }


def read_rows(path: Path | str) -> list[Row]:
    """Attempt rows from results.csv, so a run appends to the history instead of replacing it."""
    path = Path(path)
    if not path.exists():
        return []

    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        for item in csv.DictReader(fh):
            rows.append(
                Row(
                    provider=item["provider"],
                    model=item["model_key"].split("/", 1)[1],
                    task_id=item["task_id"],
                    task_type=item["task_type"],
                    outcome=item["outcome"],
                    latency_ms=int(item["latency_ms"]) if item["latency_ms"] else None,
                    ttft_ms=int(item["ttft_ms"]) if item["ttft_ms"] else None,
                    prompt_tokens=int(item["prompt_tokens"] or 0),
                    completion_tokens=int(item["completion_tokens"] or 0),
                    tokens_per_sec=float(item["tokens_per_sec"]) if item["tokens_per_sec"] else None,
                    http_status=int(item["http_status"]) if item["http_status"] else None,
                    error_class=item["error_class"] or "ok",
                    answer_hash=item["answer_hash"],
                    rate_limit=item["rate_limit"],
                    run_at=item.get("run_at", ""),
                )
            )
    return rows


def plan(catalog: Catalog, tasks: list[Task], opts: BenchOptions) -> list[PlannedCall]:
    """Dry-run plan: which model/task pairs would run, and why others are skipped."""
    calls: list[PlannedCall] = []
    per_provider: dict[str, int] = {}

    for task in tasks:
        for model in routable_models(catalog):
            if opts.providers and model.provider not in {p.lower() for p in opts.providers}:
                continue
            if opts.models and model.id not in opts.models:
                continue
            if opts.task_types and task.task_type not in opts.task_types:
                continue
            if task.task_type not in model_task_types(model):
                continue

            reason = "run"
            if not opts.keys.get(model.provider) and model.provider not in ANONYMOUS_PROVIDERS and not opts.attempt_all:
                reason = "skipped: no key"
            elif per_provider.get(model.provider, 0) >= opts.max_requests:
                reason = "skipped: provider request cap"
            elif effective_rpm(parse_rate_limit(model.rateLimit), cap=opts.max_requests) <= 0:
                reason = "skipped: rate limit"

            per_provider[model.provider] = per_provider.get(model.provider, 0) + (1 if reason == "run" else 0)
            calls.append(PlannedCall(model.provider, model.id or "", task.id, task.task_type, reason))
    return calls


def task_content(task: Task):
    """Vision tasks send the sample image as a data URL; everything else is plain text."""
    if not task.image:
        return task.prompt
    data = base64.b64encode((Path(__file__).parent / "assets" / task.image).read_bytes()).decode()
    return [
        {"type": "text", "text": task.prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
    ]


async def run_one(
    client,
    provider: str,
    model: ModelEntry,
    task: Task,
    key: str = "",
    max_tokens: int = 512,
    timeout: float = 60.0,
    base: str = "",
) -> Row:
    adapter = adapter_for(provider, key, client, base=base, timeout=timeout)
    row = Row(
        provider=provider,
        model=model.id or "",
        task_id=task.id,
        task_type=task.task_type,
        outcome="skipped",
        rate_limit=model.rateLimit or "",
        run_at=attempt_time(),
    )

    started = perf_counter()
    try:
        if task.endpoint == "embeddings":
            raw = await adapter.embeddings(model.id or "", {"input": task.prompt})
            answer = json.dumps(raw)
        else:
            raw = await adapter.chat(
                model.id or "",
                {"messages": [{"role": "user", "content": task_content(task)}], "max_tokens": max_tokens},
                stream=False,
            )
            content = raw.get("choices", [{}])[0].get("message", {}).get("content")
            # Providers can return null content (reasoning-only or tool-call replies), which is
            # data to record, not a crash.
            answer = content if isinstance(content, str) else (json.dumps(content) if content else "")
    except Exception as exc:  # provider errors are data, not crashes
        row.outcome = "error"
        row.latency_ms = int((perf_counter() - started) * 1000)
        row.error_class = getattr(exc, "error_class", "error")
        row.http_status = getattr(exc, "status", None)
        return row

    row.latency_ms = int((perf_counter() - started) * 1000)
    usage = raw.get("usage") or {}
    row.prompt_tokens = usage.get("prompt_tokens", 0)
    row.completion_tokens = usage.get("completion_tokens", 0)
    row.tokens_per_sec = round(row.completion_tokens / (row.latency_ms / 1000), 1) if row.latency_ms else None
    row.answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]
    row.outcome = "pass" if all(run_check(check, answer, task) for check in task.checks) else "fail"
    return row


async def run(catalog: Catalog, tasks: list[Task], opts: BenchOptions) -> list[Row]:
    """Sequential per provider (respecting rate limits), concurrent across providers."""
    calls = [c for c in plan(catalog, tasks, opts) if c.reason == "run"]
    by_provider: dict[str, list[PlannedCall]] = {}
    for call in calls:
        by_provider.setdefault(call.provider, []).append(call)

    models = {m.key: m for m in routable_models(catalog)}
    semaphore = asyncio.Semaphore(max(1, opts.concurrency))
    answers: list[Row] = []
    total = len(calls)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def worker(provider: str, group: list[PlannedCall]) -> None:
            async with semaphore:
                # Pacing only where quota is actually consumed; keyless attempts fail fast.
                first = models[f"{group[0].provider}/{group[0].model}"]
                budget = rpm_from_tpm(opts.tpm.get(provider), opts.tokens_per_request)
                rpm = min(effective_rpm(parse_rate_limit(first.rateLimit), cap=opts.max_requests), budget or opts.max_requests)
                delay = 60.0 / rpm if opts.keys.get(provider) or provider in ANONYMOUS_PROVIDERS else 0.0
                print(f"start {provider}: {len(group)} calls at {rpm} rpm -> {delay:.1f}s pacing", flush=True)

                for call in group:
                    model = models[f"{call.provider}/{call.model}"]
                    row = await run_one(client, provider, model, next(t for t in tasks if t.id == call.task_id), opts.keys.get(provider, ""))
                    answers.append(row)
                    print(f"  [{len(answers)}/{total}] {row.model_key} {row.outcome} {row.error_class} {row.latency_ms} ms", flush=True)
                    await asyncio.sleep(delay or 0.1)

        await asyncio.gather(*(worker(p, g) for p, g in sorted(by_provider.items())))

    opts.out_dir.mkdir(parents=True, exist_ok=True)
    (opts.out_dir / "answers.jsonl").write_text(
        "\n".join(json.dumps({"model_key": r.model_key, "task_id": r.task_id, "outcome": r.outcome}) for r in answers),
        encoding="utf-8",
    )
    return answers


def min_n_gate(rows: list[Row], min_n: int = 5) -> dict[str, bool]:
    """A candidate is ranked only when it has enough scored samples; otherwise coverage-only."""
    scored: dict[str, int] = {}
    for row in rows:
        if row.outcome in ("pass", "fail"):
            scored[row.model_key] = scored.get(row.model_key, 0) + 1
    return {key: count >= min_n for key, count in sorted(scored.items())}
