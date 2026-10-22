from __future__ import annotations

import json
from pathlib import Path

from benchmark import checks, report, router_eval
from benchmark.checks import Task, load_tasks, run_check
from benchmark.metrics import ModelMetrics, aggregate, pareto_frontier, win_rate_at_budget
from benchmark.runner import BenchOptions, Row, min_n_gate, plan, task_content
from catalog.loader import load_catalog, routable_models
from providers.ollama import normalize_ollama_response, parse_ollama_body
from providers.openai_adapter import normalize_openai_response
from routing.budget import Budget
from routing.policies import POLICIES, RoutingContext
from routing.request import RoutingRequest
from routing.scoring import DEFAULT_WEIGHTS, rank

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "benchmark" / "tasks.json"
CATALOG = load_catalog()

# Catalog models whose modality covers the task type and whose rate limit is parseable,
# so the fixture stays consistent with the source of truth.
TEXT_A = "aion_labs/aion-labs/aion-rp-llama-3.1-8b"
TEXT_B = "cloudflare_workers_ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CODE_C = "mistral_ai/codestral-2508"
MODEL = {m.key: m for m in routable_models(CATALOG)}


def row(key: str, task_id: str, task_type: str, outcome: str, latency: int | None,
        prompt: int = 200, completion: int = 300) -> Row:
    provider, model = key.split("/", 1)
    return Row(
        provider=provider,
        model=model,
        task_id=task_id,
        task_type=task_type,
        outcome=outcome,
        latency_ms=latency,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokens_per_sec=round(completion / (latency / 1000), 1) if latency else None,
        http_status=200 if outcome != "error" else 429,
        error_class="ok" if outcome != "error" else "429",
        answer_hash=f"{key}|{task_id}",
        rate_limit=MODEL[key].rateLimit if key in MODEL else "",
    )


def fixture_rows() -> list[Row]:
    return [
        row(TEXT_A, "text-pong", "text", "pass", 100),
        row(TEXT_A, "text-json", "text", "pass", 200),
        row(TEXT_A, "text-count", "text", "pass", 300),
        row(TEXT_A, "text-sentence", "text", "pass", 400),
        row(TEXT_B, "text-pong", "text", "fail", 500),
        row(TEXT_B, "text-json", "text", "pass", 600),
        row(TEXT_B, "text-count", "text", "fail", 700),
        row(TEXT_B, "text-sentence", "text", "fail", 800),
        row(CODE_C, "code-exec", "code", "pass", 900),
        row(CODE_C, "code-regex", "code", "fail", 1000),
    ]


def recorded(rows: list[Row]) -> dict[str, dict[str, Row]]:
    out: dict[str, dict[str, Row]] = {}
    for r in rows:
        out.setdefault(r.task_id, {})[r.model_key] = r
    return out


def make_ctx(metrics: dict[str, ModelMetrics], rows: list[Row], min_n: int = 2) -> RoutingContext:
    return RoutingContext(
        keys={r.provider: "recorded" for r in rows},
        metrics=metrics,
        weights=dict(DEFAULT_WEIGHTS),
        min_n=min_n,
        pass_rate_by_task={
            key: {t: m.pass_rate for t in m.task_types}
            for key, m in metrics.items()
            if m.pass_rate is not None
        },
    )


# --- tasks.json ---------------------------------------------------------------

def test_tasks_cover_task_types_and_are_scored():
    tasks = load_tasks(TASKS_PATH)
    assert len(tasks) >= 12
    assert len({t.id for t in tasks}) == len(tasks)
    assert all(t.scored and t.checks for t in tasks)
    covered = {t.task_type for t in tasks}
    assert {"text", "reasoning", "code", "vision", "embeddings"} <= covered
    assert sum(1 for t in tasks if t.task_type == "text") >= 2
    assert sum(1 for t in tasks if t.task_type == "reasoning") >= 2
    assert sum(1 for t in tasks if t.task_type == "code") >= 2
    for task in tasks:
        for check in task.checks:
            assert check["type"] in checks.CHECK_TYPES


def test_load_tasks_rejects_unknown_check_type(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"id": "x", "task_type": "text", "prompt": "p",
                                 "checks": [{"type": "llm_judge"}]}]), encoding="utf-8")
    try:
        load_tasks(path)
    except ValueError as exc:
        assert "unknown check type" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_vision_task_uses_the_sample_asset():
    tasks = load_tasks(TASKS_PATH)
    vision = next(t for t in tasks if t.task_type == "vision")
    assert vision.image and (ROOT / "benchmark" / "assets" / vision.image).exists()
    content = task_content(vision)
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert isinstance(task_content(next(t for t in tasks if t.id == "text-pong")), str)


# --- deterministic checks ---------------------------------------------------

def test_normalize_and_extract_code_block():
    assert checks.normalize_answer("  PONG\n") == "pong"
    assert checks.normalize_answer("```text\nHello\n```") == "hello"
    assert checks.extract_code_block("prefix\n```python\nprint(1)\n```") == "print(1)"
    assert checks.extract_code_block("no fence") is None


def test_exact_contains_regex_numeric_count():
    assert checks.check_exact("PONG", "pong")
    assert not checks.check_exact("PONG.", "pong")
    assert checks.check_contains("Red and Blue", ["red", "blue"])
    assert not checks.check_contains("Red", ["blue"])
    assert checks.check_regex("match 54321 here", r"\d{5}")
    assert checks.check_regex("54321", r"^\d{5}$")
    assert not checks.check_regex("match 54321 here", r"^\d{5}$")
    assert not checks.check_regex("1234", r"\d{5}")
    assert checks.check_numeric("The ball is 5 cents", 5)
    assert not checks.check_numeric("The ball is 5 cents", 50)
    assert checks.check_numeric("about 4.99", 5, tol=0.01)
    assert not checks.check_numeric("no digits", 5)
    assert checks.check_count("red\nblue\nyellow", 3)
    assert not checks.check_count("red\nblue", 3)


def test_json_schema_and_format():
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
    assert checks.check_json_schema('{"answer": "four"}', schema)
    assert checks.check_json_schema('```json\n{"answer": "four"}\n```', schema)
    assert not checks.check_json_schema('{"value": 4}', schema)
    assert not checks.check_json_schema("not json", schema)

    rules = {"max_sentences": 1, "language": "en", "must_not_contain": ["sure"]}
    assert checks.check_format("The sea is calm.", rules)
    assert not checks.check_format("Sure, the sea is calm. It is wide.", rules)
    assert not checks.check_format("你好", rules)


def test_python_exec_check_runs_generated_code():
    task = next(t for t in load_tasks(TASKS_PATH) if t.id == "code-exec")
    answer = "```python\nimport sys\nprint(int(sys.stdin.read()) ** 2)\n```"
    assert run_check(task.checks[0], answer, task)
    assert not checks.check_python_exec("print(1)", "7", "49")
    assert not checks.check_python_exec("", "7", "49")
    assert not checks.check_python_exec("raise SystemExit(1)", "7", "49")


def test_run_check_rejects_unknown_type():
    task = Task(id="x", task_type="text", prompt="p", checks=[])
    try:
        run_check({"type": "llm_judge"}, "answer", task)
    except ValueError as exc:
        assert "unknown check type" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --- metrics ----------------------------------------------------------------

def test_aggregate_percentiles_include_failed_attempts():
    metrics = aggregate(fixture_rows())
    a, b, c = metrics[TEXT_A], metrics[TEXT_B], metrics[CODE_C]

    assert a.pass_rate == 1.0 and a.scored_n == 4
    assert b.pass_rate == 0.25 and b.scored_n == 4
    assert c.pass_rate == 0.5 and c.scored_n == 2

    # p50 spans all four attempts of B, not only the passing one.
    assert b.p50_ms == 650.0
    assert b.p95_ms == 785.0
    assert b.error_rate == 0.0
    assert b.coverage == ("text",)
    assert 0.0 < b.quota_cost < 1.0
    assert a.raw_rate_limit == "15 RPM, 20K TPD"


def test_aggregate_counts_errors_and_keeps_unmeasured_unknown():
    rows = fixture_rows() + [row(TEXT_A, "text-pong", "text", "error", 120)]
    a = aggregate(rows)[TEXT_A]
    assert a.attempts == 5 and a.scored_n == 4
    assert a.error_rate == 0.2
    assert a.outcomes == {"error": 1, "pass": 4}

    unmeasured = aggregate([row("groq/no-such-model", "text-pong", "text", "error", None)])
    entry = unmeasured["groq/no-such-model"]
    assert entry.p50_ms is None
    assert entry.pass_rate is None
    assert entry.quota_cost is None


def test_pareto_frontier_prefers_higher_quality_and_lower_cost():
    metrics = aggregate(fixture_rows())
    frontier = pareto_frontier(metrics)
    assert TEXT_A in frontier
    assert CODE_C in frontier
    assert TEXT_B not in frontier  # same task type, lower quality, higher cost


def test_win_rate_at_budget_only_counts_models_inside_the_budget():
    metrics = aggregate(fixture_rows())
    wide = win_rate_at_budget(metrics, Budget(max_requests=1, max_tokens=100000, per_provider_max_requests=1))
    assert wide.get("text") == 1.0 and wide.get("code") == 0.5

    # A budget share of 1e-5 leaves only the TPM-metered code model inside the budget.
    tight = win_rate_at_budget(metrics, Budget(max_requests=100000, max_tokens=100000, per_provider_max_requests=1))
    assert tight == {"code": 0.5}


def test_min_n_gate():
    assert min_n_gate(fixture_rows(), min_n=4) == {TEXT_A: True, TEXT_B: True, CODE_C: False}
    assert min_n_gate(fixture_rows(), min_n=5) == {TEXT_A: False, TEXT_B: False, CODE_C: False}


# --- runner plan ------------------------------------------------------------

def test_plan_records_skip_reasons_and_respects_the_cap():
    tasks = [t for t in load_tasks(TASKS_PATH) if t.task_type == "text"]
    opts = BenchOptions(providers=("aion_labs",), task_types=("text",), max_requests=2, keys={"aion_labs": "k"})
    calls = plan(CATALOG, tasks, opts)
    assert all(c.provider == "aion_labs" for c in calls)
    assert sum(1 for c in calls if c.reason == "run") == 2
    assert all(c.reason == "skipped: provider request cap" for c in calls if c.reason != "run")

    no_key = plan(CATALOG, tasks, BenchOptions(providers=("aion_labs",), task_types=("text",)))
    assert no_key and all(c.reason == "skipped: no key" for c in no_key)

    anonymous = plan(CATALOG, tasks, BenchOptions(providers=("ovhcloud_ai_endpoints",), task_types=("text",), max_requests=1))
    assert any(c.reason == "run" for c in anonymous)


# --- offline policy replay --------------------------------------------------

def test_replay_reports_quality_cost_and_win_rate():
    tasks = load_tasks(TASKS_PATH)
    rows = fixture_rows()
    metrics = aggregate(rows)
    pm = router_eval.replay([POLICIES["quality"]], tasks, recorded(rows), catalog=CATALOG,
                            ctx=make_ctx(metrics, rows))

    quality = pm["quality"]
    assert quality.decisions == 6 and quality.scored == 6
    assert quality.quality == (4 + 1) / 6
    assert quality.mean_cost is not None
    assert quality.mean_latency_ms == (100 + 200 + 300 + 400 + 900 + 1000) / 6
    assert quality.win_rate == {"code": 0.5, "text": 1.0}
    assert quality.budget_used > 0.0


def test_replay_only_ranks_models_above_the_minimum_n_gate():
    metrics = aggregate(fixture_rows())
    ctx = make_ctx(metrics, fixture_rows(), min_n=5)
    candidates = [MODEL[TEXT_A], MODEL[TEXT_B]]
    assert rank(candidates, metrics, DEFAULT_WEIGHTS, None, min_n=5) == []

    req = RoutingRequest(task_type="text", messages=({"role": "user", "content": "Reply with exactly: PONG"},),
                         policy="quality")
    decision = POLICIES["quality"].route(req, candidates, ctx)
    assert decision.reason == "unknown" and decision.key in {TEXT_A, TEXT_B}


def test_cost_quality_curve_and_plot_render():
    tasks = load_tasks(TASKS_PATH)
    rows = fixture_rows()
    metrics = aggregate(rows)
    pm = router_eval.replay(list(POLICIES.values()), tasks, recorded(rows), catalog=CATALOG,
                            ctx=make_ctx(metrics, rows))

    curve = router_eval.cost_quality_curve(pm)
    assert curve and all(cost is not None and quality is not None for cost, quality in curve)
    plot = router_eval.render_plot(curve)
    assert "quality ^" in plot and "<svg" in plot and "*" in plot
    assert len(plot.splitlines()) > 12


def test_load_recorded_round_trip(tmp_path):
    rows = fixture_rows()
    path = tmp_path / "results.json"
    report.write_json(rows, path)
    reloaded = router_eval.load_recorded(path)
    assert sorted(reloaded) == sorted({r.task_id for r in rows})
    for r in rows:
        got = reloaded[r.task_id][r.model_key]
        assert got.outcome == r.outcome and got.latency_ms == r.latency_ms
        assert got.rate_limit == r.rate_limit and got.answer_hash == r.answer_hash


# --- report -----------------------------------------------------------------

def test_report_writers_and_scores_round_trip(tmp_path):
    tasks = load_tasks(TASKS_PATH)
    rows = fixture_rows()
    metrics = aggregate(rows)
    pm = router_eval.replay(list(POLICIES.values()), tasks, recorded(rows), catalog=CATALOG,
                            ctx=make_ctx(metrics, rows))

    report.write_json(rows, tmp_path / "results.json")
    report.write_csv(rows, tmp_path / "results.csv")
    report.write_scores(metrics, tmp_path / "scores.json")
    # min_n=4 puts A and B above the gate and C below it, so both gate states render.
    report.write_html(metrics, pm, tmp_path / "report.html", min_n=4)

    csv_rows = (tmp_path / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_rows) == len(rows) + 1
    assert csv_rows[0].split(",")[0] == "model_key"

    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert scores[TEXT_A]["category"] == "provider_api"
    assert scores[TEXT_A]["pass_rate"] == 1.0
    assert scores[TEXT_A]["on_pareto_frontier"] is True
    assert scores[TEXT_B]["on_pareto_frontier"] is False

    restored = report.load_scores(tmp_path / "scores.json")
    assert restored[TEXT_A].pass_rate == 1.0 and restored[TEXT_A].coverage == ("text",)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "provider_api" in html and "Policies" in html and "win rate at budget" in html
    assert "coverage only" in html and "ranked" in html


def test_ollama_body_merges_newline_delimited_chunks():
    body = (
        '{"model":"gpt-oss:20b","message":{"role":"assistant","content":"ok"},"done":false}\n'
        '{"model":"gpt-oss:20b","message":{"role":"assistant","content":""},"done":true,'
        '"done_reason":"stop","prompt_eval_count":18,"eval_count":4}'
    )
    raw = normalize_ollama_response(parse_ollama_body(body))
    assert raw["choices"][0]["message"]["content"] == "ok"
    assert raw["usage"] == {"prompt_tokens": 18, "completion_tokens": 4, "total_tokens": 22}


def test_normalize_keeps_reasoning_content():
    raw = {
        "choices": [{"message": {"role": "assistant", "content": "red", "reasoning_content": "the sign is an octagon"}}],
        "usage": {"prompt_tokens": 1186, "completion_tokens": 147, "completion_tokens_details": {"reasoning_tokens": 132}},
    }
    out = normalize_openai_response(raw)
    assert out["choices"][0]["message"]["reasoning_content"] == "the sign is an octagon"
    assert out["usage"]["total_tokens"] == 1333
    # A model that does not think has no reasoning field to preserve.
    assert "reasoning_content" not in normalize_openai_response({"choices": [{"message": {"content": "ok"}}]})["choices"][0]["message"]
