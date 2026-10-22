# LLM API Router — grouped by type, deterministic quality benchmark, policy evaluation

## Summary

Greenfield Python project in `c:\AI_Server\ClaudeCode\LLM-Router` (currently empty).

Four layers on top of the catalog from `mnfst/awesome-free-llm-apis`:

1. **Catalog** — `data.json` snapshotted locally, parsed into typed models, providers grouped **primary = provider category** (`provider_api` / `inference_provider`), **secondary = task type** (text / reasoning / code / vision / audio / embeddings / rerank / roleplay).
2. **Routing core** — `AbstractRouter` policies (explicit, quality, cheapest-quota, weighted PQR, hybrid-threshold, cascade, semantic), applied as *hard constraints first, then normalized weighted scoring*, returning an explainable decision.
3. **HTTP gateway** — FastAPI, OpenAI-SDK compatible, forwards `/v1/chat/completions` using the routing core; per-provider adapters for non-compatible upstreams; health tracking and failover.
4. **Benchmark** — deterministic quality checks (no LLM judge) + operational metrics, then **offline policy evaluation** (RouterBench-style): replay each policy against recorded per-model results and report quality-vs-quota-cost Pareto and win-rate-at-budget.

Stack: Python 3.11+, FastAPI, httpx (async), pydantic v2, typer, pytest.

## Reference inputs (what is adopted from `dstripelis/Awesome-LLM-Routing`)

| Source | Idea adopted here |
|---|---|
| RouterBench (`withmartian/routerbench`) | `AbstractRouter` extension point; pipeline shape *convert → evaluate routers → visualize*; long-format CSV results; performance-vs-cost plot. |
| LLM-PQR (`Amazed-Labs/llm-pqr`) | Hard constraints applied **before** scoring (capabilities, locality, unavailable providers); normalized weights auto-sum; decision output includes `excluded` reasons; minimum-N gate before a candidate is scored; latency percentiles include failed/interrupted attempts (not success-biased); content-free route telemetry (`route`, `reason`, `latency_ms`, `outcome`) with an allowlist. |
| FrugalGPT (05-2023) | Cascade: try the weak/cheap candidate first, escalate on low confidence, stop when budget is exhausted. |
| Hybrid LLM (06-2024) | Quality-aware threshold routing: route "easy" requests to the weak model, escalate above a per-task-type threshold. |
| RouteLLM (07-2024) | Win rate at a fixed budget as the headline routing metric, reported per task subset. |
| ORBIT / RouteJudge (2026) | Common interface for routers and benchmarks; budget-aware evaluation; reproducibility (fixed seeds, recorded raw answers). |
| Bifrost / Anyscale llm-router | Gateway concerns kept separate from policy: load balancing, failover, provider health. |
| semantic-router (Aurelio) | Intent classification decides the *task-type group*; the router then picks within that group. Implemented here as a deterministic keyword classifier (no embedding dependency). |

Cost is **not money** — every provider here is a free tier. Cost is normalized as **quota consumption** (share of the provider's published RPD/TPM budget used by the request), so quality-vs-cost curves remain meaningful.

## Current State Analysis

- `c:\AI_Server\ClaudeCode\LLM-Router` is empty — no entry points, configs, or conventions exist. All structure is new; the layout below defines the conventions.
- Upstream catalog schema (`data.json`, ~45 KB, source of truth; README is generated from it by a GitHub Action):
  - `lastUpdated`, `providers[]`
  - provider: `name`, `category` (`provider_api` | `inference_provider`), `country`, `flag`, `url`, `baseUrl`, `description`, `footnoteRef`, `models[]`
  - model: `id`, `name`, `context`, `maxOutput`, `modality`, `rateLimit` — all strings; `null` / `—` / `+ N more models` placeholder rows exist.
- `rateLimit` is free text: `"30 RPM, 14,400 RPD, 1M TPD"`, `"~1 RPS, 500K TPM"`, `"10K neurons/day (shared)"`, `"Dynamic"`, `"Tiered by region"`, `"1 concurrent request"`, `"Credit-metered"` → parser must tolerate unparseable values.
- `modality` is free text: `Text`, `Text (reasoning)`, `Text + Image`, `Code`, `Embeddings (Text + Image)`, `Reranking`, `Multimodal`, `Audio → Text`, `Text (roleplay)` → normalizer needed; a model can match multiple task types.
- Adapters: most providers are OpenAI-compatible. Exceptions/quirks: Ollama Cloud (native `/api/chat`), Cloudflare Workers AI (`/ai/run`, `account_id` in base URL, response wrapped in `{"result": {...}}`), Hugging Face (OpenAI-compatible host is `https://router.huggingface.co/v1`, not `api-inference.huggingface.co/models`), Cohere v2. Placeholder rows (`id: null`) are never routable.
- ~20 providers: provider APIs (Aion Labs, AI21 Labs, Alibaba Model Studio, Cohere, DeepSeek, Google Gemini, Mistral, Z AI) and inference providers (Cerebras, Cloudflare, GitHub Models, Groq, Hugging Face, Kilo Code, LLM7.io, ModelScope, NVIDIA NIM, Ollama Cloud, OpenRouter, OVHcloud, SiliconFlow). OVHcloud has an anonymous no-key tier (2 RPM) → usable for a live smoke test without credentials.
- Keys: env only. `ROUTER_KEY_<PROVIDER_SLUG>` plus optional `LLM_ROUTER_KEYS` JSON map. Anonymous tiers: OVHcloud, LLM7.io basic access.

## Proposed Changes

### 1. Layout

```
LLM-Router/
  requirements.txt
  .env.example
  catalog/
    data.json  schema.py  loader.py  task_types.py  rate_limits.py  grouping.py
  routing/
    request.py  decision.py  constraints.py  scoring.py  policies.py  health.py  budget.py
  providers/
    base.py  openai_adapter.py  cloudflare.py  ollama.py  huggingface.py  registry.py
  benchmark/
    tasks.json  checks.py  runner.py  metrics.py  router_eval.py  report.py
    assets/sample.png
  telemetry.py
  cli.py
  server.py
  tests/
```

`requirements.txt`: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `typer`, `pytest`, `pytest-asyncio`.

### 2. `catalog/`

**`schema.py`**
```python
class ModelEntry(BaseModel):
    id: str | None; name: str; context: str | None; maxOutput: str | None
    modality: str | None; rateLimit: str | None
    provider: str            # filled by loader, not in data.json
    key: str                 # f"{slug(provider)}/{id}", empty when id is None

class Catalog(BaseModel):
    lastUpdated: str | None
    providers: list[Provider]

class Provider(BaseModel):
    name: str; category: Literal["provider_api","inference_provider"]
    country: str | None; flag: str | None; url: str | None; baseUrl: str
    description: str | None; footnoteRef: int | None; models: list[ModelEntry]
```

**`loader.py`**
- `load_catalog(path: Path) -> Catalog` — read + validate; sets `ModelEntry.provider` and `key`.
- `slug(name: str) -> str` — lowercase, non-alphanumeric → `_`; used for keys, env vars, and `provider/model-id`.
- `routable_models(catalog) -> list[ModelEntry]` — drops `id is None` placeholder rows.
- `model_index(catalog) -> dict[str, ModelEntry]` — `"{slug}/{id}"` → entry; ambiguous bare ids map to a list of candidates.
- `resolve_model(catalog, model: str) -> list[ModelEntry]` — accepts `provider/model-id`, bare `model-id`, `auto`, `auto:<task_type>`; returns candidates (empty for `auto`).

**`task_types.py`**
```python
TASK_TYPES = ("text","reasoning","code","vision","audio","embeddings","rerank","roleplay")
def normalize_modality(modality: str | None) -> set[str]   # substring rules, 'text' fallback
def model_task_types(model: ModelEntry) -> set[str]
```
Rules (order matters, a model may match several): `reasoning` → reasoning; `code` → code; `Image|Vision|Multimodal` → vision; `Audio|Speech` → audio; `Embedding` → embeddings; `Rerank` → rerank; `roleplay` → roleplay; otherwise `text`.

**`rate_limits.py`**
```python
@dataclass(frozen=True)
class Limits:
    rpm: int|None; rps: int|None; rpd: int|None; tpm: int|None; tpd: int|None
    neurons_per_day: int|None; concurrent: int|None; parsed: bool; raw: str

def parse_rate_limit(raw: str | None) -> Limits     # handles thousands separators, ~, <=, 'Dynamic'/'Varies'/'unpublished'/'Credit-metered'
def effective_rpm(limits: Limits, cap: int = 5) -> int
def quota_cost(limits: Limits, requests: int, tokens: int) -> float   # 0..1 share of the daily budget consumed by one request
```

**`grouping.py`**
```python
def group(catalog, primary="category", secondary="task_type") -> dict[str, dict[str, list[ModelEntry]]]
def group_counts(grouped) -> dict[str, dict[str, int]]
def render_table(grouped) -> str
```
This is the "group by type" deliverable: category → task type → providers/models with context, max output, parsed rate limit.

### 3. `routing/`

**`request.py`**
```python
@dataclass(frozen=True)
class RoutingRequest:
    model: str | None = None
    task_type: str | None = None          # inferred by semantic_router when absent
    messages: list[dict] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    require: frozenset[str] = frozenset() # capability labels: vision, json, tools, code
    local_only: bool = False              # hard constraint (PQR pattern); here means self-hosted-only providers
    unavailable: frozenset[str] = frozenset()
    budget: Budget | None = None
    policy: str = "weighted"
```

**`decision.py`**
```python
@dataclass
class RouteDecision:
    provider: str; model_id: str; key: str
    reason: str            # from the telemetry allowlist only
    score: float
    excluded: dict[str, str]
    policy: str
    attempts: list[RouteDecision]   # populated by cascade/failover
```

**`constraints.py`** — hard filters applied *before* any scoring; each exclusion records a reason.
```python
def apply_constraints(candidates, req, keys, health) -> tuple[list[ModelEntry], dict[str, str]]
```
Excludes: `id is None`; no key and no anonymous tier; provider in `req.unavailable` or in health cooldown; task-type mismatch; missing required capability; `context` smaller than `input_tokens`; `maxOutput` smaller than `output_tokens`; provider budget exhausted in `QuotaLedger`.

**`scoring.py`**
```python
DEFAULT_WEIGHTS = {"quality": 0.40, "latency": 0.25, "quota_cost": 0.20, "reliability": 0.15}
def normalize_weights(weights: dict[str, float]) -> dict[str, float]   # non-negative, auto-normalized
def normalize(values: dict[str, float]) -> dict[str, float]            # min-max over the candidate pool
def candidate_metrics(model, metrics: dict[str, ModelMetrics]) -> dict[str, float]
def weighted_score(model, metrics, weights) -> float                   # lower cost is better (inverted)
```
`quality` comes from `benchmarks/scores.json`; `latency` from measured p50 (failures included); `reliability` from health success rate; `quota_cost` from `rate_limits.quota_cost`. Unmeasured values stay `None` and are reported as unknown rather than fabricated.

**`policies.py`**
```python
class AbstractRouter(ABC):
    name: str
    def route(self, req, candidates, ctx: RoutingContext) -> RouteDecision: ...

class ExplicitRouter(AbstractRouter)        # exact provider/model-id
class SemanticRouter(AbstractRouter)        # keyword intent classifier -> task_type, then QualityRouter within group
class QualityRouter(AbstractRouter)         # best measured quality within the task-type group
class CheapestQuotaRouter(AbstractRouter)   # lowest normalized quota cost
class WeightedPQRRouter(AbstractRouter)     # constraints first, then normalized weighted score
class HybridThresholdRouter(AbstractRouter) # weak-first; escalate if candidate pass_rate(task_type) < threshold
class CascadeRouter(AbstractRouter)         # FrugalGPT: ordered attempts, stop on success or budget exhaustion
POLICIES: dict[str, AbstractRouter]
```
`RoutingContext` carries `keys`, `health`, `ledger`, `metrics`, `weights`, `thresholds`, `min_n`.
`CLASSIFY_RULES` for `SemanticRouter`: ordered keyword → task-type table (e.g. `"classify|label|sentiment" → text`, `"json|schema" → json`, `"code|function|refactor" → code`, `"image|photo|ocr" → vision`), first match wins, fallback `text`. Deterministic and offline.

**`health.py`**
```python
class HealthTracker:
    def record(self, provider: str, outcome: str, latency_ms: int) -> None
    def success_rate(self, provider: str, window: int = 50) -> float
    def cooldown_until(self, provider: str) -> float | None    # set from Retry-After on 429
    def is_available(self, provider: str, now: float = time.time()) -> bool
    def snapshot(self) -> dict[str, dict]                      # for GET /health
```

**`budget.py`**
```python
@dataclass class Budget: max_requests: int; max_tokens: int; per_provider_max_requests: int
class QuotaLedger:
    def charge(self, provider: str, model: str, requests: int, tokens: int) -> None
    def remaining(self, provider: str) -> dict
    def is_exhausted(self, provider: str) -> bool
```

### 4. `providers/`

**`base.py`**
```python
class ProviderClient(Protocol):
    provider: str
    def supports_stream(self) -> bool: ...
    async def chat(self, model: str, payload: dict, stream: bool) -> dict | AsyncIterator[dict]: ...
    async def embeddings(self, model: str, payload: dict) -> dict: ...
    async def list_models(self) -> list[str]: ...
```

**`openai_adapter.py`**
```python
class OpenAIAdapter:
    def __init__(self, provider, key, client, timeout=60.0)
    def url(self, path: str = "chat/completions") -> str
    async def chat(self, model, payload, stream) -> dict | AsyncIterator[dict]
    async def embeddings(self, model, payload) -> dict
    async def list_models(self) -> list[str]
def normalize_openai_response(raw: dict) -> dict      # guarantees id/choices/usage shape
```
Single shared `httpx.AsyncClient`; one retry on 429/5xx honouring `Retry-After`; every response is normalized so `usage` is always present (some providers omit it).

**`cloudflare.py`** — `url()` requires `CF_ACCOUNT_ID`; `normalize_cf(raw)` unwraps `{"result": {...}}` into OpenAI shape.
**`ollama.py`** — `to_ollama_payload(payload)` (OpenAI messages → Ollama), `normalize_ollama_response(raw)`; `supports_stream() -> False`.
**`huggingface.py`** — overrides base URL to `https://router.huggingface.co/v1`.
**`registry.py`** — `def adapter_for(provider, key, client) -> ProviderClient` (registry keyed by provider slug, default `OpenAIAdapter`).

### 5. `benchmark/`

**`tasks.json`** — ~12 tasks, each `{id, task_type, prompt, checks[], timeout_s}`, covering every task type (≥2 text, ≥2 reasoning, ≥2 code, 1 vision, 1 embeddings, 1 rerank, 1 audio). Vision task uses `benchmark/assets/sample.png`. Each task declares `scored: true` so coverage gates are explicit.

**`checks.py`**
```python
def run_check(check: dict, answer: str, task: Task) -> bool
def check_exact(answer, expected) -> bool
def check_contains(answer, needles: list[str]) -> bool
def check_regex(answer, pattern) -> bool
def check_json_schema(answer, schema: dict) -> bool
def check_numeric(answer, expected: float, tol: float = 0.0) -> bool
def check_python_exec(code: str, stdin: str, expected_stdout: str, timeout: int = 10) -> bool
def check_count(answer, n: int) -> bool
def check_format(answer, rules: dict) -> bool          # language, max sentences, must-not-contain
def extract_code_block(answer) -> str | None
def normalize_answer(answer) -> str                    # lowercase, collapse whitespace, strip fences
```
`check_python_exec` runs model-generated code in a subprocess (stdin fed, 10 s timeout, no network) only for tasks explicitly marked `python_exec`.

**`runner.py`**
```python
@dataclass class PlannedCall: provider, model, task_id, task_type, reason   # reason = "skipped: no key" etc.
@dataclass class Row: provider, model, task_id, task_type, outcome, latency_ms, ttft_ms, prompt_tokens, completion_tokens, tokens_per_sec, http_status, error_class, answer_hash
def plan(catalog, tasks, opts) -> list[PlannedCall]              # dry-run, respects parsed limits
async def run_one(client, provider, model, task) -> Row
async def run(catalog, tasks, opts) -> list[Row]                 # sequential per provider, --concurrency across providers
def min_n_gate(rows, min_n: int = 5) -> dict[str, bool]          # candidate scored only if n_scored >= min_n
```
`outcome` ∈ `pass | fail | error | skipped`; `error_class` ∈ `ok | 429 | timeout | auth | schema | unsupported`. Raw answers are recorded (hash + answer file) so policy replay is reproducible offline.

**`metrics.py`**
```python
@dataclass class ModelMetrics: pass_rate, scored_n, p50_ms, p95_ms, tokens_per_sec, error_rate, coverage, quota_cost
def aggregate(rows) -> dict[str, ModelMetrics]        # percentiles include failed attempts, not success-biased
def pareto_frontier(metrics) -> list[str]             # quality vs normalized quota cost
def win_rate_at_budget(metrics, budget: Budget) -> dict[str, float]   # per policy, per task type
```

**`router_eval.py`** — offline replay, no extra API calls:
```python
def replay(policies: list[AbstractRouter], tasks, recorded: dict[str, dict[str, Row]], weights) -> dict[str, PolicyMetrics]
def policy_metrics(decisions, recorded) -> PolicyMetrics   # quality, mean/normalized cost, latency, escalation rate, budget used
def cost_quality_curve(policy_metrics) -> list[tuple[float, float]]
def render_plot(curves) -> str                            # ASCII + SVG, no external assets
```

**`report.py`**
- `write_json(rows, path)`, `write_csv(rows, path)` (long format, RouterBench-style), `write_scores(metrics, path)` (consumed by `routing/scoring.py`), `write_html(metrics, policy_metrics, path)`.
- HTML grouped by provider category → task type: provider, model, scored n, pass rate, p50/p95, tokens/s, 429 rate, error class, Pareto flag; then a policy table with win-rate-at-budget.

### 6. `telemetry.py`

```python
ALLOWED_REASONS = frozenset({"explicit","task_type","quality","cost","reliability","threshold","cascade","no_key","capability","context","budget","unavailable","health","unknown"})
ALLOWED_OUTCOMES = frozenset({"pass","fail","error","skipped","policy_blocked"})
def log_route(decision, latency_ms: int, outcome: str) -> None   # JSONL: route, reason, latency_ms, outcome only
def summarize(jsonl: Path, taxonomy: dict) -> dict               # rejects unknown values; percentiles include failures
```
Content-free: no prompts, responses, identifiers, or paths.

### 7. `server.py` (FastAPI)

- `POST /v1/chat/completions` — OpenAI-shaped body; `model` resolved by `resolve_model` + `POLICIES[req.policy]`; `stream` passed through for OpenAI-compatible adapters; response adds `router: {provider, model, reason, score, policy, attempts}`; logs telemetry.
- `POST /v1/embeddings` — task type `embeddings` only.
- `GET /v1/models` — routable models in `provider/model-id` form.
- `GET /providers` — grouped catalog (`grouping.group`).
- `GET /health` — `HealthTracker.snapshot()` + key availability.
- `GET /` — plain-text usage examples.
- `errors.py`: `def error_json(message, type, provider=None, model=None) -> dict` → `{"error": {...}}`.

### 8. `cli.py` (typer)

- `refresh` — download `data.json` into `catalog/data.json`
- `groups [--task-type X] [--format json|table]`
- `route "model" "prompt" [--policy weighted|quality|cheapest|hybrid|cascade|semantic] [--require code,json] [--local-only]`
- `bench [--providers ...] [--task-type ...] [--models ...] [--max-requests 15] [--concurrency 4] [--dry-run]`
- `eval [--budget N] [--policy ...]` — replay policies against recorded benchmark results
- `summary` — summarize telemetry JSONL
- `serve [--port 8000]`

## Assumptions & Decisions

- Python 3.11+; `data.json` is committed as a snapshot and refreshed by `cli.py refresh`, so runs are reproducible.
- Placeholder catalog rows (`id: null`) are reported but never routed or benchmarked.
- `stream` is only pass-through for OpenAI-compatible adapters.
- Cost = normalized quota consumption, not money (all tiers are free); unmeasured values stay `None` and are reported as unknown, never fabricated.
- Candidates are scored only when `n_scored >= min_n` (default 5); below that they are reported as coverage-only, not ranked.
- Latency percentiles include failed and interrupted attempts.
- No LLM judge; scoring is deterministic, so results are reproducible from recorded answers.
- Keys come from env only, nothing written to disk; `.env.example` lists one variable per provider.
- `local_only` here means "self-hosted-only provider" as a hard constraint flag; it is metadata, not a network attestation.

## Verification

1. `pip install -r requirements.txt`
2. `pytest` unit tests (no network):
   - `rate_limits.parse_rate_limit` against every real string in `catalog/data.json`
   - `task_types.normalize_modality` against every modality value in the catalog
   - each `check_*` against fixed pass/fail answers
   - `constraints.apply_constraints` (no key, capability, context, budget, unavailable)
   - each policy in `POLICIES` on a fixed fixture pool, asserting `reason` and `excluded`
   - `min_n_gate`, `aggregate`, `pareto_frontier`, `win_rate_at_budget` on fixture rows
   - adapters with mocked httpx transport; `normalize_cf` / `normalize_ollama_response`
   - `telemetry.summarize` rejects an unknown reason value
3. `python cli.py groups` — 2 categories × task types, counts match the catalog.
4. `python cli.py bench --dry-run` — plan respects parsed rate limits, shows skip reasons.
5. Live keyless smoke: `python cli.py route "ovhcloud/Meta-Llama-3_1-8B-Instruct" "Reply with exactly: PONG"` (anonymous tier, 2 RPM).
6. `python cli.py bench --providers ovhcloud --max-requests 3` — produces `results.json`, `results.csv`, `scores.json`, `report.html`.
7. `python cli.py eval --budget 20` — replay policies on recorded results, prints cost-vs-quality and win-rate-at-budget.
8. `python cli.py serve` + OpenAI SDK call with `model=auto:text` — response includes `router.reason`, telemetry JSONL row has only the four allowed fields.
