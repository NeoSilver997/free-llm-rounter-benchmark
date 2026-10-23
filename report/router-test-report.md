# Router function test report

Run: `python -m pytest -q --basetemp=C:\AI_Server\pytest-tmp -p no:cacheprovider`
Environment: Python 3.14.0, pytest 9.1.1, pytest-asyncio 1.4.0 (strict mode, no pytest config), httpx.
No network calls — adapters run through `httpx.MockTransport`.

**Result: 68 passed, 0 failed (1.87 s).**

| Module | Tests | Result |
| --- | --- | --- |
| `tests/test_routing.py` | 24 | 24 passed |
| `tests/test_benchmark.py` | 21 | 21 passed |
| `tests/test_web.py` | 23 | 23 passed |

## Coverage against the plan's Verification item 2

| Plan bullet | Tests | Status |
| --- | --- | --- |
| `parse_rate_limit` against every real string in `catalog/data.json` | `test_parse_rate_limit_against_every_real_catalog_string`, `test_parse_rate_limit_edge_cases` | covered |
| `normalize_modality` against every catalog modality value | `test_normalize_modality_against_every_catalog_value` | covered |
| each `check_*` against fixed pass/fail answers | `test_exact_contains_regex_numeric_count`, `test_json_schema_and_format`, `test_python_exec_check_runs_generated_code`, `test_run_check_rejects_unknown_type` | covered |
| `apply_constraints` (no key, capability, context, budget, unavailable) | `test_apply_constraints_records_a_reason_for_every_exclusion`, `test_constraint_order_and_availability` | covered |
| each policy in `POLICIES` on a fixed fixture pool, asserting `reason` and `excluded` | `test_each_policy_on_the_fixture_pool`, `test_policy_reasons_and_fallbacks` | covered (all 7) |
| `min_n_gate`, `aggregate`, `pareto_frontier`, `win_rate_at_budget` | `test_min_n_gate`, `test_aggregate_percentiles_include_failed_attempts`, `test_aggregate_counts_errors_and_keeps_unmeasured_unknown`, `test_pareto_frontier_prefers_higher_quality_and_lower_cost`, `test_win_rate_at_budget_only_counts_models_inside_the_budget`, `test_scoring_normalisation_and_min_n_gate` | covered |
| adapters with mocked httpx transport; `normalize_cf` / `normalize_ollama_response` | `test_openai_adapter_chat_and_normalise`, `test_openai_adapter_retries_once_and_classifies_errors`, `test_openai_adapter_streaming_passes_reasoning_content_through`, `test_openai_adapter_embeddings_and_list_models`, `test_cloudflare_adapter_url_and_normalise`, `test_ollama_adapter_merges_ndjson_chunks`, `test_huggingface_base_and_registry_dispatch` | covered |
| `telemetry.summarize` rejects an unknown reason | `test_telemetry_rejects_unknown_values_and_keeps_four_fields` | covered |

Extra router coverage beyond the plan: `classify`, `resolve_model` / `unique_prefix`, `group` / `group_counts` / `render_table`, `is_local` / `model_capabilities`, `HealthTracker`, `parse_tokens` / `effective_rpm` / `rpm_from_tpm` / `quota_cost`, `normalize_openai_response`, `normalize_weights`.

## Ground truth the assertions are pinned to

Catalog: 16 providers, 116 routable models, 5 placeholder rows (`id: null`, never routable), snapshot `lastUpdated: 2026-10-05`.

Group counts (`python cli.py groups --format json`) — a multi-modality model appears in more than one bucket, so the buckets sum to 128 over 116 models:

| Category | audio | code | reasoning | text | vision |
| --- | --- | --- | --- | --- | --- |
| `inference_provider` | 1 | 6 | 3 | 51 | 20 |
| `provider_api` | 9 | 3 | 6 | 10 | 19 |

Rate limits: 20 distinct strings, 16 parseable, 4 opaque (`—`, `Credit-metered`, `Dynamic quotas + dynamic concurrency`, `Session/weekly limits (unpublished)`) — the opaque ones must parse to an empty `Limits`, not a fabricated number.

Modality: 16 distinct values, all map through `normalize_modality`; nothing matches falls back to `{"text"}`.

Fixture pool used by the policy tests:

| Key | Task type | Rate limit | Context / maxOutput |
| --- | --- | --- | --- |
| `aion_labs/aion-labs/aion-rp-llama-3.1-8b` | text | 15 RPM, 20K TPD | 32K / 32K |
| `cloudflare_workers_ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast` | text | 10K neurons/day (shared) | 24K |
| `mistral_ai/codestral-2508` | code | ~1 RPS, 500K TPM | 128K |

Policy expectations on that pool (a `text` request excludes `codestral-2508` as `task_type` in every policy):

| Policy | Chosen | `reason` |
| --- | --- | --- |
| explicit | cloudflare | `explicit` |
| semantic | cloudflare | `quality` |
| quality | cloudflare | `quality` |
| cheapest | aion | `cost` |
| weighted | cloudflare | `quality` |
| hybrid | cloudflare | `quality` (aion when its pass rate ≥ 0.70 threshold → `threshold`) |
| cascade | aion | `cascade`, attempts cheapest-first |

Constraint order verified by test: placeholder → unknown provider → unavailable → local_only → **no_key** → health → budget → task_type → capability → context → maxOutput. `no_key` is checked before the task type, so a code model without a key is recorded as `no_key`, not `task_type`.

Scoring: weights are normalised and negative/unknown axes dropped (`{"quality":2,"latency":1,"bogus":5,"reliability":-1}` → quality 2/3, latency 1/3, reliability 0.0); `quota_cost` is inverted; unmeasured axes are dropped and their weight is redistributed; candidates are ranked only when `n_scored >= min_n` (5), otherwise the rank is empty.

Telemetry: exactly four fields (`route`, `reason`, `latency_ms`, `outcome`); 14 allowed reasons, 5 allowed outcomes; `log_route` raises on an unknown value and `summarize()` reports `rejected_rows` instead of bucketing them.

## Live telemetry summary (`python cli.py summary`)

```json
{"n": 9, "rejected": 0, "rejected_rows": [],
 "outcomes": {"error": 1, "pass": 6, "policy_blocked": 2},
 "overall": {"p50_ms": 1277.0, "p95_ms": 12413.4}}
```

## Offline policy replay (`python cli.py eval --budget 20`)

```
policy                quality  cost      latency   escal  budget  win rate at budget=20
cascade              1.000  0.000  8383.167  0.000  0.000733  reasoning=1.00, text=1.00
cheapest             1.000  0.000  8383.167  0.000  0.000733  reasoning=1.00, text=1.00
explicit             1.000  0.000  2905.167  0.000  0.000333  reasoning=1.00, text=1.00
hybrid               0.833  0.000  3068.333  0.000  0.000333  reasoning=1.00, text=0.75
quality              1.000  0.000  3607.167  0.000  0.000333  reasoning=1.00, text=1.00
semantic             1.000  0.000  3607.167  0.000  0.000333  reasoning=1.00, text=1.00
weighted             1.000  0.000  2905.167  0.000  0.000333  reasoning=1.00, text=1.00

win rate at budget: reasoning 1.00, text 1.00
```

Caveat: the replay cannot currently discriminate policy quality — every ranked model has a recorded pass rate of 1.0, so all policies score 1.000 and the only differences are latency and budget share. The policies separate once there are recorded failures above the min-N gate (the unit tests cover that separation on fixture rows).

## Other verification runs

`python cli.py bench --dry-run` — 329 attempts over 103 models; `run: 27`, `skipped: no key: 288`, `skipped: provider request cap: 14`. Skip reasons are recorded per attempt, matching the parsed rate limits.

`python cli.py route "ovhcloud/Meta-Llama-3_1-8B-Instruct" "Reply with exactly: PONG"` — `task_type=text`, route `ovhcloud_ai_endpoints/Mistral-Nemo-Instruct-2407`, `reason=quality score=0.9925`; the anonymous providers stay routable without a key, non-text models are excluded as `task_type`.

`python cli.py route auto:code "Write a python function..." --policy hybrid` — `route=none`, `reason=unknown`: every code-capable model is excluded as `no_key`, so the router reports no route instead of guessing.

## Issues found while completing the tests

1. **Circular import** — importing `telemetry` raised `ImportError: cannot import name 'ALLOWED_REASONS'` (`routing/__init__` → `routing.decision` → `telemetry` → `routing.health`). Fixed in [telemetry.py](file:///c:/AI_Server/ClaudeCode/LLM-Router/telemetry.py#L7-L33) by defining the allowlists before the `routing.health` import.
2. **Constraint-order assertion** — the test expected `task_type` for a keyless code model; the actual order records `no_key` first. Assertion corrected to match the documented order (this is behaviour, not a bug).

## UI flow review (live browser, http://127.0.0.1:8010)

Fixed and re-verified in the browser:

1. **Open detail row lost on every refresh** — `refreshLog()` rebuilds the tbody, so the open state now travels by `data-time`. Verified: open before and after a 5 s refresh, and the detail row is still adjacent to its request row.
2. **Filter and sort could separate a detail row from its request row** — both now handle the pair as one unit. Verified on `/benchmark`: sort then filter, no row whose detail state disagreed with its request row.
3. **Silent truncation** — `LogEntry.truncated` is a property. As a method the home template read a bound method, which is always truthy, so every row claimed `text capped`; now the note appears only when the stored text hit `MAX_TEXT`. Verified live: a 2050-char prompt stored at 2000, and the page showed exactly one `text capped` note.
4. **Client and page disagreed on the limit** — the client reads `data-limit` from the tbody instead of hardcoding 100.

5. **Rows and sortable headers were mouse-only** — request rows carry `tabindex`/`aria-expanded` and toggle on Enter or Space; sortable headers are focusable and activate from the keyboard; `aria-sort` reports the current direction. Verified live: Enter and Space both opened a detail row, and Enter on `p50 ms` sorted ascending with unmeasured values last.
6. **Focus was dropped on every refresh** — the rebuild discards the focused row, so `refreshLog()` now re-applies focus to the same entry. Verified: focus survived a 5 s refresh with the detail row still open.
7. **Favicon 404 in the console** — every page requested `/favicon.ico` and logged an error; `base.html` declares an empty icon, and the console is now clean.

Still open from the review: detail rows are rendered eagerly for all 100 entries (about 400 KB per refresh), and the Jinja and JS formatters differ (`fmt` prints `120`, `show` prints `120.0`).

## Still open

- Live verification item 8 (`serve` + OpenAI SDK call with `model=auto:text`) was verified manually earlier; it is not part of this offline suite.
