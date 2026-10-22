from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from routing.health import percentile

ALLOWED_REASONS = frozenset(
    {
        "explicit",
        "task_type",
        "quality",
        "cost",
        "reliability",
        "threshold",
        "cascade",
        "no_key",
        "capability",
        "context",
        "budget",
        "unavailable",
        "health",
        "unknown",
    }
)
ALLOWED_OUTCOMES = frozenset({"pass", "fail", "error", "skipped", "policy_blocked"})
DEFAULT_LOG = Path("routes.jsonl")

# Content-free telemetry: no prompts, responses, identifiers or paths.
FIELDS = ("route", "reason", "latency_ms", "outcome")


def log_route(decision, latency_ms: int, outcome: str, path: Path | str = DEFAULT_LOG) -> None:
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome!r}")
    if decision.reason not in ALLOWED_REASONS:
        raise ValueError(f"unknown reason: {decision.reason!r}")

    row = {
        "route": decision.key,
        "reason": decision.reason,
        "latency_ms": int(latency_ms),
        "outcome": outcome,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(jsonl: Path | str, taxonomy: dict) -> dict:
    """Aggregate the JSONL log. Unknown values are rejected, not bucketed."""
    reasons = taxonomy.get("reasons", ALLOWED_REASONS)
    outcomes = taxonomy.get("outcomes", ALLOWED_OUTCOMES)

    rows: list[dict] = []
    rejected: list[dict] = []
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bad = [f for f in FIELDS if f not in row]
        if bad:
            rejected.append({"line": line, "missing": bad})
            continue
        if row["reason"] not in reasons or row["outcome"] not in outcomes:
            rejected.append({"line": line, "unknown": [row["reason"], row["outcome"]]})
            continue
        rows.append(row)

    latencies = [row["latency_ms"] for row in rows]
    by_reason: dict[str, dict] = {}
    for reason in sorted({row["reason"] for row in rows}):
        group = [row for row in rows if row["reason"] == reason]
        by_reason[reason] = {
            "n": len(group),
            "outcomes": dict(sorted(Counter(r["outcome"] for r in group).items())),
            # Percentiles include failures and policy-blocked routes, not success-biased.
            "p50_ms": percentile([r["latency_ms"] for r in group], 0.50),
            "p95_ms": percentile([r["latency_ms"] for r in group], 0.95),
        }

    return {
        "n": len(rows),
        "rejected": len(rejected),
        "rejected_rows": rejected,
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "overall": {
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
        },
        "by_reason": by_reason,
    }
