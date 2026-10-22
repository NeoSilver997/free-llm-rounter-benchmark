from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from routing.health import percentile
from telemetry import ALLOWED_OUTCOMES

DEFAULT_LOG = Path("ui_log.jsonl")
MAX_ENTRIES = 200

# Columns shown on the home page. Values that cannot be measured stay None and render as 'unknown'.
COLUMNS = (
    "model",
    "time",
    "status",
    "prompt",
    "reused",
    "output",
    "prefill_tps",
    "decode_tps",
    "cache_hit_rate",
    "duration_ms",
)
MAX_TEXT = 500


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LogEntry:
    time: str
    model: str
    status: str
    prompt: str
    output: str
    reused: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    cache_hit_rate: float | None = None
    duration_ms: int | None = None
    ttft_ms: int | None = None
    route_reason: str = ""
    outcome: str = "error"

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "model": self.model,
            "status": self.status,
            "prompt": self.prompt,
            "reused": self.reused,
            "output": self.output,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "cache_hit_rate": self.cache_hit_rate,
            "duration_ms": self.duration_ms,
            "ttft_ms": self.ttft_ms,
            "route_reason": self.route_reason,
            "outcome": self.outcome,
        }


def derive_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: int | None,
    duration_ms: int | None,
    cached_tokens: int | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Prefill/decode split needs TTFT; cache hit rate needs reported cached tokens.

    When TTFT is unavailable (non-streaming calls), decode is measured over the whole call.
    Nothing is estimated: unmeasured values stay None.
    """
    prefill = round(prompt_tokens / (ttft_ms / 1000), 1) if ttft_ms and prompt_tokens else None

    decode = None
    if completion_tokens and duration_ms:
        window = (duration_ms - (ttft_ms or 0)) / 1000
        decode = round(completion_tokens / window, 1) if window > 0 else round(completion_tokens / (duration_ms / 1000), 1)

    hit = round(cached_tokens / prompt_tokens, 4) if cached_tokens and prompt_tokens else None
    return prefill, decode, hit


class RequestLog:
    """Ring buffer of recent requests, persisted as JSONL so restarts keep history."""

    def __init__(self, path: Path | str = DEFAULT_LOG, max_entries: int = MAX_ENTRIES):
        self.path = Path(path)
        self.max_entries = max_entries
        self.entries: list[LogEntry] = []

    def load(self) -> list[LogEntry]:
        if not self.path.exists():
            return []
        entries: list[LogEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entries.append(LogEntry(**{k: row.get(k) for k in LogEntry.__dataclass_fields__}))
        self.entries = entries[-self.max_entries :]
        return self.entries

    def record(self, entry: LogEntry) -> LogEntry:
        if entry.outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"unknown outcome: {entry.outcome!r}")
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")
        return entry

    def recent(self, n: int = 10) -> list[LogEntry]:
        return list(reversed(self.entries[-n:]))

    def stats(self) -> dict:
        durations = [e.duration_ms for e in self.entries if e.duration_ms]
        return {
            "n": len(self.entries),
            "outcomes": {o: sum(1 for e in self.entries if e.outcome == o) for o in sorted({e.outcome for e in self.entries})},
            "p50_ms": percentile(durations, 0.50),
            "p95_ms": percentile(durations, 0.95),
        }
