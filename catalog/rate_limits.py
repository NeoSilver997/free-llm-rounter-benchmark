from __future__ import annotations

import re
from dataclasses import dataclass

_MULTIPLIER = {"K": 1000, "M": 1_000_000}
_UNITS = ("RPM", "RPS", "RPD", "TPM", "TPD")


@dataclass(frozen=True)
class Limits:
    rpm: int | None = None
    rps: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    rph: int | None = None
    neurons_per_day: int | None = None
    concurrent: int | None = None
    parsed: bool = False
    raw: str = ""


def _value(number: str, multiplier: str = "") -> int | None:
    try:
        base = float(number.replace(",", ""))
    except ValueError:
        return None
    return int(round(base * _MULTIPLIER.get(multiplier.upper(), 1)))


def parse_tokens(value: str | None) -> int | None:
    """Catalog token sizes ('128K', '1M', '131072') -> int; None when unparseable."""
    if not value:
        return None
    match = re.search(r"(\d[\d,\.]*)\s*([KM]?)", value)
    if not match:
        return None
    return _value(match.group(1), match.group(2))


def parse_rate_limit(raw: str | None) -> Limits:
    """Free-text rate limit -> structured limits. Unparseable values stay None."""
    raw = raw or ""
    if not raw:
        return Limits(raw="")

    fields: dict[str, int] = {}
    for number, multiplier, unit in re.findall(
        r"(\d[\d,\.]*)\s*([KM]?)\s*(%s)" % "|".join(_UNITS), raw, re.I
    ):
        key = unit.upper().lower()
        value = _value(number, multiplier)
        if value is not None and key not in fields:
            fields[key] = value

    def _single(pattern: str, key: str) -> None:
        match = re.search(pattern, raw, re.I)
        if match:
            value = _value(match.group(1), match.group(2) if match.lastindex >= 2 else "")
            if value is not None:
                fields[key] = value

    _single(r"(\d[\d,\.]*)\s*([KM]?)\s*neurons/day", "neurons_per_day")
    _single(r"(\d[\d,\.]*)\s*([KM]?)\s*(?:req|requests)/hr", "rph")
    _single(r"(\d[\d,\.]*)\s*([KM]?)\s*concurrent", "concurrent")

    return Limits(parsed=bool(fields), raw=raw, **fields)


def effective_rpm(limits: Limits, cap: int = 5) -> int:
    """Requests per minute the runner is allowed to use for a provider."""
    if limits.rpm:
        return min(limits.rpm, cap)
    if limits.rps:
        return min(limits.rps * 60, cap)
    if limits.rph:
        return max(1, min(limits.rph // 60, cap))
    if limits.concurrent:
        return min(limits.concurrent, cap)
    return cap


def quota_cost(limits: Limits, requests: int, tokens: int) -> float | None:
    """Share (0..1) of the provider's published daily budget a request consumes.

    Free tiers have no monetary cost, so 'cost' means quota consumption.
    Returns None when nothing is measurable rather than fabricating a value.
    """
    shares: list[float] = []
    if limits.rpd:
        shares.append(requests / limits.rpd)
    if limits.tpd:
        shares.append(tokens / limits.tpd)
    if limits.neurons_per_day:
        shares.append(tokens / limits.neurons_per_day)
    if limits.rph:
        shares.append(requests / (limits.rph * 24))
    if limits.tpm:
        shares.append(tokens / (limits.tpm * 1440))
    if not shares:
        return None
    return min(max(shares), 1.0)
