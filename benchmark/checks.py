from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CHECK_TYPES = frozenset(
    {"exact", "contains", "regex", "json_schema", "numeric", "python_exec", "count", "format"}
)

# The reachability probe: one tiny chat request with no checks, so it only records
# whether the model answers and how long it took. Shared by cli.py probe and the UI Retest.
PROBE_PROMPT = "Reply with the single word ok."


@dataclass
class Task:
    id: str
    task_type: str
    prompt: str
    checks: list[dict] = field(default_factory=list)
    timeout_s: int = 60
    scored: bool = True
    endpoint: str = "chat"
    stdin: str = ""
    image: str | None = None


def load_tasks(path: Path | str) -> list[Task]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = [Task(**item) for item in raw]
    for task in tasks:
        for check in task.checks:
            if check["type"] not in CHECK_TYPES:
                raise ValueError(f"{task.id}: unknown check type {check['type']!r}")
    return tasks


def normalize_answer(answer: str) -> str:
    """Lowercase, collapse whitespace, strip code fences."""
    text = re.sub(r"```[a-z]*|```", " ", answer or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_code_block(answer: str) -> str | None:
    match = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", answer or "", re.S)
    return match.group(1).strip() if match else None


def check_exact(answer: str, expected: str) -> bool:
    return normalize_answer(answer) == normalize_answer(expected)


def check_contains(answer: str, needles: list[str]) -> bool:
    text = normalize_answer(answer)
    return all(n.lower() in text for n in needles)


def check_regex(answer: str, pattern: str) -> bool:
    return bool(re.search(pattern, answer or "", re.I | re.S))


def check_numeric(answer: str, expected: float, tol: float = 0.0) -> bool:
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", answer or "")]
    if not numbers:
        return False
    return any(abs(n - expected) <= tol for n in numbers)


def check_count(answer: str, n: int) -> bool:
    """Exactly n items on separate lines, or n enumerated items."""
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    return len(lines) == n


def check_json_schema(answer: str, schema: dict) -> bool:
    try:
        value = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        block = extract_code_block(answer)
        if not block:
            return False
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            return False

    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return False
        for key in schema.get("required", []):
            if key not in value:
                return False
        for key, expected in schema.get("properties", {}).items():
            if key in value and not _matches_type(value[key], expected):
                return False
        return True
    return _matches_type(value, schema.get("type", "any"))


def _matches_type(value, expected) -> bool:
    """Accepts a type name or a JSON-Schema property object like {"type": "string"}."""
    if isinstance(expected, dict):
        expected = expected.get("type", "any")
    kinds = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    if expected == "any":
        return True
    return isinstance(value, kinds.get(expected, object))


def check_format(answer: str, rules: dict) -> bool:
    text = (answer or "").strip()
    max_sentences = rules.get("max_sentences")
    if max_sentences:
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) > max_sentences:
            return False
    for needle in rules.get("must_not_contain", []):
        if needle.lower() in text.lower():
            return False
    if rules.get("starts_with") and not normalize_answer(text).startswith(normalize_answer(rules["starts_with"])):
        return False
    language = rules.get("language")
    if language == "en" and re.search(r"[\u3000-\u30ff\u4e00-\u9fff]", text):
        return False
    return True


def check_python_exec(code: str, stdin: str, expected_stdout: str, timeout: int = 10) -> bool:
    """Runs model-generated code in a sandboxed subprocess (stdin fed, no network calls)."""
    if not code:
        return False
    try:
        proc = subprocess.run(
            ["python", "-I", "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    return normalize_answer(proc.stdout) == normalize_answer(expected_stdout)


def run_check(check: dict, answer: str, task: Task) -> bool:
    kind = check["type"]
    if kind == "exact":
        return check_exact(answer, check["expected"])
    if kind == "contains":
        return check_contains(answer, check["needles"])
    if kind == "regex":
        return check_regex(answer, check["pattern"])
    if kind == "json_schema":
        return check_json_schema(answer, check["schema"])
    if kind == "numeric":
        return check_numeric(answer, check["expected"], check.get("tol", 0.0))
    if kind == "count":
        return check_count(answer, check["n"])
    if kind == "format":
        return check_format(answer, check["rules"])
    if kind == "python_exec":
        code = extract_code_block(answer) or answer
        return check_python_exec(code, task.stdin, check["expected_stdout"], check.get("timeout", 10))
    raise ValueError(f"unknown check type: {kind!r}")
