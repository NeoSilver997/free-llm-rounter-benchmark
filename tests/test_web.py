from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient

import server
import benchmark.runner as runner
from benchmark.report import write_csv
from benchmark.runner import Row, read_rows
from catalog.loader import load_catalog, routable_models
from catalog.rate_limits import rpm_from_tpm
import config as config_module
from config import AppConfig, ModelConfig, env_keys, load_config, save_config
from providers.registry import adapter_for
from request_log import LogEntry, RequestLog, derive_metrics


def groq_key() -> str:
    return [m.key for m in routable_models(load_catalog()) if m.provider == "groq"][0]


def test_env_keys_and_precedence(monkeypatch):
    monkeypatch.setenv("ROUTER_KEY_GROQ", "env-key")
    monkeypatch.setenv("LLM_ROUTER_KEYS", json.dumps({"groq": "map-key", "openrouter": "other"}))
    assert env_keys()["groq"] == "env-key"  # a per-provider env var overrides the bulk map

    runtime = AppConfig(models=[ModelConfig(provider="groq", model_key="groq/x", api_key="runtime")])
    assert runtime.key_for("groq/x") == "runtime"

    env_only = AppConfig(models=[ModelConfig(provider="groq", model_key="groq/x")], api_key="global")
    assert env_only.key_for("groq/x") == "env-key"

    assert AppConfig(api_key="global").key_for("unknown/model") == "global"


def test_config_round_trip_never_persists_keys(tmp_path):
    config = AppConfig(api_host="http://localhost:9000")
    config.models.append(
        ModelConfig(provider="groq", model_key="groq/llama", keyword="code", host="https://override.test/v1", api_key="secret", selected=True)
    )
    path = tmp_path / "ui_config.json"
    save_config(config, path)
    assert "secret" not in path.read_text(encoding="utf-8")

    loaded = load_config(path)
    assert loaded.api_host == "http://localhost:9000"
    assert loaded.models[0].keyword == "code"
    assert loaded.models[0].selected
    assert loaded.models[0].api_key == ""

    assert loaded.keyword_match("please refactor this code") is loaded.models[0]
    assert loaded.keyword_match("hello") is None


def test_host_falls_back_to_catalog_base_url():
    config = AppConfig(models=[ModelConfig(provider="groq", model_key="groq/llama")])
    assert config.host_for("groq/llama").startswith("https://")
    config.models[0].host = "https://override.test/v1"
    assert config.host_for("groq/llama") == "https://override.test/v1"


def test_derive_metrics_only_when_measurable():
    prefill, decode, hit = derive_metrics(200, 300, 1000, 5000, 100)
    assert (prefill, decode, hit) == (200.0, 75.0, 0.5)  # 300 tokens over the 4 s decode window

    # No time-to-first-token: prefill cannot be split, decode covers the whole call.
    prefill, decode, hit = derive_metrics(200, 300, None, 3000)
    assert prefill is None and decode == 100.0 and hit is None

    assert derive_metrics(0, 0, None, None) == (None, None, None)


def test_request_log_ring_buffer_and_round_trip(tmp_path):
    path = tmp_path / "ui_log.jsonl"
    log = RequestLog(path, max_entries=3)
    for i in range(5):
        log.record(LogEntry(time=f"2026-10-22T00:00:0{i}Z", model="groq/x", status="ok", prompt="p", output="o", outcome="pass", duration_ms=100 + i))

    assert [e.time for e in log.recent(2)] == ["2026-10-22T00:00:04Z", "2026-10-22T00:00:03Z"]
    assert len(log.entries) == 3

    # The JSONL keeps full history; the in-memory buffer is capped.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5
    reloaded = RequestLog(path, max_entries=3).load()
    assert len(reloaded) == 3
    assert reloaded[-1].model == "groq/x"

    with pytest.raises(ValueError, match="unknown outcome"):
        log.record(LogEntry(time="t", model="groq/x", status="ok", prompt="p", output="o", outcome="maybe"))


class StubAdapter:
    provider = "groq"

    def supports_stream(self) -> bool:
        return True

    async def chat(self, model: str, payload: dict, stream: bool) -> dict:
        await asyncio.sleep(0.05)  # nonzero duration so decode t/s is measurable
        return {
            "id": "1",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 100}},
        }

    async def embeddings(self, model: str, payload: dict) -> dict:
        return {"data": []}

    async def list_models(self) -> list[str]:
        return []


@pytest.fixture
def client(tmp_path, monkeypatch):
    key = groq_key()
    config = AppConfig(models=[ModelConfig(provider="groq", model_key=key, keyword="pong", selected=True)])
    monkeypatch.setattr(server, "CONFIG", config)
    monkeypatch.setattr(server, "LOG", RequestLog(tmp_path / "ui_log.jsonl"))
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(server, "ROUTE_LOG", tmp_path / "routes.jsonl")
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "ui_config.json")
    monkeypatch.setenv("ROUTER_KEY_GROQ", "test-key")
    monkeypatch.setattr(server, "adapter_for", lambda provider, key, client, base="": StubAdapter())
    # run_one resolves its own adapter, so the retest path needs the stub too.
    monkeypatch.setattr(runner, "adapter_for", lambda provider, key, client, base="", timeout=60.0: StubAdapter())
    return TestClient(server.app)


def test_pages_render(client):
    for path in ("/", "/config", "/benchmark", "/health"):
        assert client.get(path).status_code == 200

    home = client.get("/").text
    assert "VRAM hit rate" in home and "Prefill t/s" in home
    assert "Decode t/s" in home

    config_page = client.get("/config").text
    for column in ("Provider", "Model key", "Host", "Keyword", "API key"):
        assert column in config_page

    # No results yet: the benchmark page says so instead of inventing numbers.
    assert "No benchmark results yet" in client.get("/benchmark").text


def test_chat_records_the_request(client):
    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "PONG"

    entry = client.get("/api/log").json()["entries"][0]
    assert entry["status"] == "ok" and entry["outcome"] == "pass"
    assert entry["output"] == "PONG"
    assert entry["reused"] == 100
    assert entry["cache_hit_rate"] == 0.5
    assert entry["prefill_tps"] is None  # non-streaming: no time-to-first-token
    assert entry["decode_tps"] is not None
    assert entry["route_reason"] == "explicit"  # configured keyword 'pong' wins


def test_no_key_is_excluded_before_scoring(client, monkeypatch):
    monkeypatch.delenv("ROUTER_KEY_GROQ")
    client.post("/api/config", json={"models": [{"model_key": groq_key(), "keyword": ""}]})
    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 400
    assert response.json()["error"]["excluded"][groq_key()] == "no_key"


def test_host_override_reaches_the_adapter():
    client = httpx.AsyncClient()
    adapter = adapter_for("groq", "k", client, base="https://override.test/v1")
    assert adapter.url("chat/completions") == "https://override.test/v1/chat/completions"
    assert adapter.headers() == {"Authorization": "Bearer k"}

    # No override -> the catalog baseUrl.
    default = adapter_for("groq", "", client)
    assert default.url("chat/completions").startswith("https://api.groq.com")
    assert default.headers() == {}


def test_provider_key_beats_env_and_tpm_default(monkeypatch):
    monkeypatch.setenv("ROUTER_KEY_GROQ", "env-key")
    config = AppConfig(
        models=[ModelConfig(provider="groq", model_key="groq/x"), ModelConfig(provider="groq", model_key="groq/y", api_key="model-key")],
        provider_keys={"groq": "provider-key"},
    )
    assert config.key_for("groq/x") == "provider-key"  # provider key beats the env key
    assert config.key_for("groq/y") == "model-key"     # a model key still wins
    assert config.tpm_for("groq") == 20
    config.provider_tpm["groq"] = 40
    assert config.tpm_for("groq") == 40
    assert rpm_from_tpm(20, 36) == 1 and rpm_from_tpm(1000, 200) == 5 and rpm_from_tpm(None, 200) is None


def test_config_page_groups_by_provider(client):
    page = client.get("/config").text
    assert 'data-provider="groq"' in page and "Provider key" in page and "TPM" in page

    saved = client.post("/api/config", json={"providers": [{"provider": "groq", "api_key": "secret", "tpm": 25}], "models": []})
    assert saved.json()["saved"] is True

    api = client.get("/api/config").json()
    assert api["providers"]["groq"]["tpm"] == 25
    assert "secret" not in json.dumps(api)  # provider keys are runtime-only as well


def test_config_page_shows_last_result_per_model(client):
    key = groq_key()
    write_csv(
        [Row(provider="groq", model=key.split("/", 1)[1], task_id="probe", task_type="text", outcome="pass", latency_ms=1542)],
        server.RESULTS_DIR / "results.csv",
    )

    server.CONFIG.models.append(ModelConfig(provider="groq", model_key="groq/untested"))
    page = client.get("/config").text
    assert "Last result" in page
    assert "pass=1" in page and "1542 ms" in page  # provider summary and the row cell
    assert "not tested" in page                    # a model with no recorded attempt

    # The history is readable back, so a later run appends instead of replacing it.
    rows = read_rows(server.RESULTS_DIR / "results.csv")
    assert rows[0].model_key == key and rows[0].outcome == "pass" and rows[0].latency_ms == 1542


def test_benchmark_page_can_filter_and_sort(client):
    key = groq_key()
    write_csv(
        [Row(provider="groq", model=key.split("/", 1)[1], task_id="probe", task_type="text", outcome="pass", latency_ms=1542)],
        server.RESULTS_DIR / "results.csv",
    )

    page = client.get("/benchmark").text
    assert 'id="filter"' in page
    assert "data-sort" in page
    assert f'data-key="{key}"' in page and 'data-provider="groq"' in page


def test_benchmark_page_shows_last_run_time(client):
    key = groq_key()
    row = Row(
        provider="groq", model=key.split("/", 1)[1], task_id="probe", task_type="text",
        outcome="pass", latency_ms=1542, run_at="2026-10-23 09:14:02",
    )
    write_csv([row], server.RESULTS_DIR / "results.csv")

    page = client.get("/benchmark").text
    assert "Last run" in page and "2026-10-23 09:14:02" in page  # header note and the attempt row
    assert read_rows(server.RESULTS_DIR / "results.csv")[0].run_at == "2026-10-23 09:14:02"

    # A row recorded before the run_at column existed falls back to the file mtime.
    legacy = dict(row.as_dict(), run_at="")
    stamped = datetime.fromtimestamp((server.RESULTS_DIR / "results.csv").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    assert server.last_run([legacy]) == stamped and server.last_run([]) == stamped
    assert server.last_run([row.as_dict()]) == "2026-10-23 09:14:02"


def test_retest_button_and_endpoint_append_one_attempt(client):
    key = groq_key()

    config_page = client.get("/config").text
    assert 'class="retest"' in config_page and f'data-key="{key}"' in config_page
    assert "Reply with the single word ok." in config_page  # the prompt Retest sends

    bench_page = client.get("/benchmark").text
    assert "Probe prompt" in bench_page and "text-pong" in bench_page  # the prompt table

    response = client.post("/api/retest", json={"model_key": key})
    assert response.status_code == 200
    attempt = response.json()["attempt"]
    assert attempt["task_id"] == "probe" and attempt["outcome"] == "pass" and attempt["run_at"]

    rows = read_rows(server.RESULTS_DIR / "results.csv")
    assert len(rows) == 1 and rows[0].model_key == key

    # The attempt list now renders a row, and each row carries its own Retest button.
    assert 'class="retest"' in client.get("/benchmark").text

    # A second retest appends to the history instead of replacing it.
    client.post("/api/retest", json={"model_key": key})
    assert len(read_rows(server.RESULTS_DIR / "results.csv")) == 2

    assert client.post("/api/retest", json={"model_key": "groq/not-a-model"}).status_code == 404


def test_config_api_persists_host_and_keyword(client):
    key = groq_key()
    saved = client.post("/api/config", json={"api_host": "http://127.0.0.1:8080", "models": [{"model_key": key, "host": "https://override.test/v1", "keyword": "code"}]})
    assert saved.json()["saved"] is True

    api = client.get("/api/config").json()
    assert api["api_host"] == "http://127.0.0.1:8080"
    row = api["models"][0]
    assert row["host"] == "https://override.test/v1" and row["keyword"] == "code"
    assert "api_key" not in row
