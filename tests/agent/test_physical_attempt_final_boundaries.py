"""Public boundary regressions for the final #227 diagnostics repair."""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import threading
from types import SimpleNamespace

import pytest


def _enabled(monkeypatch, root):
    from agent import cache_lowhit_request_dump as dump
    from agent import physical_attempt_diagnostics as diagnostics

    monkeypatch.setattr(dump, "get_hermes_home", lambda: root)
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: root)
    monkeypatch.setattr(dump, "enabled", lambda: True)
    monkeypatch.setattr(diagnostics, "enabled", lambda: True)
    diagnostics._LAST_ATTEMPT.clear()
    dump.reset_for_tests()


def _usage(*, cached=0, prompt=1000):
    from agent.usage_pricing import CanonicalUsage

    return CanonicalUsage(input_tokens=prompt - cached, cache_read_tokens=cached)


def test_public_sdk_retry_records_each_final_body_and_skips_pre_dispatch_cancel(monkeypatch):
    from agent import relay_llm

    remembered = []
    attempts = []
    monkeypatch.setattr(
        relay_llm.cache_lowhit_request_dump,
        "remember_sent_request",
        lambda request, **kwargs: remembered.append((dict(request), kwargs)),
    )
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "prepare_cache_scope",
        lambda _scope: None,
    )
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )
    sent = []

    def sdk_create(request):
        sent.append(dict(request))
        if len(sent) == 1:
            raise ValueError("synthetic rejected cache marker")
        return {"ok": True}

    def retrying_callback(request):
        try:
            return relay_llm.physical_send(request, sdk_create)
        except ValueError:
            rewritten = {**request, "messages": [{"role": "user", "content": "rewritten"}]}
            return relay_llm.physical_send(rewritten, sdk_create)

    result = relay_llm.execute(
        {"model": "model", "messages": [{"role": "user", "content": "initial"}]},
        retrying_callback,
        session_id="",
        name="provider",
        model_name="model",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:3"},
    )
    assert result == {"ok": True}
    assert [request for request, _kwargs in attempts] == sent
    assert [request for request, _kwargs in remembered] == sent
    assert [kwargs["physical_send_ordinal"] for _request, kwargs in attempts] == [0, 1]

    attempts.clear()
    with pytest.raises(RuntimeError, match="cancelled before dispatch"):
        relay_llm.execute(
            {"model": "model", "messages": []},
            lambda _request: (_ for _ in ()).throw(RuntimeError("cancelled before dispatch")),
            session_id="",
            name="provider",
            model_name="model",
            metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:4"},
        )
    assert attempts == []


def test_bedrock_public_cachepoint_recovery_records_both_physical_sends(monkeypatch):
    from agent import bedrock_adapter, relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(relay_llm.physical_attempt_diagnostics, "prepare_cache_scope", lambda _scope: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )

    class Client:
        def __init__(self):
            self.sent = []

        def converse(self, **kwargs):
            self.sent.append(kwargs)
            if len(self.sent) == 1:
                raise ValueError("synthetic cachePoint rejection")
            return {"output": {"message": {"content": []}}, "usage": {}}

    client = Client()
    monkeypatch.setattr(bedrock_adapter, "_get_bedrock_runtime_client", lambda _region: client)
    monkeypatch.setattr(bedrock_adapter, "build_converse_kwargs", lambda *_a, **_k: {"modelId": "m", "toolConfig": {"tools": [{"cachePoint": {"type": "default"}}]}})
    monkeypatch.setattr(bedrock_adapter, "recover_from_cache_point_rejection", lambda _exc, _request: {"modelId": "m", "toolConfig": {"tools": []}})

    relay_llm.execute(
        {"model": "m", "messages": []},
        lambda _request: bedrock_adapter.call_converse("region", "m", []),
        session_id="", name="bedrock", model_name="m",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:5"},
    )
    assert [request for request, _kwargs in attempts] == client.sent
    assert [kwargs["physical_send_ordinal"] for _request, kwargs in attempts] == [0, 1]


def test_internal_aux_adapter_records_at_its_sdk_boundary_only_once(monkeypatch):
    from agent import auxiliary_client as aux
    from agent import relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(relay_llm.physical_attempt_diagnostics, "prepare_cache_scope", lambda _scope: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )
    monkeypatch.setattr(aux, "_client_streams_internally", lambda _client: True)

    class Completions:
        def create(self, **kwargs):
            return relay_llm.physical_send(
                kwargs, lambda final: SimpleNamespace(choices=[final], usage=None)
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = relay_llm.execute(
        {"model": "m", "messages": []},
        lambda request: aux._create_with_progress(client, request),
        session_id="", name="provider", model_name="m",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:6"},
    )
    assert result.choices
    assert len(attempts) == 1


def test_relay_bypassed_facade_carries_scope_without_leaking_private_envelope(monkeypatch):
    from agent import relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "prepare_cache_scope",
        lambda _scope: {"digest": "synthetic-scope"},
    )
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )

    def adapter(request):
        assert "_hermes_physical_attempt_cache_scope" not in request
        final = {"model": request["model"], "input": []}
        return relay_llm.physical_send(final, lambda sent: sent)

    result = relay_llm.run_direct(
        {"model": "m", "prompt_cache_key": "synthetic-key"}, adapter,
        name="provider", model_name="m",
        metadata={"api_mode": "codex_responses", "api_request_id": "turn:api:9"},
    )
    assert result == {"model": "m", "input": []}
    assert attempts[0][1]["scope"] == {"digest": "synthetic-scope"}


def test_aux_stream_negotiation_records_stream_then_plain_final_kwargs(monkeypatch):
    from agent import auxiliary_client as aux
    from agent import relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(relay_llm.physical_attempt_diagnostics, "prepare_cache_scope", lambda _scope: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: True)
    monkeypatch.setattr(aux, "_client_streams_internally", lambda _client: False)

    class Completions:
        def __init__(self):
            self.sent = []

        def create(self, **kwargs):
            self.sent.append(kwargs)
            if kwargs.get("stream"):
                raise ValueError("synthetic streaming unsupported")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = relay_llm.execute(
        {"model": "m", "messages": []},
        lambda request: aux._create_with_progress(client, request),
        session_id="", name="provider", model_name="m",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:7"},
    )
    assert result.choices
    assert [request for request, _kwargs in attempts] == completions.sent
    assert attempts[0][0]["stream"] is True
    assert "stream" not in attempts[1][0]


@pytest.mark.asyncio
async def test_async_aux_stream_negotiation_records_stream_then_plain_final_kwargs(monkeypatch):
    from agent import auxiliary_client as aux
    from agent import relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(relay_llm.physical_attempt_diagnostics, "prepare_cache_scope", lambda _scope: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )
    monkeypatch.setattr(aux, "_aux_progress_active", lambda: True)
    monkeypatch.setattr(aux, "_async_client_streams_internally", lambda _client: False)

    class Completions:
        def __init__(self):
            self.sent = []

        async def create(self, **kwargs):
            self.sent.append(kwargs)
            if kwargs.get("stream"):
                raise ValueError("synthetic streaming unsupported")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = await relay_llm.execute_async(
        {"model": "m", "messages": []},
        lambda request: aux._acreate_with_progress(client, request),
        session_id="", name="provider", model_name="m",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:8"},
    )
    assert result.choices
    assert [request for request, _kwargs in attempts] == completions.sent
    assert attempts[0][0]["stream"] is True
    assert "stream" not in attempts[1][0]


@pytest.mark.asyncio
async def test_public_sdk_async_and_stream_sends_are_exactly_once(monkeypatch):
    from agent import relay_llm

    attempts = []
    monkeypatch.setattr(relay_llm.cache_lowhit_request_dump, "remember_sent_request", lambda *_a, **_k: None)
    monkeypatch.setattr(relay_llm.physical_attempt_diagnostics, "prepare_cache_scope", lambda _scope: None)
    monkeypatch.setattr(
        relay_llm.physical_attempt_diagnostics,
        "start_attempt",
        lambda request, **kwargs: attempts.append((dict(request), kwargs)),
    )

    async def async_callback(request):
        return await relay_llm.physical_send_async(request, lambda final: _async_value(final))

    async def _async_value(final):
        return {"model": final["model"]}

    result = await relay_llm.execute_async(
        {"model": "async-model", "messages": []}, async_callback,
        session_id="", name="provider", model_name="async-model",
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:1"},
    )
    stream = relay_llm.stream_current(
        {"model": "stream-model", "messages": [], "stream": True},
        lambda request: relay_llm.physical_send(request, lambda final: iter([final["model"]])),
        name="provider", model_name="stream-model", finalizer=dict,
        metadata={"api_mode": "chat_completions", "api_request_id": "turn:api:2"},
    )
    assert result == {"model": "async-model"}
    assert list(stream) == ["stream-model"]
    assert [request["model"] for request, _kwargs in attempts] == ["async-model", "stream-model"]


def test_lowhit_event_is_profile_and_response_owned_and_published_once(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    roots = {"value": tmp_path / "profile-a"}
    monkeypatch.setattr(dump, "get_hermes_home", lambda: roots["value"])
    monkeypatch.setattr(dump, "enabled", lambda: True)
    monkeypatch.setattr("agent.physical_attempt_diagnostics.get_hermes_home", lambda: roots["value"])
    dump.reset_for_tests()

    event_a = dump.remember_sent_request(
        {"model": "model-a", "messages": [{"role": "user", "content": "A"}]},
        correlation="call-a",
    )
    roots["value"] = tmp_path / "profile-b"
    dump.remember_sent_request(
        {"model": "model-b", "messages": [{"role": "user", "content": "B"}]},
        correlation="call-b",
    )
    dump.remember_sent_request(
        {"model": "model-c", "messages": [{"role": "user", "content": "C"}]},
        correlation="call-c",
    )

    dump.maybe_dump_on_usage(_usage(), cache_telemetry="reported", event=event_a)
    dump.maybe_dump_on_usage(_usage(), cache_telemetry="reported", event=event_a)

    files_a = list((tmp_path / "profile-a" / "observability" / "cache_lowhit").glob("*.json"))
    assert len(files_a) == 1
    assert not (tmp_path / "profile-b" / "observability" / "cache_lowhit").exists()
    payload = json.loads(files_a[0].read_text(encoding="utf-8"))
    assert len(payload["requests"]) == 1


@pytest.mark.parametrize(
    ("usage", "reported"),
    [
        ({"prompt_tokens": 10, "prompt_tokens_details": {}}, False),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": None}}, False),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}, True),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 2}}, True),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cache_write_tokens": 2}}, False),
        (SimpleNamespace(prompt_tokens=10, prompt_tokens_details=SimpleNamespace(cached_tokens=None)), False),
        (SimpleNamespace(prompt_tokens=10, prompt_tokens_details=SimpleNamespace(cached_tokens=0)), True),
    ],
)
def test_normalize_usage_distinguishes_missing_null_zero_and_write_only(monkeypatch, usage, reported):
    from agent import cache_lowhit_request_dump as dump
    from agent.usage_pricing import normalize_usage

    observed = []
    monkeypatch.setattr(dump, "maybe_dump_on_usage", lambda _usage, **kwargs: observed.append(kwargs))
    normalize_usage(usage)
    assert observed == [{"cache_telemetry": "reported" if reported else "unavailable"}]


def test_bedrock_omitted_cache_fields_remain_unavailable_sync_and_stream(monkeypatch):
    from agent import cache_lowhit_request_dump as dump
    from agent.bedrock_adapter import normalize_converse_response, stream_converse_with_callbacks
    from agent.usage_pricing import normalize_usage

    observed = []
    monkeypatch.setattr(dump, "maybe_dump_on_usage", lambda _usage, **kwargs: observed.append(kwargs))
    sync = normalize_converse_response({"usage": {"inputTokens": 5, "outputTokens": 1}})
    stream = stream_converse_with_callbacks({"stream": [{"metadata": {"usage": {"inputTokens": 5, "outputTokens": 1}}}]})
    normalize_usage(sync.usage)
    normalize_usage(stream.usage)
    assert observed == [
        {"cache_telemetry": "unavailable"},
        {"cache_telemetry": "unavailable"},
    ]


def _dump_writer(root, event, barrier):
    from agent import cache_lowhit_request_dump as dump

    real_write = dump.atomic_json_write

    def synchronized_write(*args, **kwargs):
        try:
            barrier.wait(timeout=0.4)
        except threading.BrokenBarrierError:
            pass
        return real_write(*args, **kwargs)

    dump.atomic_json_write = synchronized_write
    dump.maybe_dump_on_usage(_usage(), cache_telemetry="reported", event=event)


def test_dump_retention_is_one_cross_process_transaction(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enabled(monkeypatch, tmp_path)
    events = []
    for index in range(dump.MAX_DUMPS + 2):
        events.append(dump.remember_sent_request({"model": "m", "messages": [{"role": "user", "content": str(index)}]}, correlation=str(index)))
    for event in events[: dump.MAX_DUMPS]:
        dump.maybe_dump_on_usage(_usage(), cache_telemetry="reported", event=event)
    barrier = multiprocessing.Barrier(2)
    workers = [multiprocessing.Process(target=_dump_writer, args=(tmp_path, event, barrier)) for event in events[-2:]]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    files = list((tmp_path / "observability" / "cache_lowhit").glob("*.json"))
    assert len(files) == dump.MAX_DUMPS
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in files)


def _append_worker(root, barrier, marker):
    from agent import physical_attempt_diagnostics as diagnostics

    diagnostics.get_hermes_home = lambda: root
    diagnostics.enabled = lambda: True
    real_write = diagnostics.os.write

    def synchronized_write(fd, payload):
        if payload.endswith(b"\n"):
            try:
                barrier.wait(timeout=0.4)
            except threading.BrokenBarrierError:
                pass
        return real_write(fd, payload)

    diagnostics.os.write = synchronized_write
    diagnostics._append({"marker": marker})


def _key_worker(root, barrier, results):
    from agent import physical_attempt_diagnostics as diagnostics

    diagnostics.get_hermes_home = lambda: root
    real_write = diagnostics.os.write

    def synchronized_write(fd, payload):
        if len(payload) == 32:
            try:
                barrier.wait(timeout=0.4)
            except threading.BrokenBarrierError:
                pass
        return real_write(fd, payload)

    diagnostics.os.write = synchronized_write
    try:
        results.put(("ok", hashlib.sha256(diagnostics._key()).hexdigest()))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def test_jsonl_cap_is_one_cross_process_transaction(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics

    _enabled(monkeypatch, tmp_path)
    diagnostics._key()
    monkeypatch.setattr(diagnostics, "_MAX_RECORDS_BYTES", 220)
    path = tmp_path / "observability" / "physical_attempt_digests.jsonl"
    path.write_bytes((json.dumps({"old": "x" * 130}) + "\n").encode())
    path.chmod(0o600)
    barrier = multiprocessing.Barrier(2)
    workers = [multiprocessing.Process(target=_append_worker, args=(tmp_path, barrier, "y" * 55)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    assert path.stat().st_size <= 220
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def test_competing_key_creators_publish_one_complete_private_key(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics

    _enabled(monkeypatch, tmp_path)
    barrier = multiprocessing.Barrier(2)
    results = multiprocessing.Queue()
    workers = [multiprocessing.Process(target=_key_worker, args=(tmp_path, barrier, results)) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    outcomes = [results.get(timeout=1) for _ in workers]
    assert [status for status, _value in outcomes] == ["ok", "ok"]
    assert len({value for _status, value in outcomes}) == 1
    key_path = tmp_path / "observability" / "physical_attempt_digests.key"
    assert key_path.stat().st_size == 32
    assert key_path.stat().st_mode & 0o077 == 0


def test_failed_first_key_write_recovers_without_manual_deletion(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics

    _enabled(monkeypatch, tmp_path)
    real_write = diagnostics.os.write
    calls = {"count": 0}

    def fail_first(fd, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("synthetic quota exhaustion")
        return real_write(fd, payload)

    monkeypatch.setattr(diagnostics.os, "write", fail_first)
    assert diagnostics.start_attempt(
        {"messages": []}, api_mode="chat_completions", route="chat_completions",
        provider="provider", model="model", retry=0, loop=1, correlation="key-failure",
    ) is None
    key_path = tmp_path / "observability" / "physical_attempt_digests.key"
    assert not key_path.exists()
    attempt = diagnostics.start_attempt(
        {"messages": []}, api_mode="chat_completions", route="chat_completions",
        provider="provider", model="model", retry=0, loop=1, correlation="key-recovery",
    )
    assert attempt is not None
    assert key_path.stat().st_size == 32


def test_short_append_failure_rolls_back_and_does_not_advance_pairing(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics

    _enabled(monkeypatch, tmp_path)
    diagnostics._key()
    real_write = diagnostics.os.write
    calls = {"count": 0}

    def short_then_fail(fd, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_write(fd, payload[: max(1, len(payload) // 2)])
        if calls["count"] == 2:
            raise OSError("synthetic continuation failure")
        return real_write(fd, payload)

    monkeypatch.setattr(diagnostics.os, "write", short_then_fail)
    failed = diagnostics.start_attempt(
        {"messages": [{"role": "user", "content": "first"}]},
        api_mode="chat_completions", route="chat_completions", provider="provider",
        model="model", retry=0, loop=1, correlation="append-failure",
    )
    assert failed is None
    assert diagnostics._LAST_ATTEMPT == {}

    succeeded = diagnostics.start_attempt(
        {"messages": [{"role": "user", "content": "second"}]},
        api_mode="chat_completions", route="chat_completions", provider="provider",
        model="model", retry=0, loop=1, correlation="append-failure",
    )
    assert succeeded is not None
    lines = (tmp_path / "observability" / "physical_attempt_digests.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["phase"] for line in lines] == ["attempt"]
