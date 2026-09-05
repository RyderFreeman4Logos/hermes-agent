"""message.complete must stamp loop end time + first-call cache_info (#126)."""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, **_kw):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _StoppedTicker:
    def join(self):
        pass


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def frames(monkeypatch):
    captured: list = []
    monkeypatch.setattr(server, "write_json", captured.append)
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_start_usage_ticker", lambda *_args: (threading.Event(), _StoppedTicker())
    )
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)


def _complete_payloads(frames):
    out = []
    for frame in frames:
        params = frame.get("params") or {}
        if params.get("type") == "message.complete":
            out.append(params.get("payload") or {})
    return out


def test_emit_stamps_completed_at_on_message_complete(frames):
    before = server.time.time()
    server._emit("message.complete", "sid", {"text": "done", "status": "complete"})
    after = server.time.time()

    payload = _complete_payloads(frames)[0]
    stamp = payload["completed_at"]
    assert before <= stamp <= after
    assert frames[0]["params"]["payload"]["text"] == "done"


def test_emit_does_not_stamp_non_complete_events(frames):
    server._emit("status.update", "sid", {"kind": "cache_hit", "text": "cache 95%"})
    params = frames[0]["params"]
    assert params["type"] == "status.update"
    assert "completed_at" not in (params.get("payload") or {})


@pytest.mark.parametrize(
    ("state", "pct", "read", "expected_state", "expected_pct"),
    [
        ("hit", 95, 1_900, "hit", 95),
        ("cold_write", 0, 0, "cold_write", 0),
        ("no_field", 0, 0, "unavailable", 0),
    ],
)
def test_message_complete_stamps_first_call_cache_info(
    frames, turn_env, state, pct, read, expected_state, expected_pct
):
    class _Agent:
        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            record = {
                "request_index": 1,
                "state": state,
                "pct": pct if state == "hit" else None,
                "timestamp": 1.0,
                "turn_origin": turn_origin,
            }
            callback = getattr(self, "_tui_cache_callback")
            callback(state, pct, read, 2_000, record)
            callback("hit", 99, 1_980, 2_000, {**record, "request_index": 2, "pct": 99})
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = f"loop-stamp-{state}"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert isinstance(payload["completed_at"], (int, float))
    assert payload["cache_info"]["state"] == expected_state
    assert payload["cache_info"]["pct"] == expected_pct
    if expected_state == "unavailable":
        assert "read_tokens" not in payload["cache_info"]
        assert "prompt_tokens" not in payload["cache_info"]
    else:
        assert payload["cache_info"]["read_tokens"] == read
        assert payload["cache_info"]["prompt_tokens"] == 2_000


def test_message_complete_unavailable_when_no_provider_usage(frames, turn_env):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {"final_response": "reply", "messages": []},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-no-usage"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert isinstance(payload["completed_at"], (int, float))
    assert payload["cache_info"]["state"] == "unavailable"
    assert "read_tokens" not in payload["cache_info"]
    assert "prompt_tokens" not in payload["cache_info"]


def test_cache_info_omits_tokens_when_counts_missing():
    info = server._cache_info_from_first_call({"state": "hit", "pct": 95})
    assert info["state"] == "hit"
    assert info["pct"] == 95
    assert "read_tokens" not in info
    assert "prompt_tokens" not in info


def test_usage_without_cache_telemetry_does_not_fake_token_counts():
    info = server._cache_info_from_usage({"prompt_tokens": 4_000})
    assert info == {"state": "unavailable", "pct": 0}


def test_message_complete_does_not_reuse_prior_loop_cache_info(frames, turn_env):
    class _Agent:
        def __init__(self):
            self._calls = 0

        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            self._calls += 1
            if self._calls == 1:
                getattr(self, "_tui_cache_callback")(
                    "hit",
                    95,
                    1_900,
                    2_000,
                    {
                        "request_index": 1,
                        "state": "hit",
                        "pct": 95,
                        "timestamp": 1.0,
                        "turn_origin": turn_origin,
                    },
                )
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-no-reuse"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid-1", sid, session, "first")
        server._run_prompt_submit("rid-2", sid, session, "second")
    finally:
        server._sessions.pop(sid, None)

    payloads = _complete_payloads(frames)
    assert payloads[0]["cache_info"]["state"] == "hit"
    assert payloads[1]["cache_info"]["state"] == "unavailable"


def test_make_agent_wires_cache_callback_on_synthetic_agent(monkeypatch):
    class _Agent:
        pass

    monkeypatch.setattr(
        "tui_gateway.synthetic_turn.maybe_build_synthetic_agent",
        lambda *_a, **_k: _Agent(),
    )
    agent = server._make_agent("sid-wire", "session-key")
    assert callable(getattr(agent, "_tui_cache_callback", None))
    assert getattr(agent, "_tui_cache_owner_session", None) == "sid-wire"


def _real_shaped_counters(**extra):
    """Session counters as AIAgent initializes them (zeros, not missing)."""
    ns = types.SimpleNamespace(
        session_id="session-key",
        session_api_calls=0,
        session_prompt_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        _awaiting_cache_usage_after_compression=False,
        _first_turn_usage=None,
        _tui_provider_response_index=0,
        **extra,
    )
    return ns


def test_unavailable_first_record_does_not_become_miss_from_zero_counters(
    frames, turn_env
):
    from agent.conversation_loop import _ingest_successful_provider_usage

    class _Agent:
        def __init__(self):
            self.__dict__.update(_real_shaped_counters().__dict__)

        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            self.session_api_calls = 1
            self.session_prompt_tokens = 4_000
            _ingest_successful_provider_usage(
                self,
                {
                    "prompt_tokens": 4_000,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_telemetry": "unavailable",
                },
                first_call=True,
            )
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-zero-counters-unavailable"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert payload["cache_info"]["state"] == "unavailable"
    assert "read_tokens" not in payload["cache_info"]
    assert "prompt_tokens" not in payload["cache_info"]
    assert payload["cache_info"].get("compression_bound") is not True


def test_same_wake_post_compression_usage_publishes_compression_bound(
    frames, turn_env
):
    from agent.conversation_loop import _ingest_successful_provider_usage

    class _Agent:
        def __init__(self):
            self.__dict__.update(_real_shaped_counters().__dict__)

        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            _ingest_successful_provider_usage(
                self,
                {
                    "prompt_tokens": 2_000,
                    "cache_read_tokens": 1_900,
                    "cache_write_tokens": 0,
                    "cache_telemetry": "reported",
                },
                first_call=True,
            )
            self._awaiting_cache_usage_after_compression = True
            _ingest_successful_provider_usage(
                self,
                {
                    "prompt_tokens": 800,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 25,
                    "cache_telemetry": "reported",
                },
                first_call=False,
            )
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-same-wake-bound"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert payload["cache_info"]["compression_bound"] is True
    assert payload["cache_info"]["attribution"] == "post_compression"
    assert payload["cache_info"]["state"] == "cold_write"
    assert agent._awaiting_cache_usage_after_compression is False


@pytest.mark.parametrize(
    "aggregator_read,advisor_read,expected_state,expected_pct",
    [(None, None, "unavailable", 0), (None, 760, "unavailable", 0),
     (0, 760, "miss", 0), (400, 760, "hit", 50),
     (400, None, "hit", 50), (400, "absent", "hit", 50)],
)
@pytest.mark.parametrize("pending", [False, True])
def test_normal_loop_cache_origin_seam(
    frames, turn_env, aggregator_read, advisor_read, expected_state, expected_pct, pending
):
    """Execute the production normalization/fold/selection seam, not a full turn."""
    import ast
    import logging
    from pathlib import Path

    import agent.conversation_loop as loop
    from agent.usage_pricing import normalize_usage

    path = Path(__file__).resolve().parents[2] / "agent/conversation_loop.py"
    assert Path(loop.__file__).resolve() == path
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "run_conversation")
    # Take the contiguous real usage block through ingest, including the actual
    # aggregator capture and advisor fold. No parallel mapping or line-number pin.
    block = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                 and n.body and isinstance(n.body[0], ast.Assign)
                 and ast.unparse(n.body[0].targets[0]) == "canonical_usage")
    end = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == "_ingest_successful_provider_usage")
    seam = compile(ast.Module(body=block.body[:end + 1], type_ignores=[]), str(path), "exec")
    raw = {"prompt_tokens": 800, "completion_tokens": 20,
           "prompt_tokens_details": {"cached_tokens": aggregator_read}}
    advisor = None if advisor_read == "absent" else normalize_usage(
        {"prompt_tokens": 800, "completion_tokens": 10,
         "prompt_tokens_details": {"cached_tokens": advisor_read}},
        provider="openai", api_mode="chat_completions",
    )

    class Agent:
        def __init__(self):
            self.__dict__.update(_real_shaped_counters().__dict__)
            self.provider, self.api_mode = "openai", "chat_completions"
            self.client = types.SimpleNamespace(consume_reference_usage=lambda: (advisor, None))

        def run_conversation(self, _prompt, **kwargs):
            self._awaiting_cache_usage_after_compression = pending
            scope = {"agent": self, "response": types.SimpleNamespace(usage=raw),
                     "api_call_count": 1, "normalize_usage": normalize_usage,
                     "logger": logging.getLogger(__name__),
                     "_ingest_successful_provider_usage": loop._ingest_successful_provider_usage}
            exec(seam, scope)
            self.accounted_usage = scope["usage_dict"]
            return {"final_response": "synthetic", "messages": []}

        def clear_interrupt(self):
            pass

    agent = Agent()
    sid = "synthetic-cache-origin"
    session = _session(agent=agent, running=True)
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("synthetic", sid, session, "synthetic")
        # Keep accounting assertions outside the host's exception handler.
        usage = agent.accounted_usage
        assert usage["prompt_tokens"] == (800 if advisor is None else 1600)
        assert usage["completion_tokens"] == (20 if advisor is None else 30)
        assert usage["cache_read_tokens"] == (aggregator_read or 0) + (
            advisor_read if isinstance(advisor_read, int) else 0)
        info = _complete_payloads(frames)[0]["cache_info"]
        assert info["state"] == expected_state
        assert info["pct"] == expected_pct
        assert (info.get("compression_bound") is True) == pending
        assert ("read_tokens" in info) == (aggregator_read is not None)
        if aggregator_read is not None:
            assert info["read_tokens"] == aggregator_read
            assert info["prompt_tokens"] == 800
        else:
            assert "prompt_tokens" not in info
        assert agent._awaiting_cache_usage_after_compression is False
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.parametrize("source", ["ingest", "callback", "record", "usage"])
@pytest.mark.parametrize("state", ["hit", "cold_write", "miss", "no_field", "unavailable"])
@pytest.mark.parametrize("epoch", ["ordinary", "before-first", "same-wake", "repeated"])
def test_public_completion_projection_matrix(frames, turn_env, source, state, epoch):
    """Unknown numeric telemetry cannot erase an authoritative boundary record."""
    from pathlib import Path

    from agent.conversation_loop import _ingest_successful_provider_usage as ingest

    root = Path(__file__).resolve().parents[2]
    assert Path(server.__file__).resolve() == root / "tui_gateway/server.py"
    assert Path(ingest.__code__.co_filename).resolve() == root / "agent/conversation_loop.py"
    known = state not in {"no_field", "unavailable"}
    bound = epoch != "ordinary"
    expected_state = state if known else "unavailable"
    read = 760 if state == "hit" else 0
    usage = {
        "prompt_tokens": 800,
        "cache_read_tokens": read,
        "cache_write_tokens": 25 if state == "cold_write" else 0,
        "cache_telemetry": "reported" if known else "unavailable",
    }

    class Agent:
        def __init__(self):
            self.__dict__.update(_real_shaped_counters().__dict__)
            self.wake = 0

        def run_conversation(self, _prompt, **_kwargs):
            self.wake += 1
            # Match run_conversation's per-wake usage reset; session-record
            # reset itself is exercised in the real _run_prompt_submit.
            self._first_turn_usage = None
            self._last_turn_usage = None
            self._tui_provider_response_index = 0
            if self.wake == 2:
                return {"final_response": "reply", "messages": []}
            if epoch in {"same-wake", "repeated"}:
                ingest(self, {**usage, "cache_telemetry": "reported",
                              "cache_read_tokens": 0, "cache_write_tokens": 0}, first_call=True)
            if epoch == "repeated":
                self._awaiting_cache_usage_after_compression = True
                ingest(self, {**usage, "cache_telemetry": "reported",
                              "cache_read_tokens": 400}, first_call=False)
            self._awaiting_cache_usage_after_compression = bound
            if source == "usage":
                self._tui_cache_callback = None
                session.pop("first_provider_response", None)
            if source in {"ingest", "usage"}:
                ingest(self, usage, first_call=epoch in {"ordinary", "before-first"})
            else:
                record = {"state": state, "pct": 95 if read else 0}
                if bound:
                    record["attribution"] = "post_compression"
                if known:
                    record.update(read_tokens=read, prompt_tokens=800)
                if source == "record":
                    session["first_provider_response"] = record
                    # An older fallback must not replace an authoritative
                    # unknown record, even when that fallback has numbers.
                    self._first_turn_usage = {**usage, "cache_telemetry": "reported"}
                else:
                    self._first_turn_usage = None
                    self._tui_cache_callback(state, record["pct"], read, 800, record)
                self._awaiting_cache_usage_after_compression = False
            # Ordinary later usage must not replace the first/boundary response.
            ingest(self, {**usage, "cache_telemetry": "reported",
                          "cache_read_tokens": 123}, first_call=False)
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            pass

    agent = Agent()
    sid = "projection-matrix"
    session = _session(agent=agent, running=True)
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("first", sid, session, "synthetic")
        info = _complete_payloads(frames)[0]["cache_info"]
        assert info["state"] == expected_state
        assert (info.get("compression_bound") is True) == bound
        assert info.get("attribution") == ("post_compression" if bound else None)
        assert ("read_tokens" in info) == known
        assert ("prompt_tokens" in info) == known
        if known:
            assert info["read_tokens"] == read
            assert info["prompt_tokens"] == 800
            assert info["pct"] == (95 if read else 0)
        for key, value in info.items():
            assert session["first_provider_response"][key] == value
        assert agent._awaiting_cache_usage_after_compression is False
        if source in {"ingest", "callback"}:
            statuses = [f["params"]["payload"]["cache_record"] for f in frames
                        if f.get("params", {}).get("type") == "status.update"
                        and "cache_record" in f["params"]["payload"]]
            if known:
                assert statuses[-1]["state"] == info["state"]
                assert statuses[-1].get("compression_bound", False) == bound
            else:
                assert not any(r.get("state") in {"no_field", "unavailable"} for r in statuses)
        server._run_prompt_submit("second", sid, session, "synthetic")
        assert _complete_payloads(frames)[1]["cache_info"] == {"state": "unavailable", "pct": 0}
    finally:
        server._sessions.pop(sid, None)
