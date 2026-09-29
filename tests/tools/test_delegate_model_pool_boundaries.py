"""Public regressions for model-pool route preparation and provenance."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.redact import RedactingFormatter
from tools import async_delegation as ad
from tools.delegate_tool import delegate_task
from tools.delegate_tool_child_run import _lease_child_credential
from tools.delegate_tool_config import _resolve_child_credential_pool
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_batch_delegation, format_process_notification


def _parent():
    parent = MagicMock()
    parent.base_url = "https://parent.invalid/v1"
    parent.api_key = "fixture-parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._fallback_chain = []
    parent.enabled_toolsets = ["terminal"]
    parent.disabled_toolsets = []
    parent.session_id = "fixture-parent-session"
    return parent


def _route(model: str, *, provider: str = "custom", port: int = 9, **extra):
    return {
        "provider": provider,
        "model": model,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key": f"fixture-{model}-key",
        "api_mode": "chat_completions",
        **extra,
    }


def _fake_child(parent, captured):
    def _build(**kwargs):
        captured.append(kwargs)
        child = SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            base_url=kwargs["override_base_url"],
            api_key=kwargs["override_api_key"],
            _delegate_role="leaf",
            _subagent_id=None,
            session_id="fixture-child",
            _delegate_depth=1,
            _interrupt_requested=False,
            tool_progress_callback=None,
            run_conversation=lambda **_kw: {"completed": True, "final_response": "ok", "api_calls": 0},
            get_activity_summary=lambda: {"api_call_count": 0, "current_tool": None, "max_iterations": 4},
            interrupt=lambda *a, **k: None,
            close=lambda: None,
        )
        parent._active_children.append(child)
        return child

    return _build


def _sync_result(_batch, _background):
    return json.dumps({"ok": True})


def test_direct_endpoint_keeps_canonical_custom_identity_through_lease():
    cfg = {"max_iterations": 4, "model_pool": {"standard": _route("fixture-model", provider="openrouter")}}
    parent = _parent()
    parent_pool = MagicMock()
    parent_pool.acquire_lease.return_value = "parent-lease"
    parent_pool.current.return_value = SimpleNamespace(
        api_key="fixture-parent-key", base_url="https://parent.invalid/v1"
    )
    parent._credential_pool = parent_pool
    captured = []

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        payload = json.loads(delegate_task(goal="inspect route identity", parent_agent=parent))

    assert payload == {"ok": True}
    assert captured[0]["override_provider"] == "custom"
    child = SimpleNamespace(_swap_credential=MagicMock())
    with patch("agent.credential_pool.get_custom_provider_pool_key", return_value=None):
        child_pool = _resolve_child_credential_pool(
            captured[0]["override_provider"], parent, captured[0]["override_base_url"]
        )
    if child_pool is not None:
        child._credential_pool = child_pool
    _lease_child_credential(child)
    assert child_pool is None
    child._swap_credential.assert_not_called()
    assert captured[0]["override_base_url"] == "http://127.0.0.1:9/v1"
    assert captured[0]["override_api_key"] == "fixture-fixture-model-key"


def test_fallback_route_log_allowlists_labels_without_inline_key():
    opaque_key = "opaque-fixture-value"
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": _route(
                "primary", fallback_chain=[{"provider": "custom", "model": "backup", "api_key": opaque_key}]
            )
        },
    }
    parent = _parent()
    captured = []
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    logger = logging.getLogger("tools.delegate_tool")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            side_effect=_fake_child(parent, captured),
        ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
            payload = json.loads(delegate_task(goal="inspect safe logging", parent_agent=parent))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    text = stream.getvalue()
    assert payload == {"ok": True}
    assert "custom" in text and "backup" in text
    assert opaque_key not in text
    assert "api_key" not in text


def test_unknown_later_profile_rejects_before_transcripts_or_children():
    cfg = {"max_iterations": 4, "model_pool": {"standard": _route("standard-model")}}
    parent = _parent()
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegation_live_log.create_live_transcripts",
        return_value=(None, [None, None], []),
    ) as live, patch("tools.delegate_tool._build_child_preserving_parent_tools") as build:
        payload = json.loads(
            delegate_task(
                tasks=[
                    {"goal": "inspect routing ownership", "model_profile": "standard"},
                    {"goal": "inspect result delivery", "model_profile": "missing-tier"},
                ],
                parent_agent=parent,
            )
        )

    assert "missing-tier" in payload["error"]
    assert (live.call_count, build.call_count) == (0, 0)
    assert parent._active_children == []


def test_all_fast_tasks_do_not_resolve_unused_standard_but_standard_is_required():
    fast = _route("fast-model", port=8)
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": {"provider": "minimax", "model": "unused-standard"},
            "fast": fast,
        },
    }
    parent = _parent()
    captured = []
    resolved = []

    def _resolve(route_cfg, _parent_agent):
        resolved.append(route_cfg.get("provider"))
        if route_cfg.get("provider") == "minimax":
            raise ValueError("unused MiniMax credentials unavailable")
        return {
            "model": route_cfg.get("model"),
            "provider": "custom",
            "base_url": route_cfg.get("base_url"),
            "api_key": route_cfg.get("api_key"),
            "api_mode": "chat_completions",
            "request_overrides": None,
        }

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._resolve_delegation_credentials", side_effect=_resolve
    ), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        payload = json.loads(
            delegate_task(
                tasks=[{"goal": "use the fast route", "model_profile": "fast"}],
                parent_agent=parent,
            )
        )

    assert payload == {"ok": True}
    assert resolved == ["custom"]
    assert captured[0]["model"] == "fast-model"

    missing_standard = {"max_iterations": 4, "model_pool": {"fast": fast}}
    with patch("tools.delegate_tool._load_config", return_value=missing_standard):
        refused = json.loads(
            delegate_task(
                tasks=[{"goal": "use the fast route", "model_profile": "fast"}],
                parent_agent=_parent(),
            )
        )
    assert "standard" in refused["error"].lower()


def _run_public_background_routes(tmp_path, monkeypatch, tasks):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "max_iterations": 4,
        "independent_completions": True,
        "model_pool": {
            "standard": _route("standard-model", provider="custom", port=9),
            "fast": _route("fast-model", provider="custom", port=8),
        },
    }
    parent = _parent()
    captured_children = []
    persisted = []

    class _NoRunExecutor:
        def submit(self, _fn):
            return SimpleNamespace(add_done_callback=lambda _cb: None)

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured_children),
    ), patch("tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value="fixture-origin"), patch(
        "gateway.session_context.session_history_delivery_supported", return_value=True
    ), patch(
        "tools.delegate_tool_dispatch._resolve_async_session_key", return_value=("fixture-key", "fixture-ui")
    ), patch("tools.delegate_tool_dispatch._detach_child"), patch(
        "tools.delegate_tool_config._get_independent_completions", return_value=True
    ), patch("tools.async_delegation._records", {}), patch(
        "tools.async_delegation._persist_dispatch", side_effect=lambda record: persisted.append(dict(record))
    ), patch("tools.async_delegation._get_executor", return_value=_NoRunExecutor()), patch(
        "tools.async_delegation._ensure_stale_monitor"
    ), patch("tools.delegate_tool._get_max_async_children", return_value=4):
        handle = json.loads(
            delegate_task(tasks=tasks, background=True, parent_agent=parent)
        )

    manifest_path = tmp_path / "cache" / "delegation" / "live" / handle["delegation_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return captured_children, persisted, manifest


def test_public_sole_fast_task_keeps_route_in_manifest_and_persisted_record(tmp_path, monkeypatch):
    children, persisted, manifest = _run_public_background_routes(
        tmp_path, monkeypatch, [{"goal": "run only fast", "model_profile": "fast"}]
    )

    assert children[0]["model"] == "fast-model"
    assert manifest["model"] == "fast-model"
    assert (manifest["tasks"][0]["model"], manifest["tasks"][0]["provider"]) == (
        "fast-model", "custom"
    )
    assert [record["model"] for record in persisted] == ["fast-model"]


def test_public_independent_units_keep_routes_in_manifest_dispatch_and_completion(tmp_path, monkeypatch):
    _children, persisted, manifest = _run_public_background_routes(
        tmp_path,
        monkeypatch,
        [
            {"goal": "run fast independently", "model_profile": "fast"},
            {"goal": "run standard independently", "model_profile": "standard"},
        ],
    )

    assert [(t["model"], t["provider"]) for t in manifest["tasks"]] == [
        ("fast-model", "custom"),
        ("standard-model", "custom"),
    ]
    assert manifest["model"] is None
    assert [record["model"] for record in persisted] == ["fast-model", "standard-model"]

    rendered = _format_batch_delegation(
        {
            "role": "leaf",
            "model": None,
            "goals": ["run fast independently", "run standard independently"],
            "results": [
                {"task_index": 0, "status": "completed", "summary": "fast done", "model": "fast-model", "provider": "custom"},
                {"task_index": 1, "status": "completed", "summary": "standard done", "model": "standard-model", "provider": "custom"},
            ],
        },
        "deleg-fixture",
        1.0,
    )
    assert "Model: fast-model" in rendered
    assert "Model: standard-model" in rendered
    assert rendered.index("Model: fast-model") < rendered.index("fast done")
    assert rendered.index("Model: standard-model") < rendered.index("standard done")


class _SelectedRouteFailureChild:
    def __init__(self, *, model: str, provider: str, outcome: str):
        self.model = model
        self.provider = provider
        self.outcome = outcome
        self.session_id = f"fixture-{outcome}-child"
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self._interrupt_requested = False
        self.tool_progress_callback = None
        self._release = threading.Event()

    def run_conversation(self, **_kwargs):
        if self.outcome == "exception":
            raise RuntimeError("synthetic selected-route failure")
        self._release.wait(timeout=5.0)
        return {"completed": False, "final_response": "", "api_calls": 1}

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "max_iterations": 4}

    def interrupt(self):
        self._release.set()

    def close(self):
        self._release.set()


@pytest.mark.parametrize(("outcome", "expected_status"), [("exception", "error"), ("timeout", "timeout")])
def test_public_background_route_errors_keep_selected_provenance(
    tmp_path, monkeypatch, outcome, expected_status,
):
    """Public result, durable completion, and formatter keep the route selected before the run."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": _route("standard-model", port=9),
            "fast": _route("fast-model", port=8),
        },
    }
    parent = _parent()

    def _build(**kwargs):
        child = _SelectedRouteFailureChild(
            model=kwargs["model"], provider=kwargs["override_provider"], outcome=outcome,
        )
        parent._active_children.append(child)
        return child

    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    try:
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_build,
        ), patch(
            "tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value="",
        ), patch(
            "tools.delegate_tool._get_child_timeout", return_value=0.05 if outcome == "timeout" else None,
        ):
            handle = json.loads(delegate_task(
                goal="exercise selected route failure",
                model_profile="fast",
                background=True,
                parent_agent=parent,
            ))

            assert handle["status"] == "dispatched"
            deadline = time.monotonic() + 5.0
            event = None
            while time.monotonic() < deadline:
                if process_registry.completion_queue.empty():
                    time.sleep(0.02)
                    continue
                candidate = process_registry.completion_queue.get_nowait()
                if candidate.get("delegation_id") == handle["delegation_id"]:
                    event = candidate
                    break
            assert event is not None
            (result,) = event["results"]
            assert (result["status"], result["model"], result["provider"]) == (
                expected_status, "fast-model", "custom",
            )

            durable = ad.get_durable_delegation(event["delegation_id"])
            assert durable is not None
            (persisted,) = durable["result"]["results"]
            assert (persisted["status"], persisted["model"], persisted["provider"]) == (
                expected_status, "fast-model", "custom",
            )

            rendered = format_process_notification(event)
            assert "Model: fast-model" in rendered
            assert "Provider: custom" in rendered
            assert rendered.index("Model: fast-model") < rendered.index("(no summary")
    finally:
        deadline = time.monotonic() + 2.0
        while ad.active_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
