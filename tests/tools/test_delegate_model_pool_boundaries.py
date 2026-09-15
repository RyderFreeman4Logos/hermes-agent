"""Public regressions for model-pool route preparation and provenance."""

from __future__ import annotations

import io
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.redact import RedactingFormatter
from tools.delegate_tool import delegate_task
from tools.delegate_tool_child_run import _lease_child_credential
from tools.delegate_tool_config import _resolve_child_credential_pool
from tools.process_registry_notifications import _format_batch_delegation


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
            return SimpleNamespace()

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured_children),
    ), patch("tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value="fixture-origin"), patch(
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
