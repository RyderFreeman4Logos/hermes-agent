from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


def _scope(session_id, *, platform="cli", db=None, parent_session_id=None):
    agent = SimpleNamespace(
        session_id=session_id,
        platform=platform,
        _session_db=db,
        _parent_session_id=parent_session_id,
    )
    return AIAgent._prompt_cache_scope_id(agent)


def test_prompt_cache_scope_preserves_only_compression_lineage(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        db.create_session("other-root", source="cli")
        db.create_session(
            "delegate-a",
            source="delegate",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        db.create_session(
            "delegate-b",
            source="delegate",
            parent_session_id="root",
            model_config={"_delegate_from": "root"},
        )
        db.create_session(
            "delegate-other",
            source="delegate",
            parent_session_id="other-root",
            model_config={"_delegate_from": "other-root"},
        )
        db.create_session("tool", source="tool", parent_session_id="root")
        db.create_session(
            "branch",
            source="cli",
            parent_session_id="root",
            model_config={"_branched_from": "root"},
        )
        db.end_session("root", "compression")
        db.create_session("continuation", source="cli", parent_session_id="root")

        db.end_session("delegate-a", "compression")
        db.create_session(
            "delegate-tip",
            source="delegate",
            parent_session_id="delegate-a",
        )

        assert db.get_compression_lineage("root") == ["root", "continuation"]
        assert _scope("root", db=db) == "root"
        assert _scope("continuation", db=db) == "root"
        assert _scope("branch", db=db) == "branch"
        assert _scope("tool", db=db) == "tool"
        assert _scope("delegate-a", db=db) == "root"
        assert _scope("delegate-b", db=db) == "root"
        assert _scope("delegate-tip", db=db) == "root"
        assert _scope("delegate-other", db=db) == "other-root"
        # A first-turn native delegate has not created its row yet. Its parent
        # hint still resolves through a nested delegate to the top-level root.
        assert (
            _scope(
                "not-created-yet",
                platform="subagent",
                db=db,
                parent_session_id="delegate-a",
            )
            == "root"
        )
        # Non-delegate children never inherit merely because they have a
        # parent hint; their explicit branch/tool namespace remains isolated.
        assert (
            _scope(
                "not-created-branch",
                platform="cli",
                db=db,
                parent_session_id="root",
            )
            == "not-created-branch"
        )
    finally:
        db.close()


def test_cron_prompt_cache_scope_is_stable_per_job():
    first = _scope("cron_my_job_20260804_151339", platform="cron")
    second = _scope("cron_my_job_20260804_161339", platform="cron")

    assert first == second == "cron:my_job"
    assert first != _scope("cron_other_job_20260804_151339", platform="cron")


def test_cron_continuation_uses_job_scope(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        root = "cron_my_job_20260804_151339"
        db.create_session(root, source="cron")
        db.end_session(root, "compression")
        db.create_session(
            "cron-continuation",
            source="cron",
            parent_session_id=root,
        )

        assert _scope(root, platform="cron", db=db) == "cron:my_job"
        assert _scope("cron-continuation", platform="cron", db=db) == "cron:my_job"
    finally:
        db.close()


def test_main_runtime_refresh_preserves_scoped_cache_namespace():
    from agent.auxiliary_client import (
        _runtime_main_value,
        reset_runtime_main,
        scoped_runtime_main,
        set_runtime_main,
    )

    with scoped_runtime_main({"cache_scope": "conversation-a"}):
        token = set_runtime_main("custom", "model")
        try:
            assert _runtime_main_value("cache_scope") == "conversation-a"
        finally:
            reset_runtime_main(token)
        assert _runtime_main_value("cache_scope") == "conversation-a"

    assert _runtime_main_value("cache_scope") == ""


def test_direct_compression_binds_scope_in_sync_and_bare_executor(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from agent.auxiliary_client import _runtime_main_value, scoped_runtime_main

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        db.end_session("root", "compression")
        db.create_session("continuation", source="cli", parent_session_id="root")
        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "continuation"
        agent.platform = "cli"
        agent._session_db = db
        agent._parent_session_id = None
        agent._conversation_root_id = lambda: None
        observed = []

        def fake_compress(_agent, messages, system_message, **_kwargs):
            observed.append(_runtime_main_value("cache_scope"))
            return messages, system_message

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context", fake_compress
        )

        with scoped_runtime_main({"cache_scope": "caller"}):
            assert agent._compress_context([], "system", commit_fence=object()) == (
                [],
                "system",
            )
            assert _runtime_main_value("cache_scope") == "caller"

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(
                agent._compress_context,
                [],
                "system",
                commit_fence=object(),
            ).result(timeout=10) == ([], "system")
            assert (
                executor.submit(_runtime_main_value, "cache_scope").result(timeout=10)
                == ""
            )

        assert observed == ["root", "root"]
        assert _runtime_main_value("cache_scope") == ""
    finally:
        db.close()


def test_direct_compression_restores_runtime_scope_after_exception(
    tmp_path, monkeypatch
):
    from agent.auxiliary_client import _runtime_main_value, scoped_runtime_main

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("root", source="cli")
        agent = AIAgent.__new__(AIAgent)
        agent.session_id = "root"
        agent.platform = "cli"
        agent._session_db = db
        agent._parent_session_id = None
        agent._conversation_root_id = lambda: None
        observed = []

        def fail_compress(*_args, **_kwargs):
            observed.append(
                {
                    field: _runtime_main_value(field)
                    for field in (
                        "provider",
                        "model",
                        "base_url",
                        "api_mode",
                        "cache_scope",
                    )
                }
            )
            raise RuntimeError("compression failed")

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context", fail_compress
        )

        caller_runtime = {
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "base_url": "https://example.invalid/openai/v1",
            "api_mode": "codex_responses",
            "cache_scope": "caller",
        }
        with scoped_runtime_main(caller_runtime):
            with pytest.raises(RuntimeError, match="compression failed"):
                agent._compress_context([], "system", commit_fence=object())
            assert {
                field: _runtime_main_value(field) for field in caller_runtime
            } == caller_runtime

        assert observed == [
            {
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "base_url": "https://example.invalid/openai/v1",
                "api_mode": "codex_responses",
                "cache_scope": "root",
            }
        ]
        assert _runtime_main_value("cache_scope") == ""
        assert "_active_compression_commit_fence" not in vars(agent)
    finally:
        db.close()
