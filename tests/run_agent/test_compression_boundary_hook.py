"""Test: the context engine is notified of a compression-boundary rollover.

When _compress_context rotates session_id (compression split), the active
context engine receives on_session_start(new_sid, boundary_reason="compression",
old_session_id=<old>). This lets plugin engines (e.g. hermes-lcm) preserve
DAG lineage across the split instead of treating it as a fresh /new.

See hermes-lcm#68: after Hermes compresses and mints a new physical session,
LCM was losing continuity (compression_count: 1, store_messages: 0,
dag_nodes: 0). With boundary_reason="compression" plugins can distinguish
this from a real user-initiated /new.
"""

import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import (
    finalize_context_engine_compression_notification,
)
from hermes_cli.model_switch import (
    ModelSwitchResult,
    get_model_switch_after_compression,
    schedule_model_switch_after_compression,
)
from hermes_state_common import AuthorityWriteIndeterminateError


def _public_shape(messages):
    fields = ("role", "content", "tool_calls", "tool_call_id", "name")
    return [{key: row[key] for key in fields if key in row} for row in messages]


def _final_response(model):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=None,
    )


class TestCompressionBoundaryHook:
    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

    def test_on_session_start_called_with_compression_boundary(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            # Stub the context compressor: we only need to observe the hook.
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            # Avoid the summary-error warning path
            compressor._last_summary_error = None
            # MagicMock auto-creates truthy attrs; explicitly clear the abort
            # flag so the post-compress abort branch in
            # conversation_compression.py does not short-circuit before the
            # session-id rotation we are asserting on.
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor

            original_sid = agent.session_id
            messages = [
                {"role": "user", "content": f"m{i}"} for i in range(10)
            ]

            agent._compress_context(messages, "sys", approx_tokens=10_000)

            # Session_id rotated
            assert agent.session_id != original_sid, \
                "compression should rotate session_id when session_db is set"

            # Hook fired with boundary_reason="compression" and old_session_id
            calls = [
                c for c in compressor.on_session_start.call_args_list
            ]
            assert calls, "on_session_start was never called on the context engine"
            # Find the compression boundary call (there may be others from init)
            comp_calls = [
                c for c in calls
                if c.kwargs.get("boundary_reason") == "compression"
            ]
            assert comp_calls, (
                f"Expected an on_session_start call with "
                f"boundary_reason='compression', got {calls!r}"
            )
            call = comp_calls[-1]
            # Positional new session_id
            assert call.args and call.args[0] == agent.session_id, \
                f"Expected new session_id as first positional arg, got {call!r}"
            assert call.kwargs.get("old_session_id") == original_sid, \
                f"Expected old_session_id={original_sid!r}, got {call.kwargs!r}"
            assert len(comp_calls) == 1

    def test_automatic_notification_follows_core_persistence(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **kwargs: events.append(
                    kwargs.get("boundary_reason")
                )
            )
            agent.context_compressor = compressor
            original_publish = db.publish_compression_child

            def _record_publish(*args, **kwargs):
                result = original_publish(*args, **kwargs)
                events.append("persist")
                return result

            with patch.object(
                db, "publish_compression_child", side_effect=_record_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert events == ["persist", "compression"]

    def test_deferred_switch_applies_between_publication_and_boundary_hook(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **kwargs: events.append(kwargs.get("boundary_reason"))
            )
            agent.context_compressor = compressor
            original_publish = db.publish_compression_child

            def _record_publish(*args, **kwargs):
                value = original_publish(*args, **kwargs)
                events.append("persist")
                return value

            def _switch(model, provider, *_args):
                events.append("switch")
                agent.model = model
                agent.provider = provider

            agent.switch_model = _switch
            schedule_model_switch_after_compression(
                agent,
                ModelSwitchResult(
                    success=True,
                    new_model="new-model",
                    target_provider="new-provider",
                    context_length=128_000,
                ),
                on_applied=lambda *_args: events.append("frontend-sync"),
            )

            with patch.object(
                db, "publish_compression_child", side_effect=_record_publish
            ):
                _, returned_prompt = agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert events == ["persist", "switch", "frontend-sync", "compression"]
            assert agent.model == "new-model"
            assert "Model: new-model" in returned_prompt
            assert get_model_switch_after_compression(agent) is None
            child = db.get_session(agent.session_id)
            assert child["model"] == "new-model"
            assert "pending_model_switch_after_compression" not in (
                child["model_config"] or ""
            )

    def test_tui_host_applies_deferred_switch_before_compression_returns(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **kwargs: events.append(kwargs.get("boundary_reason"))
            )
            agent.context_compressor = compressor
            agent._defer_host_compression_publication = True
            agent._compression_host_publication_callback = lambda: events.append("host")

            def _switch(model, provider, *_args):
                events.append("switch")
                agent.model = model
                agent.provider = provider

            agent.switch_model = _switch
            schedule_model_switch_after_compression(
                agent,
                ModelSwitchResult(
                    success=True,
                    new_model="new-model",
                    target_provider="new-provider",
                    context_length=128_000,
                ),
                on_applied=lambda *_args: events.append("frontend-sync"),
            )

            agent._compress_context(
                [{"role": "user", "content": "request"}],
                "sys",
                approx_tokens=100,
                defer_context_engine_notification=True,
            )

            assert events == ["host", "switch", "frontend-sync", "compression"]
            assert agent.model == "new-model"
            assert get_model_switch_after_compression(agent) is None
            assert getattr(agent, "_pending_context_engine_compression_notification") is None

    def test_auto_threshold_serializes_switch_publication_before_first_request(
        self, monkeypatch
    ):
        from hermes_state import SessionDB
        from tui_gateway import server

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            agent.compression_in_place = True
            compressor = agent.context_compressor
            compressor.context_length = 64_000
            compressor.threshold_tokens = 1
            compressor.protect_first_n = 0
            compressor.protect_last_n = 0
            compressor.compression_count = 0
            compress_next = [True]
            monkeypatch.setattr(
                compressor,
                "should_compress",
                lambda _tokens: compress_next[0],
            )

            def compress(_messages, **_kwargs):
                compressor.compression_count += 1
                compress_next[0] = False
                compressor._last_compression_made_progress = True
                compressor._last_summary_fallback_used = False
                compressor._last_feasibility_skip = False
                compressor._last_compress_aborted = False
                return [
                    {
                        "role": "user",
                        "content": f"compressed trigger {compressor.compression_count}",
                    }
                ]

            monkeypatch.setattr(compressor, "compress", compress)

            requests = []
            old_client = MagicMock()
            target_client = MagicMock()

            def create(**kwargs):
                requests.append(kwargs["model"])
                return _final_response(kwargs["model"])

            target_client.chat.completions.create.side_effect = create
            old_client.chat.completions.create.side_effect = create
            agent.client = old_client
            agent._create_openai_client = MagicMock(return_value=target_client)
            pending = ModelSwitchResult(
                success=True,
                new_model="next-model",
                target_provider="openrouter",
                api_key="next-key",
                base_url=agent.base_url,
                api_mode="chat_completions",
                context_length=128_000,
            )
            session = {
                "agent": agent,
                "session_key": agent.session_id,
                "history": [],
                "history_lock": threading.Lock(),
                "history_version": 0,
                "running": True,
                "attached_images": [],
                "image_counter": 0,
                "cols": 80,
                "slash_worker": None,
                "show_reasoning": False,
                "tool_progress_mode": "all",
                "after_compression_model_switch": pending,
            }
            sid = "auto-threshold"
            server._sessions[sid] = session
            server._attach_model_switch_after_compression(sid, session, agent)
            monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
            monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
            monkeypatch.setattr(server, "render_message", lambda *_args: None)
            monkeypatch.setattr(server, "_get_db", lambda: None)
            monkeypatch.setattr(
                server, "_sync_agent_model_with_config", lambda *_args: None
            )
            monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)

            publication_entered = threading.Event()
            contender_attempted = threading.Event()
            marker_finished = threading.Event()
            contender_done = threading.Event()
            contender_acquired = []
            contender_errors = []
            holder_during_publication = []
            marker_lock_holders = []
            pending_during_publication = []
            contender_holder = f"test-contender:pid={os.getpid()}"
            original_sync = server._sync_session_key_after_compress
            original_append_message = db.append_message

            def sync_with_barrier(*args, **kwargs):
                original_sync(*args, **kwargs)
                if publication_entered.is_set():
                    return
                holder_during_publication.append(
                    db.get_compression_lock_holder(agent.session_id)
                )
                pending_during_publication.append(
                    get_model_switch_after_compression(agent) is pending
                )
                publication_entered.set()
                if not contender_attempted.wait(timeout=10):
                    raise RuntimeError("contender did not attempt lease acquisition")

            def append_message(*args, **kwargs):
                if kwargs.get("display_kind") != "model_switch":
                    return original_append_message(*args, **kwargs)
                marker_lock_holders.append(kwargs.get("compression_lock_holder"))
                try:
                    return original_append_message(*args, **kwargs)
                finally:
                    marker_finished.set()

            def contend() -> None:
                acquired = False
                try:
                    if not publication_entered.wait(timeout=10):
                        contender_errors.append("host publication did not start")
                        return
                    acquired = db.try_acquire_compression_lock(
                        agent.session_id,
                        contender_holder,
                        ttl_seconds=60,
                    )
                    contender_acquired.append(acquired)
                    contender_attempted.set()
                    if acquired and not marker_finished.wait(timeout=10):
                        contender_errors.append("model marker was not attempted")
                except BaseException as exc:
                    contender_errors.append(repr(exc))
                finally:
                    contender_attempted.set()
                    if acquired:
                        db.release_compression_lock(agent.session_id, contender_holder)
                    contender_done.set()

            monkeypatch.setattr(
                server, "_sync_session_key_after_compress", sync_with_barrier
            )
            monkeypatch.setattr(db, "append_message", append_message)
            contender_thread = threading.Thread(
                target=contend,
                name="deferred-switch-lease-contender",
            )
            contender_thread.start()

            try:
                server._run_prompt_submit("request", sid, session, "trigger")
                first_run = session.get("_run_thread")
                assert isinstance(first_run, threading.Thread)
                first_run.join(timeout=15)

                assert not first_run.is_alive()
                assert contender_done.wait(timeout=10)
                contender_thread.join(timeout=1)
                assert not contender_thread.is_alive()
                assert contender_errors == []
                assert pending_during_publication == [True]
                assert holder_during_publication[0]
                assert contender_acquired == [False]
                assert marker_lock_holders == [holder_during_publication[0]]
                assert requests == ["next-model"]
                assert agent._create_openai_client.call_count == 1
                assert get_model_switch_after_compression(agent) is None
                assert agent._model_switch_after_compression_state["state"] == "applied"
                durable = db.get_session(agent.session_id)
                assert durable is not None
                assert durable["model"] == "next-model"
                assert "pending_model_switch_after_compression" not in (
                    durable["model_config"] or ""
                )
                assert (
                    sum(
                        message.get("display_kind") == "model_switch"
                        for message in session["history"]
                    )
                    == 1
                )

                compress_next[0] = True
                session["running"] = True
                server._run_prompt_submit("request-2", sid, session, "followup")
                second_run = session.get("_run_thread")
                assert isinstance(second_run, threading.Thread)
                assert second_run is not first_run
                second_run.join(timeout=15)

                assert not second_run.is_alive()
                assert compressor.compression_count == 2
                assert requests == ["next-model", "next-model"]
                assert agent.model == "next-model"
                assert agent._create_openai_client.call_count == 1
                assert get_model_switch_after_compression(agent) is None
                assert db.get_session(agent.session_id)["model"] == "next-model"
                assert db.get_compression_lock_holder(agent.session_id) is None
            finally:
                publication_entered.set()
                contender_attempted.set()
                marker_finished.set()
                contender_thread.join(timeout=1)
                if db.get_compression_lock_holder(agent.session_id) == contender_holder:
                    db.release_compression_lock(agent.session_id, contender_holder)
                server._sessions.pop(sid, None)

    def test_manual_compression_switches_the_first_followup_request_once(
        self, monkeypatch
    ):
        from hermes_state import SessionDB
        from tui_gateway import server

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = agent.context_compressor

            def compress(_messages, **_kwargs):
                compressor.compression_count += 1
                compressor._last_compression_made_progress = True
                compressor._last_summary_fallback_used = False
                compressor._last_feasibility_skip = False
                compressor._last_compress_aborted = False
                return [{"role": "user", "content": "compressed history"}]

            monkeypatch.setattr(compressor, "compress", compress)

            requests = []
            target_client = MagicMock()
            target_client.chat.completions.create.side_effect = lambda **kwargs: (
                requests.append(kwargs["model"]) or _final_response(kwargs["model"])
            )
            agent._create_openai_client = MagicMock(return_value=target_client)
            pending = ModelSwitchResult(
                success=True,
                new_model="next-model",
                target_provider="openrouter",
                api_key="next-key",
                base_url=agent.base_url,
                api_mode="chat_completions",
                context_length=128_000,
            )
            session = {
                "agent": agent,
                "session_key": agent.session_id,
                "history": [
                    {"role": "user", "content": f"message {index}"}
                    for index in range(6)
                ],
                "history_lock": threading.Lock(),
                "history_version": 0,
                "running": False,
                "slash_worker": None,
                "after_compression_model_switch": pending,
            }
            sid = "manual-compression"
            server._sessions[sid] = session
            server._attach_model_switch_after_compression(sid, session, agent)
            monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
            monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)

            try:
                response = server._methods["session.compress"](
                    "request", {"session_id": sid}
                )
                assert "error" not in response

                agent.compression_enabled = False
                result = agent.run_conversation(
                    "followup", conversation_history=session["history"]
                )

                assert result["completed"] is True
                assert requests == ["next-model"]
                assert agent._create_openai_client.call_count == 1
                assert get_model_switch_after_compression(agent) is None
            finally:
                server._sessions.pop(sid, None)

    def test_deferred_notification_still_publishes_target_route_atomically(self):
        import json

        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            agent._ensure_db_session()
            pending = ModelSwitchResult(
                success=True,
                new_model="new-model",
                target_provider="new-provider",
                api_key="new-key",
                base_url="https://new.example/v1",
                api_mode="responses",
                          context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)

            def switch_model(model, provider, api_key, base_url, api_mode):
                agent.model = model
                agent.provider = provider
                agent.api_key = api_key
                agent.base_url = base_url
                agent.api_mode = api_mode

            agent.switch_model = MagicMock(side_effect=switch_model)
            old_sid = agent.session_id

            agent._compress_context(
                [{"role": "user", "content": "request"}],
                "sys",
                approx_tokens=100,
                defer_context_engine_notification=True,
            )

            child = db.get_session(agent.session_id)
            assert child["model"] == "test/model"
            assert get_model_switch_after_compression(agent) is pending
            compressor.on_session_start.assert_not_called()

            assert finalize_context_engine_compression_notification(
                agent, committed=True
            )
            child = db.get_session(agent.session_id)
            config = json.loads(child["model_config"])
            assert child["model"] == "new-model"
            assert config["provider"] == "new-provider"
            assert "Model: new-model" in child["system_prompt"]
            assert child["billing_provider"] == "new-provider"
            assert child["billing_mode"] == "responses"
            assert get_model_switch_after_compression(agent) is None
            compressor.on_session_start.assert_called_once_with(
                agent.session_id,
                boundary_reason="compression",
                old_session_id=old_sid,
                platform="cli",
                conversation_id=None,
            )

    @pytest.mark.parametrize("in_place", [False, True])
    def test_callback_failure_is_post_commit_reconciliation(self, in_place):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            agent.compression_in_place = in_place
            compressor = MagicMock()
            compressed = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            expected_public = _public_shape(compressed)
            compressor.compress.return_value = compressed
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.has_content_to_compress.return_value = True
            agent.context_compressor = compressor
            agent._ensure_db_session()
            original_sid = agent.session_id
            pending = ModelSwitchResult(
                success=True,
                new_model="new-model",
                target_provider="new-provider",
                api_key="new-key",
                base_url="https://new.example/v1",
                api_mode="responses",
                          context_length=128_000,
            )

            def fail_callback(*_args):
                raise OSError("injected frontend publication failure")

            schedule_model_switch_after_compression(
                agent,
                pending,
                on_applied=fail_callback,
            )

            def switch_model(model, provider, api_key, base_url, api_mode):
                agent.model = model
                agent.provider = provider
                agent.api_key = api_key
                agent.base_url = base_url
                agent.api_mode = api_mode

            agent.switch_model = MagicMock(side_effect=switch_model)
            messages = [
                {"role": "user", "content": f"m{i}"} for i in range(10)
            ]

            returned, returned_prompt = agent._compress_context(
                messages,
                "sys",
                approx_tokens=10_000,
            )

            assert _public_shape(returned) == expected_public
            assert all(message.get("_db_persisted") is True for message in returned)
            assert agent._flushed_db_message_ids == {
                id(message) for message in returned
            }
            assert "Model: test/model" in returned_prompt
            assert (agent.session_id == original_sid) is in_place
            assert agent.model == "test/model"
            assert get_model_switch_after_compression(agent) is pending
            published = db.get_session(agent.session_id)
            assert published["model"] == "test/model"
            stored = db.get_messages_as_conversation(agent.session_id)
            assert _public_shape(stored) == expected_public
            if in_place:
                assert db.get_session(original_sid)["end_reason"] is None
            else:
                assert db.get_session(original_sid)["end_reason"] == "compression"

    def test_auto_compression_aborts_on_authority_indeterminate_finalizer(self):
        """The automatic path must not continue to a post-compression request."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            pending = ModelSwitchResult(
                success=True,
                new_model="target-model",
                target_provider="target-provider",
                          context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)

            with patch(
                "hermes_cli.model_switch.apply_model_switch_after_compression",
                side_effect=AuthorityWriteIndeterminateError("authority indeterminate"),
            ), pytest.raises(AuthorityWriteIndeterminateError, match="indeterminate"):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert get_model_switch_after_compression(agent) is pending
            compressor.on_session_start.assert_not_called()

    def test_apply_failure_after_rotation_keeps_pending_and_old_route(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            agent._ensure_db_session()
            original_sid = agent.session_id
            pending = ModelSwitchResult(
                success=True,
                new_model="broken-model",
                target_provider="broken-provider",
                          context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)
            agent.switch_model = MagicMock(side_effect=RuntimeError("switch failed"))

            messages = [{"role": "user", "content": "request"}]
            returned, returned_prompt = agent._compress_context(
                messages,
                "sys",
                approx_tokens=100,
            )

            assert _public_shape(returned) == [
                {"role": "user", "content": "summary"}
            ]
            assert all(message.get("_db_persisted") is True for message in returned)
            assert agent._flushed_db_message_ids == {
                id(message) for message in returned
            }
            assert "Model: test/model" in returned_prompt
            assert "broken-model" not in returned_prompt
            assert agent.session_id != original_sid
            assert agent.model == "test/model"
            assert get_model_switch_after_compression(agent) is pending
            assert db.get_session(agent.session_id)["model"] == "test/model"
            compressor.on_session_start.assert_called_once()

    def test_publication_failure_rolls_back_switch_and_keeps_pending(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            agent._ensure_db_session()
            original_sid = agent.session_id
            original_route = (agent.model, agent.provider, agent.api_key, agent.base_url)
            pending = ModelSwitchResult(
                success=True,
                new_model="new-model",
                target_provider="new-provider",
                api_key="new-key",
                base_url="https://new.example/v1",
                          context_length=128_000,
            )
            on_applied = MagicMock()
            schedule_model_switch_after_compression(
                agent,
                pending,
                on_applied=on_applied,
            )
            scheduled_config = db.get_session(original_sid)["model_config"]

            def _switch(model, provider, api_key, base_url, _api_mode):
                agent.model = model
                agent.provider = provider
                agent.api_key = api_key
                agent.base_url = base_url

            agent.switch_model = MagicMock(side_effect=_switch)
            messages = [{"role": "user", "content": "request"}]
            with patch.object(
                db,
                "publish_compression_child",
                side_effect=RuntimeError("publication failed"),
            ):
                returned, _ = agent._compress_context(
                    messages,
                    "sys",
                    approx_tokens=100,
                )

            # Compression may stamp metadata (e.g. timestamp) on the returned
            # transcript even when publication aborts the boundary rewrite.
            assert [m.get("role") for m in returned] == [m.get("role") for m in messages]
            assert [m.get("content") for m in returned] == [
                m.get("content") for m in messages
            ]
            assert agent.session_id == original_sid
            assert (agent.model, agent.provider, agent.api_key, agent.base_url) == original_route
            agent.switch_model.assert_not_called()
            assert get_model_switch_after_compression(agent) is pending
            assert db.get_session(original_sid)["model_config"] == scheduled_config
            on_applied.assert_not_called()
            compressor.on_session_start.assert_not_called()

    def test_failure_before_persistence_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = RuntimeError("synthetic compression failure")
            agent.context_compressor = compressor
            pending = ModelSwitchResult(
                success=True,
                new_model="later-model",
                target_provider="later-provider",
                          context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)

            with pytest.raises(RuntimeError, match="synthetic compression failure"):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            compressor.on_session_start.assert_not_called()
            assert get_model_switch_after_compression(agent) is pending

    def test_cancelled_compression_keeps_deferred_switch(self):
        from agent.auxiliary_client import AuxiliaryExplicitCancellation
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = AuxiliaryExplicitCancellation()
            agent.context_compressor = compressor
            pending = ModelSwitchResult(
                success=True,
                new_model="later-model",
                target_provider="later-provider",
                context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)
            agent.switch_model = MagicMock()
            messages = [{"role": "user", "content": "request"}]

            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=100,
            )

            assert [message.get("role") for message in returned] == ["user"]
            assert [message.get("content") for message in returned] == ["request"]
            agent.switch_model.assert_not_called()
            assert get_model_switch_after_compression(agent) is pending
            compressor.on_session_start.assert_not_called()

    def test_no_progress_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = lambda messages, **_kwargs: messages
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            pending = ModelSwitchResult(
                success=True,
                new_model="later-model",
                target_provider="later-provider",
                context_length=128_000,
            )
            schedule_model_switch_after_compression(agent, pending)
            agent.switch_model = MagicMock()
            messages = [{"role": "user", "content": "request"}]

            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=100,
            )

            # No-progress may still stamp metadata; content must stay intact
            # and the context-engine boundary hook must not fire.
            assert [m.get("role") for m in returned] == [m.get("role") for m in messages]
            assert [m.get("content") for m in returned] == [
                m.get("content") for m in messages
            ]
            agent.switch_model.assert_not_called()
            assert get_model_switch_after_compression(agent) is pending
            compressor.on_session_start.assert_not_called()


    def test_no_hook_when_no_session_db(self):
        """Without session_db, session_id does not rotate and the hook is not fired."""
        from run_agent import AIAgent
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )

        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "x"}]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        agent.context_compressor = compressor

        original_sid = agent.session_id
        agent._compress_context([{"role": "user", "content": "m"}], "sys", approx_tokens=100)

        # No DB => no rotation => no compression-boundary hook
        assert agent.session_id == original_sid
        comp_calls = [
            c for c in compressor.on_session_start.call_args_list
            if c.kwargs.get("boundary_reason") == "compression"
        ]
        assert not comp_calls, (
            f"No compression hook should fire without session_db rotation, "
            f"got {comp_calls!r}"
        )

    def test_hook_failure_does_not_break_compression(self):
        """If the context engine raises from on_session_start, compression still completes."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False

            # Raise only on the compression-boundary call, not on earlier calls.
            def _raise_on_compression(*args, **kwargs):
                if kwargs.get("boundary_reason") == "compression":
                    raise RuntimeError("plugin exploded")
                return None
            compressor.on_session_start.side_effect = _raise_on_compression
            agent.context_compressor = compressor

            original_sid = agent.session_id

            # Must not raise
            compressed, _prompt = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed
            assert agent.session_id != original_sid


class TestSessionCompressEvent:
    """The session:compress event_callback fires after a compression split."""

    def _make_agent(self, session_db, event_callback=None):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
                event_callback=event_callback,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

    def _stub_compressor(self):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        return compressor

    def test_event_emitted_on_compression(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(
                db, event_callback=lambda et, ctx: events.append((et, ctx))
            )
            original_sid = agent.session_id
            agent.context_compressor = self._stub_compressor()

            agent._compress_context(
                [{"role": "user", "content": f"m{i}"} for i in range(10)],
                "sys",
                approx_tokens=10_000,
            )

            compress_events = [e for e in events if e[0] == "session:compress"]
            assert compress_events, f"session:compress not emitted, got {events!r}"
            _, ctx = compress_events[-1]
            assert ctx["session_id"] == agent.session_id
            assert ctx["old_session_id"] == original_sid
            assert ctx["compression_count"] == 1

    def test_no_callback_is_safe(self):
        """Compression must work when no event_callback is wired."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, event_callback=None)
            agent.context_compressor = self._stub_compressor()
            compressed, _ = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed
