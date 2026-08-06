"""Regression tests for browser session cleanup and screenshot recovery."""

import os
from unittest.mock import MagicMock, patch


def _publish_owner(browser_tool, socket_dir, session_name):
    with patch(
        "tools.browser_tool._socket_safe_tmpdir", return_value=str(socket_dir.parent)
    ):
        assert browser_tool._write_owner_pid(str(socket_dir), session_name) is True


class TestScreenshotPathRecovery:
    def test_extracts_standard_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text("Screenshot saved to /tmp/foo.png")
            == "/tmp/foo.png"
        )

    def test_extracts_quoted_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text(
                "Screenshot saved to '/Users/david/.hermes/browser_screenshots/shot.png'"
            )
            == "/Users/david/.hermes/browser_screenshots/shot.png"
        )


class TestBrowserCleanup:
    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_active_sessions = browser_tool._active_sessions.copy()
        self.orig_session_last_activity = browser_tool._session_last_activity.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done

    def test_cleanup_browser_clears_tracking_state(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-1"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345", encoding="utf-8")
        browser_tool._active_sessions["task-1"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch("tools.browser_tool._maybe_stop_recording") as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ) as mock_run,
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else None,
            ),
            patch("tools.process_registry.ProcessRegistry._is_host_pid_alive", return_value=False),
        ):
            assert browser_tool.cleanup_browser("task-1") is True

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_called_once_with("task-1")
        mock_run.assert_called_once_with("task-1", "close", [], timeout=10)


    def test_emergency_cleanup_retains_unconfirmed_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch("tools.browser_tool.cleanup_all_browsers") as mock_cleanup_all:
            browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_called_once_with()
        assert set(browser_tool._active_sessions) == {"task-1", "task-2"}
        assert browser_tool._session_last_activity == {"task-1": 1.0, "task-2": 2.0}
        assert browser_tool._recording_sessions == {"task-1", "task-2"}
        assert browser_tool._cleanup_done is False

    def test_unconfirmed_termination_retains_owner_for_retry(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-retry"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345", encoding="utf-8")
        browser_tool._active_sessions["task-retry"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-retry"] = 123.0

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else 77,
            ),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid", return_value=False),
        ):
            assert browser_tool.cleanup_browser("task-retry") is False

        assert socket_dir.exists()
        assert "task-retry" in browser_tool._active_sessions
        assert browser_tool._session_last_activity["task-retry"] == 123.0

    def test_failed_close_result_retains_owner_for_retry(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-close-failed"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345", encoding="utf-8")
        browser_tool._active_sessions["task-close-failed"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": False, "error": "close failed"},
            ),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else 77,
            ),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid") as terminate,
        ):
            assert browser_tool.cleanup_browser("task-close-failed") is False

        terminate.assert_not_called()
        assert "task-close-failed" in browser_tool._active_sessions
        assert socket_dir.exists()

    def test_provider_close_failure_retains_expired_session(self, tmp_path):
        browser_tool = self.browser_tool
        provider = MagicMock()
        provider.close_session.side_effect = RuntimeError("provider unavailable")
        browser_tool._active_sessions["task-provider-failed"] = {
            "session_name": "missing-control-dir",
            "bb_session_id": "cloud-123",
        }

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._session_has_expired", return_value=True),
            patch("tools.browser_tool._get_cloud_provider", return_value=provider),
        ):
            assert browser_tool.cleanup_browser("task-provider-failed") is False

        assert "task-provider-failed" in browser_tool._active_sessions

    def test_expired_provider_close_retains_unknown_local_daemon(self, tmp_path):
        browser_tool = self.browser_tool
        provider = MagicMock()
        browser_tool._active_sessions["task-provider-closed"] = {
            "session_name": "missing-control-dir",
            "bb_session_id": "cloud-123",
        }

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._session_has_expired", return_value=True),
            patch("tools.browser_tool._get_cloud_provider", return_value=provider),
        ):
            assert browser_tool.cleanup_browser("task-provider-closed") is False

        provider.close_session.assert_called_once_with("cloud-123")
        assert "task-provider-closed" in browser_tool._active_sessions

    def test_replaced_control_directory_is_not_deleted(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-replaced"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345", encoding="utf-8")
        displaced = tmp_path / "displaced-control"
        browser_tool._active_sessions["task-replaced"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }

        def replace_after_verification(_pid, _expected_start):
            socket_dir.rename(displaced)
            socket_dir.mkdir(mode=0o700)
            (socket_dir / "keep.txt").write_text("unrelated", encoding="utf-8")
            return True

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else 77,
            ),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                side_effect=replace_after_verification,
            ),
        ):
            assert browser_tool.cleanup_browser("task-replaced") is False

        assert (socket_dir / "keep.txt").read_text(encoding="utf-8") == "unrelated"
        assert displaced.exists()
        assert "task-replaced" in browser_tool._active_sessions
