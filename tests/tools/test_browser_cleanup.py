"""Regression tests for browser session cleanup and screenshot recovery."""

import os
from unittest.mock import patch


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
        self.orig_last_active_session_key = browser_tool._last_active_session_key.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._last_active_session_key.clear()
        self.browser_tool._last_active_session_key.update(self.orig_last_active_session_key)
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
            patch(
                "tools.process_registry.ProcessRegistry._is_host_pid_alive",
                return_value=False,
            ),
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


    def test_unknown_daemon_identity_retains_session_control_path(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-unknown"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345")
        browser_tool._active_sessions["task-unknown"] = {"session_name": session_name, "bb_session_id": None}
        browser_tool._session_last_activity["task-unknown"] = 123.0
        browser_tool._last_active_session_key["task-unknown"] = "task-unknown"
        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else None,
            ),
            patch("tools.process_registry.ProcessRegistry._is_host_pid_alive", return_value=True),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid", return_value=False) as terminate,
        ):
            browser_tool.cleanup_browser("task-unknown")
        terminate.assert_not_called()
        assert socket_dir.exists()
        assert "task-unknown" in browser_tool._active_sessions
        assert browser_tool._session_last_activity["task-unknown"] == 123.0
        assert browser_tool._last_active_session_key["task-unknown"] == "task-unknown"

    def test_unconfirmed_daemon_termination_retains_owner_until_retry(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-retry"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345")
        browser_tool._active_sessions["task-retry"] = {"session_name": session_name, "bb_session_id": None}
        browser_tool._session_last_activity["task-retry"] = 123.0
        browser_tool._last_active_session_key["task-retry"] = "task-retry"
        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else 77,
            ),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=[False, True]) as terminate,
        ):
            browser_tool.cleanup_browser("task-retry")
            assert "task-retry" in browser_tool._active_sessions
            assert browser_tool._session_last_activity["task-retry"] == 123.0
            assert browser_tool._last_active_session_key["task-retry"] == "task-retry"
            assert socket_dir.exists()

            browser_tool.cleanup_browser("task-retry")
            browser_tool.cleanup_browser("task-retry")

        assert terminate.call_count == 2
        assert "task-retry" not in browser_tool._active_sessions
        assert "task-retry" not in browser_tool._session_last_activity
        assert "task-retry" not in browser_tool._last_active_session_key
        assert not socket_dir.exists()

    def test_confirmed_gone_daemon_releases_owner(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-gone"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345")
        browser_tool._active_sessions["task-gone"] = {"session_name": session_name, "bb_session_id": None}
        browser_tool._session_last_activity["task-gone"] = 123.0
        browser_tool._last_active_session_key["task-gone"] = "task-gone"
        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True}),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else None,
            ),
            patch("tools.process_registry.ProcessRegistry._is_host_pid_alive", return_value=False),
        ):
            browser_tool.cleanup_browser("task-gone")

        assert "task-gone" not in browser_tool._active_sessions
        assert "task-gone" not in browser_tool._session_last_activity
        assert "task-gone" not in browser_tool._last_active_session_key
        assert not socket_dir.exists()

    def test_missing_daemon_evidence_retains_owner_and_control(self, tmp_path):
        browser_tool = self.browser_tool
        session_name = "sess-missing"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        browser_tool._active_sessions["task-missing"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-missing"] = 123.0
        browser_tool._last_active_session_key["task-missing"] = "task-missing"

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
        ):
            assert browser_tool.cleanup_browser("task-missing") is False

        assert socket_dir.exists()
        assert "task-missing" in browser_tool._active_sessions
        assert browser_tool._session_last_activity["task-missing"] == 123.0
        assert browser_tool._last_active_session_key["task-missing"] == "task-missing"

    def test_symlinked_daemon_evidence_is_never_followed(self, tmp_path):
        browser_tool = self.browser_tool
        session_name = "sess-symlink"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        external = tmp_path / "unrelated.pid"
        external.write_text("12345", encoding="utf-8")
        (socket_dir / f"{session_name}.pid").symlink_to(external)
        browser_tool._active_sessions["task-symlink"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-symlink"] = 123.0

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                return_value=True,
            ) as terminate,
        ):
            assert browser_tool.cleanup_browser("task-symlink") is False

        terminate.assert_not_called()
        assert external.read_text(encoding="utf-8") == "12345"
        assert socket_dir.exists()
        assert "task-symlink" in browser_tool._active_sessions

    def test_rename_replacement_cannot_delete_unrelated_tree(self, tmp_path):
        browser_tool = self.browser_tool
        from tools.process_registry import ProcessRegistry

        owner_start = ProcessRegistry._safe_host_start_time(os.getpid())
        session_name = "sess-rename"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        (socket_dir / f"{session_name}.pid").write_text("12345", encoding="utf-8")
        displaced = tmp_path / "displaced-control"
        browser_tool._active_sessions["task-rename"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-rename"] = 123.0

        def replace_after_verification(_pid, _expected_start):
            socket_dir.rename(displaced)
            socket_dir.mkdir(mode=0o700)
            (socket_dir / "keep.txt").write_text("unrelated", encoding="utf-8")
            return True

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch(
                "tools.browser_tool._verify_reapable_browser_daemon",
                return_value=True,
            ),
            patch(
                "tools.process_registry.ProcessRegistry._safe_host_start_time",
                side_effect=lambda pid: owner_start if pid == os.getpid() else 77,
            ),
            patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                side_effect=replace_after_verification,
            ),
        ):
            assert browser_tool.cleanup_browser("task-rename") is False

        assert (socket_dir / "keep.txt").read_text(encoding="utf-8") == "unrelated"
        assert displaced.exists()
        assert "task-rename" in browser_tool._active_sessions

    def test_inactivity_scanner_retries_unknown_cleanup(self, tmp_path):
        browser_tool = self.browser_tool
        session_name = "sess-inactive-unknown"
        socket_dir = tmp_path / f"agent-browser-{session_name}"
        socket_dir.mkdir(mode=0o700)
        _publish_owner(browser_tool, socket_dir, session_name)
        browser_tool._active_sessions["task-inactive"] = {
            "session_name": session_name,
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-inactive"] = 1.0
        cleanup_calls = []
        real_cleanup = browser_tool.cleanup_browser

        def cleanup(task_id):
            cleanup_calls.append(task_id)
            return real_cleanup(task_id)

        with (
            patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)),
            patch("tools.browser_tool.time.time", return_value=1000.0),
            patch.object(browser_tool, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 30),
            patch("tools.browser_tool._maybe_stop_recording"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True},
            ),
            patch.object(browser_tool, "cleanup_browser", side_effect=cleanup),
        ):
            browser_tool._cleanup_inactive_browser_sessions()
            browser_tool._cleanup_inactive_browser_sessions()

        assert cleanup_calls == ["task-inactive", "task-inactive"]
        assert browser_tool._session_last_activity["task-inactive"] == 1.0
        assert "task-inactive" in browser_tool._active_sessions
