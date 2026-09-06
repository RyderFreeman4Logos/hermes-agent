"""Synthetic-only URL userinfo redaction at non-navigation egress (#256).

Navigation/tool default redaction still leaves ``user:pass@`` intact (#34029).
Logs, monitoring export, and structured error descriptors are not navigation
surfaces and must not persist or serialize the fixture userinfo.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict

from agent.error_classifier import ClassifiedError, FailoverReason, classify_api_error
from agent.error_surface import build_error_surface_from_exception
from agent.monitoring.redaction import redact_for_export
from agent.redact import RedactingFormatter, redact_cdp_url, redact_sensitive_text

FIXTURE_USER = "fixture-user"
FIXTURE_PASS = "fixture-passw0rd"
FIXTURE_URL = f"http://{FIXTURE_USER}:{FIXTURE_PASS}@example.test/private/path"
BENIGN_URL = "http://example.test/public"


def _assert_userinfo_absent(text: str) -> None:
    assert FIXTURE_PASS not in text
    assert f"{FIXTURE_USER}:{FIXTURE_PASS}" not in text


class TestDefaultNavigationPreserved:
    def test_default_text_redaction_keeps_userinfo(self):
        assert redact_sensitive_text(FIXTURE_URL) == FIXTURE_URL

    def test_strict_and_cdp_redaction_remove_userinfo(self):
        strict = redact_sensitive_text(FIXTURE_URL, redact_url_credentials=True)
        _assert_userinfo_absent(strict)
        _assert_userinfo_absent(redact_cdp_url(FIXTURE_URL))
        assert "example.test" in strict


class TestMonitoringExport:
    def test_export_strips_userinfo_and_keeps_benign_host(self):
        out = redact_for_export(f"probe {FIXTURE_URL} ok")
        assert out is not None
        _assert_userinfo_absent(out)
        assert "example.test" in out

    def test_export_preserves_benign_url(self):
        out = redact_for_export(f"probe {BENIGN_URL}")
        assert out is not None
        assert BENIGN_URL in out


class TestRedactingFormatterLogs:
    def test_formatter_strips_compressor_warning_userinfo(self):
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="agent.context_compressor",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="lean chunk digest %d/%d failed: %s",
            args=(1, 2, ConnectionError(FIXTURE_URL)),
            exc_info=None,
        )
        result = formatter.format(record)
        _assert_userinfo_absent(result)
        assert "example.test" in result
        assert "lean chunk digest 1/2 failed:" in result

    def test_formatter_preserves_benign_url(self):
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="agent.context_compressor",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Summary model '%s' %s (%s). Falling back.",
            args=("aux", "failed", ConnectionError(BENIGN_URL)),
            exc_info=None,
        )
        assert BENIGN_URL in formatter.format(record)


class TestStructuredErrorDescriptors:
    def test_sync_classified_message_json_pickle_omit_userinfo(self):
        classified = classify_api_error(ConnectionError(f"connect {FIXTURE_URL}"))
        _assert_userinfo_absent(classified.message)
        _assert_userinfo_absent(repr(classified))
        _assert_userinfo_absent(json.dumps(asdict(classified), default=str))
        _assert_userinfo_absent(pickle.dumps(classified).decode("latin-1"))
        assert "example.test" in classified.message

    def test_async_surface_and_direct_descriptor_omit_userinfo(self):
        classified = ClassifiedError(
            reason=FailoverReason.timeout,
            message=f"async connect {FIXTURE_URL}",
        )
        _assert_userinfo_absent(classified.message)
        _assert_userinfo_absent(json.dumps(asdict(classified), default=str))
        _assert_userinfo_absent(pickle.dumps(classified).decode("latin-1"))

        surface = build_error_surface_from_exception(
            ConnectionError(f"async connect {FIXTURE_URL}"),
            provider="custom",
            model="local",
        )
        assert surface is not None
        _assert_userinfo_absent(json.dumps(surface, default=str))
        _assert_userinfo_absent(pickle.dumps(surface).decode("latin-1"))

    def test_benign_error_message_preserved(self):
        classified = classify_api_error(ConnectionError(f"connect {BENIGN_URL}"))
        assert BENIGN_URL in classified.message
