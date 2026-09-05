"""Focused contract tests for authoritative external memory routing."""

import asyncio
import json
import io
import logging
from contextlib import redirect_stdout
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.memory_manager import MemoryManager
from agent import agent_init
from tools.memory_tool import (
    check_memory_requirements,
    get_memory_provider_mode,
    memory_tool,
)


class RecordingAuthoritativeProvider:
    name = "synthetic-provider"

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "success": True,
            "drawer_id": "drawer-synthetic",
            "operation_id": "op-synthetic",
        }

    def get_tool_schemas(self):
        return []

    def authoritative_memory_write(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return json.dumps(self.result)


class RaisingAuthoritativeProvider(RecordingAuthoritativeProvider):
    def authoritative_memory_write(self, request, **kwargs):
        self.calls.append((request, kwargs))
        _traceback_local_sentinel = "traceback-local-sentinel"
        cause = ValueError("exception-cause-sentinel")
        raise RuntimeError(
            "exception-message-sentinel", "exception-args-sentinel"
        ) from cause


class RecordingBuiltinProvider:
    name = "builtin"

    def __init__(self):
        self.calls = []

    def get_tool_schemas(self):
        return []

    def authoritative_memory_write(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return json.dumps({"success": True})


class LegacyMemoryProvider:
    name = "legacy-provider"

    def __init__(self):
        self.handle_calls = []

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.handle_calls.append((tool_name, args, kwargs))
        return json.dumps({"success": True})


class RecordingToolProvider(LegacyMemoryProvider):
    name = "external-provider"

    def __init__(self, result):
        super().__init__()
        self.result = result

    def get_tool_schemas(self):
        return [{"name": "external_memory", "parameters": {}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.handle_calls.append((tool_name, args, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class RaisingToolProvider(RecordingToolProvider):
    def handle_tool_call(self, tool_name, args, **kwargs):
        self.handle_calls.append((tool_name, args, kwargs))
        _traceback_local_sentinel = "tool-traceback-local-sentinel"
        raise RuntimeError("tool-exception-sentinel", "tool-args-sentinel") from ValueError(
            "tool-cause-sentinel"
        )


class MemoryTarget(Enum):
    MEMORY = "memory"
    USER = "user"


@pytest.mark.parametrize("target", ["", " ", False, 0, [], {}, "other"])
def test_authoritative_rejects_invalid_targets_without_provider_call(target):
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": target, "content": "synthetic fact"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "invalid_target"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("memory", "memory"),
        ("user", "user"),
        (MemoryTarget.USER, "user"),
        (None, "memory"),
    ],
)
def test_authoritative_accepts_exact_targets_and_enum_values(target, expected):
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)
    request = {"action": "add", "content": "synthetic fact"}
    if target is not None:
        request["target"] = target

    result = json.loads(manager.authoritative_memory_write(request))

    assert result["success"] is True
    assert provider.calls[0][0]["target"] == expected


@pytest.mark.parametrize("bad_action", [[], {}], ids=["list", "dict"])
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_authoritative_rejects_unhashable_actions(bad_action, batch):
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)
    request = (
        {"operations": [{"action": bad_action}]}
        if batch
        else {"action": bad_action}
    )

    result = json.loads(manager.authoritative_memory_write(request))

    assert result["success"] is False
    assert result["error_class"] == "invalid_action"
    assert provider.calls == []



@pytest.mark.parametrize("target", ["", " ", False, 0, [], {}, "other"])
def test_builtin_target_validation_rejects_before_store_access(target):
    store_calls = []
    store = SimpleNamespace(
        target_enabled=lambda value: store_calls.append(value) or True,
    )

    result = json.loads(
        memory_tool(action="add", target=target, content="synthetic fact", store=store)
    )

    assert result["success"] is False
    assert result["error_class"] == "invalid_target"
    assert store_calls == []


def test_authoritative_explicit_none_target_defaults_to_memory():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": None, "content": "synthetic fact"}
        )
    )

    assert result["success"] is True
    assert provider.calls[0][0]["target"] == "memory"


def test_authoritative_manager_routes_batch_without_local_store():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {
                "target": "user",
                "operations": [
                    {"action": "remove", "old_text": "synthetic old fact"},
                    {"action": "add", "content": "synthetic new fact"},
                ],
            },
            metadata={"session_id": "synthetic-session"},
        )
    )

    assert result["success"] is True
    assert result["provider_mode"] == "authoritative"
    assert result["operation_id"] == "op-synthetic"
    assert provider.calls[0][0]["target"] == "user"
    assert provider.calls[0][1]["metadata"]["session_id"] == "synthetic-session"


def test_authoritative_requires_explicit_provider_write_capability():
    provider = LegacyMemoryProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "provider_capability_missing"
    assert provider.handle_calls == []


def test_authoritative_provider_receipt_is_content_free():
    provider = RecordingAuthoritativeProvider(
        {
            "success": False,
            "partial_write": True,
            "operation_id": "op-synthetic",
            "secret_payload": "must-not-escape",
        }
    )
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {
                "action": "replace",
                "target": "memory",
                "old_text": "synthetic old",
                "content": "synthetic new",
            }
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "partial_write"
    assert result["operation_id"] == "op-synthetic"
    assert "secret_payload" not in json.dumps(result)
    assert "synthetic new" not in json.dumps(result)


@pytest.mark.parametrize(
    "receipt",
    [
        '{"success":false,"success":true,"operation_id":"A"}',
        '{"success":true,"metadata":{"safe":true,"safe":false}}',
    ],
)
def test_authoritative_receipt_rejects_duplicate_json_keys(receipt):
    provider = RecordingAuthoritativeProvider()
    provider.authoritative_memory_write = lambda *_args, **_kwargs: receipt
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "provider_protocol_error"


def test_authoritative_provider_mutation_cannot_change_caller_or_receipt(caplog):
    class MutatingProvider(RecordingAuthoritativeProvider):
        def authoritative_memory_write(self, request, **kwargs):
            self.calls.append((request, kwargs))
            request["target"] = "provider-target-sentinel"
            request["action"] = "provider-action-sentinel"
            request["operations"].append({"action": "remove", "old_text": "sentinel"})
            request["operations"][0]["content"] = "provider-content-sentinel"
            request["operations"][0]["nested"]["value"] = "provider-nested-sentinel"
            return json.dumps({"success": True})

    provider = MutatingProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)
    request = {
        "target": "user",
        "operations": [
            {
                "action": "add",
                "new_text": "caller-content-sentinel",
                "nested": {"value": "caller-nested-sentinel"},
            }
        ],
    }
    original = json.loads(json.dumps(request))
    caplog.set_level(logging.WARNING, logger="agent.memory_manager")
    caplog.clear()

    result = json.loads(manager.authoritative_memory_write(request))

    assert request == original
    assert len(provider.calls) == 1
    seen = provider.calls[0][0]
    assert seen is not request
    assert seen["operations"] is not request["operations"]
    assert result == {
        "success": True,
        "provider_mode": "authoritative",
        "target": "user",
        "operation_count": 1,
    }
    rendered = json.dumps(result)
    assert "content-sentinel" not in rendered
    assert "nested-sentinel" not in rendered
    assert "provider-target-sentinel" not in rendered
    assert "provider-action-sentinel" not in rendered
    assert not caplog.records


@pytest.mark.parametrize("bad_value", [object(), "x" * (16 * 1024 + 1)])
def test_authoritative_request_rejects_noncanonical_or_oversized_graph(bad_value):
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "ok", "extra": bad_value}
        )
    )

    assert result["error_class"] == "invalid_request"
    assert provider.calls == []


def test_authoritative_request_rejects_cycle_without_provider_call():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)
    request = {"action": "add", "target": "memory", "content": "ok"}
    request["cycle"] = request

    result = json.loads(manager.authoritative_memory_write(request))

    assert result["error_class"] == "invalid_request"
    assert provider.calls == []


@pytest.mark.parametrize(
    "raw_result",
    [
        {"success": False, "error": "failure-sentinel", "payload": {"nested": "payload-sentinel"}},
        {"success": "yes", "content": "protocol-sentinel"},
        {"status": "failed", "response": {"nested": "response-sentinel"}},
        {"unknown": "unknown-field-sentinel"},
        ["unknown-result-sentinel"],
        "malformed-result-sentinel",
    ],
)
def test_provider_tool_failure_and_protocol_results_are_content_free(raw_result):
    provider = RecordingToolProvider(raw_result)
    manager = MemoryManager()
    manager.add_provider(provider)

    result = json.loads(manager.handle_tool_call("external_memory", {"secret": "request-sentinel"}))

    assert result == {
        "error": "Memory provider tool failed.",
        "success": False,
        "error_class": (
            "provider_protocol_error" if isinstance(raw_result, str) else "provider_error"
        ),
    }
    assert len(provider.handle_calls) == 1
    rendered = json.dumps(result)
    assert not any(
        sentinel in rendered
        for sentinel in (
            "failure-sentinel",
            "payload-sentinel",
            "protocol-sentinel",
            "response-sentinel",
            "unknown-field-sentinel",
            "unknown-result-sentinel",
            "malformed-result-sentinel",
            "request-sentinel",
        )
    )


def test_provider_tool_rejects_undocumented_success_content():
    raw_result = {
        "success": True,
        "content": {
            "items": [
                {
                    "text": "approved model content",
                    "metadata": {"secret": "nested-metadata-sentinel"},
                }
            ]
        },
        "metadata": {"nested": {"secret": "metadata-sentinel"}},
        "payload": {"secret": "payload-sentinel"},
        "response": {"secret": "response-sentinel"},
    }
    provider = RecordingToolProvider(raw_result)
    manager = MemoryManager()
    manager.add_provider(provider)

    serialized = manager.handle_tool_call("external_memory", {})

    assert json.loads(serialized)["error_class"] == "provider_error"
    assert "sentinel" not in serialized


def test_provider_tool_documented_handled_success_is_projected():
    args = {"query": "approved query"}
    provider = RecordingToolProvider(
        {
            "handled": "external_memory",
            "args": args,
        }
    )
    manager = MemoryManager()
    manager.add_provider(provider)

    serialized = manager.handle_tool_call("external_memory", args)

    assert json.loads(serialized) == {
        "success": True,
        "handled": "external_memory",
        "args": {"query": "approved query"},
    }


class TestProviderToolSuccessProjection:
    @staticmethod
    def call(raw_result, args=None, expected_calls=1):
        provider = RecordingToolProvider(raw_result)
        builtin = RecordingBuiltinProvider()
        manager = MemoryManager()
        manager.add_provider(builtin)
        manager.add_provider(provider)
        result = json.loads(
            manager.handle_tool_call("external_memory", {} if args is None else args)
        )
        assert len(provider.handle_calls) == expected_calls
        assert builtin.calls == []
        return result

    def test_provider_gets_detached_args_and_legacy_success_remains_exact(self):
        class EchoingProvider(RecordingToolProvider):
            def handle_tool_call(self, tool_name, args, **kwargs):
                self.handle_calls.append((tool_name, args, kwargs))
                return {"handled": tool_name, "args": args}

        original = {"query": "approved", "nested": {"value": "trusted"}}
        provider = EchoingProvider(None)
        manager = MemoryManager()
        manager.add_provider(provider)

        result = json.loads(manager.handle_tool_call("external_memory", original))
        seen_args = provider.handle_calls[0][1]

        assert seen_args is not original
        assert seen_args["nested"] is not original["nested"]
        assert original == {"query": "approved", "nested": {"value": "trusted"}}
        assert result == {
            "success": True,
            "handled": "external_memory",
            "args": original,
        }

    def test_provider_mutation_cannot_change_invocation_or_projected_result(self):
        class MutatingProvider(RecordingToolProvider):
            def handle_tool_call(self, tool_name, args, **kwargs):
                self.handle_calls.append((tool_name, args, kwargs))
                args["provider_top_sentinel"] = True
                args["nested"]["provider_nested_sentinel"] = True
                return {"handled": tool_name, "args": args}

        original = {"query": "approved", "nested": {"value": "trusted"}}
        provider = MutatingProvider(None)
        manager = MemoryManager()
        manager.add_provider(provider)

        serialized = manager.handle_tool_call("external_memory", original)

        assert original == {"query": "approved", "nested": {"value": "trusted"}}
        assert json.loads(serialized) == {
            "error": "Memory provider tool failed.",
            "success": False,
            "error_class": "provider_error",
        }
        assert "sentinel" not in serialized

    @pytest.mark.parametrize(
        "raw_result",
        [
            '{"handled":"external_memory","handled":"external_memory","args":{"query":"approved"}}',
            '{"handled":"external_memory","args":{"query":"approved","query":"approved"}}',
        ],
    )
    def test_duplicate_json_keys_fail_closed(self, raw_result):
        result = self.call(raw_result, {"query": "approved"})

        assert result["error_class"] == "provider_protocol_error"

    def test_ordinary_canonical_json_is_accepted(self):
        result = self.call(
            '{"args":{"query":"approved"},"handled":"external_memory"}',
            {"query": "approved"},
        )

        assert result == {
            "success": True,
            "handled": "external_memory",
            "args": {"query": "approved"},
        }

    @pytest.mark.parametrize(
        "raw_result",
        [
            {"handled": "other_memory", "args": {}},
            {"handled": "external-memory", "args": {}},
            {"handled": object(), "args": {}},
            {"handled": "external_memory", "args": {"query": "provider-changed"}},
            {"handled": "external_memory", "args": {}, "debug": "secret-sentinel"},
            {
                "handled": "external_memory",
                "args": {"query": "approved", "raw": {"secret": "secret-sentinel"}},
            },
        ],
    )
    def test_rejects_unbound_or_unknown_generic_forms(self, raw_result):
        result = self.call(raw_result, {"query": "approved"})

        assert result == {
            "error": "Memory provider tool failed.",
            "success": False,
            "error_class": "provider_error",
        }
        assert "sentinel" not in json.dumps(result)

    @pytest.mark.parametrize("kind", ["dict", "list"])
    def test_cycles_fail_closed(self, kind):
        value = {} if kind == "dict" else []
        if kind == "dict":
            value["cycle"] = value
        else:
            value.append(value)

        result = self.call(
            {"handled": "external_memory", "args": value}, value, expected_calls=0
        )

        assert result["error_class"] == "provider_error"

    def test_depth_boundary_is_fixed(self):
        def nested(level):
            value = "ok"
            for _ in range(level):
                value = [value]
            return {"value": value}

        accepted = nested(18)
        rejected = nested(19)

        assert self.call(
            {"handled": "external_memory", "args": accepted}, accepted
        )["success"] is True
        assert self.call(
            {"handled": "external_memory", "args": rejected}, rejected
        )["error_class"] == "provider_error"

    def test_encoded_byte_cap_boundary(self):
        cap = MemoryManager._PROVIDER_TOOL_RESULT_MAX_BYTES
        empty = {
            "success": True,
            "handled": "external_memory",
            "args": {"value": ""},
        }
        overhead = len(
            json.dumps(
                empty,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
        accepted = {"value": "x" * (cap - overhead)}
        rejected = {"value": "x" * (cap - overhead + 1)}

        assert self.call(
            {"handled": "external_memory", "args": accepted}, accepted
        )["success"] is True
        assert self.call(
            {"handled": "external_memory", "args": rejected}, rejected
        )["error_class"] == "provider_error"

    @pytest.mark.parametrize(
        "value",
        [
            "x" * 20_000,
            [None] * 20_000,
            float("nan"),
            float("inf"),
            object(),
        ],
    )
    def test_noncanonical_or_oversized_args_fail_closed(self, value):
        args = {"value": value}

        result = self.call(
            {"handled": "external_memory", "args": args}, args, expected_calls=0
        )

        assert result["error_class"] == "provider_error"


def test_provider_tool_exception_is_content_free_once_without_fallback(caplog):
    provider = RaisingToolProvider(None)
    builtin = RecordingBuiltinProvider()
    manager = MemoryManager()
    manager.add_provider(builtin)
    manager.add_provider(provider)

    caplog.set_level(logging.ERROR, logger="agent.memory_manager")
    caplog.clear()
    result = json.loads(manager.handle_tool_call("external_memory", {"secret": "request-sentinel"}))

    assert result["error_class"] == "provider_error"
    assert len(provider.handle_calls) == 1
    assert builtin.calls == []
    assert "sentinel" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert all(record.args == ("handle_tool_call",) for record in caplog.records)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_provider_tool_preserves_base_exceptions(error):
    provider = RecordingToolProvider(error)
    manager = MemoryManager()
    manager.add_provider(provider)

    with pytest.raises(type(error)):
        manager.handle_tool_call("external_memory", {})

    assert len(provider.handle_calls) == 1


def test_authoritative_provider_exception_is_content_free(caplog):
    provider = RaisingAuthoritativeProvider()
    builtin = RecordingBuiltinProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(builtin)
    manager.add_provider(provider)

    caplog.set_level(logging.WARNING, logger="agent.memory_manager")
    caplog.clear()
    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "sentinel-content"}
        )
    )

    assert result == {
        "error": "Authoritative memory provider failed.",
        "success": False,
        "error_class": "provider_error",
    }
    assert provider.calls and len(provider.calls) == 1
    assert builtin.calls == []
    assert not any(
        sentinel in caplog.text
        for sentinel in (
            "exception-message-sentinel",
            "exception-args-sentinel",
            "exception-cause-sentinel",
            "traceback-local-sentinel",
            "sentinel-content",
        )
    )
    assert all(record.exc_info is None for record in caplog.records)
    assert all(record.args == ("authoritative_memory_write",) for record in caplog.records)
    assert all(
        sentinel not in json.dumps(result)
        for sentinel in (
            "exception-message-sentinel",
            "exception-args-sentinel",
            "exception-cause-sentinel",
            "traceback-local-sentinel",
            "sentinel-content",
        )
    )


def test_authoritative_receipt_drops_provider_metadata_except_operation_id():
    provider = RecordingAuthoritativeProvider(
        {
            "success": True,
            "operation_id": "op-synthetic",
            "drawer_id": "drawer-secret",
            "operation_ids": ["op-synthetic"],
            "status": "committed",
            "provider_payload": {"secret": "must-not-escape"},
        }
    )
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert result["operation_id"] == "op-synthetic"
    assert "drawer_id" not in result
    assert "operation_ids" not in result
    assert "status" not in result
    assert "provider_payload" not in result


@pytest.mark.parametrize(
    ("operation_id", "preserved"),
    [
        ("A", True),
        ("op_1.2:three-four", True),
        (None, False),
        (7, False),
        (" op-1", False),
        ("op 1", False),
        ("op\n1", False),
        ("https://provider.invalid/op/1", False),
        ("op-é", False),
        ("x" * 65, False),
    ],
)
def test_authoritative_receipt_bounds_operation_id(operation_id, preserved):
    provider = RecordingAuthoritativeProvider(
        {"success": True, "operation_id": operation_id}
    )
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert ("operation_id" in result) is preserved
    if preserved:
        assert result["operation_id"] == operation_id


@pytest.mark.parametrize(
    ("success", "error_class", "expected"),
    [
        (False, "partial_write", "partial_write"),
        (False, "provider_rejected", "provider_rejected"),
        (False, "provider_error", "provider_error"),
        (False, "unknown-error-sentinel", "provider_error"),
        (True, "unknown-error-sentinel", None),
        (False, {"nested": "error-sentinel"}, "provider_error"),
    ],
)
def test_authoritative_receipt_allows_only_safe_error_classes(
    success, error_class, expected
):
    provider = RecordingAuthoritativeProvider(
        {"success": success, "error_class": error_class}
    )
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    serialized = manager.authoritative_memory_write(
        {"action": "add", "target": "memory", "content": "synthetic fact"}
    )
    result = json.loads(serialized)

    assert result.get("error_class") == expected
    assert "sentinel" not in serialized


def test_provider_schema_and_unavailable_reason_logs_are_content_free(caplog):
    class MalformedSchemaProvider(RecordingToolProvider):
        name = "provider-name-sentinel"

        def get_tool_schemas(self):
            return [{"metadata": {"secret": "schema-sentinel"}}]

    manager = MemoryManager()
    manager.add_provider(MalformedSchemaProvider({"success": True}))
    agent_init._warned_unavailable_providers.clear()
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        assert manager.get_all_tool_schemas() == []
        agent_init._warn_memory_provider_unavailable(
            "unavailable-name-sentinel", "unavailable-reason-sentinel"
        )

    assert "sentinel" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert all(record.args == () for record in caplog.records)


def test_authoritative_manager_fails_closed_without_provider():
    result = json.loads(
        MemoryManager(provider_mode="authoritative").authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "provider_unavailable"
    assert "current_entries" not in result


def test_mempal_without_provider_mode_defaults_authoritative_when_markdown_off():
    memory_config = {
        "provider": "mempal",
        "memory_enabled": False,
        "user_profile_enabled": False,
    }

    assert "provider_mode" not in memory_config
    assert get_memory_provider_mode(memory_config) == "authoritative"


def test_authoritative_mode_keeps_memory_tool_available_when_markdown_off():
    with patch(
        "hermes_cli.config.load_config",
        return_value={
            "memory": {
                "provider": "mempal",
                "provider_mode": "authoritative",
                "memory_enabled": False,
                "user_profile_enabled": False,
            }
        },
    ):
        assert check_memory_requirements() is True


def test_hybrid_mode_hides_memory_tool_when_markdown_off():
    with patch(
     "hermes_cli.config.load_config",
     return_value={
         "memory": {
             "provider": "mempal",
             "provider_mode": "hybrid",
             "memory_enabled": False,
             "user_profile_enabled": False,
         }
     },
 ), patch(
     "hermes_cli.config.load_config_readonly",
     return_value={
         "memory": {
             "provider": "mempal",
             "provider_mode": "hybrid",
             "memory_enabled": False,
             "user_profile_enabled": False,
         }
     },
 ):
        assert check_memory_requirements() is False


def test_status_reports_authoritative_storage_without_context_mode():
    from hermes_cli.memory_setup import cmd_status

    config = {
        "memory": {
            "provider": "mempal",
            "memory_enabled": False,
            "user_profile_enabled": False,
        }
    }
    output = io.StringIO()
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.memory_setup._get_available_providers", return_value=[]),
        patch("hermes_cli.tools_config._get_platform_tools", return_value=["memory"]),
        redirect_stdout(output),
    ):
        cmd_status(SimpleNamespace())

    text = output.getvalue()
    assert "provider_mode=authoritative" in text
    assert "storage_mode=authoritative_provider" in text
    assert "built_in_injection=disabled" in text
    assert "core_tool_routing=authoritative_provider" in text
    assert "provider_context_mode" not in text
