"""Focused contract tests for authoritative provider-backed core memory."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.memory_manager import MemoryManager
from hermes_cli.memory_setup import cmd_status
from tools.memory_tool import check_memory_requirements


class RecordingAuthoritativeProvider:
    name = "synthetic-provider"

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "success": True,
            "drawer_id": "drawer-synthetic",
            "operation_id": "op-synthetic",
        }

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_scope = {"session_id": session_id, **kwargs}

    def unavailable_reason(self):
        return ""

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args, kwargs))
        return json.dumps(self.result)



def test_authoritative_mode_keeps_core_tool_when_markdown_flags_are_off(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "memory": {
                "provider_mode": "authoritative",
                "memory_enabled": False,
                "user_profile_enabled": False,
            }
        },
    )

    assert check_memory_requirements() is True



def test_authoritative_manager_routes_batch_without_local_store():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    raw = manager.authoritative_memory_write(
        {
            "target": "user",
            "operations": [
                {"action": "remove", "old_text": "synthetic old fact"},
                {"action": "add", "content": "synthetic new fact"},
            ],
        },
        metadata={"session_id": "synthetic-session"},
    )

    result = json.loads(raw)
    assert result["success"] is True
    assert result["operation_id"] == "op-synthetic"
    assert provider.calls[0][0] == "memory"
    assert provider.calls[0][1]["target"] == "user"
    assert len(provider.calls[0][1]["operations"]) == 2



def test_authoritative_manager_fails_closed_for_missing_provider():
    manager = MemoryManager(provider_mode="authoritative")

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "add", "target": "memory", "content": "synthetic fact"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "provider_unavailable"
    assert "current_entries" not in result



def test_authoritative_status_reports_mode_and_routing(monkeypatch, capfd):
    config = {
        "memory": {
            "provider": "synthetic-provider",
            "provider_mode": "authoritative",
            "memory_enabled": False,
            "user_profile_enabled": False,
        }
    }
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.memory_setup._get_available_providers", return_value=[]),
        patch(
            "hermes_cli.tools_config._get_platform_tools",
            return_value={"memory"},
        ),
    ):
        cmd_status(args=None)

    output = capfd.readouterr().out
    assert "provider_mode=authoritative" in output
    assert "core_tool_routing=authoritative_provider" in output
    assert "built_in_injection=disabled" in output



def test_authoritative_direct_actions_preserve_targets_and_scope():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")

    manager.add_provider(provider)
    for action, target, content, old_text in (
        ("add", "memory", "synthetic add", None),
        ("replace", "user", "synthetic replacement", "synthetic old"),
        ("remove", "memory", None, "synthetic remove"),
    ):
        raw = manager.authoritative_memory_write(
            {
                "action": action,
                "target": target,
                "content": content,
                "old_text": old_text,
            },
            metadata={"session_id": "synthetic-session", "gateway": "synthetic-gateway"},
        )
        assert json.loads(raw)["success"] is True

    assert [call[0] for call in provider.calls] == ["memory", "memory", "memory"]
    assert [call[1]["target"] for call in provider.calls] == ["memory", "user", "memory"]
    assert provider.calls[1][2]["metadata"]["session_id"] == "synthetic-session"



def test_authoritative_missing_old_text_does_not_call_provider():
    provider = RecordingAuthoritativeProvider()
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "remove", "target": "user"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "missing_old_text"
    assert provider.calls == []
    assert "current_entries" not in result



def test_authoritative_provider_errors_are_truthful_and_content_free():
    provider = RecordingAuthoritativeProvider(
        result={
            "success": False,
            "error_class": "ambiguous_match",
            "partial_write": True,
            "operation_id": "op-synthetic",
            "secret_payload": "must-not-escape",
        }
    )
    manager = MemoryManager(provider_mode="authoritative")
    manager.add_provider(provider)

    result = json.loads(
        manager.authoritative_memory_write(
            {"action": "replace", "target": "memory", "old_text": "synthetic old", "content": "synthetic new"}
        )
    )

    assert result["success"] is False
    assert result["error_class"] == "partial_write"
    assert result["operation_id"] == "op-synthetic"
    assert "secret_payload" not in result
    assert "synthetic new" not in json.dumps(result)



def test_hybrid_mode_keeps_core_tool_hidden_when_markdown_flags_are_off(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "memory": {
                "provider_mode": "hybrid",
                "memory_enabled": False,
                "user_profile_enabled": False,
            }
        },
    )

    assert check_memory_requirements() is False



def test_authoritative_agent_init_keeps_core_route_without_memory_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    provider = RecordingAuthoritativeProvider()
    config = {
        "memory": {
            "provider": "synthetic-provider",
            "provider_mode": "authoritative",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="synthetic-key",
            base_url="https://synthetic.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="synthetic-session",
        )

    assert agent._memory_store is None
    assert agent._memory_provider_mode == "authoritative"
    assert agent._memory_manager is not None
    assert agent._memory_manager.provider_mode == "authoritative"
    assert agent._memory_manager.providers == [provider]

    config["memory"]["provider_mode"] = "hybrid"
    assert agent._memory_provider_mode == "authoritative"
    assert agent._memory_manager.provider_mode == "authoritative"
