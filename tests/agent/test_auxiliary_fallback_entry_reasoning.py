"""#130 — per-entry reasoning_effort on auxiliary fallback_chain."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import _call_fallback_candidate_sync, _call_llm_impl


def test_fallback_entries_apply_own_reasoning_effort():
    """Entry A high and entry B low apply; omitted gets no local-only payload."""
    chain = [
        {"provider": "custom", "model": "a", "reasoning_effort": "high"},
        {"provider": "custom", "model": "b", "reasoning_effort": "low"},
        {"provider": "custom", "model": "c"},
    ]
    task_body = {"reasoning": {"enabled": True, "effort": "medium"}}

    def _effort_for(label):
        seen = {}

        class _FakeCompletions:
            def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        client = SimpleNamespace(
            base_url="https://example.invalid/v1",
            chat=SimpleNamespace(completions=_FakeCompletions()),
        )
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": chain, "reasoning_effort": "medium"},
        ):
            resp = _call_fallback_candidate_sync(
                client,
                "model",
                label,
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
                temperature=None,
                max_tokens=None,
                tools=None,
                effective_timeout=30.0,
                effective_extra_body=task_body,
                reasoning_config=None,
            )
        assert resp is not None
        return (seen.get("extra_body") or {}).get("reasoning")

    assert _effort_for("fallback_chain[0](custom)") == {
        "enabled": True,
        "effort": "high",
    }
    assert _effort_for("fallback_chain[1](custom)") == {
        "enabled": True,
        "effort": "low",
    }
    assert _effort_for("fallback_chain[2](custom)") is None
    assert task_body == {"reasoning": {"enabled": True, "effort": "medium"}}


def _rate_limit_err():
    err = Exception("rate limited")
    err.status_code = 429
    return err


def _ok_resp(text):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _client_recording(seen, *, fail=False, text="ok"):
    client = MagicMock()
    client.base_url = "https://example.invalid/v1"

    def _create(**kwargs):
        seen.append(dict(kwargs.get("extra_body") or {}))
        if fail:
            raise _rate_limit_err()
        return _ok_resp(text)

    client.chat.completions.create.side_effect = _create
    return client


def test_unavailable_primary_walk_does_not_rebind_task_extra_body():
    """Promotion + sequential walk: omitted entry avoids task-local controls."""
    chain = [
        {
            "provider": "custom",
            "model": "a",
            "reasoning_effort": "high",
            "thinking": {"type": "enabled"},
        },
        {"provider": "custom", "model": "b", "reasoning_effort": "low"},
        {"provider": "custom", "model": "c"},
    ]
    task_cfg = {"fallback_chain": chain, "reasoning_effort": "medium"}
    seen0, seen1, seen2 = [], [], []
    entry0 = _client_recording(seen0, fail=True)
    entry1 = _client_recording(seen1, fail=True)
    entry2 = _client_recording(seen2, text="from omitted")

    with patch(
        "agent.auxiliary_client._get_cached_client", return_value=(None, None)
    ), patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("ollama-cloud", "missing", None, None, None),
    ), patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=task_cfg,
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        side_effect=[
            (entry0, "a", "fallback_chain[0](custom)"),
            (entry1, "b", "fallback_chain[1](custom)"),
            (entry2, "c", "fallback_chain[2](custom)"),
        ],
    ), patch(
        "agent.auxiliary_client._try_main_agent_model_fallback",
        return_value=(None, None, ""),
    ), patch(
        "agent.auxiliary_client._provider_requires_stream", return_value=False
    ):
        resp = _call_llm_impl(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert resp.choices[0].message.content == "from omitted"
    assert seen0[0]["reasoning"] == {"enabled": True, "effort": "high"}
    assert seen0[0]["thinking"] == {"type": "enabled"}
    assert seen1[0]["reasoning"] == {"enabled": True, "effort": "low"}
    assert "thinking" not in seen1[0]
    assert seen2[0] == {}


def test_chunk_digests_reuse_selected_fallback_route():
    """Sibling digest chunks keep the first call's configured fallback route."""
    import agent.context_compressor as compressor_module

    seen = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
    )

    def _call(**kwargs):
        seen.append({
            "provider": kwargs.get("provider"),
            "model": kwargs.get("model"),
            "route_info": dict(kwargs.get("route_info") or {}),
        })
        if len(seen) == 1:
            kwargs["route_info"].update(
                provider="custom",
                model="fallback",
                fallback_label="fallback_chain[1](custom)",
            )
        return response

    compressor = object.__new__(compressor_module.ContextCompressor)
    compressor._lean_pristine_tools = {}
    with patch("agent.auxiliary_client.call_llm", side_effect=_call):
        result = compressor._build_chunk_digests(
            [{"role": "user", "content": "x" * 72_001}]
        )

    assert "### Segment 1/2" in result
    assert len(seen) == 2
    assert seen[0] == {"provider": None, "model": None, "route_info": {}}
    assert seen[1] == {
        "provider": "custom",
        "model": "fallback",
        "route_info": {"fallback_label": "fallback_chain[1](custom)"},
    }
