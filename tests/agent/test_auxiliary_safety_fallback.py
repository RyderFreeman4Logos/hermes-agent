from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as auxiliary_client
from agent.context_compressor import ContextCompressor


def _response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class ProviderError(Exception):
    pass


def _quota_error():
    error = ProviderError("structured primary quota")
    error.status_code = 429
    error.body = {"error": {"code": "quota_exhausted"}}
    return error


class Harness:
    def __init__(
        self, mode, config, primary_error, candidate_outcomes, *, primary_outcomes=None
    ):
        self.mode = mode
        self.config = config
        self.primary_error = primary_error
        self.order = []
        self.primary = MagicMock()
        primary_call = self._call(
            "primary", primary_outcomes or [primary_error] * 3
        )
        self.primary.chat.completions.create = primary_call
        self.candidates = []
        for index, outcomes in enumerate(candidate_outcomes):
            client = MagicMock()
            client.api_key = config["fallback_chain"][index]["api_key"]
            client.base_url = f"https://fallback-{index}.example/v1"
            client.chat.completions.create = self._call(
                f"fallback-{index}", outcomes
            )
            self.candidates.append(client)

    def _call(self, label, outcomes):
        remaining = list(outcomes)

        def run(**_kwargs):
            self.order.append(label)
            outcome = remaining.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return AsyncMock(side_effect=run) if self.mode == "async" else MagicMock(side_effect=run)

    async def invoke(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                auxiliary_client, "_get_auxiliary_task_config",
                return_value=self.config,
            ))
            stack.enter_context(patch.object(
                auxiliary_client, "_resolve_task_provider_model",
                return_value=("primary", "primary-model", None, None, None),
            ))
            stack.enter_context(patch.object(
                auxiliary_client, "_get_cached_client",
                return_value=(self.primary, "primary-model"),
            ))
            stack.enter_context(patch.object(
                auxiliary_client, "_resolve_fallback_entry",
                side_effect=lambda entry: (
                    self.candidates[int(entry["model"].removeprefix("model-"))],
                    entry["model"],
                ),
            ))
            stack.enter_context(patch.object(
                auxiliary_client, "_to_async_client",
                side_effect=lambda client, model, **_kwargs: (client, model),
            ))
            stack.enter_context(patch.object(
                auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0,
            ))
            call = {
                "task": "compression",
                "messages": [{"role": "user", "content": "summarize"}],
            }
            if self.mode == "async":
                return await auxiliary_client.async_call_llm(**call)
            return auxiliary_client.call_llm(**call)


def _config(count=2, *, fallback_on=None):
    config = {
        "fallback_chain": [
            {
                "provider": f"provider-{index}",
                "model": f"model-{index}",
                "base_url": f"https://fallback-{index}.example/v1",
                "api_key": f"entry-key-{index}",
            }
            for index in range(count)
        ]
    }
    if fallback_on is not None:
        config["fallback_on"] = fallback_on
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_absent_fallback_on_walks_any_error_in_order_with_three_attempts(mode):
    safety_error = ProviderError("provider policy rejection")
    malformed = SimpleNamespace(choices=[])
    harness = Harness(
        mode,
        _config(),
        safety_error,
        [[malformed, malformed, malformed], [_response("second fallback")]],
    )

    result = await harness.invoke()

    assert result.choices[0].message.content == "second fallback"
    assert harness.order == ["primary"] * 3 + ["fallback-0"] * 3 + ["fallback-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_explicit_quota_gate_blocks_nonquota_primary_error(mode):
    safety_error = ProviderError("provider policy rejection")
    harness = Harness(
        mode,
        _config(fallback_on=["quota_exhausted"]),
        safety_error,
        [[_response("must not run")], [_response("must not run")]],
    )

    with pytest.raises(ProviderError) as caught:
        await harness.invoke()

    assert caught.value is safety_error
    assert harness.order == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_quota_admitted_chain_advances_past_safety_error(mode):
    candidate_error = ProviderError("provider safety rejection")
    harness = Harness(
        mode,
        _config(fallback_on=["quota_exhausted"]),
        _quota_error(),
        [[candidate_error] * 3, [_response("safe fallback")]],
    )

    result = await harness.invoke()

    assert result.choices[0].message.content == "safe fallback"
    assert harness.order == ["primary"] * 3 + ["fallback-0"] * 3 + ["fallback-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_exhaustion_preserves_original_primary_error(mode):
    primary_error = ProviderError("first primary error")
    later_primary_errors = [
        ProviderError("second primary error"),
        ProviderError("third primary error"),
    ]
    harness = Harness(
        mode,
        _config(),
        primary_error,
        [[ProviderError("first failed")] * 3, [TimeoutError("second failed")] * 3],
        primary_outcomes=[primary_error, *later_primary_errors],
    )

    with pytest.raises(ProviderError) as caught:
        await harness.invoke()

    assert caught.value is primary_error
    assert harness.order == ["primary"] * 3 + ["fallback-0"] * 3 + ["fallback-1"] * 3


def test_chain_credentials_are_captured_per_entry_and_missing_key_fails_closed(monkeypatch):
    monkeypatch.setenv("ENTRY_KEY", "captured-key")
    monkeypatch.setenv("PROVIDER_1_API_KEY", "ambient-key-must-not-be-used")
    config = {
        "fallback_chain": [
            {
                "provider": "custom",
                "model": "entry-model",
                "base_url": "https://entry.example/v1",
                "key_env": "ENTRY_KEY",
            },
            {"provider": "provider-1", "model": "missing-key"},
        ]
    }
    with patch.object(
        auxiliary_client, "_get_auxiliary_task_config", return_value=config
    ):
        chain, fallback_on = auxiliary_client._capture_configured_fallback_chain(
            "compression"
        )

    monkeypatch.setenv("ENTRY_KEY", "changed-after-capture")
    assert fallback_on is None
    assert chain[0]["api_key"] == "captured-key"
    assert chain[1]["api_key"] == ""

    client = MagicMock()
    with patch.object(
        auxiliary_client, "resolve_provider_client",
        return_value=(client, "entry-model"),
    ) as resolve:
        auxiliary_client._resolve_fallback_entry(chain[0])

    assert resolve.call_args.kwargs == {
        "model": "entry-model",
        "explicit_base_url": "https://entry.example/v1",
        "explicit_api_key": "captured-key",
        "api_mode": None,
    }
    assert resolve.call_args.args == ("custom",)

    with patch.object(
        auxiliary_client, "_resolve_fallback_entry"
    ) as missing_resolve, patch.object(
        auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0,
    ):
        assert auxiliary_client._try_configured_candidates_sync(
            "compression", (chain[1],),
            messages=[], temperature=None, max_tokens=None, tools=None,
            effective_timeout=1, effective_extra_body={}, reasoning_config=None,
        ) is None
    missing_resolve.assert_not_called()

    ambient_client = MagicMock()
    ambient_client.api_key = "ambient-key-must-not-be-used"
    ambient_client.base_url = "https://ambient.example/v1"
    with patch.object(
        auxiliary_client, "_resolve_fallback_entry",
        return_value=(ambient_client, chain[0]["model"]),
    ):
        assert auxiliary_client._try_configured_candidates_sync(
            "compression", (chain[0],),
            messages=[], temperature=None, max_tokens=None, tools=None,
            effective_timeout=1, effective_extra_body={}, reasoning_config=None,
        ) is None
    ambient_client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "actual_base",
    [
        "http://entry.example/v1",
        "https://entry.example:8443/v1",
        "https://entry.example/ambient/v1",
    ],
)
def test_chain_endpoint_authority_rejects_same_host_substitution(actual_base):
    entry = {
        "api_key": "entry-key",
        "base_url": "https://entry.example/v1/",
    }
    client = MagicMock(api_key="entry-key", base_url=actual_base)

    assert auxiliary_client._fallback_client_matches_entry(client, entry) is False


def test_chain_endpoint_authority_normalizes_default_port_and_trailing_slash():
    entry = {
        "api_key": "entry-key",
        "base_url": "https://entry.example/v1/",
    }
    client = MagicMock(
        api_key="entry-key", base_url="https://ENTRY.example:443/v1"
    )

    assert auxiliary_client._fallback_client_matches_entry(client, entry) is True


def test_compression_main_model_retry_does_not_rewalk_exhausted_chain():
    config = _config(count=1)
    primary_error = ProviderError("primary rejected")
    candidate_error = ProviderError("fallback rejected")
    harness = Harness(
        "sync",
        config,
        primary_error,
        [[candidate_error] * 3],
        primary_outcomes=[primary_error] * 4,
    )
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        compressor = ContextCompressor(
            model="main-model",
            summary_model_override="summary-model",
            quiet_mode=True,
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            auxiliary_client, "_get_auxiliary_task_config", return_value=config,
        ))
        stack.enter_context(patch.object(
            auxiliary_client, "_resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch.object(
            auxiliary_client, "_get_cached_client",
            return_value=(harness.primary, "primary-model"),
        ))
        stack.enter_context(patch.object(
            auxiliary_client, "_resolve_fallback_entry",
            return_value=(harness.candidates[0], "model-0"),
        ))
        stack.enter_context(patch.object(
            auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0,
        ))
        assert compressor._generate_summary([
            {"role": "user", "content": "summarize this"},
            {"role": "assistant", "content": "working"},
        ]) is None

    assert harness.order == ["primary"] * 3 + ["fallback-0"] * 3 + ["primary"]


@pytest.mark.asyncio
async def test_sync_progress_stream_uses_one_request_per_outer_attempt():
    harness = Harness(
        "sync",
        _config(count=1),
        ProviderError("primary rejected"),
        [[_response("fallback")]],
    )
    ticks = []

    with auxiliary_client.aux_progress_hook(lambda: ticks.append(1)):
        result = await harness.invoke()

    assert result.choices[0].message.content == "fallback"
    assert harness.order == ["primary"] * 3 + ["fallback-0"]
    assert ticks
