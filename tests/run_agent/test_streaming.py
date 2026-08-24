"""Tests for streaming token delivery infrastructure.

Tests the unified streaming API call, delta callbacks, tool-call
suppression, provider fallback, and CLI streaming display.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_stream_chunk(
    content=None, tool_calls=None, finish_reason=None,
    model=None, reasoning_content=None, usage=None,
):
    """Build a mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        reasoning=None,
    )
    choice = SimpleNamespace(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )
    chunk = SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )
    return chunk


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None, extra_content=None, model_extra=None):
    """Build a mock tool call delta."""
    func = SimpleNamespace(name=name, arguments=arguments)
    delta = SimpleNamespace(index=index, id=tc_id, function=func)
    if extra_content is not None:
        delta.extra_content = extra_content
    if model_extra is not None:
        delta.model_extra = model_extra
    return delta


def _make_empty_chunk(model=None, usage=None):
    """Build a chunk with no choices (usage-only final chunk)."""
    return SimpleNamespace(choices=[], model=model, usage=usage)


# ── Test: Streaming Accumulator ──────────────────────────────────────────


class TestStreamingAccumulator:
    """Verify that _interruptible_streaming_api_call accumulates content
    and tool calls into a response matching the non-streaming shape."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_text_only_response(self, mock_close, mock_create):
        """Text-only stream produces correct response shape."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            _make_stream_chunk(content="!", finish_reason="stop", model="test-model"),
            _make_empty_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Hello world!"
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.completion_tokens == 3

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sparse_delta_allows_missing_optional_fields(self, mock_close, mock_create):
        """Managed stream deltas may omit both content and tool_calls."""
        from run_agent import AIAgent

        sparse_delta = SimpleNamespace(reasoning_content=None, reasoning=None)
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=sparse_delta,
                        finish_reason=None,
                    )
                ],
                model="test-model",
                usage=None,
            ),
            _make_stream_chunk(
                content="done", finish_reason="stop", model="test-model"
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "done"
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sparse_tool_delta_allows_missing_nested_fields(
        self, mock_close, mock_create
    ):
        """A partial tool delta may contain arguments before its other fields."""
        from run_agent import AIAgent

        sparse_tool_delta = SimpleNamespace(
            index=0,
            function=SimpleNamespace(arguments='{"city":"Paris"}'),
        )
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(tool_calls=[sparse_tool_delta]),
                    )
                ],
                model="test-model",
                usage=None,
            ),
            _make_stream_chunk(finish_reason="tool_calls", model="test-model"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == ""
        assert tool_call.function.arguments == '{"city":"Paris"}'
        assert response.choices[0].finish_reason == "tool_calls"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_chat_stream_closes_original_provider_resource(
        self,
        mock_close,
        mock_create,
    ):
        from run_agent import AIAgent

        class ProviderStream:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return iter([
                    _make_stream_chunk(
                        content="Hello",
                        finish_reason="stop",
                        model="test-model",
                    )
                ])

            def close(self):
                self.closed = True

        provider_stream = ProviderStream()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = provider_stream
        mock_create.return_value = mock_client
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Hello"
        assert provider_stream.closed is True

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_native_gemini_endpoint_omits_stream_options(self, mock_close, mock_create):
        """Google's native Gemini REST endpoint rejects OpenAI-only stream_options."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([
            _make_stream_chunk(content="Paris", finish_reason="stop", model="gemini"),
        ])
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-3-flash-preview",
            provider="gemini",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Paris"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "stream_options" not in call_kwargs



    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_tool_call_response(self, mock_close, mock_create):
        """Tool call stream accumulates ID, name, and arguments."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_123", name="terminal")
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"command":')
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments=' "ls"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        tc = response.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].id == "call_123"
        assert tc[0].function.name == "terminal"
        assert tc[0].function.arguments == '{"command": "ls"}'





# ── Test: Streaming Callbacks ────────────────────────────────────────────


class TestStreamingCallbacks:
    """Verify that delta callbacks fire correctly."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_deltas_fire_in_order(self, mock_close, mock_create):
        """Callbacks receive text deltas in order."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="a"),
            _make_stream_chunk(content="b"),
            _make_stream_chunk(content="c"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        agent._interruptible_streaming_api_call({})

        assert deltas == ["a", "b", "c"]

    @pytest.mark.parametrize("pending", [False, True], ids=["direct", "pending"])
    @pytest.mark.parametrize(
        "delivery",
        ["display", "tts", "display+tts", "all-fail", "none"],
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_suppressed_tool_text_uses_physical_delivery_for_partial_recovery(
        self, mock_close, mock_create, pending, delivery, monkeypatch,
    ):
        """Suppressed direct/pending text shares display/TTS retry authority."""
        import httpx

        from hermes_constants import PARTIAL_STREAM_STUB_ID
        from run_agent import AIAgent

        text = "suppressed text"
        if pending:
            monkeypatch.setattr(
                "agent.chat_completion_helpers._provider_stream_text_may_be_sse",
                lambda _text: True,
            )
        tool_chunk = _make_stream_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call_1", name="terminal"),
        ])
        chunks = (
            [_make_stream_chunk(content=text), tool_chunk]
            if pending
            else [tool_chunk, _make_stream_chunk(content=text)]
        )

        def _stream(*_args, **_kwargs):
            yield from chunks
            raise httpx.RemoteProtocolError("peer closed connection")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _stream
        mock_create.return_value = mock_client
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

        display = []
        tts = []

        def _display(delta):
            display.append(delta)
            if delivery == "all-fail":
                raise RuntimeError("display failed")

        def _tts(delta):
            tts.append(delta)
            if delivery == "all-fail":
                raise RuntimeError("tts failed")

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=(
                _display if delivery in {"display", "display+tts", "all-fail"} else None
            ),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._stream_callback = (
            _tts if delivery in {"tts", "display+tts", "all-fail"} else None
        )

        physically_delivered = delivery in {"display", "tts", "display+tts"}
        if physically_delivered:
            response = agent._interruptible_streaming_api_call({})
            assert response.id == PARTIAL_STREAM_STUB_ID
            assert response.choices[0].message.content.startswith(text)
            assert "the action was not executed" in response.choices[0].message.content
        else:
            with pytest.raises(httpx.RemoteProtocolError):
                agent._interruptible_streaming_api_call({})

        for callback, observed in (
            (agent.stream_delta_callback, display),
            (agent._stream_callback, tts),
        ):
            if callback and physically_delivered:
                assert observed[0] == text
                assert len(observed) == 2
                assert "the action was not executed" in observed[1]
            elif callback:
                assert observed == [text]
            else:
                assert observed == []
        if physically_delivered:
            assert agent._current_streamed_assistant_text.startswith(text)
            assert "the action was not executed" in agent._current_streamed_assistant_text
        else:
            assert agent._current_streamed_assistant_text == ""

    @pytest.mark.parametrize(
        ("delivery", "expected_attempts"),
        [
            pytest.param("all-fail", 2, id="callback-failures-retry"),
            pytest.param("plugin-only", 2, id="plugin-only-retries"),
            pytest.param("tts", 1, id="tts-delivery-does-not-retry"),
        ],
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_transient_text_drop_retries_only_without_physical_delivery(
        self, mock_close, mock_create, delivery, expected_attempts, monkeypatch,
    ):
        """Retry authority comes only from successful display/TTS delivery."""
        from run_agent import AIAgent
        import httpx

        attempts = 0

        def _stream(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            yield _make_stream_chunk(content="first" if attempts == 1 else "retry")
            if attempts == 1:
                raise httpx.RemoteProtocolError("peer closed connection")
            yield _make_stream_chunk(finish_reason="stop")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _stream
        mock_create.return_value = mock_client

        def _fail(_text):
            raise RuntimeError("delivery failed")

        plugin_deltas = []
        monkeypatch.setattr(
            "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
            lambda hook, **payload: (
                plugin_deltas.append(payload["delta"])
                if hook == "on_stream_delta"
                else None
            ),
        )
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=None if delivery == "plugin-only" else _fail,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        tts = []
        agent._stream_callback = tts.append if delivery == "tts" else (
            None if delivery == "plugin-only" else _fail
        )

        response = agent._interruptible_streaming_api_call({})

        assert attempts == expected_attempts
        assert response.choices[0].message.content == (
            "retry" if expected_attempts == 2 else "first"
        )
        assert plugin_deltas == (["first", "retry"] if expected_attempts == 2 else ["first"])
        assert tts == (["first"] if delivery == "tts" else [])





# ── Test: Streaming Fallback ────────────────────────────────────────────


class TestStreamingFallback:
    """Verify streaming errors propagate to the main retry loop.

    Previously, streaming errors triggered an inline fallback to
    non-streaming.  Now they propagate so the main retry loop can apply
    richer recovery (credential rotation, provider fallback, backoff).
    The only special case: 'stream not supported' sets _disable_streaming
    so the *next* main-loop retry uses non-streaming automatically.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stream_not_supported_sets_flag_and_raises(self, mock_close, mock_create):
        """'not supported' error sets _disable_streaming and propagates."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Streaming is not supported for this model"
        )
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(Exception, match="Streaming is not supported"):
            agent._interruptible_streaming_api_call({})

        # The flag should be set so the main retry loop switches to non-streaming
        assert agent._disable_streaming is True


    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_response_object_disables_streaming_and_returns_final_response(
        self, mock_close, mock_create
    ):
        """Adapters that ignore stream=True should fall back cleanly."""
        from run_agent import AIAgent

        final_response = SimpleNamespace(
            model="copilot-acp",
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from ACP",
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = final_response
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="claude-sonnet-4.6",
            provider="copilot-acp",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        deltas = []
        agent._stream_callback = lambda text: deltas.append(text)

        response = agent._interruptible_streaming_api_call({})

        assert response is final_response
        assert agent._disable_streaming is True
        assert deltas == ["Hello from ACP"]


    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_moa_interrupt_closes_stream_handle(
        self, mock_create, mock_close_openai, mock_abort_openai
    ):
        """MoA interrupts must close the per-request stream, not the facade client."""
        from run_agent import AIAgent

        class _BlockingClosableStream:
            def __init__(self):
                self.entered = threading.Event()
                self.closed = threading.Event()
                self.close_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.entered.set()
                if not self.closed.wait(timeout=5):
                    raise TimeoutError("MoA test stream was not closed")
                raise RuntimeError("stream closed")

            def close(self):
                self.close_calls += 1
                self.closed.set()

        stream = _BlockingClosableStream()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="default",
            provider="moa",
            api_key="test-key",
            base_url="moa://local",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent.client = mock_client

        def _request_interrupt():
            assert stream.entered.wait(timeout=2)
            agent._interrupt_requested = True

        interrupter = threading.Thread(target=_request_interrupt, daemon=True)
        interrupter.start()

        with pytest.raises(InterruptedError):
            agent._interruptible_streaming_api_call({"model": "default", "messages": []})

        assert stream.closed.wait(timeout=2)
        assert stream.close_calls == 1
        mock_create.assert_called_once()
        mock_close_openai.assert_not_called()
        mock_abort_openai.assert_not_called()




    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sse_connection_lost_retried_as_transient(self, mock_close, mock_create):
        """SSE 'Network connection lost' (APIError w/ no status_code) retries like httpx errors.

        OpenRouter sends {"error":{"message":"Network connection lost."}} as an SSE
        event when the upstream stream drops.  The OpenAI SDK raises APIError from
        this.  It should be retried at the streaming level, same as httpx connection
        errors, then propagate to the main retry loop after exhaustion.
        """
        from run_agent import AIAgent
        import httpx

        # Create an APIError that mimics what the OpenAI SDK raises from SSE error events.
        # Key: no status_code attribute (unlike APIStatusError which has one).
        from openai import APIError as OAIAPIError
        sse_error = OAIAPIError(
            message="Network connection lost.",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            body={"message": "Network connection lost."},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = sse_error
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(OAIAPIError):
            agent._interruptible_streaming_api_call({})

        # Should retry 3 times (default HERMES_STREAM_RETRIES=2 → 3 attempts)
        assert mock_client.chat.completions.create.call_count == 3
        # Connection cleanup should happen for each failed retry
        assert mock_close.call_count >= 2

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_streaming_retry_pairs_final_retry_with_next_loop(
        self, mock_close, mock_create, monkeypatch, tmp_path
    ):
        import httpx
        import json
        from openai import APIError as OAIAPIError

        from agent import physical_attempt_diagnostics as diagnostics
        from hermes_cli import config
        from run_agent import AIAgent

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
        monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {
            "observability": {"physical_attempt_digests": {"enabled": True}}
        })
        diagnostics._LAST_ATTEMPT.clear()

        dropped = OAIAPIError(
            message="Network connection lost.",
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
            body={"message": "Network connection lost."},
        )

        def complete():
            return iter([
                _make_stream_chunk(content="ok"),
                _make_stream_chunk(finish_reason="stop"),
            ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [dropped, complete(), complete()]
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent.session_id = "session-1"
        request = {
            "model": "test-model",
            "messages": [{"role": "system", "content": "fixed"}],
        }

        for loop in (1, 2):
            agent._current_api_request_id = f"turn:api:{loop}"
            agent._interruptible_streaming_api_call(request)

        records = [
            json.loads(line)
            for line in (
                tmp_path / "observability" / "physical_attempt_digests.jsonl"
            ).read_text().splitlines()
        ]
        pair = next(record for record in records if record["phase"] == "pair")
        assert pair["previous_attempt_retry"] == 1



# ── Test: Reasoning Streaming ────────────────────────────────────────────


class TestReasoningStreaming:
    """Verify reasoning content is accumulated and callback fires."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reasoning_callback_fires(self, mock_close, mock_create):
        """Reasoning deltas fire the reasoning_callback."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(reasoning_content="Let me think"),
            _make_stream_chunk(reasoning_content=" about this"),
            _make_stream_chunk(content="The answer is 42"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        reasoning_deltas = []
        text_deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: text_deltas.append(t),
            reasoning_callback=lambda t: reasoning_deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert reasoning_deltas == ["Let me think", " about this"]
        assert text_deltas == ["The answer is 42"]
        assert response.choices[0].message.reasoning_content == "Let me think about this"
        assert response.choices[0].message.content == "The answer is 42"


# ── Test: _has_stream_consumers ──────────────────────────────────────────


class TestHasStreamConsumers:
    """Verify _has_stream_consumers() detects registered callbacks."""

    def test_no_consumers(self):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent._has_stream_consumers() is False




# ── Test: Codex stream fires callbacks ────────────────────────────────


class TestCodexStreamCallbacks:
    """Verify _run_codex_stream fires delta callbacks."""

    def test_codex_text_delta_fires_callback(self):
        from run_agent import AIAgent

        deltas = []

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False

        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello from Codex!",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="r1", usage=None),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self_inner):
                return iter(events)
            def close(self_inner):
                return None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _FakeCreateStream()

        agent._run_codex_stream({}, client=mock_client)
        assert "Hello from Codex!" in deltas


    @pytest.mark.parametrize(
        ("event_type", "callback_name"),
        [
            ("response.output_text.delta", "on_text_delta"),
            ("response.reasoning_summary_text.delta", "on_reasoning_delta"),
        ],
    )
    def test_codex_callback_propagates_stream_payload_bound(
        self, event_type, callback_name,
    ):
        from agent.codex_runtime import _consume_codex_event_stream
        from agent.stream_payload_bound import StreamPayloadBoundExceeded

        overflow = StreamPayloadBoundExceeded(256 * 1024 + 1)

        def _overflow(_text):
            raise overflow

        with pytest.raises(StreamPayloadBoundExceeded) as caught:
            _consume_codex_event_stream(
                [SimpleNamespace(type=event_type, delta="x")],
                model="test/model",
                **{callback_name: _overflow},
            )

        assert caught.value is overflow

    @pytest.mark.parametrize(
        ("delivery", "expected_attempts"),
        [
            pytest.param("display", 1, id="display-no-retry"),
            pytest.param("tts", 1, id="tts-no-retry"),
            pytest.param("all-fail", 2, id="callback-failures-retry"),
            pytest.param("plugin-only", 2, id="plugin-only-retry"),
        ],
    )
    def test_codex_midstream_drop_retries_only_without_physical_delivery(
        self, delivery, expected_attempts, monkeypatch,
    ):
        import httpx

        from run_agent import AIAgent

        attempts = []

        class _FakeCreateStream:
            def __init__(self, text, *, fail):
                self.text = text
                self.fail = fail

            def __iter__(self):
                yield SimpleNamespace(
                    type="response.output_text.delta",
                    delta=self.text,
                )
                if self.fail:
                    raise httpx.RemoteProtocolError("peer closed connection")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(status="completed", id="r2", usage=None),
                )

            def close(self):
                return None

        def _create(**_kwargs):
            attempt = len(attempts) + 1
            attempts.append(attempt)
            return _FakeCreateStream(
                "first" if attempt == 1 else "retry",
                fail=attempt == 1,
            )

        delivered = []

        def _display(text):
            if delivery == "all-fail":
                raise RuntimeError("display failed")
            delivered.append(text)

        def _tts(text):
            if delivery == "all-fail":
                raise RuntimeError("tts failed")
            delivered.append(text)

        plugin_deltas = []
        monkeypatch.setattr(
            "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
            lambda hook, **payload: (
                plugin_deltas.append(payload["delta"])
                if hook == "on_stream_delta"
                else None
            ),
        )
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=(
                None if delivery in {"tts", "plugin-only"} else _display
            ),
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False
        agent._stream_callback = (
            _tts if delivery == "tts" else (_display if delivery == "all-fail" else None)
        )
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _create

        response = agent._run_codex_stream({}, client=mock_client)

        assert len(attempts) == expected_attempts
        assert response.output_text == (
            "retry" if expected_attempts == 2 else "first"
        )
        assert delivered == (["first"] if delivery in {"display", "tts"} else [])
        assert plugin_deltas == (
            ["first", "retry"] if expected_attempts == 2 else ["first"]
        )

    def test_codex_remote_protocol_error_retries_then_raises(self):
        """Transport errors from ``responses.create`` retry once then re-raise.

        With the migration from ``responses.stream(...)`` to
        ``responses.create(stream=True)``, there is no longer a separate
        fallback function — the same call IS the streaming path.  When it
        raises ``httpx.RemoteProtocolError``, we retry once (matching the
        old behavior on the helper) and re-raise on the second failure.
        """
        from run_agent import AIAgent
        import httpx

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False

        call_count = {"n": 0}

        def _create_side_effect(**kwargs):
            call_count["n"] += 1
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _create_side_effect

        with pytest.raises(httpx.RemoteProtocolError):
            agent._run_codex_stream({}, client=mock_client)

        # 1 initial + 1 retry = 2 calls
        assert call_count["n"] == 2

    def test_codex_retry_metadata_combines_outer_and_inner_attempts(self, monkeypatch):
        import httpx

        from agent import relay_llm
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False
        agent._current_api_retry_count = 2

        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="r1", usage=None),
            ),
        ]
        retries = []

        def fake_stream(*_args, metadata, **_kwargs):
            retries.append(metadata["retry_count"])
            if len(retries) == 1:
                raise httpx.ConnectError(
                    "retry",
                    request=httpx.Request("POST", "https://example.invalid"),
                )
            return iter(events)

        monkeypatch.setattr(relay_llm, "stream", fake_stream)

        agent._run_codex_stream({}, client=MagicMock())

        assert retries == [2, 3]

    def test_codex_create_stream_fallback_refreshes_activity_on_every_event(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Hello"),
            SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(type="message")),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="Hello")],
                    )]
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self_inner):
                return iter(events)

            def close(self_inner):
                return None

        mock_stream = _FakeCreateStream()

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_stream

        agent._run_codex_create_stream_fallback(
            {"model": "test/model", "instructions": "hi", "input": []},
            client=mock_client,
        )

        assert touch_calls.count("receiving stream response") == len(events)


class TestAnthropicStreamCallbacks:
    """Verify Anthropic streaming refreshes activity on every event."""

    def test_anthropic_stream_refreshes_activity_on_every_event(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="thinking"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", name="terminal"),
            ),
        ]

        final_message = SimpleNamespace(
            content=[],
            stop_reason="end_turn",
        )

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter(events))
        mock_stream.get_final_message.return_value = final_message

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.return_value = mock_stream
        # #67142: streaming now runs on a request-local anthropic client; route
        # it to the test mock so .messages.stream is exercised.
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        agent._interruptible_streaming_api_call({})

        assert touch_calls.count("receiving stream response") == len(events)
        mock_stream.close.assert_called_once()

    @patch("run_agent.AIAgent._rebuild_anthropic_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_anthropic_stream_parser_valueerror_retries_before_delivery(
        self, mock_replace, mock_rebuild, monkeypatch,
    ):
        """Malformed Anthropic event-stream frames retry instead of surfacing HTTP None.

        On the Anthropic-native path the stream-retry cleanup must close + rebuild the
        Anthropic client, NOT the OpenAI primary client (which would fail with
        Missing-credentials and leave the wedged stream open). See #28161.
        """
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.minimax.io/anthropic",
            provider="minimax",
            model="MiniMax-M2.7",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class _BadStream:
            response = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                raise ValueError("expected ident at line 1 column 149")

        final_message = SimpleNamespace(content=[], stop_reason="end_turn")
        good_stream = MagicMock()
        good_stream.__enter__ = MagicMock(return_value=good_stream)
        good_stream.__exit__ = MagicMock(return_value=False)
        good_stream.__iter__ = MagicMock(return_value=iter([]))
        good_stream.get_final_message.return_value = final_message

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = [
            _BadStream(),
            good_stream,
        ]
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        response = agent._interruptible_streaming_api_call({})

        assert response is final_message
        assert agent._anthropic_client.messages.stream.call_count == 2
        # #67142: cleanup runs on the request-local anthropic client (closed,
        # worker-owned, via _close_request_client_once), never rebuilding the
        # shared client and never touching the OpenAI primary client.
        assert mock_replace.call_count == 0
        assert mock_rebuild.call_count == 0
        assert agent._anthropic_client.close.call_count >= 1

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_generic_anthropic_valueerror_still_propagates_without_stream_retry(
        self, mock_replace, monkeypatch,
    ):
        """Only known provider stream parser ValueErrors are treated as transient."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.minimax.io/anthropic",
            provider="minimax",
            model="MiniMax-M2.7",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = ValueError(
            "invalid local request shape"
        )
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        with pytest.raises(ValueError, match="invalid local request shape"):
            agent._interruptible_streaming_api_call({})

        assert agent._anthropic_client.messages.stream.call_count == 1
        assert mock_replace.call_count == 0


    @patch("run_agent.AIAgent._try_refresh_anthropic_client_credentials")
    @patch("run_agent.AIAgent._rebuild_anthropic_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_anthropic_eventless_sdk_assertion_normalized_to_empty_stream(
        self, mock_replace, mock_rebuild, mock_refresh,
    ):
        """Real-SDK shape: an eventless stream has no message_start, so
        get_final_message() raises AssertionError (final snapshot is None).
        That must be normalized to EmptyStreamError and retried as
        transient — not surface as a raw AssertionError."""
        from agent.errors import EmptyStreamError
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            model="claude-test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False

        empty_stream = MagicMock()
        empty_stream.__enter__ = MagicMock(return_value=empty_stream)
        empty_stream.__exit__ = MagicMock(return_value=False)
        empty_stream.__iter__ = MagicMock(side_effect=lambda: iter([]))
        empty_stream.get_final_message.side_effect = AssertionError()

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.return_value = empty_stream
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        with pytest.raises(EmptyStreamError):
            agent._interruptible_streaming_api_call({})

        assert agent._anthropic_client.messages.stream.call_count == 3
        assert mock_replace.call_count == 0
        assert mock_rebuild.call_count == 0


class TestPartialToolCallWarning:
    """Regression: when a stream dies mid tool-call argument generation after
    text was already delivered, the partial-stream stub at run_agent.py
    line ~6107 used to silently set ``tool_calls=None`` and return
    ``finish_reason=stop``, losing the attempted action with zero user-facing
    signal.  Live-observed Apr 2026 with MiniMax M2.7 on a 6-minute audit
    task — agent streamed commentary, emitted a write_file tool call,
    MiniMax stalled for 240 s mid-arguments, stale-stream detector killed
    the connection, the stub returned, session ended with no file written
    and no error shown.

    Fix: when the stream accumulator captured any tool-call names before the
    error, the stub now appends a user-visible warning to content AND fires
    it as a stream delta so the user sees it immediately.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_tool_call_surfaces_warning(self, mock_close, mock_create):
        """Stream with text + partial tool-call name + mid-stream error
        produces a stub whose content contains the user-visible warning
        and whose tool_calls is None."""
        from run_agent import AIAgent

        class _StallError(RuntimeError):
            pass

        def _stalling_stream():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise _StallError("simulated upstream stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        fired_deltas: list = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text) or True
        agent._current_streamed_assistant_text = "Let me write the audit: "

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "0"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        content = response.choices[0].message.content or ""
        assert "Let me write the audit:" in content, (
            f"Partial text not preserved in stub: {content!r}"
        )
        assert "Stream stalled mid tool-call" in content, (
            f"Stub content is missing the dropped-tool-call warning; users "
            f"get silent failure.  Got content={content!r}"
        )
        assert "write_file" in content, (
            f"Warning should name the dropped tool. Got: {content!r}"
        )
        assert response.choices[0].message.tool_calls is None
        assert any("Stream stalled mid tool-call" in d for d in fired_deltas), (
            f"Warning was not surfaced as a live stream delta. "
            f"fired_deltas={fired_deltas}"
        )


    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_empty_partial_stream_stub_stays_empty_for_loop_guard(
        self, mock_close, mock_create,
    ):
        """Stream dies with 0 recovered chars and no tool call → the stub
        keeps its empty content ON PURPOSE.

        The conversation loop's truncation path detects an EMPTY
        partial-stream stub (PARTIAL_STREAM_STUB_ID + no content) and skips
        appending it to history entirely — only the continuation nudge is
        sent (the #68041 class fix).  An earlier iteration substituted
        '[response interrupted]' placeholder text HERE, which defeated that
        guard: the stub no longer looked empty, entered history, and the
        placeholder leaked into the stitched final response.  Transcripts
        that already carry a persisted empty turn are healed at the send
        boundary by repair_empty_non_final_messages instead.
        """
        from run_agent import AIAgent
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        class _StallError(RuntimeError):
            pass

        def _stalling_stream():
            yield _make_stream_chunk(content="partial token")
            raise _StallError("simulated upstream stall after a delta")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _stalling_stream()
        )
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._fire_stream_delta = lambda text: True
        # A physical sink accepted the delta while recovered text is empty —
        # the exact partial-stub condition.
        agent._current_streamed_assistant_text = ""

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "0"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        # The stub must be RECOGNIZABLY empty so the loop guard can skip it.
        assert getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
        content = response.choices[0].message.content
        assert not content, (
            f"Empty-partial-stream stub must keep empty content so the "
            f"conversation loop's empty-stub guard can detect and skip it — "
            f"substituted text defeats the guard and leaks into the final "
            f"response. Got content={content!r}"
        )
        assert response.choices[0].message.tool_calls is None


class TestSilentRetryMidToolCall:
    """A physical display/TTS write commits the provider attempt."""

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_mid_tool_drop_after_delivery_does_not_retry(
        self, mock_close, mock_create, mock_replace,
    ):
        """A dropped partial tool call returns one safe partial, never replay."""
        from run_agent import AIAgent
        import httpx as _httpx

        attempts = {"n": 0}

        def _stream(*_args, **_kwargs):
            attempts["n"] += 1
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise _httpx.RemoteProtocolError("peer closed connection")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _stream
        mock_create.return_value = mock_client
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        fired_deltas = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text) or True

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        assert attempts["n"] == 1
        msg = response.choices[0].message
        assert msg.tool_calls is None
        assert fired_deltas[0] == "Let me write the audit: "
        assert "the action was not executed" in msg.content
        assert not any("reconnecting" in delta.lower() for delta in fired_deltas)

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_silent_retry_exhausted_falls_back_to_stub(
        self, mock_close, mock_create, mock_replace,
    ):
        """When all retry attempts fail with connection errors, fall back
        to the original stub-with-warning behaviour so the user isn't left
        with zero signal."""
        from run_agent import AIAgent
        import httpx as _httpx

        def _always_fails():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            raise _httpx.RemoteProtocolError("peer closed connection")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _always_fails()
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        fired_deltas: list = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text) or True

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "1"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        # After retries exhaust, the stub-with-warning path must engage.
        content = response.choices[0].message.content or ""
        assert "Stream stalled mid tool-call" in content, (
            f"Exhausted-retry fallback dropped the user-visible warning: {content!r}"
        )
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_no_silent_retry_for_text_only_stall(
        self, mock_close, mock_create, mock_replace,
    ):
        """Text-only stall (no tool call in flight) must NOT trigger silent
        retry — that's the case where the user saw the model's text reply
        and retrying would duplicate it with no benefit."""
        from run_agent import AIAgent
        import httpx as _httpx

        attempts = {"n": 0}

        def _text_stall(*a, **kw):
            attempts["n"] += 1

            def _gen():
                yield _make_stream_chunk(content="Here's my answer so far")
                raise _httpx.RemoteProtocolError("peer closed connection")
            return _gen()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _text_stall
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda _text: None,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        # Only one attempt: text-only stall short-circuits retry.
        assert attempts["n"] == 1, (
            f"Text-only stall should not silent-retry, got {attempts['n']} attempts"
        )
        content = response.choices[0].message.content or ""
        assert content == "Here's my answer so far", (
            f"Text-only stall regressed: {content!r}"
        )
        assert "Stream stalled" not in content, (
            f"Text-only stall should not emit tool-call warning: {content!r}"
        )


# ── Test: CopilotACP Streaming Decision ──────────────────────────────────


def _valid_acp_response():
    """Build a minimal valid non-streaming API response for copilot-acp."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from ACP",
                    tool_calls=None,
                    role="assistant",
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        model="claude-opus-4.7",
    )


def _make_acp_agent(provider="copilot-acp", base_url="acp://copilot"):
    """Create an AIAgent configured for copilot-acp with a stream consumer
    so _has_stream_consumers() returns True (ensuring the test exercises the
    ACP exclusion, not the no-consumer branch)."""
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-acp-key",
        base_url=base_url,
        provider=provider,
        model="claude-opus-4.7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=lambda text: None,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestCopilotACPStreamingDecision:
    """Verify that copilot-acp routes to the non-streaming path.

    CopilotACPClient communicates via subprocess stdio and returns a plain
    SimpleNamespace — not an iterable stream.  The streaming decision logic
    must detect ACP runtimes and route to _interruptible_api_call instead.
    """

    @patch("run_agent.get_tool_definitions", return_value=[])
    @patch("run_agent.check_toolset_requirements", return_value={})
    @patch("agent.copilot_acp_client.CopilotACPClient")
    def test_provider_name_triggers_non_streaming(
        self, mock_acp_cls, _mock_check, _mock_tools
    ):
        """provider='copilot-acp' → non-streaming path."""
        mock_acp_cls.return_value = MagicMock()
        agent = _make_acp_agent(provider="copilot-acp", base_url="acp://copilot")

        with (
            patch.object(agent, "_interruptible_api_call",
                         return_value=_valid_acp_response()) as mock_non_stream,
            patch.object(agent, "_interruptible_streaming_api_call") as mock_stream,
        ):
            # Verify the decision logic correctly disables streaming
            _use_streaming = True
            if getattr(agent, "_disable_streaming", False):
                _use_streaming = False
            elif (
                agent.provider == "copilot-acp"
                or str(agent.base_url or "").lower().startswith("acp://copilot")
                or str(agent.base_url or "").lower().startswith("acp+tcp://")
            ):
                _use_streaming = False

            assert _use_streaming is False
            # Call the non-streaming path as the loop would
            response = mock_non_stream({})
            mock_stream.assert_not_called()

    @patch("run_agent.get_tool_definitions", return_value=[])
    @patch("run_agent.check_toolset_requirements", return_value={})
    @patch("agent.copilot_acp_client.CopilotACPClient")
    def test_acp_base_url_triggers_non_streaming(
        self, mock_acp_cls, _mock_check, _mock_tools
    ):
        """base_url='acp://copilot' → non-streaming even without provider name."""
        mock_acp_cls.return_value = MagicMock()
        agent = _make_acp_agent(provider="custom", base_url="acp://copilot")
        agent.provider = "custom"

        _use_streaming = True
        if (
            agent.provider == "copilot-acp"
            or str(agent.base_url or "").lower().startswith("acp://copilot")
            or str(agent.base_url or "").lower().startswith("acp+tcp://")
        ):
            _use_streaming = False

        assert _use_streaming is False


    def test_non_acp_provider_allows_streaming(self):
        """Regular providers still get streaming enabled."""
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda text: None,
        )
        agent.api_mode = "chat_completions"

        _use_streaming = True
        if getattr(agent, "_disable_streaming", False):
            _use_streaming = False
        elif (
            agent.provider == "copilot-acp"
            or str(agent.base_url or "").lower().startswith("acp://copilot")
            or str(agent.base_url or "").lower().startswith("acp+tcp://")
        ):
            _use_streaming = False

        assert _use_streaming is True


class TestBedrockIamStreamingFallback:
    """bedrock_converse streaming branch: IAM denial of
    InvokeModelWithResponseStream falls back to converse() inline and sets
    _disable_streaming for the rest of the session."""

    def _make_bedrock_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "bedrock_converse"
        agent._interrupt_requested = False
        return agent

    def test_iam_denial_falls_back_inline_and_disables_streaming(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from botocore.exceptions import ClientError

        agent = self._make_bedrock_agent()

        client = MagicMock()
        client.converse_stream.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": (
                        "User is not authorized to perform: "
                        "bedrock:InvokeModelWithResponseStream"
                    ),
                }
            },
            operation_name="ConverseStream",
        )
        client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            response = agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )

        client.converse.assert_called_once()
        assert response.choices[0].message.content == "hi"
        assert getattr(agent, "_disable_streaming", False) is True


@pytest.mark.parametrize(
    "delivery",
    ["display", "tts", "all-fail", "plugin-only"],
)
def test_bedrock_drop_communicates_physical_delivery_to_outer_recovery(
    delivery, monkeypatch,
):
    """Bedrock returns a partial only after display/TTS acknowledged text."""
    pytest.importorskip(
        "botocore.exceptions",
        reason="botocore (with working exceptions module) required",
    )
    from hermes_constants import PARTIAL_STREAM_STUB_ID
    from run_agent import AIAgent

    delivered = []

    def _display(text):
        if delivery == "all-fail":
            raise RuntimeError("display failed")
        delivered.append(text)

    def _tts(text):
        if delivery == "all-fail":
            raise RuntimeError("tts failed")
        delivered.append(text)

    plugin_deltas = []
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.has_stream_observer_hooks",
        lambda: delivery == "plugin-only",
    )
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
        lambda hook, **payload: (
            plugin_deltas.append(payload["delta"])
            if hook == "on_stream_delta"
            else None
        ),
    )
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=(
            None if delivery in {"tts", "plugin-only"} else _display
        ),
    )
    agent.api_mode = "bedrock_converse"
    agent._interrupt_requested = False
    agent._stream_callback = (
        _tts if delivery == "tts" else (_display if delivery == "all-fail" else None)
    )

    def _events():
        yield {"contentBlockDelta": {"delta": {"text": "first"}}}
        raise ConnectionError("bedrock connection dropped")

    client = MagicMock()
    client.converse_stream.return_value = {"stream": _events()}
    with patch(
        "agent.bedrock_adapter._get_bedrock_runtime_client",
        return_value=client,
    ):
        if delivery in {"display", "tts"}:
            response = agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )
            assert response.id == PARTIAL_STREAM_STUB_ID
            assert response.choices[0].message.content == "first"
        else:
            with pytest.raises(ConnectionError, match="bedrock connection dropped"):
                agent._interruptible_streaming_api_call(
                    {"modelId": agent.model, "messages": []}
                )

    assert client.converse_stream.call_count == 1
    assert delivered == (["first"] if delivery in {"display", "tts"} else [])
    assert plugin_deltas == ["first"]


def test_bedrock_delivery_does_not_swallow_stream_payload_bound():
    pytest.importorskip(
        "botocore.exceptions",
        reason="botocore (with working exceptions module) required",
    )
    from agent.stream_payload_bound import (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
        StreamPayloadBoundExceeded,
    )
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=lambda _text: None,
    )
    agent.api_mode = "bedrock_converse"
    agent._interrupt_requested = False
    client = MagicMock()
    client.converse_stream.return_value = {
        "stream": iter([
            {"contentBlockDelta": {"delta": {"text": "first"}}},
            {
                "contentBlockDelta": {
                    "delta": {"text": "x" * DEFAULT_STREAM_PAYLOAD_BOUND_BYTES}
                }
            },
        ])
    }

    with patch(
        "agent.bedrock_adapter._get_bedrock_runtime_client",
        return_value=client,
    ):
        with pytest.raises(StreamPayloadBoundExceeded):
            agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )


class _BlockingEventStream:
    """Mock boto3 ``converse_stream()`` response whose event iterator blocks
    forever — simulates a provider that opens the stream then stops yielding
    events. The worker thread sits inside ``for event in event_stream`` exactly
    as a wedged Bedrock stream would, giving the liveness watchdog something to
    trip on."""

    def __init__(self, release):
        self._release = release

    def get(self, key, default=None):
        if key == "stream":
            return self
        return default

    def __iter__(self):
        return self

    def __next__(self):
        # Never yields — blocks until the test releases it (teardown) so the
        # daemon worker can exit instead of leaking a truly-hung thread.
        self._release.wait(timeout=30)
        raise StopIteration


def test_on_event_fires_per_bedrock_event():
    """FIX 1: on_event fires once for EVERY yielded Bedrock event — text,
    tool-input delta, messageStop, and metadata alike — providing wire-level
    liveness (not just text deltas)."""
    from agent.bedrock_adapter import stream_converse_with_callbacks

    events = [
        {"contentBlockDelta": {"delta": {"text": "a"}}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "x"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
    ]
    calls = {"n": 0}

    stream_converse_with_callbacks(
        {"stream": iter(events)},
        on_event=lambda: calls.__setitem__("n", calls["n"] + 1),
    )

    assert calls["n"] == len(events)


def test_on_event_exception_is_swallowed():
    """FIX 1: a raising on_event callback must never abort the stream."""
    from agent.bedrock_adapter import stream_converse_with_callbacks

    events = [{"messageStop": {"stopReason": "end_turn"}}]

    def _boom():
        raise ValueError("liveness hook blew up")

    resp = stream_converse_with_callbacks({"stream": iter(events)}, on_event=_boom)
    assert resp is not None
    assert resp.choices[0].finish_reason == "stop"


class TestBedrockStreamLivenessWatchdog:
    """FIX 1: Bedrock streaming participates in the #58962 cross-turn stale
    breaker and no longer hangs when the stream stops yielding events."""

    def _make_bedrock_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "bedrock_converse"
        agent._interrupt_requested = False
        return agent

    def test_stalled_stream_bumps_streak_and_aborts(self, monkeypatch):
        """A Bedrock stream that opens then stops yielding events trips the
        watchdog: it bumps the cross-turn stale streak and raises TimeoutError
        instead of hanging forever."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        import threading as _t

        # Tiny stale timeout so the watchdog trips quickly; give-up threshold
        # kept above 1 so a single call raises TimeoutError (not the breaker).
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.5")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 0
        release = _t.Event()

        client = MagicMock()
        client.converse_stream.return_value = _BlockingEventStream(release)

        try:
            with patch(
                "agent.bedrock_adapter._get_bedrock_runtime_client",
                return_value=client,
            ):
                with pytest.raises(TimeoutError):
                    agent._interruptible_streaming_api_call(
                        {"modelId": agent.model, "messages": []}
                    )
        finally:
            release.set()

        # Watchdog counted exactly one stale kill in the cross-turn breaker.
        assert agent._consecutive_stale_streams == 1

    def test_pre_elevated_streak_aborts_before_streaming(self, monkeypatch):
        """A streak already past the give-up threshold aborts at entry with
        RuntimeError — Bedrock never even opens a stream (cross-turn breaker)."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 5

        client = MagicMock()
        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            with pytest.raises(RuntimeError, match="unresponsive"):
                agent._interruptible_streaming_api_call(
                    {"modelId": agent.model, "messages": []}
                )

        client.converse_stream.assert_not_called()

    def test_successful_stream_resets_streak(self, monkeypatch):
        """A Bedrock stream that completes normally clears any prior stale
        streak so a recovered provider doesn't carry it into later turns."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "60")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 3  # simulate a prior wedged streak

        events = [
            {"contentBlockDelta": {"delta": {"text": "hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
        ]
        client = MagicMock()
        client.converse_stream.return_value = {"stream": iter(events)}

        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            response = agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )

        assert response.choices[0].message.content == "hi"
        assert agent._consecutive_stale_streams == 0


class TestBedrockReasoningStaleFloor:
    """The Bedrock inference-profile id -> reasoning stale-timeout floor
    normalizer must match floor-table keys regardless of whether the model
    is keyed with a dashed version (``claude-opus-4``) or a dotted version
    (``claude-sonnet-4.5``). Bedrock always dashes the version, so the
    normalizer has to try the alternate separator form."""

    @pytest.mark.parametrize(
        "model_id, expected",
        [
            # opus is keyed dashed/base (``claude-opus-4`` -> 240) and
            # matches the Bedrock dashed id unchanged.
            ("us.anthropic.claude-opus-4-6-v1:0", 240.0),
            # sonnet is keyed DOTTED (``claude-sonnet-4.5`` /
            # ``claude-sonnet-4.6`` -> 180). The Bedrock dashed id must
            # now resolve via the alternate version-separator form.
            ("us.anthropic.claude-sonnet-4-5-v1:0", 180.0),
            ("us.anthropic.claude-sonnet-4-6-v1:0", 180.0),
            # region prefix variations still strip correctly.
            ("eu.anthropic.claude-sonnet-4-5-v1:0", 180.0),
        ],
    )
    def test_bedrock_reasoning_models_resolve_floor(self, model_id, expected):
        from agent.chat_completion_helpers import _bedrock_reasoning_stale_floor

        assert _bedrock_reasoning_stale_floor(model_id) == expected


def test_conversation_loop_stream_limit_is_partial_not_interrupted():
    """A streamed turn reaches the provider and returns its partial text."""
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="[REDACTED]",
            base_url="https://example.com/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = iter([
        _make_stream_chunk(content="partial", finish_reason="stop", model="test/model"),
    ])
    stream_callback = MagicMock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("go", stream_callback=stream_callback)

    assert agent.client.chat.completions.create.called
    assert result["completed"] is True
    assert result["interrupted"] is False
    assert result["final_response"] == "partial"
    stream_callback.assert_called_once_with("partial")
