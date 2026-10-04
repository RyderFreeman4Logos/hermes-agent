"""A typed denial must not be mistaken for failed stream negotiation."""
import asyncio
from types import SimpleNamespace
import pytest
from agent import auxiliary_client as aux

@pytest.mark.parametrize('wire', ['sync', 'async'])
def test_public_progress_stream_denial_has_no_plain_retry(monkeypatch, tmp_path, wire):
    from hermes_cli import config
    assert '/ord52-recover-52/base/' in aux.__file__ or '/hermes-lane-52-59-4a3e16bf-20261003/' in aux.__file__
    (tmp_path / 'config.yaml').write_text('model: {provider: custom, default: primary}\nauxiliary:\n  classifier: {provider: custom, model: primary, base_url: https://primary.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    aux._reset_aux_unhealthy_cache()
    class Policy(RuntimeError):
        body = {'error': {'code': 'content_policy_violation'}}
    denial = Policy('connection reset')
    attempts = []
    def send(kwargs):
        attempts.append(bool(kwargs.get('stream')))
        if kwargs.get('stream'): raise denial
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='FORBIDDEN_PLAIN_RETRY', tool_calls=None))])
    def create(**kwargs): return send(kwargs)
    async def acreate(**kwargs): return send(kwargs)
    client = SimpleNamespace(base_url='https://primary.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=acreate if wire == 'async' else create)))
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client, 'primary'))
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    monkeypatch.setattr(aux, '_transient_retry_count', lambda: 0)
    try:
        with aux.aux_progress_hook(lambda: None), pytest.raises(Policy) as raised:
            kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'offline'}])
            asyncio.run(aux.async_call_llm(**kwargs)) if wire == 'async' else aux.call_llm(**kwargs)
        assert raised.value is denial
        assert attempts == [True]
    finally:
        print('PROGRESS_DENIAL_WITNESS', {'wire': wire, 'attempts': attempts})
        aux._reset_aux_unhealthy_cache()


@pytest.mark.parametrize('wire', ['sync', 'async'])
@pytest.mark.parametrize('stage', ['primary', 'primary_parameter', 'candidate', 'candidate_parameter', 'transient_retry'])
@pytest.mark.parametrize('shape,status,code', [('wrapped', None, 'content_policy_violation'), ('inner', 402, 'approval_denied'), ('type', 401, 'content_policy_violation'), ('response', 500, 'approval_denied'), ('cause', 429, 'content_policy_violation')])
def test_progress_consumers_preserve_all_terminal_denial_shapes(monkeypatch, tmp_path, wire, stage, shape, status, code):
    async_mode = wire == "async"
    from hermes_cli import config
    home = tmp_path / 'home'; home.mkdir()
    (home / 'config.yaml').write_text('auxiliary:\n  classifier:\n    provider: custom\n    model: primary\n    base_url: https://primary.invalid/v1\n    fallback_chain:\n      - {provider: openai-codex, model: first, base_url: https://first.invalid/v1}\n      - {provider: openai-codex, model: second, base_url: https://second.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    aux._reset_aux_unhealthy_cache()
    class Payment(RuntimeError): status_code = 402
    class Parameter(RuntimeError): status_code = 400
    class Policy(RuntimeError): pass
    denial = Policy("Unsupported parameter: 'temperature'; payment required; connection reset")
    denial.status_code = status
    body = {'error': {'code': code}} if shape == 'wrapped' else {'type' if shape == 'type' else 'code': code}
    if shape == 'response':
        denial.response = SimpleNamespace(status_code=status, json=lambda: body)
    elif shape == 'cause':
        cause = RuntimeError('explicit denial'); cause.body = body
        denial.__cause__ = cause
    else:
        denial.body = body
    attempts = []
    def send(model, kwargs):
        attempts.append((model, 'temperature' in kwargs))
        if model == 'primary':
            if stage == 'primary': raise denial
            if stage == 'primary_parameter':
                if 'temperature' in kwargs: raise Parameter("Unsupported parameter: 'temperature'")
                raise denial
            if stage == 'transient_retry':
                if len(attempts) == 1: raise ConnectionError('connection reset')
                raise denial
            raise Payment('payment required')
        if model == 'first':
            if stage == 'candidate_parameter' and 'temperature' in kwargs:
                raise Parameter("Unsupported parameter: 'temperature'")
            raise denial
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='FORBIDDEN', tool_calls=None))])
    def client(model):
        def create(**kwargs): return send(model, kwargs)
        async def acreate(**kwargs): return send(model, kwargs)
        return SimpleNamespace(base_url=f'https://{model}.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=acreate if async_mode else create)))
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    if wire == 'forced_stream':
        monkeypatch.setattr(aux, '_provider_requires_stream', lambda *a: True)
    primary = client('primary')
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (primary, 'primary'))
    monkeypatch.setattr(aux, 'resolve_provider_client', lambda provider, model, **kw: (client(model), model))
    monkeypatch.setattr(aux, '_to_async_client', lambda c, m, **kw: (c, m))
    monkeypatch.setattr(aux, '_try_main_agent_model_fallback', lambda *a, **kw: (None, None, ''))
    try:
        kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'hi'}], temperature=0.2)
        with aux.aux_progress_hook(lambda: None), pytest.raises(Policy) as raised:
            asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
        assert raised.value is denial
        # Progress negotiates stream then plain before parameter recovery; denial itself is terminal.
        expected = [('primary', True)]
        if stage == 'primary_parameter': expected += [('primary', True), ('primary', False)]
        elif stage == 'transient_retry': expected += [('primary', True)]
        elif stage.startswith('candidate'): expected += [('first', True)] + ([('first', True), ('first', False)] if stage.endswith('parameter') else [])
        assert attempts == expected
    finally:
        print('CONTROL_DENIAL_ATTEMPTS', attempts)
        aux._reset_aux_unhealthy_cache()


@pytest.mark.parametrize('wire', ['sync', 'async', 'forced_stream'])
@pytest.mark.parametrize('progress', [False, True])
def test_affordable_capacity_text_cannot_retry_a_typed_denial(monkeypatch, tmp_path, wire, progress):
    from hermes_cli import config
    (tmp_path / 'config.yaml').write_text('model: {provider: openrouter, default: primary}\nauxiliary:\n  classifier: {provider: openrouter, model: primary, base_url: https://primary.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    class Policy(RuntimeError):
        status_code = 402
        body = {'error': {'code': 'content_policy_violation'}}
    denial = Policy('payment required; can only afford 2048 tokens')
    attempts = []
    def send(**kwargs):
        attempts.append((kwargs.get('stream', False), kwargs.get('max_tokens', kwargs.get('max_completion_tokens'))))
        if len(attempts) == 1: raise denial
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='FORBIDDEN', tool_calls=None))])
    async def asend(**kwargs): return send(**kwargs)
    client = SimpleNamespace(base_url='https://primary.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=asend if wire == 'async' else send)))
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client, 'primary'))
    if wire == 'forced_stream': monkeypatch.setattr(aux, '_provider_requires_stream', lambda *a: True)
    aux._reset_aux_unhealthy_cache()
    try:
        with aux.aux_progress_hook((lambda: None) if progress else None), pytest.raises(Policy) as raised:
            kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'offline'}], max_tokens=4096)
            asyncio.run(aux.async_call_llm(**kwargs)) if wire == 'async' else aux.call_llm(**kwargs)
        assert raised.value is denial
        assert attempts == [(progress or wire == 'forced_stream', 4096)]
    finally:
        print('AFFORDABLE_DENIAL_WITNESS', attempts)
        aux._reset_aux_unhealthy_cache()
