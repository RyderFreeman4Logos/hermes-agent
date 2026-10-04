"""Public configured-chain counterexample: denial after parameter recovery."""
import asyncio
from types import SimpleNamespace
import pytest
from agent import auxiliary_client as aux

@pytest.mark.parametrize('wire', ['sync', 'async', 'forced_stream'])
@pytest.mark.parametrize('stage', ['primary', 'primary_parameter', 'candidate', 'candidate_parameter', 'transient_retry'])
@pytest.mark.parametrize('shape,status,code', [('wrapped', None, 'content_policy_violation'), ('inner', 402, 'approval_denied'), ('type', 401, 'content_policy_violation'), ('response', 500, 'approval_denied'), ('cause', 429, 'content_policy_violation')])
def test_control_denial_is_terminal(monkeypatch, tmp_path, wire, stage, shape, status, code):
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
        with pytest.raises(Policy) as raised:
            asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
        assert raised.value is denial
        expected = [('primary', True)]
        if stage == 'primary_parameter': expected += [('primary', False)]
        elif stage == 'transient_retry': expected += [('primary', True)]
        elif stage.startswith('candidate'): expected += [('first', True)] + ([('first', False)] if stage.endswith('parameter') else [])
        assert attempts == expected
    finally:
        print('CONTROL_DENIAL_ATTEMPTS', attempts)
        aux._reset_aux_unhealthy_cache()
