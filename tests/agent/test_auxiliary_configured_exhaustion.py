"""Independent offline public exhaustion/admission boundary oracle."""
import asyncio
from types import SimpleNamespace
import pytest
from agent import auxiliary_client as aux

@pytest.mark.parametrize('wire', ['sync', 'async', 'forced_stream'])
@pytest.mark.parametrize('failure', ['generic', 'capacity'])
def test_configured_exhaustion_preserves_primary_without_unconfigured_send(monkeypatch, tmp_path, wire, failure):
    from hermes_cli import config
    assert '/ord52-recover-52/base/' in aux.__file__ or '/hermes-lane-52-59-4a3e16bf-20261003/' in aux.__file__
    home = tmp_path / 'home'; home.mkdir()
    (home / 'config.yaml').write_text('model:\n  provider: openai-codex\n  default: unconfigured-main\nauxiliary:\n  classifier:\n    provider: custom\n    model: primary\n    base_url: https://primary.invalid/v1\n    fallback_chain:\n      - {provider: openai-codex, model: first, base_url: https://first.invalid/v1}\n      - {provider: openai-codex, model: second, base_url: https://second.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    aux._reset_aux_unhealthy_cache()
    class Payment(RuntimeError): status_code = 402
    original = ValueError('primary execution error') if failure == 'generic' else Payment('payment required')
    attempts = []
    resolutions = []
    def send(model, kwargs):
        attempts.append(model)
        if model == 'primary': raise original
        if model in ('first', 'second'): raise ValueError('candidate execution error')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='UNCONFIGURED_SEND', tool_calls=None))])
    def client(model):
        def create(**kwargs): return send(model, kwargs)
        async def acreate(**kwargs): return send(model, kwargs)
        return SimpleNamespace(base_url=f'https://{model}.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=acreate if wire == 'async' else create)))
    monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client('primary'), 'primary'))
    def resolve(provider, model, **kwargs):
        resolutions.append((provider, model))
        return client(model), model
    monkeypatch.setattr(aux, 'resolve_provider_client', resolve)
    monkeypatch.setattr(aux, '_to_async_client', lambda c, m, **kw: (c, m))
    if wire == 'forced_stream': monkeypatch.setattr(aux, '_provider_requires_stream', lambda *a: True)
    try:
        with pytest.raises(type(original)) as raised:
            kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'offline'}])
            asyncio.run(aux.async_call_llm(**kwargs)) if wire == 'async' else aux.call_llm(**kwargs)
        assert raised.value is original
        assert 'unconfigured-main' not in attempts
    finally:
        print('BOUNDARY_WITNESS', {'wire': wire, 'failure': failure, 'attempts': attempts, 'resolutions': resolutions})
        aux._reset_aux_unhealthy_cache()


@pytest.mark.parametrize('wire', ['sync', 'async'])
@pytest.mark.parametrize('phase', ['resolution', 'execution'])
@pytest.mark.parametrize('chain', ['task', 'main'])
def test_auto_exhaustion_never_discovers_unconfigured_account(monkeypatch, tmp_path, wire, phase, chain):
    from hermes_cli import config
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable
    policy = ('auxiliary:\n  classifier:\n    provider: auto\n    fallback_chain:\n'
              if chain == 'task' else 'auxiliary:\n  classifier: {provider: auto}\nfallback_providers:\n')
    (tmp_path / 'config.yaml').write_text('model: {provider: auto}\n' + policy +
        '      - {provider: openai-codex, model: first, base_url: https://first.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    aux._reset_aux_unhealthy_cache()
    attempts = []
    discovered = []
    original = ValueError('primary execution failed')
    def client(model):
        def send(**kwargs):
            attempts.append(model)
            if model == 'primary': raise original
            if model == 'first': raise ValueError('configured candidate failed')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='FORBIDDEN', tool_calls=None))])
        async def asend(**kwargs): return send(**kwargs)
        return SimpleNamespace(base_url=f'https://{model}.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=asend if wire == 'async' else send)))
    def discover():
        discovered.append('unconfigured')
        return client('unconfigured'), 'unconfigured'
    monkeypatch.setattr(aux, '_get_provider_chain', lambda: [('openai-codex', discover)])
    real_resolve = aux.resolve_provider_client
    def resolve(provider, model=None, *args, **kw):
        if phase == 'resolution' and provider == 'auto':
            return real_resolve(provider, model, *args, **kw)
        return (None, None) if phase == 'resolution' else (client(model), model)
    monkeypatch.setattr(aux, '_client_cache', {})
    monkeypatch.setattr(aux, 'resolve_provider_client', resolve)
    monkeypatch.setattr(aux, '_to_async_client', lambda c, m, **kw: (c, m))
    if phase == 'execution':
        monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client('primary'), 'primary'))
    try:
        with pytest.raises(AuxiliaryClientUnavailable if phase == 'resolution' else ValueError) as raised:
            kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'offline'}], main_runtime={'provider': 'auto'})
            asyncio.run(aux.async_call_llm(**kwargs)) if wire == 'async' else aux.call_llm(**kwargs)
        if phase == 'execution':
            assert raised.value is original
            assert attempts == ['primary', 'first']
        assert discovered == []
    finally:
        print('AUTO_EXHAUSTION_WITNESS', {'wire': wire, 'phase': phase, 'chain': chain, 'attempts': attempts, 'discovered': discovered})
        aux._reset_aux_unhealthy_cache()


@pytest.mark.parametrize('wire', ['sync', 'async'])
@pytest.mark.parametrize('phase', ['resolution', 'execution'])
@pytest.mark.parametrize('chain', ['task', 'main'])
def test_configured_resolution_denial_cannot_advance_to_next_destination(monkeypatch, tmp_path, wire, phase, chain):
    from hermes_cli import config
    policy = ('auxiliary:\n  classifier:\n    provider: auto\n    fallback_chain:\n'
              if chain == 'task' else 'auxiliary:\n  classifier: {provider: auto}\nfallback_providers:\n')
    (tmp_path / 'config.yaml').write_text('model: {provider: auto}\n' + policy +
        '      - {provider: openai-codex, model: first, base_url: https://first.invalid/v1}\n'
        '      - {provider: openai-codex, model: second, base_url: https://second.invalid/v1}\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(config, '_LOAD_CONFIG_CACHE', {})
    monkeypatch.setattr(config, '_RAW_CONFIG_CACHE', {})
    monkeypatch.setattr(aux, '_client_cache', {})
    class Payment(RuntimeError): status_code = 402
    class Policy(RuntimeError):
        status_code = 402
        body = {'error': {'code': 'approval_denied'}}
    denial = Policy('payment required; connection reset')
    attempts, resolutions = [], []
    def client(model):
        def send(**kwargs):
            attempts.append(model)
            if model == 'primary': raise Payment('payment required')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='FORBIDDEN', tool_calls=None))])
        async def asend(**kwargs): return send(**kwargs)
        return SimpleNamespace(base_url=f'https://{model}.invalid/v1', chat=SimpleNamespace(completions=SimpleNamespace(create=asend if wire == 'async' else send)))
    real_resolve = aux.resolve_provider_client
    def resolve(provider, model=None, *args, **kwargs):
        if provider == 'auto': return real_resolve(provider, model, *args, **kwargs)
        resolutions.append(model)
        if model == 'first': raise denial
        return client(model), model
    monkeypatch.setattr(aux, 'resolve_provider_client', resolve)
    monkeypatch.setattr(aux, '_to_async_client', lambda c, m, **kw: (c, m))
    if phase == 'execution': monkeypatch.setattr(aux, '_get_cached_client', lambda *a, **kw: (client('primary'), 'primary'))
    aux._reset_aux_unhealthy_cache()
    try:
        with pytest.raises(Policy) as raised:
            kwargs = dict(task='classifier', messages=[{'role': 'user', 'content': 'offline'}], main_runtime={'provider': 'auto'})
            asyncio.run(aux.async_call_llm(**kwargs)) if wire == 'async' else aux.call_llm(**kwargs)
        assert raised.value is denial
        assert resolutions == ['first']
        assert attempts == (['primary'] if phase == 'execution' else [])
    finally:
        print('RESOLUTION_DENIAL_WITNESS', {'attempts': attempts, 'resolutions': resolutions})
        aux._reset_aux_unhealthy_cache()
