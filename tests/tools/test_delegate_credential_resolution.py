"""Frozen public-entrypoint failure matrix; no product mutations/inference."""
import json
from unittest.mock import patch

import pytest

from tests.tools.test_review_identity_dispatch import _dispatch, _named, _pool_profile, _runtime, URL, OWNED_URL
from tests.tools.test_delegate_model_pool_boundaries import _parent, _fake_child
from tools.delegate_tool import delegate_task


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))


def config_for(branch):
    config = _pool_profile()
    route = config['model_pool']['standard']
    if branch == 'derived-direct':
        del route['api_key']
    elif branch == 'provider-only':
        del route['base_url']
        del route['api_key']
    return config


@pytest.mark.parametrize('branch', ['fixed-direct', 'derived-direct', 'provider-only'])
@pytest.mark.parametrize('failure', [None, {}, [], 'invalid-runtime', 7, {'provider': 'custom'},
    _runtime(key='wrong-owned-key'), RuntimeError('resolver-unavailable'), ValueError('invalid-resolver'),
    TypeError('resolver-type-error'), OSError('resolver-io-error')],
    ids=['none', 'empty', 'list', 'string', 'int', 'incomplete', 'wrong-owner', 'runtime-error', 'value-error', 'type-error', 'os-error'])
def test_all_resolver_failures_return_refusal_before_child(branch, failure, caplog):
    captured = []
    result = _dispatch(config_for(branch), failure, _named(), captured)
    assert 'error' in result, result
    assert not captured
    if isinstance(failure, Exception):
        assert str(failure) not in result['error']
        assert str(failure) not in caplog.text


@pytest.mark.parametrize('branch', ['fixed-direct', 'derived-direct', 'provider-only'])
def test_matching_owner_is_accepted(branch):
    captured = []
    result = _dispatch(config_for(branch), _runtime(overrides={'extra_body': {'owned': True}}), _named(), captured)
    assert result == {'ok': True}
    assert (captured[0]['override_requested_provider'] or captured[0]['override_provider']) == 'named-tier'
    assert captured[0]['override_api_key'] == ('fixture-tier-owned-key' if branch == 'fixed-direct' else 'named-tier-owned-key')
    assert captured[0]['override_request_overrides'] == {'extra_body': {'owned': True}}


def test_explicit_different_endpoint_retains_own_key_and_overrides():
    captured = []
    result = _dispatch(_pool_profile(request_overrides={'extra_body': {'tier': True}}),
        _runtime(url=OWNED_URL, overrides={'extra_body': {'provider': True}}), _named(OWNED_URL), captured)
    assert result == {'ok': True}
    assert captured[0]['override_base_url'] == URL
    assert captured[0]['override_api_key'] == 'fixture-tier-owned-key'
    assert captured[0]['override_request_overrides'] == {'extra_body': {'tier': True}}


def test_nonexclusive_resolver_exception_stays_compatible():
    captured = []
    config = {'provider': 'named-tier', 'model': 'legacy-model', 'base_url': URL, 'api_key': 'legacy-fixed'}
    assert _dispatch(config, RuntimeError('unavailable'), _named(), captured) == {'ok': True}
    assert captured[0]['override_api_key'] == 'legacy-fixed'


@pytest.mark.parametrize('provider', [None, 'custom'])
def test_complete_explicit_endpoint_requires_no_ambient_provider_credentials(provider):
    captured = []
    parent = _parent()
    route = {'model': 'fixture-direct-model', 'base_url': URL, 'api_key': 'fixture-explicit-key'}
    if provider is not None:
        route['provider'] = provider
    config = {'model_pool': {'standard': route}}
    with patch('tools.delegate_tool._load_config', return_value=config), patch(
        'tools.delegate_tool._build_child_preserving_parent_tools', side_effect=_fake_child(parent, captured)
    ), patch('tools.delegate_tool._run_batch', return_value=json.dumps({'ok': True})), patch(
        'hermes_cli.runtime_provider.resolve_runtime_provider', side_effect=AssertionError('no ambient resolution')
    ) as resolver:
        result = json.loads(delegate_task(goal='complete explicit direct route', parent_agent=parent))
    resolver.assert_not_called()
    assert result == {'ok': True}, result
    assert captured[0]['override_api_key'] == 'fixture-explicit-key'
    assert captured[0]['override_base_url'] == URL


@pytest.mark.parametrize('second', [None, {}, [], 'invalid-runtime', 7, {'provider': 'custom'}, _runtime(key='wrong-owned-key'), RuntimeError('second-resolution-failed'), ValueError('invalid-resolver'), TypeError('resolver-type-error'), OSError('resolver-io-error')], ids=['none','empty','list','string','int','incomplete','wrong-owner','runtime-error','value-error','type-error','os-error'])
def test_derived_direct_second_resolution_refuses(second):
    captured = []
    parent = _parent()
    with patch('tools.delegate_tool._load_config', return_value=config_for('derived-direct')), patch(
        'hermes_cli.runtime_provider.resolve_runtime_provider', side_effect=[_runtime(), second]
    ), patch('hermes_cli.runtime_provider_custom._get_named_custom_provider', return_value=_named()), patch(
        'tools.delegate_tool._build_child_preserving_parent_tools', side_effect=_fake_child(parent, captured)
    ), patch('tools.delegate_tool._run_batch', return_value=json.dumps({'ok': True})):
        result = json.loads(delegate_task(goal='finite failure probe', parent_agent=parent))
    assert 'error' in result
    assert not captured


@pytest.mark.parametrize('with_url', [False, True])
@pytest.mark.parametrize('failure', [None, {}, [], 'invalid-runtime', 7,
    {'provider': 'google'}, {'provider': 'openrouter', 'base_url': URL, 'api_key': 'wrong'},
    RuntimeError('unavailable'), ValueError('invalid'), TypeError('wrong-type'), OSError('io')],
    ids=['none','empty','list','string','int','incomplete','wrong-owner','runtime-error','value-error','type-error','os-error'])
def test_native_sdk_resolver_branch_refuses_incomplete_and_wrong_owner(with_url, failure):
    config = {'model_pool': {'standard': {'model': 'fixture-native', 'provider': 'google'}}}
    if with_url:
        config['model_pool']['standard']['base_url'] = URL
    captured=[]
    result=_dispatch(config,failure,None,captured)
    assert 'error' in result, result
    assert not captured


@pytest.mark.parametrize(("configured_provider", "resolved_provider"), [("google", "gemini"), ("gemini", "google")])
def test_native_sdk_registered_aliases_share_canonical_identity(configured_provider, resolved_provider):
    config = {'model_pool': {'standard': {'model': 'fixture-native', 'provider': configured_provider}}}
    captured = []
    runtime = {'provider': resolved_provider, 'base_url': URL, 'api_key': 'fixture-owned', 'api_mode': 'google_genai', 'request_overrides': {}}
    assert _dispatch(config, runtime, None, captured) == {'ok': True}
    assert captured[0]['override_provider'] == resolved_provider


@pytest.mark.parametrize('with_url', [False, True])
def test_native_sdk_resolver_matching_owner_is_preserved(with_url):
    config = {'model_pool': {'standard': {'model': 'fixture-native', 'provider': 'google'}}}
    if with_url:
        config['model_pool']['standard']['base_url'] = URL
    captured=[]
    runtime={'provider':'google','api_key':'fixture-owned','api_mode':'google_genai','request_overrides':{}}
    if with_url:
        runtime['base_url']=URL
    assert _dispatch(config,runtime,None,captured)=={'ok':True}
    assert captured[0]['override_provider']=='google'
    assert captured[0]['override_api_key']=='fixture-owned'
    assert captured[0]['override_acp_command'] is None


@pytest.mark.parametrize('branch', ['fixed-direct','derived-direct','provider-only'])
@pytest.mark.parametrize('failure', [RuntimeError('identity lookup unavailable'), ValueError('identity invalid'), TypeError('identity type'), OSError('identity IO')], ids=['runtime-error','value-error','type-error','os-error'])
def test_identity_resolver_revalidation_errors_return_typed_refusal(branch,failure,caplog):
    parent=_parent()
    captured=[]
    with patch('tools.delegate_tool._load_config', return_value=config_for(branch)), patch(
        'hermes_cli.runtime_provider.resolve_runtime_provider',return_value=_runtime()), patch(
        'hermes_cli.runtime_provider_custom._get_named_custom_provider', side_effect=failure), patch(
        'tools.delegate_tool._build_child_preserving_parent_tools',side_effect=_fake_child(parent,captured)), patch(
        'tools.delegate_tool._run_batch',return_value=json.dumps({'ok':True})):
        result=json.loads(delegate_task(goal='identity revalidation probe',parent_agent=parent))
    assert 'error' in result
    assert not captured
    assert str(failure) not in result['error']
    assert str(failure) not in caplog.text


@pytest.mark.parametrize('with_url', [False, True])
def test_bedrock_sdk_auth_marker_is_not_a_static_api_key_requirement(with_url):
    route = {'provider': 'bedrock', 'model': 'fixture-native'}
    runtime = {'provider': 'bedrock', 'api_key': 'aws-sdk', 'api_mode': 'bedrock_converse'}
    if with_url:
        route['base_url'] = runtime['base_url'] = URL
    captured = []
    assert _dispatch({'model_pool': {'standard': route}}, runtime, None, captured) == {'ok': True}
    assert captured[0]['override_api_key'] == 'aws-sdk'
    assert captured[0]['override_provider'] == 'bedrock'
