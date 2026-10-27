"""Bind a provider route onto a live agent in place, and put the primary route back.

``bind_route_entry`` is the fallback chain's swap (client, pool, compressor, reasoning,
extra_body) shared with per-turn routes such as the voice-chat model; ``reinstall_primary_runtime``
is the inverse that ``restore_primary_runtime`` runs once its gates pass.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from utils import base_url_host_matches

logger = logging.getLogger(__name__)


def bind_route_entry(agent: Any, entry: dict[str, Any], provider: str, model: str, *, publish_compression: bool = True, accept_resolved=None) -> Optional[tuple[str, str]]:
    """Swap ``agent`` onto ``entry`` (a fallback-chain-shaped dict). Returns ``(old_model,
    old_provider)``, or None when the provider has no usable client. Sets ``_fallback_activated``
    so the next turn's ``restore_primary_runtime`` reverts it; raises on a failed swap."""
    from agent.auxiliary_client import resolve_provider_client
    from agent.chat_completion_helpers import (
        _fallback_api_mode_hint, _fallback_api_mode_resolved, _rebind_fallback_credential_pool,
        _reresolve_fallback_reasoning_config, _rescope_fallback_extra_body, _update_fallback_context_compressor,
    )
    from hermes_cli.fallback_config import resolve_entry_api_key
    # Pass the entry's base_url/api_key so custom endpoints (Ollama Cloud) resolve instead
    # of falling through to OpenRouter defaults.
    base_url_hint = (entry.get("base_url") or "").strip() or None
    api_key_hint = resolve_entry_api_key(entry)
    api_mode_explicit, api_mode = _fallback_api_mode_hint(entry, provider, base_url_hint)
    # Ollama Cloud: OLLAMA_API_KEY from env when the entry has no key. Host match, not
    # substring — GHSA-76xc-57q6-vm5m.
    if base_url_hint and base_url_host_matches(base_url_hint, "ollama.com") and not api_key_hint:
        from agent.secret_scope import get_secret
        api_key_hint = get_secret("OLLAMA_API_KEY") or None
    # raw_codex=True: the main agent needs direct responses.stream() access for Codex providers.
    client, _resolved_model = resolve_provider_client(
        provider, model=model, raw_codex=True, explicit_base_url=base_url_hint, explicit_api_key=api_key_hint,
        api_mode=api_mode)
    if client is None:
        return None
    if provider == "moa":
        # A MoA entry means the preset itself, exactly like ``provider: moa`` in config. The client
        # is the preset's aggregator and only proves the preset resolves; bind the facade with the
        # same pins every other MoA build site uses (#112525, #112623).
        base_url, api_mode = "moa://local", "chat_completions"
    else:
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider
            model = normalize_model_for_provider(model, provider)
        except Exception as norm_err:  # health: allow BLE001 -- moved verbatim from try_activate_fallback; a bad catalog entry keeps the raw id
            logger.warning("Could not normalize fallback model %r for provider %r: %s", model, provider, norm_err)
        base_url = str(client.base_url)
        from hermes_cli.providers import is_actual_route
        if is_actual_route(provider, base_url):
            api_mode = "chat_completions"
        elif not api_mode_explicit and api_mode == "chat_completions":
            api_mode = _fallback_api_mode_resolved(agent, provider, model, base_url)

    if accept_resolved is not None and not accept_resolved(client, model, base_url):
        if client is not getattr(agent, "client", None) and client is not getattr(agent, "_anthropic_client", None):
            from agent.auxiliary_client import _close_cached_client
            _close_cached_client(client)
        return None

    old_model, old_provider, old_base_url = agent.model, agent.provider, agent.base_url
    from agent.agent_runtime_helpers import _copy_request_overrides
    if not getattr(agent, "_fallback_activated", False):
        primary_runtime = getattr(agent, "_primary_runtime", None)
        if isinstance(primary_runtime, dict):
            primary_runtime["request_overrides"] = _copy_request_overrides(
                getattr(agent, "request_overrides", {}) or {}
            )
    # Clear the per-config context_length override so the new model's own context window is
    # resolved instead of the previous model's stale value (#22387).
    agent._config_context_length = None
    agent.model, agent.provider, agent.requested_provider = model, provider, provider
    agent.base_url, agent.api_mode = base_url, api_mode
    # reasoning_content echo opt-in travels with the active provider; restore_primary_runtime reverts it.
    agent._reasoning_echo_flag = bool(entry.get("reasoning_echo", False))
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    from agent.turn_recovery_codex import reset_codex_reasoning_replay
    reset_codex_reasoning_replay(agent)
    agent._fallback_activated = True

    _rebind_fallback_credential_pool(agent, provider, model)
    if provider == "moa":
        from agent.moa_loop import bind_moa_runtime
        bind_moa_runtime(agent, model)
    else:
        from agent.client_lifecycle import _swap_fallback_clients
        _swap_fallback_clients(agent, client, provider, model, base_url, api_mode)

    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
        provider=provider, base_url=base_url, api_mode=api_mode, model=model)
    agent._ensure_lmstudio_runtime_loaded()  # LM Studio: preload before probing context length
    if publish_compression:
        _update_fallback_context_compressor(agent)
    _reresolve_fallback_reasoning_config(agent)
    _rescope_fallback_extra_body(agent, old_model, old_provider, old_base_url)
    return old_model, old_provider


def reinstall_runtime_snapshot(agent: Any, rt: dict[str, Any]) -> None:
    """Put ``agent`` on the runtime recorded in ``rt`` (a ``_primary_runtime``-shaped snapshot):
    identity, client, caching flags, compressor, reasoning and prompt identity. Credential pool
    and fallback bookkeeping are the caller's. Raises on failure."""
    from agent.agent_runtime_helpers import (
        _apply_primary_runtime_fields, _rebuild_primary_client, _restore_runtime_capabilities,
    )
    _apply_primary_runtime_fields(agent, rt)
    from agent.turn_recovery_codex import reset_codex_reasoning_replay
    reset_codex_reasoning_replay(agent)
    _restore_runtime_capabilities(agent, rt)
    agent._use_prompt_caching = rt["use_prompt_caching"]
    # Default to native layout for snapshots predating the native-vs-proxy split.
    agent._use_native_cache_layout = rt.get(
        "use_native_cache_layout",
        agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
    )
    # An operator cache disable (_cache_disabled) must survive snapshot restoration.
    if getattr(agent, "_cache_disabled", False):
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
    _rebuild_primary_client(agent, rt, reason="restore_primary")
    agent.context_compressor.update_model(
        model=rt["compressor_model"], context_length=rt["compressor_context_length"],
        base_url=rt["compressor_base_url"], api_key=rt["compressor_api_key"],
        provider=rt["compressor_provider"], api_mode=rt.get("compressor_api_mode", ""),
    )
    # Same rule as fallback activation: refresh an existing verdict only; never-probed sessions stay lazy.
    if getattr(agent, "_compression_feasibility_checked", False) is True:
        from agent.conversation_compression import revalidate_compression_feasibility
        revalidate_compression_feasibility(agent)
    # Older snapshots have no reasoning_config; keep the current value.
    saved_reasoning = rt.get("reasoning_config")
    if saved_reasoning is not None:
        agent.reasoning_config = dict(saved_reasoning)
    # Reset the stale-call circuit breaker: its streak measured the route being left.
    from agent.chat_completion_helpers import _reset_stale_streak, rewrite_prompt_model_identity
    _reset_stale_streak(agent)
    # Undo the identity rewrite so the prompt is byte-identical to the stored copy again
    # (prefix cache match).
    rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])


def _primary_pool_owner(agent, primary_provider):
    """The primary's pool identity, or None when its pinned key excludes pools."""
    if getattr(agent, "_delegation_fixed_api_key", False):
        return None
    if primary_provider == "custom":
        return (agent._primary_runtime or {}).get("requested_provider") or primary_provider
    return primary_provider


def _ensure_primary_pool(agent, matches_primary, load_primary_pool, prefetched_pool, prefetched):
    """Attach the primary pool when fallback left a different provider's pool."""
    pool = getattr(agent, "_credential_pool", None)
    pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
    if pool is not None and pool_provider and not matches_primary(pool):
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        try:
            # Reuse the pool the reset-aware gate already loaded (one auth.json read).
            agent._credential_pool = prefetched_pool if prefetched else load_primary_pool()
        except Exception as exc:
            logger.warning(
                "Restore could not reload primary credential pool: %s", exc
            )
        pool = getattr(agent, "_credential_pool", None)
    return pool


def _owned_restore_entry(agent, primary_model):
    """The gate's admitted entry if that id is still eligible, else None.

    Does not clear the handoff and does not call select()."""
    owned_id = getattr(agent, "_restore_selected_entry_id", None)
    if owned_id is None:
        return None
    pool = getattr(agent, "_credential_pool", None)
    return pool.reclaim(owned_id, model=primary_model or None) if pool is not None else None


def _rebind_primary_credential_pool(agent, primary_provider, primary_model, matches_primary, load_primary_pool, prefetched_pool, prefetched):
    """Prepare the admitted row; None means no pool route, False means refusal."""
    if _primary_pool_owner(agent, primary_provider) is None:
        agent._credential_pool = None
        agent._credential_pool_entry_id = None
        return None  # The snapshot's pinned tier key owns the primary route.
    pool = _ensure_primary_pool(agent, matches_primary, load_primary_pool, prefetched_pool, prefetched)
    agent._credential_pool_entry_id = None
    if getattr(agent, "_restore_selected_entry_id", None) is not None:
        entry = _owned_restore_entry(agent, primary_model)
        if entry is None:
            return False
    else:
        entry = pool.select(model=primary_model or None) if pool is not None else None
        if entry is None:
            return None  # No pool credential: retain the configured snapshot route.
        agent._restore_selected_entry_id = entry.id
    return entry if matches_primary(entry) else False


def reinstall_primary_runtime(
    agent: Any, rt: dict[str, Any], primary_provider: str, primary_model: str, matches_primary,
    load_primary_pool, prefetched_pool=None, prefetched: bool = False,
) -> bool:
    """Publish the one admitted primary credential after staged client construction."""
    import contextlib
    import copy
    from agent.agent_runtime_helpers import (
        _apply_primary_runtime_fields, _rebuild_primary_client, _restore_runtime_capabilities,
    )
    # Stage runtime writes, but bind retained client callbacks to the live session.
    staged = copy.copy(agent)
    staged._client_owner = agent
    staged.client = staged._anthropic_client = None
    staged._transport_cache = {}
    prepared = []
    try:
        selected = _rebind_primary_credential_pool(
            staged, primary_provider, primary_model, matches_primary, load_primary_pool, prefetched_pool, prefetched
        )
        if selected is False:
            return False
        build_rt = dict(rt)
        if selected is not None:
            build_rt["api_key"] = selected.runtime_api_key
            build_rt["client_kwargs"] = {**rt["client_kwargs"], "api_key": selected.runtime_api_key}
        _apply_primary_runtime_fields(staged, build_rt)
        _rebuild_primary_client(staged, build_rt, reason="restore_primary")
        prepared.extend((staged.client, staged._anthropic_client))
        if selected is not None:
            selected = _owned_restore_entry(staged, primary_model)
            if selected is None or not matches_primary(selected) or staged._swap_credential(selected) is False:
                return False
        # Reclaim may refresh outside the pool lock. Fence its returned live object
        # under that lock through publication; a concurrent replace/bench invalidates it.
        pool = getattr(staged, "_credential_pool", None)
        owned = _owned_restore_entry(staged, primary_model)
        with agent._openai_client_lock(), pool._lock if selected is not None and pool is not None else contextlib.nullcontext():
            if selected is not None and (
                pool is None or owned is not selected or pool.current() is not owned
                or owned.runtime_api_key != staged.api_key
            ):
                return False
            _apply_primary_runtime_fields(agent, {
                **rt, "api_key": staged.api_key, "base_url": staged.base_url,
                "api_mode": staged.api_mode, "client_kwargs": staged._client_kwargs,
            })
            for name in ("client", "_anthropic_client", "_anthropic_api_key", "_anthropic_base_url",
                         "_is_anthropic_oauth", "_bedrock_region", "_bedrock_guardrail_config",
                         "_credential_pool", "_credential_pool_entry_id"):
                if hasattr(staged, name):
                    setattr(agent, name, getattr(staged, name))
        _restore_runtime_capabilities(agent, rt)
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout for snapshots predating the native-vs-proxy split.
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )
        # An operator cache disable (_cache_disabled) must survive snapshot restoration.
        if getattr(agent, "_cache_disabled", False):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False
        agent.context_compressor.update_model(
            model=rt["compressor_model"], context_length=rt["compressor_context_length"],
            base_url=agent.base_url, api_key=agent.api_key,
            provider=rt["compressor_provider"], api_mode=rt.get("compressor_api_mode", ""),
        )
        # Same rule as fallback activation: refresh an existing verdict only; never-probed sessions stay lazy.
        if getattr(agent, "_compression_feasibility_checked", False) is True:
            from agent.conversation_compression import revalidate_compression_feasibility
            revalidate_compression_feasibility(agent)
        # Older snapshots have no reasoning_config; keep the current value.
        saved_reasoning = rt.get("reasoning_config")
        if saved_reasoning is not None:
            agent.reasoning_config = dict(saved_reasoning)
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._rate_limit_backoff_count = 0
        # Reset the stale-call circuit breaker: its streak measured the fallback provider.
        from agent.chat_completion_helpers import _reset_stale_streak, rewrite_prompt_model_identity
        _reset_stale_streak(agent)
        # Undo the fallback's identity rewrite so the prompt is byte-identical to the stored copy
        # again (prefix cache match).
        rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])
        from agent.turn_recovery_codex import reset_codex_reasoning_replay
        reset_codex_reasoning_replay(agent)
        return True
    finally:
        agent._restore_selected_entry_id = None
        # Only unpublished clients are ours to close. Never close fallback borrowers.
        for client in (*prepared, staged.client, staged._anthropic_client):
            if client is not None and client is not getattr(agent, "client", None) and client is not getattr(agent, "_anthropic_client", None):
                with contextlib.suppress(Exception):
                    client.close()
