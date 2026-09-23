#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Callable, List, Optional

from tools import write_approval as wa


logger = logging.getLogger(__name__)


def load_authoritative_memory_manager(
    *, session_id: str, platform: str = "cli", identity: Optional[dict] = None,
):
    """Load the active profile's provider for a cold ``/memory approve`` command."""
    try:
        from agent.memory_manager import MemoryManager
        from hermes_cli.config import load_config_readonly
        from hermes_constants import get_hermes_home
        from plugins.memory import load_memory_provider
        from tools.memory_tool import get_builtin_memory_config

        memory_config = get_builtin_memory_config(load_config_readonly())
        provider_name = str(memory_config.get("provider") or "").strip()
        provider = (
            load_memory_provider(provider_name, register_skills=False)
            if provider_name else None
        )
        if provider is None or not provider.is_available():
            return None

        manager = MemoryManager(provider_mode="authoritative")
        manager.add_provider(provider)
        init_kwargs = {
            "platform": platform or "cli",
            "hermes_home": str(get_hermes_home()),
            "agent_context": "primary",
        }
        allowed_identity = {
            "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type",
            "thread_id", "gateway_session_key", "session_title",
        }
        init_kwargs.update({
            key: value for key, value in (identity or {}).items()
            if key in allowed_identity and value
        })
        try:
            from hermes_cli.profiles import get_active_profile_name
            init_kwargs.update(agent_identity=get_active_profile_name(), agent_workspace="hermes")
        except Exception:
            pass
        manager.initialize_all(str(session_id or ""), **init_kwargs)
        return manager
    except Exception as exc:
        logger.warning("Memory provider plugin init failed during approval: %s", exc)
        return None


@contextmanager
def _approval_memory_manager(current, factory, *, required: bool):
    """Yield the live manager, or a command-scoped manager when approval needs one."""
    manager = current
    owned = False
    if manager is None and required and factory is not None:
        manager = factory()
        owned = manager is not None
    try:
        yield manager
    finally:
        if owned:
            manager.shutdown_all()


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    lines = [f"Pending {subsystem} writes ({len(records)}):"]
    for r in records:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
    lines.append("")
    lines.append(f"Apply: /{subsystem} approve <id>   Reject: /{subsystem} reject <id>")
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    return "\n".join(lines)


def handle_pending_subcommand(
    subsystem: str, args: List[str], *, memory_store=None, memory_manager=None,
    memory_manager_factory: Optional[Callable] = None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn`` persists the write_approval boolean. Returns text for the user,
    or None when the args are not a write-approval subcommand so the caller falls through to its
    other handling (e.g. /skills search).
    """
    if not args:
        return f"{_fmt_state(subsystem)}\n\n" + _fmt_pending_list(subsystem)
    sub, rest = args[0].lower(), args[1:]
    if sub == "pending":
        return _fmt_pending_list(subsystem)
    if sub in {"approve", "apply"}:
        return _approve(
            subsystem, rest, memory_store, memory_manager,
            memory_manager_factory=memory_manager_factory,
        )
    if sub in {"reject", "deny", "drop"}:
        return _reject(subsystem, rest)
    if sub == "diff" and subsystem == wa.SKILLS:
        return _diff(rest)
    if sub in {"approval", "mode"}:  # 'mode' kept as a back-compat alias
        return _set_approval(subsystem, rest, set_mode_fn)
    return None  # not ours — caller handles


def _usage(subsystem: str) -> str:
    return f"Usage: /{subsystem} approve|reject <id>  (or 'all')"


def _approve(
    subsystem: str, rest: List[str], memory_store, memory_manager=None,
    *, memory_manager_factory=None,
) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    if target.lower() == "all":
        targets = list(records)
    else:
        rec = wa.get_pending(subsystem, target)
        if not rec:
            return f"No pending {subsystem} write with id '{target}'."
        targets = [rec]

    needs_authoritative = (
        subsystem == wa.MEMORY
        and any(rec.get("payload", {}).get("memory_provider_mode") == "authoritative"
                for rec in targets)
    )
    applied, failed, overwritten = 0, [], []
    with _approval_memory_manager(
        memory_manager, memory_manager_factory, required=needs_authoritative,
    ) as resolved_manager:
        for rec in targets:
            ok, msg, result = _apply_one(subsystem, rec, memory_store, resolved_manager)
            if ok:
                wa.discard_pending(subsystem, rec["id"])
                applied += 1
                overwritten.extend(f"  {rec['id']}: {text}" for text in _replaced_entries(result))
            else:
                failed.append(f"{rec['id']}: {msg}")

    out = [f"Approved {applied} {subsystem} write(s)."]
    if overwritten:
        # A memory 'replace' overwrites the WHOLE matched entry (#117952); the approver
        # is the last person who can notice a clause went missing, so show what was lost.
        out.append("Overwrote entire entry (re-add anything you still need):")
        out.extend(overwritten)
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    return "\n".join(out)


def _replaced_entries(result: dict) -> List[str]:
    """Full text of every entry a memory replace overwrote, single-op or batch shape."""
    single = result.get("replaced_entry")
    batch = result.get("replaced_entries") or {}
    return ([single] if single else []) + [batch[k] for k in sorted(batch, key=int)]


def _apply_one(subsystem: str, rec, memory_store, memory_manager=None):
    """``(ok, error, result)`` — *result* is the applier's full payload (empty on exceptions)."""
    payload = rec.get("payload", {})
    try:
        if subsystem == wa.MEMORY:
            if payload.get("memory_provider_mode") == "authoritative":
                if memory_manager is None:
                    return False, "authoritative memory provider unavailable", {}
                provider_payload = dict(payload)
                provider_payload.pop("memory_provider_mode", None)
                result = json.loads(memory_manager.authoritative_memory_write(provider_payload))
                return bool(result.get("success")), result.get("error", ""), result
            if memory_store is None:
                return False, "memory store unavailable", {}
            from tools.memory_tool import apply_memory_pending
            result = apply_memory_pending(payload, memory_store)
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), result.get("error", ""), result
    except Exception as e:
        return False, str(e), {}


def _reject(subsystem: str, rest: List[str]) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    if target.lower() == "all":
        n = sum(1 for rec in wa.list_pending(subsystem) if wa.discard_pending(subsystem, rec["id"]))
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.discard_pending(subsystem, target):
        return f"Rejected pending {subsystem} write '{target}'."
    return f"No pending {subsystem} write with id '{target}'."


def _diff(rest: List[str]) -> str:
    if not rest:
        return "Usage: /skills diff <id>"
    rec = wa.get_pending(wa.SKILLS, rest[0])
    if not rec:
        return f"No pending skill write with id '{rest[0]}'."
    return f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n\n" + wa.skill_pending_diff(rec)


_APPROVAL_VALUES = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), True),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), False)}


def _set_approval(subsystem: str, rest: List[str], set_mode_fn) -> str:
    """Turn the approval gate on/off for a subsystem."""
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off>")
    arg = rest[0].strip().lower()
    enabled = _APPROVAL_VALUES.get(arg)
    if enabled is None:
        return f"Invalid value '{arg}'. Use: on or off."
    if set_mode_fn is None:
        val = "true" if enabled else "false"
        return (f"To change the {subsystem} approval gate, run:\n"
                f"  hermes config set {subsystem}.write_approval {val}")
    try:
        set_mode_fn(enabled)
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    return f"{subsystem}.write_approval set to '{'on' if enabled else 'off'}'."
