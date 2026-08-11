"""Pure completion identities shared by persistence and the process registry."""

import hashlib
import json
import math
from typing import Optional

_COMPLETION_DELIVERY_PRIVATE_PREFIX = "_completion_delivery_"


def completion_durable_payload(evt: dict) -> dict:
    """Return the public ordinary-completion projection used for persistence."""
    return {
        key: value
        for key, value in evt.items()
        if not key.startswith(_COMPLETION_DELIVERY_PRIVATE_PREFIX)
    }


def completion_authority_fingerprint(
    evt: dict, *, allow_nonfinite: bool = False
) -> Optional[str]:
    """Hash the current terminal envelope without transport-only routing fields."""
    if evt.get("type", "completion") != "completion":
        return None
    payload = completion_durable_payload(evt)
    for key in (
        "restored",
        "platform",
        "chat_type",
        "chat_id",
        "thread_id",
        "user_id",
        "user_name",
        "message_id",
        "completion_ack_recorded_at",
    ):
        payload.pop(key, None)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=allow_nonfinite,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return f"completion-event-v1:{hashlib.sha256(encoded).hexdigest()}"


def canonical_completion_fingerprint(evt: dict) -> str:
    """Hash the fixed public fields that distinguish canonical terminal envelopes."""
    encoded = json.dumps(
        (
            evt["command"],
            evt["output"],
            evt["exit_code"],
            evt["completion_reason"],
            evt["termination_source"],
            bool(evt.get("delegated_child", False)),
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"completion-canonical-v1:{hashlib.sha256(encoded).hexdigest()}"


def completion_event_fingerprint(evt: dict) -> Optional[str]:
    """Return a replay-stable identity for one sanitized malformed envelope."""
    if (
        evt.get("type", "completion") != "completion"
        or completion_core_identity(evt) is None
    ):
        return None
    payload = completion_durable_payload(evt)
    payload.pop("restored", None)
    payload.pop("completion_ack_recorded_at", None)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return f"completion-event-v1:{hashlib.sha256(encoded).hexdigest()}"


def completion_core_identity(evt: dict) -> Optional[tuple]:
    """Return the stable producer tuple without granting public authority."""
    session_id = evt.get("session_id")
    session_key = evt.get("session_key", "")
    started_at = evt.get("started_at")
    if (
        evt.get("type", "completion") != "completion"
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(session_key, str)
        or isinstance(started_at, bool)
        or not isinstance(started_at, (int, float))
        or (isinstance(started_at, float) and not math.isfinite(started_at))
        or started_at <= 0
    ):
        return None
    return "completion", session_id, started_at, session_key


def completion_identity(evt: dict) -> Optional[tuple]:
    """Return tuple authority only for a fully canonical success envelope."""
    identity = completion_core_identity(evt)
    if (
        identity is None
        or evt.get("type") != "completion"
        or "session_key" not in evt
        or evt.get("termination_source") != ""
        or type(evt.get("exit_code")) is not int
        or evt["exit_code"] != 0
        or evt.get("completion_reason") != "exited"
        or not isinstance(evt.get("command"), str)
        or not isinstance(evt.get("output"), str)
        or any(
            evt.get(key)
            for key in (
                "error",
                "stderr",
                "error_message",
                "exception",
                "warning",
                "safety_alert",
                "timed_out",
                "cancelled",
                "canceled",
                "failed",
                "lost",
            )
        )
    ):
        return None
    return identity


def completion_durable_identity(evt: dict) -> Optional[tuple]:
    """Return durable authority for a public persisted completion envelope."""
    if evt.get("type") == "async_delegation":
        delegation_id = evt.get("delegation_id")
        if isinstance(delegation_id, str) and delegation_id:
            return "async-delegation", delegation_id
        return None
    identity = completion_identity(evt)
    if identity is not None:
        return *identity, canonical_completion_fingerprint(evt)
    if completion_core_identity(evt) is None:
        return None
    fingerprint = completion_authority_fingerprint(evt)
    return ("completion-event", fingerprint) if fingerprint is not None else None
