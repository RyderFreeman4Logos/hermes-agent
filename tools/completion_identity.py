"""Pure completion identities shared by persistence and the process registry."""

import hashlib
import json
import math
from typing import Optional

_COMPLETION_DELIVERY_PRIVATE_PREFIX = "_completion_delivery_"


class CompletionDeliveryToken(str):
    """Process-local authority whose type cannot survive JSON persistence."""


class CompletionDeliveryBinding(str):
    """Exact private subtype binding a local token to its producer event."""


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


def bound_completion_delivery_metadata(evt: dict) -> Optional[tuple]:
    """Decode exact local authority bound to its producer and original envelope."""
    token = evt.get("_completion_delivery_token")
    binding = evt.get("_completion_delivery_binding")
    if (
        type(token) is not CompletionDeliveryToken
        or type(binding) is not CompletionDeliveryBinding
        or not token
        or not binding
    ):
        return None
    try:
        metadata = json.loads(str(binding))
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(metadata, list)
        or len(metadata) not in {3, 5}
        or not isinstance(metadata[1], str)
        or token.rpartition("|")[2] != metadata[1]
    ):
        return None
    if len(metadata) == 3:
        metadata.extend((None, None))
    elif (
        not isinstance(metadata[2], str)
        or not metadata[2].startswith("completion-event-v1:")
        or not isinstance(metadata[3], str)
        or not metadata[3].startswith("completion-event-v1:")
        or not isinstance(metadata[4], list)
    ):
        return None
    return token, metadata[0], metadata[2], metadata[3], metadata[4]


def bound_completion_projection(evt: dict) -> Optional[tuple]:
    """Recover producer identity only from a binding valid for this projection."""
    metadata = bound_completion_delivery_metadata(evt)
    current_core = completion_core_identity(evt)
    current_fingerprint = completion_authority_fingerprint(evt)
    if (
        metadata is None
        or current_core is None
        or metadata[1] != list(current_core)
        or metadata[3] != current_fingerprint
        or not metadata[4]
        or any(
            isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in metadata[4]
        )
    ):
        return None
    return metadata[0], metadata[2], tuple(metadata[4])


def bound_completion_delivery_token(evt: dict) -> Optional[CompletionDeliveryToken]:
    """Return an exact local token only when its producer binding still matches."""
    metadata = bound_completion_delivery_metadata(evt)
    return metadata[0] if metadata is not None else None


def bound_completion_event_fingerprint(evt: dict) -> Optional[str]:
    """Return the original envelope fingerprint from valid local authority."""
    metadata = bound_completion_delivery_metadata(evt)
    current_core = completion_core_identity(evt)
    current_fingerprint = completion_authority_fingerprint(evt)
    fingerprint = metadata[2] if metadata is not None else None
    if (
        current_core is None
        or current_fingerprint is None
        or metadata is None
        or metadata[1] != list(current_core)
        or fingerprint != current_fingerprint
    ):
        return None
    if isinstance(fingerprint, str) and fingerprint.startswith("completion-event-v1:"):
        return fingerprint
    return None


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
    """Return canonical, bound-producer, or sanitized-envelope authority."""
    if evt.get("type") == "async_delegation":
        delegation_id = evt.get("delegation_id")
        if isinstance(delegation_id, str) and delegation_id:
            return "async-delegation", delegation_id
        return None
    projection = bound_completion_projection(evt)
    if projection is not None:
        return projection[2]
    identity = completion_identity(evt)
    if identity is not None:
        return *identity, canonical_completion_fingerprint(evt)
    fingerprint = bound_completion_event_fingerprint(evt)
    if fingerprint is None and completion_core_identity(evt) is not None:
        fingerprint = completion_authority_fingerprint(evt)
    return ("completion-event", fingerprint) if fingerprint is not None else None
