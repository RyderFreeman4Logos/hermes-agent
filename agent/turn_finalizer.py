"""Post-loop turn finalization for ``run_conversation``.

Extracted from ``agent/conversation_loop.py`` as part of the god-file
decomposition campaign (``~/.hermes/plans/god-file-decomposition.md``, Phase 1
step 4 — the post-loop ``TurnFinalizer`` seam). ``run_conversation``'s tail
(everything after the main tool-calling ``while`` loop) is lifted here verbatim:
budget-exhaustion summary, trajectory save, session persist, turn diagnostics,
response transforms, result-dict assembly, steer drain, and the memory/skill
review trigger.

Behavior-neutral: the body is moved unchanged. All ``agent.*`` side effects fire
exactly as before; only the post-loop *locals* are passed in as keyword args, and
the assembled ``result`` dict is returned to ``run_conversation`` which returns it
to the caller. The function is synchronous with a single return — mirroring the
region it replaces (no awaits, no early returns).

Module ``logger`` is imported lazily inside the body (``from
agent.conversation_loop import logger``) so this module never imports
``agent.conversation_loop`` at import time -> no import cycle, and the log records
keep the exact logger name (``"agent.conversation_loop"``).
"""

from __future__ import annotations

import copy
import errno
import math
import os
import time

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.message_content import flatten_message_text
from agent.message_sanitization import (
    COMPLETION_DELIVERY_INTERRUPTED_CLOSURE,
    COMPLETION_DELIVERY_TOOL_CLOSURE,
    _is_ephemeral_scaffolding,
    close_interrupted_tool_sequence,
    completion_delivery_suffix_has_meaningful_work,
    completion_delivery_suffix_start,
    completion_delivery_transcript_content,
)
from agent.retry_utils import jittered_backoff


_COMPLETION_COMMIT_DEFAULTS = (32, 0.05, 5.0, 120.0)
_COMPLETION_COMMIT_MAX_ATTEMPTS = 1_000
_COMPLETION_COMMIT_MAX_BACKOFF_S = 60.0
_COMPLETION_COMMIT_MAX_PATIENCE_S = 3_600.0


def _get_completion_delivery_commit_config() -> tuple[int, float, float, float]:
    """Read the completion commit retry policy at its use site."""
    try:
        from hermes_cli.config import load_config_readonly

        session = load_config_readonly().get("session") or {}
        retry = session.get("completion_delivery_commit") or {}
        values = (
            float(retry.get("initial_backoff_s", 0.05)),
            float(retry.get("max_backoff_s", 5.0)),
            float(retry.get("patience_s", 120.0)),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("retry durations must be finite")
        # ponytail: one-hour ceiling; raise it only if real SessionDB recovery
        # needs a longer foreground turn budget.
        return (
            min(
                max(1, int(retry.get("max_attempts", 32))),
                _COMPLETION_COMMIT_MAX_ATTEMPTS,
            ),
            min(max(0.0, values[0]), _COMPLETION_COMMIT_MAX_BACKOFF_S),
            min(max(0.0, values[1]), _COMPLETION_COMMIT_MAX_BACKOFF_S),
            min(max(0.0, values[2]), _COMPLETION_COMMIT_MAX_PATIENCE_S),
        )
    except Exception:
        return _COMPLETION_COMMIT_DEFAULTS


def _completion_delivery_commit_failure_kind(error) -> str:
    try:
        from hermes_state import is_disk_full_error

        if is_disk_full_error(error):
            return "storage_full"
    except Exception:
        pass
    text = str(error or "").lower()
    if (
        isinstance(error, PermissionError)
        or getattr(error, "errno", None) in {errno.EPERM, errno.EACCES, errno.EROFS}
        or any(
            marker in text
            for marker in (
                "permission denied",
                "operation not permitted",
                "not writable",
                "read-only",
                "readonly",
            )
        )
    ):
        return "permission"
    if any(
        marker in text
        for marker in (
            "compare-and-set",
            "compare and set",
            "does not match durable transcript tail",
        )
    ):
        return "cas_conflict"
    if any(
        marker in text
        for marker in (
            "locked",
            "busy",
            "timed out waiting for compression",
        )
    ):
        return "lock_busy"
    return "unknown"


def completion_delivery_commit_error_message(agent, *, prior: bool = False) -> str:
    """Describe why a retained completion still cannot become durable."""
    failure = getattr(agent, "_completion_delivery_commit_failure", "unknown")
    details = {
        "lock_busy": (
            "SessionDB is locked or busy; retries for the exact retained suffix "
            "were exhausted"
        ),
        "storage_full": (
            "SessionDB has no space left (ENOSPC/database full); the exact "
            "retained suffix was not published"
        ),
        "permission": (
            "SessionDB has a permission or read-only write failure; the exact "
            "retained suffix was not published"
        ),
        "cas_conflict": (
            "SessionDB completion compare-and-set conflict; the retained suffix "
            "does not match the active durable event"
        ),
        "missing_pending_suffix": (
            "the pending completion suffix is missing and its durable recovery "
            "record could not be written"
        ),
        "missing_active_marker": (
            "the active completion marker is missing and its durable recovery "
            "record could not be written"
        ),
        "unknown": (
            "SessionDB rejected the exact retained suffix for an unclassified "
            "write failure"
        ),
    }
    prefix = (
        "A prior completion delivery is still pending: "
        if prior
        else "Completion delivery is still pending: "
    )
    return (
        prefix
        + details.get(failure, details["unknown"])
        + ". Retry after resolving this SessionDB condition; a process restart "
        "is not required."
    )


def _flush_completion_delivery_with_retry(
    agent, staged, metadata_cas, pending
) -> bool:
    """Settle one pending suffix once under the hot-read count/time budget."""
    attempts = 0
    started = time.monotonic()
    while True:
        max_attempts, _initial, _maximum, patience = (
            _get_completion_delivery_commit_config()
        )
        if attempts and (
            attempts >= max_attempts or time.monotonic() - started >= patience
        ):
            return False

        attempts += 1
        error = None
        persist_lock = getattr(agent, "_session_persist_lock", None)
        persist_lock_acquired = persist_lock is None
        try:
            remaining_patience = max(0.0, patience - (time.monotonic() - started))
            if persist_lock is not None:
                persist_lock_acquired = persist_lock.acquire(
                    timeout=remaining_patience
                )
            if not persist_lock_acquired:
                raise TimeoutError("session persistence lock remained busy")
            # Another finalizer may have published this exact generation while
            # this caller waited. Re-check under the same lock as the append.
            if getattr(agent, "_pending_completion_delivery_suffix", None) is not pending:
                return True
            committed = (
                agent._flush_messages_to_session_db(
                    staged,
                    display_metadata_cas=metadata_cas,
                    patience_s=max(
                        0.0, patience - (time.monotonic() - started)
                    ),
                )
                is True
            )
            if committed:
                agent._pending_completion_delivery_suffix = None
                agent._pending_completion_delivery_display_metadata_cas = None
                agent._completion_delivery_commit_failed = False
        except Exception as exc:
            committed = False
            error = exc
        finally:
            if persist_lock is not None and persist_lock_acquired:
                persist_lock.release()
        if committed:
            agent._completion_delivery_commit_failure = None
            return True

        if error is None:
            error = getattr(agent, "_last_session_db_flush_error", None)
        agent._completion_delivery_commit_failure = (
            _completion_delivery_commit_failure_kind(error)
        )

        try:
            # Re-read after the failed flush: an operator may tune the policy while
            # a long SessionDB attempt is blocked, without restarting the process.
            max_attempts, initial, maximum, patience = (
                _get_completion_delivery_commit_config()
            )
            elapsed = time.monotonic() - started
            if attempts >= max_attempts or elapsed >= patience:
                return False
            remaining = patience - elapsed
            delay = 0.0
            if initial > 0 and maximum > 0:
                delay = min(
                    jittered_backoff(
                        attempts,
                        base_delay=initial,
                        max_delay=maximum,
                        jitter_ratio=0.5,
                    ),
                    maximum,
                    remaining,
                )
            if delay > 0:
                time.sleep(delay)
        except Exception as exc:
            agent._completion_delivery_commit_failure = (
                _completion_delivery_commit_failure_kind(exc)
            )
            return False


def _is_pure_tool_call_tail(msg: dict) -> bool:
    """An assistant row with ``tool_calls`` but no visible text content of its own.

    Such a row satisfies the role check (``tail role == "assistant"``) while
    carrying none of the delivered answer — see the #43849/#44100 invariant
    block in :func:`finalize_turn`. Uses :func:`flatten_message_text` so that
    multimodal (list-type) content is evaluated by its text parts, not just
    its type.
    """
    if not msg.get("tool_calls"):
        return False
    return not flatten_message_text(msg.get("content")).strip()


def _drop_ephemeral_scaffolding(messages) -> None:
    """Remove model-only scaffolding while preserving every real message."""
    messages[:] = [m for m in messages if not _is_ephemeral_scaffolding(m)]


def finalize_completion_delivery_suffix(
    agent,
    messages,
    *,
    final_response,
    failed: bool,
    interrupted: bool,
    commit_tool_intent: bool = False,
) -> str:
    """Publish one completion suffix, using SessionDB as commit authority.

    The marked user row and everything after it were withheld from append-only
    persistence.  A meaningful turn is staged as a hidden durable event plus
    its assistant/tool sequence.  The staged suffix is atomically appended to
    SessionDB before the live list is changed, so JSON/live state can never
    report a commit that cold resume cannot replay. ``api_content`` retains the
    exact prompt the provider saw; ``content`` contains only the canonical event
    text.

    Zero-work no-ops and failures are discarded.  Once a tool/effect or visible
    assistant action exists, failure/interruption commits that audit trail with
    a provider-safe closure instead of making restart retry the side effect.

    Returns ``"none"``, ``"dropped"``, ``"committed"``, or ``"pending"``.
    ``pending`` means the atomic DB publish failed after a retry; the marked
    suffix remains live for a later durability retry and must not be discarded.
    """
    start = completion_delivery_suffix_start(messages)
    if start is None:
        return "none"

    response_text = flatten_message_text(final_response).strip()
    suffix_has_work = completion_delivery_suffix_has_meaningful_work(messages, start)
    meaningful = suffix_has_work or bool(
        not failed
        and not interrupted
        and response_text
        and response_text != "(empty)"
    )
    if commit_tool_intent:
        meaningful = meaningful or any(
            isinstance(row, dict)
            and row.get("role") == "assistant"
            and row.get("tool_calls")
            for row in messages[start + 1:]
        )

    if not meaningful:
        del messages[start:]
        agent._pending_completion_delivery_suffix = None
        agent._pending_completion_delivery_display_metadata_cas = None
        agent._completion_delivery_commit_failed = False
        agent._completion_delivery_commit_failure = None
        agent._persist_user_message_idx = None
        agent._persist_user_message_override = None
        agent._db_flush_scan_prefix = None
        return "dropped"

    # Keep the durable prefix objects by identity and isolate only the pending
    # suffix.  The flush protocol uses those prefix identities/markers to tell
    # already-published compression rows from a turn-start boundary whose DB
    # append failed and still needs to land with the event.
    staged = list(messages[:start]) + copy.deepcopy(messages[start:])
    user = staged[start]
    wire_content = user.get("content")
    provider_content = user.get("api_content")
    if not isinstance(provider_content, str) or not provider_content:
        provider_content = wire_content
    clean_content = completion_delivery_transcript_content(wire_content)
    user["content"] = clean_content
    if isinstance(provider_content, str) and provider_content != clean_content:
        user["api_content"] = provider_content
    user["display_kind"] = "hidden"
    metadata = user.get("display_metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata_cas = None
    if not commit_tool_intent and user.get("_completion_delivery_active"):
        expected_metadata = dict(metadata)
        expected_metadata["completion_delivery_status"] = "effect_started"
    metadata["completion_delivery_status"] = (
        "effect_started"
        if commit_tool_intent
        else "interrupted" if interrupted else "failed" if failed else "complete"
    )
    user["display_metadata"] = metadata
    if not commit_tool_intent and user.get("_completion_delivery_active"):
        metadata_cas = {
            "role": "user",
            "content": clean_content,
            "api_content": (
                provider_content
                if isinstance(provider_content, str)
                and provider_content != clean_content
                else None
            ),
            "display_kind": "hidden",
            "expected_display_metadata": expected_metadata,
            "display_metadata": metadata,
        }
    user.pop("_completion_delivery_synthetic", None)
    # Active is a live finalization marker and therefore a persistence barrier.
    # Keep it off the staged DB copy until the tool intent has actually landed.
    user.pop("_completion_delivery_active", None)

    staged_tail = staged[-1] if staged and isinstance(staged[-1], dict) else None
    if response_text and not failed and not interrupted and (
        staged_tail is None or staged_tail.get("role") != "assistant"
    ):
        staged.append({"role": "assistant", "content": final_response})
    elif (
        response_text
        and not failed
        and not interrupted
        and isinstance(staged_tail, dict)
        and staged_tail.get("content") != final_response
        and _is_pure_tool_call_tail(staged_tail)
    ):
        staged_tail["content"] = final_response

    if staged and isinstance(staged[-1], dict) and staged[-1].get("role") == "tool":
        closure = (
            COMPLETION_DELIVERY_INTERRUPTED_CLOSURE
            if failed or interrupted
            else COMPLETION_DELIVERY_TOOL_CLOSURE
        )
        if close_interrupted_tool_sequence(staged, closure):
            staged[-1]["display_kind"] = "hidden"

    # The TUI stages completion delivery with the provider-facing prompt as
    # ``persist_user_message``.  Override it before the first atomic intent
    # flush: otherwise the generic persistence layer writes that prompt over
    # the canonical event in SQLite while this staged/live copy keeps the
    # canonical text.  Every later durable-tail comparison and final metadata
    # CAS then targets two different ``content`` values.
    previous_persist_idx = getattr(agent, "_persist_user_message_idx", None)
    previous_persist_override = getattr(
        agent, "_persist_user_message_override", None
    )
    agent._persist_user_message_idx = start
    agent._persist_user_message_override = clean_content

    # The normal incremental writer is already atomic for a new suffix. Use its
    # marker protocol on the staged copy; only publish those exact dicts to the
    # live list after SessionDB accepts them. Failed transactions stamp no
    # message markers, so the configured retries cannot duplicate the suffix.
    db_bound = (
        getattr(agent, "_session_db", None) is not None
        and not getattr(agent, "_persist_disabled", False)
    )
    committed = not db_bound
    if db_bound:
        # Install recovery authority before entering any fallible retry/timing
        # code. An unexpected exception can never leave a visible effect free
        # to start a new provider turn without its exact canonical suffix.
        pending = copy.deepcopy(staged[start:])
        agent._pending_completion_delivery_suffix = pending
        agent._pending_completion_delivery_display_metadata_cas = copy.deepcopy(
            metadata_cas
        )
        agent._completion_delivery_commit_failed = True
        committed = _flush_completion_delivery_with_retry(
            agent, staged, metadata_cas, pending
        )
    if not committed:
        if commit_tool_intent:
            agent._persist_user_message_idx = previous_persist_idx
            agent._persist_user_message_override = previous_persist_override
        agent._db_flush_scan_prefix = None
        return "pending"

    if commit_tool_intent:
        staged[start]["_completion_delivery_active"] = True
    messages[:] = staged
    if not db_bound:
        agent._pending_completion_delivery_suffix = None
        agent._pending_completion_delivery_display_metadata_cas = None
        agent._completion_delivery_commit_failed = False
    agent._completion_delivery_commit_failure = None
    # Keep the canonical override after response repair so later incremental
    # suffix flushes cannot restore the API-only instruction.
    # Earlier incremental flushes intentionally stopped at this same dict.
    # Force the marker scanner to revisit it now that its disposition changed.
    if not db_bound:
        agent._db_flush_scan_prefix = None
    return "committed"


def retry_pending_completion_delivery_commit(agent, messages) -> str:
    """Retry the exact canonical suffix retained after a failed DB append."""
    pending = getattr(agent, "_pending_completion_delivery_suffix", None)
    start = completion_delivery_suffix_start(messages)
    if not isinstance(pending, list) or not pending or start is None:
        outcome = (
            "missing_pending_suffix"
            if not isinstance(pending, list) or not pending
            else "missing_active_marker"
        )
        recovery_event = (
            pending[0]
            if isinstance(pending, list) and pending and isinstance(pending[0], dict)
            else messages[start]
            if start is not None and isinstance(messages[start], dict)
            else None
        )
        try:
            from tools.async_delegation import record_completion_delivery_recovery

            recovery_committed = bool(recovery_event) and (
                record_completion_delivery_recovery(
                    str(getattr(agent, "session_id", "") or ""),
                    outcome,
                    recovery_event,
                )
            )
        except Exception:
            recovery_committed = False
        if not recovery_committed:
            agent._completion_delivery_commit_failure = outcome
            return "pending"
        if start is not None:
            del messages[start:]
        agent._pending_completion_delivery_suffix = None
        agent._pending_completion_delivery_display_metadata_cas = None
        agent._completion_delivery_commit_failed = False
        agent._completion_delivery_commit_failure = None
        agent._persist_user_message_idx = None
        agent._persist_user_message_override = None
        agent._db_flush_scan_prefix = None
        try:
            agent._save_session_log(messages)
        except Exception:
            pass
        return outcome
    staged = list(messages[:start]) + copy.deepcopy(pending)
    metadata_cas = copy.deepcopy(getattr(
        agent, "_pending_completion_delivery_display_metadata_cas", None
    ))
    committed = _flush_completion_delivery_with_retry(
        agent, staged, metadata_cas, pending
    )
    if not committed:
        return "pending"
    messages[:] = staged
    agent._completion_delivery_commit_failure = None
    try:
        agent._save_session_log(messages)
    except Exception:
        pass
    return "committed"


def finalize_turn(
    agent,
    *,
    final_response,
    api_call_count,
    interrupted,
    failed,
    messages,
    conversation_history,
    effective_task_id,
    turn_id,
    user_message,
    original_user_message,
    _should_review_memory,
    _turn_exit_reason,
    logical_iteration_count=None,
    _pending_verification_response=None,
    _pending_verification_response_previewed=False,
):
    """Run the post-loop finalization and return the turn ``result`` dict.

    Lifted verbatim from ``run_conversation`` (the region after the main agent
    loop). See module docstring.
    """
    from agent.conversation_loop import logger


    if logical_iteration_count is None:
        logical_iteration_count = api_call_count
    budget_exhausted = (
        logical_iteration_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    )
    budget_fallback_eligible = (
        budget_exhausted
        and not interrupted
        and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    )
    continuation_budget_exhausted = (
        final_response is None
        and bool(_pending_verification_response)
        and budget_fallback_eligible
    )

    iteration_limit_fallback = False
    preserved_verification_fallback = False
    if continuation_budget_exhausted:
        # A verification/continuation gate deliberately withheld a composed
        # answer, then consumed the remaining budget before producing a newer
        # one. Preserve that exact answer instead of replacing it with another
        # fallible model call. The explicit pending value is the provenance
        # guard: unrelated error/recovery exits can never enter this branch.
        final_response = _pending_verification_response
        # Mark the turn as previewed only when the reused candidate was
        # actually streamed to the user as interim content. (#65919 review:
        # response-loss blocker)
        if _pending_verification_response_previewed:
            agent._response_was_previewed = True
        _turn_exit_reason = f"max_iterations_reached({logical_iteration_count}/{agent.max_iterations})"
        iteration_limit_fallback = True
        preserved_verification_fallback = True
    elif final_response is None and budget_fallback_eligible:
        # Budget exhausted — ask the model for a summary via one extra
        # API call with tools stripped.  _handle_max_iterations injects a
        # user message and makes a single toolless request.
        _turn_exit_reason = f"max_iterations_reached({logical_iteration_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({logical_iteration_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({logical_iteration_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, logical_iteration_count)
        iteration_limit_fallback = True

    if iteration_limit_fallback:
        # If running as a kanban worker, signal the dispatcher that the
        # worker could not complete (rather than treating it as a
        # protocol violation). This applies whether the user-facing fallback
        # came from the summary call or an explicitly pending continuation;
        # both exhausted the task budget and must advance the failure circuit.
        #
        # We route through ``_record_task_failure(outcome="timed_out")``
        # rather than ``kanban_block`` so this counts toward the dispatcher's
        # consecutive-failure circuit breaker (#29747 gap 2).
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                from hermes_cli import kanban_db as _kb
                _conn = _kb.connect()
                try:
                    _kb._record_task_failure(
                        _conn,
                        _kanban_task,
                        error=(
                            f"Iteration budget exhausted "
                            f"({logical_iteration_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                        outcome="timed_out",
                        release_claim=True,
                        end_run=True,
                        event_payload_extra={
                            "budget_used": logical_iteration_count,
                            "budget_max": agent.max_iterations,
                        },
                    )
                    logger.info(
                        "recorded budget-exhausted failure for task %s (%d/%d)",
                        _kanban_task, logical_iteration_count, agent.max_iterations,
                    )
                finally:
                    try:
                        _conn.close()
                    except Exception:
                        pass
            except Exception:
                logger.warning(
                    "Failed to record budget-exhausted failure for task %s",
                    _kanban_task,
                    exc_info=True,
                )

    # Determine if conversation completed successfully
    normal_text_response = str(_turn_exit_reason).startswith("text_response(")
    completed = (
        final_response is not None
        and not failed
        and (
            logical_iteration_count < agent.max_iterations
            or normal_text_response
        )
    ) or (
        str(_turn_exit_reason) in {
            "completion_delivery_noop",
            "completion_delivery_effect_complete",
        }
        and not failed
        and not interrupted
    )

    # Preflight can seed the display count before the provider receives the
    # request. Roll that estimate back only when an interrupt wins the race
    # before any successful provider response. Compaction state remains owned
    # by the real-usage/post-compaction path, including its ``-1`` sentinel.
    # Guard rules (test-double density on this path is high):
    #  - snapshot is type-pinned to a real int — MagicMock agents auto-create
    #    truthy Mock attributes that must never arm the rollback;
    #  - the received-response flag is pinned to ``is not True`` — its real
    #    domain is True/False, and only a literal True means a provider
    #    response completed;
    #  - the compressor method gets a getattr+callable guard — SimpleNamespace
    #    compressor doubles and plugin context engines lack it.
    _preflight_snapshot = getattr(
        agent, "_turn_preflight_display_snapshot", None
    )
    if (
        interrupted is True
        and isinstance(_preflight_snapshot, int)
        and not isinstance(_preflight_snapshot, bool)
        and getattr(agent, "_turn_received_provider_response", False) is not True
        and getattr(agent, "context_compressor", None) is not None
    ):
        _rollback_fn = getattr(
            agent.context_compressor,
            "rollback_interrupted_preflight_display_tokens",
            None,
        )
        if callable(_rollback_fn):
            _rollback_fn(_preflight_snapshot)

    # Post-loop cleanup must never lose the response.  Trajectory save,
    # resource teardown, and session persistence all touch fallible
    # surfaces — file I/O / JSON serialization (_save_trajectory), remote
    # VM/browser teardown over the network (_cleanup_task_resources), and
    # SQLite writes (_persist_session).  A raise from any of them used to
    # propagate straight out of run_conversation, discarding the partial
    # final_response the caller is waiting for (subprocess wrappers saw an
    # empty stdout with no traceback — #8049).  Each step is now guarded
    # independently so one failure can't skip the others, and any errors
    # are surfaced on the result dict via ``cleanup_errors`` rather than
    # killing the turn.
    _cleanup_errors = []
    _completion_delivery_status = "none"

    # Save trajectory if enabled.  ``user_message`` may be a multimodal
    # list of parts; the trajectory format wants a plain string.
    try:
        agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)
    except Exception as _save_err:
        _cleanup_errors.append(f"save_trajectory: {_save_err}")
        logger.error("finalize_turn: _save_trajectory failed: %s", _save_err, exc_info=True)

    # Clean up VM and browser for this task after conversation completes
    try:
        agent._cleanup_task_resources(effective_task_id)
    except Exception as _cleanup_err:
        _cleanup_errors.append(f"cleanup_task_resources: {_cleanup_err}")
        logger.error("finalize_turn: _cleanup_task_resources failed: %s", _cleanup_err, exc_info=True)

    # Persist session to both JSON log and SQLite only after private retry
    # scaffolding has been removed. Otherwise a later user "continue" turn
    # can replay assistant("(empty)") / recovery nudges and fall into the
    # same empty-response loop again.
    try:
        agent._drop_trailing_empty_response_scaffolding(messages)

        _completion_delivery_status = finalize_completion_delivery_suffix(
            agent,
            messages,
            final_response=final_response,
            failed=failed,
            interrupted=interrupted,
        )
        _completion_delivery_committed = (
            _completion_delivery_status == "committed"
        )
        if _completion_delivery_status == "pending":
            completed = False
            failed = True
            _cleanup_errors.append("completion_delivery_commit: SessionDB append failed")

        # Drop model-only nudges only after the completion suffix has a final
        # disposition.  On a DB failure the completion marker is the atomic
        # write barrier for the still-pending assistant/tool rows; deleting it
        # here would let the generic persist below append an orphan response.
        # Keep the whole suffix intact so a later durability retry still has
        # the event plus every effect it owns.
        if _completion_delivery_status != "pending":
            _drop_ephemeral_scaffolding(messages)

        # A completion event may do real tool work and then honor the prompt's
        # literal-empty final-answer contract.  That is meaningful (so the
        # whole event/tool suffix is durable), but it would otherwise leave a
        # raw tool-result tail.  Reuse the established provider-safe closure
        # helper and keep the synthetic closure out of transcript surfaces.
        if (
            _completion_delivery_committed
            and not flatten_message_text(final_response).strip()
            and close_interrupted_tool_sequence(
                messages, COMPLETION_DELIVERY_TOOL_CLOSURE
            )
        ):
            messages[-1]["display_kind"] = "hidden"

        # When the turn was interrupted and the last message is a tool
        # result, append a synthetic assistant message to close the
        # tool-call sequence. Without this, the session persists a
        # ``tool → user`` alternation that strict providers (Gemini,
        # Claude) reject, causing them to hallucinate a continuation of
        # the user's message on the next turn (#48879).
        #
        # ``_drop_trailing_empty_response_scaffolding`` only rewinds the
        # tool tail when an empty-response scaffolding flag is present; a
        # clean ``/stop`` interrupt after a successful tool sets no such
        # flag, so the tool result survives as the tail and we close it
        # here instead. On an interrupt ``final_response`` is typically
        # empty, so fall back to an explicit placeholder rather than
        # persisting an empty-content assistant turn.
        if interrupted:
            close_interrupted_tool_sequence(messages, final_response)

        # Some recovery/fallback paths return a real final_response without
        # adding a closing assistant message to the transcript (e.g. the
        # partial-stream and prior-turn-content recovery ``break`` sites in
        # ``conversation_loop``). If persisted as-is, the durable session can
        # end at a tool/user message even though the caller — and the gateway
        # platform — already saw a completed assistant response. The next turn
        # then replays a user-only backlog and the model re-answers every
        # "unanswered" message. Close the durable turn at the source, at the
        # single chokepoint every recovery ``break`` flows through, so the
        # invariant "delivered final_response ⇒ assistant row in transcript"
        # holds regardless of which path produced it. (#43849 / #44100)
        #
        # Compare content (not just role) so a verification candidate that
        # matches the final response is not duplicated at budget
        # exhaustion. (#65919 §7)
        if final_response and not interrupted:
            try:
                _tail = messages[-1] if messages else None
            except Exception:
                _tail = None
            _tail_role = _tail.get("role") if isinstance(_tail, dict) else None
            if _tail_role != "assistant":
                # Tail is not an assistant row — append the final response
                # so the durable turn closes with the answer (#43849/#44100).
                messages.append({"role": "assistant", "content": final_response})
            elif isinstance(_tail, dict) and _tail.get("content") != final_response and _is_pure_tool_call_tail(_tail):
                # The tail IS an assistant row, but a *pure tool-call turn*:
                # tool_calls with no text of its own. The role check alone
                # leaves the #43849/#44100 invariant unmet — the user saw a
                # response that never reached the transcript, and the next turn
                # replays the user backlog and re-answers it (the very symptom
                # this block was added for). Fill that row's empty content
                # instead of appending, so the durable turn ends with the answer
                # without disturbing the tool-call structure or creating an
                # assistant→assistant pair.
                #
                # The ``content != final_response`` guard prevents filling when
                # the tail already carries the final response text (verification
                # candidate collapse — the provisional answer was persisted and
                # reused as the terminal response, #65919 §7).
                _tail["content"] = final_response
                # The row may have already been flushed to SQLite by the
                # incremental tool-call persist (conversation_loop.py:4990),
                # which stamps ``_DB_PERSISTED_MARKER`` so subsequent flushes
                # skip it. Pop the marker so the next ``_persist_session``
                # re-writes the filled content to the durable store —
                # otherwise ``/resume`` reloads ``content=""`` and the bug
                # resurfaces cross-session.
                _tail.pop("_db_persisted", None)
                # The bounded flush-scan cursor (run_agent.py) skips the
                # identity-matched prefix of its previous snapshot on the
                # assumption that no live dict loses the marker in place —
                # this pop is the one place that does. Invalidate it so the
                # filled row is re-examined instead of skipped.
                agent._db_flush_scan_prefix = None

        # The model has completed its request, so replace API-local
        # voice/model/skill guidance with the clean user input before writing the
        # final durable snapshot and returning the continuation history. Earlier
        # turn-start flushes use the DB-only override because their messages are
        # still needed for the API request; this finalizer runs after that request
        # is complete (#48677 / #63766).
        _apply_override = getattr(agent, "_apply_persist_user_message_override", None)
        if callable(_apply_override):
            _apply_override(messages)
        # ── Post-turn micro-compaction ────────────────────────────
        # After the assistant response is finalized but before the session is
        # persisted, run micro-compaction to absorb the oldest uncompacted
        # exchange into the rolling summary.  This amortizes compression
        # across turns rather than batching it into one big pause.
        if not interrupted and not failed:
            try:
                _compressor = getattr(agent, "context_compressor", None)
                # Strict `is True` + isinstance gates: plugin context engines
                # (and MagicMock compressors in tests) satisfy getattr/duck
                # checks with truthy auto-attributes — a bare truthiness check
                # here called _micro_compact on a mock and spliced its (empty-
                # iterating) return value over the transcript, wiping it.
                if (
                    _compressor
                    and getattr(_compressor, '_micro_compact_enabled', False) is True
                    and callable(getattr(_compressor, '_micro_compact', None))
                    and final_response
                    # Persistence-isolated agents (background review fork)
                    # must not micro-compact: the pass burns a real aux-LLM
                    # call on a throwaway replay transcript, and if the
                    # compressor ever holds a session_db binding it would
                    # archive_and_compact the CANONICAL session rows — the
                    # exact write class _persist_disabled exists to stop.
                    and not getattr(agent, "_persist_disabled", False)
                ):
                    # A DB-bound micro pass must include this just-finalized
                    # turn in the authoritative active view it reloads under
                    # the compression lease. The existing flush protocol
                    # excludes any live-only pending completion suffix.
                    _micro_flush_ready = True
                    if (
                        getattr(_compressor, "_session_db", None)
                        and getattr(_compressor, "_session_id", "")
                    ):
                        _flush = getattr(
                            agent, "_flush_messages_to_session_db", None
                        )
                        try:
                            _micro_flush_ready = bool(
                                callable(_flush)
                                and _flush(
                                    messages,
                                    conversation_history=conversation_history,
                                )
                                is True
                            )
                        except Exception as exc:
                            logger.warning(
                                "Micro-compaction pre-snapshot flush failed; "
                                "skipping this pass: %s",
                                exc,
                            )
                            _micro_flush_ready = False
                    _before = len(messages)
                    _compacted = (
                        _compressor._micro_compact(messages)
                        if _micro_flush_ready
                        else messages
                    )
                    # Micro-compaction defrag rewrites the newest MICRO
                    # marker's content and pops _db_persisted from the live
                    # dict in place — the sibling of the pop site above. The
                    # compressor has no agent reference, so it raises a flag
                    # for us to invalidate the bounded flush-scan cursor;
                    # otherwise the rewritten marker row is identity-skipped
                    # and the stale summary persists to state.db.
                    if getattr(
                        _compressor, "_flush_scan_cursor_invalidated", False
                    ):
                        _compressor._flush_scan_cursor_invalidated = False
                        agent._db_flush_scan_prefix = None
                    if isinstance(_compacted, list) and _compacted:
                        messages[:] = _compacted
                        if (
                            getattr(_compressor, "_session_db", None)
                            and getattr(_compressor, "_session_id", "")
                            == getattr(agent, "session_id", None)
                        ):
                            from agent.message_sanitization import (
                                durable_messages_before_pending_completion,
                            )

                            _durable_compacted = (
                                durable_messages_before_pending_completion(
                                    messages,
                                )
                            )
                            agent._last_flushed_db_idx = len(_durable_compacted)
                            agent._flushed_db_message_session_id = agent.session_id
                            agent._flushed_db_message_ids = {
                                id(message)
                                for message in _durable_compacted
                                if isinstance(message, dict)
                            }
                    _after = len(messages)
                    if _before != _after:
                        logger.info(
                            "Micro-compaction: %d -> %d messages",
                            _before, _after,
                        )
            except Exception as _mc_err:
                logger.info("Micro-compaction failed: %s", _mc_err)

        agent._persist_session(messages, conversation_history)
    except Exception as _persist_err:
        _cleanup_errors.append(f"persist_session: {_persist_err}")
        logger.error("finalize_turn: _persist_session failed: %s", _persist_err, exc_info=True)

    # The gateway owns a separate in-memory history snapshot. Keep it current
    # even when finalization reports a cleanup error: a later prompt must not be
    # sent with the pre-turn snapshot while the durable DB already has this turn.
    try:
        agent._session_messages = messages
    except Exception:
        pass

    # ── Turn-exit diagnostic log ─────────────────────────────────────
    # Always logged at INFO so agent.log captures WHY every turn ended.
    # When the last message is a tool result (agent was mid-work), log
    # at WARNING — this is the "just stops" scenario users report.
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to find the assistant message with the tool call
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d logical_iterations=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count,
        logical_iteration_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # Agent was mid-work — this is the "just stops" case.
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # File-mutation verifier footer.
    # If one or more ``write_file`` / ``patch`` calls failed during this
    # turn and were never superseded by a successful write to the same
    # path, append an advisory footer to the assistant response.  This
    # catches the specific case — reported by Ben Eng (#15524-adjacent)
    # — where a model issues a batch of parallel patches, half of them
    # fail with "Could not find old_string", and the model summarises
    # the turn claiming every file was edited.  The user then has to
    # manually run ``git status`` to catch the lie.  With this footer
    # the truth is surfaced on every turn, so over-claiming is
    # structurally impossible past the model.
    #
    # Gate: only applied when a real text response exists for this
    # turn and the user didn't interrupt.  Empty/interrupted turns
    # already have other surface text that shouldn't be augmented.
    if final_response and not interrupted:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    # Turn-completion explainer.
    # When a turn ends abnormally after substantive work — empty content
    # after retries, a partial/truncated stream, a still-pending tool
    # result, or an iteration/budget limit — the user otherwise gets a
    # blank or fragmentary response box with no consolidated reason why
    # the agent stopped (#34452).  Surface a single user-visible
    # explanation derived from ``_turn_exit_reason``, mirroring the
    # file-mutation verifier footer pattern above.
    #
    # Gate carefully so healthy turns stay quiet:
    #   - ``text_response(...)`` exits never produce an explanation
    #     (handled inside the formatter), so a terse ``Done.`` is silent.
    #   - We only ACT when there is no genuinely usable reply this turn:
    #     an empty response, the "(empty)" terminal sentinel, or a
    #     suspiciously short partial fragment with no terminating
    #     punctuation (e.g. "The").  A real short answer keeps its text.
    if not interrupted and str(_turn_exit_reason) not in {
        "completion_delivery_noop",
        "completion_delivery_effect_complete",
    }:
        try:
            if agent._turn_completion_explainer_enabled():
                _stripped = (final_response or "").strip()
                _is_empty_terminal = _stripped == "" or _stripped == "(empty)"
                # A short fragment that is not a normal text_response exit
                # and lacks sentence-ending punctuation is treated as a
                # truncated partial (the "The" case from #34452).
                _is_partial_fragment = (
                    not _is_empty_terminal
                    and not preserved_verification_fallback
                    and not str(_turn_exit_reason).startswith("text_response")
                    and len(_stripped) <= 24
                    and _stripped[-1:] not in {".", "!", "?", "。", "！", "？", "`", ")"}
                )
                _is_partial_stream_recovery = (
                    str(_turn_exit_reason) == "partial_stream_recovery"
                )
                if (
                    _is_empty_terminal
                    or _is_partial_fragment
                    or _is_partial_stream_recovery
                ):
                    _explanation = agent._format_turn_completion_explanation(
                        _turn_exit_reason
                    )
                    if _explanation:
                        if _is_empty_terminal:
                            # Replace the bare "(empty)"/blank sentinel with
                            # the actionable explanation.
                            final_response = _explanation
                        else:
                            # Keep the partial fragment, append the reason so
                            # the user sees both what arrived and why it
                            # stopped.
                            final_response = (
                                _stripped + "\n\n" + _explanation
                            )
        except Exception as _exp_err:
            logger.debug("turn-completion explainer failed: %s", _exp_err)

    _response_transformed = False

    # Plugin hook: transform_llm_output
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can transform the LLM's output text before it's returned.
    # First hook to return a string wins; None/empty return leaves text unchanged.
    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # Plugin hook: post_llm_call
    # Fired once per turn after the tool-calling loop completes.
    # Plugins can use this to persist conversation data (e.g. sync
    # to an external memory system).
    if final_response and not interrupted:
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                task_id=effective_task_id,
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # Context engine observation hook: notify the active engine that this
    # turn has finished, with the finalized transcript. Complements the
    # per-request select_context() hook (selection before the request;
    # observation after the turn). No-op default, fail-open.
    try:
        from agent.conversation_loop import _notify_context_engine_turn_complete
        # Forward the turn's canonical usage when the host has it. The loop
        # stashes the most recent API response's usage dict (the same
        # canonical buckets fed to ``update_from_response``) on the agent as
        # ``_last_turn_usage``. It is ``None`` on turns that never reached a
        # provider response (early failure / interrupt), which is exactly the
        # contract: real usage when available, ``None`` otherwise.
        _turn_usage = getattr(agent, "_last_turn_usage", None)
        _notify_context_engine_turn_complete(
            agent,
            messages,
            usage=_turn_usage,
            logger=logger,
            turn_id=turn_id,
            task_id=effective_task_id,
            api_call_count=api_call_count,
            interrupted=interrupted,
            failed=failed,
            turn_exit_reason=_turn_exit_reason,
        )
    except Exception as exc:
        logger.warning("on_turn_complete notification failed: %s", exc)

    # Extract reasoning from the CURRENT turn only.  Walk backwards
    # but stop at the user message that started this turn — anything
    # earlier is from a prior turn and must not leak into the reasoning
    # box (confusing stale display; #17055).  Within the current turn
    # we still want the *most recent* non-empty reasoning: many
    # providers (Claude thinking, DeepSeek v4, Codex Responses) emit
    # reasoning on the tool-call step and leave the final-answer step
    # with reasoning=None, so picking only the last assistant would
    # silently drop legitimate same-turn reasoning.
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    result_messages = messages
    if _completion_delivery_status == "pending":
        from agent.message_sanitization import (
            durable_messages_before_pending_completion,
        )

        result_messages = list(durable_messages_before_pending_completion(messages))

    # Build result with interrupt info if applicable
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": result_messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": _completion_delivery_status == "pending",
        "interrupted": interrupted,
        "completion_delivery_status": _completion_delivery_status,
        "response_transformed": _response_transformed,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        # Requested service tier (from request_overrides.extra_body), for
        # billing audits by callers like `hermes -z --usage-file`.
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # Persistence failures already set failed=True + an explanation in
    # final_response; also stamp `error` so gateway surfaces status="error"
    # (and desktop can toast disk-full) instead of a quiet complete frame.
    if failed and str(_turn_exit_reason) == "session_persistence_failed":
        result["error"] = final_response or (
            "session storage could not be written — free disk space and try again"
        )
    elif _completion_delivery_status == "pending":
        result["error"] = completion_delivery_commit_error_message(agent)
    # Surface any post-loop cleanup failures so the caller can distinguish a
    # clean turn from one whose trajectory/session/resource teardown raised
    # (the response is still returned either way — #8049).
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # If a /steer landed after the final assistant turn (no more tool
    # batches to drain into), hand it back to the caller so it can be
    # delivered as the next user turn instead of being silently lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False

    # Include interrupt message if one triggered the interrupt
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # Clear interrupt state after handling
    agent.clear_interrupt()

    # Clear stream callback so it doesn't leak into future calls
    agent._stream_callback = None

    # Check skill trigger NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # Background memory/skill review — runs AFTER the response is delivered
    # so it never competes with the user's task for model attention.
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

    # Note: Memory provider on_session_end() + shutdown_all() are NOT
    # called here — run_conversation() is called once per user message in
    # multi-turn sessions. Shutting down after every turn would kill the
    # provider before the second message. Actual session-end cleanup is
    # handled by the CLI (atexit / /reset) and gateway (session expiry /
    # _reset_session).

    # Plugin hook: on_session_end
    # Fired at the very end of every run_conversation call.
    # Plugins can use this for cleanup, flushing buffers, etc.
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            completed=completed,
            failed=failed,
            interrupted=interrupted,
            turn_exit_reason=_turn_exit_reason,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    from agent.cache_attribution import (
        clear_post_compression_cache_pending_after_empty_usage,
    )

    clear_post_compression_cache_pending_after_empty_usage(agent)
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False

    return result
