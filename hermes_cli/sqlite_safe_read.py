"""Lock-safe inspection of SQLite database files.

POSIX advisory locks are cancelled **process-wide** by ``close()`` on *any* fd for that file, so a
bare ``open(db_path, "rb") ... close()`` on a live database drops every lock SQLite holds from this
process (a VACUUM's EXCLUSIVE lock, an in-flight BEGIN IMMEDIATE's RESERVED lock). This module
tracks live connections so raw reads happen only when none exist.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Guards BOTH the registry and the lifecycle syscalls it describes. Reentrant
# because connect_tracked -> _canonical_db_path -> ... stays on one thread.
_live_lock = threading.RLock()
_admission_changed = threading.Condition(_live_lock)
# canonical path -> number of live connections opened by this process
_live_connections: dict[str, int] = {}
# Immutable admission tokens survive rename/replacement; never restat at close.
_live_tokens: dict[object, tuple[str, frozenset[tuple[int, int]]]] = {}
_unknown_tokens: set[object] = set()
_manual_tokens: dict[str, list[object]] = {}
_reservations: list[_Handoff] = []
_SIDECARS = ("-wal", "-shm", "-journal")


def _identity(path):
    try:
        info = os.stat(path)
        return info.st_dev, info.st_ino
    except FileNotFoundError:
        return None


def _bundle(key):
    return (key, *(key + suffix for suffix in _SIDECARS))


def _identities(key):
    return frozenset(identity for path in _bundle(key) if (identity := _identity(path)) is not None)


class _Handoff:
    """Strongly owned reservation and explicit capability for repair-only I/O."""

    def __init__(self, key, tokens, *, snapshot=False):
        self.key = key
        self.main_identity = _identity(key)
        self.paths = frozenset(_bundle(key))
        self.identities = _identities(key)
        self.tokens = set(tokens)
        self.thread = threading.get_ident()
        self.pending = 0
        self.raw = 0
        self.snapshot = snapshot

    def matches(self, key):
        # Keep sidecars created by owner SQL claimed before any alias opener.
        self.identities |= _identities(self.key)
        return key in self.paths or _identity(key) in self.identities


def _reservation(key, handoff=None):
    for claim in _reservations:
        if claim.matches(key):
            if claim is not handoff or claim.thread != threading.get_ident():
                raise ConnectionAdmissionError(f"SQLite lifecycle reserved for recovery: {key}")
            # Inode ownership is not permission to reopen a replaced path/generation.
            if (_identity(claim.key) != claim.main_identity
                    or os.stat(claim.key).st_nlink != 1):
                raise ConnectionAdmissionError(f"SQLite recovery pathname changed: {claim.key}")
            return claim
    if handoff is not None and handoff not in _reservations:
        raise ConnectionAdmissionError("SQLite recovery capability has expired")
    return None


class UntrackableConnectionError(RuntimeError):
    """A connection to a probe-able database could not be tracked. Raised rather than returning an
    untracked connection: on these paths tracking is part of the correctness contract."""


class LiveConnectionError(RuntimeError):
    """A raw file operation was attempted on a database with live connections."""


class ConnectionAdmissionError(LiveConnectionError):
    """An expected pre-open reservation refusal; readers may use writer fallback."""


def _key(path: Path | str) -> str:
    """Canonicalise a *filesystem* path for use as a registry key."""
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _canonical_db_path(conn: sqlite3.Connection) -> Optional[str]:
    """The on-disk path of ``main`` as SQLite reports it (immune to ``file:`` URIs, relative paths,
    symlinks). ``None`` for in-memory/unnamed databases, which cannot be byte-probed."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return None
    if not row or len(row) < 3 or not row[2]:
        return None
    return _key(row[2])


def track_connection(path: Path | str) -> None:
    """Record that this process holds a connection to *path* (prefer :func:`connect_tracked`; this
    is for callers managing their own connection objects, and for tests)."""
    with _live_lock:
        key = _key(path)
        _reservation(key)
        token = object()
        _manual_tokens.setdefault(key, []).append(token)
        _live_tokens[token] = (key, _identities(key))
        if _identity(key) is None:
            _unknown_tokens.add(token)
        _track_key(key)


def _track_key(key: str, delta: int = 1) -> None:
    """Adjust the live count for an already-canonical key (caller holds ``_live_lock``)."""
    remaining = _live_connections.get(key, 0) + delta
    if remaining > 0:
        _live_connections[key] = remaining
    else:
        _live_connections.pop(key, None)


def untrack_connection(path: Path | str) -> None:
    """Record that one connection to *path* has been closed."""
    with _live_lock:
        key = _key(path)
        tokens = _manual_tokens.get(key, [])
        if tokens:
            _untrack_token(tokens.pop())
        if not tokens:
            _manual_tokens.pop(key, None)


def _untrack_token(token):
    _unknown_tokens.discard(token)
    entry = _live_tokens.pop(token, None)
    if entry is not None:
        _track_key(entry[0], -1)
    for claim in _reservations:
        claim.tokens.discard(token)


def _live_main_key(key: str) -> Optional[str]:
    """The tracked main-database key that makes *key* live, or ``None`` (caller holds ``_live_lock``).

    SQLite locks the main file and its WAL sidecars; a raw ``close()`` of any of those
    inodes cancels this process's POSIX locks. Unknown disk bindings fence every raw
    operation until a successful physical close settles their custody."""
    if _unknown_tokens:
        return _live_tokens[next(iter(_unknown_tokens))][0]
    identity = _identity(key)
    for main, captured in _live_tokens.values():
        if key in _bundle(main) or (identity is not None and identity in captured):
            return main
        # Sidecars may be created after connect; stat only, never open a raw FD.
        if _identity(main) in captured and identity is not None and identity in _identities(main):
            return main
    return None


def has_live_connection(path: Path | str) -> bool:
    """Whether this process holds a connection to *path* (or to the database it is a sidecar of).

    Point-in-time answer: a raw open/close right after it returns ``False`` can still race a
    new connection. Hold :func:`offline_file_access` across the I/O whenever possible."""
    with _live_lock:
        return _live_main_key(_key(path)) is not None


class _TrackingMixin:
    """Untrack-on-close behaviour, mixable into any Connection subclass.

    The real ``close()`` and the unregister happen together under ``_live_lock`` so a concurrent
    probe can never observe "no live connection" while this descriptor is still open. Unregister
    runs only after ``close()`` succeeds: untracking before a failing close (e.g. cross-thread
    ProgrammingError) would leave the FD open while the byte-probe guard thinks nothing is live.
    """

    _hermes_tracked_path: str | None = None
    _hermes_tracking_token: object | None = None
    _hermes_identity_uncertain: bool = False

    def close(self) -> None:  # type: ignore[misc]
        with _live_lock:
            token = getattr(self, "_hermes_tracking_token", None)
            claim = next((r for r in _reservations if token in r.tokens), None)
            if claim is not None and claim.thread != threading.get_ident():
                raise LiveConnectionError("Only the admitted recovery owner may close this connection")
            if claim is not None:
                claim.pending += 1
        # ponytail: ordinary open/close keep the inherited global syscall boundary;
        # admitted recovery I/O alone leaves the metadata lock during slow work.
        guard = contextlib.nullcontext() if claim is not None else _live_lock
        try:
            with guard:
                with _live_lock:
                    current = next((r for r in _reservations if token in r.tokens), None)
                    if current is not None and current is not claim:
                        raise LiveConnectionError("Connection was reserved before close")
                super().close()  # type: ignore[misc]
                with _live_lock:
                    if token is not None:
                        self._hermes_tracking_token = None
                        self._hermes_tracked_path = None
                        _untrack_token(token)
        finally:
            if claim is not None:
                with _live_lock:
                    claim.pending -= 1


class TrackedConnection(_TrackingMixin, sqlite3.Connection):
    """A ``sqlite3.Connection`` that untracks its path exactly once on close (callers close in many
    places, some via ``contextlib.closing``, so counting closes by hand is unreliable)."""


_tracked_factory_cache: dict[type, type] = {}


def _tracking_factory(factory: type) -> type:
    """Return *factory* augmented with untrack-on-close. Callers legitimately pass their own
    ``Connection`` subclasses (tests simulate FTS5-less or pragma-failing runtimes); leaving them
    untracked would quietly unguard the database, so the tracking ``close()`` is mixed in."""
    if factory is sqlite3.Connection:
        return TrackedConnection
    if issubclass(factory, _TrackingMixin):
        return factory
    cached = _tracked_factory_cache.get(factory)
    if cached is None:
        cached = type(f"Tracked{factory.__name__}", (_TrackingMixin, factory), {})
        _tracked_factory_cache[factory] = cached
    return cached


def connect_tracked(
    path: Path | str, *, tracking_path: Path | str | None = None, connect_fn=None, handoff=None, **kwargs,
) -> sqlite3.Connection:
    """Open and publish a tracked handle atomically against raw access and recovery.

    Ordinary native calls wait for foreign snapshots, without holding the metadata
    lock during that wait. Recovery reservations and uncertain/custom targets refuse.
    Physical ordinary opens retain the inherited global syscall boundary; only an
    explicit reservation capability permits target I/O outside it, before publication.
    """
    from urllib.parse import quote, unquote, urlsplit
    opener = connect_fn if connect_fn is not None else sqlite3.connect
    factory = kwargs.get("factory", sqlite3.Connection)
    opaque = opener is not sqlite3.connect or factory not in (sqlite3.Connection, TrackedConnection)
    kwargs["factory"] = _tracking_factory(factory)
    uri = urlsplit(str(path)) if kwargs.get("uri") and str(path).startswith("file:") else None
    memory = str(path) == ':memory:' or (uri is not None and
             (uri.path == ':memory:' or 'mode=memory' in uri.query.split('&')))
    spelling = unquote(uri.path) if uri is not None else path
    key = _key(spelling)
    label = _key(tracking_path) if tracking_path is not None else key
    # Resolve native filesystem aliases once; an advisory label never changes the target.
    target = str(path) if opaque or memory or not str(path) else (
        uri._replace(path=quote(key, safe='/')).geturl() if uri is not None else key)

    def admit(*resources, wait_snapshot=False):
        candidates = [candidate for resource in (key, *resources) for candidate in _bundle(resource)]
        # Recheck the entire resource set after a wake: waiting for one sidecar
        # must not leave an earlier main-file admission stale.
        while (wait_snapshot and not opaque and label == key and handoff is None
               and any(r.snapshot and r.thread != threading.get_ident()
                       and any(r.matches(candidate) for candidate in candidates)
                       for r in _reservations)):
            _admission_changed.wait()
        claim = _reservation(key, handoff)
        for candidate in candidates:
            reservation = _reservation(candidate, handoff)
            if reservation is not None and reservation.raw:
                raise ConnectionAdmissionError(f"SQLite resource is in raw access: {candidate}")
        # A user opener/factory may ignore path or retarget it. It can run only
        # under the global syscall boundary with no foreign raw-I/O reservation.
        if opaque and any(r.raw or r is not handoff or r.thread != threading.get_ident() for r in _reservations):
            raise ConnectionAdmissionError("Custom SQLite opener cannot prove its reserved resource")
        return claim

    with _live_lock:
        claim = admit(label, wait_snapshot=True)
        if claim is not None:
            claim.pending += 1
    guard = contextlib.nullcontext() if claim is not None and not opaque else _live_lock
    try:
        with guard:
            with _live_lock:
                post_key = _key(spelling)
                admit(label, post_key, wait_snapshot=post_key == key)
                before = _identity(key)
            conn = opener(target, **kwargs)
            try:
                actual = _canonical_db_path(conn)
                resolved = label if tracking_path is not None else actual
                if resolved is None:
                    if memory:
                        return conn  # known memory connections have no disk descriptor
                    resolved = key  # unknown file-backed identity must retain custody
                if not isinstance(conn, _TrackingMixin):
                    conn = _retrofit_tracking(conn, resolved)
                with _live_lock:
                    token = object()
                    post_key = _key(spelling)
                    admit(resolved, post_key, actual or key)
                    captured = _identities(resolved) | _identities(key) | _identities(post_key)
                    if actual is not None:
                        captured |= _identities(actual)
                    conn._hermes_identity_uncertain = (
                        actual is None or actual != key or resolved != key or post_key != key
                        or (before is not None and before != _identity(resolved))
                    )
                    if before is not None:
                        captured = captured | {before}
                    conn._hermes_tracked_path = resolved
                    conn._hermes_tracking_token = token
                    _live_tokens[token] = (resolved, captured)
                    if actual is None:
                        _unknown_tokens.add(token)
                    _track_key(resolved)
                    if claim is not None:
                        claim.tokens.add(token)
                return conn
            except BaseException:
                # The ordinary syscall boundary or the admitted reservation still
                # excludes raw I/O while setup unwinds.
                sqlite3.Connection.close(conn)
                raise
    finally:
        if claim is not None:
            with _live_lock:
                claim.pending -= 1


def _retrofit_tracking(conn: sqlite3.Connection, resolved: str) -> sqlite3.Connection:
    """Give an already-open connection untrack-on-close semantics by swapping ``__class__`` for
    one mixing in the tracking ``close()`` (used when an opener ignored the factory we asked for)."""
    cls = type(conn)
    try:
        conn.__class__ = _tracking_factory(cls)  # type: ignore[assignment]
        return conn
    except TypeError as exc:
        raise UntrackableConnectionError(
            f"connection to {resolved} uses factory {cls.__name__}, which "
            "cannot release its tracking entry on close; byte-probe safety "
            "for this database would be silently lost"
        ) from exc


def page_count_bytes(conn: sqlite3.Connection) -> Optional[int]:
    """Logical database size in bytes (``page_count * page_size``, the header field at offset 28)
    read via PRAGMA over *conn* so no new fd is opened. ``None`` when the pragmas cannot be read."""
    try:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)
    except (sqlite3.Error, TypeError, IndexError, ValueError) as exc:
        logger.debug("page_count/page_size unavailable: %s", exc)
        return None


def file_length_matches_header(conn: sqlite3.Connection) -> Optional[bool]:
    """Whether the file on disk is at least as long as the header claims ("torn extend" check),
    without opening the file (PRAGMA over *conn* + ``stat()``). Advisory in WAL mode: a freshly
    committed page may still live in ``-wal`` so the main file legitimately lags."""
    path_str = _canonical_db_path(conn)
    if path_str is None:
        return None
    logical = page_count_bytes(conn)
    if not logical:
        return None
    try:
        return os.path.getsize(path_str) >= logical
    except OSError:
        return None


def read_header_bytes_preopen(path: Path | str, *, length: int = 100, force: bool = False) -> Optional[bytes]:
    """Read the first *length* bytes of *path* -- only when no connection is live. The ONLY
    sanctioned byte-level read of a database file, for first-open validation (real SQLite? zeroed?
    overwritten?). Check and open/read/close run together under ``_live_lock`` so a connection
    cannot be opened between deciding "nothing is live" and closing this descriptor."""
    with _live_lock:
        try:
            _reservation(_key(path))
        except LiveConnectionError:
            return None
        if not force and _live_main_key(_key(path)) is not None:
            logger.debug(
                "refusing byte-level read of %s: a live connection exists in "
                "this process and close() would cancel its POSIX locks",
                path)
            return None
        try:
            with open(path, "rb") as handle:
                return handle.read(length)
        except OSError:
            return None


@contextlib.contextmanager
def connection_handoff(path: Path | str, connection: sqlite3.Connection, *, owned_readers=()):
    """Atomically try sole-owner admission, then retain only target-path/inode claims.

    Never wait while the caller owns a writer lock. Ordinary physical opens/closes
    still hold the short lifecycle boundary, so unregistered/pending handles refuse.
    Hardlinks are not repair authority: distinct WAL bundles are ambiguous.
    """
    if not _live_lock.acquire(blocking=False):
        yield False
        return
    claim = None
    try:
        key = _key(path)
        owned = (connection, *owned_readers)
        tokens = {getattr(conn, "_hermes_tracking_token", None) for conn in owned}
        identity = _identity(key)
        live = {token for token, (main, captured) in _live_tokens.items()
                if main == key or (identity is not None and identity in captured)}
        if (identity is not None and os.stat(key).st_nlink == 1
                and not _unknown_tokens
                and len(tokens) == len(owned) and None not in tokens and live == tokens
                and all(getattr(conn, "_hermes_tracked_path", None) == key
                        and not getattr(conn, "_hermes_identity_uncertain", False) for conn in owned)
                and all(identity in _live_tokens[token][1] for token in tokens if token in _live_tokens)
                and not any(r.matches(key) for r in _reservations)):
            claim = _Handoff(key, tokens)
            _reservations.append(claim)
    finally:
        _live_lock.release()
    try:
        yield claim if claim is not None else False
    finally:
        if claim is not None:
            with _live_lock:
                if claim.pending:
                    raise RuntimeError("SQLite recovery ended with pending physical I/O")
                _reservations.remove(claim)


@contextlib.contextmanager
def offline_file_access(path: Path | str, *, what: str = "read", handoff=None):
    """Reserve raw I/O per resource without fencing unrelated databases.

    Native ordinary opens wait until raw custody ends. Same-thread/capability
    overlap refuses, so an owner never waits for its own scope to exit.
    """
    key = _key(path)
    with _live_lock:
        claim = _reservation(key, handoff)
        main = _live_main_key(key)
        if main is not None:
            raise LiveConnectionError(
                f"Refusing to {what} {path}: a connection to {main} is still open "
                "in this process, and raw file access would cancel its POSIX locks. "
                "Close all database handles and retry.")
        temporary = claim is None
        if temporary:
            claim = _Handoff(key, (), snapshot=True)
            _reservations.append(claim)
        claim.pending += 1
        claim.raw += 1
    try:
        yield
    finally:
        with _live_lock:
            claim.raw -= 1
            claim.pending -= 1
            if temporary:
                _reservations.remove(claim)
                _admission_changed.notify_all()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"
# ---- END PLUGIN-COMPAT ----
