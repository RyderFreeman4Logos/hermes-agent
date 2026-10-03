"""Live, process-owned issuance for cooperative native delegation (Linux only).

Environment strings locate an issuer and carry its MAC, never grant authority by
PID alone. The issuer retains its key and validates requests over a peer-checked
Unix socket. Shells need no inherited descriptor or stdio changes. Grants expire
with the issuing process; native descendants may renew only after validation.

This authenticates trusted Hermes issuance, not hostile same-UID Python code:
code that replaces this module, impersonates the issuer protocol, reads process
memory, or steals an issued capability is outside this cooperative boundary.
No capability/key is persisted, logged, or put in a prompt.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import select
import socket
import struct
import sys
import threading

__all__ = ["issue_lineage", "validate_lineage"]

_MAX_STAMP_BYTES = 4096
_TIMEOUT = 1.0
_lock = threading.Lock()
_construction_lock = threading.Lock()
_issuer: tuple[socket.socket, bytes, int, str] | None = None
_sockets: set[socket.socket] = set()


def _open_socket(listener: socket.socket | None = None) -> socket.socket:
    # Fork must not land between FD allocation/accept and ownership registration.
    # The listener is nonblocking; all readiness and protocol waits stay outside.
    with _construction_lock:
        conn = listener.accept()[0] if listener is not None else socket.socket(
            socket.AF_UNIX, socket.SOCK_SEQPACKET,
        )
        _sockets.add(conn)
        return conn


def _close_socket(conn: socket.socket) -> None:
    with _construction_lock:
        conn.close()
        _sockets.discard(conn)


def _address(pid: int, start: str) -> str:
    return "\0hermes-delegation-" + hashlib.sha256(f"{pid}:{start}".encode()).hexdigest()


def _peer(sock: socket.socket) -> tuple[int, int, int]:
    return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))


def _descendant(pid: int, ancestor: int) -> bool:
    import psutil

    try:
        # ponytail: cap ancestry work; deeper trees fail closed, not an unbounded walk.
        for _ in range(64):
            if pid == ancestor:
                return True
            if pid <= 1:
                return False
            pid = psutil.Process(pid).ppid()
    except psutil.Error:
        return False
    return False


def _serve(listener: socket.socket, key: bytes, pid: int) -> None:
    ready = select.poll()
    ready.register(listener, select.POLLIN)
    while True:
        try:
            ready.poll()
            conn = _open_socket(listener)
        except BlockingIOError:
            continue
        except OSError:
            return
        try:
            conn.settimeout(_TIMEOUT)
            peer_pid, uid, _ = _peer(conn)
            packet = conn.recv(_MAX_STAMP_BYTES + 65)
            proof, stamp = packet[:64], packet[64:]
            valid = (
                uid == os.getuid() and 0 < len(stamp) <= _MAX_STAMP_BYTES
                and hmac.compare_digest(proof, hmac.digest(key, stamp, "sha256").hex().encode())
                and _descendant(peer_pid, pid)
            )
            conn.sendall(b"1" if valid else b"0")
        except OSError:
            continue
        finally:
            _close_socket(conn)


def _before_fork() -> None:
    _construction_lock.acquire()


def _after_fork_parent() -> None:
    _construction_lock.release()


def _after_fork() -> None:
    global _issuer, _lock, _construction_lock
    for conn in _sockets:
        conn.close()
    _sockets.clear()
    _issuer = None
    _lock = threading.Lock()
    _construction_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork,
    )


def issue_lineage(stamp: str) -> tuple[str, str, str] | None:
    """Issue (pid, start, proof) only for a caller-validated delegation stamp.

    One close-on-exec listener/key per process, no per-session table, no retained
    grants. The OS closes the listener at exit. Forked children discard it before
    any child issuance, so neither a vanished thread nor its lock is reused.
    """
    global _issuer
    try:
        payload = stamp.encode()
    except UnicodeError:
        return None
    if sys.platform != "linux" or not payload or len(payload) > _MAX_STAMP_BYTES:
        return None
    from gateway.status import get_process_start_time

    with _lock:
        if _issuer is None:
            pid = os.getpid()
            start = get_process_start_time(pid)
            if start is None:
                return None
            listener = _open_socket()
            try:
                listener.bind(_address(pid, str(start)))
                listener.listen(16)
                listener.setblocking(False)
                key = secrets.token_bytes(32)
                worker = threading.Thread(
                    target=_serve, args=(listener, key, pid), daemon=True,
                    name="delegation-lineage",
                )
                worker.start()
            except BaseException as exc:
                _close_socket(listener)
                if isinstance(exc, (OSError, RuntimeError)):
                    return None
                raise
            _issuer = listener, key, pid, str(start)
        _, key, pid, start = _issuer
        return str(pid), start, hmac.digest(key, payload, "sha256").hex()


def validate_lineage(stamp: str, pid_text: str, start: str, proof: str) -> bool:
    """Require live issuer incarnation, peer ownership, and issuer-validated MAC.

    Malformed, absent, stale, dead, foreign or unavailable authority fails closed.
    PID/start identify the IPC peer; only its private-key proof grants authority.
    """
    try:
        payload = stamp.encode()
    except UnicodeError:
        return False
    if sys.platform != "linux" or not payload or len(payload) > _MAX_STAMP_BYTES:
        return False
    if not pid_text.isascii() or not pid_text.isdecimal() or not start.isascii() or not start.isdecimal():
        return False
    if len(pid_text) > 10 or len(start) > 20 or len(proof) != 64:
        return False
    if any(c not in "0123456789abcdef" for c in proof):
        return False
    pid = int(pid_text)
    if pid <= 0:
        return False
    from gateway.status import get_process_start_time

    if str(get_process_start_time(pid)) != start:
        return False
    try:
        conn = _open_socket()
        try:
            conn.settimeout(_TIMEOUT)
            conn.connect(_address(pid, start))
            peer_pid, uid, _ = _peer(conn)
            if peer_pid != pid or uid != os.getuid():
                return False
            conn.sendall(proof.encode() + payload)
            accepted = conn.recv(2) == b"1"
        finally:
            _close_socket(conn)
        return accepted and str(get_process_start_time(pid)) == start
    except OSError:
        return False
