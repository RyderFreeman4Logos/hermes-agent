"""Shared validator for an explicitly configured test scratch root."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from hermes_constants import SOCKET_TMPDIR_MAX_LEN

_ENV = "HERMES_TEST_TMP_ROOT"
# A worker sets TMPDIR to ``<root>/r-<8-char-token>`` before test imports.
_CHILD_TEMP_COMPONENT_BYTES = len(os.fsencode("/r-")) + 8


def configured_test_temp_root() -> Path | None:
    """Return the validated opt-in root; leave the canonical defaults untouched."""
    raw = os.environ.get(_ENV, "")
    if not raw:
        return None
    getuid = getattr(os, "getuid", None)
    if os.name == "nt" or not callable(getuid):
        raise ValueError(f"{_ENV} ownership validation is supported on POSIX only")

    root = Path(raw)
    if not root.is_absolute():
        raise ValueError(f"{_ENV} must be an absolute path")
    if len(os.fsencode(str(root))) + _CHILD_TEMP_COMPONENT_BYTES > SOCKET_TMPDIR_MAX_LEN:
        raise ValueError(
            f"{_ENV} leaves no room for the runner's per-file directory within "
            f"the {SOCKET_TMPDIR_MAX_LEN}-byte socket-safe limit"
        )
    if any(part.startswith(".") for part in root.parts if part not in (root.anchor, ".", "..")):
        raise ValueError(f"{_ENV} must not contain hidden path components")

    try:
        resolved = root.resolve(strict=True)
        root_stat = root.stat()
        ancestors = [(path, path.stat()) for path in root.parents]
    except OSError as exc:
        raise ValueError(f"{_ENV} must name an existing real directory: {root}: {exc}") from exc
    if resolved != root:
        raise ValueError(f"{_ENV} must not use a symlinked or non-canonical path: {root}")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"{_ENV} must name a directory: {root}")
    if root_stat.st_uid != getuid():
        raise ValueError(f"{_ENV} must be owned by the current user: {root}")
    if stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise ValueError(f"{_ENV} must have mode 0700: {root}")
    for path, info in ancestors:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{_ENV} has a non-directory ancestor: {path}")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not info.st_mode & stat.S_ISVTX:
            raise ValueError(f"{_ENV} has an unsafe writable ancestor: {path}")
    if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(f"{_ENV} is not readable, writable, and searchable: {root}")
    return root
