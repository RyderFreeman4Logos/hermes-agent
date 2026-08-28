"""CLI adapters for compression trace export and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _db(args: argparse.Namespace) -> Path:
    if getattr(args, "db", None):
        return Path(args.db)
    from hermes_state import DEFAULT_DB_PATH
    return Path(DEFAULT_DB_PATH)


def cmd_compression_trace(args: argparse.Namespace) -> int:
    from hermes_cli.compression_trace import export_compression_trace
    if args.redaction == "none":
        print("WARNING: --redaction none is local-only and may expose credentials", file=sys.stderr)
    manifest = export_compression_trace(_db(args), args.session_id, Path(args.output), redaction=args.redaction)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def cmd_compression_sessions(args: argparse.Namespace) -> int:
    from hermes_cli.compression_trace import discover_compression_sessions, inspect_compression_session
    if args.session_id:
        value = inspect_compression_session(_db(args), args.session_id)
    else:
        value = discover_compression_sessions(_db(args), min_compressions=args.min_compressions)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def cmd_compression_replay(args: argparse.Namespace) -> int:
    from hermes_cli.compression_replay import ReplayConfig, ReplayRunner
    config = ReplayConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning,
        fallback_policy=args.fallback_policy,
        execute_real_tools=args.execute_real_tools,
        timeout=args.timeout,
        repetitions=args.repetitions,
    )
    result = ReplayRunner(Path(args.corpus), config).run(mode=args.mode, output=Path(args.output) if args.output else None, rerun=args.rerun)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0
