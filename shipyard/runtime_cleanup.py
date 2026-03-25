from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage_paths import (
    DATA_ROOT,
    LOGS_ROOT,
    SESSIONS_ROOT,
    SNAPSHOTS_ROOT,
    TRACES_ROOT,
    WORKSPACES_ROOT,
    ensure_dir,
)


def cleanup_runtime_data(
    keep_traces: int = 20,
    keep_snapshots: int = 20,
    keep_sessions: int = 20,
    keep_logs: int = 20,
    remove_empty_workspaces: bool = True,
    remove_empty_spec_dirs: bool = True,
) -> dict[str, Any]:
    trace_dir = ensure_dir(TRACES_ROOT)
    snapshot_dir = ensure_dir(SNAPSHOTS_ROOT)
    workspace_dir = ensure_dir(WORKSPACES_ROOT)
    session_dir = ensure_dir(SESSIONS_ROOT)
    log_dir = ensure_dir(LOGS_ROOT)
    specs_dir = ensure_dir(DATA_ROOT / "specs")

    removed_traces = _remove_oldest(trace_dir, keep_traces)
    removed_snapshots = _remove_oldest(snapshot_dir, keep_snapshots)
    removed_sessions = _remove_oldest(session_dir, keep_sessions)
    removed_logs = _remove_oldest(log_dir, keep_logs)
    removed_workspaces = 0
    if remove_empty_workspaces:
        for path in workspace_dir.iterdir():
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                removed_workspaces += 1
    removed_spec_dirs = 0
    if remove_empty_spec_dirs:
        for path in specs_dir.iterdir():
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                removed_spec_dirs += 1

    return {
        "removed_traces": removed_traces,
        "removed_snapshots": removed_snapshots,
        "removed_sessions": removed_sessions,
        "removed_logs": removed_logs,
        "removed_empty_workspaces": removed_workspaces,
        "removed_empty_spec_dirs": removed_spec_dirs,
    }


def _remove_oldest(directory: Path, keep: int) -> int:
    keep = max(0, keep)
    files = sorted((path for path in directory.iterdir() if path.is_file()), key=lambda p: p.stat().st_mtime)
    if len(files) <= keep:
        return 0
    removed = 0
    for path in files[: len(files) - keep]:
        path.unlink()
        removed += 1
    return removed
