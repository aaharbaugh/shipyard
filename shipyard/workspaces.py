from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid

from .storage_paths import DATA_ROOT, GRAPH_RUNTIME_ROOT, WORKSPACES_ROOT, ensure_dir

WORKSPACE_ROOT = WORKSPACES_ROOT
LIVE_GRAPH_RUNTIME_ROOT = GRAPH_RUNTIME_ROOT
DEFAULT_WORKSPACE_NAME = "default"


def get_workspace_root() -> Path:
    return ensure_dir(WORKSPACE_ROOT)


def get_live_graph_runtime_root() -> Path:
    return ensure_dir(LIVE_GRAPH_RUNTIME_ROOT)


def create_temp_workspace(prefix: str = "run") -> Path:
    root = get_workspace_root()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        name = f"{prefix}-{stamp}" if suffix == 0 else f"{prefix}-{stamp}-{suffix}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def get_session_workspace(session_id: str | None = None) -> Path:
    root = get_workspace_root()
    name = DEFAULT_WORKSPACE_NAME
    workspace = root / name
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def create_temp_target_path(
    session_id: str | None = None,
    filename: str = "file.py",
    unique: bool = False,
) -> Path:
    workspace = get_session_workspace(session_id)
    target = workspace / filename
    if not unique and not target.exists():
        return target
    if not unique:
        return target

    suffix = target.suffix
    stem = target.stem
    while True:
        candidate = workspace / f"{stem}-{uuid.uuid4().hex[:6]}{suffix}"
        if not candidate.exists():
            return candidate


def get_workspace_status() -> dict[str, str | bool]:
    workspace_root = get_workspace_root()
    live_graph_root = get_live_graph_runtime_root()
    return {
        "data_root": str(ensure_dir(DATA_ROOT).resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "workspace_exists": workspace_root.exists(),
        "live_graph_runtime_root": str(live_graph_root.resolve()),
        "live_graph_runtime_exists": live_graph_root.exists(),
    }
