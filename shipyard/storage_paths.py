from __future__ import annotations

from pathlib import Path


SHIPYARD_ROOT = Path(".shipyard")
DATA_ROOT = SHIPYARD_ROOT / "data"
LOGS_ROOT = DATA_ROOT / "logs"
SESSIONS_ROOT = DATA_ROOT / "sessions"
TRACES_ROOT = DATA_ROOT / "traces"
SNAPSHOTS_ROOT = DATA_ROOT / "snapshots"
WORKSPACES_ROOT = DATA_ROOT / "workspace"
GRAPH_ROOT = DATA_ROOT / "graph"
GRAPH_INDEX_ROOT = GRAPH_ROOT / "index"
GRAPH_RUNTIME_ROOT = GRAPH_ROOT / "runtime"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
