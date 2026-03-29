from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

from .storage_paths import DATA_ROOT, GRAPH_RUNTIME_ROOT, WORKSPACES_ROOT, ensure_dir

WORKSPACE_ROOT = WORKSPACES_ROOT
LIVE_GRAPH_RUNTIME_ROOT = GRAPH_RUNTIME_ROOT
DEFAULT_WORKSPACE_NAME = "default"
WORKSPACE_SELECTIONS_PATH = DATA_ROOT / "workspace_selections.json"
IGNORED_REPO_DIRS = {".git", ".venv", ".shipyard", "__pycache__", "node_modules"}


def get_workspace_root() -> Path:
    return ensure_dir(WORKSPACE_ROOT)


def get_live_graph_runtime_root() -> Path:
    return ensure_dir(LIVE_GRAPH_RUNTIME_ROOT)


def get_workspace_registry_path() -> Path:
    ensure_dir(DATA_ROOT)
    return WORKSPACE_SELECTIONS_PATH


def normalize_repo_workspace_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    repo_root = Path.cwd().resolve()
    candidate = Path(text)
    resolved = (repo_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.exists() or not resolved.is_dir():
        return None
    return resolved


def _read_workspace_registry() -> dict[str, str]:
    path = get_workspace_registry_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if value}


def _write_workspace_registry(registry: dict[str, str]) -> None:
    path = get_workspace_registry_path()
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def set_session_workspace(session_id: str | None, workspace_path: str | None) -> dict[str, str | bool | None]:
    normalized = normalize_repo_workspace_path(workspace_path)
    if not session_id:
        return {
            "mode": "managed",
            "workspace_path": None,
            "workspace_label": DEFAULT_WORKSPACE_NAME,
            "managed": True,
            "exists": True,
        }

    registry = _read_workspace_registry()
    if normalized is None:
        registry.pop(str(session_id), None)
        _write_workspace_registry(registry)
        workspace = get_managed_workspace()
        return {
            "mode": "managed",
            "workspace_path": None,
            "workspace_label": workspace.name,
            "managed": True,
            "exists": True,
            "resolved_path": str(workspace.resolve()),
        }

    repo_root = Path.cwd().resolve()
    try:
        relative = str(normalized.relative_to(repo_root)) if normalized != repo_root else "."
    except ValueError:
        # External workspace — store absolute path
        relative = str(normalized)
    registry[str(session_id)] = relative
    _write_workspace_registry(registry)
    return {
        "mode": "repo_folder",
        "workspace_path": relative,
        "workspace_label": relative,
        "managed": False,
        "exists": True,
        "resolved_path": str(normalized),
    }


def get_session_workspace_selection(session_id: str | None = None) -> dict[str, str | bool | None]:
    managed = get_managed_workspace()
    if not session_id:
        return {
            "mode": "managed",
            "workspace_path": None,
            "workspace_label": managed.name,
            "managed": True,
            "exists": True,
            "resolved_path": str(managed.resolve()),
        }

    registry = _read_workspace_registry()
    raw_path = registry.get(str(session_id))
    normalized = normalize_repo_workspace_path(raw_path)
    if normalized is None:
        if raw_path:
            registry.pop(str(session_id), None)
            _write_workspace_registry(registry)
        return {
            "mode": "managed",
            "workspace_path": None,
            "workspace_label": managed.name,
            "managed": True,
            "exists": True,
            "resolved_path": str(managed.resolve()),
        }

    repo_root = Path.cwd().resolve()
    try:
        relative = str(normalized.relative_to(repo_root)) if normalized != repo_root else "."
    except ValueError:
        relative = str(normalized)
    return {
        "mode": "repo_folder",
        "workspace_path": relative,
        "workspace_label": relative,
        "managed": False,
        "exists": True,
        "resolved_path": str(normalized),
    }


def list_repo_workspace_folders(max_depth: int = 3) -> list[dict[str, str | bool]]:
    repo_root = Path.cwd().resolve()
    items: list[dict[str, str | bool]] = [
        {
            "path": ".",
            "label": ".",
            "resolved_path": str(repo_root),
            "managed": False,
        }
    ]
    for path in sorted(repo_root.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_dir():
            continue
        if any(part in IGNORED_REPO_DIRS for part in path.relative_to(repo_root).parts):
            continue
        depth = len(path.relative_to(repo_root).parts)
        if depth > max_depth:
            continue
        relative = str(path.relative_to(repo_root))
        items.append(
            {
                "path": relative,
                "label": relative,
                "resolved_path": str(path.resolve()),
                "managed": False,
            }
        )
    return items


def get_managed_workspace() -> Path:
    root = get_workspace_root()
    workspace = root / DEFAULT_WORKSPACE_NAME
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


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


def get_session_workspace(session_id: str | None = None, workspace_path: str | None = None) -> Path:
    normalized = normalize_repo_workspace_path(workspace_path)
    if normalized is not None:
        return normalized

    selection = get_session_workspace_selection(session_id)
    resolved_path = selection.get("resolved_path")
    if selection.get("mode") == "repo_folder" and resolved_path:
        return Path(str(resolved_path))

    return get_managed_workspace()


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
        "managed_workspace": str(get_managed_workspace().resolve()),
    }
