from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage_paths import DATA_ROOT
from .workspaces import get_session_workspace


def _resolve_testing_mode_path(raw_path: str, session_id: str | None) -> tuple[str, str]:
    return _resolve_testing_mode_path_with_context(raw_path, session_id, None)


def resolve_target_path(
    explicit_target_path: str | None,
    context: dict[str, Any] | None,
    edit_mode: str,
    session_id: str | None = None,
    instruction: str | None = None,
) -> tuple[str | None, str]:
    context = context or {}
    testing_mode = bool(context.get("testing_mode"))
    workspace_path = context.get("workspace_path")

    if explicit_target_path:
        if testing_mode:
            return _resolve_testing_mode_path_with_context(str(explicit_target_path), session_id, workspace_path)
        return explicit_target_path, "explicit_target_path"

    file_hint = context.get("file_hint")
    if file_hint:
        if testing_mode:
            return _resolve_testing_mode_path_with_context(str(file_hint), session_id, workspace_path)
        return str(file_hint), "file_hint"

    return None, "unresolved"


def _resolve_testing_mode_path_with_context(
    raw_path: str,
    session_id: str | None,
    workspace_path: str | None,
) -> tuple[str, str]:
    path = Path(raw_path)
    workspace = get_session_workspace(session_id, workspace_path).resolve()
    data_root = DATA_ROOT.resolve()
    repo_root = Path.cwd().resolve()

    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(data_root)
            return str(resolved), "explicit_target_path"
        except ValueError:
            try:
                resolved.relative_to(repo_root)
                return str(resolved), "explicit_target_path"
            except ValueError:
                return str((workspace / path.name).resolve()), "sandboxed_target_path"

    shipyard_relative = (repo_root / path).resolve()
    try:
        shipyard_relative.relative_to(data_root)
        return str(shipyard_relative), "explicit_target_path"
    except ValueError:
        pass

    try:
        shipyard_relative.relative_to(repo_root)
        if workspace == repo_root or workspace in shipyard_relative.parents or shipyard_relative == workspace:
            return str(shipyard_relative), "explicit_target_path"
    except ValueError:
        pass

    return str((workspace / path).resolve()), "sandboxed_target_path"
