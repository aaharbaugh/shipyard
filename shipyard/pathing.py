from __future__ import annotations

from pathlib import Path
from typing import Any

from .storage_paths import DATA_ROOT
from .workspaces import get_session_workspace


def _resolve_testing_mode_path(raw_path: str, session_id: str | None) -> tuple[str, str]:
    path = Path(raw_path)
    workspace = get_session_workspace(session_id).resolve()
    data_root = DATA_ROOT.resolve()
    repo_root = Path.cwd().resolve()

    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(data_root)
            return str(resolved), "explicit_target_path"
        except ValueError:
            return str((workspace / path.name).resolve()), "sandboxed_target_path"

    shipyard_relative = (repo_root / path).resolve()
    try:
        shipyard_relative.relative_to(data_root)
        return str(shipyard_relative), "explicit_target_path"
    except ValueError:
        pass

    return str((workspace / path).resolve()), "sandboxed_target_path"


def resolve_target_path(
    explicit_target_path: str | None,
    context: dict[str, Any] | None,
    edit_mode: str,
    session_id: str | None = None,
    instruction: str | None = None,
) -> tuple[str | None, str]:
    context = context or {}
    testing_mode = bool(context.get("testing_mode"))

    if explicit_target_path:
        if testing_mode:
            return _resolve_testing_mode_path(str(explicit_target_path), session_id)
        return explicit_target_path, "explicit_target_path"

    file_hint = context.get("file_hint")
    if file_hint:
        if testing_mode:
            return _resolve_testing_mode_path(str(file_hint), session_id)
        return str(file_hint), "file_hint"

    return None, "unresolved"
