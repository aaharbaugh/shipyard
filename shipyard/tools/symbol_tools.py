from __future__ import annotations

from pathlib import Path
from typing import Any


def find_symbol_offset(file_path: str, symbol_name: str, occurrence: int = 1) -> int | None:
    """
    Return the byte offset of the nth occurrence of symbol_name in the file.
    Used to give rope a cursor position for rename.
    Returns None if not found.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    search = symbol_name
    pos = 0
    for _ in range(occurrence):
        idx = content.find(search, pos)
        if idx == -1:
            return None
        pos = idx + 1
    return idx


def rename_symbol_rope(
    repo_root: str,
    target_path: str,
    old_name: str,
    new_name: str,
    offset: int | None = None,
) -> dict[str, Any]:
    """
    Rename a symbol across a rope project rooted at repo_root.
    target_path is the file containing the definition.
    offset is the byte offset of the symbol in target_path (auto-detected if None).
    Returns {"changed_files": [...], "error": None} on success,
            {"changed_files": [], "error": str} on failure.
    """
    try:
        from rope.base.project import Project
        from rope.refactor.rename import Rename
    except ImportError:
        return {"changed_files": [], "error": "rope is not installed"}

    if offset is None:
        offset = find_symbol_offset(target_path, old_name)
        if offset is None:
            return {"changed_files": [], "error": f"Symbol {old_name!r} not found in {target_path}"}

    project = None
    try:
        project = Project(repo_root, ropefolder=None)
        resource = project.get_resource(str(Path(target_path).relative_to(repo_root)))
        renamer = Rename(project, resource, offset)
        changes = renamer.get_changes(new_name)
        changed_files: list[str] = []
        for change in changes.changes:
            # Each change has a resource attribute
            changed_files.append(str(Path(repo_root) / change.resource.path))
        project.do(changes)
        return {"changed_files": changed_files, "error": None}
    except Exception as exc:
        return {"changed_files": [], "error": str(exc)}
    finally:
        if project is not None:
            try:
                project.close()
            except Exception:
                pass
