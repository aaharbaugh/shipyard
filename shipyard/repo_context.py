from __future__ import annotations

from pathlib import Path

from .planning_hints import extract_explicit_filenames
from .workspaces import get_session_workspace, get_session_workspace_selection


IGNORED_NAMES = {".git", ".venv", "__pycache__"}


def build_repo_context_lines(
    session_id: str | None,
    target_path: str | None = None,
    max_top_level: int = 12,
    max_workspace_files: int = 20,
    max_target_siblings: int = 12,
) -> list[str]:
    repo_root = Path.cwd().resolve()
    lines: list[str] = [f"Repository root: {repo_root.name}"]

    top_level: list[str] = []
    for child in sorted(repo_root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.name in IGNORED_NAMES:
            continue
        if child.name == ".shipyard":
            top_level.append(".shipyard/")
            continue
        top_level.append(f"{child.name}/" if child.is_dir() else child.name)
        if len(top_level) >= max_top_level:
            break
    if top_level:
        lines.append("Top-level tree: " + ", ".join(top_level))

    workspace_selection = get_session_workspace_selection(session_id)
    workspace = get_session_workspace(session_id).resolve()
    if workspace.exists():
        workspace_files = _collect_relative_files(workspace, workspace, max_workspace_files)
        if workspace_files:
            label = str(workspace_selection.get("workspace_label") or workspace.name)
            lines.append(f"Session workspace: {label}")
            lines.append("Workspace files: " + ", ".join(workspace_files))

    if target_path:
        target = Path(target_path).resolve()
        if target.exists():
            lines.append(f"Resolved target: {target.name}")
            if target.parent.exists():
                sibling_files = []
                for child in sorted(target.parent.iterdir(), key=lambda item: item.name.lower()):
                    if child.name == target.name:
                        continue
                    sibling_files.append(child.name + ("/" if child.is_dir() else ""))
                    if len(sibling_files) >= max_target_siblings:
                        break
                if sibling_files:
                    lines.append("Target siblings: " + ", ".join(sibling_files))

    return lines


def build_existing_file_context_lines(
    session_id: str | None,
    instruction: str,
    max_files: int = 4,
    max_chars: int = 700,
) -> list[str]:
    workspace = get_session_workspace(session_id).resolve()
    lines: list[str] = []
    for name in extract_explicit_filenames(instruction)[:max_files]:
        candidate = workspace / name
        if not candidate.exists() or not candidate.is_file():
            continue
        content = candidate.read_text(encoding="utf-8")
        snippet = content[:max_chars]
        if len(content) > max_chars:
            snippet = snippet.rstrip() + "\n..."
        lines.append(f"Existing file: {name}")
        lines.append(snippet)
    return lines


def any_explicit_files_exist(session_id: str | None, instruction: str) -> bool:
    workspace = get_session_workspace(session_id).resolve()
    return any((workspace / name).exists() for name in extract_explicit_filenames(instruction))


def _collect_relative_files(root: Path, base: Path, limit: int) -> list[str]:
    items: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if path.name in IGNORED_NAMES:
            continue
        if path.is_file():
            items.append(str(path.relative_to(base)))
            if len(items) >= limit:
                break
    return items
