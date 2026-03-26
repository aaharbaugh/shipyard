from __future__ import annotations

from pathlib import Path

from .workspaces import get_session_workspace


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

    workspace = get_session_workspace(session_id).resolve()
    if workspace.exists():
        workspace_files = _collect_relative_files(workspace, workspace, max_workspace_files)
        if workspace_files:
            lines.append(f"Session workspace: {workspace.name}")
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
