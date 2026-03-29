from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .tools.read_file import read_file
from .workspaces import get_session_workspace

IGNORED_DIRS = {".git", ".venv", "__pycache__", ".shipyard", "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build"}
KEY_FILENAMES = {
    "README.md", "README.rst", "readme.md", "CODEAGENT.md", "CONTRIBUTING.md",
    "pyproject.toml", "setup.py", "setup.cfg", "package.json", "tsconfig.json",
    "Makefile", "makefile", "main.py", "index.py", "app.py", "__init__.py",
}
# Directories that commonly hold PRDs, specs, plans, or design docs
DOC_DIRS = {"plans", "docs", "specs", "design", "rfcs", "proposals", "prds"}
# File patterns that look like planning/spec docs
DOC_PATTERNS = {".prd.json", ".prd.md", ".spec.md", ".plan.md"}
MAX_FILE_TREE = 1000
MAX_SAMPLED_FILES = 12
MAX_FILE_CONTENT = 8000


def build_broad_context(session_id: str | None, instruction: str) -> dict[str, Any]:
    """
    Walk the repo and sample key files to give the LLM broad orientation before planning.
    Returns file_tree (all relative paths) and sampled_files (path -> content snippets).
    No LLM call — just surfaces structure so the planning call has richer context.
    """
    workspace = get_session_workspace(session_id).resolve()
    repo_root = Path.cwd().resolve()

    def _is_ignored(path: Path, root: Path) -> bool:
        """Check IGNORED_DIRS against path parts relative to root, not absolute parts."""
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            rel_parts = path.parts
        return any(part in IGNORED_DIRS for part in rel_parts)

    # Use workspace if it has files, else fall back to repo root
    if workspace.exists() and any(
        p for p in workspace.rglob("*") if p.is_file() and not _is_ignored(p, workspace)
    ):
        root = workspace
    else:
        root = repo_root

    file_tree: list[str] = []
    for path in sorted(root.rglob("*")):
        if _is_ignored(path, root):
            continue
        if path.is_file():
            file_tree.append(str(path.relative_to(root)))
        if len(file_tree) >= MAX_FILE_TREE:
            break

    # Sample key files: named matches first, then top-level files
    sampled_files: dict[str, str] = {}
    candidates: list[Path] = []

    for path in sorted(root.rglob("*")):
        if _is_ignored(path, root):
            continue
        if not path.is_file():
            continue
        if path.name in KEY_FILENAMES:
            candidates.insert(0, path)
        elif len(path.parts) - len(root.parts) <= 2:
            candidates.append(path)

    # Also prioritize any file explicitly named in the instruction
    mentioned = re.findall(r'[\w./\\-]+\.(?:py|js|ts|html|css|json|md|yaml|yml|toml|txt)', instruction)
    for name in mentioned:
        # Try relative to root
        p = root / name
        if p.is_file() and p not in candidates:
            candidates.insert(0, p)

    for path in candidates:
        if len(sampled_files) >= MAX_SAMPLED_FILES:
            break
        try:
            content = read_file(str(path))
            sampled_files[str(path.relative_to(root))] = content[:MAX_FILE_CONTENT]
        except (OSError, UnicodeDecodeError):
            continue

    # Discover planning/spec docs so the LLM knows they exist
    discovered_docs: list[str] = []
    for path in sorted(root.rglob("*")):
        if _is_ignored(path, root):
            continue
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        rel_parts = path.relative_to(root).parts
        # Files inside doc directories
        if rel_parts and rel_parts[0].lower() in DOC_DIRS:
            discovered_docs.append(rel)
        # Files matching doc patterns anywhere
        elif any(path.name.endswith(pat) for pat in DOC_PATTERNS):
            discovered_docs.append(rel)
        # Top-level markdown that looks like a spec/plan
        elif len(rel_parts) <= 2 and path.suffix.lower() == ".md" and path.name not in KEY_FILENAMES:
            discovered_docs.append(rel)
        if len(discovered_docs) >= 30:
            break

    git_status = _build_git_status(root)
    project_stack = detect_project_stack(root)

    return {
        "root": str(root),
        "file_tree": file_tree,
        "sampled_files": sampled_files,
        "discovered_docs": discovered_docs,
        "git_status": git_status,
        "project_stack": project_stack,
    }


def _build_git_status(root: Path) -> dict[str, Any]:
    """Return branch, dirty files, diff stat and a diff preview for the LLM."""
    def _git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    branch = _git(["branch", "--show-current"]).strip()
    if not branch:
        return {}

    status_output = _git(["status", "--short"])
    status_lines = [l for l in status_output.splitlines() if l.strip()]
    is_clean = len(status_lines) == 0

    diff_stat = _git(["diff", "--stat", "HEAD"]).strip()
    diff_full = _git(["diff", "HEAD"])

    return {
        "branch": branch,
        "is_clean": is_clean,
        "status_lines": status_lines,
        "diff_stat": diff_stat[:4000] if diff_stat else "",
        "diff_preview": diff_full[:8000] if diff_full else "",
    }


def detect_project_stack(root: Path) -> dict[str, Any]:
    """Detect the technology stack from project files at the given root.

    Returns a dict with keys like 'language', 'framework', 'package_manager'.
    Never raises — returns an empty dict on any failure.
    """
    stack: dict[str, Any] = {}
    try:
        pkg_json = root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                stack["package_manager"] = "npm"
                if "typescript" in deps or (root / "tsconfig.json").exists():
                    stack["language"] = "TypeScript"
                else:
                    stack["language"] = "JavaScript"
                for fw, key in [
                    ("React", "react"),
                    ("Next.js", "next"),
                    ("Vue", "vue"),
                    ("Svelte", "svelte"),
                    ("Angular", "@angular/core"),
                    ("Express", "express"),
                    ("Fastify", "fastify"),
                ]:
                    if key in deps:
                        stack["framework"] = fw
                        break
            except Exception:
                pass

        # Python detection
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "setup.cfg").exists():
            stack.setdefault("language", "Python")
            req_files = [root / "requirements.txt", root / "requirements-dev.txt"]
            for req_path in req_files:
                if req_path.exists():
                    try:
                        reqs = req_path.read_text().lower()
                        for fw, key in [("Django", "django"), ("Flask", "flask"), ("FastAPI", "fastapi")]:
                            if key in reqs:
                                stack.setdefault("framework", fw)
                                break
                    except Exception:
                        pass
                    break
    except Exception:
        pass
    return stack


def load_context_files(paths: list[str], max_files: int = 5, max_content: int = 3000) -> dict[str, str]:
    """
    Read a list of file paths and return their contents.
    Used by fetch_step_context to load context_files declared in a coarse action.
    """
    loaded: dict[str, str] = {}
    for raw_path in paths[:max_files]:
        path = Path(str(raw_path))
        if path.exists() and path.is_file():
            try:
                content = read_file(str(path))
                loaded[str(path)] = content[:max_content]
            except (OSError, UnicodeDecodeError):
                continue
    return loaded
