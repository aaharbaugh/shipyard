from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitAutomationError(RuntimeError):
    """Raised when a git automation operation cannot be completed safely."""


class GitAutomation:
    def __init__(self, workdir: str = ".") -> None:
        self.workdir = Path(workdir).resolve()

    def get_status(self) -> dict[str, Any]:
        repo_root = self._discover_repo_root()
        branch = self._git(["branch", "--show-current"], repo_root).strip()
        status_output = self._git(["status", "--short"], repo_root)
        return {
            "repo_root": str(repo_root),
            "branch": branch,
            "is_clean": status_output.strip() == "",
            "status_lines": [line for line in status_output.splitlines() if line.strip()],
        }

    def create_branch(self, branch_name: str) -> dict[str, str]:
        if not branch_name.strip():
            raise GitAutomationError("Branch name cannot be empty.")

        repo_root = self._discover_repo_root()
        existing_branches = self._git(["branch", "--list", branch_name], repo_root).strip()
        if existing_branches:
            self._git(["switch", branch_name], repo_root)
            return {
                "repo_root": str(repo_root),
                "branch": branch_name,
                "action": "switched_existing",
            }

        self._git(["switch", "-c", branch_name], repo_root)
        return {
            "repo_root": str(repo_root),
            "branch": branch_name,
            "action": "created",
        }

    def commit(self, message: str, paths: list[str] | None = None) -> dict[str, str]:
        if not message.strip():
            raise GitAutomationError("Commit message cannot be empty.")

        repo_root = self._discover_repo_root()
        stage_paths = paths or ["."]
        self._git(["add", *stage_paths], repo_root)

        try:
            self._git(["commit", "-m", message], repo_root)
        except GitAutomationError as exc:
            if "nothing to commit" in str(exc).lower():
                raise GitAutomationError("No staged changes were available to commit.") from exc
            raise

        commit_sha = self._git(["rev-parse", "HEAD"], repo_root).strip()
        branch = self._git(["branch", "--show-current"], repo_root).strip()
        return {
            "repo_root": str(repo_root),
            "branch": branch,
            "commit": commit_sha,
            "message": message,
        }

    def _discover_repo_root(self) -> Path:
        try:
            output = self._git(
                ["rev-parse", "--show-toplevel"],
                self.workdir,
                discover=True,
            )
        except GitAutomationError as exc:
            raise GitAutomationError(
                f"No git repository found from {self.workdir}."
            ) from exc

        return Path(output.strip())

    def _git(self, args: list[str], cwd: Path, discover: bool = False) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            if discover and "not a git repository" in stderr.lower():
                raise GitAutomationError(stderr)
            raise GitAutomationError(stderr or "Git command failed.")
        return completed.stdout
