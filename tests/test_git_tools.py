from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from shipyard.tools.git_tools import GitAutomation, GitAutomationError


class GitAutomationTests(unittest.TestCase):
    def test_status_raises_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            automation = GitAutomation(tmpdir)
            with self.assertRaises(GitAutomationError):
                automation.get_status()

    def test_branch_and_commit_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            self._git(["init"], repo)
            self._git(["config", "user.name", "Shipyard Test"], repo)
            self._git(["config", "user.email", "shipyard@example.com"], repo)

            file_path = repo / "demo.txt"
            file_path.write_text("hello\n", encoding="utf-8")

            automation = GitAutomation(tmpdir)
            commit_result = automation.commit("Initial commit")
            self.assertEqual(commit_result["message"], "Initial commit")

            branch_result = automation.create_branch("feature/test-branch")
            self.assertEqual(branch_result["branch"], "feature/test-branch")

            file_path.write_text("updated\n", encoding="utf-8")
            second_commit = automation.commit("Update demo file", ["demo.txt"])
            self.assertEqual(second_commit["branch"], "feature/test-branch")

            status = automation.get_status()
            self.assertTrue(status["is_clean"])
            self.assertEqual(status["branch"], "feature/test-branch")

    def _git(self, args: list[str], cwd: Path) -> None:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
