from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.workspaces import (
    create_temp_workspace,
    get_live_graph_runtime_root,
    get_session_workspace,
    get_workspace_root,
    get_workspace_status,
    create_temp_target_path,
)


class WorkspaceTests(unittest.TestCase):
    def test_create_temp_workspace_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = create_temp_workspace("demo")

            self.assertTrue(workspace.exists())
            self.assertTrue(workspace.is_dir())
            self.assertEqual(workspace.parent, Path(tmpdir) / "workspaces")

    def test_workspace_status_reports_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ), patch(
            "shipyard.workspaces.LIVE_GRAPH_RUNTIME_ROOT",
            Path(tmpdir) / "runtime" / "live_graph",
        ):
            status = get_workspace_status()

            self.assertTrue(status["workspace_exists"])
            self.assertTrue(status["live_graph_runtime_exists"])

    def test_live_graph_runtime_root_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.LIVE_GRAPH_RUNTIME_ROOT",
            Path(tmpdir) / "runtime" / "live_graph",
        ):
            root = get_live_graph_runtime_root()

            self.assertTrue(root.exists())
            self.assertTrue(root.is_dir())

    def test_workspace_root_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            root = get_workspace_root()

            self.assertTrue(root.exists())
            self.assertTrue(root.is_dir())

    def test_session_workspace_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            first = get_session_workspace("abc123")
            second = get_session_workspace("abc123")

            self.assertEqual(first, second)
            self.assertTrue(first.exists())

    def test_temp_target_path_reuses_session_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            target = create_temp_target_path(session_id="abc123")

            self.assertEqual(target, Path(tmpdir) / "workspaces" / "abc123" / "scratch.py")

    def test_temp_target_path_can_generate_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "abc123"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "scratch.py").write_text("", encoding="utf-8")

            target = create_temp_target_path(session_id="abc123", unique=True)

            self.assertNotEqual(target.name, "scratch.py")
            self.assertTrue(target.name.startswith("scratch-"))


if __name__ == "__main__":
    unittest.main()
