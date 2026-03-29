from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.workspaces import (
    create_temp_workspace,
    get_managed_workspace,
    get_live_graph_runtime_root,
    get_session_workspace_selection,
    get_session_workspace,
    get_workspace_root,
    get_workspace_status,
    create_temp_target_path,
    list_repo_workspace_folders,
    set_session_workspace,
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
        ), patch("shipyard.workspaces.WORKSPACE_SELECTIONS_PATH", Path(tmpdir) / "workspace-selections.json"):
            first = get_session_workspace("abc123")
            second = get_session_workspace("abc123")

            self.assertEqual(first, second)
            self.assertTrue(first.exists())
            self.assertEqual(first.name, "default")

    def test_session_workspace_can_bind_to_repo_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ), patch("shipyard.workspaces.WORKSPACE_SELECTIONS_PATH", Path(tmpdir) / "workspace-selections.json"), patch(
            "pathlib.Path.cwd",
            return_value=Path(tmpdir),
        ):
            folder = Path(tmpdir) / "demo"
            folder.mkdir()

            result = set_session_workspace("abc123", "demo")

            self.assertEqual(result["mode"], "repo_folder")
            self.assertEqual(get_session_workspace("abc123"), folder.resolve())
            self.assertEqual(get_session_workspace_selection("abc123")["workspace_path"], "demo")

    def test_temp_target_path_reuses_session_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            target = create_temp_target_path(session_id="abc123")

            self.assertEqual(target, Path(tmpdir) / "workspaces" / "default" / "file.py")

    def test_temp_target_path_uses_selected_repo_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ), patch("shipyard.workspaces.WORKSPACE_SELECTIONS_PATH", Path(tmpdir) / "workspace-selections.json"), patch(
            "pathlib.Path.cwd",
            return_value=Path(tmpdir),
        ):
            folder = Path(tmpdir) / "app"
            folder.mkdir()
            set_session_workspace("abc123", "app")

            target = create_temp_target_path(session_id="abc123")

            self.assertEqual(target, folder / "file.py")

    def test_temp_target_path_can_generate_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "file.py").write_text("", encoding="utf-8")

            target = create_temp_target_path(session_id="abc123", unique=True)

            self.assertNotEqual(target.name, "file.py")
            self.assertTrue(target.name.startswith("file-"))

    def test_workspace_status_reports_managed_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            status = get_workspace_status()

            self.assertEqual(status["managed_workspace"], str(get_managed_workspace().resolve()))

    def test_list_repo_workspace_folders_ignores_shipyard_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / ".shipyard").mkdir()

            folders = list_repo_workspace_folders()
            paths = [item["path"] for item in folders]

            self.assertIn(".", paths)
            self.assertIn("src", paths)
            self.assertNotIn(".shipyard", paths)


if __name__ == "__main__":
    unittest.main()
