from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.tools.code_graph import inspect_code_graph_artifacts, inspect_code_graph_status, index_code_graph


class CodeGraphToolTests(unittest.TestCase):
    def test_inspect_code_graph_artifacts_reports_existing_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proto_dir = Path(tmpdir) / ".shipyard" / "data" / "graph" / "index"
            proto_dir.mkdir(parents=True, exist_ok=True)
            (proto_dir / "nodes.bin").write_text("demo", encoding="utf-8")

            result = inspect_code_graph_artifacts(tmpdir)

            self.assertTrue(result["exists"])
            self.assertTrue(result["has_index"])
            self.assertIn(".shipyard/data/graph/index/nodes.bin", result["files"])

    def test_inspect_code_graph_status_uses_env_override(self) -> None:
        with patch.dict("os.environ", {"SHIPYARD_ASSUME_CODE_GRAPH_READY": "1"}):
            result = inspect_code_graph_status()

        self.assertTrue(result["ready"])
        self.assertEqual(result["source"], "env_override")
        self.assertIn("index_state", result)

    def test_index_code_graph_reports_missing_cli(self) -> None:
        with patch("shipyard.tools.code_graph._find_cgr_executable", return_value=None):
            result = index_code_graph()

        self.assertFalse(result["ok"])
        self.assertIn("cgr", result["reason"])

    def test_inspect_code_graph_artifacts_marks_stale_when_source_is_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proto_dir = root / ".shipyard" / "data" / "graph" / "index"
            proto_dir.mkdir(parents=True, exist_ok=True)
            artifact = proto_dir / "nodes.bin"
            source = root / "demo.py"

            artifact.write_text("artifact", encoding="utf-8")
            os.utime(artifact, (100, 100))
            source.write_text("print('newer')\n", encoding="utf-8")
            os.utime(source, (200, 200))

            result = inspect_code_graph_artifacts(tmpdir)

            self.assertTrue(result["stale"])
            self.assertIsNotNone(result["latest_artifact_at"])
            self.assertIsNotNone(result["latest_source_at"])

    def test_inspect_code_graph_status_marks_empty_live_graph_not_ready(self) -> None:
        mock_completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "Node Statistics\nTotal Nodes 0\nRelationship Statistics\nTotal Relationships 0\n",
                "stderr": "",
            },
        )()

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.tools.code_graph._find_cgr_executable",
            return_value="/tmp/fake-cgr",
        ), patch(
            "shipyard.tools.code_graph.subprocess.run",
            return_value=mock_completed,
        ):
            result = inspect_code_graph_status(tmpdir)

        self.assertFalse(result["ready"])
        self.assertTrue(result["available"])
        self.assertEqual(result["reason"], "Memgraph is reachable, but the live graph is still empty.")
        self.assertEqual(result["live_graph_state"]["node_count"], 0)
        self.assertFalse(result["live_graph_state"]["populated"])

    def test_inspect_code_graph_status_parses_ansi_wrapped_counts(self) -> None:
        mock_completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": (
                    "\x1b[1;32mTotal Nodes\x1b[0m 25\n"
                    "\x1b[1;32mTotal Relationships\x1b[0m 24\n"
                ),
                "stderr": "",
            },
        )()

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.tools.code_graph._find_cgr_executable",
            return_value="/tmp/fake-cgr",
        ), patch(
            "shipyard.tools.code_graph.subprocess.run",
            return_value=mock_completed,
        ):
            result = inspect_code_graph_status(tmpdir)

        self.assertTrue(result["ready"])
        self.assertEqual(result["live_graph_state"]["node_count"], 25)
        self.assertEqual(result["live_graph_state"]["relationship_count"], 24)
        self.assertTrue(result["live_graph_state"]["populated"])


if __name__ == "__main__":
    unittest.main()
