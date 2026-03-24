from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shipyard.graph import build_graph


class GraphFlowTests(unittest.TestCase):
    def test_graph_completes_successful_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            path.write_text('print("old")\n', encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": 'replace "old" with "new"',
                    "proposal_mode": "heuristic",
                    "context": {"file_hint": str(path)},
                    "verification_commands": [f"python3 -m py_compile {path}"],
                }
            )

            self.assertEqual(result["status"], "verified")
            self.assertEqual(path.read_text(encoding="utf-8"), 'print("new")\n')
            self.assertEqual(result["helper_output"]["provider"], "heuristic")
            self.assertTrue(result["snapshot_path"])

    def test_graph_blocks_missing_anchor_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            original = 'print("original")\n'
            path.write_text(original, encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "blocked path",
                    "target_path": str(path),
                    "anchor": 'print("missing")',
                    "replacement": 'print("new")',
                    "proposal_mode": "heuristic",
                }
            )

            self.assertEqual(result["status"], "edit_blocked")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertIn("Anchor was not found", result["error"])

    def test_graph_reverts_after_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            original = 'print("safe")\n'
            path.write_text(original, encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "force bad edit",
                    "target_path": str(path),
                    "anchor": 'print("safe")',
                    "replacement": 'print("broken)',
                    "proposal_mode": "heuristic",
                    "verification_commands": [f"python3 -m py_compile {path}"],
                    "max_edit_attempts": 1,
                }
            )

            self.assertEqual(result["status"], "failed_after_retries")
            self.assertTrue(result["reverted_to_snapshot"])
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
