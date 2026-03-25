from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from shipyard.main import _maybe_sync_graph, _run_action_plan, _should_skip_graph_sync, main, parse_user_input, read_user_input
from shipyard.main import _attach_file_outcome


class MainInputTests(unittest.TestCase):
    def test_parse_user_input_accepts_multiline_json(self) -> None:
        payload = """{
  "instruction": "replace \\"before\\" with \\"after\\"",
  "proposal_mode": "heuristic",
  "context": {
    "file_hint": "/tmp/demo.py"
  }
}"""

        result = parse_user_input(payload)

        self.assertEqual(result["instruction"], 'replace "before" with "after"')
        self.assertEqual(result["proposal_mode"], "heuristic")
        self.assertEqual(result["context"]["file_hint"], "/tmp/demo.py")

    def test_read_user_input_collects_multiline_json(self) -> None:
        lines = [
            "{",
            '  "instruction": "replace \\"before\\" with \\"after\\"",',
            '  "context": {"file_hint": "/tmp/demo.py"}',
            "}",
        ]

        with patch("builtins.input", side_effect=lines):
            raw = read_user_input()

        parsed = json.loads(raw)
        self.assertEqual(parsed["instruction"], 'replace "before" with "after"')
        self.assertEqual(parsed["context"]["file_hint"], "/tmp/demo.py")

    def test_main_handles_invalid_json_without_crashing(self) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["shipyard.main"]), patch(
            "builtins.input",
            side_effect=["{", "", "exit"],
        ), patch("shipyard.main.build_graph", return_value=Mock()), patch(
            "shipyard.main.SessionStore", return_value=Mock()
        ), redirect_stdout(output):
            main()

        rendered = output.getvalue()
        self.assertIn("input_error=Invalid JSON", rendered)
        self.assertIn("hint=Paste the full JSON block", rendered)
        self.assertIn("Stopping Shipyard MVP runner.", rendered)

    def test_main_prints_proposal_summary(self) -> None:
        output = io.StringIO()
        fake_app = Mock()
        fake_app.invoke.return_value = {
            "status": "verified",
            "proposal_summary": {
                "provider": "heuristic",
                "is_valid": True,
                "edit_mode": "anchor",
            },
        }
        fake_store = Mock()

        with patch("sys.argv", ["shipyard.main"]), patch(
            "builtins.input",
            side_effect=["replace \"a\" with \"b\"", "exit"],
        ), patch("shipyard.main.build_graph", return_value=fake_app), patch(
            "shipyard.main.SessionStore", return_value=fake_store
        ), patch(
            "shipyard.main.write_trace", return_value=".shipyard/data/traces/demo.json"
        ), redirect_stdout(output):
            main()

        rendered = output.getvalue()
        self.assertIn("proposal_summary:", rendered)
        self.assertIn('"provider": "heuristic"', rendered)

    def test_maybe_sync_graph_skips_internal_runtime_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch("shipyard.main.Path.cwd", return_value=Path(tmpdir)):
            target = Path(tmpdir) / ".shipyard" / "data" / "graph" / "runtime" / "scratch.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            result = _maybe_sync_graph({"status": "edited", "target_path": str(target)})

        self.assertFalse(result["attempted"])
        self.assertIn("internal Shipyard runtime data", result["reason"])

    def test_should_skip_graph_sync_for_trivial_testing_mode_write(self) -> None:
        self.assertTrue(
            _should_skip_graph_sync(
                {
                    "instruction": "make a new file",
                    "edit_mode": "write_file",
                    "context": {"testing_mode": True},
                }
            )
        )

    def test_attach_file_outcome_adds_preview_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "demo.txt"
            target.write_text("hello world", encoding="utf-8")

            result = _attach_file_outcome({"status": "edited", "target_path": str(target)})

        self.assertEqual(result["changed_files"], [str(target.resolve())])
        self.assertEqual(result["file_preview"], "hello world")
        self.assertIn("content_hash", result)

    def test_attach_file_outcome_preserves_existing_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "demo.txt"
            target.write_text("hello world", encoding="utf-8")
            other = Path(tmpdir) / "copy.txt"
            other.write_text("hello world", encoding="utf-8")

            result = _attach_file_outcome(
                {
                    "status": "edited",
                    "target_path": str(target),
                    "changed_files": [str(other.resolve())],
                }
            )

        self.assertEqual(result["changed_files"], [str(other.resolve())])
        self.assertEqual(result["file_preview"], "hello world")

    def test_run_action_plan_executes_each_step_and_merges_changed_files(self) -> None:
        app = Mock()
        app.invoke.side_effect = [
            {"status": "edited", "changed_files": ["/tmp/file1.txt"]},
            {"status": "edited", "changed_files": ["/tmp/file2.txt"]},
        ]

        result = _run_action_plan(
            app,
            {"session_id": "demo", "instruction": "create 2 files and write hello"},
            {
                "actions": [
                    {"instruction": "create 2 files"},
                    {"instruction": "write hello"},
                ]
            },
        )

        self.assertEqual(app.invoke.call_count, 2)
        self.assertEqual(result["changed_files"], ["/tmp/file1.txt", "/tmp/file2.txt"])
        self.assertEqual(result["instruction_steps"], ["create 2 files", "write hello"])


if __name__ == "__main__":
    unittest.main()
