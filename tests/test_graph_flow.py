from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            self.assertEqual(result["helper_output"]["helper_agent"]["agent_name"], "helper-planner")
            self.assertEqual(result["helper_output"]["helper_agent"]["delegation_mode"], "sequential")
            self.assertEqual(result["helper_output"]["proposal"]["provider"], "heuristic")
            self.assertTrue(result["proposal_summary"]["is_valid"])
            self.assertEqual(result["proposal_summary"]["edit_mode"], "anchor")
            self.assertEqual(result["helper_output"]["edit_context"]["mode"], "anchor")
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

    def test_graph_stops_early_on_invalid_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            original = "original\n"
            path.write_text(original, encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": 'replace "hello" with "world"',
                    "proposal_mode": "heuristic",
                }
            )

            self.assertEqual(result["status"], "invalid_proposal")
            self.assertIn("Missing target_path.", result["error"])
            self.assertEqual(result["human_gate"]["action"], "clarify_request")
            self.assertFalse(result.get("edit_applied", False))
            self.assertEqual(path.read_text(encoding="utf-8"), original)

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
            self.assertEqual(result["human_gate"]["action"], "inspect_failure")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_graph_blocks_named_function_edit_when_code_graph_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            original = "def boot_system():\n    return 'old'\n"
            path.write_text(original, encoding="utf-8")

            with patch(
                "shipyard.graph.inspect_code_graph_status",
                return_value={
                    "ready": False,
                    "available": True,
                    "source": "cgr_stats",
                    "reason": "Memgraph is not reachable from the current environment.",
                },
            ):
                result = build_graph().invoke(
                    {
                        "instruction": "Update boot_system",
                        "proposal_mode": "heuristic",
                        "context": {
                            "file_hint": str(path),
                            "function_name": "boot_system",
                        },
                    }
                )

            self.assertEqual(result["status"], "graph_unavailable")
            self.assertIn("Named-function edits require a ready Code-Graph-RAG runtime", result["error"])
            self.assertEqual(result["human_gate"]["action"], "sync_graph")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_graph_applies_named_function_edit_after_graph_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            original = "def boot_system():\n    return 'old'\n"
            path.write_text(original, encoding="utf-8")

            with patch(
                "shipyard.graph.inspect_code_graph_status",
                return_value={
                    "ready": True,
                    "available": True,
                    "source": "cgr_stats",
                    "reason": "Code graph statistics are available.",
                },
            ):
                result = build_graph().invoke(
                    {
                        "instruction": "Update boot_system",
                        "proposal_mode": "heuristic",
                        "replacement": "def boot_system():\n    return 'new'\n",
                        "context": {
                            "file_hint": str(path),
                            "function_name": "boot_system",
                        },
                        "verification_commands": [f"python3 -m py_compile {path}"],
                    }
                )

            self.assertEqual(result["status"], "verified")
            self.assertEqual(
                path.read_text(encoding="utf-8").strip(),
                "def boot_system():\n    return 'new'",
            )
            self.assertTrue(result["code_graph_status"]["refresh_required"])
            self.assertTrue(result["code_graph_status"]["index_state"]["stale"])
            self.assertTrue(result["code_graph_status"]["context_collected"])
            self.assertEqual(result["code_graph_status"]["query_mode"], "function_source_only")
            self.assertEqual(result["helper_output"]["helper_agent"]["task_type"], "function_edit_planning")
            self.assertEqual(result["helper_output"]["edit_context"]["mode"], "named_function")
            self.assertEqual(result["helper_output"]["edit_context"]["function_name"], "boot_system")
            self.assertIn("return 'old'", result["helper_output"]["edit_context"]["current_source"])

    def test_graph_retry_updates_helper_notes_from_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            path.write_text('print("safe")\n', encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "force bad edit",
                    "target_path": str(path),
                    "anchor": 'print("safe")',
                    "replacement": 'print("broken)',
                    "proposal_mode": "heuristic",
                    "verification_commands": [f"python3 -m py_compile {path}"],
                    "max_edit_attempts": 2,
                }
            )

            self.assertEqual(result["status"], "failed_after_retries")
            self.assertIn("Verification failed previously", result["context"]["helper_notes"])

    def test_graph_supports_write_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("old\n", encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": 'write "fresh" to file',
                    "proposal_mode": "heuristic",
                    "context": {"file_hint": str(path)},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), "fresh")
            self.assertEqual(result["helper_output"]["edit_context"]["mode"], "write_file")

    def test_graph_supports_unquoted_write_file_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "temptest.py"

            result = build_graph().invoke(
                {
                    "instruction": "write hello world to new file",
                    "proposal_mode": "heuristic",
                    "target_path": str(path),
                    "context": {"file_hint": str(path)},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), "hello world")

    def test_graph_supports_copy_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pledge.txt"
            path.write_text("hello\n", encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "make some copies of pledge.txt",
                    "proposal_mode": "heuristic",
                    "target_path": str(path),
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(len(result["changed_files"]), 3)
            for copy_path in result["changed_files"]:
                self.assertTrue(Path(copy_path).exists())
                self.assertEqual(Path(copy_path).read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(result["proposal_summary"]["edit_mode"], "copy_file")
            self.assertTrue(result["proposal_summary"]["is_valid"])

    def test_graph_write_file_without_target_creates_managed_workspace_file(self) -> None:
        result = build_graph().invoke(
            {
                "instruction": 'write "hello world" to file',
                "proposal_mode": "heuristic",
            }
        )

        target_path = Path(result["target_path"])
        self.assertEqual(result["status"], "edited")
        self.assertTrue(target_path.exists())
        self.assertEqual(target_path.read_text(encoding="utf-8"), "hello world")
        self.assertIn(".shipyard/data/workspace/", str(target_path))

    def test_graph_make_new_file_creates_blank_managed_workspace_file(self) -> None:
        result = build_graph().invoke(
            {
                "instruction": "make a new file",
                "proposal_mode": "heuristic",
            }
        )

        target_path = Path(result["target_path"])
        self.assertEqual(result["status"], "edited")
        self.assertTrue(target_path.exists())
        self.assertEqual(target_path.read_text(encoding="utf-8"), "")
        self.assertEqual(result["proposal_summary"]["edit_mode"], "write_file")

    def test_graph_make_multiple_new_files_creates_blank_managed_workspace_files(self) -> None:
        result = build_graph().invoke(
            {
                "instruction": "make 2 new files",
                "proposal_mode": "heuristic",
                "session_id": "demo-multi",
            }
        )

        self.assertEqual(result["status"], "edited")
        self.assertEqual(result["proposal_summary"]["edit_mode"], "create_files")
        self.assertEqual(result["proposal_summary"]["quantity"], 2)
        self.assertEqual(len(result["changed_files"]), 2)
        for created_path in result["changed_files"]:
            path = Path(created_path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_graph_create_files_can_focus_content_on_numbered_file(self) -> None:
        result = build_graph().invoke(
            {
                "instruction": "create 4 random files and in file 3 write hello world",
                "proposal_mode": "heuristic",
                "session_id": "demo-batch",
            }
        )

        self.assertEqual(result["status"], "edited")
        self.assertEqual(len(result["changed_files"]), 4)
        paths = [Path(path) for path in result["changed_files"]]
        for path in paths:
            self.assertTrue(path.exists())
        by_name = {path.name: path.read_text(encoding="utf-8") for path in paths}
        self.assertEqual(by_name["file3.txt"], "hello world")
        self.assertEqual(by_name["file1.txt"], "")
        self.assertEqual(by_name["file2.txt"], "")
        self.assertEqual(by_name["file4.txt"], "")

    def test_graph_write_file_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "folder" / "test.py"

            result = build_graph().invoke(
                {
                    "instruction": 'write "print(123)" to file',
                    "proposal_mode": "heuristic",
                    "target_path": str(path),
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "print(123)")

    def test_graph_replaces_middle_occurrence_in_small_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unknown.txt"
            path.write_text("hahaha", encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "make the middle ha into ho",
                    "proposal_mode": "heuristic",
                    "context": {"file_hint": str(path)},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(result["proposal_summary"]["occurrence_selector"], "middle")
            self.assertEqual(path.read_text(encoding="utf-8"), "hahoha")

    def test_graph_supports_delete_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unknown-Copy.txt"
            path.write_text("bye", encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "delete unknown-Copy.txt",
                    "proposal_mode": "heuristic",
                    "context": {"file_hint": str(path)},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertFalse(path.exists())

    def test_graph_supports_symbol_rename_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            path.write_text(
                "base = 1\nvalue = base + 2\nprint(base)\n",
                encoding="utf-8",
            )

            result = build_graph().invoke(
                {
                    "instruction": "change base to boos and update other places where base appears as boos",
                    "proposal_mode": "heuristic",
                    "context": {"file_hint": str(path)},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(result["proposal_summary"]["edit_mode"], "rename_symbol")
            self.assertIn("boos = 1", path.read_text(encoding="utf-8"))
            self.assertIn("value = boos + 2", path.read_text(encoding="utf-8"))
            self.assertIn("print(boos)", path.read_text(encoding="utf-8"))

    def test_graph_prompt_reflects_final_planned_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actual.py"

            result = build_graph().invoke(
                {
                    "instruction": 'write "hello" to file',
                    "proposal_mode": "heuristic",
                    "target_path": str(path),
                    "context": {"file_hint": "/tmp/ignored.py"},
                }
            )

            self.assertIn(f"Target path: {path}", result["prompt"])
            self.assertNotIn("/tmp/ignored.py", result["prompt"])

    def test_graph_supports_append_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("first", encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "append text",
                    "proposal_mode": "heuristic",
                    "context": {
                        "file_hint": str(path),
                        "append_text": "\nsecond",
                    },
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\nsecond")
            self.assertEqual(result["helper_output"]["edit_context"]["mode"], "append")

    def test_graph_removes_new_file_after_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.py"

            result = build_graph().invoke(
                {
                    "instruction": "force broken file write",
                    "proposal_mode": "heuristic",
                    "target_path": str(path),
                    "edit_mode": "write_file",
                    "replacement": 'print("broken)',
                    "context": {"file_hint": str(path)},
                    "verification_commands": [f"python3 -m py_compile {path}"],
                    "max_edit_attempts": 1,
                }
            )

            self.assertEqual(result["status"], "failed_after_retries")
            self.assertTrue(result["reverted_to_snapshot"])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
