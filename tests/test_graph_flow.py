from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.graph import apply_edit, build_graph, plan_edit
from shipyard.workspaces import get_session_workspace


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
            self.assertEqual(result["helper_output"]["helper_agent"]["agent_name"], "helper-anchor-planner")
            self.assertEqual(result["helper_output"]["helper_agent"]["delegation_mode"], "sequential")
            self.assertEqual(result["tasks"][0]["role"], "helper-anchor-planner")
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

    def test_apply_edit_replans_stale_anchor_from_current_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "formatter.py"
            path.write_text('def format_result(value, unit):\n    return f"Average latency: {value} {unit}"\n', encoding="utf-8")

            with patch(
                "shipyard.graph.propose_edit",
                return_value={
                    "edit_mode": "anchor",
                    "anchor": 'Average latency',
                    "replacement": 'Processed latency',
                    "is_valid": True,
                    "validation_errors": [],
                    "pointers": None,
                    "occurrence_selector": None,
                },
            ):
                result = apply_edit(
                    {
                        "instruction": 'Change "Processed" to "Average latency"',
                        "target_path": str(path),
                        "edit_mode": "anchor",
                        "anchor": 'Processed latency',
                        "replacement": 'Average latency',
                        "file_before": path.read_text(encoding="utf-8"),
                        "context": {"helper_notes": "Use the exact current file"},
                    }
                )

            self.assertEqual(result["status"], "edited")
            self.assertIn("repair_reason", result["proposal_summary"])

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
            self.assertEqual(result["helper_output"]["helper_agent"]["agent_name"], "helper-function-planner")
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

            self.assertEqual(result["status"], "invalid_proposal")
            self.assertIn("full_file_rewrite=true", result["error"])

    def test_graph_scaffold_files_writes_into_session_workspace_in_testing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            session_id = "scaffold-session"
            workspace = get_session_workspace(session_id)

            result = build_graph().invoke(
                {
                    "instruction": "Create tiny repo files",
                    "session_id": session_id,
                    "edit_mode": "scaffold_files",
                    "files": [
                        {"path": "main.py", "content": 'print("hi")\n'},
                        {"path": "config.json", "content": '{"unit":"ms"}\n'},
                    ],
                    "preplanned_action": {
                        "instruction": "Create tiny repo files",
                        "edit_mode": "scaffold_files",
                        "files": [
                            {"path": "main.py", "content": 'print("hi")\n'},
                            {"path": "config.json", "content": '{"unit":"ms"}\n'},
                        ],
                        "valid": True,
                        "validation_errors": [],
                    },
                    "context": {"testing_mode": True},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertTrue((workspace / "main.py").exists())
            self.assertTrue((workspace / "config.json").exists())
            self.assertTrue(all(str(workspace.resolve()) in path for path in result["changed_files"]))

    def test_apply_edit_replaces_all_anchor_occurrences_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "formatter.py"
            original = 'return "Average Average"\n'
            path.write_text(original, encoding="utf-8")

            result = apply_edit(
                {
                    "instruction": "Replace Average with Processed everywhere in formatter.py",
                    "target_path": str(path),
                    "edit_mode": "anchor",
                    "anchor": "Average",
                    "replacement": "Processed",
                    "occurrence_selector": "all",
                    "file_before": original,
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), 'return "Processed Processed"\n')

    def test_apply_edit_replaces_pointer_spans(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "formatter.py"
            original = 'return "Average Average"\n'
            path.write_text(original, encoding="utf-8")

            result = apply_edit(
                {
                    "instruction": "Replace Average with Processed everywhere in formatter.py",
                    "target_path": str(path),
                    "edit_mode": "anchor",
                    "anchor": "Average",
                    "replacement": "Processed",
                    "pointers": [{"start": 8, "end": 15}, {"start": 16, "end": 23}],
                    "file_before": original,
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), 'return "Processed Processed"\n')

    def test_apply_edit_repairs_ambiguous_anchor_with_pointer_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "formatter.py"
            original = 'return "Average Average"\n'
            path.write_text(original, encoding="utf-8")

            with patch(
                "shipyard.graph.propose_edit",
                return_value={
                    "edit_mode": "anchor",
                    "anchor": "Average",
                    "replacement": "Processed",
                    "pointers": [{"start": 8, "end": 15}, {"start": 16, "end": 23}],
                },
            ):
                result = apply_edit(
                    {
                        "instruction": "In formatter.py, change the word Average to Processed everywhere it appears in that file.",
                        "target_path": str(path),
                        "edit_mode": "anchor",
                        "anchor": "Average",
                        "replacement": "Processed",
                        "file_before": original,
                        "context": {},
                        "proposal_summary": {},
                    }
                )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), 'return "Processed Processed"\n')
            self.assertEqual(result["pointers"], [{"start": 8, "end": 15}, {"start": 16, "end": 23}])

    def test_apply_edit_autocorrects_single_exact_anchor_when_invalid_pointers_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "formatter.py"
            original = 'return "Average latency"\n'
            path.write_text(original, encoding="utf-8")

            result = apply_edit(
                {
                    "instruction": 'Change "Average" to "Processed" in formatter.py',
                    "target_path": str(path),
                    "edit_mode": "anchor",
                    "anchor": "Average",
                    "replacement": "Processed",
                    "pointers": [{"start": 0, "end": 7}],
                    "file_before": original,
                    "context": {},
                    "proposal_summary": {},
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertEqual(path.read_text(encoding="utf-8"), 'return "Processed latency"\n')
            self.assertEqual(result["pointers"], [{"start": 8, "end": 15}])

    def test_plan_edit_refines_preplanned_anchor_from_current_file_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "main.py"
            path.write_text("alpha\nbeta\nbeta\n", encoding="utf-8")

            with patch(
                "shipyard.graph.propose_edit",
                return_value={
                    "edit_mode": "anchor",
                    "target_path": str(path),
                    "anchor": "beta\nbeta\n",
                    "replacement": "beta\n",
                    "pointers": [{"start": 6, "end": 16}],
                    "is_valid": True,
                    "validation_errors": [],
                    "provider": "openai",
                    "provider_reason": "Refined against current file contents.",
                },
            ) as propose_mock:
                result = plan_edit(
                    {
                        "instruction": "Remove all repeated beta lines except one.",
                        "target_path": str(path),
                        "file_before": "alpha\nbeta\nbeta\n",
                        "tool_outputs": [{"tool": "read_file", "target_path": str(path), "content": "alpha\nbeta\nbeta\n"}],
                        "action_plan": {"provider": "openai", "provider_reason": "planned"},
                        "preplanned_action": {
                            "instruction": "Edit main.py to remove all repeated beta lines except one.",
                            "edit_mode": "anchor",
                            "target_path": str(path),
                            "anchor": "beta",
                            "replacement": "beta",
                            "valid": True,
                            "validation_errors": [],
                        },
                    }
                )

            propose_mock.assert_called_once()
            self.assertEqual(result["anchor"], "beta\nbeta\n")
            self.assertEqual(result["pointers"], [{"start": 6, "end": 16}])
            self.assertIn("Refined", result["proposal_summary"]["provider_reason"])

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

    def test_graph_run_command_returns_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = get_session_workspace("cmd-demo")
            (workspace / "main.py").write_text('print("hi")\n', encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "Run the program",
                    "session_id": "cmd-demo",
                    "edit_mode": "run_command",
                    "command": "python3 main.py",
                    "preplanned_action": {
                        "instruction": "Run the program",
                        "edit_mode": "run_command",
                        "command": "python3 main.py",
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "observed")
            self.assertEqual(result["tool_output"]["tool"], "run_command")
            self.assertEqual(result["tool_output"]["returncode"], 0)

    def test_graph_verify_command_returns_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = get_session_workspace("verify-demo")
            (workspace / "main.py").write_text('print("hi")\n', encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "Verify the program",
                    "session_id": "verify-demo",
                    "edit_mode": "verify_command",
                    "command": "python3 main.py",
                    "preplanned_action": {
                        "instruction": "Verify the program",
                        "edit_mode": "verify_command",
                        "command": "python3 main.py",
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["tool_output"]["tool"], "verify_command")

    def test_graph_create_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "reports"

            result = build_graph().invoke(
                {
                    "instruction": "Create reports directory",
                    "edit_mode": "create_directory",
                    "target_path": str(directory),
                    "preplanned_action": {
                        "instruction": "Create reports directory",
                        "edit_mode": "create_directory",
                        "target_path": str(directory),
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertTrue(directory.exists())

    def test_graph_rename_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "report.py"
            source.write_text("print('hi')\n", encoding="utf-8")
            destination = Path(tmpdir) / "reporter.py"

            result = build_graph().invoke(
                {
                    "instruction": "Rename report.py",
                    "edit_mode": "rename_file",
                    "source_path": str(source),
                    "destination_path": str(destination),
                    "target_path": str(source),
                    "preplanned_action": {
                        "instruction": "Rename report.py",
                        "edit_mode": "rename_file",
                        "source_path": str(source),
                        "destination_path": str(destination),
                        "target_path": str(source),
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertTrue(destination.exists())
            self.assertFalse(source.exists())

    def test_graph_list_files_allows_directory_target_without_preread_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = get_session_workspace("list-demo")
            (workspace / "main.py").write_text('print("hi")\n', encoding="utf-8")

            result = build_graph().invoke(
                {
                    "instruction": "Inspect the current tiny repo files before editing",
                    "session_id": "list-demo",
                    "edit_mode": "list_files",
                    "target_path": str(workspace),
                    "preplanned_action": {
                        "instruction": "Inspect the current tiny repo files before editing",
                        "edit_mode": "list_files",
                        "target_path": str(workspace),
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "observed")
            self.assertEqual(result["tool_output"]["tool"], "list_files")
            self.assertIn("main.py", result["tool_output"]["files"])

    def test_graph_read_file_blocks_directory_target_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            result = build_graph().invoke(
                {
                    "instruction": "Read the directory",
                    "edit_mode": "read_file",
                    "target_path": str(root),
                    "preplanned_action": {
                        "instruction": "Read the directory",
                        "edit_mode": "read_file",
                        "target_path": str(root),
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "edit_blocked")
            self.assertIn("Target file was not found for read_file.", result["error"])

    def test_graph_read_file_missing_target_is_non_blocking_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "reporter.py"

            result = build_graph().invoke(
                {
                    "instruction": "Read reporter.py before creating it",
                    "edit_mode": "read_file",
                    "target_path": str(missing),
                    "preplanned_action": {
                        "instruction": "Read reporter.py before creating it",
                        "edit_mode": "read_file",
                        "target_path": str(missing),
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "observed")
            self.assertTrue(result["tool_output"]["missing"])

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
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            result = build_graph().invoke(
                {
                    "instruction": 'write "hello world" to file',
                    "proposal_mode": "heuristic",
                }
            )

            self.assertEqual(result["status"], "invalid_proposal")
            self.assertIn("Missing target_path.", result["error"])

    def test_graph_make_new_file_without_target_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            result = build_graph().invoke(
                {
                    "instruction": "make a new file",
                    "proposal_mode": "heuristic",
                }
            )

            self.assertEqual(result["status"], "invalid_proposal")
            self.assertIn("Missing target_path.", result["error"])

    def test_graph_make_multiple_new_files_without_target_is_blocked(self) -> None:
        result = build_graph().invoke(
            {
                "instruction": "make 2 new files",
                "proposal_mode": "heuristic",
                "session_id": "demo-multi",
            }
        )

        self.assertEqual(result["status"], "invalid_proposal")
        self.assertIn("Missing target_path.", result["error"])

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

    def test_graph_supports_preplanned_scaffold_files_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            main_path = Path(tmpdir) / "main.py"
            config_path = Path(tmpdir) / "config.json"

            result = build_graph().invoke(
                {
                    "instruction": "Create a tiny repo with main.py and config.json",
                    "edit_mode": "scaffold_files",
                    "files": [
                        {"path": str(main_path), "content": "print('hi')\n"},
                        {"path": str(config_path), "content": "{}\n"},
                    ],
                    "preplanned_action": {
                        "instruction": "Create a tiny repo with main.py and config.json",
                        "edit_mode": "scaffold_files",
                        "files": [
                            {"path": str(main_path), "content": "print('hi')\n"},
                            {"path": str(config_path), "content": "{}\n"},
                        ],
                        "valid": True,
                        "validation_errors": [],
                    },
                }
            )

            self.assertEqual(result["status"], "edited")
            self.assertTrue(main_path.exists())
            self.assertTrue(config_path.exists())
            self.assertEqual(main_path.read_text(encoding="utf-8"), "print('hi')\n")
            self.assertEqual(config_path.read_text(encoding="utf-8"), "{}\n")

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
