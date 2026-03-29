from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from shipyard.main import (
    _plan_actions_with_cancellation,
    _sanitize_runtime_result,
    _maybe_sync_graph,
    _run_action_plan,
    _sanitize_stale_target_request,
    _should_skip_graph_sync,
    main,
    parse_user_input,
    read_user_input,
    run_once,
)
from shipyard.main import _attach_file_outcome
from shipyard.action_planner import PlanningCancelledError


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
            "shipyard.main.plan_actions",
            return_value={
                "actions": [{"instruction": "replace \"a\" with \"b\"", "valid": True}],
                "provider": "heuristic",
                "provider_reason": "test",
                "is_valid": True,
                "validation_errors": [],
            },
        ), patch(
            "shipyard.main.write_trace", return_value=".shipyard/data/traces/demo.json"
        ), redirect_stdout(output):
            main()

        rendered = output.getvalue()
        self.assertNotIn("proposal_summary:", rendered)
        self.assertIn("status=verified", rendered)

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

    def test_sanitize_stale_target_request_clears_generic_file_when_prompt_names_real_file(self) -> None:
        result = _sanitize_stale_target_request(
            {
                "instruction": "remove all the runs from main.py except for one.",
                "target_path": "/tmp/file.py",
                "context": {"file_hint": "/tmp/file.py", "testing_mode": True},
            }
        )

        self.assertIsNone(result["target_path"])
        self.assertNotIn("file_hint", result["context"])

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

    def test_sanitize_runtime_result_strips_runtime_callables(self) -> None:
        result = _sanitize_runtime_result(
            {
                "session_id": "demo",
                "instruction": "make a todo app",
                "cancel_check": lambda: False,
                "context": {"testing_mode": True},
                "tasks": [
                    {
                        "task_id": "run-demo",
                        "callback": lambda: True,
                    }
                ],
            }
        )

        self.assertNotIn("cancel_check", result)
        self.assertEqual(result["tasks"], [{"task_id": "run-demo"}])

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
        self.assertEqual(result["instruction"], "create 2 files and write hello")
        self.assertEqual(result["request_instruction"], "create 2 files and write hello")

    def test_run_once_short_circuits_invalid_action_plan(self) -> None:
        app = Mock()
        session_store = Mock()

        with patch("shipyard.main.PromptLog") as prompt_log_cls, patch(
            "shipyard.main.generate_spec_bundle",
            return_value={"mode": "direct_edit", "created": False},
        ), patch(
            "shipyard.main.plan_actions",
            return_value={
                "actions": [{"instruction": "write main.py", "valid": True}],
                "provider": "openai",
                "provider_reason": "planned",
                "is_valid": False,
                "validation_errors": ["Action plan did not cover all explicitly named files: config.json."],
            },
        ):
            prompt_log_cls.return_value.append = Mock()
            result = run_once(
                app,
                session_store,
                {
                    "session_id": "demo",
                    "instruction": "Create a tiny repo with main.py and config.json",
                },
            )

        app.invoke.assert_not_called()
        self.assertEqual(result["status"], "invalid_action_plan")
        self.assertIn("config.json", result["execution"]["error"])
        self.assertIn("troubleshooting_path", result["artifacts"])

    def test_run_once_persists_failed_result_when_execution_raises(self) -> None:
        app = Mock()
        session_store = Mock()

        with patch("shipyard.main.PromptLog") as prompt_log_cls, patch(
            "shipyard.main.generate_spec_bundle",
            return_value={"mode": "direct_edit", "created": False},
        ), patch(
            "shipyard.main.plan_actions",
            return_value={
                "actions": [{"instruction": "Inspect repo", "edit_mode": "list_files", "valid": True, "validation_errors": []}],
                "provider": "openai",
                "provider_reason": "planned",
                "is_valid": True,
                "validation_errors": [],
            },
        ), patch(
            "shipyard.main._run_action_plan",
            side_effect=IsADirectoryError("boom"),
        ):
            prompt_log_cls.return_value.append = Mock()
            result = run_once(
                app,
                session_store,
                {
                    "session_id": "web-test",
                    "instruction": "Inspect repo and continue",
                },
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("boom", result["execution"]["error"])
        self.assertIn("troubleshooting_path", result["artifacts"])

    def test_plan_actions_with_cancellation_raises_cancelled(self) -> None:
        started = {"ready": False}

        def fake_plan_actions(state):
            started["ready"] = True
            time.sleep(2)

        cancel_state = {"cancel_check": lambda: started["ready"]}
        with patch("shipyard.main.plan_actions", side_effect=fake_plan_actions):
            with self.assertRaises(PlanningCancelledError):
                _plan_actions_with_cancellation(cancel_state)

    def test_run_action_plan_keeps_edited_status_when_final_step_is_observation(self) -> None:
        app = Mock()
        app.invoke.side_effect = [
            {"status": "observed", "no_op": True, "tool_output": {"tool": "read_file"}},
            {"status": "edited", "no_op": False, "target_path": "/tmp/main.py", "changed_files": ["/tmp/main.py"]},
            {"status": "observed", "no_op": True, "tool_output": {"tool": "run_command", "returncode": 0}},
        ]

        result = _run_action_plan(
            app,
            {"session_id": "demo", "instruction": "edit and verify", "request_instruction": "edit and verify"},
            {
                "actions": [
                    {"instruction": "Read main.py", "edit_mode": "read_file"},
                    {"instruction": "Edit main.py", "edit_mode": "anchor"},
                    {"instruction": "Run main.py", "edit_mode": "run_command"},
                ]
            },
        )

        self.assertEqual(result["status"], "edited")
        self.assertFalse(result["no_op"])


    def test_run_action_plan_marks_no_op_edit_as_edit_skipped(self) -> None:
        app = Mock()
        app.invoke.return_value = {"status": "edited", "no_op": True, "target_path": "/tmp/main.py"}

        result = _run_action_plan(
            app,
            {"session_id": "demo", "instruction": "touch main.py", "request_instruction": "touch main.py"},
            {"actions": [{"id": "step-1", "instruction": "Touch main.py", "edit_mode": "anchor", "action_class": "mutate"}]},
        )

        self.assertEqual(result["action_steps"][0]["status"], "edit_skipped")

    def test_run_action_plan_does_not_blind_retry_verification_failures(self) -> None:
        app = Mock()
        app.invoke.return_value = {
            "status": "verification_failed",
            "no_op": True,
            "verification_results": [{"command": "python3 main.py", "returncode": 1, "stderr": "AssertionError"}],
            "error": "Command failed with exit code 1.",
        }

        result = _run_action_plan(
            app,
            {"session_id": "demo", "instruction": "verify app", "request_instruction": "verify app"},
            {"actions": [{"id": "step-1", "instruction": "Verify app", "edit_mode": "verify_command", "action_class": "verify", "max_retries": 2}]},
        )

        self.assertEqual(app.invoke.call_count, 1)
        self.assertEqual(result["action_steps"][0]["retry_count"], 0)
        self.assertEqual(result["action_steps"][0]["status"], "verification_failed")


class ParallelBatchTests(unittest.TestCase):
    """Tests for parallel batch execution of independent actions."""

    def test_find_parallel_batch_groups_independent_actions(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s1", "action_class": "mutate", "edit_mode": "write_file", "target_path": "a.py"},
            {"id": "s2", "action_class": "mutate", "edit_mode": "write_file", "target_path": "b.py"},
            {"id": "s3", "action_class": "mutate", "edit_mode": "write_file", "target_path": "c.py"},
        ]
        batch = _find_parallel_batch(actions, 0, set())
        self.assertEqual(batch, [0, 1, 2])

    def test_find_parallel_batch_stops_at_dependency(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s1", "action_class": "inspect", "edit_mode": "read_file", "target_path": "a.py"},
            {"id": "s2", "action_class": "inspect", "edit_mode": "read_file", "target_path": "b.py", "depends_on": ["s1"]},
        ]
        batch = _find_parallel_batch(actions, 0, set())
        # s2 depends on s1 (in batch) → can't parallelize
        self.assertEqual(batch, [0])

    def test_find_parallel_batch_allows_completed_deps(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s3", "action_class": "mutate", "edit_mode": "write_file", "target_path": "a.py", "depends_on": ["s1"]},
            {"id": "s4", "action_class": "mutate", "edit_mode": "write_file", "target_path": "b.py", "depends_on": ["s2"]},
        ]
        # s1 and s2 already completed
        batch = _find_parallel_batch(actions, 0, {"s1", "s2"})
        self.assertEqual(batch, [0, 1])

    def test_find_parallel_batch_stops_at_same_target(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s1", "action_class": "mutate", "edit_mode": "write_file", "target_path": "a.py"},
            {"id": "s2", "action_class": "mutate", "edit_mode": "write_file", "target_path": "a.py"},
        ]
        batch = _find_parallel_batch(actions, 0, set())
        self.assertEqual(batch, [0])

    def test_find_parallel_batch_stops_at_different_action_class(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s1", "action_class": "inspect", "edit_mode": "read_file", "target_path": "a.py"},
            {"id": "s2", "action_class": "mutate", "edit_mode": "write_file", "target_path": "b.py"},
        ]
        batch = _find_parallel_batch(actions, 0, set())
        self.assertEqual(batch, [0])

    def test_find_parallel_batch_requires_target_path(self) -> None:
        from shipyard.main import _find_parallel_batch

        actions = [
            {"id": "s1", "action_class": "mutate", "edit_mode": "write_file"},
            {"id": "s2", "action_class": "mutate", "edit_mode": "write_file", "target_path": "b.py"},
        ]
        batch = _find_parallel_batch(actions, 0, set())
        self.assertEqual(batch, [0])  # no target → single-step batch

    def test_parallel_batch_executes_and_merges(self) -> None:
        app = Mock()
        app.invoke.side_effect = [
            {"status": "edited", "changed_files": ["/tmp/a.py"]},
            {"status": "edited", "changed_files": ["/tmp/b.py"]},
            {"status": "edited", "changed_files": ["/tmp/c.py"]},
        ]

        result = _run_action_plan(
            app,
            {"session_id": "demo", "instruction": "edit three files"},
            {
                "actions": [
                    {"id": "s1", "instruction": "edit a", "action_class": "mutate", "edit_mode": "write_file", "target_path": "a.py"},
                    {"id": "s2", "instruction": "edit b", "action_class": "mutate", "edit_mode": "write_file", "target_path": "b.py"},
                    {"id": "s3", "instruction": "edit c", "action_class": "mutate", "edit_mode": "write_file", "target_path": "c.py"},
                ]
            },
        )

        self.assertEqual(app.invoke.call_count, 3)
        self.assertIn("/tmp/a.py", result["changed_files"])
        self.assertIn("/tmp/b.py", result["changed_files"])
        self.assertIn("/tmp/c.py", result["changed_files"])
        # All steps should be marked as parallel_batch
        for step in result["action_steps"]:
            self.assertTrue(step.get("parallel_batch"))


class PathSandboxingTests(unittest.TestCase):
    """Tests for workspace path sandboxing in apply_edit."""

    def test_sandbox_target_path_resolves_relative_to_workspace(self) -> None:
        from shipyard.graph import _sandbox_target_path

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT", Path(tmpdir) / "workspaces"
        ):
            # get_session_workspace maps session_id to "default" workspace
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True)
            (workspace / "app.js").write_text("hello", encoding="utf-8")

            result = _sandbox_target_path("app.js", {"session_id": "demo"})
            self.assertEqual(result, str((workspace / "app.js").resolve()))

    def test_sandbox_target_path_leaves_absolute_unchanged(self) -> None:
        from shipyard.graph import _sandbox_target_path

        result = _sandbox_target_path("/absolute/path/file.py", {"session_id": "demo"})
        self.assertEqual(result, "/absolute/path/file.py")

    def test_sandbox_target_path_no_session_returns_as_is(self) -> None:
        from shipyard.graph import _sandbox_target_path

        result = _sandbox_target_path("app.js", {})
        self.assertEqual(result, "app.js")


if __name__ == "__main__":
    unittest.main()
