from __future__ import annotations

import unittest

from shipyard.actions import normalize_action


class ActionNormalizationTests(unittest.TestCase):
    def test_normalize_action_uses_fallback_target_and_validation(self) -> None:
        action = normalize_action(
            {"instruction": "replace total with totality", "edit_mode": "rename_symbol"},
            fallback={
                "target_path": "demo.py",
                "target_path_source": "file_hint",
                "anchor": "total",
                "replacement": "totality",
            },
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["target_path"], "demo.py")
        self.assertEqual(action["edit_mode"], "rename_symbol")
        self.assertTrue(action["valid"])

    def test_normalize_action_keeps_model_declared_edit_mode(self) -> None:
        action = normalize_action(
            {
                "instruction": "add a random algorithm to scratch.py",
                "edit_mode": "write_file",
                "replacement": "def random_algorithm():\n    return 1\n",
            },
            fallback={"target_path": "scratch.py", "target_path_source": "explicit_target_path"},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["edit_mode"], "write_file")
        self.assertTrue(action["valid"])

    def test_normalize_action_supports_scaffold_files(self) -> None:
        action = normalize_action(
            {
                "instruction": "scaffold tiny repo",
                "edit_mode": "scaffold_files",
                "files": [
                    {"path": "main.py", "content": "print('hi')\n"},
                    {"path": "config.json", "content": "{}\n"},
                ],
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["edit_mode"], "scaffold_files")
        self.assertTrue(action["valid"])

    def test_normalize_action_accepts_search_and_replace_aliases(self) -> None:
        action = normalize_action(
            {
                "instruction": "update formatter.py",
                "edit_mode": "anchor",
                "target_path": "formatter.py",
                "search_text": "old text",
                "replace_text": "new text",
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["anchor"], "old text")
        self.assertEqual(action["replacement"], "new text")
        self.assertTrue(action["valid"])

    def test_normalize_action_supports_run_command(self) -> None:
        action = normalize_action(
            {
                "instruction": "Run the program",
                "edit_mode": "run_command",
                "command": "python3 main.py",
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["edit_mode"], "run_command")
        self.assertEqual(action["command"], "python3 main.py")
        self.assertTrue(action["valid"])

    def test_normalize_action_supports_dependency_metadata(self) -> None:
        action = normalize_action(
            {
                "id": "step-2",
                "role": "helper-planner",
                "agent_type": "specialist",
                "parent_task_id": "step-1",
                "child_task_ids": ["step-3"],
                "allowed_actions": ["verify_command"],
                "instruction": "Verify the repo",
                "action_class": "verify",
                "edit_mode": "verify_command",
                "target_path": "main.py",
                "command": "python3 main.py",
                "depends_on": ["step-1"],
                "inputs_from": ["step-1"],
                "timeout_seconds": 15,
                "max_retries": 2,
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["id"], "step-2")
        self.assertEqual(action["role"], "helper-planner")
        self.assertEqual(action["agent_type"], "specialist")
        self.assertEqual(action["parent_task_id"], "step-1")
        self.assertEqual(action["child_task_ids"], ["step-3"])
        self.assertEqual(action["allowed_actions"], ["verify_command"])
        self.assertEqual(action["action_class"], "verify")
        self.assertEqual(action["depends_on"], ["step-1"])
        self.assertEqual(action["inputs_from"], ["step-1"])
        self.assertTrue(action["valid"])

    def test_normalize_action_supports_move_file(self) -> None:
        action = normalize_action(
            {
                "instruction": "Rename report.py to reporter.py",
                "edit_mode": "rename_file",
                "target_path": "report.py",
                "source_path": "report.py",
                "destination_path": "reporter.py",
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["edit_mode"], "rename_file")
        self.assertTrue(action["valid"])

    def test_normalize_action_maps_read_many_files_files_to_paths(self) -> None:
        action = normalize_action(
            {
                "instruction": "Read the main UI files",
                "edit_mode": "read_many_files",
                "files": ["index.html", "app.js", "styles.css"],
            },
            fallback={},
            provider="openai",
            provider_reason="planned",
        )

        self.assertEqual(action["paths"], ["index.html", "app.js", "styles.css"])
        self.assertTrue(action["valid"])


if __name__ == "__main__":
    unittest.main()
