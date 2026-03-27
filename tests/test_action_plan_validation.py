from __future__ import annotations

import unittest

from shipyard.action_plan_validation import validate_action_plan


class ActionPlanValidationTests(unittest.TestCase):
    def test_validate_action_plan_accepts_complete_file_list(self) -> None:
        errors = validate_action_plan(
            "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
            [
                {"target_path": "main.py", "valid": True},
                {"target_path": "math_utils.py", "valid": True},
                {"target_path": "formatter.py", "valid": True},
                {"target_path": "config.json", "valid": True},
            ],
        )

        self.assertEqual(errors, [])

    def test_validate_action_plan_rejects_missing_explicit_files(self) -> None:
        errors = validate_action_plan(
            "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
            [
                {"target_path": "main.py", "valid": True},
            ],
        )

        self.assertTrue(any("math_utils.py" in error for error in errors))
        self.assertTrue(any("config.json" in error for error in errors))

    def test_validate_action_plan_rejects_too_few_steps(self) -> None:
        errors = validate_action_plan(
            "create 4 files and then write hello world",
            [
                {"target_path": "file1.txt", "valid": True},
            ],
        )

        self.assertTrue(any("2 steps" in error for error in errors))

    def test_validate_action_plan_accepts_scaffold_files_covering_explicit_names(self) -> None:
        errors = validate_action_plan(
            "Create a tiny repo with main.py and config.json",
            [
                {
                    "edit_mode": "scaffold_files",
                    "valid": True,
                    "files": [
                        {"path": "main.py", "content": "print('hi')\n"},
                        {"path": "config.json", "content": "{}\n"},
                    ],
                }
            ],
        )

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
