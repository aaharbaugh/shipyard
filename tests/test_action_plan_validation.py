from __future__ import annotations

import unittest

from shipyard.action_plan_validation import validate_action_plan


class ActionPlanValidationTests(unittest.TestCase):
    def test_validate_action_plan_accepts_valid_plan(self) -> None:
        errors = validate_action_plan(
            "Create files",
            [
                {"target_path": "main.py", "valid": True},
                {"target_path": "config.json", "valid": True},
            ],
        )
        self.assertEqual(errors, [])

    def test_validate_action_plan_rejects_empty(self) -> None:
        errors = validate_action_plan("do something", [])
        self.assertTrue(any("did not include any actions" in e for e in errors))

    def test_validate_action_plan_flags_invalid_actions(self) -> None:
        errors = validate_action_plan(
            "create stuff",
            [{"target_path": "a.py", "valid": False}],
        )
        self.assertTrue(any("invalid actions" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
