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

    def test_normalize_action_promotes_add_generation_to_append(self) -> None:
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

        self.assertEqual(action["edit_mode"], "append")
        self.assertTrue(action["valid"])


if __name__ == "__main__":
    unittest.main()
