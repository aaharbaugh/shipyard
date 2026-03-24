from __future__ import annotations

import unittest

from shipyard.proposal import propose_edit


class AgentPlanningTests(unittest.TestCase):
    def test_heuristic_proposal_uses_context_search_and_replace(self) -> None:
        result = propose_edit(
            {
                "instruction": "Update the file",
                "proposal_mode": "heuristic",
                "context": {
                    "file_hint": "demo.py",
                    "search_text": 'print("old")',
                    "replace_text": 'print("new")',
                },
            }
        )

        self.assertEqual(result["target_path"], "demo.py")
        self.assertEqual(result["anchor"], 'print("old")')
        self.assertEqual(result["replacement"], 'print("new")')
        self.assertEqual(result["provider"], "heuristic")

    def test_heuristic_proposal_parses_simple_replace_instruction(self) -> None:
        result = propose_edit(
            {
                "instruction": 'replace "before" with "after"',
                "proposal_mode": "heuristic",
                "context": {"file_hint": "demo.py"},
            }
        )

        self.assertEqual(result["target_path"], "demo.py")
        self.assertEqual(result["anchor"], "before")
        self.assertEqual(result["replacement"], "after")

    def test_openai_mode_without_key_falls_back_to_heuristic(self) -> None:
        result = propose_edit(
            {
                "instruction": 'replace "one" with "two"',
                "proposal_mode": "openai",
                "context": {"file_hint": "demo.py"},
            }
        )

        self.assertEqual(result["provider"], "heuristic")
        self.assertIn("OPENAI_API_KEY", result["provider_reason"])


if __name__ == "__main__":
    unittest.main()
