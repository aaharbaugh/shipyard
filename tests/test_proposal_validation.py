from __future__ import annotations

import unittest

from shipyard.proposal_validation import validate_proposal


class ProposalValidationTests(unittest.TestCase):
    def test_anchor_mode_accepts_pointer_edits_without_unique_anchor_selection(self) -> None:
        errors = validate_proposal(
            {
                "edit_mode": "anchor",
                "target_path": "formatter.py",
                "anchor": "Average",
                "replacement": "Processed",
                "pointers": [{"start": 8, "end": 15}, {"start": 16, "end": 23}],
            }
        )

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
