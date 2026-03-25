from __future__ import annotations

import unittest

from shipyard.proposal_validation import attach_validation


class ProposalValidationTests(unittest.TestCase):
    def test_attach_validation_marks_missing_target_invalid(self) -> None:
        result = attach_validation(
            {
                "edit_mode": "anchor",
                "anchor": "old",
                "replacement": "new",
            }
        )

        self.assertFalse(result["is_valid"])
        self.assertIn("Missing target_path.", result["validation_errors"])


if __name__ == "__main__":
    unittest.main()
