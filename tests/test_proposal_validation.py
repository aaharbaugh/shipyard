from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shipyard.proposal_validation import validate_proposal


class ProposalValidationTests(unittest.TestCase):
    def test_anchor_mode_accepts_pointer_edits_without_unique_anchor_selection(self) -> None:
        errors = validate_proposal(
            {
                "action_class": "mutate",
                "edit_mode": "anchor",
                "intent": "localized_edit",
                "edit_scope": "multi_span",
                "expected_existing_state": "existing_file",
                "recovery_strategy": "replan_step",
                "target_path": "formatter.py",
                "anchor": "Average",
                "replacement": "Processed",
                "pointers": [{"start": 8, "end": 15}, {"start": 16, "end": 23}],
            }
        )

        self.assertEqual(errors, [])

    def test_scaffold_files_rejects_overwriting_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "index.html"
            target.write_text("<h1>Old</h1>\n", encoding="utf-8")

            errors = validate_proposal(
                {
                    "edit_mode": "scaffold_files",
                    "files": [{"path": str(target), "content": "<h1>New</h1>\n"}],
                }
            )

        self.assertTrue(any("existing file" in error for error in errors))

    def test_anchor_mode_rejects_no_op_replacement(self) -> None:
        errors = validate_proposal(
            {
                "action_class": "mutate",
                "edit_mode": "anchor",
                "intent": "localized_edit",
                "edit_scope": "single_span",
                "expected_existing_state": "existing_file",
                "recovery_strategy": "replan_step",
                "target_path": "app.js",
                "anchor": "sidebar",
                "replacement": "sidebar",
            }
        )

        self.assertTrue(any("must differ from anchor" in error for error in errors))

    def test_anchor_mode_rejects_placeholder_comment_replacement(self) -> None:
        errors = validate_proposal(
            {
                "action_class": "mutate",
                "edit_mode": "anchor",
                "intent": "localized_edit",
                "edit_scope": "single_span",
                "expected_existing_state": "existing_file",
                "recovery_strategy": "replan_step",
                "target_path": "app.js",
                "anchor": "renderSidebar();",
                "replacement": "// TODO: update this section",
            }
        )

        self.assertTrue(any("placeholder" in error for error in errors))

    def test_mutate_action_requires_explicit_contract_fields(self) -> None:
        errors = validate_proposal(
            {
                "action_class": "mutate",
                "edit_mode": "anchor",
                "target_path": "app.js",
                "anchor": "old",
                "replacement": "new",
            }
        )

        self.assertTrue(any("intent" in error for error in errors))
        self.assertTrue(any("edit_scope" in error for error in errors))
        self.assertTrue(any("expected_existing_state" in error for error in errors))
        self.assertTrue(any("recovery_strategy" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
