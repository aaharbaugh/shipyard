from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shipyard.tools.edit_file import AnchorEditError, apply_anchor_edit, apply_symbol_rename, validate_anchor_edit


class EditFileTests(unittest.TestCase):
    def test_validate_anchor_edit_rejects_missing_anchor(self) -> None:
        with self.assertRaises(AnchorEditError):
            validate_anchor_edit('print("hello")', 'print("missing")')

    def test_validate_anchor_edit_rejects_multiple_matches(self) -> None:
        content = 'print("same")\nprint("same")\n'
        with self.assertRaises(AnchorEditError):
            validate_anchor_edit(content, 'print("same")')

    def test_apply_anchor_edit_replaces_one_unique_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            path.write_text('print("old")\n', encoding="utf-8")

            updated = apply_anchor_edit(
                str(path),
                'print("old")',
                'print("new")',
            )

            self.assertEqual(updated, 'print("new")\n')
            self.assertEqual(path.read_text(encoding="utf-8"), 'print("new")\n')

    def test_apply_anchor_edit_replaces_middle_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("hahaha", encoding="utf-8")

            updated = apply_anchor_edit(
                str(path),
                "ha",
                "ho",
                "middle",
            )

            self.assertEqual(updated, "hahoha")
            self.assertEqual(path.read_text(encoding="utf-8"), "hahoha")

    def test_apply_anchor_edit_uses_case_insensitive_single_match_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("HELLO world", encoding="utf-8")

            updated = apply_anchor_edit(
                str(path),
                "hello",
                "goodbye",
            )

            self.assertEqual(updated, "goodbye world")
            self.assertEqual(path.read_text(encoding="utf-8"), "goodbye world")

    def test_apply_symbol_rename_updates_all_whole_word_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.py"
            path.write_text(
                "base = 1\nvalue = base + 2\nbase_value = base\n",
                encoding="utf-8",
            )

            updated = apply_symbol_rename(str(path), "base", "boos")

            self.assertIn("boos = 1", updated)
            self.assertIn("value = boos + 2", updated)
            self.assertIn("base_value = boos", updated)
            self.assertNotIn(" base ", f" {updated} ")


if __name__ == "__main__":
    unittest.main()
