from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shipyard.tools.edit_file import AnchorEditError, apply_anchor_edit, validate_anchor_edit


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


if __name__ == "__main__":
    unittest.main()
