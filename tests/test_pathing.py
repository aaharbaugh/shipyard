from __future__ import annotations

import unittest

from shipyard.pathing import resolve_target_path


class PathingTests(unittest.TestCase):
    def test_explicit_target_path_wins(self) -> None:
        path, source = resolve_target_path(
            "/tmp/explicit.py",
            {"file_hint": "/tmp/hint.py"},
            "write_file",
            session_id="demo",
        )

        self.assertEqual(path, "/tmp/explicit.py")
        self.assertEqual(source, "explicit_target_path")

    def test_testing_mode_rebases_relative_target_into_workspace(self) -> None:
        path, source = resolve_target_path(
            "file1.txt",
            {"testing_mode": True},
            "write_file",
            session_id="demo",
        )

        self.assertIn("/.shipyard/data/workspace/demo/file1.txt", path)
        self.assertEqual(source, "sandboxed_target_path")

    def test_testing_mode_rebases_absolute_target_outside_data_root(self) -> None:
        path, source = resolve_target_path(
            "/tmp/explicit.py",
            {"testing_mode": True},
            "write_file",
            session_id="demo",
        )

        self.assertIn("/.shipyard/data/workspace/demo/explicit.py", path)
        self.assertEqual(source, "sandboxed_target_path")

    def test_file_hint_used_when_no_explicit_target(self) -> None:
        path, source = resolve_target_path(
            None,
            {"file_hint": "/tmp/hint.py"},
            "write_file",
            session_id="demo",
        )

        self.assertEqual(path, "/tmp/hint.py")
        self.assertEqual(source, "file_hint")

    def test_write_file_without_path_uses_managed_workspace(self) -> None:
        path, source = resolve_target_path(
            None,
            {},
            "write_file",
            session_id="demo",
        )

        self.assertIn("/demo/", path)
        self.assertTrue(path.endswith("scratch.py"))
        self.assertEqual(source, "managed_workspace")
        self.assertTrue(path.startswith("/"))

    def test_new_file_without_path_uses_unique_name(self) -> None:
        path, source = resolve_target_path(
            None,
            {},
            "write_file",
            session_id="demo",
            instruction="make a new file",
        )

        self.assertIn("/demo/", path)
        self.assertRegex(path, r"scratch-[0-9a-f]{6}\.py$")
        self.assertEqual(source, "managed_workspace")

    def test_write_file_without_path_infers_javascript_extension(self) -> None:
        path, source = resolve_target_path(
            None,
            {},
            "write_file",
            session_id="demo",
            instruction="make a file with javascript code",
        )

        self.assertIn("/demo/", path)
        self.assertRegex(path, r"scratch-[0-9a-f]{6}\.js$")
        self.assertEqual(source, "managed_workspace")

    def test_testing_mode_preserves_relative_shipyard_data_path(self) -> None:
        path, source = resolve_target_path(
            ".shipyard/data/workspace/demo/scratch.py",
            {"testing_mode": True},
            "write_file",
            session_id="demo",
        )

        self.assertIn("/.shipyard/data/workspace/demo/scratch.py", path)
        self.assertEqual(source, "explicit_target_path")


if __name__ == "__main__":
    unittest.main()
