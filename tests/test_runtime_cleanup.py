from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.runtime_cleanup import cleanup_runtime_data


class RuntimeCleanupTests(unittest.TestCase):
    def test_cleanup_runtime_data_prunes_old_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            traces = Path(tmpdir) / "traces"
            snapshots = Path(tmpdir) / "snapshots"
            workspaces = Path(tmpdir) / "workspace"
            sessions = Path(tmpdir) / "sessions"
            logs = Path(tmpdir) / "logs"
            specs = Path(tmpdir) / "specs"
            traces.mkdir()
            snapshots.mkdir()
            workspaces.mkdir()
            sessions.mkdir()
            logs.mkdir()
            specs.mkdir()
            for index in range(3):
                (traces / f"trace-{index}.json").write_text("x", encoding="utf-8")
                (snapshots / f"snap-{index}.txt").write_text("x", encoding="utf-8")
                (sessions / f"session-{index}.json").write_text("x", encoding="utf-8")
                (logs / f"log-{index}.jsonl").write_text("x", encoding="utf-8")
            (workspaces / "empty").mkdir()
            (specs / "empty").mkdir()

            with patch("shipyard.runtime_cleanup.TRACES_ROOT", traces), patch(
                "shipyard.runtime_cleanup.SNAPSHOTS_ROOT",
                snapshots,
            ), patch(
                "shipyard.runtime_cleanup.WORKSPACES_ROOT",
                workspaces,
            ), patch(
                "shipyard.runtime_cleanup.SESSIONS_ROOT",
                sessions,
            ), patch(
                "shipyard.runtime_cleanup.LOGS_ROOT",
                logs,
            ), patch(
                "shipyard.runtime_cleanup.DATA_ROOT",
                Path(tmpdir),
            ):
                result = cleanup_runtime_data(
                    keep_traces=1,
                    keep_snapshots=1,
                    keep_sessions=1,
                    keep_logs=1,
                )

            self.assertEqual(result["removed_traces"], 2)
            self.assertEqual(result["removed_snapshots"], 2)
            self.assertEqual(result["removed_sessions"], 2)
            self.assertEqual(result["removed_logs"], 2)
            self.assertEqual(result["removed_empty_workspaces"], 1)
            self.assertEqual(result["removed_empty_spec_dirs"], 1)


if __name__ == "__main__":
    unittest.main()
