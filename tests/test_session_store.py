from __future__ import annotations

import tempfile
import unittest

from shipyard.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_append_run_and_load_latest_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(tmpdir)
            state = {
                "session_id": "abc123",
                "instruction": "demo",
                "status": "verified",
                "trace_path": ".shipyard/traces/demo.json",
            }

            session_dir = store.append_run(state)
            loaded = store.load_latest_state("abc123")

            self.assertTrue(session_dir.endswith("abc123"))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["status"], "verified")


if __name__ == "__main__":
    unittest.main()
