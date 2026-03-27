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
                "edit_mode": "named_function",
                "changed_files": ["/tmp/demo.py"],
                "content_hash": "abc123",
                "trace_path": ".shipyard/data/traces/demo.json",
                "proposal_summary": {
                    "provider": "openai",
                    "is_valid": True,
                },
                "code_graph_status": {
                    "ready": True,
                    "refresh_required": True,
                },
            }

            session_dir = store.append_run(state)
            loaded = store.load_latest_state("abc123")
            sessions = store.list_sessions()

            self.assertTrue(session_dir.endswith("abc123"))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["status"], "verified")
            self.assertEqual(loaded["request"]["instruction"], "demo")
            self.assertEqual(loaded["plan"]["provider"], "openai")
            self.assertEqual(loaded["execution"]["changed_files"], ["/tmp/demo.py"])
            self.assertEqual(sessions[0]["changed_files"], ["/tmp/demo.py"])
            self.assertEqual(sessions[0]["content_hash"], "abc123")


if __name__ == "__main__":
    unittest.main()
