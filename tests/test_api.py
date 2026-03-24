from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shipyard.api import (
    InstructionRequest,
    get_session,
    health,
    instruct,
    list_sessions,
)
from shipyard.session_store import SessionStore


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        import shipyard.api as api_module

        self.api_module = api_module
        self.original_store = api_module.session_store
        api_module.session_store = SessionStore(self.tempdir.name)
        self.addCleanup(self._restore_store)

    def _restore_store(self) -> None:
        self.api_module.session_store = self.original_store

    def test_health_returns_ok(self) -> None:
        self.assertEqual(health(), {"status": "ok"})

    def test_instruct_persists_session_result(self) -> None:
        target = Path(self.tempdir.name) / "demo.py"
        target.write_text('print("api-old")\n', encoding="utf-8")

        result = instruct(
            InstructionRequest(
                session_id="api-test",
                instruction='replace "api-old" with "api-new"',
                proposal_mode="heuristic",
                context={"file_hint": str(target)},
                verification_commands=[f"python3 -m py_compile {target}"],
            )
        )

        self.assertEqual(result["status"], "verified")
        sessions = list_sessions()["sessions"]
        self.assertTrue(any(item["session_id"] == "api-test" for item in sessions))

        loaded = get_session("api-test")
        self.assertEqual(loaded["status"], "verified")
        self.assertEqual(target.read_text(encoding="utf-8"), 'print("api-new")\n')


if __name__ == "__main__":
    unittest.main()
