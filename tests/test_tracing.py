from __future__ import annotations

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from shipyard.tracing import _build_troubleshooting_payload, write_troubleshooting_log


class TracingTests(unittest.TestCase):
    def test_troubleshooting_payload_omits_null_fields(self) -> None:
        payload = _build_troubleshooting_payload(
            {
                "session_id": "demo",
                "instruction": "Create a tiny repo with main.py and config.json",
                "status": "invalid_action_plan",
                "error": "missing config.json",
                "action_plan": {
                    "provider": "openai",
                    "is_valid": False,
                    "validation_errors": ["missing config.json"],
                    "actions": [
                        {
                            "instruction": "write main.py",
                            "valid": True,
                            "validation_errors": [],
                        }
                    ],
                },
                "human_gate": {
                    "action": "clarify_request",
                    "prompt": "Clarify the missing files or steps, then run again.",
                },
            }
        )

        self.assertNotIn("target_path", payload)
        self.assertNotIn("edit_mode", payload)
        self.assertNotIn("context", payload)
        self.assertEqual(payload["action_plan_summary"]["action_count"], 1)

    def test_demo_sessions_do_not_clobber_global_latest_troubleshooting(self) -> None:
        with TemporaryDirectory() as tmpdir, patch("shipyard.tracing.LOGS_ROOT", Path(tmpdir)):
            write_troubleshooting_log({"session_id": "web-mn88u4wi-rodmwq", "instruction": "real run", "status": "edited"})
            write_troubleshooting_log({"session_id": "demo", "instruction": "test run", "status": "failed"})
            write_troubleshooting_log({"session_id": "web-test", "instruction": "fake test web run", "status": "failed"})

            latest = (Path(tmpdir) / "latest-troubleshooting.json").read_text(encoding="utf-8")
            self.assertIn("web-mn88u4wi-rodmwq", latest)
            self.assertNotIn('"session_id": "demo"', latest)
            self.assertNotIn('"session_id": "web-test"', latest)
            self.assertTrue((Path(tmpdir) / "latest-demo-troubleshooting.json").exists())
            self.assertTrue((Path(tmpdir) / "latest-web-test-troubleshooting.json").exists())


if __name__ == "__main__":
    unittest.main()
