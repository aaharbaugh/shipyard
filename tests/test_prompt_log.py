from __future__ import annotations

import tempfile
import unittest

from shipyard.prompt_log import PromptLog


class PromptLogTests(unittest.TestCase):
    def test_append_persists_prompt_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = PromptLog(f"{tmpdir}/prompt_log.jsonl")
            path = log.append(
                {
                    "session_id": "demo-session",
                    "instruction": "write hello world to new file",
                    "target_path": "/tmp/demo.py",
                    "proposal_mode": "openai",
                    "context": {"file_hint": "/tmp/demo.py"},
                    "verification_commands": ["python3 -m py_compile /tmp/demo.py"],
                }
            )

            entries = log.load()

        self.assertTrue(path.endswith("prompt_log.jsonl"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["session_id"], "demo-session")
        self.assertEqual(entries[0]["instruction"], "write hello world to new file")
        self.assertEqual(entries[0]["proposal_mode"], "openai")


if __name__ == "__main__":
    unittest.main()
