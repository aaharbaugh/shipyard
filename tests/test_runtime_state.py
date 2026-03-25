from __future__ import annotations

import unittest

from shipyard.runtime_state import enrich_state_sections


class RuntimeStateTests(unittest.TestCase):
    def test_enrich_state_sections_groups_runtime_state(self) -> None:
        enriched = enrich_state_sections(
            {
                "session_id": "demo",
                "instruction": "write hello",
                "target_path": "/tmp/demo.py",
                "proposal_mode": "openai",
                "context": {"file_hint": "/tmp/demo.py"},
                "verification_commands": ["python3 -m py_compile /tmp/demo.py"],
                "proposal_summary": {
                    "provider": "openai",
                    "provider_reason": "planned",
                    "edit_mode": "write_file",
                    "target_path_source": "explicit_target_path",
                    "is_valid": True,
                    "validation_errors": [],
                },
                "status": "verified",
                "changed_files": ["/tmp/demo.py"],
                "file_preview": "print('hi')",
                "content_hash": "abc123",
                "trace_path": ".shipyard/data/traces/demo.json",
            }
        )

        self.assertEqual(enriched["request"]["instruction"], "write hello")
        self.assertEqual(enriched["plan"]["edit_mode"], "write_file")
        self.assertEqual(enriched["plan"]["target_path_source"], "explicit_target_path")
        self.assertEqual(enriched["execution"]["status"], "verified")
        self.assertEqual(enriched["execution"]["changed_files"], ["/tmp/demo.py"])
        self.assertEqual(enriched["execution"]["file_preview"], "print('hi')")
        self.assertEqual(enriched["execution"]["content_hash"], "abc123")
        self.assertIsNone(enriched["graph"]["sync_attempted"])
        self.assertEqual(enriched["artifacts"]["trace_path"], ".shipyard/data/traces/demo.json")

    def test_enrich_state_sections_exposes_graph_and_spec_summaries(self) -> None:
        enriched = enrich_state_sections(
            {
                "proposal_summary": {"is_valid": True},
                "code_graph_status": {
                    "ready": True,
                    "available": True,
                    "reason": "ready",
                    "index_state": {"stale": False},
                    "live_graph_state": {"populated": True},
                },
                "graph_sync": {"attempted": True, "ok": True},
                "spec_bundle": {"created": True},
            }
        )

        self.assertTrue(enriched["graph"]["ready"])
        self.assertTrue(enriched["graph"]["live_graph_populated"])
        self.assertTrue(enriched["graph"]["sync_attempted"])
        self.assertTrue(enriched["artifacts"]["spec_created"])


if __name__ == "__main__":
    unittest.main()
