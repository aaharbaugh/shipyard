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
                "troubleshooting_path": ".shipyard/data/logs/demo-troubleshooting.json",
            }
        )

        self.assertEqual(enriched["request"]["instruction"], "write hello")
        self.assertEqual(enriched["plan"]["edit_mode"], "write_file")
        self.assertEqual(enriched["plan"]["provider"], "openai")
        self.assertEqual(enriched["execution"]["status"], "verified")
        self.assertEqual(enriched["execution"]["changed_files"], ["/tmp/demo.py"])
        self.assertEqual(enriched["execution"]["file_preview"], "print('hi')")
        self.assertEqual(enriched["execution"]["content_hash"], "abc123")
        self.assertEqual(enriched["artifacts"]["trace_path"], ".shipyard/data/traces/demo.json")
        self.assertEqual(
            enriched["artifacts"]["troubleshooting_path"],
            ".shipyard/data/logs/demo-troubleshooting.json",
        )

    def test_enrich_state_sections_prefers_request_instruction(self) -> None:
        enriched = enrich_state_sections(
            {
                "instruction": "step instruction",
                "request_instruction": "original user instruction",
                "status": "edited",
            }
        )

        self.assertEqual(enriched["request"]["instruction"], "original user instruction")

    def test_enrich_state_sections_compacts_to_public_schema(self) -> None:
        enriched = enrich_state_sections(
            {
                "proposal_summary": {"is_valid": True},
                "spec_bundle": {"created": True},
            }
        )

        self.assertEqual(sorted(enriched.keys()), ["artifacts", "execution", "human_gate", "plan", "request", "steps", "tasks"])
        self.assertTrue(enriched["artifacts"]["spec_created"])

    def test_enrich_state_sections_includes_tasks(self) -> None:
        enriched = enrich_state_sections(
            {
                "action_plan": {
                    "actions": [
                        {
                            "id": "step-1",
                            "instruction": "Read main.py",
                            "edit_mode": "read_file",
                            "depends_on": [],
                            "inputs_from": [],
                            "valid": True,
                        }
                    ]
                },
                "action_steps": [
                    {
                        "id": "step-1",
                        "instruction": "Read main.py",
                        "edit_mode": "read_file",
                        "status": "observed",
                        "no_op": True,
                    }
                ],
            }
        )

        self.assertEqual(enriched["plan"]["task_count"], 1)
        self.assertEqual(enriched["tasks"][0]["task_id"], "step-1")


if __name__ == "__main__":
    unittest.main()
