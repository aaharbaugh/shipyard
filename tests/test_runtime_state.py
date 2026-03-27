from __future__ import annotations

import unittest

from shipyard.runtime_state import build_public_job, enrich_state_sections


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
                "tasks": [
                    {
                        "task_id": "helper-planner-task",
                        "role": "helper-planner",
                        "agent_type": "helper",
                        "goal": "Inspect and suggest a targeted edit plan",
                        "allowed_actions": ["read_file", "search_files"],
                        "status": "planned",
                    }
                ],
                "action_plan": {
                    "actions": [
                        {
                            "id": "step-1",
                            "role": "helper-planner",
                            "agent_type": "specialist",
                            "parent_task_id": "root",
                            "child_task_ids": ["step-2"],
                            "allowed_actions": ["read_file"],
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

        self.assertEqual(enriched["plan"]["task_count"], 2)
        self.assertEqual(enriched["tasks"][0]["task_id"], "helper-planner-task")
        self.assertEqual(enriched["tasks"][1]["task_id"], "step-1")
        self.assertEqual(enriched["tasks"][1]["agent_type"], "specialist")
        self.assertEqual(enriched["tasks"][1]["parent_task_id"], "root")

    def test_build_public_job_prefers_queue_tasks_for_live_drilldown(self) -> None:
        job = build_public_job(
            {
                "job_id": "job-1",
                "session_id": "demo",
                "state": {"instruction": "inspect repo"},
                "status": "running",
                "queue_state": "running",
                "tasks": [
                    {
                        "task_id": "run-job-1",
                        "role": "orchestrator",
                        "agent_type": "supervisor",
                        "goal": "inspect repo",
                        "status": "running",
                    },
                    {
                        "task_id": "step-1",
                        "role": "lead-agent",
                        "agent_type": "primary",
                        "goal": "Read main.py",
                        "status": "running",
                    },
                ],
            }
        )

        self.assertEqual(job["tasks"][0]["task_id"], "run-job-1")
        self.assertEqual(job["tasks"][1]["task_id"], "step-1")


if __name__ == "__main__":
    unittest.main()
