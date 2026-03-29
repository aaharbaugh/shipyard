from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shipyard.supervisor import should_use_supervisor, plan_subtasks
from shipyard.worker_orchestrator import execute_workers, _detect_conflicts


class SupervisorDetectionTests(unittest.TestCase):
    def test_simple_instruction_uses_single_agent(self) -> None:
        self.assertFalse(should_use_supervisor({
            "instruction": "fix the bug in app.js",
        }))

    def test_short_instruction_uses_single_agent(self) -> None:
        self.assertFalse(should_use_supervisor({
            "instruction": "add logging",
        }))

    def test_refactor_keyword_triggers_supervisor(self) -> None:
        self.assertTrue(should_use_supervisor({
            "instruction": "refactor the authentication system to use JWT tokens across all services",
            "broad_context": {"file_tree": ["api/routes.ts", "api/auth.ts", "web/auth.tsx", "web/app.tsx", "shared/types.ts", "docs/auth.md"], "discovered_docs": ["docs/auth.md"]},
        }))

    def test_explicit_multi_agent_request(self) -> None:
        self.assertTrue(should_use_supervisor({
            "instruction": "use multi-agent to update the api and web packages",
        }))

    def test_monorepo_with_multiple_areas_mentioned(self) -> None:
        self.assertTrue(should_use_supervisor({
            "instruction": "update the api and web packages to use the new shared types",
            "broad_context": {
                "file_tree": ["api/src/index.ts", "api/src/routes.ts", "web/src/App.tsx", "web/src/main.tsx", "shared/types.ts", "package.json"],
            },
        }))

    def test_implement_keyword_triggers_supervisor(self) -> None:
        self.assertTrue(should_use_supervisor({
            "instruction": "implement the new feature described in the spec",
            "broad_context": {"file_tree": ["src/app.ts", "src/routes.ts", "src/db.ts", "src/types.ts", "docs/spec.md", "package.json"]},
        }))


class ConflictDetectionTests(unittest.TestCase):
    def test_no_conflicts_with_separate_files(self) -> None:
        results = [
            {"worker_id": "w1", "changed_files": ["/a.py"]},
            {"worker_id": "w2", "changed_files": ["/b.py"]},
        ]
        conflicts = _detect_conflicts(results)
        self.assertEqual(conflicts, [])

    def test_detects_same_file_edited_by_two_workers(self) -> None:
        results = [
            {"worker_id": "w1", "changed_files": ["/shared.py"]},
            {"worker_id": "w2", "changed_files": ["/shared.py", "/other.py"]},
        ]
        conflicts = _detect_conflicts(results)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["file"], "/shared.py")
        self.assertEqual(set(conflicts[0]["workers"]), {"w1", "w2"})

    def test_no_conflicts_when_empty(self) -> None:
        self.assertEqual(_detect_conflicts([]), [])


class WorkerOrchestratorTests(unittest.TestCase):
    def test_single_worker_runs_directly(self) -> None:
        mock_run = Mock(return_value={
            "status": "edited",
            "changed_files": ["/tmp/app.js"],
            "action_steps": [{"id": "s1", "status": "edited"}],
            "file_transactions": [],
        })

        result = execute_workers(
            {"session_id": "test", "instruction": "fix it"},
            [{"id": "w1", "instruction": "fix app.js", "scope": ".", "files": [], "depends_on": []}],
            mock_run,
        )

        self.assertEqual(result["status"], "edited")
        self.assertIn("/tmp/app.js", result["changed_files"])
        self.assertTrue(result["multi_agent"])
        self.assertEqual(result["worker_count"], 1)

    def test_parallel_workers_merge_results(self) -> None:
        call_count = 0
        def mock_run(app, state, plan, cb=None):
            nonlocal call_count
            call_count += 1
            return {
                "status": "edited",
                "changed_files": [f"/tmp/file{call_count}.py"],
                "action_steps": [{"id": f"s{call_count}", "status": "edited"}],
                "file_transactions": [],
            }

        result = execute_workers(
            {"session_id": "test", "instruction": "fix everything"},
            [
                {"id": "w1", "instruction": "fix a", "scope": "a/", "files": [], "depends_on": []},
                {"id": "w2", "instruction": "fix b", "scope": "b/", "files": [], "depends_on": []},
            ],
            mock_run,
        )

        self.assertEqual(result["status"], "edited")
        self.assertEqual(len(result["changed_files"]), 2)
        self.assertEqual(result["worker_count"], 2)
        self.assertEqual(result["wave_count"], 1)  # both in same wave
        self.assertEqual(len(result["conflicts"]), 0)

    def test_sequential_workers_respect_dependencies(self) -> None:
        execution_order = []
        def mock_run(app, state, plan, cb=None):
            wid = state.get("context", {}).get("helper_notes", "")
            execution_order.append(wid)
            return {
                "status": "edited",
                "changed_files": [],
                "action_steps": [],
                "file_transactions": [],
            }

        result = execute_workers(
            {"session_id": "test", "instruction": "migrate"},
            [
                {"id": "w1", "instruction": "update types", "scope": "shared/", "files": [], "depends_on": []},
                {"id": "w2", "instruction": "update api", "scope": "api/", "files": [], "depends_on": ["w1"]},
            ],
            mock_run,
        )

        self.assertEqual(result["wave_count"], 2)  # w1 in wave 1, w2 in wave 2

    def test_deadlock_detected(self) -> None:
        result = execute_workers(
            {"session_id": "test", "instruction": "deadlock"},
            [
                {"id": "w1", "instruction": "a", "scope": ".", "files": [], "depends_on": ["w2"]},
                {"id": "w2", "instruction": "b", "scope": ".", "files": [], "depends_on": ["w1"]},
            ],
            Mock(),
        )

        blocked = [wr for wr in result["worker_results"] if wr["status"] == "blocked"]
        self.assertEqual(len(blocked), 2)


if __name__ == "__main__":
    unittest.main()
