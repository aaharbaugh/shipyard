from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from shipyard.api import (
    GraphIndexRequest,
    InstructionRequest,
    graph_index,
    graph_sync,
    queue_instruct,
    queue_status,
    planner_status,
    graph_status,
    workspace_status,
    workspace_temp,
    WorkspaceCreateRequest,
    get_session_history,
    get_session,
    health,
    instruct,
    list_sessions,
    workbench,
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

    def test_workbench_returns_html(self) -> None:
        response = workbench()
        self.assertIn("Shipyard Workbench", response.body.decode("utf-8"))
        self.assertIn("Side Panel", response.body.decode("utf-8"))
        self.assertIn("LLM planning is active", response.body.decode("utf-8"))
        self.assertIn("Details", response.body.decode("utf-8"))
        self.assertIn("Graph", response.body.decode("utf-8"))
        self.assertIn("Raw", response.body.decode("utf-8"))
        self.assertIn("Live Graph", response.body.decode("utf-8"))
        self.assertIn("Connectivity", response.body.decode("utf-8"))
        self.assertIn("New Temp Workspace", response.body.decode("utf-8"))
        self.assertIn("Rebuild Graph", response.body.decode("utf-8"))

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

        history = get_session_history("api-test")["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "verified")
        self.assertEqual(history[0]["changed_files"], [str(target)])

    def test_graph_status_returns_payload(self) -> None:
        with patch(
            "shipyard.api.inspect_code_graph_status",
            return_value={"ready": False, "available": True, "source": "cgr_stats"},
        ):
            result = graph_status()

        self.assertEqual(result["source"], "cgr_stats")

    def test_planner_status_reports_configuration(self) -> None:
        with patch(
            "shipyard.api.get_planner_status",
            return_value={"default_mode": "openai", "proposal_model": "gpt-5.4-mini"},
        ):
            result = planner_status()

        self.assertEqual(result["default_mode"], "openai")

    def test_queue_instruct_does_not_bypass_llm_when_openai_is_configured(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.api.run_queue.enqueue",
            return_value={"status": "queued", "job_id": "job-1", "session_id": "api-test"},
        ) as mocked_enqueue:
            result = queue_instruct(
                InstructionRequest(
                    session_id="api-test",
                    instruction="create 3 python files",
                    context={"testing_mode": True},
                )
            )

        mocked_enqueue.assert_called_once()
        self.assertEqual(result["status"], "queued")

    def test_graph_index_returns_result(self) -> None:
        with patch(
            "shipyard.api.index_code_graph",
            return_value={"ok": True, "reason": "Code graph index created."},
        ):
            result = graph_index(GraphIndexRequest())

        self.assertTrue(result["ok"])

    def test_graph_index_raises_http_error_on_failure(self) -> None:
        with patch(
            "shipyard.api.index_code_graph",
            return_value={"ok": False, "reason": "failed"},
        ):
            with self.assertRaises(HTTPException) as context:
                graph_index(GraphIndexRequest())

        self.assertEqual(context.exception.status_code, 400)

    def test_graph_sync_returns_result(self) -> None:
        with patch(
            "shipyard.api.sync_live_code_graph",
            return_value={
                "ok": True,
                "reason": "Live code graph synchronized.",
                "ready": True,
                "index_state": {"has_index": True},
                "live_graph_state": {"node_count": 10},
            },
        ):
            result = graph_sync(GraphIndexRequest())

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "graph_rebuilt")
        self.assertEqual(result["code_graph_status"]["live_graph_state"]["node_count"], 10)

    def test_workspace_status_returns_paths(self) -> None:
        result = workspace_status()
        self.assertIn("workspace_root", result)
        self.assertIn("live_graph_runtime_root", result)

    def test_workspace_temp_creates_directory(self) -> None:
        result = workspace_temp(WorkspaceCreateRequest(prefix="api"))
        self.assertTrue(Path(result["path"]).exists())

    def test_queue_instruct_returns_queued_status(self) -> None:
        with patch.object(
            self.api_module.run_queue,
            "enqueue",
            return_value={"status": "queued", "session_id": "demo", "job_id": "job1", "current_task": "Waiting"},
        ):
            result = queue_instruct(
                InstructionRequest(
                    session_id="demo",
                    instruction='replace "a" with "b"',
                )
            )

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["session_id"], "demo")

    def test_queue_instruct_writes_request_receipt_for_web_session(self) -> None:
        with patch.object(
            self.api_module.run_queue,
            "enqueue",
            return_value={"status": "queued", "session_id": "web-demo", "job_id": "job1", "current_task": "Waiting"},
        ), patch("shipyard.api.LOGS_ROOT", Path(self.tempdir.name) / "logs"):
            result = queue_instruct(
                InstructionRequest(
                    session_id="web-demo",
                    instruction="inspect the repo",
                )
            )

        self.assertEqual(result["status"], "queued")
        receipt = Path(self.tempdir.name) / "logs" / "latest-web-demo-request.json"
        self.assertTrue(receipt.exists())
        self.assertIn("inspect the repo", receipt.read_text(encoding="utf-8"))

    def test_queue_status_returns_payload(self) -> None:
        with patch.object(
            self.api_module.run_queue,
            "get_status",
            return_value={"active": None, "queued": [], "session": None},
        ):
            result = queue_status("demo")

        self.assertEqual(result["queued"], [])

    def test_queue_instruct_queues_trivial_testing_mode_request_without_openai(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch.object(
            self.api_module.run_queue,
            "enqueue",
            return_value={"status": "queued", "job_id": "job-direct", "session_id": "demo"},
        ) as enqueue_mock:
            result = queue_instruct(
                InstructionRequest(
                    session_id="demo",
                    instruction="make a new file",
                    context={"testing_mode": True},
                )
            )

        self.assertEqual(result["status"], "queued")
        enqueue_mock.assert_called_once()
        self.assertEqual(result["job_id"], "job-direct")


if __name__ == "__main__":
    unittest.main()
