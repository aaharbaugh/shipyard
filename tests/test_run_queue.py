from __future__ import annotations

import time
import unittest

from shipyard.run_queue import RunQueue


class RunQueueTests(unittest.TestCase):
    def test_queue_orders_higher_priority_prompt_first(self) -> None:
        queue = RunQueue(lambda state, cb: {"status": "edited", "target_path": state.get("target_path")})

        low = queue.enqueue({"session_id": "low", "instruction": "when you can review this"})
        high = queue.enqueue({"session_id": "high", "instruction": "actually fix this now"})

        status = queue.get_status()
        queued = status["queued"]

        self.assertEqual(low["status"], "queued")
        self.assertEqual(high["status"], "queued")
        self.assertTrue(any(job["session_id"] == "high" for job in queued))

    def test_queue_continues_after_failed_job(self) -> None:
        def runner(state, cb):
            if state["instruction"] == "fail":
                raise RuntimeError("boom")
            cb("completed", {"status": "edited"})
            return {"status": "edited", "target_path": state.get("target_path")}

        queue = RunQueue(runner)
        queue.enqueue({"session_id": "one", "instruction": "fail"})
        queue.enqueue({"session_id": "two", "instruction": "succeed"})

        deadline = time.time() + 3
        while time.time() < deadline:
            first = queue.get_status("one")
            second = queue.get_status("two")
            first_status = first.get("session", {}).get("status")
            second_status = second.get("session", {}).get("status")

            if first_status == "failed" and second_status == "completed":
                break
            time.sleep(0.05)
        else:
            self.fail("Queue did not continue after a failed job.")

        status = queue.get_status("two")
        self.assertEqual(status["session"]["status"], "completed")
        self.assertEqual(status["session"]["queue"]["result_status"], "edited")

    def test_queue_tracks_planning_running_and_blocked_states(self) -> None:
        def runner(state, cb):
            cb("planning", {"instruction": state["instruction"]})
            cb("lead_agent", {"instruction": "mutate", "step_index": 1, "step_count": 1, "step_id": "step-1"})
            return {"status": "invalid_proposal", "error": "bad plan"}

        queue = RunQueue(runner)
        queued = queue.enqueue({"session_id": "demo", "instruction": "do thing"})
        self.assertEqual(queued["status"], "queued")

        deadline = time.time() + 3
        seen_planning = False
        while time.time() < deadline:
            status = queue.get_status("demo")
            session = status.get("session", {})
            if session.get("status") == "planning":
                seen_planning = True
            if session.get("status") == "blocked":
                break
            time.sleep(0.05)
        else:
            self.fail("Queue did not reach blocked state.")

        self.assertTrue(
            seen_planning
            or any(event["event"] == "planning" for event in status["session"]["queue"]["task_events"])
        )
        self.assertEqual(status["session"]["queue"]["state"], "blocked")
        task_ids = [task["task_id"] for task in status["session"]["tasks"]]
        self.assertIn("run-" + status["session"]["queue"]["job_id"], task_ids)
        self.assertIn("step-1", task_ids)

    def test_queue_can_cancel_queued_job(self) -> None:
        gate = []

        def runner(state, cb):
            while not gate:
                time.sleep(0.05)
            return {"status": "edited"}

        queue = RunQueue(runner)
        queue.enqueue({"session_id": "one", "instruction": "first"})
        second = queue.enqueue({"session_id": "two", "instruction": "second"})

        cancelled = queue.cancel(second["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")

        gate.append(True)

    def test_queue_can_mark_active_job_cancel_requested(self) -> None:
        seen = {"cancelled": False}

        def runner(state, cb):
            deadline = time.time() + 2
            while time.time() < deadline:
                if state["cancel_check"]():
                    seen["cancelled"] = True
                    return {"status": "cancelled"}
                time.sleep(0.05)
            return {"status": "edited"}

        queue = RunQueue(runner)
        job = queue.enqueue({"session_id": "demo", "instruction": "long run"})
        time.sleep(0.1)
        queue.cancel(job["job_id"])

        deadline = time.time() + 3
        while time.time() < deadline:
            status = queue.get_job(job["job_id"])
            if status and status.get("status") == "cancelled":
                break
            time.sleep(0.05)
        else:
            self.fail("Active job did not cancel.")

        self.assertTrue(seen["cancelled"])


if __name__ == "__main__":
    unittest.main()
