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
        self.assertEqual(status["session"]["result_status"], "edited")


if __name__ == "__main__":
    unittest.main()
