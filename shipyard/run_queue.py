from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Callable

from .runtime_state import build_public_job
from .router import route_message
from .state import ShipyardState


TASK_LABELS = {
    "accepted": "Accepted",
    "planning": "Planning",
    "spec_bundle": "Spec Check",
    "lead_agent": "Lead Agent",
    "step_retry": "Retrying",
    "verifying": "Verifying",
    "result_ready": "Edit Result",
    "graph_sync": "Graph Sync",
    "persisting": "Persisting",
    "cancel_requested": "Cancel Requested",
    "completed": "Completed",
}

EVENT_TO_STATUS = {
    "accepted": "queued",
    "planning": "planning",
    "spec_bundle": "planning",
    "lead_agent": "running",
    "step_retry": "running",
    "verifying": "verifying",
    "result_ready": "running",
    "graph_sync": "running",
    "persisting": "running",
    "cancel_requested": "running",
}

RESULT_TO_QUEUE_STATUS = {
    "verified": "completed",
    "edited": "completed",
    "observed": "completed",
    "invalid_action_plan": "blocked",
    "invalid_proposal": "blocked",
    "edit_blocked": "blocked",
    "graph_unavailable": "blocked",
    "awaiting_edit_spec": "blocked",
    "verification_failed": "failed",
    "failed_after_retries": "failed",
    "failed": "failed",
    "blocked": "blocked",
    "cancelled": "cancelled",
}


class RunQueue:
    def __init__(self, runner: Callable[[ShipyardState, Callable[[str, dict[str, Any]], None]], dict[str, Any]]) -> None:
        self._runner = runner
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def enqueue(self, state: ShipyardState) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        session_id = state.get("session_id", "")
        job = {
            "job_id": job_id,
            "session_id": session_id,
            "state": dict(state),
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": None,
            "finished_at": None,
            "result_status": None,
            "target_path": state.get("target_path"),
            "error": None,
            "current_task": "Waiting",
            "queue_state": "queued",
            "task_events": [],
            "agents": _infer_agents(state),
            "routing": route_message(
                state.get("instruction", ""),
                {"previous_text": state.get("context", {}).get("previous_text")},
            ),
        }
        with self._condition:
            self._jobs[job_id] = job
            position = self._insert_by_priority(job_id)
            is_running = self._active_job_id is not None
            self._condition.notify()
        return {
            "status": "queued",
            "job_id": job_id,
            "session_id": session_id,
            "queue_position": position if is_running else max(position - 1, 0),
            "queue_depth": position + (1 if is_running else 0),
            "current_task": "Waiting",
            "routing": job["routing"],
        }

    def record_direct_run(self, state: ShipyardState, result: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        session_id = state.get("session_id", "")
        job = {
            "job_id": job_id,
            "session_id": session_id,
            "state": dict(state),
            "status": "completed" if result.get("status") not in {"failed", "invalid_proposal", "edit_blocked"} else "failed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "result_status": result.get("status"),
            "target_path": result.get("target_path"),
            "error": result.get("error"),
            "current_task": "Completed",
            "queue_state": "completed" if result.get("status") not in {"failed", "invalid_proposal", "edit_blocked", "blocked"} else "failed",
            "task_events": list(result.get("task_events", [])),
            "agents": _infer_agents(state),
            "routing": route_message(
                state.get("instruction", ""),
                {"previous_text": state.get("context", {}).get("previous_text")},
            ),
        }
        if not job["task_events"]:
            job["task_events"] = [
                {
                    "event": "accepted",
                    "label": TASK_LABELS["accepted"],
                    "timestamp": job["created_at"],
                    "payload": {"session_id": session_id},
                },
                {
                    "event": "completed",
                    "label": TASK_LABELS["completed"],
                    "timestamp": job["finished_at"],
                    "payload": {"status": result.get("status")},
                },
            ]
        with self._condition:
            self._jobs[job_id] = job
        return _public_job(job)

    def get_status(self, session_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            active = self._jobs.get(self._active_job_id) if self._active_job_id else None
            queued = [self._jobs[job_id] for job_id in self._queue]
            session_job = None
            if session_id:
                matching = [job for job in self._jobs.values() if job.get("session_id") == session_id]
                if matching:
                    session_job = sorted(matching, key=lambda item: item["created_at"])[-1]
            return {
                "active": _public_job(active) if active else None,
                "queued": [_public_job(job) for job in queued],
                "session": _public_job(session_job) if session_job else None,
            }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            return _public_job(self._jobs.get(job_id))

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job_id in self._queue:
                self._queue = deque(queued_id for queued_id in self._queue if queued_id != job_id)
                job["status"] = "cancelled"
                job["queue_state"] = "cancelled"
                job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                self._record_progress(job_id, "completed", {"status": "cancelled"})
                return _public_job(job)
            job["cancel_requested"] = True
            self._record_progress(job_id, "cancel_requested", {"status": "cancel_requested"})
            return _public_job(job)

    def _work_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job_id = self._queue.popleft()
                self._active_job_id = job_id
                job = self._jobs[job_id]
                job["status"] = "planning"
                job["queue_state"] = "planning"
                job["started_at"] = datetime.now().isoformat(timespec="seconds")
                job["current_task"] = "Accepted"

            try:
                job["state"]["cancel_check"] = lambda job_id=job_id: self._is_cancel_requested(job_id)
                result = self._runner(job["state"], lambda event, payload: self._record_progress(job_id, event, payload))
                job["result"] = result
                job["status"] = RESULT_TO_QUEUE_STATUS.get(result.get("status"), "completed")
                job["queue_state"] = job["status"]
                job["result_status"] = result.get("status")
                job["target_path"] = result.get("execution", {}).get("target_path") or result.get("target_path")
                job["error"] = result.get("execution", {}).get("error") or result.get("error")
                self._record_progress(job_id, "completed", {"status": result.get("status"), "error": job["error"]})
            except Exception as exc:
                job["status"] = "failed"
                job["queue_state"] = "failed"
                job["error"] = str(exc)
                self._record_progress(job_id, "completed", {"status": "failed", "error": str(exc)})
            finally:
                job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                with self._condition:
                    self._active_job_id = None

    def _insert_by_priority(self, job_id: str) -> int:
        priority = self._jobs[job_id]["routing"].get("priority_score", 0)
        position = len(self._queue)
        for index, queued_id in enumerate(self._queue):
            queued_priority = self._jobs[queued_id]["routing"].get("priority_score", 0)
            if priority > queued_priority:
                self._queue.insert(index, job_id)
                return index + 1
        self._queue.append(job_id)
        return position + 1

    def _record_progress(self, job_id: str, event: str, payload: dict[str, Any]) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if not job:
                return
            label = TASK_LABELS.get(event, event.replace("_", " ").title())
            job["current_task"] = label
            next_status = EVENT_TO_STATUS.get(event)
            if next_status:
                job["status"] = next_status
                job["queue_state"] = next_status
            task_events = list(job.get("task_events", []))
            task_events.append(
                {
                    "event": event,
                    "label": label,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "payload": payload,
                }
            )
            job["task_events"] = task_events[-12:]

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._condition:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))


def _infer_agents(state: ShipyardState) -> list[str]:
    context = state.get("context", {})
    agents = ["lead-agent"]
    if context.get("function_name"):
        agents.append("helper-planner:function")
    else:
        agents.append("helper-planner:edit")
    return agents


def _public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    return build_public_job(job)
