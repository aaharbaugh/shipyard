from __future__ import annotations

from typing import Any

from .state import ShipyardState


def enrich_state_sections(state: ShipyardState) -> ShipyardState:
    action_plan = dict(state.get("action_plan", {}) or {})
    proposal_summary = dict(state.get("proposal_summary", {}) or {})
    tasks = _build_tasks(state)
    execution = {
        "status": state.get("status"),
        "error": state.get("error"),
        "edit_applied": state.get("edit_applied"),
        "changed_files": list(state.get("changed_files", []) or []),
        "target_path": state.get("target_path"),
        "file_preview": state.get("file_preview"),
        "file_preview_truncated": state.get("file_preview_truncated"),
        "content_hash": state.get("content_hash"),
        "no_op": state.get("no_op"),
        "tool_output": state.get("tool_output"),
        "edit_attempts": state.get("edit_attempts"),
        "max_edit_attempts": state.get("max_edit_attempts"),
        "reverted_to_snapshot": state.get("reverted_to_snapshot"),
        "target_existed_before_edit": state.get("target_existed_before_edit"),
    }
    public_state: ShipyardState = {
        "request": _compact_object(
            {
                "instruction": state.get("request_instruction") or state.get("instruction") or state.get("request", {}).get("instruction"),
                "target_path": state.get("request", {}).get("target_path") if state.get("request") else state.get("target_path"),
                "context": state.get("context", {}) or state.get("request", {}).get("context", {}),
                "verification_commands": state.get("verification_commands", []) or state.get("request", {}).get("verification_commands", []),
            }
        ),
        "plan": _compact_object(
            {
                "provider": action_plan.get("provider") or proposal_summary.get("provider") or state.get("plan", {}).get("provider"),
                "provider_reason": action_plan.get("provider_reason") or proposal_summary.get("provider_reason") or state.get("plan", {}).get("provider_reason"),
                "valid": action_plan.get("is_valid", proposal_summary.get("is_valid", state.get("plan", {}).get("valid"))),
                "validation_errors": action_plan.get("validation_errors", proposal_summary.get("validation_errors", state.get("plan", {}).get("validation_errors", []))),
                "action_count": len(action_plan.get("actions", []) or []),
                "task_count": len(tasks),
                "target_path": state.get("target_path") or state.get("plan", {}).get("target_path"),
                "edit_mode": state.get("edit_mode") or proposal_summary.get("edit_mode") or state.get("plan", {}).get("edit_mode"),
            }
        ),
        "steps": _build_steps(state),
        "tasks": tasks,
        "execution": _compact_object(execution),
        "human_gate": _compact_object(state.get("human_gate", {}) or {}),
        "artifacts": _compact_object(
            {
                "trace_path": state.get("trace_path") or state.get("artifacts", {}).get("trace_path"),
                "troubleshooting_path": state.get("troubleshooting_path") or state.get("artifacts", {}).get("troubleshooting_path"),
                "snapshot_path": state.get("snapshot_path") or state.get("artifacts", {}).get("snapshot_path"),
                "spec_created": (state.get("spec_bundle", {}) or {}).get("created", state.get("artifacts", {}).get("spec_created")),
            }
        ),
    }
    if state.get("session_id") is not None:
        public_state["session_id"] = state.get("session_id")
    if state.get("status") is not None:
        public_state["status"] = state.get("status")
    return public_state


def build_public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None

    base = enrich_state_sections(job.get("result") or job.get("state") or {})
    base["session_id"] = job.get("session_id") or base.get("session_id")
    base["status"] = job.get("status") or base.get("status")
    base["queue"] = _compact_object(
        {
            "job_id": job.get("job_id"),
            "state": job.get("queue_state") or job.get("status"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "result_status": job.get("result_status"),
            "current_task": job.get("current_task"),
            "task_events": list(job.get("task_events", []) or []),
            "agents": list(job.get("agents", []) or []),
            "routing": job.get("routing", {}),
            "error": job.get("error"),
        }
    )
    return _compact_object(base)


def _build_steps(state: ShipyardState) -> list[dict[str, Any]]:
    steps = list(state.get("action_steps", []) or [])
    if steps:
        return [
            _compact_object(
                {
                    "id": step.get("id"),
                    "instruction": step.get("instruction"),
                    "action_class": step.get("action_class"),
                    "edit_mode": step.get("edit_mode"),
                    "target_path": step.get("target_path"),
                    "anchor": step.get("anchor"),
                    "pattern": step.get("pattern"),
                    "command": step.get("command"),
                    "pointers": step.get("pointers"),
                    "replacement_preview": step.get("replacement_preview"),
                    "depends_on": step.get("depends_on", []),
                    "inputs_from": step.get("inputs_from", []),
                    "status": step.get("status"),
                    "changed_files": step.get("changed_files", []),
                    "no_op": step.get("no_op"),
                    "timeout_seconds": step.get("timeout_seconds"),
                    "max_retries": step.get("max_retries"),
                }
            )
            for step in steps
        ]

    action_plan = dict(state.get("action_plan", {}) or {})
    return [
        _compact_object(
            {
                "id": action.get("id"),
                "instruction": action.get("instruction"),
                "action_class": action.get("action_class"),
                "edit_mode": action.get("edit_mode"),
                "target_path": action.get("target_path"),
                "anchor": action.get("anchor"),
                "pointers": action.get("pointers"),
                "depends_on": action.get("depends_on", []),
                "inputs_from": action.get("inputs_from", []),
                "timeout_seconds": action.get("timeout_seconds"),
                "max_retries": action.get("max_retries"),
                "validation_errors": action.get("validation_errors", []),
            }
        )
        for action in (action_plan.get("actions", []) or [])
        if isinstance(action, dict)
    ]


def _build_tasks(state: ShipyardState) -> list[dict[str, Any]]:
    action_plan = dict(state.get("action_plan", {}) or {})
    action_steps = list(state.get("action_steps", []) or [])
    step_by_id = {
        str(step.get("id")): step
        for step in action_steps
        if step.get("id")
    }
    tasks: list[dict[str, Any]] = []
    for index, action in enumerate(action_plan.get("actions", []) or [], start=1):
        if not isinstance(action, dict):
            continue
        task_id = str(action.get("id") or f"step-{index}")
        step = step_by_id.get(task_id, {})
        tasks.append(
            _compact_object(
                {
                    "task_id": task_id,
                    "role": "lead-agent",
                    "goal": action.get("instruction"),
                    "allowed_actions": [action.get("edit_mode")] if action.get("edit_mode") else [],
                    "status": step.get("status") or ("planned" if action.get("valid") else "invalid"),
                    "result": {
                        "changed_files": step.get("changed_files", []),
                        "no_op": step.get("no_op"),
                    },
                    "artifacts": {},
                    "depends_on": action.get("depends_on", []),
                    "inputs_from": action.get("inputs_from", []),
                }
            )
        )
    return tasks


def _compact_object(value: object) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            compact_item = _compact_object(item)
            if compact_item in (None, "", [], {}):
                continue
            compacted[key] = compact_item
        return compacted
    if isinstance(value, list):
        compacted_list = [_compact_object(item) for item in value]
        return [item for item in compacted_list if item not in (None, "", [], {})]
    return value
