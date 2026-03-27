from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .storage_paths import LOGS_ROOT, TRACES_ROOT, ensure_dir
from .state import ShipyardState


def write_trace(state: ShipyardState) -> str:
    trace_dir = ensure_dir(TRACES_ROOT)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = state.get("session_id", "unknown")
    trace_path = trace_dir / f"{stamp}-{session_id}.json"

    trace_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(trace_path)


def write_troubleshooting_log(state: ShipyardState) -> str:
    logs_dir = ensure_dir(LOGS_ROOT)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = str(state.get("session_id", "unknown"))
    log_path = logs_dir / f"{stamp}-{session_id}-troubleshooting.json"
    latest_path = logs_dir / "latest-troubleshooting.json"
    session_latest_path = logs_dir / f"latest-{session_id}-troubleshooting.json"

    payload = _build_troubleshooting_payload(state)
    text = json.dumps(payload, indent=2, sort_keys=True)
    log_path.write_text(text, encoding="utf-8")
    session_latest_path.write_text(text, encoding="utf-8")
    if _should_update_global_latest(session_id):
        latest_path.write_text(text, encoding="utf-8")
    return str(log_path)


def _should_update_global_latest(session_id: str) -> bool:
    return session_id.startswith("web-") or session_id in {"default", "unknown"}


def _build_troubleshooting_payload(state: ShipyardState) -> dict[str, object]:
    context = state.get("context", {}) or {}
    action_plan = state.get("action_plan", {}) or {}
    graph_sync = state.get("graph_sync", {}) or {}
    code_graph_status = state.get("code_graph_status", {}) or {}
    human_gate = state.get("human_gate", {}) or {}
    proposal_summary = state.get("proposal_summary", {}) or {}
    changed_files = list(state.get("changed_files", []) or [])
    action_steps = list(state.get("action_steps", []) or [])

    payload = {
        "session_id": state.get("session_id"),
        "instruction": state.get("request_instruction") or state.get("instruction"),
        "status": state.get("status"),
        "error": state.get("error"),
        "final_step_target_path": state.get("target_path"),
        "primary_changed_file": changed_files[-1] if changed_files else None,
        "edit_mode": state.get("edit_mode"),
        "changed_files": changed_files,
        "file_preview": state.get("file_preview"),
        "content_hash": state.get("content_hash"),
        "no_op": state.get("no_op"),
        "action_plan_summary": {
            "provider": action_plan.get("provider"),
            "provider_reason": action_plan.get("provider_reason"),
            "is_valid": action_plan.get("is_valid"),
            "validation_errors": action_plan.get("validation_errors", []),
            "action_count": len(action_plan.get("actions", []) or []),
            "actions": [
                {
                    "instruction": action.get("instruction"),
                    "edit_mode": action.get("edit_mode"),
                    "target_path": action.get("target_path"),
                    "anchor": action.get("anchor"),
                    "replacement_preview": str(action.get("replacement") or "")[:120],
                    "file_count": len(action.get("files", []) or []),
                    "valid": action.get("valid"),
                    "validation_errors": action.get("validation_errors", []),
                }
                for action in (action_plan.get("actions", []) or [])
            ],
        },
        "execution_steps": [
            {
                "instruction": step.get("instruction"),
                "edit_mode": step.get("edit_mode"),
                "target_path": step.get("target_path"),
                "anchor": step.get("anchor"),
                "replacement_preview": step.get("replacement_preview"),
                "status": step.get("status"),
                "changed_files": step.get("changed_files", []),
                "no_op": step.get("no_op"),
            }
            for step in action_steps
        ],
        "proposal_summary": {
            "provider": proposal_summary.get("provider"),
            "edit_mode": proposal_summary.get("edit_mode"),
            "target_path_source": proposal_summary.get("target_path_source"),
            "is_valid": proposal_summary.get("is_valid"),
            "validation_errors": proposal_summary.get("validation_errors", []),
        },
        "human_gate": {
            "action": human_gate.get("action"),
            "prompt": human_gate.get("prompt"),
            "reason": human_gate.get("reason"),
            "details": human_gate.get("details", {}),
        },
        "graph": {
            "ready": code_graph_status.get("ready"),
            "available": code_graph_status.get("available"),
            "reason": code_graph_status.get("reason"),
            "sync_attempted": graph_sync.get("attempted"),
            "sync_ok": graph_sync.get("ok"),
        },
        "context": {
            "file_hint": context.get("file_hint"),
            "function_name": context.get("function_name"),
            "testing_mode": context.get("testing_mode"),
        },
        "artifacts": {
            "trace_path": state.get("trace_path"),
            "snapshot_path": state.get("snapshot_path"),
        },
    }
    return _compact_object(payload)


def _compact_object(value: object) -> object:
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
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
