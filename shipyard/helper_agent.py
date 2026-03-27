from __future__ import annotations

from typing import Any

from .state import ShipyardState


def run_helper_agent(state: ShipyardState) -> dict[str, Any]:
    context = state.get("context", {})
    function_name = context.get("function_name")
    file_hint = context.get("file_hint") or state.get("target_path") or ""
    test_failure = context.get("test_failure")
    verification_commands = list(state.get("verification_commands", []) or [])
    edit_mode = str(state.get("edit_mode") or "").strip()

    if test_failure:
        task_id = "helper-retry-task"
        agent_name = "helper-verifier"
        agent_type = "verifier"
        task_type = "retry_summary"
        recommendation = "Review the latest verification failure and revise the proposed edit before retrying."
        notes = f"Verification failed previously. Focus the next attempt on: {test_failure.splitlines()[0]}"
        allowed_actions = ["read_file", "search_files", "verify_command", "run_tests"]
    elif function_name:
        task_id = "helper-function-task"
        agent_name = "helper-function-planner"
        agent_type = "function-editor"
        task_type = "function_edit_planning"
        recommendation = (
            "Prepare a candidate replacement body for the named function and keep file writes with the lead agent."
        )
        notes = f"Target named function `{function_name}` in {file_hint}."
        allowed_actions = ["read_file", "inspect_imports", "named_function"]
    elif verification_commands and edit_mode in {"run_command", "verify_command", "run_tests"}:
        task_id = "helper-verify-task"
        agent_name = "helper-verifier"
        agent_type = "verifier"
        task_type = "verification_planning"
        recommendation = "Prepare verification-oriented follow-up work and keep command execution bounded."
        notes = f"Verification is expected after editing {file_hint or 'the requested files'}."
        allowed_actions = ["read_file", "search_files", "verify_command", "run_tests"]
    else:
        task_id = "helper-anchor-task"
        agent_name = "helper-anchor-planner"
        agent_type = "editor"
        task_type = "anchor_edit_planning"
        recommendation = "Confirm the target file and isolate a unique anchor before editing."
        notes = f"Use anchor-based editing for {file_hint or 'the requested file'}."
        allowed_actions = ["read_file", "search_files", "anchor", "search_and_replace"]

    return {
        "task_id": task_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "delegation_mode": "sequential",
        "task_type": task_type,
        "allowed_actions": allowed_actions,
        "recommendation": recommendation,
        "notes": notes,
    }
