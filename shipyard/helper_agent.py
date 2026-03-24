from __future__ import annotations

from .state import ShipyardState


def run_helper_agent(state: ShipyardState) -> dict[str, str]:
    context = state.get("context", {})
    function_name = context.get("function_name")
    file_hint = context.get("file_hint") or state.get("target_path") or ""
    test_failure = context.get("test_failure")

    if test_failure:
        task_type = "retry_summary"
        recommendation = "Review the latest verification failure and revise the proposed edit before retrying."
        notes = f"Verification failed previously. Focus the next attempt on: {test_failure.splitlines()[0]}"
    elif function_name:
        task_type = "function_edit_planning"
        recommendation = (
            "Prepare a candidate replacement body for the named function and keep file writes with the lead agent."
        )
        notes = f"Target named function `{function_name}` in {file_hint}."
    else:
        task_type = "anchor_edit_planning"
        recommendation = "Confirm the target file and isolate a unique anchor before editing."
        notes = f"Use anchor-based editing for {file_hint or 'the requested file'}."

    return {
        "agent_name": "helper-planner",
        "delegation_mode": "sequential",
        "task_type": task_type,
        "recommendation": recommendation,
        "notes": notes,
    }
