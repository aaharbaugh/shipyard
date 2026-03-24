from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .prompts import build_runtime_prompt
from .proposal import propose_edit
from .state import ShipyardState
from .tools.edit_file import AnchorEditError, apply_anchor_edit, validate_anchor_edit
from .tools.read_file import read_file
from .tools.revert import revert_file
from .tools.snapshot import snapshot_file
from .tools.verify import run_verification


def seed_defaults(state: ShipyardState) -> dict:
    return {
        "edit_attempts": state.get("edit_attempts", 0),
        "max_edit_attempts": state.get("max_edit_attempts", 2),
        "reverted_to_snapshot": False,
    }


def prepare_prompt(state: ShipyardState) -> dict:
    return {
        "prompt": build_runtime_prompt(state),
        "status": "prepared",
    }


def plan_edit(state: ShipyardState) -> dict:
    planned = propose_edit(state)
    return {
        "target_path": planned.get("target_path"),
        "anchor": planned.get("anchor"),
        "replacement": planned.get("replacement"),
        "helper_output": {
            "provider": planned.get("provider"),
            "provider_reason": planned.get("provider_reason"),
        },
        "status": "planned",
    }


def read_target_file(state: ShipyardState) -> dict:
    target_path = state.get("target_path")
    if not target_path:
        return {"status": "no_target"}

    return {
        "file_before": read_file(target_path),
        "status": "file_read",
    }


def apply_edit(state: ShipyardState) -> dict:
    target_path = state.get("target_path")
    anchor = state.get("anchor")
    replacement = state.get("replacement")
    file_before = state.get("file_before", "")
    edit_attempts = state.get("edit_attempts", 0) + 1

    if not target_path or not anchor or replacement is None:
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "awaiting_edit_spec",
            "error": "Missing target path, anchor, or replacement.",
        }

    try:
        validate_anchor_edit(file_before, anchor)
    except AnchorEditError as exc:
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "edit_blocked",
            "error": str(exc),
        }

    snapshot_path = snapshot_file(target_path)
    apply_anchor_edit(target_path, anchor, replacement)

    return {
        "edit_applied": True,
        "edit_attempts": edit_attempts,
        "snapshot_path": snapshot_path,
        "status": "edited",
    }


def verify_edit(state: ShipyardState) -> dict:
    commands = state.get("verification_commands", [])
    if not commands:
        return {"verification_results": [], "status": state.get("status", "edited")}

    results = run_verification(commands)
    has_failure = any(result["returncode"] != 0 for result in results)

    return {
        "verification_results": results,
        "status": "verification_failed" if has_failure else "verified",
    }


def recover_or_finish(state: ShipyardState) -> dict:
    if state.get("status") != "verification_failed":
        return {}

    target_path = state.get("target_path")
    snapshot_path = state.get("snapshot_path")
    if target_path and snapshot_path:
        revert_file(target_path, snapshot_path)

    attempts = state.get("edit_attempts", 0)
    max_attempts = state.get("max_edit_attempts", 2)
    verification_results = state.get("verification_results", [])
    failure_messages = [
        result.get("stderr") or result.get("stdout") or ""
        for result in verification_results
        if result.get("returncode") != 0
    ]
    failure_text = "\n".join(message for message in failure_messages if message).strip()

    if attempts < max_attempts:
        context = dict(state.get("context", {}))
        if failure_text:
            context["test_failure"] = failure_text
        return {
            "context": context,
            "reverted_to_snapshot": True,
            "status": "retry_ready",
            "error": "Verification failed. File reverted to latest snapshot.",
        }

    return {
        "reverted_to_snapshot": True,
        "status": "failed_after_retries",
        "error": "Verification failed after maximum retry attempts. File reverted to latest snapshot.",
    }


def should_retry(state: ShipyardState) -> str:
    if state.get("status") == "retry_ready":
        return "retry"
    return "done"


def build_graph():
    graph = StateGraph(ShipyardState)
    graph.add_node("seed_defaults", seed_defaults)
    graph.add_node("prepare_prompt", prepare_prompt)
    graph.add_node("plan_edit", plan_edit)
    graph.add_node("read_target_file", read_target_file)
    graph.add_node("apply_edit", apply_edit)
    graph.add_node("verify_edit", verify_edit)
    graph.add_node("recover_or_finish", recover_or_finish)
    graph.add_edge(START, "seed_defaults")
    graph.add_edge("seed_defaults", "prepare_prompt")
    graph.add_edge("prepare_prompt", "plan_edit")
    graph.add_edge("plan_edit", "read_target_file")
    graph.add_edge("read_target_file", "apply_edit")
    graph.add_edge("apply_edit", "verify_edit")
    graph.add_edge("verify_edit", "recover_or_finish")
    graph.add_conditional_edges(
        "recover_or_finish",
        should_retry,
        {
            "retry": "prepare_prompt",
            "done": END,
        },
    )
    return graph.compile()
