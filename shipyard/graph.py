from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pathlib import Path
import re
import shutil

from .human_gate import make_human_gate
from .helper_agent import run_helper_agent
from .intent_parser import parse_occurrence_selector
from .prompts import build_runtime_prompt
from .proposal import propose_edit
from .state import ShipyardState
from .tools.code_graph import inspect_code_graph_status
from .tools.edit_file import AnchorEditError, apply_anchor_edit, apply_symbol_rename, validate_anchor_edit
from .tools.function_edit import FunctionEditError, apply_function_edit, get_function_source
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


def consult_helper_agent(state: ShipyardState) -> dict:
    helper_result = run_helper_agent(state)
    context = dict(state.get("context", {}))
    context["helper_notes"] = helper_result["notes"]
    return {
        "context": context,
        "helper_output": {
            "helper_agent": helper_result,
        },
        "status": "helper_consulted",
    }


def plan_edit(state: ShipyardState) -> dict:
    planned = propose_edit(state)
    helper_output = dict(state.get("helper_output", {}))
    context = dict(state.get("context", {}))
    planned_target = planned.get("target_path")
    planned_mode = planned.get("edit_mode") or state.get("edit_mode") or "anchor"
    function_name = context.get("function_name")

    helper_agent = dict(helper_output.get("helper_agent", {}))
    if context.get("test_failure"):
        refreshed_notes = helper_agent.get("notes") or context.get("helper_notes", "")
    elif planned_mode == "named_function" and function_name and planned_target:
        refreshed_notes = f"Target named function `{function_name}` in {planned_target}."
    elif planned_target:
        refreshed_notes = f"Use {planned_mode}-based editing for {planned_target}."
    else:
        refreshed_notes = helper_agent.get("notes") or context.get("helper_notes", "")

    if refreshed_notes:
        helper_agent["notes"] = refreshed_notes
        helper_output["helper_agent"] = helper_agent
        context["helper_notes"] = refreshed_notes

    helper_output["proposal"] = {
        "provider": planned.get("provider"),
        "provider_reason": planned.get("provider_reason"),
        "is_valid": planned.get("is_valid"),
        "validation_errors": planned.get("validation_errors", []),
    }
    proposal_summary = {
        "provider": planned.get("provider"),
        "provider_reason": planned.get("provider_reason"),
        "edit_mode": planned.get("edit_mode") or state.get("edit_mode") or "anchor",
        "target_path": planned.get("target_path"),
        "target_path_source": planned.get("target_path_source"),
        "occurrence_selector": planned.get("occurrence_selector"),
        "quantity": planned.get("quantity"),
        "copy_count": planned.get("copy_count"),
        "is_valid": planned.get("is_valid"),
        "validation_errors": planned.get("validation_errors", []),
        "has_anchor": bool(planned.get("anchor")),
        "has_replacement": planned.get("replacement") is not None,
    }
    return {
        "target_path": planned.get("target_path"),
        "anchor": planned.get("anchor"),
        "replacement": planned.get("replacement"),
        "occurrence_selector": planned.get("occurrence_selector"),
        "quantity": planned.get("quantity"),
        "copy_count": planned.get("copy_count"),
        "edit_mode": planned.get("edit_mode") or state.get("edit_mode") or "anchor",
        "prompt": build_runtime_prompt(
            {
                **state,
                "target_path": planned_target,
                "anchor": planned.get("anchor"),
                "replacement": planned.get("replacement"),
                "edit_mode": planned_mode,
                "context": context,
            }
        ),
        "context": context,
        "helper_output": helper_output,
        "proposal_summary": proposal_summary,
        "status": "planned",
    }


def validate_proposal(state: ShipyardState) -> dict:
    proposal_summary = state.get("proposal_summary", {})
    if proposal_summary.get("is_valid"):
        return {"status": "proposal_valid"}

    errors = proposal_summary.get("validation_errors", [])
    message = "Invalid edit proposal."
    if errors:
        message = f"Invalid edit proposal. {' '.join(errors)}"
    return {
        "status": "invalid_proposal",
        "error": message,
        "human_gate": make_human_gate(
            message,
            action="clarify_request",
            prompt="Add a target path or missing edit details, then run again.",
            details={"validation_errors": errors},
        ),
    }


def check_edit_readiness(state: ShipyardState) -> dict:
    if state.get("edit_mode") != "named_function":
        return {
            "code_graph_status": {
                "ready": False,
                "available": False,
                "source": "skipped",
                "reason": "Code graph readiness is only required for named-function edits.",
            },
            "status": "ready_for_file_read",
        }

    status = inspect_code_graph_status(state.get("target_path"))
    if not status.get("ready"):
        return {
            "code_graph_status": status,
            "status": "graph_unavailable",
            "error": (
                "Named-function edits require a ready Code-Graph-RAG runtime. "
                f"{status.get('reason', 'Code graph is unavailable.')}"
            ),
            "human_gate": make_human_gate(
                status.get("reason", "Code graph is unavailable."),
                action="sync_graph",
                prompt="Sync the live graph, then retry the named-function edit.",
                details={"code_graph_status": status},
            ),
        }

    return {
        "code_graph_status": status,
        "status": "ready_for_file_read",
    }


def collect_edit_context(state: ShipyardState) -> dict:
    if state.get("edit_mode") != "named_function":
        helper_output = dict(state.get("helper_output", {}))
        helper_output["edit_context"] = {
            "mode": state.get("edit_mode") or "anchor",
            "status": "skipped",
            "reason": "No named-function context collection required.",
        }
        return {
            "helper_output": helper_output,
            "status": "ready_for_file_read",
        }

    target_path = state.get("target_path")
    function_name = state.get("context", {}).get("function_name")
    if not target_path or not function_name:
        return {
            "status": "awaiting_edit_spec",
            "error": "Missing target path or function name for named-function context collection.",
            "human_gate": make_human_gate(
                "Missing target path or function name for named-function context collection.",
                action="provide_function_target",
                prompt="Set the target path and function name, then retry.",
            ),
        }

    try:
        current_source = get_function_source(target_path, function_name)
    except FunctionEditError as exc:
        return {
            "status": "edit_blocked",
            "error": str(exc),
            "human_gate": make_human_gate(
                str(exc),
                action="inspect_function_target",
                prompt="Confirm the function name exists in the selected file, then retry.",
            ),
        }

    helper_output = dict(state.get("helper_output", {}))
    helper_output["edit_context"] = {
        "mode": "named_function",
        "function_name": function_name,
        "current_source": current_source,
        "line_count": len(current_source.splitlines()),
        "query_mode": "function_source_only",
    }
    code_graph_status = dict(state.get("code_graph_status", {}))
    code_graph_status["context_collected"] = True
    code_graph_status["query_mode"] = "function_source_only"

    return {
        "current_function_source": current_source,
        "helper_output": helper_output,
        "code_graph_status": code_graph_status,
        "status": "ready_for_file_read",
    }


def read_target_file(state: ShipyardState) -> dict:
    target_path = state.get("target_path")
    if not target_path:
        return {"status": "no_target"}

    file_path = Path(target_path)
    if not file_path.exists() and state.get("edit_mode") in {"write_file", "append", "prepend", "create_files"}:
        return {
            "file_before": "",
            "target_existed_before_edit": False,
            "status": "file_read",
        }

    return {
        "file_before": read_file(target_path),
        "target_existed_before_edit": file_path.exists(),
        "status": "file_read",
    }


def apply_edit(state: ShipyardState) -> dict:
    if state.get("edit_mode") == "named_function":
        target_path = state.get("target_path")
        function_name = state.get("context", {}).get("function_name")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1
        code_graph_status = dict(state.get("code_graph_status", {}))

        if not target_path or not function_name or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, function name, or replacement.",
                "human_gate": make_human_gate(
                    "Missing target path, function name, or replacement.",
                    action="provide_function_replacement",
                    prompt="Provide the full replacement function body, then retry.",
                ),
            }

        snapshot_path = snapshot_file(target_path)
        try:
            apply_function_edit(target_path, function_name, replacement)
        except FunctionEditError as exc:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "snapshot_path": snapshot_path,
                "status": "edit_blocked",
                "error": str(exc),
                "human_gate": make_human_gate(
                    str(exc),
                    action="inspect_function_target",
                    prompt="Review the named function target and retry after correcting the request.",
                ),
            }

        index_state = dict(code_graph_status.get("index_state", {}))
        index_state["stale"] = True
        code_graph_status["index_state"] = index_state
        code_graph_status["refresh_required"] = True
        code_graph_status["reason"] = "Graph-backed function edit succeeded; re-index to refresh graph state."

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "code_graph_status": code_graph_status,
            "status": "edited",
        }

    if state.get("edit_mode") == "delete_file":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1

        if not target_path:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path for delete mode.",
                "human_gate": make_human_gate(
                    "Missing target path for delete mode.",
                    action="provide_target_path",
                    prompt="Provide the file you want to delete, then retry.",
                ),
            }

        file_path = Path(target_path)
        if not file_path.exists():
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "edit_blocked",
                "error": "Target file was not found for deletion.",
                "human_gate": make_human_gate(
                    "Target file was not found for deletion.",
                    action="inspect_target_path",
                    prompt="Confirm the file path exists, then retry.",
                ),
            }

        snapshot_path = snapshot_file(target_path)
        file_path.unlink()
        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "status": "edited",
        }

    if state.get("edit_mode") == "copy_file":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1
        copy_count = state.get("copy_count")

        if not target_path or copy_count is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path or copy count for copy mode.",
                "human_gate": make_human_gate(
                    "Missing target path or copy count for copy mode.",
                    action="clarify_request",
                    prompt="Specify which file to copy and how many copies you want, then retry.",
                ),
            }

        source_path = Path(target_path)
        if not source_path.exists():
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "edit_blocked",
                "error": "Target file was not found for copy mode.",
                "human_gate": make_human_gate(
                    "Target file was not found for copy mode.",
                    action="inspect_target_path",
                    prompt="Confirm the file path exists, then retry.",
                ),
            }

        count = int(copy_count)
        created_files = []
        snapshot_path = snapshot_file(target_path)
        for index in range(1, count + 1):
            copy_path = _build_copy_path(source_path, index)
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copy_path)
            created_files.append(str(copy_path.resolve()))

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "changed_files": created_files,
            "status": "edited",
        }

    if state.get("edit_mode") == "create_files":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1
        quantity = state.get("quantity")
        replacement = state.get("replacement")

        if not target_path or quantity is None or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, quantity, or content for create_files mode.",
                "human_gate": make_human_gate(
                    "Missing target path, quantity, or content for create_files mode.",
                    action="clarify_request",
                    prompt="Specify how many files to create and any file contents, then retry.",
                ),
            }

        count = int(quantity)
        first_path = Path(target_path)
        first_path.parent.mkdir(parents=True, exist_ok=True)
        created_files = []
        focused_index = _extract_numbered_file_index(first_path)

        for index in range(1, count + 1):
            file_path = _build_numbered_batch_path(first_path, index) if focused_index is not None else (
                first_path if index == 1 else _build_copy_path(first_path, index)
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = replacement if focused_index is None or index == focused_index else ""
            file_path.write_text(content, encoding="utf-8")
            created_files.append(str(file_path.resolve()))

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "changed_files": created_files,
            "target_existed_before_edit": False,
            "status": "edited",
        }

    if state.get("edit_mode") == "rename_symbol":
        target_path = state.get("target_path")
        anchor = state.get("anchor")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1

        if not target_path or not anchor or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, symbol, or replacement for rename mode.",
                "human_gate": make_human_gate(
                    "Missing target path, symbol, or replacement for rename mode.",
                    action="clarify_request",
                    prompt="Specify the file and symbol rename, then retry.",
                ),
            }

        snapshot_path = snapshot_file(target_path)
        try:
            apply_symbol_rename(target_path, anchor, replacement)
        except AnchorEditError as exc:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "snapshot_path": snapshot_path,
                "status": "edit_blocked",
                "error": str(exc),
                "human_gate": make_human_gate(
                    str(exc),
                    action="inspect_target_path",
                    prompt="Confirm the symbol exists in the target file, then retry.",
                ),
            }

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "status": "edited",
        }

    if state.get("edit_mode") in {"write_file", "append", "prepend"}:
        target_path = state.get("target_path")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1
        target_existed = state.get("target_existed_before_edit", True)

        if not target_path or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path or content for the requested edit mode.",
                "human_gate": make_human_gate(
                    "Missing target path or content for the requested edit mode.",
                    action="provide_target_or_content",
                    prompt="Add a target path or explicit content, then retry.",
                ),
            }

        snapshot_path = snapshot_file(target_path)
        file_path = Path(target_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        current_content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        updated_content = current_content

        if state.get("edit_mode") == "write_file":
            updated_content = replacement
        elif state.get("edit_mode") == "append":
            updated_content = current_content + replacement
        elif state.get("edit_mode") == "prepend":
            updated_content = replacement + current_content

        file_path.write_text(updated_content, encoding="utf-8")
        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "target_existed_before_edit": target_existed,
            "status": "edited",
        }

    target_path = state.get("target_path")
    anchor = state.get("anchor")
    replacement = state.get("replacement")
    occurrence_selector = state.get("occurrence_selector") or parse_occurrence_selector(
        state.get("instruction", "")
    )
    file_before = state.get("file_before", "")
    edit_attempts = state.get("edit_attempts", 0) + 1

    if not target_path or not anchor or replacement is None:
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "awaiting_edit_spec",
            "error": "Missing target path, anchor, or replacement.",
            "human_gate": make_human_gate(
                "Missing target path, anchor, or replacement.",
                action="provide_anchor_edit",
                prompt="Provide a unique anchor and replacement, then retry.",
            ),
        }

    try:
        validate_anchor_edit(file_before, anchor, occurrence_selector)
    except AnchorEditError as exc:
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "edit_blocked",
            "error": str(exc),
            "human_gate": make_human_gate(
                str(exc),
                action="inspect_anchor",
                prompt="Choose a unique anchor in the file, or specify first, middle, or last, then retry.",
            ),
        }

    snapshot_path = snapshot_file(target_path)
    apply_anchor_edit(target_path, anchor, replacement, occurrence_selector)

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
    target_existed = state.get("target_existed_before_edit", True)
    if target_path and snapshot_path:
        if not target_existed and Path(target_path).exists():
            Path(target_path).unlink()
        else:
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
        "human_gate": make_human_gate(
            "Verification failed after maximum retry attempts. File reverted to latest snapshot.",
            action="inspect_failure",
            prompt="Review the verification failure output and adjust the request before retrying.",
            status="blocked",
            details={"verification_results": verification_results},
        ),
    }


def should_retry(state: ShipyardState) -> str:
    if state.get("status") == "retry_ready":
        return "retry"
    return "done"


def should_continue_after_readiness(state: ShipyardState) -> str:
    if state.get("status") == "ready_for_file_read":
        return "continue"
    return "done"


def should_continue_after_proposal_validation(state: ShipyardState) -> str:
    if state.get("status") == "proposal_valid":
        return "continue"
    return "done"


def _build_copy_path(source_path: Path, index: int) -> Path:
    suffix = source_path.suffix
    stem = source_path.stem
    return source_path.with_name(f"{stem}_copy_{index}{suffix}")


def _extract_numbered_file_index(path: Path) -> int | None:
    match = re.search(r"^file(?P<index>\d+)$", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group("index"))


def _build_numbered_batch_path(source_path: Path, index: int) -> Path:
    return source_path.with_name(f"file{index}{source_path.suffix}")


def build_graph():
    graph = StateGraph(ShipyardState)
    graph.add_node("seed_defaults", seed_defaults)
    graph.add_node("prepare_prompt", prepare_prompt)
    graph.add_node("consult_helper_agent", consult_helper_agent)
    graph.add_node("plan_edit", plan_edit)
    graph.add_node("validate_proposal", validate_proposal)
    graph.add_node("check_edit_readiness", check_edit_readiness)
    graph.add_node("collect_edit_context", collect_edit_context)
    graph.add_node("read_target_file", read_target_file)
    graph.add_node("apply_edit", apply_edit)
    graph.add_node("verify_edit", verify_edit)
    graph.add_node("recover_or_finish", recover_or_finish)
    graph.add_edge(START, "seed_defaults")
    graph.add_edge("seed_defaults", "prepare_prompt")
    graph.add_edge("prepare_prompt", "consult_helper_agent")
    graph.add_edge("consult_helper_agent", "plan_edit")
    graph.add_edge("plan_edit", "validate_proposal")
    graph.add_conditional_edges(
        "validate_proposal",
        should_continue_after_proposal_validation,
        {
            "continue": "check_edit_readiness",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "check_edit_readiness",
        should_continue_after_readiness,
        {
            "continue": "collect_edit_context",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "collect_edit_context",
        should_continue_after_readiness,
        {
            "continue": "read_target_file",
            "done": END,
        },
    )
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
