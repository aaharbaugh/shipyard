from __future__ import annotations

from typing import Any, TypedDict

from .intent_parser import (
    infer_edit_mode,
    parse_instruction,
    parse_occurrence_selector,
)
from .pathing import resolve_target_path
from .planning_hints import (
    infer_copy_count,
    infer_create_quantity,
    infer_target_path_from_instruction,
    resolve_requested_target_hint,
)
from .proposal_validation import attach_validation
from .state import ShipyardState


SUPPORTED_ACTIONS = {
    "anchor",
    "named_function",
    "write_file",
    "append",
    "prepend",
    "delete_file",
    "copy_file",
    "create_files",
    "scaffold_files",
    "rename_symbol",
    "list_files",
    "read_file",
    "search_files",
    "run_command",
    "verify_command",
    "create_directory",
    "move_file",
    "rename_file",
    "read_many_files",
    "search_and_replace",
    "run_tests",
    "inspect_imports",
    "synthesize_tool",
    "invoke_tool",
}

ACTION_CLASSES = {
    "list_files": "inspect",
    "read_file": "inspect",
    "read_many_files": "inspect",
    "search_files": "inspect",
    "inspect_imports": "inspect",
    "anchor": "mutate",
    "named_function": "mutate",
    "write_file": "mutate",
    "append": "mutate",
    "prepend": "mutate",
    "delete_file": "mutate",
    "copy_file": "mutate",
    "create_files": "mutate",
    "scaffold_files": "mutate",
    "rename_symbol": "mutate",
    "create_directory": "mutate",
    "move_file": "mutate",
    "rename_file": "mutate",
    "search_and_replace": "mutate",
    "run_command": "verify",
    "verify_command": "verify",
    "run_tests": "verify",
    "synthesize_tool": "synthesize",
    "invoke_tool": "inspect",
}


class Action(TypedDict, total=False):
    id: str
    instruction: str
    role: str
    agent_type: str
    parent_task_id: str | None
    child_task_ids: list[str]
    allowed_actions: list[str]
    target_path: str | None
    target_path_source: str | None
    edit_mode: str
    action_class: str
    intent: str | None
    edit_scope: str | None
    expected_existing_state: str | None
    recovery_strategy: str | None
    anchor: str | None
    replacement: str | None
    quantity: int | None
    copy_count: int | None
    files: list[dict[str, Any]] | None
    pattern: str | None
    command: str | None
    pointers: list[dict[str, int]] | None
    occurrence_selector: str | None
    source_path: str | None
    destination_path: str | None
    paths: list[str] | None
    depends_on: list[str]
    inputs_from: list[str]
    timeout_seconds: int | None
    max_retries: int | None
    full_file_rewrite: bool
    tool_name: str | None
    tool_source: str | None
    tool_args: dict[str, Any] | None
    valid: bool
    validation_errors: list[str]
    provider: str
    provider_reason: str


def _default_mutate_contract(edit_mode: str) -> dict[str, str | None]:
    if edit_mode in {"anchor", "search_and_replace", "rename_symbol"}:
        return {
            "intent": "localized_edit",
            "edit_scope": "single_span",
            "expected_existing_state": "existing_file",
            "recovery_strategy": "replan_step",
        }
    if edit_mode == "named_function":
        return {
            "intent": "function_rewrite",
            "edit_scope": "function_block",
            "expected_existing_state": "existing_file",
            "recovery_strategy": "replan_step",
        }
    if edit_mode == "write_file":
        return {
            "intent": "full_rewrite",
            "edit_scope": "full_file",
            "expected_existing_state": "either",
            "recovery_strategy": "replan_step",
        }
    if edit_mode == "append":
        return {
            "intent": "append_content",
            "edit_scope": "file_tail",
            "expected_existing_state": "existing_file",
            "recovery_strategy": "replan_step",
        }
    if edit_mode == "prepend":
        return {
            "intent": "prepend_content",
            "edit_scope": "file_head",
            "expected_existing_state": "existing_file",
            "recovery_strategy": "replan_step",
        }
    if edit_mode == "scaffold_files":
        return {
            "intent": "scaffold",
            "edit_scope": "new_files",
            "expected_existing_state": "new_file",
            "recovery_strategy": "replan_slice",
        }
    if edit_mode in {"create_files", "create_directory", "delete_file", "copy_file", "move_file", "rename_file"}:
        return {
            "intent": "filesystem_mutation",
            "edit_scope": "filesystem",
            "expected_existing_state": "either",
            "recovery_strategy": "block",
        }
    return {
        "intent": None,
        "edit_scope": None,
        "expected_existing_state": None,
        "recovery_strategy": None,
    }


def _normalize_pointer_payload(value: Any) -> list[dict[str, int]] | None:
    if value in (None, "", []):
        return None
    return value


def _coerce_enum(value: Any, valid: set[str], default: str | None) -> str | None:
    """Return value if it's in the valid set, otherwise return default.

    Handles the LLM returning freeform descriptions instead of enum values.
    """
    if value is not None and str(value).strip() in valid:
        return str(value).strip()
    return default


def _normalize_list_field(value: Any) -> list[str]:
    """Normalize a field that should be a list of strings.

    Handles the case where the LLM returns a bare string like "s1"
    instead of ["s1"] — list("s1") would split into ["s", "1"].
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalize_paths_payload(raw_action: dict[str, Any], fallback: dict[str, Any]) -> list[str] | None:
    direct_paths = raw_action.get("paths", fallback.get("paths"))
    if isinstance(direct_paths, list) and direct_paths:
        return [str(item) for item in direct_paths if item not in (None, "")]

    if str(raw_action.get("edit_mode") or fallback.get("edit_mode") or "").strip() != "read_many_files":
        return None

    files_value = raw_action.get("files", fallback.get("files"))
    if not isinstance(files_value, list) or not files_value:
        return None

    normalized: list[str] = []
    for item in files_value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
        elif isinstance(item, dict):
            path = item.get("path") or item.get("target_path")
            if path:
                normalized.append(str(path).strip())
    return normalized or None


def normalize_action(
    raw_action: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    provider: str,
    provider_reason: str,
) -> Action:
    fallback = fallback or {}
    action_id = str(raw_action.get("id") or fallback.get("id") or "").strip() or None
    instruction = str(raw_action.get("instruction") or fallback.get("instruction") or "").strip()
    edit_mode = str(raw_action.get("edit_mode") or fallback.get("edit_mode") or "anchor").strip()
    if edit_mode not in SUPPORTED_ACTIONS:
        edit_mode = str(fallback.get("edit_mode") or "anchor")
    # Auto-convert create_files with a single target to write_file
    if edit_mode == "create_files" and raw_action.get("target_path") and not raw_action.get("quantity"):
        edit_mode = "write_file"
    action_class = str(raw_action.get("action_class") or fallback.get("action_class") or ACTION_CLASSES.get(edit_mode, "mutate")).strip()

    replacement = raw_action.get("replacement")
    if replacement is None and raw_action.get("replace_text") is not None:
        replacement = raw_action.get("replace_text")
    if replacement is None and raw_action.get("content") is not None:
        replacement = raw_action.get("content")
    if replacement is None:
        replacement = fallback.get("replacement")

    anchor = raw_action.get("anchor")
    if anchor is None and raw_action.get("search_text") is not None:
        anchor = raw_action.get("search_text")
    if anchor is None:
        anchor = fallback.get("anchor")

    mutate_contract = _default_mutate_contract(edit_mode) if action_class == "mutate" else {
        "intent": None,
        "edit_scope": None,
        "expected_existing_state": None,
        "recovery_strategy": None,
    }

    normalized: Action = {
        "id": action_id,
        "instruction": instruction,
        "role": str(raw_action.get("role") or fallback.get("role") or "lead-agent"),
        "agent_type": str(raw_action.get("agent_type") or fallback.get("agent_type") or "primary"),
        "parent_task_id": raw_action.get("parent_task_id", fallback.get("parent_task_id")),
        "child_task_ids": _normalize_list_field(raw_action.get("child_task_ids", fallback.get("child_task_ids", []))),
        "allowed_actions": _normalize_list_field(
            raw_action.get("allowed_actions", fallback.get("allowed_actions", [edit_mode])) or [edit_mode]
        ),
        "target_path": raw_action.get("target_path") or fallback.get("target_path"),
        "target_path_source": raw_action.get("target_path_source") or fallback.get("target_path_source"),
        "edit_mode": edit_mode,
        "action_class": action_class,
        "intent": _coerce_enum(
            raw_action.get("intent", fallback.get("intent")),
            {"localized_edit", "function_rewrite", "full_rewrite", "append_content", "prepend_content", "scaffold", "filesystem_mutation"},
            mutate_contract.get("intent"),
        ),
        "edit_scope": _coerce_enum(
            raw_action.get("edit_scope", fallback.get("edit_scope")),
            {"single_span", "multi_span", "function_block", "full_file", "file_tail", "file_head", "new_files", "filesystem"},
            mutate_contract.get("edit_scope"),
        ),
        "expected_existing_state": _coerce_enum(
            raw_action.get("expected_existing_state", fallback.get("expected_existing_state")),
            {"existing_file", "new_file", "either"},
            mutate_contract.get("expected_existing_state"),
        ),
        "recovery_strategy": _coerce_enum(
            raw_action.get("recovery_strategy", fallback.get("recovery_strategy")),
            {"replan_step", "replan_slice", "block"},
            mutate_contract.get("recovery_strategy"),
        ),
        "anchor": anchor,
        "replacement": replacement,
        "quantity": raw_action.get("quantity", fallback.get("quantity")),
        "copy_count": raw_action.get("copy_count", fallback.get("copy_count")),
        "files": raw_action.get("files", fallback.get("files")),
        "pattern": raw_action.get("pattern", fallback.get("pattern")),
        "command": raw_action.get("command", fallback.get("command")),
        "pointers": _normalize_pointer_payload(raw_action.get("pointers", fallback.get("pointers"))),
        "occurrence_selector": raw_action.get("occurrence_selector", fallback.get("occurrence_selector")),
        "source_path": raw_action.get("source_path", fallback.get("source_path")),
        "destination_path": raw_action.get("destination_path", fallback.get("destination_path")),
        "paths": _normalize_paths_payload(raw_action, fallback),
        "depends_on": _normalize_list_field(raw_action.get("depends_on", fallback.get("depends_on", []))),
        "inputs_from": _normalize_list_field(raw_action.get("inputs_from", fallback.get("inputs_from", []))),
        "timeout_seconds": raw_action.get("timeout_seconds", fallback.get("timeout_seconds")),
        "max_retries": raw_action.get("max_retries", fallback.get("max_retries")),
        "full_file_rewrite": bool(raw_action.get("full_file_rewrite", fallback.get("full_file_rewrite"))),
        "tool_name": raw_action.get("tool_name", fallback.get("tool_name")),
        "tool_source": raw_action.get("tool_source", fallback.get("tool_source")),
        "tool_args": raw_action.get("tool_args", fallback.get("tool_args")),
        "provider": provider,
        "provider_reason": provider_reason,
    }
    validated = attach_validation(normalized)
    normalized["valid"] = validated.get("is_valid", False)
    normalized["validation_errors"] = validated.get("validation_errors", [])
    return normalized


def build_action_fallback(state: ShipyardState, preferred_mode: str | None = None) -> dict[str, Any]:
    context = state.get("context", {}) or {}
    instruction = state.get("instruction", "")
    parsed = parse_instruction(instruction) if not preferred_mode else None
    edit_mode = preferred_mode or state.get("edit_mode") or (parsed[0] if parsed else None) or infer_edit_mode(state)
    inferred_target = infer_target_path_from_instruction(instruction, context)
    requested_target = resolve_requested_target_hint(state, context, inferred_target)
    target_path, target_path_source = resolve_target_path(
        requested_target,
        context,
        edit_mode,
        session_id=state.get("session_id"),
        instruction=instruction,
    )

    fallback: dict[str, Any] = {
        "id": state.get("task_id"),
        "instruction": instruction,
        "role": state.get("role") or "lead-agent",
        "agent_type": state.get("agent_type") or "primary",
        "parent_task_id": state.get("parent_task_id"),
        "child_task_ids": state.get("child_task_ids", []),
        "allowed_actions": [edit_mode] if edit_mode else [],
        "target_path": target_path,
        "target_path_source": target_path_source,
        "edit_mode": edit_mode,
        "action_class": ACTION_CLASSES.get(edit_mode, "mutate"),
        "anchor": state.get("anchor"),
        "replacement": state.get("replacement") if state.get("replacement") is not None else (parsed[2] if parsed else None),
        "quantity": state.get("quantity"),
        "copy_count": state.get("copy_count"),
        "pointers": state.get("pointers"),
        "pattern": state.get("pattern"),
        "command": state.get("command"),
        "source_path": state.get("source_path"),
        "destination_path": state.get("destination_path"),
        "paths": state.get("paths"),
        "occurrence_selector": state.get("occurrence_selector") or parse_occurrence_selector(instruction),
        "depends_on": state.get("depends_on", []),
        "inputs_from": state.get("inputs_from", []),
        "timeout_seconds": state.get("timeout_seconds"),
        "max_retries": state.get("max_retries"),
        "full_file_rewrite": False,
    }
    if fallback["anchor"] is None and parsed:
        fallback["anchor"] = parsed[1]
    if fallback["copy_count"] is None and edit_mode == "copy_file":
        fallback["copy_count"] = infer_copy_count(instruction)
    if fallback["quantity"] is None and edit_mode == "create_files":
        fallback["quantity"] = infer_create_quantity(instruction)
        if fallback.get("replacement") is None:
            fallback["replacement"] = ""
    return fallback
