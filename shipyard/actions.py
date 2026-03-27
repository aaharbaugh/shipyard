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
}


class Action(TypedDict, total=False):
    id: str
    instruction: str
    target_path: str | None
    target_path_source: str | None
    edit_mode: str
    action_class: str
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
    valid: bool
    validation_errors: list[str]
    provider: str
    provider_reason: str


def _normalize_pointer_payload(value: Any) -> list[dict[str, int]] | None:
    if value in (None, "", []):
        return None
    return value


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

    normalized: Action = {
        "id": action_id,
        "instruction": instruction,
        "target_path": raw_action.get("target_path") or fallback.get("target_path"),
        "target_path_source": raw_action.get("target_path_source") or fallback.get("target_path_source"),
        "edit_mode": edit_mode,
        "action_class": action_class,
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
        "paths": raw_action.get("paths", fallback.get("paths")),
        "depends_on": list(raw_action.get("depends_on", fallback.get("depends_on", [])) or []),
        "inputs_from": list(raw_action.get("inputs_from", fallback.get("inputs_from", [])) or []),
        "timeout_seconds": raw_action.get("timeout_seconds", fallback.get("timeout_seconds")),
        "max_retries": raw_action.get("max_retries", fallback.get("max_retries")),
        "full_file_rewrite": bool(raw_action.get("full_file_rewrite", fallback.get("full_file_rewrite"))),
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
