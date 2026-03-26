from __future__ import annotations

from typing import Any, TypedDict

from .intent_parser import (
    infer_edit_mode,
    parse_instruction,
    parse_occurrence_selector,
    prefers_append_for_generation,
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
    "rename_symbol",
}


class Action(TypedDict, total=False):
    instruction: str
    target_path: str | None
    target_path_source: str | None
    edit_mode: str
    anchor: str | None
    replacement: str | None
    quantity: int | None
    copy_count: int | None
    occurrence_selector: str | None
    valid: bool
    validation_errors: list[str]
    provider: str
    provider_reason: str


def normalize_action(
    raw_action: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    provider: str,
    provider_reason: str,
) -> Action:
    fallback = fallback or {}
    instruction = str(raw_action.get("instruction") or fallback.get("instruction") or "").strip()
    edit_mode = str(raw_action.get("edit_mode") or fallback.get("edit_mode") or "anchor").strip()
    if edit_mode not in SUPPORTED_ACTIONS:
        edit_mode = str(fallback.get("edit_mode") or "anchor")

    replacement = raw_action.get("replacement")
    if replacement is None and raw_action.get("content") is not None:
        replacement = raw_action.get("content")
    if replacement is None:
        replacement = fallback.get("replacement")

    anchor = raw_action.get("anchor", fallback.get("anchor"))

    if (
        edit_mode == "write_file"
        and instruction
        and prefers_append_for_generation(instruction)
    ):
        edit_mode = "append"

    normalized: Action = {
        "instruction": instruction,
        "target_path": raw_action.get("target_path") or fallback.get("target_path"),
        "target_path_source": raw_action.get("target_path_source") or fallback.get("target_path_source"),
        "edit_mode": edit_mode,
        "anchor": anchor,
        "replacement": replacement,
        "quantity": raw_action.get("quantity", fallback.get("quantity")),
        "copy_count": raw_action.get("copy_count", fallback.get("copy_count")),
        "occurrence_selector": raw_action.get("occurrence_selector", fallback.get("occurrence_selector")),
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
        "instruction": instruction,
        "target_path": target_path,
        "target_path_source": target_path_source,
        "edit_mode": edit_mode,
        "anchor": state.get("anchor"),
        "replacement": state.get("replacement") if state.get("replacement") is not None else (parsed[2] if parsed else None),
        "quantity": state.get("quantity"),
        "copy_count": state.get("copy_count"),
        "occurrence_selector": state.get("occurrence_selector") or parse_occurrence_selector(instruction),
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
