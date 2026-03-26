from __future__ import annotations

from typing import Any, TypedDict

from .intent_parser import prefers_append_for_generation
from .proposal_validation import attach_validation


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
