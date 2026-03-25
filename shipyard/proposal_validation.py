from __future__ import annotations

from typing import Any


def attach_validation(proposal: dict[str, Any]) -> dict[str, Any]:
    validated = dict(proposal)
    errors = validate_proposal(validated)
    validated["is_valid"] = not errors
    validated["validation_errors"] = errors
    return validated


def validate_proposal(proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    edit_mode = proposal.get("edit_mode") or "anchor"
    target_path = proposal.get("target_path")
    anchor = proposal.get("anchor")
    replacement = proposal.get("replacement")

    if not target_path:
        errors.append("Missing target_path.")

    if edit_mode == "anchor":
        if not anchor:
            errors.append("Anchor mode requires anchor.")
        if replacement is None:
            errors.append("Anchor mode requires replacement.")
    elif edit_mode == "named_function":
        pass
    elif edit_mode == "rename_symbol":
        if not anchor:
            errors.append("rename_symbol mode requires anchor.")
        if replacement is None:
            errors.append("rename_symbol mode requires replacement.")
    elif edit_mode in {"write_file", "append", "prepend"}:
        if replacement is None:
            errors.append(f"{edit_mode} mode requires replacement content.")
    elif edit_mode == "create_files":
        quantity = proposal.get("quantity")
        if quantity is None:
            errors.append("create_files mode requires quantity.")
        else:
            try:
                if int(quantity) < 1:
                    errors.append("create_files mode requires quantity >= 1.")
            except (TypeError, ValueError):
                errors.append("create_files mode requires a numeric quantity.")
    elif edit_mode == "delete_file":
        pass
    elif edit_mode == "copy_file":
        copy_count = proposal.get("copy_count")
        if copy_count is None:
            errors.append("copy_file mode requires copy_count.")
        else:
            try:
                if int(copy_count) < 1:
                    errors.append("copy_file mode requires copy_count >= 1.")
            except (TypeError, ValueError):
                errors.append("copy_file mode requires a numeric copy_count.")
    else:
        errors.append(f"Unsupported edit_mode: {edit_mode}.")

    return errors
