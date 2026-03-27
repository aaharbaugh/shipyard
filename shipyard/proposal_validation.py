from __future__ import annotations

from pathlib import Path
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
    pointers = proposal.get("pointers")
    source_path = proposal.get("source_path")
    destination_path = proposal.get("destination_path")
    paths = proposal.get("paths")

    if not target_path and edit_mode not in {"scaffold_files", "run_command", "verify_command", "run_tests", "move_file", "rename_file", "read_many_files"}:
        errors.append("Missing target_path.")

    if edit_mode == "anchor":
        if not anchor and not pointers:
            errors.append("Anchor mode requires anchor.")
        if replacement is None:
            errors.append("Anchor mode requires replacement.")
        if pointers is not None:
            if not isinstance(pointers, list) or not pointers:
                errors.append("Anchor mode pointers must be a non-empty list.")
            else:
                for index, pointer in enumerate(pointers, start=1):
                    if not isinstance(pointer, dict):
                        errors.append(f"Anchor mode pointer {index} must be an object.")
                        continue
                    start = pointer.get("start")
                    end = pointer.get("end")
                    if not isinstance(start, int) or not isinstance(end, int):
                        errors.append(f"Anchor mode pointer {index} requires integer start and end.")
                        continue
                    if start < 0 or end < 0 or end < start:
                        errors.append(f"Anchor mode pointer {index} has an invalid range.")
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
        if edit_mode == "write_file" and target_path and Path(str(target_path)).exists():
            if not proposal.get("full_file_rewrite"):
                errors.append("write_file on an existing file requires full_file_rewrite=true.")
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
    elif edit_mode == "scaffold_files":
        files = proposal.get("files")
        if not isinstance(files, list) or not files:
            errors.append("scaffold_files mode requires files.")
        else:
            for index, file_spec in enumerate(files, start=1):
                if not isinstance(file_spec, dict):
                    errors.append(f"scaffold_files entry {index} must be an object.")
                    continue
                if not file_spec.get("path"):
                    errors.append(f"scaffold_files entry {index} requires path.")
                if file_spec.get("content") is None:
                    errors.append(f"scaffold_files entry {index} requires content.")
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
    elif edit_mode == "list_files":
        if replacement is not None or anchor or pointers or proposal.get("command") or proposal.get("pattern"):
            errors.append("list_files mode cannot include edit content.")
    elif edit_mode == "read_file":
        if replacement is not None or anchor or pointers or proposal.get("command") or proposal.get("pattern"):
            errors.append("read_file mode cannot include edit content.")
    elif edit_mode == "read_many_files":
        if not isinstance(paths, list) or not paths:
            errors.append("read_many_files mode requires paths.")
        if replacement is not None or anchor or pointers or proposal.get("command") or proposal.get("pattern"):
            errors.append("read_many_files mode cannot include edit content.")
    elif edit_mode == "search_files":
        if not proposal.get("pattern"):
            errors.append("search_files mode requires pattern.")
        if replacement is not None or anchor or pointers or proposal.get("command"):
            errors.append("search_files mode cannot include edit content.")
    elif edit_mode == "inspect_imports":
        if replacement is not None or anchor or pointers or proposal.get("command") or proposal.get("pattern"):
            errors.append("inspect_imports mode cannot include edit content.")
    elif edit_mode == "search_and_replace":
        if not proposal.get("pattern") and not anchor:
            errors.append("search_and_replace mode requires pattern or anchor.")
        if replacement is None:
            errors.append("search_and_replace mode requires replacement.")
    elif edit_mode == "create_directory":
        if not target_path:
            errors.append("create_directory mode requires target_path.")
    elif edit_mode in {"move_file", "rename_file"}:
        if not source_path:
            errors.append(f"{edit_mode} mode requires source_path.")
        if not destination_path:
            errors.append(f"{edit_mode} mode requires destination_path.")
    elif edit_mode == "run_command":
        if not proposal.get("command"):
            errors.append("run_command mode requires command.")
        if replacement is not None or anchor or pointers or proposal.get("pattern"):
            errors.append("run_command mode cannot include edit content.")
    elif edit_mode == "verify_command":
        if not proposal.get("command"):
            errors.append("verify_command mode requires command.")
        if replacement is not None or anchor or pointers or proposal.get("pattern"):
            errors.append("verify_command mode cannot include edit content.")
    elif edit_mode == "run_tests":
        if not proposal.get("command"):
            errors.append("run_tests mode requires command.")
        if replacement is not None or anchor or pointers or proposal.get("pattern"):
            errors.append("run_tests mode cannot include edit content.")
    else:
        errors.append(f"Unsupported edit_mode: {edit_mode}.")

    return errors
