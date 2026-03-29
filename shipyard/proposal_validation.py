from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MUTATE_INTENTS = {
    "localized_edit",
    "function_rewrite",
    "full_rewrite",
    "append_content",
    "prepend_content",
    "scaffold",
    "filesystem_mutation",
}

EDIT_SCOPES = {
    "single_span",
    "multi_span",
    "function_block",
    "full_file",
    "file_tail",
    "file_head",
    "new_files",
    "filesystem",
}

EXPECTED_EXISTING_STATES = {"existing_file", "new_file", "either"}
RECOVERY_STRATEGIES = {"replan_step", "replan_slice", "block"}


def _looks_like_placeholder_replacement(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    # Structural check 1: machine-generated ALL_CAPS_UPDATED token (LLM stub pattern)
    if re.fullmatch(r"[A-Z][A-Z0-9_]*_UPDATED", text):
        return True
    # Structural check 2: every line is a comment (pure stub, no actual code)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    comment_prefixes = ("#", "//", "/*", "*", "<!--")
    return all(line.startswith(comment_prefixes) for line in lines)


def attach_validation(proposal: dict[str, Any]) -> dict[str, Any]:
    validated = dict(proposal)
    errors = validate_proposal(validated)
    validated["is_valid"] = not errors
    validated["validation_errors"] = errors
    return validated


def validate_proposal(proposal: dict[str, Any]) -> list[str]:
    # Determinism boundary: validation may reject unsafe or structurally invalid
    # proposals, but it should not reinterpret user intent or synthesize new edits.
    errors: list[str] = []
    edit_mode = proposal.get("edit_mode") or "anchor"
    target_path = proposal.get("target_path")
    anchor = proposal.get("anchor")
    replacement = proposal.get("replacement")
    pointers = proposal.get("pointers")
    source_path = proposal.get("source_path")
    destination_path = proposal.get("destination_path")
    paths = proposal.get("paths")
    action_class = proposal.get("action_class")
    intent = proposal.get("intent")
    edit_scope = proposal.get("edit_scope")
    expected_existing_state = proposal.get("expected_existing_state")
    recovery_strategy = proposal.get("recovery_strategy")

    if action_class == "mutate":
        if intent not in MUTATE_INTENTS:
            errors.append("Mutate actions require a valid intent.")
        if edit_scope not in EDIT_SCOPES:
            errors.append("Mutate actions require a valid edit_scope.")
        if expected_existing_state not in EXPECTED_EXISTING_STATES:
            errors.append("Mutate actions require a valid expected_existing_state.")
        if recovery_strategy not in RECOVERY_STRATEGIES:
            errors.append("Mutate actions require a valid recovery_strategy.")

    if not target_path and edit_mode not in {"scaffold_files", "run_command", "verify_command", "run_tests", "move_file", "rename_file", "read_many_files"}:
        errors.append("Missing target_path.")

    if edit_mode == "anchor":
        if intent and intent != "localized_edit":
            errors.append("Anchor mode requires intent=localized_edit.")
        if edit_scope and edit_scope not in {"single_span", "multi_span"}:
            errors.append("Anchor mode requires edit_scope=single_span or multi_span.")
        if not anchor and not pointers:
            errors.append("Anchor mode requires anchor.")
        if replacement is None:
            errors.append("Anchor mode requires replacement.")
        elif anchor is not None and str(anchor) == str(replacement):
            errors.append("Anchor mode replacement must differ from anchor.")
        elif _looks_like_placeholder_replacement(replacement):
            errors.append("Anchor mode replacement must contain actual code or content, not a placeholder.")
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
        if intent and intent != "function_rewrite":
            errors.append("named_function mode requires intent=function_rewrite.")
        if edit_scope and edit_scope != "function_block":
            errors.append("named_function mode requires edit_scope=function_block.")
    elif edit_mode == "rename_symbol":
        if intent and intent != "localized_edit":
            errors.append("rename_symbol mode requires intent=localized_edit.")
        if not anchor:
            errors.append("rename_symbol mode requires anchor.")
        if replacement is None:
            errors.append("rename_symbol mode requires replacement.")
        elif str(anchor) == str(replacement):
            errors.append("rename_symbol replacement must differ from anchor.")
    elif edit_mode in {"write_file", "append", "prepend"}:
        if edit_mode == "write_file":
            if intent and intent != "full_rewrite":
                errors.append("write_file mode requires intent=full_rewrite.")
            if edit_scope and edit_scope != "full_file":
                errors.append("write_file mode requires edit_scope=full_file.")
        elif edit_mode == "append":
            if intent and intent != "append_content":
                errors.append("append mode requires intent=append_content.")
            if edit_scope and edit_scope != "file_tail":
                errors.append("append mode requires edit_scope=file_tail.")
        elif edit_mode == "prepend":
            if intent and intent != "prepend_content":
                errors.append("prepend mode requires intent=prepend_content.")
            if edit_scope and edit_scope != "file_head":
                errors.append("prepend mode requires edit_scope=file_head.")
        # Allow deferred content when the step depends on inspect steps (inspect-first pattern).
        # The runtime's _refine_preplanned_action will fill in replacement after reading the file.
        has_dependencies = bool(proposal.get("depends_on") or proposal.get("inputs_from"))
        if replacement is None and not has_dependencies:
            errors.append(f"{edit_mode} mode requires replacement content.")
        elif replacement is not None and _looks_like_placeholder_replacement(replacement):
            errors.append(f"{edit_mode} mode requires actual content, not a placeholder.")
        # full_file_rewrite is advisory metadata — write_file mode already implies a full
        # rewrite, so the LLM not setting this flag should not block execution.
    elif edit_mode == "create_files":
        # If target_path is set, treat as a single file creation (quantity=1 implied)
        quantity = proposal.get("quantity")
        if quantity is None and not target_path:
            errors.append("create_files mode requires quantity or target_path.")
        elif quantity is not None:
            try:
                if int(quantity) < 1:
                    errors.append("create_files mode requires quantity >= 1.")
            except (TypeError, ValueError):
                errors.append("create_files mode requires a numeric quantity.")
    elif edit_mode == "scaffold_files":
        if intent and intent != "scaffold":
            errors.append("scaffold_files mode requires intent=scaffold.")
        if edit_scope and edit_scope != "new_files":
            errors.append("scaffold_files mode requires edit_scope=new_files.")
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
                path = file_spec.get("path")
                path_obj = Path(str(path)) if path else None
                if path_obj and path_obj.is_absolute() and path_obj.exists():
                    errors.append(
                        f"scaffold_files entry {index} targets an existing file. Use a localized edit or explicit full rewrite instead."
                    )
    elif edit_mode == "delete_file":
        if intent and intent != "filesystem_mutation":
            errors.append("delete_file mode requires intent=filesystem_mutation.")
    elif edit_mode == "copy_file":
        if intent and intent != "filesystem_mutation":
            errors.append("copy_file mode requires intent=filesystem_mutation.")
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
        # Allow no paths when target_path is a directory — runtime auto-populates from dir
        has_paths = isinstance(paths, list) and paths
        has_dir_target = bool(target_path and Path(str(target_path)).is_dir())
        if not has_paths and not has_dir_target:
            errors.append("read_many_files mode requires paths or a directory target_path.")
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
        if intent and intent != "localized_edit":
            errors.append("search_and_replace mode requires intent=localized_edit.")
        if edit_scope and edit_scope not in {"single_span", "multi_span"}:
            errors.append("search_and_replace mode requires edit_scope=single_span or multi_span.")
        has_dependencies = bool(proposal.get("depends_on") or proposal.get("inputs_from"))
        if not proposal.get("pattern") and not anchor and not has_dependencies:
            errors.append("search_and_replace mode requires pattern or anchor.")
        if replacement is None and not has_dependencies:
            errors.append("search_and_replace mode requires replacement.")
        elif (proposal.get("pattern") and str(proposal.get("pattern")) == str(replacement)) or (anchor and str(anchor) == str(replacement)):
            errors.append("search_and_replace replacement must differ from the search text.")
        elif _looks_like_placeholder_replacement(replacement):
            errors.append("search_and_replace replacement must contain actual content, not a placeholder.")
    elif edit_mode == "create_directory":
        if intent and intent != "filesystem_mutation":
            errors.append("create_directory mode requires intent=filesystem_mutation.")
        if not target_path:
            errors.append("create_directory mode requires target_path.")
    elif edit_mode in {"move_file", "rename_file"}:
        if intent and intent != "filesystem_mutation":
            errors.append(f"{edit_mode} mode requires intent=filesystem_mutation.")
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
