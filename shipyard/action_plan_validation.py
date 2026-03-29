from __future__ import annotations

from pathlib import Path
from typing import Any

from .planning_hints import extract_explicit_filenames


def validate_action_plan(instruction: str, actions: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not actions:
        return ["Action plan did not include any actions."]

    invalid_actions = [index + 1 for index, action in enumerate(actions) if not action.get("valid")]
    if invalid_actions:
        errors.append(
            "Action plan contains invalid actions at positions: "
            + ", ".join(str(index) for index in invalid_actions)
            + "."
        )

    explicit_files = extract_explicit_filenames(instruction)
    action_ids = [str(action.get("id")) for action in actions if action.get("id")]
    if len(action_ids) != len(set(action_ids)):
        errors.append("Action plan contains duplicate step ids.")
    known_ids = set(action_ids)
    for index, action in enumerate(actions, start=1):
        deps = action.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        for dependency in deps:
            if dependency not in known_ids:
                errors.append(f"Action {index} depends on unknown step id `{dependency}`.")
        inputs = action.get("inputs_from") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        for dependency in inputs:
            if dependency not in known_ids:
                errors.append(f"Action {index} references unknown inputs_from step id `{dependency}`.")
    if explicit_files:
        covered_files: set[str] = set()
        for action in actions:
            tp = action.get("target_path")
            if tp:
                tp_str = str(tp)
                covered_files.add(Path(tp_str).name)
                covered_files.add(tp_str)  # also match full paths
                # Match the relative suffix (api/src/routes/auth.ts matches auth.ts)
                if "/" in tp_str:
                    covered_files.add(tp_str.rsplit("/", 1)[-1])
            for file_spec in (action.get("files") or []):
                if isinstance(file_spec, dict) and file_spec.get("path"):
                    fp = str(file_spec["path"])
                    covered_files.add(Path(fp).name)
                    covered_files.add(fp)
        missing_files = [
            name for name in explicit_files
            if name not in covered_files and Path(name).name not in covered_files
        ]
        if missing_files:
            errors.append(
                "Action plan did not cover all explicitly named files: " + ", ".join(missing_files) + "."
            )

    return errors


def check_inspect_first(actions: list[dict[str, Any]]) -> list[str]:
    """Check that mutate steps on existing files depend on an inspect step.

    Returns warnings (not hard errors) — plans missing inspection are suboptimal
    but the runtime's fetch_step_context will still load file contents before editing.
    """
    warnings: list[str] = []
    _INSPECT_MODES = {"read_file", "read_many_files", "search_files", "list_files", "run_command", "inspect_imports"}
    _MUTATE_MODES = {"write_file", "search_and_replace", "anchor", "named_function", "rename_symbol", "append", "prepend"}
    inspect_ids: set[str] = set()
    for action in actions:
        if action.get("action_class") == "inspect" or str(action.get("edit_mode", "")) in _INSPECT_MODES:
            aid = str(action.get("id", ""))
            if aid:
                inspect_ids.add(aid)

    for index, action in enumerate(actions, start=1):
        mode = str(action.get("edit_mode", ""))
        if mode not in _MUTATE_MODES:
            continue
        if action.get("expected_existing_state") == "new_file":
            continue
        deps = set(action.get("depends_on", []) or []) | set(action.get("inputs_from", []) or [])
        if deps and deps & inspect_ids:
            continue
        target = action.get("target_path", "unknown")
        warnings.append(
            f"Action {index} ({mode} on {target}) has no inspect dependency. "
            "Plans should inspect files (read_file, run_command) before editing them."
        )

    return warnings
