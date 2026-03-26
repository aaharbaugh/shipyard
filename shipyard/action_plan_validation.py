from __future__ import annotations

from pathlib import Path
from typing import Any

from .intent_parser import split_instruction_steps
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

    expected_steps = split_instruction_steps(instruction)
    if len(expected_steps) > 1 and len(actions) < len(expected_steps):
        errors.append(
            f"Instruction implies {len(expected_steps)} steps, but the action plan only returned {len(actions)} action(s)."
        )

    explicit_files = extract_explicit_filenames(instruction)
    if explicit_files:
        covered_files = {
            Path(str(action.get("target_path", ""))).name
            for action in actions
            if action.get("target_path")
        }
        missing_files = [name for name in explicit_files if name not in covered_files]
        if missing_files:
            errors.append(
                "Action plan did not cover all explicitly named files: " + ", ".join(missing_files) + "."
            )

    return errors
