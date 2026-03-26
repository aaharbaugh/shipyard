from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .action_plan_validation import validate_action_plan
from .actions import Action, build_action_fallback, normalize_action
from .intent_parser import split_instruction_steps
from .planning_hints import extract_explicit_filenames
from .repo_context import build_repo_context_lines
from .state import ShipyardState


def plan_actions(state: ShipyardState) -> dict[str, Any]:
    explicit_mode = (state.get("proposal_mode") or "").strip().lower()
    if explicit_mode == "heuristic":
        return _heuristic_action_plan(state)
    if explicit_mode == "openai":
        return _openai_action_plan_or_fallback(state)
    if os.getenv("OPENAI_API_KEY"):
        return _openai_action_plan_or_fallback(state)
    return _heuristic_action_plan(state)


def _heuristic_action_plan(state: ShipyardState) -> dict[str, Any]:
    actions: list[Action] = []
    steps = split_instruction_steps(state.get("instruction", ""))
    for step in steps or [state.get("instruction", "")]:
        step_state: ShipyardState = {
            **state,
            "instruction": step,
        }
        proposal = build_action_fallback(step_state)
        actions.append(
            normalize_action(
                {"instruction": step},
                fallback=proposal,
                provider="heuristic",
                provider_reason="Derived action sequence from instruction text.",
            )
        )
    validation_errors = validate_action_plan(state.get("instruction", ""), actions)
    return {
        "actions": actions,
        "provider": "heuristic",
        "provider_reason": "Derived action sequence from instruction text.",
        "is_valid": not validation_errors,
        "validation_errors": validation_errors,
    }


def _openai_action_plan_or_fallback(state: ShipyardState) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = state.get("proposal_model") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    if not api_key:
        return _heuristic_action_plan(state)

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": _build_action_plan_prompt(state),
                    "text": {"format": {"type": "json_object"}},
                },
            )
            response.raise_for_status()
    except Exception as exc:
        actions = [
            normalize_action(
                {"instruction": state.get("instruction", ""), "edit_mode": state.get("edit_mode") or "anchor"},
                fallback={},
                provider="openai",
                provider_reason=f"OpenAI action planning failed: {exc}",
            )
        ]
        validation_errors = validate_action_plan(state.get("instruction", ""), actions)
        return {
            "actions": actions,
            "provider": "openai",
            "provider_reason": f"OpenAI action planning failed: {exc}",
            "is_valid": False,
            "validation_errors": validation_errors,
        }

    body = response.json()
    output_text = body.get("output_text", "{}")
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        actions = [
            normalize_action(
                {"instruction": state.get("instruction", ""), "edit_mode": state.get("edit_mode") or "anchor"},
                fallback={},
                provider="openai",
                provider_reason="OpenAI action plan response was not valid JSON.",
            )
        ]
        validation_errors = validate_action_plan(state.get("instruction", ""), actions)
        return {
            "actions": actions,
            "provider": "openai",
            "provider_reason": "OpenAI action plan response was not valid JSON.",
            "is_valid": False,
            "validation_errors": validation_errors,
        }

    actions = parsed.get("actions")
    if not isinstance(actions, list) or not actions:
        actions = [
            normalize_action(
                {"instruction": state.get("instruction", ""), "edit_mode": state.get("edit_mode") or "anchor"},
                fallback={},
                provider="openai",
                provider_reason="OpenAI action plan response did not include actions.",
            )
        ]
        validation_errors = validate_action_plan(state.get("instruction", ""), actions)
        return {
            "actions": actions,
            "provider": "openai",
            "provider_reason": "OpenAI action plan response did not include actions.",
            "is_valid": False,
            "validation_errors": validation_errors,
        }

    normalized: list[Action] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        fallback = build_action_fallback(
            {**state, "instruction": step_instruction},
            preferred_mode=action.get("edit_mode"),
        )
        action_payload = {
            **action,
            "instruction": step_instruction,
        }
        if fallback.get("target_path_source") in {"explicit_target_path", "sandboxed_target_path", "file_hint"}:
            action_payload.pop("target_path", None)
            action_payload.pop("target_path_source", None)
        normalized.append(
            normalize_action(
                action_payload,
                fallback=fallback,
                provider="openai",
                provider_reason=f"OpenAI model {model} produced action plan.",
            )
        )
    if not normalized:
        actions = [
            normalize_action(
                {"instruction": state.get("instruction", ""), "edit_mode": state.get("edit_mode") or "anchor"},
                fallback={},
                provider="openai",
                provider_reason="OpenAI action plan response could not be normalized.",
            )
        ]
        validation_errors = validate_action_plan(state.get("instruction", ""), actions)
        return {
            "actions": actions,
            "provider": "openai",
            "provider_reason": "OpenAI action plan response could not be normalized.",
            "is_valid": False,
            "validation_errors": validation_errors,
        }
    validation_errors = validate_action_plan(state.get("instruction", ""), normalized)
    return {
        "actions": normalized,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} produced action plan.",
        "is_valid": not validation_errors,
        "validation_errors": validation_errors,
    }


def _build_action_plan_prompt(state: ShipyardState) -> str:
    instruction = state.get("instruction", "").strip()
    explicit_files = extract_explicit_filenames(instruction)
    lines = [
        "Return only JSON.",
        "Plan the user request as an ordered list of actions.",
        "Use the key actions with an array of objects.",
        "Each action should include: instruction, edit_mode, target_path, anchor, replacement, quantity, copy_count, occurrence_selector.",
        "Allowed edit_mode values: anchor, named_function, write_file, append, prepend, delete_file, copy_file, create_files, rename_symbol.",
        "Do not omit separate steps just because they appear in one sentence.",
        "If the user explicitly names a file, preserve that exact file as target_path instead of inventing a scratch file.",
        "If the user explicitly lists multiple files for a tiny repo or project scaffold, return concrete actions that cover every listed file.",
        "Do not collapse an explicit file list into a single generic file action.",
        "If the instruction names multiple files, ensure the final plan covers each named file with a concrete action.",
        "Prefer one explicit action per file when scaffolding a tiny repo with named files.",
        "If the user asks to add or insert code into a named file, prefer append and preserve the existing file content.",
        "If the user asks to write or create code in a named file, use write_file unless the wording clearly implies insertion.",
        "For code-generation requests, put concrete code in replacement.",
        "Do not leave replacement empty for content-generation requests.",
        "Do not return placeholder text as replacement for code-generation requests.",
        f"Instruction: {instruction}",
        "Lightweight repository context:",
    ]
    if explicit_files:
        lines.append("Explicit files mentioned by the user:")
        lines.extend(f"- {name}" for name in explicit_files)
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))
    return "\n".join(lines)
