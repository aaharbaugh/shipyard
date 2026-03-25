from __future__ import annotations

import json
import os
from typing import Any, TypedDict

import httpx

from .intent_parser import parse_instruction, prefers_append_for_generation, split_instruction_steps
from .proposal import propose_edit
from .state import ShipyardState


class PlannedAction(TypedDict, total=False):
    instruction: str
    target_path: str | None
    edit_mode: str
    anchor: str | None
    replacement: str | None
    quantity: int | None
    copy_count: int | None
    occurrence_selector: str | None
    target_path_source: str | None
    valid: bool
    validation_errors: list[str]
    provider: str
    provider_reason: str


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
    actions: list[PlannedAction] = []
    steps = split_instruction_steps(state.get("instruction", ""))
    for step in steps or [state.get("instruction", "")]:
        step_state: ShipyardState = {
            **state,
            "instruction": step,
        }
        proposal = propose_edit(step_state)
        actions.append(
            {
                "instruction": step,
                "target_path": proposal.get("target_path"),
                "target_path_source": proposal.get("target_path_source"),
                "edit_mode": proposal.get("edit_mode"),
                "anchor": proposal.get("anchor"),
                "replacement": proposal.get("replacement"),
                "quantity": proposal.get("quantity"),
                "copy_count": proposal.get("copy_count"),
                "occurrence_selector": proposal.get("occurrence_selector"),
                "valid": proposal.get("is_valid"),
                "validation_errors": proposal.get("validation_errors", []),
                "provider": proposal.get("provider"),
                "provider_reason": proposal.get("provider_reason"),
            }
        )
    return {
        "actions": actions,
        "provider": "heuristic",
        "provider_reason": "Derived action sequence from instruction text.",
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
    except Exception:
        return _heuristic_action_plan(state)

    body = response.json()
    output_text = body.get("output_text", "{}")
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return _heuristic_action_plan(state)

    actions = parsed.get("actions")
    if not isinstance(actions, list) or not actions:
        return _heuristic_action_plan(state)

    normalized: list[PlannedAction] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        heuristic = propose_edit({**state, "instruction": step_instruction})
        resolved_target = heuristic.get("target_path") if heuristic.get("target_path_source") in {
            "explicit_target_path",
            "sandboxed_target_path",
            "file_hint",
        } else action.get("target_path") or heuristic.get("target_path")
        resolved_target_source = heuristic.get("target_path_source") if heuristic.get("target_path_source") in {
            "explicit_target_path",
            "sandboxed_target_path",
            "file_hint",
        } else action.get("target_path_source") or heuristic.get("target_path_source")
        normalized.append(
            {
                "instruction": step_instruction,
                "target_path": resolved_target,
                "target_path_source": resolved_target_source,
                "edit_mode": (
                    "append"
                    if prefers_append_for_generation(step_instruction)
                    and (action.get("edit_mode") or heuristic.get("edit_mode")) == "write_file"
                    else action.get("edit_mode") or heuristic.get("edit_mode")
                ),
                "anchor": action.get("anchor", heuristic.get("anchor")),
                "replacement": action.get("replacement", heuristic.get("replacement")),
                "quantity": action.get("quantity", heuristic.get("quantity")),
                "copy_count": action.get("copy_count", heuristic.get("copy_count")),
                "occurrence_selector": action.get("occurrence_selector", heuristic.get("occurrence_selector")),
                "valid": heuristic.get("is_valid"),
                "validation_errors": heuristic.get("validation_errors", []),
                "provider": "openai",
                "provider_reason": f"OpenAI model {model} produced action plan.",
            }
        )
    if not normalized:
        return _heuristic_action_plan(state)
    return {
        "actions": normalized,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} produced action plan.",
    }


def _build_action_plan_prompt(state: ShipyardState) -> str:
    instruction = state.get("instruction", "").strip()
    return "\n".join(
        [
            "Return only JSON.",
            "Plan the user request as an ordered list of actions.",
            "Use the key actions with an array of objects.",
            "Each action should include: instruction, edit_mode, target_path, anchor, replacement, quantity, copy_count, occurrence_selector.",
            "Do not omit separate steps just because they appear in one sentence.",
            "If the user explicitly names a file, preserve that exact file as target_path instead of inventing a scratch file.",
            "If the user asks to add or insert code into a named file, prefer append and preserve the existing file content.",
            "If the user asks to write or create code in a named file, use write_file unless the wording clearly implies insertion.",
            "For code-generation requests, put concrete code in replacement.",
            "Do not leave replacement empty for content-generation requests.",
            f"Instruction: {instruction}",
        ]
    )
