from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .actions import Action, normalize_action
from .intent_parser import split_instruction_steps
from .proposal import propose_edit
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
        proposal = propose_edit(step_state)
        actions.append(
            normalize_action(
                {"instruction": step},
                fallback=proposal,
                provider=str(proposal.get("provider") or "heuristic"),
                provider_reason=str(proposal.get("provider_reason") or "Derived action sequence from instruction text."),
            )
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

    normalized: list[Action] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        heuristic = propose_edit({**state, "instruction": step_instruction})
        fallback = dict(heuristic)
        action_payload = {
            **action,
            "instruction": step_instruction,
        }
        if heuristic.get("target_path_source") in {"explicit_target_path", "sandboxed_target_path", "file_hint"}:
            fallback["target_path"] = heuristic.get("target_path")
            fallback["target_path_source"] = heuristic.get("target_path_source")
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
            "Allowed edit_mode values: anchor, named_function, write_file, append, prepend, delete_file, copy_file, create_files, rename_symbol.",
            "Do not omit separate steps just because they appear in one sentence.",
            "If the user explicitly names a file, preserve that exact file as target_path instead of inventing a scratch file.",
            "If the user asks to add or insert code into a named file, prefer append and preserve the existing file content.",
            "If the user asks to write or create code in a named file, use write_file unless the wording clearly implies insertion.",
            "For code-generation requests, put concrete code in replacement.",
            "Do not leave replacement empty for content-generation requests.",
            f"Instruction: {instruction}",
        ]
    )
