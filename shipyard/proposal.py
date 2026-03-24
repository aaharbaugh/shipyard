from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .prompts import build_proposal_prompt
from .state import ShipyardState


REPLACE_PATTERNS = (
    r'replace\s+"(?P<old>.+?)"\s+with\s+"(?P<new>.+?)"',
    r"replace\s+'(?P<old>.+?)'\s+with\s+'(?P<new>.+?)'",
    r'change\s+"(?P<old>.+?)"\s+to\s+"(?P<new>.+?)"',
    r"change\s+'(?P<old>.+?)'\s+to\s+'(?P<new>.+?)'",
)


def propose_edit(state: ShipyardState) -> dict[str, Any]:
    explicit_mode = state.get("proposal_mode")
    if explicit_mode == "heuristic":
        return _heuristic_proposal(state)
    if explicit_mode == "openai":
        return _openai_or_fallback(state)

    if os.getenv("OPENAI_API_KEY"):
        return _openai_or_fallback(state)
    return _heuristic_proposal(state)


def _openai_or_fallback(state: ShipyardState) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = state.get("proposal_model") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    if not api_key:
        proposal = _heuristic_proposal(state)
        proposal["provider"] = "heuristic"
        proposal["provider_reason"] = "OPENAI_API_KEY not configured."
        return proposal

    prompt = build_proposal_prompt(state)
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
                    "input": prompt,
                    "text": {"format": {"type": "json_object"}},
                },
            )
            response.raise_for_status()
    except Exception as exc:
        proposal = _heuristic_proposal(state)
        proposal["provider"] = "heuristic"
        proposal["provider_reason"] = f"OpenAI proposal failed: {exc}"
        return proposal

    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        proposal = _heuristic_proposal(state)
        proposal["provider"] = "heuristic"
        proposal["provider_reason"] = "OpenAI response was not valid JSON."
        return proposal

    return {
        "target_path": parsed.get("target_path") or state.get("target_path") or state.get("context", {}).get("file_hint"),
        "anchor": parsed.get("anchor"),
        "replacement": parsed.get("replacement"),
        "edit_mode": "named_function" if state.get("context", {}).get("function_name") else "anchor",
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} generated proposal.",
    }


def _extract_response_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]

    outputs = body.get("output", [])
    for item in outputs:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return "{}"


def _heuristic_proposal(state: ShipyardState) -> dict[str, Any]:
    context = state.get("context", {})
    anchor = state.get("anchor")
    replacement = state.get("replacement")
    notes: list[str] = []
    edit_mode = "anchor"

    if context.get("function_name"):
        edit_mode = "named_function"
        notes.append("Using named-function edit mode from injected context.")

    if anchor and replacement is not None:
        notes.append("Edit spec already provided by caller.")
    else:
        if context.get("search_text") and context.get("replace_text"):
            anchor = context["search_text"]
            replacement = context["replace_text"]
            notes.append("Derived anchor and replacement from injected context.")
        else:
            parsed = _parse_replace_instruction(state.get("instruction", ""))
            if parsed:
                anchor, replacement = parsed
                notes.append("Derived anchor and replacement from instruction text.")

    target_path = state.get("target_path") or context.get("file_hint")
    if context.get("file_hint") and not state.get("target_path"):
        notes.append("Using file_hint as target path.")

    return {
        "target_path": target_path,
        "anchor": anchor,
        "replacement": replacement,
        "edit_mode": edit_mode,
        "provider": "heuristic",
        "provider_reason": "; ".join(notes) if notes else "No heuristic edit proposal could be derived.",
    }


def _parse_replace_instruction(instruction: str) -> tuple[str, str] | None:
    for pattern in REPLACE_PATTERNS:
        match = re.search(pattern, instruction.strip(), flags=re.IGNORECASE)
        if match:
            return match.group("old"), match.group("new")
    return None
