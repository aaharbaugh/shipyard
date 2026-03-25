from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .intent_parser import (
    derive_edit_spec,
    infer_edit_mode,
    parse_instruction,
    parse_occurrence_selector,
)
from .pathing import resolve_target_path
from .prompts import build_proposal_prompt
from .proposal_validation import attach_validation
from .state import ShipyardState

FILENAME_PATTERN = re.compile(r"\b(?P<name>[\w.-]+\.[A-Za-z0-9]+)\b")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def propose_edit(state: ShipyardState) -> dict[str, Any]:
    explicit_mode = (state.get("proposal_mode") or "").strip().lower()
    if explicit_mode == "heuristic":
        return _heuristic_proposal(state)
    if explicit_mode == "openai":
        return _openai_or_fallback(state)

    if os.getenv("OPENAI_API_KEY"):
        return _openai_or_fallback(state)
    return _heuristic_proposal(state)


def get_planner_status() -> dict[str, Any]:
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    default_mode = "openai" if has_api_key else "heuristic"
    return {
        "default_mode": default_mode,
        "openai_configured": has_api_key,
        "proposal_model": model if has_api_key else None,
        "summary": (
            f"LLM planner is active via {model}."
            if has_api_key
            else "OpenAI is not configured. Shipyard will use heuristic planning."
        ),
    }


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

    normalized = _normalize_openai_proposal(parsed, state)
    normalized = attach_validation(normalized)
    if not normalized.get("is_valid"):
        proposal = _heuristic_proposal(state)
        if proposal.get("is_valid"):
            proposal["provider"] = "heuristic"
            proposal["provider_reason"] = (
                f"OpenAI model {model} returned an unusable proposal; falling back to heuristic planning."
            )
            return proposal
    normalized["provider"] = "openai"
    normalized["provider_reason"] = (
        f"OpenAI model {model} generated proposal for {normalized['edit_mode']} mode."
    )
    return normalized


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
    edit_mode, anchor, replacement, notes = derive_edit_spec(state)
    inferred_target = _infer_target_path_from_instruction(state.get("instruction", ""), context)
    generated_batch_target = _infer_batch_target_path_from_instruction(state.get("instruction", ""))
    generated_batch_content = _infer_batch_content_from_instruction(state.get("instruction", ""))
    occurrence_selector = parse_occurrence_selector(state.get("instruction", ""))
    copy_count = None
    quantity = None
    if inferred_target and not state.get("target_path") and not context.get("file_hint"):
        notes.append(f"Derived target path `{inferred_target}` from instruction text.")
    if generated_batch_target and not state.get("target_path") and not context.get("file_hint"):
        inferred_target = generated_batch_target
        notes.append(f"Derived focused batch target `{generated_batch_target}` from instruction text.")
    if occurrence_selector:
        notes.append(f"Derived occurrence selector `{occurrence_selector}` from instruction text.")
    if edit_mode == "copy_file" and replacement is not None:
        try:
            copy_count = max(1, int(str(replacement)))
            replacement = None
            notes.append(f"Derived copy count `{copy_count}` from instruction text.")
        except ValueError:
            pass
    if edit_mode == "create_files" and replacement is not None:
        try:
            quantity = max(1, int(str(replacement)))
            replacement = generated_batch_content if generated_batch_content is not None else ""
            notes.append(f"Derived quantity `{quantity}` from instruction text.")
        except ValueError:
            pass

    resolved_target_hint = _resolve_requested_target_hint(state, context, inferred_target)

    target_path, target_path_source = resolve_target_path(
        resolved_target_hint,
        context,
        edit_mode,
        session_id=state.get("session_id"),
        instruction=state.get("instruction"),
    )
    if target_path_source == "file_hint":
        notes.append("Using file_hint as target path.")
    elif target_path_source == "managed_workspace":
        notes.append(f"Allocated managed workspace target path at {target_path}.")

    proposal = {
        "target_path": target_path,
        "target_path_source": target_path_source,
        "anchor": anchor,
        "replacement": replacement,
        "quantity": quantity,
        "copy_count": copy_count,
        "edit_mode": edit_mode,
        "occurrence_selector": occurrence_selector,
        "provider": "heuristic",
        "provider_reason": "; ".join(notes) if notes else "No heuristic edit proposal could be derived.",
    }
    return attach_validation(proposal)


def _infer_target_path_from_instruction(instruction: str, context: dict[str, Any]) -> str | None:
    if context.get("function_name"):
        return None

    match = FILENAME_PATTERN.search(instruction or "")
    if not match:
        return None
    return match.group("name")


def _resolve_requested_target_hint(
    state: ShipyardState,
    context: dict[str, Any],
    inferred_target: str | None,
) -> str | None:
    explicit_target = state.get("target_path")
    file_hint = context.get("file_hint")

    if inferred_target and _is_stale_scratch_target(explicit_target):
        return inferred_target
    if explicit_target:
        return explicit_target

    if inferred_target and _is_stale_scratch_target(file_hint):
        return inferred_target
    if file_hint:
        return file_hint

    return inferred_target


def _is_stale_scratch_target(value: Any) -> bool:
    if not value:
        return False
    name = str(value).strip().split("/")[-1]
    return bool(re.fullmatch(r"scratch(?:-[0-9a-f]{6})?\.[A-Za-z0-9]+", name))


def _normalize_openai_proposal(parsed: dict[str, Any], state: ShipyardState) -> dict[str, Any]:
    context = state.get("context", {})
    edit_mode = parsed.get("edit_mode") or infer_edit_mode(state)
    heuristic = _heuristic_proposal(state)
    inferred_target = _infer_target_path_from_instruction(state.get("instruction", ""), context)
    explicit_target_path = _resolve_requested_target_hint(state, context, inferred_target)
    if not explicit_target_path and heuristic.get("target_path") and heuristic.get("target_path_source") in {
        "explicit_target_path",
        "sandboxed_target_path",
        "file_hint",
    }:
        explicit_target_path = heuristic.get("target_path")
    if not explicit_target_path:
        explicit_target_path = parsed.get("target_path")

    target_path, target_path_source = resolve_target_path(
        explicit_target_path,
        context,
        edit_mode,
        session_id=state.get("session_id"),
        instruction=state.get("instruction"),
    )

    anchor = parsed.get("anchor")
    replacement = parsed.get("replacement")

    # Allow the model to use a more semantic `content` field for non-anchor modes.
    if replacement is None and parsed.get("content") is not None:
        replacement = parsed.get("content")

    if edit_mode in {"write_file", "append", "prepend", "named_function", "delete_file", "copy_file", "create_files"}:
        anchor = None

    if edit_mode == "anchor" and anchor is None:
        anchor = parsed.get("search_text") or context.get("search_text")
    if edit_mode == "anchor" and replacement is None:
        replacement = parsed.get("replace_text") or context.get("replace_text")

    if (
        edit_mode == "anchor"
        and heuristic.get("edit_mode") == "rename_symbol"
        and isinstance(anchor, str)
        and isinstance(replacement, str)
        and IDENTIFIER_PATTERN.fullmatch(anchor)
        and IDENTIFIER_PATTERN.fullmatch(replacement)
    ):
        edit_mode = "rename_symbol"

    normalized = {
        "target_path": target_path,
        "target_path_source": target_path_source,
        "anchor": anchor,
        "replacement": replacement,
        "quantity": parsed.get("quantity"),
        "copy_count": parsed.get("copy_count"),
        "edit_mode": edit_mode,
        "occurrence_selector": parsed.get("occurrence_selector") or parse_occurrence_selector(state.get("instruction", "")),
    }
    if edit_mode == "copy_file" and heuristic.get("edit_mode") == "copy_file" and heuristic.get("copy_count"):
        normalized["copy_count"] = heuristic.get("copy_count")
    if edit_mode == "create_files" and heuristic.get("edit_mode") == "create_files" and heuristic.get("quantity"):
        normalized["quantity"] = heuristic.get("quantity")
        if normalized.get("replacement") is None:
            normalized["replacement"] = heuristic.get("replacement")
        if not parsed.get("target_path") and heuristic.get("target_path"):
            normalized["target_path"] = heuristic.get("target_path")
            normalized["target_path_source"] = heuristic.get("target_path_source")
    return normalized


def _has_embedded_generation_request(instruction: str) -> bool:
    text = instruction.lower()
    return "create" in text and "file" in text and "write " in text


def _infer_batch_target_path_from_instruction(instruction: str) -> str | None:
    match = re.search(r"\bfile\s+(?P<index>\d+)\b", instruction, flags=re.IGNORECASE)
    if not match:
        return None
    index = match.group("index")
    extension = ".py" if "python" in instruction.lower() else ".txt"
    return f"file{index}{extension}"


def _infer_batch_content_from_instruction(instruction: str) -> str | None:
    match = re.search(r"\bwrite\s+(?P<content>.+)$", instruction.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("content").strip().rstrip(".")
