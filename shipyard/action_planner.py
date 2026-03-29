from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from .action_plan_validation import check_inspect_first, validate_action_plan
from .actions import Action, build_action_fallback, normalize_action
from .intent_parser import split_instruction_steps
from .pathing import resolve_target_path
from .planning_hints import extract_explicit_filenames, infer_target_path_from_instruction
from .repo_context import any_explicit_files_exist, build_existing_file_context_lines, build_repo_context_lines
from .state import ShipyardState
from .planning_hints import is_stale_scratch_target
from .tools.edit_file import find_anchor_pointers


class PlanningCancelledError(RuntimeError):
    pass


def _get_primary_model(state: ShipyardState | None = None) -> str:
    """Model for core planning and code generation (quality-critical)."""
    if state:
        m = state.get("proposal_model")
        if m:
            return str(m)
    return os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def _get_nano_model() -> str:
    """Cheaper/faster model for low-stakes calls: repair, exploration, replan."""
    return os.getenv("OPENAI_NANO_MODEL", "gpt-4.1-nano")


def plan_actions(state: ShipyardState) -> dict[str, Any]:
    state = _sanitize_state_for_scaffold_planning(state)
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
    model = _get_primary_model(state)
    if not api_key:
        return _heuristic_action_plan(state)
    explicit_files = extract_explicit_filenames(state.get("instruction", ""))
    has_existing_explicit_files = any_explicit_files_exist(state.get("session_id"), state.get("instruction", ""))
    is_explicit_scaffold = len(explicit_files) > 1 and not has_existing_explicit_files

    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(model, _build_action_plan_prompt(state)),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
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
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        repaired = _repair_invalid_plan(
            state,
            model=model,
            api_key=api_key,
            validation_errors=["OpenAI action plan response was not valid JSON."],
            explicit_scaffold=is_explicit_scaffold,
        )
        if repaired is not None:
            return repaired
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

    actions = _extract_candidate_actions(parsed)
    if not actions:
        repaired = _repair_invalid_plan(
            state,
            model=model,
            api_key=api_key,
            validation_errors=["OpenAI action plan response did not include actions."],
            explicit_scaffold=is_explicit_scaffold,
        )
        if repaired is not None:
            return repaired
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
        fallback = _build_openai_action_fallback(state, action, step_instruction)
        action_payload = {
            **action,
            "instruction": step_instruction,
        }
        used_fallback_target = False
        fallback_target = fallback.get("target_path")
        raw_target = action_payload.get("target_path")
        if fallback_target and action_payload.get("edit_mode") in {"list_files", "search_files", "read_file"}:
            action_payload["target_path"] = fallback_target
            action_payload["target_path_source"] = fallback.get("target_path_source")
            used_fallback_target = True
        if fallback_target and raw_target:
            try:
                if Path(str(raw_target)).name == Path(str(fallback_target)).name:
                    action_payload["target_path"] = fallback_target
                    action_payload["target_path_source"] = fallback.get("target_path_source")
                    used_fallback_target = True
            except (TypeError, ValueError):
                pass
        if (
            fallback.get("target_path_source") in {"explicit_target_path", "sandboxed_target_path", "file_hint"}
            and not used_fallback_target
        ):
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
    normalized = _autofill_unique_anchor_pointers(normalized)
    validation_errors = validate_action_plan(state.get("instruction", ""), normalized)
    if validation_errors:
        repaired = _repair_invalid_plan(
            state,
            model=model,
            api_key=api_key,
            validation_errors=validation_errors,
            explicit_scaffold=is_explicit_scaffold,
        )
        if repaired is not None:
            return repaired
    inspect_warnings = check_inspect_first(normalized)
    return {
        "actions": normalized,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} produced action plan.",
        "is_valid": not validation_errors,
        "validation_errors": validation_errors,
        "inspect_first_warnings": inspect_warnings,
        "needs_more_batches": bool(parsed.get("needs_more_batches", False)),
    }


def _repair_invalid_plan(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    validation_errors: list[str],
    explicit_scaffold: bool,
) -> dict[str, Any] | None:
    # Determinism boundary: repair should stay model-driven. This function may route
    # to alternate prompts/repair calls, but it should not invent semantic edit intent
    # or replacement content locally.
    # Use nano model for repair — it's just fixing JSON structure, not generating code.
    nano = _get_nano_model()
    if not explicit_scaffold:
        scaffold = _plan_broad_scaffold_files(
            state,
            model=model,  # scaffold needs quality — keep primary
            api_key=api_key,
            validation_errors=validation_errors,
        )
        if scaffold is not None and scaffold.get("is_valid"):
            return scaffold

    repaired = _repair_action_plan(
        state,
        model=nano,
        api_key=api_key,
        validation_errors=validation_errors,
    )
    if repaired is not None and repaired.get("is_valid"):
        return repaired
    if repaired is not None and repaired.get("actions"):
        targeted = _repair_invalid_actions_only(
            state,
            model=nano,
            api_key=api_key,
            repaired_plan=repaired,
        )
        if targeted is not None:
            return targeted

    if explicit_scaffold:
        scaffold = _plan_explicit_scaffold_files(
            state,
            model=model,
            api_key=api_key,
            validation_errors=validation_errors,
        )
        if scaffold is not None and scaffold.get("is_valid"):
            return scaffold

    return repaired


def _extract_candidate_actions(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if not isinstance(parsed, dict):
        return []

    actions = parsed.get("actions")
    if isinstance(actions, list):
        return [item for item in actions if isinstance(item, dict)]

    plan = parsed.get("plan")
    if isinstance(plan, list):
        return [item for item in plan if isinstance(item, dict)]

    if parsed.get("edit_mode"):
        return [parsed]

    return []


def _repair_action_plan(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    validation_errors: list[str],
) -> dict[str, Any] | None:
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(
                model,
                _build_action_plan_repair_prompt(state, validation_errors),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return None

    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None

    actions = parsed.get("actions")
    if not isinstance(actions, list) or not actions:
        return None

    normalized: list[Action] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        fallback = _build_openai_action_fallback(state, action, step_instruction)
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
                provider_reason=f"OpenAI model {model} repaired the action plan.",
            )
        )

    if not normalized:
        return None

    normalized = _autofill_unique_anchor_pointers(normalized)
    repaired_errors = validate_action_plan(state.get("instruction", ""), normalized)
    return {
        "actions": normalized,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} repaired the action plan.",
        "is_valid": not repaired_errors,
        "validation_errors": repaired_errors,
    }


def _repair_invalid_actions_only(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    repaired_plan: dict[str, Any],
) -> dict[str, Any] | None:
    actions = [dict(action) for action in list(repaired_plan.get("actions", []) or []) if isinstance(action, dict)]
    invalid_indexes = [index for index, action in enumerate(actions) if not action.get("valid", True)]
    if not invalid_indexes:
        return repaired_plan

    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(
                model,
                _build_invalid_action_repair_prompt(state, actions, invalid_indexes),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return repaired_plan

    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return repaired_plan

    replacements = _extract_candidate_actions(parsed)
    if len(replacements) != len(invalid_indexes):
        return repaired_plan

    normalized_replacements: list[Action] = []
    for replacement_action in replacements:
        if not isinstance(replacement_action, dict):
            return repaired_plan
        step_instruction = str(replacement_action.get("instruction") or "").strip()
        if not step_instruction:
            return repaired_plan
        fallback = _build_openai_action_fallback(state, replacement_action, step_instruction)
        action_payload = {**replacement_action, "instruction": step_instruction}
        if fallback.get("target_path_source") in {"explicit_target_path", "sandboxed_target_path", "file_hint"}:
            action_payload.pop("target_path", None)
            action_payload.pop("target_path_source", None)
        normalized_replacements.append(
            normalize_action(
                action_payload,
                fallback=fallback,
                provider="openai",
                provider_reason=f"OpenAI model {model} repaired invalid plan actions.",
            )
        )

    for replacement_index, action_index in enumerate(invalid_indexes):
        actions[action_index] = normalized_replacements[replacement_index]

    actions = _autofill_unique_anchor_pointers(actions)
    repaired_errors = validate_action_plan(state.get("instruction", ""), actions)
    return {
        "actions": actions,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} repaired invalid plan actions.",
        "is_valid": not repaired_errors,
        "validation_errors": repaired_errors,
    }


def _autofill_unique_anchor_pointers(actions: list[Action]) -> list[Action]:
    patched: list[Action] = []
    for action in actions:
        current = dict(action)
        if current.get("edit_mode") in {"anchor", "search_and_replace", "rename_symbol"}:
            anchor_text = current.get("anchor") or current.get("pattern")
            replacement = current.get("replacement")
            if anchor_text is not None and replacement is not None and str(anchor_text) == str(replacement):
                current["edit_mode"] = "read_file"
                current["anchor"] = None
                current["pattern"] = None
                current["replacement"] = None
                current["pointers"] = None
                current["valid"] = True
                current["validation_errors"] = []
        if (
            current.get("edit_mode") == "anchor"
            and current.get("anchor")
            and not current.get("pointers")
            and current.get("target_path")
        ):
            try:
                target_path = Path(str(current["target_path"]))
                if target_path.exists() and target_path.is_file():
                    content = target_path.read_text(encoding="utf-8")
                    pointers = find_anchor_pointers(content, str(current["anchor"]))
                    if len(pointers) == 1:
                        current["pointers"] = pointers
                        current["valid"] = True
                        current["validation_errors"] = []
                    else:
                        replacement = current.get("replacement")
                        if (
                            isinstance(replacement, str)
                            and replacement
                            and str(current["anchor"]) not in content
                            and content.count(replacement) >= 1
                        ):
                            current["edit_mode"] = "read_file"
                            current["anchor"] = None
                            current["pointers"] = None
                            current["valid"] = True
                            current["validation_errors"] = []
            except (OSError, UnicodeDecodeError, ValueError):
                pass
        patched.append(current)
    return patched


def _plan_explicit_scaffold_files(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    validation_errors: list[str],
) -> dict[str, Any] | None:
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_structured_request(
                model,
                _build_scaffold_files_prompt(state, validation_errors),
                name="scaffold_files",
                schema=_scaffold_files_schema(),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return None

    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None

    files = _extract_candidate_scaffold_files(
        parsed,
        extract_explicit_filenames(state.get("instruction", "")),
    )
    if not files:
        return None

    action = normalize_action(
        {
            "instruction": state.get("instruction", ""),
            "edit_mode": "scaffold_files",
            "files": files,
        },
        fallback={},
        provider="openai",
        provider_reason=f"OpenAI model {model} generated scaffold files.",
    )
    scaffold_errors = validate_action_plan(state.get("instruction", ""), [action])
    return {
        "actions": [action],
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} generated scaffold files.",
        "is_valid": not scaffold_errors,
        "validation_errors": scaffold_errors,
    }


def _plan_broad_scaffold_files(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    validation_errors: list[str],
) -> dict[str, Any] | None:
    if extract_explicit_filenames(state.get("instruction", "")):
        return None
    if _session_workspace_has_files(state):
        return None

    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_structured_request(
                model,
                _build_broad_scaffold_prompt(state, validation_errors),
                name="broad_scaffold_files",
                schema=_scaffold_files_schema(),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return None

    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None

    files = _extract_candidate_scaffold_files(parsed, [])
    if not files:
        return None

    action = normalize_action(
        {
            "instruction": state.get("instruction", ""),
            "edit_mode": "scaffold_files",
            "files": files,
        },
        fallback={},
        provider="openai",
        provider_reason=f"OpenAI model {model} generated a broad scaffold plan.",
    )
    scaffold_errors = validate_action_plan(state.get("instruction", ""), [action])
    return {
        "actions": [action],
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} generated a broad scaffold plan.",
        "is_valid": not scaffold_errors,
        "validation_errors": scaffold_errors,
    }


def _session_workspace_has_files(state: ShipyardState) -> bool:
    context = dict(state.get("context", {}) or {})
    workspace = Path(
        str(
            resolve_target_path(
                ".",
                context,
                "list_files",
                session_id=state.get("session_id"),
                instruction=state.get("instruction"),
            )[0]
            or "."
        )
    )
    if not workspace.exists() or not workspace.is_dir():
        return False
    return any(path.is_file() for path in workspace.rglob("*"))


def _extract_candidate_scaffold_files(parsed: Any, explicit_files: list[str]) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []

    files = parsed.get("files")
    if isinstance(files, list):
        return [item for item in files if isinstance(item, dict)]

    scaffold = parsed.get("scaffold")
    if isinstance(scaffold, list):
        return [item for item in scaffold if isinstance(item, dict)]

    if explicit_files:
        collected: list[dict[str, Any]] = []
        for name in explicit_files:
            content = parsed.get(name)
            if content is None:
                return []
            collected.append({"path": name, "content": content})
        return collected

    return []


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


def _build_openai_action_fallback(
    state: ShipyardState,
    action: dict[str, Any],
    step_instruction: str,
) -> dict[str, Any]:
    context = dict(state.get("context", {}) or {})
    edit_mode = str(action.get("edit_mode") or state.get("edit_mode") or "anchor").strip() or "anchor"
    explicit_files = extract_explicit_filenames(step_instruction) or extract_explicit_filenames(
        state.get("instruction", "")
    )
    requested_target = action.get("target_path") or state.get("target_path")
    if edit_mode in {"list_files", "search_files"} and not requested_target:
        requested_target = "."
    if len(explicit_files) == 1 and (not requested_target or is_stale_scratch_target(requested_target)):
        requested_target = explicit_files[0]
    elif not requested_target:
        inferred_target = infer_target_path_from_instruction(step_instruction, context)
        if inferred_target:
            requested_target = inferred_target

    target_path = None
    target_path_source = None
    if requested_target:
        target_path, target_path_source = resolve_target_path(
            str(requested_target) if requested_target else None,
            context,
            edit_mode,
            session_id=state.get("session_id"),
            instruction=step_instruction,
        )
        if (
            edit_mode in {"list_files", "search_files"}
            and target_path
            and not Path(str(target_path)).exists()
        ):
            target_path, target_path_source = resolve_target_path(
                ".",
                context,
                edit_mode,
                session_id=state.get("session_id"),
                instruction=step_instruction,
            )

    return {
        "instruction": step_instruction,
        "target_path": target_path,
        "target_path_source": target_path_source,
        "edit_mode": edit_mode,
        "anchor": None,
        "replacement": None,
        "quantity": action.get("quantity"),
        "copy_count": action.get("copy_count"),
        "files": action.get("files"),
        "pattern": action.get("pattern"),
        "command": action.get("command"),
        "occurrence_selector": action.get("occurrence_selector"),
        "full_file_rewrite": bool(action.get("full_file_rewrite")),
    }


def _openai_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _log_openai_call(payload: dict[str, Any], response: httpx.Response, elapsed: float) -> None:
    """Print a one-line summary of an OpenAI API call to server stdout."""
    model = payload.get("model", "?")
    try:
        body = response.json()
        usage = body.get("usage") or {}
        input_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or "?"
        output_tok = usage.get("output_tokens") or usage.get("completion_tokens") or "?"
        total_tok = usage.get("total_tokens") or (
            (input_tok + output_tok) if isinstance(input_tok, int) and isinstance(output_tok, int) else "?"
        )
    except Exception:
        input_tok = output_tok = total_tok = "?"
    print(
        f"[openai] {model}  in={input_tok} out={output_tok} total={total_tok}  {elapsed:.2f}s",
        flush=True,
    )


def _post_openai_with_retry(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    cancel_check: Any = None,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(3):
        if callable(cancel_check) and cancel_check():
            raise PlanningCancelledError("Run cancelled during planning.")
        try:
            t0 = time.monotonic()
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers=_openai_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                _log_openai_call(payload, response, time.monotonic() - t0)
                return response
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code is None or status_code < 500 or status_code >= 600 or attempt == 2:
                raise
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                raise
        jitter = random.uniform(0, 0.5)
        sleep_deadline = time.monotonic() + (0.5 * (attempt + 1)) + jitter
        while time.monotonic() < sleep_deadline:
            if callable(cancel_check) and cancel_check():
                raise PlanningCancelledError("Run cancelled during planning.")
            time.sleep(0.1)
    if last_exc:
        raise last_exc
    raise RuntimeError("OpenAI request failed without a response.")


def _openai_action_plan_request(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "text": {"format": {"type": "json_object"}},
    }


def _openai_structured_request(model: str, prompt: str, *, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        },
    }


def _build_action_plan_prompt(state: ShipyardState) -> str:
    instruction = state.get("instruction", "").strip()
    explicit_files = extract_explicit_filenames(instruction)
    has_existing_explicit_files = any_explicit_files_exist(state.get("session_id"), instruction)
    is_explicit_scaffold = len(explicit_files) > 1 and not has_existing_explicit_files
    batch_size = int(state.get("plan_batch_size") or 0)
    lines = [
        "Return only JSON.",
        "Plan the user request as an ordered list of actions.",
        "Use the key actions with an array of objects.",
        "Each action should include: id, instruction, action_class, edit_mode, target_path, and any mode-specific fields.",
        "Optional fields: anchor, replacement, quantity, copy_count, files, pattern, command, source_path, destination_path, paths, depends_on, inputs_from, timeout_seconds, max_retries, full_file_rewrite, context_files, intent, edit_scope, expected_existing_state, recovery_strategy.",
        "Allowed action_class values: inspect, mutate, verify.",
        "",
        "## DISCOVER → INSPECT → EDIT → VERIFY",
        "Before editing anything, DISCOVER what exists. Before coding, understand the codebase.",
        "1. DISCOVER: If the instruction is high-level (feature name, bug description, refactor goal), first search for related docs:",
        "   - search_files with patterns like the feature name, PRD, spec, plan, README",
        "   - list_files on docs/, plans/, specs/, or similar directories if they exist",
        "   - read_file on any matching .md, .prd.json, .txt files to understand requirements",
        "2. INSPECT: read_file on the code files you need to change. run_command with grep to find relevant functions/components.",
        "3. EDIT: Apply changes based on what you discovered and inspected.",
        "4. VERIFY: run_command to check syntax, run tests, or validate the result.",
        "For simple direct instructions ('add X to Y'), skip DISCOVER and go straight to INSPECT.",
        "",
        "## EDITING APPROACH",
        "Use whatever edit mode gets the job done. Available modes:",
        "write_file, search_and_replace, append, prepend, anchor, rename_symbol, delete_file, copy_file, scaffold_files, list_files, read_file, read_many_files, search_files, run_command, verify_command, create_directory, move_file, rename_file, run_tests, inspect_imports.",
        "When using write_file on an existing file, the replacement must contain the COMPLETE file — all existing content plus your changes.",
        "For creating multiple new files: use scaffold_files with files=[{path, content}, ...]. Include the ACTUAL file content in each entry — not a description of what should go there.",
        "For creating a single new file: use write_file with target_path and replacement containing the full file content.",
        "CRITICAL: Every mutate action that creates or edits a file MUST include the actual content in replacement (for write_file) or files[].content (for scaffold_files). Do NOT plan abstract 'create this file' steps without content.",
        "",
        "Use id for stable step references. depends_on and inputs_from should reference earlier ids.",
        "If the user explicitly names a file, preserve that exact file as target_path.",
        "For scaffolds, use scaffold_files with files=[{path, content}, ...].",
        "Use list_files, read_file, read_many_files, search_files for inspect steps.",
        "Use run_command, verify_command, or run_tests for verify steps.",
        "Mutate actions should declare intent and edit_scope when applicable.",
        "If the file does not exist yet, write_file is fine and full_file_rewrite should be false.",
        "For code-generation requests, put concrete code in replacement.",
        "Do not leave replacement empty or use placeholder text for code-generation requests.",
        f"Instruction: {instruction}",
        "Lightweight repository context:",
    ]
    session_journal = state.get("session_journal") or []
    if session_journal:
        lines.append("Recent session history (most recent last):")
        for entry in session_journal:
            ts = entry.get("timestamp", "")
            instr = entry.get("instruction", "")
            status = entry.get("status", "")
            changed = ", ".join(entry.get("changed_files") or [])
            error = entry.get("error", "")
            summary = f"  [{ts}] status={status} instruction={instr!r}"
            if changed:
                summary += f" changed={changed}"
            if error:
                summary += f" error={error!r}"
            lines.append(summary)
    if batch_size:
        lines.append(f"Plan only the first {batch_size} steps needed to begin this task.")
        lines.append("If the task will require more steps after these complete, include needs_more_batches: true in your response.")
        lines.append("The runtime will call you again with actual file state and tool outputs after each batch.")
    if explicit_files:
        lines.append("Explicit files mentioned by the user:")
        lines.extend(f"- {name}" for name in explicit_files)
    if is_explicit_scaffold:
        lines.append("This request is an explicit multi-file scaffold.")
        lines.append("Preferred output: one scaffold_files action with files entries for every named file.")
        lines.append("Every listed file must appear exactly once in files[].path unless the request clearly requires separate ordered actions.")
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))

    # Broad context: key file contents surfaced before planning so the LLM can ground
    # anchors/pointers in reality rather than generating them blind.
    broad_context = state.get("broad_context") or {}
    sampled_files = broad_context.get("sampled_files") or {}
    if sampled_files:
        lines.append("Sampled repository files (for orientation — do not assume these are complete):")
        for file_path, file_content in list(sampled_files.items())[:12]:
            lines.append(f"--- {file_path} ---")
            lines.append(str(file_content)[:8000])
    broad_file_tree = broad_context.get("file_tree") or []
    if broad_file_tree:
        lines.append("Full file tree: " + ", ".join(broad_file_tree[:500]))
    discovered_docs = broad_context.get("discovered_docs") or []
    if discovered_docs:
        lines.append("Planning/spec docs found in repo (read these if the instruction is high-level):")
        lines.extend(f"  - {doc}" for doc in discovered_docs[:25])
    project_stack = broad_context.get("project_stack") or {}
    if project_stack:
        stack_parts = []
        if project_stack.get("language"):
            stack_parts.append(f"language={project_stack['language']}")
        if project_stack.get("framework"):
            stack_parts.append(f"framework={project_stack['framework']}")
        if project_stack.get("package_manager"):
            stack_parts.append(f"package_manager={project_stack['package_manager']}")
        if stack_parts:
            lines.append("Project stack: " + ", ".join(stack_parts))
            pass  # stack info is sufficient context
    git_status = broad_context.get("git_status") or {}
    if git_status.get("branch"):
        lines.append(f"Git branch: {git_status['branch']}")
        if git_status.get("status_lines"):
            lines.append("Uncommitted changes: " + ", ".join(git_status["status_lines"][:20]))
        if git_status.get("diff_stat"):
            lines.append("Diff stat vs HEAD:\n" + git_status["diff_stat"])
        if git_status.get("diff_preview"):
            lines.append("Diff preview (first 3000 chars):\n" + git_status["diff_preview"])

    existing_file_context = build_existing_file_context_lines(state.get("session_id"), instruction)
    if existing_file_context:
        lines.append("Existing file context:")
        for line in existing_file_context:
            lines.append(line)
        pass  # existing file context is enough for the LLM to ground on
    lines.append("Example workflow: inspect → edit → verify:")
    lines.append('{"actions":[{"id":"s1","instruction":"Read files","action_class":"inspect","edit_mode":"read_file","target_path":"app.js"},{"id":"s2","instruction":"Update app.js","action_class":"mutate","edit_mode":"write_file","target_path":"app.js","depends_on":["s1"]},{"id":"s3","instruction":"Verify","action_class":"verify","edit_mode":"run_command","command":"node --check app.js","depends_on":["s2"]}]}')
    if is_explicit_scaffold:
        lines.append("For this request, do not return a single write_file for only one file.")

    # Dynamic tool registry — expose available synthesised tools.
    try:
        from .tool_registry import get_registry
        dynamic_tools = get_registry().list_tools()
    except Exception:  # noqa: BLE001
        dynamic_tools = []
    if dynamic_tools:
        lines.append("Available dynamic tools (use invoke_tool to call them):")
        for tool in dynamic_tools:
            desc = tool.get("description") or "(no description)"
            lines.append(f"  - {tool['name']}: {desc}")

    return "\n".join(lines)


def _build_action_plan_repair_prompt(state: ShipyardState, validation_errors: list[str]) -> str:
    base_prompt = _build_action_plan_prompt(state)
    explicit_files = extract_explicit_filenames(state.get("instruction", ""))
    lines = [
        base_prompt,
        "The previous action plan was incomplete.",
        "Repair it so the final actions cover the full request.",
        "Validation errors:",
    ]
    lines.extend(f"- {error}" for error in validation_errors)
    if len(explicit_files) > 1:
        lines.append("Return a corrected scaffold_files action unless the request clearly requires multiple ordered actions.")
        lines.append("The corrected files array must cover every explicitly named file.")
    lines.append("Return only corrected JSON.")
    return "\n".join(lines)


def _build_invalid_action_repair_prompt(
    state: ShipyardState,
    actions: list[dict[str, Any]],
    invalid_indexes: list[int],
) -> str:
    lines = [
        _build_action_plan_prompt(state),
        "The previous repair still contained invalid actions.",
        "Return only JSON with the key actions.",
        "Return replacements only for the invalid actions, in the same order they are listed below.",
        "Keep the same intent, but replace placeholders with actual code or concrete content.",
        "For mutate steps, include explicit intent, edit_scope, expected_existing_state, and recovery_strategy.",
        "Do not return placeholder strings such as TODO_* or *_UPDATED.",
        "Invalid actions to replace:",
    ]
    for offset, action_index in enumerate(invalid_indexes, start=1):
        action = actions[action_index]
        lines.append(f"- Invalid action {offset}:")
        lines.append(json.dumps(action, ensure_ascii=True))
    lines.append("Return only corrected JSON.")
    return "\n".join(lines)


def _build_scaffold_files_prompt(state: ShipyardState, validation_errors: list[str]) -> str:
    instruction = state.get("instruction", "").strip()
    explicit_files = extract_explicit_filenames(instruction)
    lines = [
        "Return only JSON.",
        "Create file contents for an explicit tiny repo scaffold.",
        "Use the key files with an array of objects.",
        "Each file object must have: path, content.",
        "Cover every explicitly named file exactly once.",
        "Do not return actions. Return only files.",
        f"Instruction: {instruction}",
        "Explicit files mentioned by the user:",
    ]
    lines.extend(f"- {name}" for name in explicit_files)
    lines.append("Lightweight repository context:")
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))
    lines.append("Previous planner validation errors:")
    lines.extend(f"- {error}" for error in validation_errors)
    lines.append(
        'Example output: {"files":[{"path":"main.py","content":"print(\\"hello\\")\\n"},{"path":"config.json","content":"{\\"unit\\": \\"ms\\"}\\n"}]}'
    )
    lines.append(
        'Alternative acceptable output: {"main.py":"print(\\"hello\\")\\n","config.json":"{\\"unit\\": \\"ms\\"}\\n"}'
    )
    return "\n".join(lines)


def _build_broad_scaffold_prompt(state: ShipyardState, validation_errors: list[str]) -> str:
    instruction = state.get("instruction", "").strip()
    lines = [
        "Return only JSON.",
        "Convert this broad generation request into a minimal concrete file scaffold.",
        "Use the key files with an array of objects.",
        "Each file object must have: path, content.",
        "Return only files, not actions.",
        "Choose the smallest realistic set of files needed to satisfy the request.",
        "If the request is for a browser app, prefer a minimal scaffold like index.html, styles.css, and app.js unless a different set is clearly better.",
        "Do not leave file contents blank.",
        f"Instruction: {instruction}",
        "Lightweight repository context:",
    ]
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))
    lines.append("Previous planner validation errors:")
    lines.extend(f"- {error}" for error in validation_errors)
    lines.append(
        'Example output: {"files":[{"path":"index.html","content":"<!doctype html>\\n<html>\\n  <head>\\n    <meta charset=\\"utf-8\\" />\\n    <title>Todo App</title>\\n    <link rel=\\"stylesheet\\" href=\\"styles.css\\" />\\n  </head>\\n  <body>\\n    <div id=\\"app\\"></div>\\n    <script src=\\"app.js\\"></script>\\n  </body>\\n</html>\\n"},{"path":"styles.css","content":"body { font-family: sans-serif; }\\n"},{"path":"app.js","content":"document.getElementById(\\"app\\").textContent = \\"Todo app\\";\\n"}]}'
    )
    return "\n".join(lines)


def _scaffold_files_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            }
        },
        "required": ["files"],
    }


def replan_remaining_actions(
    state: ShipyardState,
    *,
    completed_steps: list[dict[str, Any]],
    failed_step: dict[str, Any],
    remaining_actions: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    After a step failure, ask the LLM what to do with the remaining plan.
    Returns revised remaining actions, empty list (skip and continue), or None (hard stop).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not remaining_actions:
        return None
    model = _get_nano_model()  # replan is structural, not code-generating
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(
                model,
                _build_replan_prompt(state, completed_steps, failed_step, remaining_actions),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return None
    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None
    decision = str(parsed.get("action") or "stop").strip().lower()
    if decision == "stop":
        return None
    if decision == "continue":
        return list(remaining_actions)
    if decision == "modify":
        raw_revised = parsed.get("revised_remaining") or []
        if not isinstance(raw_revised, list) or not raw_revised:
            return list(remaining_actions)
        normalized: list[Action] = []
        for action in raw_revised:
            if not isinstance(action, dict):
                continue
            step_instruction = str(action.get("instruction") or "").strip()
            if not step_instruction:
                continue
            fallback = _build_openai_action_fallback(state, action, step_instruction)
            normalized.append(
                normalize_action(
                    {**action, "instruction": step_instruction},
                    fallback=fallback,
                    provider="openai",
                    provider_reason="Adaptive replan after step failure.",
                )
            )
        return normalized or list(remaining_actions)
    return None


def plan_next_batch(
    state: ShipyardState,
    *,
    completed_steps: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
    changed_files: list[str],
    batch_size: int = 4,
) -> dict[str, Any] | None:
    """
    After a batch completes, plan the next batch of steps with actual execution context.
    Returns an action plan dict (same shape as plan_actions), or None if done.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = _get_primary_model(state)
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(
                model,
                _build_next_batch_prompt(state, completed_steps, tool_outputs, changed_files, batch_size),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return None
    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None
    actions = _extract_candidate_actions(parsed)
    if not actions:
        return None
    normalized: list[Action] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        fallback = _build_openai_action_fallback(state, action, step_instruction)
        normalized.append(
            normalize_action(
                {**action, "instruction": step_instruction},
                fallback=fallback,
                provider="openai",
                provider_reason="Next batch plan.",
            )
        )
    needs_more = bool(parsed.get("needs_more_batches", False))
    return {
        "actions": normalized,
        "needs_more_batches": needs_more,
        "provider": "openai",
        "is_valid": bool(normalized),
    }


def plan_edits_for_matches(
    state: ShipyardState,
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Given search matches (file, line, text), ask the LLM to generate one edit action per file.
    Used by search_then_edit to expand into per-file edits.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not matches:
        return []
    model = _get_primary_model(state)
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(
                model,
                _build_search_then_edit_prompt(state, matches),
            ),
            timeout=45.0,
            cancel_check=state.get("cancel_check"),
        )
    except Exception:
        return []
    body = response.json()
    output_text = _extract_response_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return []
    actions = _extract_candidate_actions(parsed)
    normalized: list[Action] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        step_instruction = str(action.get("instruction") or "").strip()
        if not step_instruction:
            continue
        fallback = _build_openai_action_fallback(state, action, step_instruction)
        normalized.append(
            normalize_action(
                {**action, "instruction": step_instruction},
                fallback=fallback,
                provider="openai",
                provider_reason="search_then_edit expansion.",
            )
        )
    return normalized


def _build_replan_prompt(
    state: ShipyardState,
    completed_steps: list[dict[str, Any]],
    failed_step: dict[str, Any],
    remaining_actions: list[dict[str, Any]],
) -> str:
    instruction = state.get("instruction", "").strip()
    lines = [
        "A step in the action plan has failed. Decide what to do with the remaining steps.",
        f"Original task: {instruction}",
        "",
        "Completed steps:",
    ]
    for step in completed_steps:
        status = step.get("status", "unknown")
        lines.append(f"  [{step.get('id', '?')}] {step.get('instruction', '')[:80]} → {status}")
    lines.append("")
    lines.append("Failed step:")
    lines.append(f"  [{failed_step.get('id', '?')}] {failed_step.get('instruction', '')[:80]}")
    lines.append(f"  edit_mode: {failed_step.get('edit_mode', 'unknown')}")
    lines.append(f"  error: {str(failed_step.get('error') or failed_step.get('status', 'unknown'))[:200]}")
    lines.append("")
    lines.append("Remaining planned steps:")
    for action in remaining_actions:
        lines.append(f"  [{action.get('id', '?')}] {str(action.get('instruction') or '')[:80]} ({action.get('edit_mode', '?')})")
    lines.append("")
    lines.append("Decide what to do. Return JSON with:")
    lines.append('  "action": "continue" | "modify" | "stop"')
    lines.append('  "revised_remaining": [...full action objects if action=modify...]')
    lines.append('  "reason": "brief explanation"')
    lines.append("")
    lines.append("continue = proceed with the remaining plan unchanged (skip the failed step)")
    lines.append("modify = replace remaining steps with revised_remaining (use same action schema as planning)")
    lines.append("stop = abort the run entirely")
    lines.append(f"Allowed edit_mode values: write_file, search_and_replace, anchor, append, prepend, delete_file, scaffold_files, rename_symbol, list_files, read_file, read_many_files, search_files, run_command, verify_command, run_tests.")
    return "\n".join(lines)


def _build_next_batch_prompt(
    state: ShipyardState,
    completed_steps: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
    changed_files: list[str],
    batch_size: int,
) -> str:
    instruction = state.get("instruction", "").strip()
    lines = [
        f"Continue planning. The following steps were completed toward this task: {instruction}",
        "",
        "Completed steps:",
    ]
    for step in completed_steps:
        status = step.get("status", "unknown")
        lines.append(f"  [{step.get('id', '?')}] {step.get('instruction', '')[:80]} → {status}")
    if changed_files:
        lines.append("")
        lines.append("Changed files so far: " + ", ".join(changed_files[:20]))
    if tool_outputs:
        lines.append("")
        lines.append("Recent tool outputs (file reads, searches):")
        for output in tool_outputs[-4:]:
            tool = output.get("tool", "?")
            if tool == "read_file":
                path = output.get("target_path", "?")
                content = str(output.get("content") or "")[:500]
                lines.append(f"  read_file {path}:\n{content}")
            elif tool in {"list_files", "search_files"}:
                lines.append(f"  {tool}: {str(output)[:300]}")
    lines.append("")
    lines.append(f"Plan the next {batch_size} steps to continue the task.")
    lines.append("Return JSON with 'actions' array using the same action schema.")
    lines.append("If more steps will be needed after these, include needs_more_batches: true.")
    lines.append("If the task is complete or no more steps are needed, return an empty actions array or needs_more_batches: false.")
    lines.append(f"Allowed edit_mode values: write_file, search_and_replace, anchor, append, prepend, delete_file, scaffold_files, rename_symbol, list_files, read_file, read_many_files, search_files, run_command, verify_command, run_tests.")
    lines.append("Repository context:")
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))
    return "\n".join(lines)


def _build_search_then_edit_prompt(
    state: ShipyardState,
    matches: list[dict[str, Any]],
) -> str:
    instruction = state.get("instruction", "").strip()
    lines = [
        "Generate edit actions for search matches.",
        f"Task: {instruction}",
        "",
        "Search matches (file, line, text):",
    ]
    files_seen: dict[str, list[dict]] = {}
    for match in matches[:50]:
        path = str(match.get("path", "?"))
        files_seen.setdefault(path, []).append(match)
    for path, file_matches in files_seen.items():
        lines.append(f"  {path}:")
        for m in file_matches[:5]:
            lines.append(f"    line {m.get('line', '?')}: {str(m.get('text', ''))[:100]}")
    lines.append("")
    lines.append("Return JSON with an 'actions' array — one action per file that needs editing.")
    lines.append("Each action: id, instruction, action_class, edit_mode, target_path, anchor, replacement, pointers.")
    lines.append("Prefer search_and_replace or anchor edits. Do not invent changes not supported by the matches.")
    return "\n".join(lines)


def _sanitize_state_for_scaffold_planning(state: ShipyardState) -> ShipyardState:
    instruction = state.get("instruction", "")
    explicit_files = extract_explicit_filenames(instruction)
    if len(explicit_files) <= 1:
        return state

    sanitized: ShipyardState = dict(state)
    context = dict(state.get("context", {}) or {})
    target_path = sanitized.get("target_path")
    if target_path and is_stale_scratch_target(target_path):
        sanitized["target_path"] = None
    file_hint = context.get("file_hint")
    if file_hint and is_stale_scratch_target(file_hint):
        context.pop("file_hint", None)
    sanitized["context"] = context
    return sanitized


def request_exploration_files(state: ShipyardState) -> list[str]:
    """
    Ask the LLM which files it needs to read before planning.
    Returns a list of existing file paths (up to 6).
    Falls back to [] if the API is unavailable or the response is malformed.
    Skip when proposal_mode is heuristic or OPENAI_API_KEY is absent.
    """
    if (state.get("proposal_mode") or "").strip().lower() == "heuristic":
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    prompt = _build_exploration_prompt(state)
    model = _get_nano_model()  # exploration is just picking filenames
    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(model, prompt),
            timeout=30.0,
            cancel_check=state.get("cancel_check"),
        )
        body = response.json()
        text = _extract_response_text(body)
        parsed = json.loads(text)
        raw_paths = parsed.get("files") or []
        validated: list[str] = []
        for p in raw_paths[:6]:
            path = Path(str(p))
            if not path.is_absolute():
                broad_root = (state.get("broad_context") or {}).get("root") or "."
                path = Path(broad_root) / path
            if path.exists() and path.is_file():
                validated.append(str(path.resolve()))
        return validated
    except Exception:
        return []


def _build_exploration_prompt(state: ShipyardState) -> str:
    instruction = state.get("instruction", "").strip()
    broad_context = state.get("broad_context") or {}
    file_tree = broad_context.get("file_tree") or []
    sampled_files = broad_context.get("sampled_files") or {}

    lines = [
        "You are about to plan code changes. Before planning, identify which files you need to read.",
        f"Task: {instruction}",
        "",
        "Repository file tree:",
    ]
    lines.extend(f"  {p}" for p in file_tree[:100])
    if sampled_files:
        lines.append("")
        lines.append("Already sampled files:")
        for path in sampled_files:
            lines.append(f"  {path}")
    lines.append("")
    lines.append("Return ONLY JSON with a 'files' key listing up to 6 file paths you need to read before planning.")
    lines.append('Example: {"files": ["src/main.py", "src/utils.py"]}')
    lines.append("List only files from the tree above. Do not invent paths.")
    return "\n".join(lines)
