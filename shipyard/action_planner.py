from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .action_plan_validation import validate_action_plan
from .actions import Action, build_action_fallback, normalize_action
from .intent_parser import split_instruction_steps
from .pathing import resolve_target_path
from .planning_hints import extract_explicit_filenames, infer_target_path_from_instruction
from .repo_context import any_explicit_files_exist, build_existing_file_context_lines, build_repo_context_lines
from .state import ShipyardState
from .planning_hints import is_stale_scratch_target
from .tools.edit_file import find_anchor_pointers


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
    model = state.get("proposal_model") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    if not api_key:
        return _heuristic_action_plan(state)
    explicit_files = extract_explicit_filenames(state.get("instruction", ""))
    has_existing_explicit_files = any_explicit_files_exist(state.get("session_id"), state.get("instruction", ""))
    is_explicit_scaffold = len(explicit_files) > 1 and not has_existing_explicit_files

    try:
        response = _post_openai_with_retry(
            api_key=api_key,
            payload=_openai_action_plan_request(model, _build_action_plan_prompt(state)),
            timeout=20.0,
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
    if validation_errors and explicit_files:
        repaired = _repair_invalid_plan(
            state,
            model=model,
            api_key=api_key,
            validation_errors=validation_errors,
            explicit_scaffold=is_explicit_scaffold,
        )
        if repaired is not None:
            return repaired
    return {
        "actions": normalized,
        "provider": "openai",
        "provider_reason": f"OpenAI model {model} produced action plan.",
        "is_valid": not validation_errors,
        "validation_errors": validation_errors,
    }


def _repair_invalid_plan(
    state: ShipyardState,
    *,
    model: str,
    api_key: str,
    validation_errors: list[str],
    explicit_scaffold: bool,
) -> dict[str, Any] | None:
    repaired = _repair_action_plan(
        state,
        model=model,
        api_key=api_key,
        validation_errors=validation_errors,
    )
    if repaired is not None and repaired.get("is_valid"):
        return repaired

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
            timeout=20.0,
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


def _autofill_unique_anchor_pointers(actions: list[Action]) -> list[Action]:
    patched: list[Action] = []
    for action in actions:
        current = dict(action)
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
            timeout=30.0,
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


def _post_openai_with_retry(*, api_key: str, payload: dict[str, Any], timeout: float) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers=_openai_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
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
        time.sleep(0.5 * (attempt + 1))
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
    lines = [
        "Return only JSON.",
        "Plan the user request as an ordered list of actions.",
        "Use the key actions with an array of objects.",
        "Each action should include: id, instruction, action_class, edit_mode, target_path, anchor, replacement, quantity, copy_count, files, pattern, command, pointers, source_path, destination_path, paths, depends_on, inputs_from, timeout_seconds, max_retries, full_file_rewrite.",
        "Allowed action_class values: inspect, mutate, verify.",
        "Allowed edit_mode values: anchor, named_function, write_file, append, prepend, delete_file, copy_file, create_files, scaffold_files, rename_symbol, list_files, read_file, read_many_files, search_files, run_command, verify_command, create_directory, move_file, rename_file, search_and_replace, run_tests, inspect_imports.",
        "Use id for stable step references. depends_on and inputs_from should reference earlier ids when a later step relies on them.",
        "Do not omit separate steps just because they appear in one sentence.",
        "If the user explicitly names a file, preserve that exact file as target_path instead of inventing a scratch file.",
        "If the user explicitly lists multiple files for a tiny repo or project scaffold, return concrete actions that cover every listed file.",
        "Do not collapse an explicit file list into a single generic file action.",
        "If the instruction names multiple files, ensure the final plan covers each named file with a concrete action.",
        "Prefer one explicit action per file when scaffolding a tiny repo with named files.",
        "For tiny repo scaffolds, you may use scaffold_files with files=[{path, content}, ...] when that is more concise and still covers every named file.",
        "Use list_files, read_file, read_many_files, search_files, and inspect_imports for inspect steps.",
        "Use run_command, verify_command, or run_tests for verify steps.",
        "For larger multi-step repo workflows, prefer plans that inspect files first, then edit, then verify with run_command.",
        "If the user asks to add or insert code into a named file, prefer append and preserve the existing file content.",
        "If a named file already exists, prefer anchor, rename_symbol, append, prepend, or named_function over write_file.",
        "Use write_file on an existing file only when you truly intend a full file rewrite, and set full_file_rewrite to true.",
        "For localized edits on existing files, include either exact pointers or a concrete anchor, plus replacement.",
        "Pointers must be a list of {start, end} spans into the current file contents.",
        "Pointer offsets must be 0-based character offsets into the exact current file string, with end exclusive.",
        "If the user asks to change a literal everywhere in one file, prefer anchor with exact pointers covering each occurrence.",
        "If the user asks to remove repeated sections except one, read the whole target file first and anchor a full contiguous block that removes all unwanted repetitions, not just the first nearby occurrence.",
        "For reduction requests like 'keep only one run', ensure the replacement removes every extra repeated call requested by the user.",
        "If the file does not exist yet, write_file is fine and full_file_rewrite should be false.",
        "For code-generation requests, put concrete code in replacement.",
        "Do not leave replacement empty for content-generation requests.",
        "Do not return placeholder text as replacement for code-generation requests.",
        f"Instruction: {instruction}",
        "Lightweight repository context:",
    ]
    if explicit_files:
        lines.append("Explicit files mentioned by the user:")
        lines.extend(f"- {name}" for name in explicit_files)
    if is_explicit_scaffold:
        lines.append("This request is an explicit multi-file scaffold.")
        lines.append("Preferred output: one scaffold_files action with files entries for every named file.")
        lines.append("Every listed file must appear exactly once in files[].path unless the request clearly requires separate ordered actions.")
    lines.extend(f"- {line}" for line in build_repo_context_lines(state.get("session_id"), state.get("target_path")))
    existing_file_context = build_existing_file_context_lines(state.get("session_id"), instruction)
    if existing_file_context:
        lines.append("Existing file context:")
        for line in existing_file_context:
            if line.startswith("Existing file: "):
                lines.append(line)
            else:
                lines.append(line)
        lines.append("Example localized edit action:")
        lines.append(
            '{"id":"step-2","instruction":"Update formatter.py formatting","action_class":"mutate","edit_mode":"anchor","target_path":"formatter.py","anchor":"return f\\"{value:.2f} {unit}\\"","replacement":"return f\\"Average latency: {value:.2f} {unit}\\"","pointers":[{"start":31,"end":59}],"depends_on":["step-1"],"inputs_from":["step-1"],"full_file_rewrite":false}'
        )
        lines.append("Example multi-pointer localized edit action:")
        lines.append(
            '{"id":"step-3","instruction":"Replace Average with Processed everywhere in formatter.py","action_class":"mutate","edit_mode":"anchor","target_path":"formatter.py","anchor":"Average","replacement":"Processed","pointers":[{"start":8,"end":15},{"start":42,"end":49}],"full_file_rewrite":false}'
        )
    lines.append("Example scaffold_files action:")
    lines.append('{"id":"step-1","instruction":"Create tiny repo files","action_class":"mutate","edit_mode":"scaffold_files","files":[{"path":"main.py","content":"print(\\"hello\\")\\n"},{"path":"config.json","content":"{\\"unit\\": \\"ms\\"}\\n"}]}')
    lines.append("Example tool actions:")
    lines.append('{"id":"step-1","instruction":"Inspect the repo files","action_class":"inspect","edit_mode":"list_files","target_path":"."}')
    lines.append('{"id":"step-2","instruction":"Read main.py","action_class":"inspect","edit_mode":"read_file","target_path":"main.py","depends_on":["step-1"]}')
    lines.append('{"id":"step-3","instruction":"Search for format_result","action_class":"inspect","edit_mode":"search_files","target_path":".","pattern":"format_result","depends_on":["step-1"]}')
    lines.append('{"id":"step-4","instruction":"Run the program","action_class":"verify","edit_mode":"verify_command","command":"python3 main.py","depends_on":["step-2"]}')
    lines.append("Example larger workflow:")
    lines.append('{"actions":[')
    lines.append('{"id":"step-1","instruction":"Scaffold the tiny repo","action_class":"mutate","edit_mode":"scaffold_files","files":[{"path":"main.py","content":"from report import build_report\\nprint(build_report(12.5, \\"ms\\"))\\n"},{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"Average latency: {value} {unit}\\"\\n"},{"path":"report.py","content":"import formatter\\n\\ndef build_report(value, unit):\\n    return formatter.format_result(value, unit)\\n"}]},')
    lines.append('{"id":"step-2","instruction":"Read main.py to confirm the import","action_class":"inspect","edit_mode":"read_file","target_path":"main.py","depends_on":["step-1"],"inputs_from":["step-1"]},')
    lines.append('{"id":"step-3","instruction":"Run the repo","action_class":"verify","edit_mode":"verify_command","command":"python3 main.py","depends_on":["step-2"],"inputs_from":["step-2"]}')
    lines.append(']}')
    lines.append("Prefer ending a multi-step coding workflow with run_command verification when a simple local command can prove the result works.")
    if is_explicit_scaffold:
        lines.append("For this request, do not return a single write_file for only one file.")
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
