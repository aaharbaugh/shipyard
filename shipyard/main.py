from __future__ import annotations

import argparse
import concurrent.futures
import json
import uuid
import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

from .action_planner import plan_actions
from .graph import build_graph
from .intent_parser import parse_instruction
from .langsmith_config import build_langgraph_config
from .plan_feature import generate_spec_bundle
from .planning_hints import extract_explicit_filenames, is_stale_scratch_target
from .prompt_log import PromptLog
from .runtime_state import enrich_state_sections
from .session_store import SessionStore
from .state import ShipyardState
from .tools.code_graph import inspect_code_graph_status, sync_live_code_graph
from .tracing import write_trace, write_troubleshooting_log


def parse_user_input(raw: str) -> ShipyardState:
    text = raw.strip()
    if not text:
        return {"instruction": ""}

    if text.startswith("{"):
        payload = json.loads(text)
        return _normalize_payload(payload)

    return {"instruction": text, "context": {}}


def read_user_input() -> str:
    first_line = input("> ")
    stripped = first_line.strip()

    if stripped.lower() in {"exit", "quit"}:
        return stripped

    if not stripped.startswith("{"):
        return first_line

    lines = [first_line]
    while True:
        try:
            json.loads("\n".join(lines).strip())
            return "\n".join(lines)
        except json.JSONDecodeError:
            next_line = input("... ")
            if not next_line.strip():
                return "\n".join(lines)
            lines.append(next_line)


def _normalize_payload(payload: dict[str, Any]) -> ShipyardState:
    return {
        "session_id": str(payload.get("session_id", "")).strip() or None,
        "instruction": str(payload.get("instruction", "")).strip(),
        "target_path": _coerce_optional_str(payload.get("target_path")),
        "anchor": _coerce_optional_str(payload.get("anchor")),
        "replacement": _coerce_optional_str(payload.get("replacement")),
        "quantity": payload.get("quantity"),
        "copy_count": payload.get("copy_count"),
        "proposal_mode": _coerce_optional_str(payload.get("proposal_mode")),
        "proposal_model": _coerce_optional_str(payload.get("proposal_model")),
        "edit_mode": _coerce_optional_str(payload.get("edit_mode")),
        "context": payload.get("context", {}) or {},
        "edit_attempts": int(payload.get("edit_attempts", 0) or 0),
        "max_edit_attempts": int(payload.get("max_edit_attempts", 2) or 2),
        "verification_commands": list(payload.get("verification_commands", []) or []),
    }


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def run_once(
    app,
    session_store: SessionStore,
    state: ShipyardState,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> ShipyardState:
    state = _sanitize_stale_target_request(state)
    state["session_id"] = _ensure_session_id(state.get("session_id"))
    state["request_instruction"] = str(state.get("request_instruction") or state.get("instruction") or "")
    action_plan: dict[str, Any] | None = None
    try:
        _emit_progress(progress_callback, "accepted", {"session_id": state["session_id"]})
        PromptLog().append(state)
        _emit_progress(progress_callback, "spec_bundle", {"instruction": state.get("instruction")})
        spec_bundle = generate_spec_bundle(state.get("session_id"), state.get("instruction", ""))
        if spec_bundle.get("created") or spec_bundle.get("mode"):
            state["spec_bundle"] = spec_bundle
        action_plan = plan_actions(state)
        if not action_plan.get("is_valid", True):
            result = _invalid_action_plan_result(state, action_plan)
            result = _persist_result(session_store, result)
            _emit_progress(progress_callback, "completed", {"status": result.get("status"), "trace_path": result.get("trace_path")})
            return result
        result = _run_action_plan(app, state, action_plan, progress_callback)
        if spec_bundle:
            result["spec_bundle"] = spec_bundle
        _emit_progress(progress_callback, "result_ready", {"status": result.get("status")})
        graph_sync = _maybe_sync_graph(result)
        if graph_sync is not None:
            _emit_progress(progress_callback, "graph_sync", {"attempted": graph_sync.get("attempted")})
            result["graph_sync"] = graph_sync
            if graph_sync.get("status"):
                result["code_graph_status"] = graph_sync["status"]
        result = _attach_file_outcome(result)
        result = _persist_result(session_store, result)
        _emit_progress(progress_callback, "completed", {"status": result.get("status"), "trace_path": result.get("trace_path")})
        return result
    except Exception as exc:
        result = _failed_runtime_result(state, str(exc), action_plan=action_plan)
        result = _persist_result(session_store, result)
        _emit_progress(progress_callback, "completed", {"status": result.get("status"), "trace_path": result.get("trace_path"), "error": result.get("error")})
        return result


def _sanitize_stale_target_request(state: ShipyardState) -> ShipyardState:
    instruction = state.get("instruction", "")
    explicit_files = extract_explicit_filenames(instruction)
    if not explicit_files:
        return state

    sanitized = dict(state)
    context = dict(state.get("context", {}) or {})
    target_path = sanitized.get("target_path")
    if target_path and is_stale_scratch_target(target_path):
        sanitized["target_path"] = None
    file_hint = context.get("file_hint")
    if file_hint and is_stale_scratch_target(file_hint):
        context.pop("file_hint", None)
    sanitized["context"] = context
    return sanitized


def _run_action_plan(
    app,
    state: ShipyardState,
    action_plan: dict[str, Any],
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> ShipyardState:
    request_instruction = str(state.get("request_instruction") or state.get("instruction") or "")
    current_state: ShipyardState = dict(state)
    actions = list(action_plan.get("actions", []) or [])
    if not actions:
        actions = [{"instruction": state.get("instruction", "")}]
    aggregated_changed_files: list[str] = []
    action_steps: list[dict[str, Any]] = []
    completed_step_ids: set[str] = set()
    latest_result: ShipyardState = dict(state)
    steps: list[str] = []

    for index, action in enumerate(actions, start=1):
        cancel_check = current_state.get("cancel_check")
        if callable(cancel_check) and cancel_check():
            latest_result = {
                **current_state,
                "status": "cancelled",
                "error": "Run cancelled.",
            }
            break
        step_id = str(action.get("id") or f"step-{index}")
        depends_on = list(action.get("depends_on", []) or [])
        inputs_from = list(action.get("inputs_from", []) or [])
        step = str(action.get("instruction") or "").strip() or state.get("instruction", "")
        steps.append(step)
        unmet_dependencies = [dependency for dependency in depends_on if dependency not in completed_step_ids]
        if unmet_dependencies:
            latest_result = {
                **current_state,
                "status": "blocked",
                "error": f"Unmet step dependencies: {', '.join(unmet_dependencies)}",
                "human_gate": {
                    "status": "blocked",
                    "reason": f"Unmet step dependencies: {', '.join(unmet_dependencies)}",
                    "action": "inspect_plan_dependencies",
                    "prompt": "Review the plan dependencies, then retry.",
                },
            }
            action_steps.append(
                {
                    "id": step_id,
                    "instruction": step,
                    "action_class": action.get("action_class"),
                    "edit_mode": action.get("edit_mode"),
                    "target_path": action.get("target_path"),
                    "depends_on": depends_on,
                    "inputs_from": inputs_from,
                    "status": "blocked",
                    "changed_files": [],
                    "no_op": True,
                }
            )
            break
        _emit_progress(
            progress_callback,
            "verifying" if action.get("action_class") == "verify" else "lead_agent",
            {"instruction": step, "step_index": index, "step_count": len(actions), "step_id": step_id},
        )
        max_retries = int(action.get("max_retries") or (1 if action.get("action_class") in {"inspect", "verify"} else 0))
        attempt = 0
        while True:
            attempt += 1
            step_state: ShipyardState = {
                **current_state,
                "request_instruction": request_instruction,
                "instruction": step,
                "task_id": step_id,
                "target_path": action.get("target_path"),
                "edit_mode": action.get("edit_mode"),
                "anchor": action.get("anchor"),
                "replacement": action.get("replacement"),
                "pattern": action.get("pattern"),
                "command": action.get("command"),
                "pointers": action.get("pointers"),
                "quantity": action.get("quantity"),
                "copy_count": action.get("copy_count"),
                "files": action.get("files"),
                "source_path": action.get("source_path"),
                "destination_path": action.get("destination_path"),
                "paths": action.get("paths"),
                "preplanned_action": action,
                "occurrence_selector": action.get("occurrence_selector"),
                "changed_files": [],
                "depends_on": depends_on,
                "inputs_from": inputs_from,
                "timeout_seconds": action.get("timeout_seconds"),
                "max_retries": max_retries,
            }
            latest_result = _invoke_step_with_timeout(
                app,
                step_state,
                state.get("session_id"),
                step,
                index,
                len(actions),
                int(action.get("timeout_seconds") or 45),
            )
            if latest_result.get("status") in {"failed", "verification_failed", "invalid_proposal", "edit_blocked"} and attempt <= max_retries:
                _emit_progress(
                    progress_callback,
                    "step_retry",
                    {"instruction": step, "step_index": index, "attempt": attempt + 1, "step_id": step_id},
                )
                continue
            break
        step_changed_files = list(latest_result.get("changed_files", []) or [])
        if not step_changed_files and not latest_result.get("no_op") and latest_result.get("target_path"):
            target = Path(str(latest_result["target_path"]))
            if target.exists() and target.is_file():
                step_changed_files = [str(target.resolve())]
        aggregated_changed_files.extend(step_changed_files)
        tool_outputs = list(current_state.get("tool_outputs", []) or [])
        if latest_result.get("tool_output"):
            tool_outputs.append(latest_result["tool_output"])
        current_state = {
            **current_state,
            **latest_result,
            "tool_outputs": tool_outputs,
            "context": {
                **dict(current_state.get("context", {}) or {}),
                **dict(latest_result.get("context", {}) or {}),
                "tool_outputs": tool_outputs,
            },
        }
        action_steps.append(
            {
                "id": step_id,
                "instruction": step,
                "action_class": action.get("action_class"),
                "edit_mode": action.get("edit_mode"),
                "target_path": latest_result.get("target_path"),
                "anchor": action.get("anchor"),
                "pattern": action.get("pattern"),
                "command": action.get("command"),
                "pointers": action.get("pointers"),
                "replacement_preview": str(action.get("replacement") or "")[:120],
                "depends_on": depends_on,
                "inputs_from": inputs_from,
                "timeout_seconds": action.get("timeout_seconds"),
                "max_retries": max_retries,
                "status": latest_result.get("status"),
                "changed_files": step_changed_files,
                "no_op": bool(latest_result.get("no_op")),
            }
        )
        if latest_result.get("status") in {"edited", "verified", "observed"}:
            completed_step_ids.add(step_id)
        if latest_result.get("status") not in {"edited", "verified", "observed"}:
            break

    had_material_edit = any(
        step.get("status") in {"edited", "verified"} and not step.get("no_op")
        for step in action_steps
    )
    deduped: list[str] = []
    if aggregated_changed_files:
        seen: set[str] = set()
        for path in aggregated_changed_files:
            if path in seen:
                continue
            deduped.append(path)
            seen.add(path)
    latest_result["changed_files"] = deduped
    if latest_result.get("status") == "observed" and had_material_edit:
        latest_result["status"] = "edited"
        latest_result["no_op"] = False
    latest_result["request_instruction"] = request_instruction
    latest_result["instruction"] = request_instruction
    latest_result["instruction_steps"] = steps
    latest_result["action_steps"] = action_steps
    latest_result["action_plan"] = action_plan
    return latest_result


def _invoke_step_with_timeout(
    app,
    step_state: ShipyardState,
    session_id: str | None,
    instruction: str,
    step_index: int,
    step_count: int,
    timeout_seconds: int,
) -> ShipyardState:
    config = build_langgraph_config(
        session_id,
        instruction=instruction,
        step_index=step_index,
        step_count=step_count,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(app.invoke, step_state, config=config)
        try:
            return future.result(timeout=max(timeout_seconds, 1))
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {
                **step_state,
                "status": "failed",
                "error": f"Step timed out after {timeout_seconds} seconds.",
                "human_gate": {
                    "status": "blocked",
                    "reason": f"Step timed out after {timeout_seconds} seconds.",
                    "action": "inspect_runtime_failure",
                    "prompt": "Review the timed out step, then retry.",
                },
            }


def _ensure_session_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return uuid.uuid4().hex[:8]
    return text


def _invalid_action_plan_result(state: ShipyardState, action_plan: dict[str, Any]) -> ShipyardState:
    errors = list(action_plan.get("validation_errors", []) or [])
    message = "Action plan was incomplete or invalid."
    if errors:
        message = f"{message} {' '.join(errors)}"
    return {
        **state,
        "status": "invalid_action_plan",
        "error": message,
        "action_plan": action_plan,
        "human_gate": {
            "status": "blocked",
            "reason": message,
            "action": "clarify_request",
            "prompt": "Clarify the missing files or steps, then run again.",
            "details": {"validation_errors": errors},
        },
    }


def _failed_runtime_result(
    state: ShipyardState,
    error: str,
    *,
    action_plan: dict[str, Any] | None = None,
) -> ShipyardState:
    result = {
        **state,
        "status": "failed",
        "error": error,
        "human_gate": {
            "status": "blocked",
            "reason": error,
            "action": "inspect_runtime_failure",
            "prompt": "Review the runtime failure details, then retry.",
        },
    }
    if action_plan:
        result["action_plan"] = action_plan
    return result


def _persist_result(session_store: SessionStore, result: ShipyardState) -> ShipyardState:
    result = _sanitize_runtime_result(result)
    trace_path = write_trace(result)
    result["trace_path"] = trace_path
    troubleshooting_path = write_troubleshooting_log(result)
    result["troubleshooting_path"] = troubleshooting_path
    result = enrich_state_sections(result)
    session_store.append_run(result)
    return result


def _maybe_sync_graph(state: ShipyardState) -> dict[str, Any] | None:
    status = state.get("status")
    target_path = state.get("target_path")
    if status not in {"edited", "verified"} or not target_path:
        return None
    if _should_skip_graph_sync(state):
        return {
            "attempted": False,
            "reason": "Graph sync was skipped for a trivial testing-mode file operation.",
            "status": inspect_code_graph_status(),
        }

    target = Path(target_path).resolve()
    repo_root = Path.cwd().resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        return {
            "attempted": False,
            "reason": "Target path is outside the repo, so graph sync was skipped.",
            "status": inspect_code_graph_status(),
        }

    shipyard_data_root = (repo_root / ".shipyard" / "data").resolve()
    graph_runtime_root = (shipyard_data_root / "graph" / "runtime").resolve()
    graph_index_root = (shipyard_data_root / "graph" / "index").resolve()
    logs_root = (shipyard_data_root / "logs").resolve()
    disallowed_roots = (graph_runtime_root, graph_index_root, logs_root)
    if any(_is_relative_to(target, root) for root in disallowed_roots):
        return {
            "attempted": False,
            "reason": "Target path is internal Shipyard runtime data, so graph sync was skipped.",
            "status": inspect_code_graph_status(),
        }

    sync_result = sync_live_code_graph()
    return {
        "attempted": True,
        "reason": sync_result.get("reason"),
        "ok": sync_result.get("ok"),
        "status": inspect_code_graph_status(),
        "output": sync_result.get("output"),
    }


def _should_skip_graph_sync(state: ShipyardState) -> bool:
    context = state.get("context", {}) or {}
    if not context.get("testing_mode"):
        return False
    instruction = state.get("instruction", "")
    parsed = parse_instruction(instruction)
    mode = state.get("edit_mode") or (parsed[0] if parsed else None)
    return mode in {"write_file", "append", "prepend", "delete_file", "copy_file", "create_files", "rename_symbol"}


def _sanitize_runtime_result(state: ShipyardState) -> ShipyardState:
    cleaned = copy.deepcopy(state)
    code_graph_status = cleaned.get("code_graph_status")
    if isinstance(code_graph_status, dict):
        code_graph_status.pop("details", None)
    graph_sync = cleaned.get("graph_sync")
    if isinstance(graph_sync, dict):
        graph_sync.pop("output", None)
        nested_status = graph_sync.get("status")
        if isinstance(nested_status, dict):
            nested_status.pop("details", None)
    return cleaned


def _attach_file_outcome(state: ShipyardState) -> ShipyardState:
    enriched = copy.deepcopy(state)
    target_path = enriched.get("target_path")
    status = enriched.get("status")
    if not target_path or status not in {"edited", "verified"}:
        return enriched

    existing_changed_files = list(enriched.get("changed_files", []) or [])
    if existing_changed_files:
        enriched["changed_files"] = existing_changed_files
    if enriched.get("no_op") and not existing_changed_files:
        return enriched

    preview_target = existing_changed_files[-1] if existing_changed_files else target_path
    file_path = Path(preview_target)
    if not file_path.exists() or not file_path.is_file():
        return enriched

    content = file_path.read_text(encoding="utf-8")
    preview = content[:240]
    if not existing_changed_files:
        enriched["changed_files"] = [str(file_path.resolve())]
    enriched["file_preview"] = preview
    enriched["file_preview_truncated"] = len(content) > len(preview)
    enriched["content_hash"] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return enriched


def _emit_progress(
    callback: Callable[[str, dict[str, Any] | None], None] | None,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    callback(event, payload or {})


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Shipyard MVP agent.")
    parser.add_argument("--session-id", help="Reuse an existing session id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load and display the latest saved state for the session id at startup.",
    )
    args = parser.parse_args()

    app = build_graph()
    session_store = SessionStore()
    session_id = args.session_id or uuid.uuid4().hex[:8]

    print("Shipyard MVP runner")
    print(f"session_id={session_id}")
    print("Enter plain text or JSON. Type 'exit' to stop.")

    if args.resume:
        resumed = session_store.load_latest_state(session_id)
        if resumed:
            print(f"resumed_status={resumed.get('status', 'unknown')}")
            print(f"resumed_instruction={resumed.get('instruction', '')}")
        else:
            print("resume_requested_but_no_saved_state_found")

    while True:
        raw = read_user_input()
        if raw.strip().lower() in {"exit", "quit"}:
            print("Stopping Shipyard MVP runner.")
            return

        try:
            state = parse_user_input(raw)
        except json.JSONDecodeError as exc:
            print(f"input_error=Invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}")
            print("hint=Paste the full JSON block, then press Enter on a blank line.")
            continue

        state["session_id"] = state.get("session_id") or session_id
        result = run_once(app, session_store, state)
        _print_result(result)
        print(f"trace={result.get('trace_path')}")


def _print_result(result: ShipyardState) -> None:
    print(f"status={result.get('status', 'unknown')}")

    prompt = result.get("prompt")
    if prompt:
        print("prompt:")
        print(prompt)

    snapshot_path = result.get("snapshot_path")
    if snapshot_path:
        print(f"snapshot={snapshot_path}")

    helper_output = result.get("helper_output")
    if helper_output:
        print("helper_output:")
        print(json.dumps(helper_output, indent=2))

    proposal_summary = result.get("proposal_summary")
    if proposal_summary:
        print("proposal_summary:")
        print(json.dumps(proposal_summary, indent=2))

    if result.get("reverted_to_snapshot"):
        print("reverted_to_snapshot=true")

    error = result.get("error")
    if error:
        print(f"error={error}")

    verification_results = result.get("verification_results", [])
    if verification_results:
        print("verification:")
        print(json.dumps(verification_results, indent=2))


if __name__ == "__main__":
    main()
