from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import uuid
import copy
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

from .action_planner import PlanningCancelledError, plan_actions, plan_next_batch, replan_remaining_actions, request_exploration_files
from .context_explorer import build_broad_context, load_context_files
from .graph import build_graph, _check_file_syntax as _check_file_syntax_fast
from .supervisor import plan_subtasks, should_use_supervisor
from .worker_orchestrator import execute_workers
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
from .workspaces import get_session_workspace, set_session_workspace


def _detect_workspace_syntax_errors(session_id: str | None) -> dict[str, str]:
    """Scan the session workspace for files with syntax errors. Returns {rel_path: error}."""
    from .workspaces import get_session_workspace
    _CHECKABLE = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"}
    _IGNORED = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "dist", "build"}
    errors: dict[str, str] = {}
    try:
        root = get_session_workspace(session_id)
        if not root.exists():
            return {}
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(part in _IGNORED for part in p.relative_to(root).parts):
                continue
            if p.suffix.lower() not in _CHECKABLE:
                continue
            err = _check_file_syntax_fast(str(p))
            if err:
                errors[str(p.relative_to(root))] = err
    except Exception:
        pass
    return errors


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
        "wide_impact_approved": bool(payload.get("wide_impact_approved", False)),
    }


def _record_file_transaction(
    transactions: list[dict[str, Any]],
    result: ShipyardState,
) -> list[dict[str, Any]]:
    target_path = result.get("target_path")
    snapshot_path = result.get("snapshot_path")
    if not target_path or not snapshot_path:
        return transactions
    resolved_target = str(Path(str(target_path)).resolve())
    updated = [dict(item) for item in transactions if str(item.get("target_path")) != resolved_target]
    updated.append(
        {
            "target_path": resolved_target,
            "snapshot_path": str(snapshot_path),
            "target_existed_before_edit": bool(result.get("target_existed_before_edit", True)),
        }
    )
    return updated


def _default_step_retries(action_class: str) -> int:
    return 1 if action_class == "inspect" else 0


def _is_transient_verification_failure(result: ShipyardState) -> bool:
    if "timed out" in str(result.get("error") or "").lower():
        return True
    for verification in list(result.get("verification_results", []) or []):
        if not isinstance(verification, dict):
            continue
        returncode = verification.get("returncode")
        stderr = str(verification.get("stderr") or "").lower()
        stdout = str(verification.get("stdout") or "").lower()
        combined = f"{stderr}\n{stdout}"
        if returncode in {124, 137, 143}:
            return True
        transient_markers = ("temporar", "timed out", "connection reset", "try again", "resource busy")
        if any(marker in combined for marker in transient_markers):
            return True
    return False


def _should_retry_step(action_class: str, result: ShipyardState, attempt: int, max_retries: int) -> bool:
    if attempt > max_retries:
        return False
    latest_status = str(result.get("status") or "")
    if action_class == "inspect":
        return latest_status in {"failed", "verification_failed", "invalid_proposal", "edit_blocked"}
    if action_class == "verify":
        return latest_status == "verification_failed" and _is_transient_verification_failure(result)
    return False


def _step_status_for_result(result: ShipyardState) -> str:
    status = str(result.get("status") or "unknown")
    if status == "edited" and result.get("no_op"):
        return "edit_skipped"
    return status


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _auto_branch(state: ShipyardState) -> dict[str, str] | None:
    """Create a feature branch before edits so damage stays off main."""
    try:
        from .tools.git_tools import GitAutomation
        workspace = get_session_workspace(state.get("session_id"))
        git = GitAutomation(str(workspace))
        status = git.get_status()
        branch = status.get("branch", "")
        # Only branch if we're on main/master — don't nest branches
        if branch not in ("main", "master"):
            return None
        # Slug from instruction
        instruction = state.get("instruction", "")[:40].strip()
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", instruction).strip("-").lower() or "shipyard"
        ts = time.strftime("%m%d-%H%M")
        branch_name = f"shipyard/{slug}-{ts}"
        result = git.create_branch(branch_name)
        print(f"[auto-branch] {result.get('action', '?')} → {branch_name}", flush=True)
        return result
    except Exception as exc:
        print(f"[auto-branch] skipped: {exc}", flush=True)
        return None


def _auto_rollback(state: ShipyardState, result: ShipyardState) -> ShipyardState:
    """If the run failed due to an EDIT failure, revert changed files.

    Does NOT rollback if edits succeeded but a verify/test step failed —
    the edits are likely correct, the test just timed out or found a pre-existing issue.
    """
    status = result.get("status", "")
    if status in ("edited", "verified", "observed", "completed"):
        return result  # success — keep changes

    # Check if any edit step actually succeeded — if so, keep the edits
    steps = result.get("action_steps") or []
    has_successful_edit = any(
        s.get("action_class") == "mutate" and s.get("status") in ("edited", "verified")
        for s in steps
    )
    if has_successful_edit:
        # Edits worked, only a verify/test step failed — don't rollback
        return result

    transactions = result.get("file_transactions") or []
    if not transactions:
        return result
    reverted: list[str] = []
    for tx in transactions:
        snap = tx.get("snapshot_path")
        target = tx.get("target_path")
        if snap and target and Path(str(snap)).exists() and Path(str(target)).exists():
            try:
                from .tools.revert import revert_file
                revert_file(str(target), str(snap))
                reverted.append(str(target))
            except Exception:
                pass
    if reverted:
        print(f"[auto-rollback] reverted {len(reverted)} file(s): {', '.join(Path(p).name for p in reverted)}", flush=True)
        result["auto_rollback"] = {"reverted": reverted, "reason": status}
    return result


def _collect_diffs(result: ShipyardState) -> str:
    """Generate unified diffs for all changed files vs their snapshots."""
    import difflib
    transactions = result.get("file_transactions") or []
    diffs: list[str] = []
    for tx in transactions:
        snap = tx.get("snapshot_path")
        target = tx.get("target_path")
        if not snap or not target:
            continue
        try:
            before = Path(str(snap)).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if Path(str(snap)).exists() else []
            after = Path(str(target)).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if Path(str(target)).exists() else []
            name = Path(str(target)).name
            diff = difflib.unified_diff(before, after, fromfile=f"a/{name}", tofile=f"b/{name}")
            diff_text = "".join(diff)
            if diff_text:
                diffs.append(diff_text)
        except Exception:
            continue
    return "\n".join(diffs)


def _discover_test_command(state: ShipyardState) -> str | None:
    """Auto-detect the project's test runner from config files."""
    try:
        workspace = get_session_workspace(state.get("session_id"))
        # Check package.json scripts
        pkg = workspace / "package.json"
        if pkg.exists():
            import json as _json
            data = _json.loads(pkg.read_text())
            scripts = data.get("scripts", {})
            # Prefer test, then test:unit, then check
            for key in ("test", "test:unit", "check", "lint"):
                if key in scripts:
                    # Detect package manager
                    if (workspace / "pnpm-lock.yaml").exists():
                        return f"pnpm {key}"
                    if (workspace / "yarn.lock").exists():
                        return f"yarn {key}"
                    return f"npm run {key}"
        # Check pyproject.toml
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                return "python -m pytest -x -q"
        # Check for pytest directly
        if (workspace / "tests").is_dir() or (workspace / "test").is_dir():
            return "python -m pytest -x -q"
        return None
    except Exception:
        return None


def run_once(
    app,
    session_store: SessionStore,
    state: ShipyardState,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> ShipyardState:
    state = _sanitize_stale_target_request(state)
    state["session_id"] = _ensure_session_id(state.get("session_id"))
    _register_requested_workspace(state)
    state["request_instruction"] = str(state.get("request_instruction") or state.get("instruction") or "")
    action_plan: dict[str, Any] | None = None
    try:
        _emit_progress(progress_callback, "accepted", {"session_id": state["session_id"]})
        PromptLog().append(state)
        _emit_progress(progress_callback, "spec_bundle", {"instruction": state.get("instruction")})
        spec_bundle = generate_spec_bundle(state.get("session_id"), state.get("instruction", ""))
        if spec_bundle.get("created") or spec_bundle.get("mode"):
            state["spec_bundle"] = spec_bundle
        print(f"[run] planning  instruction={state.get('instruction', '')[:80]!r}", flush=True)
        _emit_progress(progress_callback, "planning", {"instruction": state.get("instruction")})
        # Build broad repo context once before planning so the LLM sees the file tree
        # and key file contents when generating the action plan.
        state["broad_context"] = build_broad_context(state.get("session_id"), state.get("instruction", ""))
        history = session_store.load_history(state["session_id"])
        if isinstance(history, list) and history:
            state["session_journal"] = history[-5:]
        # Skip exploration when the caller has already named the target file
        # or when broad_context already sampled files — the extra LLM call
        # (using nano model) is pure overhead in these cases.
        broad_has_files = bool((state.get("broad_context") or {}).get("sampled_files"))
        if not state.get("target_path") and not broad_has_files:
            explore_paths = request_exploration_files(state)
            if explore_paths:
                state["live_file_context"] = load_context_files(explore_paths, max_files=6, max_content=4000)
        # Clear stale tool_outputs from prior runs. They accumulate in the session
        # context and mislead the LLM into thinking prior failures are still relevant.
        # The session_journal already captures what happened in prior runs.
        if state.get("tool_outputs"):
            state["tool_outputs"] = []
        if state.get("context") and state["context"].get("tool_outputs"):
            state["context"] = {**state["context"], "tool_outputs": []}
        # Detect pre-existing syntax errors in workspace files so the planning LLM knows
        # to use write_file instead of anchor edits on broken files.
        _syntax_errors = _detect_workspace_syntax_errors(state.get("session_id"))
        if _syntax_errors:
            _notes = "\n".join(f"  {f}: {e[:120]}" for f, e in _syntax_errors.items())
            _ctx = dict(state.get("context") or {})
            _existing = _ctx.get("helper_notes", "")
            _ctx["helper_notes"] = (
                f"{_existing}\nWARNING: These workspace files have pre-existing syntax errors.\n"
                f"Plan write_file rewrites (not anchor edits) for them:\n{_notes}"
            ).strip()
            state["context"] = _ctx
        action_plan = _plan_actions_with_cancellation(state)
        actions_count = len(action_plan.get("actions") or [])
        provider = action_plan.get("provider") or "?"
        print(f"[run] plan ready  provider={provider} steps={actions_count}", flush=True)
        if not action_plan.get("is_valid", True):
            result = _invalid_action_plan_result(state, action_plan)
            result = _persist_result(session_store, result)
            _emit_progress(progress_callback, "completed", {"status": result.get("status"), "trace_path": result.get("trace_path")})
            return result
        # Auto-branch: create a feature branch before any edits touch files
        has_mutate = any(
            (a.get("action_class") or "") == "mutate"
            for a in (action_plan.get("actions") or [])
        )
        if has_mutate:
            branch_result = _auto_branch(state)
            if branch_result:
                state["auto_branch"] = branch_result

        # Auto-test: discover project test runner and append a verify step
        test_cmd = _discover_test_command(state)
        if test_cmd and has_mutate:
            actions = action_plan.get("actions") or []
            last_id = actions[-1].get("id", f"step-{len(actions)}") if actions else "step-0"
            actions.append({
                "id": f"auto-test",
                "instruction": f"Run project tests: {test_cmd}",
                "action_class": "verify",
                "edit_mode": "run_command",
                "command": test_cmd,
                "depends_on": [last_id],
                "timeout_seconds": 120,
            })
            action_plan["actions"] = actions
            print(f"[auto-test] appended: {test_cmd}", flush=True)

        # Multi-agent: if the task is complex, decompose into parallel workers
        if should_use_supervisor(state):
            _emit_progress(progress_callback, "supervisor", {"instruction": state.get("instruction")})
            print("[supervisor] decomposing into sub-tasks...", flush=True)
            supervisor_plan = plan_subtasks(state)
            subtasks = supervisor_plan.get("subtasks") or []
            if len(subtasks) > 1:
                print(f"[supervisor] {len(subtasks)} workers: {', '.join(t['id'] for t in subtasks)}", flush=True)
                _emit_progress(progress_callback, "multi_agent", {
                    "worker_count": len(subtasks),
                    "workers": [{"id": t["id"], "scope": t.get("scope"), "instruction": t.get("instruction", "")[:80]} for t in subtasks],
                })
                result = execute_workers(
                    state, subtasks, _run_action_plan,
                    progress_callback=progress_callback,
                )
                result["supervisor_plan"] = supervisor_plan
                result["action_plan"] = action_plan  # keep original plan for reference
            else:
                # Supervisor decided single-agent is fine
                print("[supervisor] single worker sufficient", flush=True)
                result = _run_action_plan(app, state, action_plan, progress_callback)
        else:
            result = _run_action_plan(app, state, action_plan, progress_callback)
        # Chunked planning: if the LLM indicated more batches are needed, keep planning
        # and executing with full context of what was done so far.
        batch_limit = 8  # guard against runaway chunked plans
        batch_count = 1
        while action_plan.get("needs_more_batches") and batch_count < batch_limit:
            batch_count += 1
            _emit_progress(progress_callback, "planning",
                {"instruction": state.get("instruction"), "batch": batch_count})
            next_batch = plan_next_batch(
                {**state, **result},
                completed_steps=list(result.get("action_steps") or []),
                tool_outputs=list(result.get("tool_outputs") or []),
                changed_files=list(result.get("changed_files") or []),
                batch_size=int(state.get("plan_batch_size") or 4),
            )
            if not next_batch or not next_batch.get("actions"):
                break
            action_plan = next_batch
            batch_result = _run_action_plan(
                app,
                {**state, **result},
                action_plan,
                progress_callback,
            )
            # Merge batch result into overall result
            result = {
                **result,
                **batch_result,
                "changed_files": list({*list(result.get("changed_files") or []), *list(batch_result.get("changed_files") or [])}),
                "action_steps": list(result.get("action_steps") or []) + list(batch_result.get("action_steps") or []),
                "tool_outputs": list(result.get("tool_outputs") or []) + list(batch_result.get("tool_outputs") or []),
            }
        if spec_bundle:
            result["spec_bundle"] = spec_bundle
        _emit_progress(progress_callback, "result_ready", {"status": result.get("status")})
        graph_sync = _maybe_sync_graph(result)
        if graph_sync is not None:
            _emit_progress(progress_callback, "graph_sync", {"attempted": graph_sync.get("attempted")})
            result["graph_sync"] = graph_sync
            if graph_sync.get("status"):
                result["code_graph_status"] = graph_sync["status"]
        # Collect unified diffs for audit
        diff_text = _collect_diffs(result)
        if diff_text:
            result["diff"] = diff_text
            print(f"[diff] {len(diff_text)} chars across {diff_text.count('--- a/')} file(s)", flush=True)

        # Auto-rollback: if run failed, revert all changes
        result = _auto_rollback(state, result)

        # Copy branch info to result
        if state.get("auto_branch"):
            result["auto_branch"] = state["auto_branch"]

        result = _attach_file_outcome(result)
        result = _persist_result(session_store, result)
        _emit_progress(progress_callback, "completed", {"status": result.get("status"), "trace_path": result.get("trace_path")})
        return result
    except PlanningCancelledError:
        result = {
            **state,
            "status": "cancelled",
            "error": "Run cancelled.",
        }
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


def _register_requested_workspace(state: ShipyardState) -> None:
    session_id = state.get("session_id")
    context = state.get("context", {}) or {}
    workspace_path = context.get("workspace_path")
    set_session_workspace(session_id, workspace_path)


def _run_action_plan(
    app,
    state: ShipyardState,
    action_plan: dict[str, Any],
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> ShipyardState:
    request_instruction = str(state.get("request_instruction") or state.get("instruction") or "")
    current_state: ShipyardState = dict(state)

    # Inject pre-existing syntax errors into context so the LLM knows the file
    # is already broken before planning any edits.
    actions = list(action_plan.get("actions", []) or [])
    _primary_target = state.get("target_path") or (actions[0].get("target_path") if actions else None)
    if _primary_target:
        _tp = Path(str(_primary_target))
        if _tp.is_file():
            _pre_err = _check_file_syntax_fast(str(_tp))
            if _pre_err:
                current_state = {
                    **current_state,
                    "context": {
                        **dict(current_state.get("context", {}) or {}),
                        "pre_existing_syntax_errors": f"{_primary_target}: {_pre_err}",
                        "helper_notes": (
                            f"{current_state.get('context', {}).get('helper_notes', '')}\n"
                            f"WARNING: Target file has pre-existing syntax errors. "
                            f"Use write_file to rewrite it cleanly rather than anchor edits.\n{_pre_err}"
                        ).strip(),
                    },
                }
    if not actions:
        actions = [{"instruction": state.get("instruction", "")}]
    aggregated_changed_files: list[str] = []
    file_transactions: list[dict[str, Any]] = [dict(item) for item in list(state.get("file_transactions", []) or []) if isinstance(item, dict)]
    action_steps: list[dict[str, Any]] = []
    completed_step_ids: set[str] = set()
    latest_result: ShipyardState = dict(state)
    steps: list[str] = []
    replan_count = 0
    max_replans = 2
    # While loop so we can splice in expand_to sub-actions and adaptive replan actions.
    action_index = 0

    while action_index < len(actions):
        action = actions[action_index]
        index = action_index + 1  # 1-based for display / step IDs

        cancel_check = current_state.get("cancel_check")
        if callable(cancel_check) and cancel_check():
            latest_result = {**current_state, "status": "cancelled", "error": "Run cancelled."}
            break

        step_id = str(action.get("id") or f"step-{index}")
        depends_on = list(action.get("depends_on", []) or [])
        inputs_from = list(action.get("inputs_from", []) or [])
        step = str(action.get("instruction") or "").strip() or state.get("instruction", "")
        steps.append(step)

        unmet_dependencies = [dep for dep in depends_on if dep not in completed_step_ids]
        if unmet_dependencies:
            # Check whether each unmet dep actually failed (vs never ran)
            _failed_status = {"failed_after_retries", "edit_blocked", "failed", "invalid_proposal"}
            failed_dep_msgs = []
            for dep_id in unmet_dependencies:
                for prev in action_steps:
                    if str(prev.get("id")) == dep_id and prev.get("status") in _failed_status:
                        prev_err = prev.get("error") or prev.get("status") or "unknown error"
                        failed_dep_msgs.append(f"Step '{dep_id}' failed: {prev_err}")
                        break
            if failed_dep_msgs:
                error_msg = "; ".join(failed_dep_msgs)
                latest_result = {
                    **current_state,
                    "status": "failed",
                    "error": error_msg,
                }
            else:
                error_msg = f"Unmet step dependencies: {', '.join(unmet_dependencies)}"
                latest_result = {
                    **current_state,
                    "status": "failed",
                    "error": error_msg,
                }
            action_steps.append({
                "id": step_id, "instruction": step,
                "action_class": action.get("action_class"), "edit_mode": action.get("edit_mode"),
                "target_path": action.get("target_path"), "depends_on": depends_on,
                "inputs_from": inputs_from, "status": "blocked", "changed_files": [], "no_op": True,
            })
            break

        # --- Parallel batch detection ---
        # Check if this action and subsequent ones can run in parallel
        # (different target files, no cross-dependencies, same action_class).
        batch_indices = _find_parallel_batch(actions, action_index, completed_step_ids)
        if len(batch_indices) > 1:
            batch_labels = ", ".join(
                f"{actions[i].get('edit_mode', '?')}:{actions[i].get('target_path', '?')}"
                for i in batch_indices
            )
            print(f"[parallel batch] {len(batch_indices)} steps: {batch_labels}", flush=True)
            _emit_progress(progress_callback, "parallel_batch", {
                "step_count": len(batch_indices),
                "step_ids": [str(actions[i].get("id") or f"step-{i+1}") for i in batch_indices],
            })
            batch_results = _execute_batch_parallel(
                app, actions, batch_indices, current_state, state,
                completed_step_ids, len(actions), progress_callback,
            )
            for b_idx, b_action, b_result in batch_results:
                b_index = b_idx + 1
                b_step_id = str(b_action.get("id") or f"step-{b_index}")
                b_step = str(b_action.get("instruction") or "").strip() or state.get("instruction", "")
                b_status = b_result.get("status") or "?"
                b_changed = b_result.get("changed_files") or []
                b_no_op = b_result.get("no_op")
                b_label = "no-op" if b_no_op else (f"{len(b_changed)} file(s) changed" if b_changed else b_status)
                print(f"  [{b_step_id}] {b_action.get('edit_mode', '?')} {b_action.get('target_path', '')} → {b_label}", flush=True)

                step_changed_files = list(b_changed)
                aggregated_changed_files.extend(step_changed_files)
                file_transactions = _record_file_transaction(file_transactions, b_result)
                tool_outputs = list(current_state.get("tool_outputs", []) or [])
                if b_result.get("tool_output"):
                    tool_outputs.append(b_result["tool_output"])

                # Merge file content cache across batch results (reads populate it)
                merged_cache = {
                    **dict(current_state.get("file_content_cache") or {}),
                    **dict(b_result.get("file_content_cache") or {}),
                }
                current_state = {
                    **current_state,
                    **b_result,
                    "tool_outputs": tool_outputs,
                    "file_transactions": [dict(item) for item in file_transactions],
                    "file_content_cache": merged_cache,
                    "context": {
                        **dict(current_state.get("context", {}) or {}),
                        **dict(b_result.get("context", {}) or {}),
                        "tool_outputs": tool_outputs,
                    },
                }
                b_step_status = _step_status_for_result(b_result)
                action_steps.append({
                    "id": b_step_id, "instruction": b_step,
                    "action_class": b_action.get("action_class"),
                    "edit_mode": b_action.get("edit_mode"),
                    "target_path": b_result.get("target_path"),
                    "status": b_step_status,
                    "changed_files": step_changed_files,
                    "no_op": bool(b_no_op),
                    "parallel_batch": True,
                })
                if b_step_status in {"edited", "verified", "observed", "edit_skipped"}:
                    completed_step_ids.add(b_step_id)

            latest_result = current_state
            action_index = batch_indices[-1] + 1
            continue
        # --- End parallel batch ---

        _emit_progress(
            progress_callback,
            "verifying" if action.get("action_class") == "verify" else "lead_agent",
            {
                "instruction": step, "step_index": index, "step_count": len(actions),
                "step_id": step_id, "role": _infer_subagent_role(action),
                "agent_type": _infer_subagent_type(action),
                "allowed_actions": list(action.get("allowed_actions", [])) or ([action.get("edit_mode")] if action.get("edit_mode") else []),
                "depends_on": depends_on, "inputs_from": inputs_from,
            },
        )

        action_class = str(action.get("action_class") or "").strip()
        max_retries = int(action.get("max_retries") if action.get("max_retries") is not None else _default_step_retries(action_class))
        edit_mode_label = action.get("edit_mode") or action_class or "?"
        target_label = action.get("target_path") or ""
        print(
            f"[step {index}/{len(actions)}] {edit_mode_label}"
            + (f"  {target_label}" if target_label else ""),
            flush=True,
        )
        attempt = 0
        while True:
            attempt += 1
            # Clear snapshot_path for non-mutate steps (verify_command, run_command, etc.)
            # so recover_or_finish doesn't accidentally revert a prior step's edit.
            _clear_snapshot = action_class != "mutate"
            step_state: ShipyardState = {
                **current_state,
                "request_instruction": request_instruction,
                "instruction": step,
                "task_id": step_id,
                "target_path": action.get("target_path"),
                "edit_mode": action.get("edit_mode"),
                **({"snapshot_path": None} if _clear_snapshot else {}),
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
                "file_transactions": [dict(item) for item in file_transactions],
                "depends_on": depends_on,
                "inputs_from": inputs_from,
                "timeout_seconds": action.get("timeout_seconds"),
                "max_retries": max_retries,
                "tool_name": action.get("tool_name"),
                "tool_source": action.get("tool_source"),
                "tool_args": action.get("tool_args") or {},
                "verification_retry_count": max(attempt - 1, 0) if action_class == "verify" else int(current_state.get("verification_retry_count") or 0),
            }
            latest_result = _invoke_step_with_timeout(
                app, step_state, state.get("session_id"), step, index, len(actions),
                int(action.get("timeout_seconds") or 90),
            )
            result_status = latest_result.get("status") or "?"
            verification_results = latest_result.get("verification_results") or []
            if verification_results:
                passed = sum(1 for r in verification_results if r.get("returncode") == 0)
                failed = len(verification_results) - passed
                print(f"  verify: {passed} passed, {failed} failed", flush=True)
            if result_status not in {"edited", "verified", "observed", "edit_skipped"}:
                err = latest_result.get("error") or ""
                print(f"  → {result_status}" + (f": {err[:120]}" if err else ""), flush=True)
            else:
                changed = latest_result.get("changed_files") or []
                no_op = latest_result.get("no_op")
                label = "no-op" if no_op else (f"{len(changed)} file(s) changed" if changed else result_status)
                print(f"  → {label}", flush=True)
            if _should_retry_step(action_class, latest_result, attempt, max_retries):
                _emit_progress(progress_callback, "step_retry",
                    {"instruction": step, "step_index": index, "attempt": attempt + 1, "step_id": step_id})
                continue
            break

        # expand_to: search_then_edit and similar actions return sub-actions to splice in
        expand_to = list(latest_result.pop("expand_to", None) or [])
        if expand_to:
            actions[action_index + 1:action_index + 1] = expand_to

        step_changed_files = list(latest_result.get("changed_files", []) or [])
        if not step_changed_files and not latest_result.get("no_op") and latest_result.get("target_path"):
            target = Path(str(latest_result["target_path"]))
            if target.exists() and target.is_file():
                step_changed_files = [str(target.resolve())]
        aggregated_changed_files.extend(step_changed_files)
        file_transactions = _record_file_transaction(file_transactions, latest_result)
        tool_outputs = list(current_state.get("tool_outputs", []) or [])
        if latest_result.get("tool_output"):
            tool_outputs.append(latest_result["tool_output"])
        current_state = {
            **current_state,
            **latest_result,
            "tool_outputs": tool_outputs,
            "file_transactions": [dict(item) for item in file_transactions],
            "file_content_cache": {
                **dict(current_state.get("file_content_cache") or {}),
                **dict(latest_result.get("file_content_cache") or {}),
            },
            "context": {
                **dict(current_state.get("context", {}) or {}),
                **dict(latest_result.get("context", {}) or {}),
                "tool_outputs": tool_outputs,
            },
        }

        step_status = _step_status_for_result(latest_result)
        action_steps.append({
            "id": step_id, "instruction": step, "action_class": action_class,
            "edit_mode": action.get("edit_mode"), "target_path": latest_result.get("target_path"),
            "anchor": action.get("anchor"), "pattern": action.get("pattern"),
            "command": action.get("command"), "pointers": action.get("pointers"),
            "replacement_preview": str(action.get("replacement") or "")[:120],
            "depends_on": depends_on, "inputs_from": inputs_from,
            "timeout_seconds": action.get("timeout_seconds"), "max_retries": max_retries,
            "retry_count": max(attempt - 1, 0), "status": step_status,
            "changed_files": step_changed_files, "no_op": bool(latest_result.get("no_op")),
        })

        if step_status in {"edited", "verified", "observed", "edit_skipped"}:
            completed_step_ids.add(step_id)
            action_index += 1
            continue

        # Wide-impact gate triggered — surface human_gate and hard-stop immediately
        if step_status == "needs_approval":
            for remaining_index, remaining_action in enumerate(actions[action_index + 1:], start=action_index + 2):
                action_steps.append({
                    "id": str(remaining_action.get("id") or f"step-{remaining_index}"),
                    "instruction": str(remaining_action.get("instruction") or "").strip() or "Pending step",
                    "action_class": remaining_action.get("action_class"),
                    "edit_mode": remaining_action.get("edit_mode"),
                    "target_path": remaining_action.get("target_path"),
                    "status": "skipped",
                    "no_op": True,
                })
            break

        # Step failed — try adaptive replan before giving up, but only for
        # recoverable failures.  Exhausted-retry failures are terminal: the step
        # was already retried the maximum number of times, so replanning cannot
        # help and would only obscure the real error for downstream steps.
        remaining = actions[action_index + 1:]
        if remaining and replan_count < max_replans and step_status != "failed_after_retries":
            _emit_progress(progress_callback, "replanning",
                {"step_id": step_id, "step_index": index, "replan_count": replan_count + 1})
            failed_step_info = {
                "id": step_id, "instruction": step,
                "edit_mode": action.get("edit_mode"),
                "error": latest_result.get("error") or step_status,
                "status": step_status,
            }
            revised = replan_remaining_actions(
                current_state,
                completed_steps=action_steps,
                failed_step=failed_step_info,
                remaining_actions=remaining,
            )
            if revised is not None:
                actions[action_index + 1:] = revised
                replan_count += 1
                action_index += 1  # skip the failed step, continue with revised plan
                continue

        # Hard stop: mark all remaining as skipped
        for remaining_index, remaining_action in enumerate(actions[action_index + 1:], start=action_index + 2):
            action_steps.append({
                "id": str(remaining_action.get("id") or f"step-{remaining_index}"),
                "instruction": str(remaining_action.get("instruction") or "").strip() or "Pending step",
                "action_class": remaining_action.get("action_class"),
                "edit_mode": remaining_action.get("edit_mode"),
                "target_path": remaining_action.get("target_path"),
                "anchor": remaining_action.get("anchor"),
                "pattern": remaining_action.get("pattern"),
                "command": remaining_action.get("command"),
                "pointers": remaining_action.get("pointers"),
                "replacement_preview": str(remaining_action.get("replacement") or "")[:120],
                "depends_on": list(remaining_action.get("depends_on", []) or []),
                "inputs_from": list(remaining_action.get("inputs_from", []) or []),
                "timeout_seconds": remaining_action.get("timeout_seconds"),
                "max_retries": remaining_action.get("max_retries"),
                "status": "skipped", "changed_files": [], "no_op": True,
            })
        break

    had_material_edit = any(
        step.get("status") in {"edited", "verified"} and not step.get("no_op")
        for step in action_steps
    )
    deduped: list[str] = []
    if aggregated_changed_files:
        seen: set[str] = set()
        for path in aggregated_changed_files:
            if path not in seen:
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
    latest_result["file_transactions"] = file_transactions
    latest_result["tasks"] = _merge_runtime_tasks(latest_result.get("tasks"), actions, action_steps)
    return latest_result


def _find_parallel_batch(
    actions: list[dict[str, Any]],
    start_index: int,
    completed_step_ids: set[str],
) -> list[int]:
    """Find consecutive actions starting at start_index that can run in parallel.

    Actions are parallelizable when they:
    - Target different files (or no file)
    - Have no unmet depends_on or inputs_from
    - Are all the same action_class (e.g., all mutate or all inspect)

    Returns list of action indices in the batch (at least 1 element).
    """
    if start_index >= len(actions):
        return []
    first = actions[start_index]
    first_class = first.get("action_class") or ""
    first_target = str(first.get("target_path") or "")
    # Can't parallelize if the first action has no target (ambiguous scope)
    if not first_target:
        return [start_index]
    batch = [start_index]
    seen_targets: set[str] = {first_target}

    for i in range(start_index + 1, len(actions)):
        action = actions[i]
        # Must be same action_class
        if (action.get("action_class") or "") != first_class:
            break
        # Must have no unmet dependencies
        deps = set(action.get("depends_on", []) or []) | set(action.get("inputs_from", []) or [])
        # Allow deps on completed steps, but not on steps in this batch
        batch_ids = {str(actions[j].get("id") or f"step-{j+1}") for j in batch}
        if deps - completed_step_ids - batch_ids:
            break  # has unmet deps outside batch
        if deps & batch_ids:
            break  # depends on something in this batch — must wait
        # Must target a different file (and must have a target)
        target = str(action.get("target_path") or "")
        if not target:
            break  # no target — can't parallelize safely
        if target in seen_targets:
            break  # same file — can't parallelize
        seen_targets.add(target)
        batch.append(i)

    return batch


def _execute_batch_parallel(
    app,
    actions: list[dict[str, Any]],
    action_indices: list[int],
    current_state: ShipyardState,
    state: ShipyardState,
    completed_step_ids: set[str],
    total_actions: int,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> list[tuple[int, dict[str, Any], ShipyardState]]:
    """Execute a batch of independent actions in parallel.

    Returns list of (action_index, action, result) tuples in order.
    """
    request_instruction = str(state.get("request_instruction") or state.get("instruction") or "")

    def _run_one(action_index: int) -> tuple[int, dict[str, Any], ShipyardState]:
        action = actions[action_index]
        index = action_index + 1
        step_id = str(action.get("id") or f"step-{index}")
        step = str(action.get("instruction") or "").strip() or state.get("instruction", "")
        action_class = action.get("action_class") or ""
        max_retries = int(action.get("max_retries") or (2 if action_class == "mutate" else 0))
        _clear_snapshot = action_class != "mutate"

        step_state: ShipyardState = {
            **current_state,
            "request_instruction": request_instruction,
            "instruction": step,
            "task_id": step_id,
            "target_path": action.get("target_path"),
            "edit_mode": action.get("edit_mode"),
            **({"snapshot_path": None} if _clear_snapshot else {}),
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
            "depends_on": list(action.get("depends_on", []) or []),
            "inputs_from": list(action.get("inputs_from", []) or []),
            "timeout_seconds": action.get("timeout_seconds"),
            "max_retries": max_retries,
            "tool_name": action.get("tool_name"),
            "tool_source": action.get("tool_source"),
            "tool_args": action.get("tool_args") or {},
        }
        result = _invoke_step_with_timeout(
            app, step_state, state.get("session_id"), step, index, total_actions,
            int(action.get("timeout_seconds") or 90),
        )
        return (action_index, action, result)

    results: list[tuple[int, dict[str, Any], ShipyardState]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(action_indices), 4)) as executor:
        futures = {executor.submit(_run_one, idx): idx for idx in action_indices}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                idx = futures[future]
                action = actions[idx]
                results.append((idx, action, {
                    **current_state,
                    "status": "failed",
                    "error": f"Parallel execution error: {exc}",
                }))
    # Sort by original index order
    results.sort(key=lambda x: x[0])
    return results


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
    cancel_check = step_state.get("cancel_check")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(app.invoke, step_state, config=config)
        deadline = time.monotonic() + max(timeout_seconds, 1)
        while True:
            if callable(cancel_check) and cancel_check():
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return {
                    **step_state,
                    "status": "cancelled",
                    "error": "Run cancelled.",
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return {
                    **step_state,
                    "status": "failed",
                    "error": f"Step timed out after {timeout_seconds} seconds.",
                }
            try:
                return future.result(timeout=min(0.25, max(remaining, 0.01)))
            except concurrent.futures.TimeoutError:
                continue
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _plan_actions_with_cancellation(state: ShipyardState) -> dict[str, Any]:
    cancel_check = state.get("cancel_check")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(plan_actions, state)
        while True:
            if callable(cancel_check) and cancel_check():
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise PlanningCancelledError("Run cancelled during planning.")
            try:
                return future.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                continue
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _merge_runtime_tasks(
    existing_tasks: Any,
    actions: list[dict[str, Any]],
    action_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        dict(task)
        for task in list(existing_tasks or [])
        if isinstance(task, dict)
    ]
    seen = {str(task.get("task_id")) for task in tasks if task.get("task_id")}
    step_by_id = {
        str(step.get("id")): step
        for step in action_steps
        if isinstance(step, dict) and step.get("id")
    }
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            continue
        task_id = str(action.get("id") or f"step-{index}")
        if task_id in seen:
            continue
        step = step_by_id.get(task_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "role": action.get("role") or _infer_subagent_role(action),
                "agent_type": action.get("agent_type") or _infer_subagent_type(action),
                "parent_task_id": action.get("parent_task_id"),
                "child_task_ids": list(action.get("child_task_ids", []) or []),
                "goal": action.get("instruction"),
                "allowed_actions": list(action.get("allowed_actions", [])) or ([action.get("edit_mode")] if action.get("edit_mode") else []),
                "status": step.get("status") or ("planned" if action.get("valid", True) else "invalid"),
                "result": {
                    "changed_files": list(step.get("changed_files", []) or []),
                    "no_op": bool(step.get("no_op")),
                },
                "artifacts": {
                    "target_path": step.get("target_path") or action.get("target_path"),
                    "command": step.get("command") or action.get("command"),
                },
                "depends_on": list(action.get("depends_on", []) or []),
                "inputs_from": list(action.get("inputs_from", []) or []),
            }
        )
        seen.add(task_id)
    return tasks


def _infer_subagent_role(action: dict[str, Any]) -> str:
    action_class = str(action.get("action_class") or "").strip()
    if action_class == "inspect":
        return "inspector-agent"
    if action_class == "verify":
        return "verifier-agent"
    return "editor-agent"


def _infer_subagent_type(action: dict[str, Any]) -> str:
    action_class = str(action.get("action_class") or "").strip()
    if action_class == "inspect":
        return "inspector"
    if action_class == "verify":
        return "verifier"
    return "editor"


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
    return _strip_nonserializable_runtime_values(cleaned)


def _strip_nonserializable_runtime_values(value: Any) -> Any:
    if callable(value):
        return None
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            sanitized = _strip_nonserializable_runtime_values(item)
            if sanitized is None:
                continue
            cleaned[key] = sanitized
        return cleaned
    if isinstance(value, list):
        return [item for item in (_strip_nonserializable_runtime_values(item) for item in value) if item is not None]
    if isinstance(value, tuple):
        return [item for item in (_strip_nonserializable_runtime_values(item) for item in value) if item is not None]
    return value


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
