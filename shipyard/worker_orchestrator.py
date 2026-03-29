"""Worker orchestrator: runs multiple agents in parallel with isolated contexts.

Each worker gets:
- Its own instruction (scoped sub-task from supervisor)
- Its own broad_context filtered to its scope
- Its own action plan (generated independently)
- The shared graph (thread-safe, stateless)

Workers execute via _run_action_plan, same as single-agent mode.
"""
from __future__ import annotations

import concurrent.futures
import difflib
import time
from pathlib import Path
from typing import Any, Callable

from .action_planner import plan_actions
from .context_explorer import build_broad_context, load_context_files
from .graph import build_graph
from .state import ShipyardState
from .workspaces import get_session_workspace


def execute_workers(
    state: ShipyardState,
    subtasks: list[dict[str, Any]],
    run_action_plan: Callable,
    *,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
    max_parallel: int = 4,
) -> dict[str, Any]:
    """Execute sub-tasks as independent worker agents.

    Sequential tasks (with depends_on) run in order.
    Independent tasks run in parallel up to max_parallel.

    Returns merged result with all changed_files, action_steps, diffs.
    """
    app = build_graph()
    completed: dict[str, dict[str, Any]] = {}
    all_changed: list[str] = []
    all_steps: list[dict[str, Any]] = []
    all_transactions: list[dict[str, Any]] = []
    all_diffs: list[str] = []
    worker_results: list[dict[str, Any]] = []
    had_failure = False

    # Group subtasks into execution waves: tasks with no pending deps run together
    remaining = list(subtasks)
    wave_num = 0

    while remaining:
        wave_num += 1
        completed_ids = set(completed.keys())

        # Find tasks whose dependencies are all completed
        ready = []
        still_waiting = []
        for task in remaining:
            deps = set(task.get("depends_on") or [])
            if deps <= completed_ids:
                ready.append(task)
            else:
                still_waiting.append(task)

        if not ready:
            # Deadlock — deps can't be satisfied
            for task in still_waiting:
                worker_results.append({
                    "worker_id": task["id"],
                    "status": "blocked",
                    "error": f"Deadlocked: unmet deps {task.get('depends_on')}",
                })
            break

        remaining = still_waiting

        # Execute ready tasks in parallel
        if progress_callback:
            ids = [t["id"] for t in ready]
            progress_callback("worker_wave", {
                "wave": wave_num,
                "workers": ids,
                "total_remaining": len(remaining),
            })

        print(
            f"[supervisor] wave {wave_num}: "
            f"{len(ready)} worker(s) — {', '.join(t['id'] for t in ready)}",
            flush=True,
        )

        wave_results = _execute_wave(
            ready, state, app, run_action_plan,
            progress_callback=progress_callback,
            max_parallel=max_parallel,
        )

        for task, result in zip(ready, wave_results):
            worker_id = task["id"]
            status = result.get("status", "unknown")
            changed = result.get("changed_files") or []
            steps = result.get("action_steps") or []

            completed[worker_id] = result
            all_changed.extend(changed)
            all_steps.extend(steps)
            all_transactions.extend(result.get("file_transactions") or [])

            if result.get("diff"):
                all_diffs.append(result["diff"])

            label = f"{len(changed)} file(s)" if changed else status
            print(f"  [{worker_id}] {status} → {label}", flush=True)

            worker_results.append({
                "worker_id": worker_id,
                "instruction": task.get("instruction", ""),
                "scope": task.get("scope", ""),
                "status": status,
                "changed_files": changed,
                "step_count": len(steps),
                "error": result.get("error"),
            })

            if status in ("failed", "failed_after_retries", "edit_blocked"):
                had_failure = True

    # Detect conflicts: files edited by multiple workers
    conflicts = _detect_conflicts(worker_results)

    # Dedup changed files
    seen: set[str] = set()
    deduped: list[str] = []
    for f in all_changed:
        if f not in seen:
            deduped.append(f)
            seen.add(f)

    final_status = "edited" if deduped and not had_failure else (
        "failed" if had_failure else "observed"
    )

    return {
        "status": final_status,
        "changed_files": deduped,
        "action_steps": all_steps,
        "file_transactions": all_transactions,
        "diff": "\n".join(all_diffs) if all_diffs else "",
        "worker_results": worker_results,
        "conflicts": conflicts,
        "multi_agent": True,
        "wave_count": wave_num,
        "worker_count": len(subtasks),
    }


def _execute_wave(
    tasks: list[dict[str, Any]],
    parent_state: ShipyardState,
    app: Any,
    run_action_plan: Callable,
    *,
    progress_callback: Callable | None = None,
    max_parallel: int = 4,
) -> list[dict[str, Any]]:
    """Execute a wave of independent tasks in parallel."""
    if len(tasks) == 1:
        return [_run_worker(tasks[0], parent_state, app, run_action_plan, progress_callback)]

    results: list[tuple[int, dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), max_parallel)) as executor:
        future_map = {
            executor.submit(
                _run_worker, task, parent_state, app, run_action_plan, progress_callback
            ): i
            for i, task in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                results.append((idx, future.result()))
            except Exception as exc:
                results.append((idx, {
                    "status": "failed",
                    "error": f"Worker thread error: {exc}",
                    "changed_files": [],
                    "action_steps": [],
                }))

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


def _run_worker(
    task: dict[str, Any],
    parent_state: ShipyardState,
    app: Any,
    run_action_plan: Callable,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Run a single worker agent with its own scoped context."""
    worker_id = task["id"]
    instruction = task.get("instruction", "")
    scope = task.get("scope", ".")
    files = task.get("files") or []

    # Build worker-scoped state
    session_id = parent_state.get("session_id")
    workspace = get_session_workspace(session_id)

    # Filter broad context to worker's scope
    parent_broad = parent_state.get("broad_context") or {}
    scoped_tree = [
        f for f in (parent_broad.get("file_tree") or [])
        if f.startswith(scope) or scope == "."
    ]
    scoped_docs = [
        d for d in (parent_broad.get("discovered_docs") or [])
        if d.startswith(scope) or scope == "."
    ]

    # Load worker's specific files into context
    file_cache: dict[str, str] = {}
    if files:
        for fpath in files[:10]:
            full = workspace / fpath if not Path(fpath).is_absolute() else Path(fpath)
            if full.exists() and full.is_file():
                try:
                    file_cache[str(full)] = full.read_text(encoding="utf-8", errors="replace")[:8000]
                except OSError:
                    pass

    worker_state: ShipyardState = {
        "session_id": session_id,
        "instruction": instruction,
        "request_instruction": instruction,
        "broad_context": {
            **parent_broad,
            "file_tree": scoped_tree,
            "discovered_docs": scoped_docs,
            # Keep project_stack and git_status from parent
        },
        "file_content_cache": file_cache,
        "context": {
            **(parent_state.get("context") or {}),
            "helper_notes": f"Worker agent {worker_id}: scope={scope}",
        },
        "proposal_mode": parent_state.get("proposal_mode"),
        "proposal_model": parent_state.get("proposal_model"),
        "cancel_check": parent_state.get("cancel_check"),
    }

    # Plan actions for this worker's scope
    try:
        action_plan = plan_actions(worker_state)
        if not action_plan.get("is_valid", True):
            return {
                "status": "invalid_action_plan",
                "error": " ".join(action_plan.get("validation_errors") or []),
                "changed_files": [],
                "action_steps": [],
            }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"Worker planning failed: {exc}",
            "changed_files": [],
            "action_steps": [],
        }

    # Execute the plan
    try:
        result = run_action_plan(app, worker_state, action_plan, progress_callback)

        # Collect diff for this worker
        diff_text = ""
        transactions = result.get("file_transactions") or []
        if transactions:
            diffs = []
            for tx in transactions:
                snap = tx.get("snapshot_path")
                target = tx.get("target_path")
                if not snap or not target:
                    continue
                try:
                    before = Path(snap).read_text(errors="replace").splitlines(keepends=True) if Path(snap).exists() else []
                    after = Path(target).read_text(errors="replace").splitlines(keepends=True) if Path(target).exists() else []
                    name = Path(target).name
                    d = "".join(difflib.unified_diff(before, after, fromfile=f"a/{name}", tofile=f"b/{name}"))
                    if d:
                        diffs.append(d)
                except Exception:
                    pass
            diff_text = "\n".join(diffs)

        result["diff"] = diff_text
        result["worker_id"] = worker_id
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"Worker execution failed: {exc}",
            "changed_files": [],
            "action_steps": [],
        }


def _detect_conflicts(worker_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find files that were edited by multiple workers."""
    file_owners: dict[str, list[str]] = {}
    for wr in worker_results:
        worker_id = wr.get("worker_id", "?")
        for f in (wr.get("changed_files") or []):
            file_owners.setdefault(f, []).append(worker_id)

    conflicts = []
    for filepath, owners in file_owners.items():
        if len(owners) > 1:
            conflicts.append({
                "file": filepath,
                "workers": owners,
                "severity": "high",
            })
            print(
                f"[conflict] {Path(filepath).name} edited by {', '.join(owners)}",
                flush=True,
            )

    return conflicts
