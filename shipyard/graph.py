from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pathlib import Path
import re
import shlex
import subprocess
import shutil
import time

from .context_explorer import load_context_files
from .human_gate import make_human_gate
from .helper_agent import run_helper_agent
from .intent_parser import parse_occurrence_selector
from .pathing import resolve_target_path
from .prompts import build_runtime_prompt
from .proposal import propose_edit
from .state import ShipyardState
from .tools.code_graph import inspect_code_graph_status
from .tools.edit_file import (
    AnchorEditError,
    apply_anchor_edit,
    apply_pointer_edits,
    apply_symbol_rename,
    find_anchor_pointers,
    validate_anchor_edit,
    validate_pointer_edits,
)
from .tools.function_edit import FunctionEditError, apply_function_edit, get_function_source
from .tools.read_file import read_file
from .tools.revert import revert_file
from .tools.snapshot import snapshot_file
from .tools.verify import run_verification
from .workspaces import get_session_workspace

TOOL_EDIT_MODES = {"list_files", "read_file", "search_files", "run_command"}
REFINABLE_PREPLANNED_MODES = {"anchor", "rename_symbol", "named_function", "append", "prepend", "write_file", "search_and_replace"}
_WIDE_IMPACT_THRESHOLD = 50  # raised for unsupervised execution on real codebases


def _check_file_syntax(path: str) -> str | None:
    """Return None if syntax is OK or check is skipped; return error string if file has syntax errors.

    Supports .py files (python3 -m py_compile) and .js/.mjs/.cjs files (node --check).
    Times out after 5 seconds. Returns None silently on any exception.
    """
    try:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".py":
            if not shutil.which("python3"):
                return None
            result = subprocess.run(
                ["python3", "-m", "py_compile", path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return result.stderr.strip() or "Syntax error (py_compile)"
            return None
        elif suffix in {".js", ".mjs", ".cjs"}:
            if not shutil.which("node"):
                return None
            result = subprocess.run(
                ["node", "--check", path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return result.stderr.strip() or "Syntax error (node --check)"
            return None
        elif suffix in {".ts", ".tsx", ".mts", ".cts"}:
            if not shutil.which("tsc"):
                return None
            result = subprocess.run(
                ["tsc", "--noEmit", "--allowJs", "--checkJs", "false",
                 "--skipLibCheck", "--isolatedModules", "--noResolve", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return result.stdout.strip() or result.stderr.strip() or "Syntax error (tsc)"
            return None
        return None
    except Exception:
        return None


def _is_path_within_bounds(path: Path, *allowed_roots: Path) -> bool:
    """
    Return True if *path* resolves to a location inside at least one of *allowed_roots*.
    Prevents path-traversal attacks (e.g. ../../../etc/passwd).
    """
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _check_bounds(path: Path, allowed: list[Path] | None, label: str) -> str | None:
    """Return an error string if *path* is outside *allowed*, or None if OK.

    When *allowed* is None (no active session) the check is skipped.
    """
    if allowed is None:
        return None
    if not _is_path_within_bounds(path, *allowed):
        return f"{label} is outside allowed workspace: {path}"
    return None


def _count_affected_files(root: Path, search_text: str) -> int:
    """Count .py files under root that contain search_text (for gate checks)."""
    count = 0
    for file_path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in file_path.parts):
            continue
        try:
            if search_text in file_path.read_text(encoding="utf-8"):
                count += 1
        except (OSError, UnicodeDecodeError):
            continue
    return count


def _repair_anchor_with_pointers(state: ShipyardState, error: str) -> dict | None:
    context = dict(state.get("context", {}))
    helper_notes = context.get("helper_notes", "")
    repair_note = (
        "Return exact edit pointers for each intended replacement span. "
        "Pointers must be 0-based character offsets into the exact current file string, with end exclusive."
    )
    context["helper_notes"] = f"{helper_notes}\n{repair_note}\nPrevious error: {error}".strip()
    repaired = propose_edit(
        {
            **state,
            "context": context,
            "file_before": state.get("file_before", ""),
        }
    )
    pointers = repaired.get("pointers")
    if repaired.get("edit_mode") != "anchor" or not pointers or repaired.get("replacement") is None:
        return None
    return {
        "anchor": repaired.get("anchor") or state.get("anchor"),
        "replacement": repaired.get("replacement"),
        "pointers": pointers,
        "proposal_summary": {
            **dict(state.get("proposal_summary", {}) or {}),
            "pointers": pointers,
            "repair_reason": "Applied pointer repair after ambiguous anchor match.",
        },
        "context": context,
    }


def _replan_mutate_step_from_current_file(state: ShipyardState, error: str) -> dict | None:
    target_path = state.get("target_path")
    if not target_path:
        return None
    target = Path(str(target_path))
    if not target.exists() or not target.is_file():
        return None

    try:
        current_file_before = read_file(str(target))
    except OSError:
        return None

    context = dict(state.get("context", {}) or {})
    helper_notes = context.get("helper_notes", "")
    syntax_err = _check_file_syntax(str(target))
    file_line_count = len(current_file_before.splitlines())
    use_rewrite = (state.get("edit_attempts", 0) >= 1 or syntax_err is not None) and file_line_count <= 300
    repair_note = (
        "Replan only this failed mutate step from the exact current file contents. "
        + (
            "The file has syntax errors or prior edits failed — use write_file mode to rewrite the entire file cleanly. "
            "Put the complete corrected file content in replacement. "
            if use_rewrite else
            "Use a localized edit. Do not trust stale anchors from earlier planning. "
            "Prefer exact pointers when needed."
        )
    )
    context["helper_notes"] = f"{helper_notes}\n{repair_note}\nPrevious error: {error}".strip()
    repaired = propose_edit(
        {
            **state,
            "context": context,
            "file_before": current_file_before,
        }
    )
    if not repaired.get("is_valid"):
        return None
    if repaired.get("edit_mode") not in {"anchor", "rename_symbol", "write_file"}:
        return None
    result = {
        "edit_mode": repaired.get("edit_mode"),
        "anchor": repaired.get("anchor"),
        "replacement": repaired.get("replacement"),
        "pointers": repaired.get("pointers"),
        "proposal_summary": {
            **dict(state.get("proposal_summary", {}) or {}),
            "edit_mode": repaired.get("edit_mode"),
            "anchor": repaired.get("anchor"),
            "pointers": repaired.get("pointers"),
            "repair_reason": "Replanned failed mutate step from the current file contents.",
        },
        "context": context,
        "file_before": current_file_before,
    }
    if repaired.get("edit_mode") == "write_file":
        result["full_file_rewrite"] = True
    return result


def seed_defaults(state: ShipyardState) -> dict:
    return {
        "edit_attempts": state.get("edit_attempts", 0),
        "max_edit_attempts": state.get("max_edit_attempts", 2),
        "reverted_to_snapshot": False,
    }


def fetch_step_context(state: ShipyardState) -> dict:
    """
    Read the target file (and any context_files from the coarse action) before plan_edit runs.
    This gives the LLM live file content so anchors/pointers are generated against reality.
    Runs both on the first pass and after a retry (re-reads after potential revert).
    """
    target_path = state.get("target_path")
    edit_mode = state.get("edit_mode") or (state.get("preplanned_action") or {}).get("edit_mode")

    # Modes that don't operate on a single readable file
    skip_read_modes = {
        "list_files", "search_files", "run_command", "verify_command", "run_tests",
        "create_directory", "move_file", "rename_file", "read_many_files", "scaffold_files",
    }

    result: dict = {}

    # Sandbox relative paths to session workspace early so file reads
    # are consistent with where writes will land.
    if target_path:
        sandboxed = _sandbox_target_path(target_path, state)
        if sandboxed and sandboxed != target_path:
            target_path = sandboxed
            result["target_path"] = target_path
            # Also update the preplanned action so downstream nodes see the sandboxed path
            preplanned = state.get("preplanned_action")
            if isinstance(preplanned, dict) and preplanned.get("target_path"):
                updated_preplanned = {**preplanned, "target_path": target_path}
                result["preplanned_action"] = updated_preplanned

    if target_path and edit_mode not in skip_read_modes:
        # Check file content cache first (populated by prior read_file steps
        # in parallel batches) to avoid redundant disk reads.
        file_cache = state.get("file_content_cache") or {}
        cached = file_cache.get(str(target_path))
        if cached is not None:
            result["file_before"] = cached
            result["target_existed_before_edit"] = True
        else:
            file_path = Path(str(target_path))
            if file_path.exists() and file_path.is_file():
                try:
                    result["file_before"] = read_file(str(target_path))
                    result["target_existed_before_edit"] = True
                except OSError:
                    pass
            elif edit_mode in {"write_file", "append", "prepend", "create_files"}:
                result["file_before"] = ""
                result["target_existed_before_edit"] = False

    # Load any additional context files declared by the coarse action so the LLM
    # can see related files before generating fine-grained edits.
    preplanned = state.get("preplanned_action") or {}
    context_file_paths = preplanned.get("context_files") or []
    if isinstance(context_file_paths, list) and context_file_paths:
        loaded = load_context_files(context_file_paths)
        if loaded:
            context = dict(state.get("context") or {})
            existing_notes = context.get("helper_notes", "")
            file_snippets = "\n".join(
                f"--- {path} ---\n{content}" for path, content in loaded.items()
            )
            context["helper_notes"] = f"{existing_notes}\nRelated file context:\n{file_snippets}".strip()
            result["context"] = context
            result["live_file_context"] = loaded

    return result


def prepare_prompt(state: ShipyardState) -> dict:
    return {
        "prompt": build_runtime_prompt(state),
        "status": "prepared",
    }


def consult_helper_agent(state: ShipyardState) -> dict:
    helper_result = run_helper_agent(state)
    context = dict(state.get("context", {}))
    context["helper_notes"] = helper_result["notes"]
    existing_tasks = list(state.get("tasks", []) or [])
    helper_task_id = str(helper_result.get("task_id") or "helper-planner-task")
    helper_task = {
        "task_id": helper_task_id,
        "role": helper_result.get("agent_name", "helper-planner"),
        "agent_type": helper_result.get("agent_type", "helper"),
        "parent_task_id": state.get("task_id") or "run-root",
        "goal": helper_result.get("recommendation") or helper_result.get("notes"),
        "allowed_actions": list(helper_result.get("allowed_actions", [])),
        "status": "planned",
        "artifacts": {
            "task_type": helper_result.get("task_type"),
            "delegation_mode": helper_result.get("delegation_mode"),
            "notes": helper_result.get("notes"),
        },
    }
    return {
        "context": context,
        "tasks": [*existing_tasks, helper_task],
        "helper_output": {
            "helper_agent": helper_result,
        },
        "status": "helper_consulted",
    }


def plan_edit(state: ShipyardState) -> dict:
    preplanned = state.get("preplanned_action")
    if isinstance(preplanned, dict) and preplanned.get("edit_mode"):
        if _should_refine_preplanned_action(state, preplanned):
            planned = _refine_preplanned_action(state, preplanned)
        else:
            planned = dict(preplanned)
            planned.setdefault("provider", state.get("action_plan", {}).get("provider"))
            planned.setdefault("provider_reason", state.get("action_plan", {}).get("provider_reason"))
            planned.setdefault("is_valid", preplanned.get("valid", True))
            planned.setdefault("validation_errors", preplanned.get("validation_errors", []))
    else:
        planned = propose_edit(state)
    helper_output = dict(state.get("helper_output", {}))
    context = dict(state.get("context", {}))
    planned_target = planned.get("target_path")
    planned_mode = planned.get("edit_mode") or state.get("edit_mode") or "anchor"
    function_name = context.get("function_name")

    helper_agent = dict(helper_output.get("helper_agent", {}))
    if context.get("test_failure"):
        refreshed_notes = helper_agent.get("notes") or context.get("helper_notes", "")
    elif planned_mode == "named_function" and function_name and planned_target:
        refreshed_notes = f"Target named function `{function_name}` in {planned_target}."
    elif planned_target:
        refreshed_notes = f"Use {planned_mode}-based editing for {planned_target}."
    else:
        refreshed_notes = helper_agent.get("notes") or context.get("helper_notes", "")

    if refreshed_notes:
        helper_agent["notes"] = refreshed_notes
        helper_output["helper_agent"] = helper_agent
        context["helper_notes"] = refreshed_notes

    helper_output["proposal"] = {
        "provider": planned.get("provider"),
        "provider_reason": planned.get("provider_reason"),
        "is_valid": planned.get("is_valid"),
        "validation_errors": planned.get("validation_errors", []),
    }
    proposal_summary = {
        "provider": planned.get("provider"),
        "provider_reason": planned.get("provider_reason"),
        "id": planned.get("id"),
        "action_class": planned.get("action_class"),
        "edit_mode": planned.get("edit_mode") or state.get("edit_mode") or "anchor",
        "target_path": planned.get("target_path"),
        "target_path_source": planned.get("target_path_source"),
        "pointers": planned.get("pointers"),
        "occurrence_selector": planned.get("occurrence_selector"),
        "quantity": planned.get("quantity"),
        "copy_count": planned.get("copy_count"),
        "files": planned.get("files"),
        "source_path": planned.get("source_path"),
        "destination_path": planned.get("destination_path"),
        "paths": planned.get("paths"),
        "depends_on": planned.get("depends_on"),
        "inputs_from": planned.get("inputs_from"),
        "timeout_seconds": planned.get("timeout_seconds"),
        "max_retries": planned.get("max_retries"),
        "is_valid": planned.get("is_valid"),
        "validation_errors": planned.get("validation_errors", []),
        "has_anchor": bool(planned.get("anchor")),
        "has_replacement": planned.get("replacement") is not None,
    }
    return {
        "task_id": planned.get("id"),
        "target_path": planned.get("target_path"),
        "anchor": planned.get("anchor"),
        "replacement": planned.get("replacement"),
        "pointers": planned.get("pointers"),
        "occurrence_selector": planned.get("occurrence_selector"),
        "quantity": planned.get("quantity"),
        "copy_count": planned.get("copy_count"),
        "files": planned.get("files"),
        "source_path": planned.get("source_path"),
        "destination_path": planned.get("destination_path"),
        "paths": planned.get("paths"),
        "action_class": planned.get("action_class"),
        "depends_on": planned.get("depends_on"),
        "inputs_from": planned.get("inputs_from"),
        "timeout_seconds": planned.get("timeout_seconds"),
        "max_retries": planned.get("max_retries"),
        "edit_mode": planned.get("edit_mode") or state.get("edit_mode") or "anchor",
        "prompt": build_runtime_prompt(
            {
                **state,
                "target_path": planned_target,
                "anchor": planned.get("anchor"),
                "replacement": planned.get("replacement"),
                "pointers": planned.get("pointers"),
                "edit_mode": planned_mode,
                "context": context,
            }
        ),
        "context": context,
        "helper_output": helper_output,
        "proposal_summary": proposal_summary,
        "status": "planned",
    }


def _should_refine_preplanned_action(state: ShipyardState, preplanned: dict) -> bool:
    mode = str(preplanned.get("edit_mode") or "").strip()
    if mode not in REFINABLE_PREPLANNED_MODES:
        return False
    if not preplanned.get("target_path"):
        return False
    # For anchor/search_and_replace: refine if the anchor is stale (0) or ambiguous (>1),
    # OR if the file has pre-existing syntax errors (anchor edit would be blocked by
    # apply_edit's pre-syntax guard, so escalate to write_file now via plan_edit).
    if mode in {"anchor", "search_and_replace"}:
        anchor = preplanned.get("anchor") or preplanned.get("pattern")
        file_before = state.get("file_before") or ""
        if anchor and file_before:
            count = file_before.count(anchor)
            if count != 1:
                return True  # stale or ambiguous anchor
        # Escalate if the file is syntactically broken — apply_edit will block it anyway
        target = preplanned.get("target_path")
        if target and Path(str(target)).is_file() and _check_file_syntax(str(target)):
            return True
    # For other refinable modes, refine when we have explicit fresh file content
    return bool(state.get("file_before"))


def _refine_preplanned_action(state: ShipyardState, preplanned: dict) -> dict:
    target_path = preplanned.get("target_path") or state.get("target_path")
    # Sandbox relative paths to session workspace
    if target_path:
        sandboxed = _sandbox_target_path(target_path, state)
        if sandboxed:
            target_path = sandboxed
    current_file_before = state.get("file_before")
    if target_path:
        target = Path(str(target_path))
        if target.exists() and target.is_file():
            try:
                current_file_before = read_file(str(target))
            except OSError:
                pass
    context = dict(state.get("context", {}) or {})
    syntax_err = _check_file_syntax(str(Path(str(target_path)))) if target_path else None
    file_line_count = len((current_file_before or "").splitlines())
    planned_mode = preplanned.get("edit_mode") or state.get("edit_mode")

    if syntax_err and file_line_count <= 300:
        hint = (
            "The target file has syntax errors. Use write_file mode. "
            "Put the COMPLETE corrected file in replacement."
        )
    else:
        hint = (
            "Return the COMPLETE file with your changes applied. "
            "Preserve all existing content — only change what the instruction asks for."
        )

    # Guide write_file on valid existing files: prefer search_and_replace via hints.
    # The 30% content loss guard in apply_edit is the hard safety net — if the LLM
    # uses write_file and drops too much content, it gets blocked there.
    if (
        planned_mode == "write_file"
        and not syntax_err
        and current_file_before
        and file_line_count > 10
    ):
        hint += (
            "\nPrefer search_and_replace over write_file for targeted changes. "
            "Set anchor to the EXACT text to change (verbatim from the file) and replacement to the new text. "
            "write_file will be blocked if it loses >30% of existing content."
        )

    context["helper_notes"] = f"{context.get('helper_notes', '')}\n{hint}".strip()
    seeded_state = {
        **state,
        "target_path": target_path,
        "edit_mode": planned_mode,
        "anchor": preplanned.get("anchor"),
        "replacement": preplanned.get("replacement"),
        "pointers": preplanned.get("pointers"),
        "pattern": preplanned.get("pattern"),
        "command": preplanned.get("command"),
        "quantity": preplanned.get("quantity"),
        "copy_count": preplanned.get("copy_count"),
        "files": preplanned.get("files"),
        "occurrence_selector": preplanned.get("occurrence_selector"),
        "file_before": current_file_before,
        "context": context,
    }
    planned = propose_edit(seeded_state)
    planned.setdefault("target_path", preplanned.get("target_path"))
    planned.setdefault("target_path_source", preplanned.get("target_path_source"))
    planned.setdefault("provider", state.get("action_plan", {}).get("provider"))
    planned["provider_reason"] = (
        f"{planned.get('provider_reason') or ''} Refined from the current file/tool context."
    ).strip()
    # If the LLM switched from write_file to anchor during refinement, keep it —
    # the LLM knows the file contents and picked the appropriate mode.
    if not planned.get("validation_errors") and preplanned.get("validation_errors"):
        planned["validation_errors"] = list(preplanned.get("validation_errors") or [])
    if planned.get("is_valid") is None:
        planned["is_valid"] = preplanned.get("valid", True)
    return planned


def validate_proposal(state: ShipyardState) -> dict:
    proposal_summary = state.get("proposal_summary", {})
    if proposal_summary.get("is_valid"):
        return {"status": "proposal_valid"}

    errors = proposal_summary.get("validation_errors", [])
    message = "Invalid edit proposal."
    if errors:
        message = f"Invalid edit proposal. {' '.join(errors)}"
    return {
        "status": "invalid_proposal",
        "error": message,
    }


def check_edit_readiness(state: ShipyardState) -> dict:
    """Check edit readiness. For named_function mode, degrade to write_file
    (grep+cat is sufficient — no Code Graph RAG required)."""
    if state.get("edit_mode") != "named_function":
        return {
            "code_graph_status": {
                "ready": False,
                "available": False,
                "source": "skipped",
                "reason": "Code graph check skipped — not a named-function edit.",
            },
            "status": "ready_for_file_read",
        }

    # named_function mode: always degrade to write_file. The LLM will use
    # the file contents (loaded by fetch_step_context) to generate the edit.
    # This removes the hard dependency on Memgraph/Code Graph RAG.
    target_path = state.get("target_path")
    reason = "named_function degraded to write_file — code graph not required, using grep+cat approach."
    return {
        "code_graph_status": {
            "ready": False,
            "available": False,
            "source": "degraded",
            "reason": reason,
        },
        "edit_mode": "write_file",
        "status": "ready_for_file_read",
        "tool_outputs": list(state.get("tool_outputs") or []) + [reason],
    }


def collect_edit_context(state: ShipyardState) -> dict:
    """Collect edit context. For named_function mode, try grep-based function
    extraction from file contents (no Code Graph RAG required). Falls back
    gracefully if function can't be found."""
    helper_output = dict(state.get("helper_output", {}))

    if state.get("edit_mode") != "named_function":
        helper_output["edit_context"] = {
            "mode": state.get("edit_mode") or "anchor",
            "status": "skipped",
            "reason": "No named-function context collection required.",
        }
        return {
            "helper_output": helper_output,
            "status": "ready_for_file_read",
        }

    # For named_function: attempt to extract function source using the existing
    # get_function_source helper. If it fails, degrade gracefully — the LLM
    # already has the full file contents from fetch_step_context.
    target_path = state.get("target_path")
    function_name = state.get("context", {}).get("function_name")

    if not target_path or not function_name:
        helper_output["edit_context"] = {
            "mode": "named_function",
            "status": "skipped",
            "reason": "Missing target path or function name — LLM will use full file contents.",
        }
        return {
            "helper_output": helper_output,
            "status": "ready_for_file_read",
        }

    try:
        current_source = get_function_source(target_path, function_name)
    except (FunctionEditError, Exception):
        # Graceful degradation: function extraction failed, but the LLM has
        # the full file from fetch_step_context and can grep for the function.
        helper_output["edit_context"] = {
            "mode": "named_function",
            "function_name": function_name,
            "status": "degraded",
            "reason": f"Could not extract function '{function_name}' — using full file contents instead.",
        }
        return {
            "helper_output": helper_output,
            "status": "ready_for_file_read",
        }

    helper_output["edit_context"] = {
        "mode": "named_function",
        "function_name": function_name,
        "current_source": current_source,
        "line_count": len(current_source.splitlines()),
        "query_mode": "function_source_only",
    }
    code_graph_status = dict(state.get("code_graph_status", {}))
    code_graph_status["context_collected"] = True

    return {
        "current_function_source": current_source,
        "helper_output": helper_output,
        "code_graph_status": code_graph_status,
        "status": "ready_for_file_read",
    }


def read_target_file(state: ShipyardState) -> dict:
    target_path = state.get("target_path")
    if not target_path:
        return {"status": "no_target"}

    edit_mode = state.get("edit_mode")
    if edit_mode in {"list_files", "search_files", "run_command", "verify_command", "run_tests", "create_directory", "move_file", "rename_file", "read_many_files"}:
        return {"status": "file_read"}

    file_path = Path(target_path)
    if not file_path.exists() and state.get("edit_mode") in {"write_file", "append", "prepend", "create_files", "scaffold_files"}:
        return {
            "file_before": "",
            "target_existed_before_edit": False,
            "status": "file_read",
        }
    if edit_mode == "read_file" and not file_path.is_file():
        return {
            "status": "file_read",
        }

    return {
        "file_before": read_file(target_path),
        "target_existed_before_edit": file_path.exists(),
        "status": "file_read",
    }


def _get_allowed_roots(state: ShipyardState) -> list[Path] | None:
    """Compute the allowed filesystem roots for the current session.

    Returns None when there is no active session (no session_id), meaning
    bounds checking should be skipped.  In production the agent always runs
    with a session_id; None only happens in bare unit-test invocations.
    """
    if not state.get("session_id"):
        return None
    from .storage_paths import DATA_ROOT
    workspace = get_session_workspace(state.get("session_id")).resolve()
    repo_root = Path.cwd().resolve()
    data_root = DATA_ROOT.resolve()
    return [workspace, repo_root, data_root]


_FILE_REQUIRED_MODES = {
    "anchor", "named_function", "write_file", "append", "prepend",
    "search_and_replace", "rename_symbol", "delete_file", "copy_file",
    "update_imports",
}


def _sandbox_target_path(target_path: str | None, state: ShipyardState) -> str | None:
    """Resolve relative target paths to the session workspace.

    Prevents writes to CWD when the LLM returns a bare filename like 'app.js'.
    Absolute paths that already fall within allowed roots are left unchanged.
    """
    if not target_path:
        return target_path
    tp = Path(str(target_path))
    if tp.is_absolute():
        return target_path  # already resolved
    session_id = state.get("session_id")
    if not session_id:
        return target_path  # no session — bare unit tests
    workspace = get_session_workspace(session_id).resolve()
    repo_root = Path.cwd().resolve()
    # If workspace is external (not inside CWD), always sandbox there
    # This handles the rebuild case: workspace=/ship-rebuild, CWD=/shipyard
    try:
        workspace.relative_to(repo_root)
        is_external = False
    except ValueError:
        is_external = True
    if is_external:
        return str((workspace / tp).resolve())
    # For internal workspaces, prefer workspace if file exists there
    sandboxed = (workspace / tp).resolve()
    if sandboxed.exists():
        return str(sandboxed)
    # For new files in managed workspace, sandbox if parent exists there
    if sandboxed.parent.exists():
        return str(sandboxed)
    return target_path


def apply_edit(state: ShipyardState) -> dict:
    edit_mode = state.get("edit_mode") or ""
    target_path = state.get("target_path")

    # Sandbox relative paths to session workspace for mutation modes
    if edit_mode in _FILE_REQUIRED_MODES and target_path:
        sandboxed = _sandbox_target_path(target_path, state)
        if sandboxed and sandboxed != target_path:
            target_path = sandboxed
            state = {**state, "target_path": target_path}

    if edit_mode in _FILE_REQUIRED_MODES and target_path:
        tp = Path(str(target_path))
        if tp.exists() and tp.is_dir():
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": f"Target '{target_path}' is a directory — expected a file for {edit_mode}.",
            }

    # For anchor-based modes: if the file is already broken, block immediately.
    # plan_edit's _should_refine_preplanned_action detects syntax errors and
    # escalates to write_file there (before apply_edit runs), so this guard
    # only fires when the plan was not refined (e.g. direct heuristic proposals).
    if edit_mode in {"anchor", "search_and_replace"} and target_path:
        tp = Path(str(target_path))
        if tp.is_file():
            pre_syntax_error = _check_file_syntax(str(tp))
            if pre_syntax_error:
                return {
                    "edit_applied": False,
                    "status": "edit_blocked",
                    "error": (
                        f"Target file has pre-existing syntax errors — anchor editing is unsafe. "
                        f"Use write_file to rewrite the file from scratch.\n{pre_syntax_error}"
                    ),
                }

    # Also sandbox inspect modes so reads and writes are consistent
    if edit_mode in {"read_file", "read_many_files", "search_files", "list_files", "inspect_imports"} and target_path:
        sandboxed = _sandbox_target_path(target_path, state)
        if sandboxed and sandboxed != target_path:
            target_path = sandboxed
            state = {**state, "target_path": target_path}

    if state.get("edit_mode") == "list_files":
        target_path = state.get("target_path") or str(get_session_workspace(state.get("session_id")))
        root = Path(target_path)
        allowed = _get_allowed_roots(state)
        if err := _check_bounds(root, allowed, "list_files target"):
            return {"edit_applied": False, "status": "edit_blocked", "error": err}
        if not root.exists():
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": "Target path was not found for list_files.",
            }
        items = sorted(
            str(path.relative_to(root)) if path != root else "."
            for path in root.rglob("*")
            if path.is_file()
        )[:200]
        return {
            "edit_applied": False,
            "status": "observed",
            "tool_output": {"tool": "list_files", "target_path": str(root), "files": items},
            "no_op": True,
        }

    if state.get("edit_mode") == "read_file":
        target_path = state.get("target_path")
        if not target_path:
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": "Target file was not found for read_file.",
            }
        target = Path(target_path)
        if err := _check_bounds(target, _get_allowed_roots(state), "read_file target"):
            return {"edit_applied": False, "status": "edit_blocked", "error": err}
        if target.exists() and not target.is_file():
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": "Target file was not found for read_file.",
            }
        if not target.exists():
            return {
                "edit_applied": False,
                "status": "observed",
                "tool_output": {"tool": "read_file", "target_path": target_path, "missing": True, "content": ""},
                "no_op": True,
            }
        content = read_file(target_path)
        # Cache file content so subsequent write steps don't re-read from disk
        existing_cache = dict(state.get("file_content_cache") or {})
        existing_cache[str(target_path)] = content
        return {
            "edit_applied": False,
            "status": "observed",
            "file_before": content,
            "file_content_cache": existing_cache,
            "tool_output": {"tool": "read_file", "target_path": target_path, "content": content},
            "no_op": True,
        }

    if state.get("edit_mode") == "read_many_files":
        allowed = _get_allowed_roots(state)
        paths_to_read = list(state.get("paths", []) or [])

        # Fallback: if no explicit paths are provided but target_path is a directory,
        # auto-populate by reading all relevant files from that directory (up to 10 files).
        if not paths_to_read:
            _dir_target = state.get("target_path")
            if _dir_target:
                _dir = Path(str(_dir_target))
                if _dir.is_dir():
                    _IGNORED = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "dist", "build", ".shipyard"}
                    _READABLE_SUFFIXES = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
                    for _p in sorted(_dir.rglob("*")):
                        if any(part in _IGNORED for part in _p.relative_to(_dir).parts):
                            continue
                        if _p.is_file() and _p.suffix.lower() in _READABLE_SUFFIXES:
                            paths_to_read.append(str(_p))
                            if len(paths_to_read) >= 10:
                                break

        files = []
        for path in paths_to_read:
            p = Path(str(path))
            if _check_bounds(p, allowed, "read_many_files path"):
                continue
            if p.is_file():
                files.append({"path": str(path), "content": read_file(str(path))})
        return {
            "edit_applied": False,
            "status": "observed",
            "tool_output": {"tool": "read_many_files", "files": files},
            "no_op": True,
        }

    if state.get("edit_mode") == "search_files":
        target_path = state.get("target_path") or str(get_session_workspace(state.get("session_id")))
        pattern = state.get("pattern")
        root = Path(target_path)
        if err := _check_bounds(root, _get_allowed_roots(state), "search_files target"):
            return {"edit_applied": False, "status": "edit_blocked", "error": err}
        if not pattern:
            return {
                "edit_applied": False,
                "status": "awaiting_edit_spec",
                "error": "Missing pattern for search_files.",
            }
        matches: list[dict[str, str | int]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "text": line[:200],
                        }
                    )
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break
        return {
            "edit_applied": False,
            "status": "observed",
            "tool_output": {"tool": "search_files", "target_path": str(root), "pattern": pattern, "matches": matches},
            "no_op": True,
        }

    if state.get("edit_mode") == "inspect_imports":
        target_path = state.get("target_path")
        if not target_path or not Path(target_path).is_file():
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": "Target file was not found for inspect_imports.",
            }
        imports = []
        for line in read_file(target_path).splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        return {
            "edit_applied": False,
            "status": "observed",
            "tool_output": {"tool": "inspect_imports", "target_path": target_path, "imports": imports},
            "no_op": True,
        }

    if state.get("edit_mode") in {"run_command", "verify_command", "run_tests"}:
        command = str(state.get("command") or "").strip()
        if not command:
            return {
                "edit_applied": False,
                "status": "awaiting_edit_spec",
                "error": f"Missing command for {state.get('edit_mode')}.",
            }
        # Block only truly dangerous constructs: null bytes and command substitution.
        # Pipes, redirects, and chaining (&&, ||, ;) are allowed — the command runs
        # inside the session workspace sandbox with shell=True when needed.
        _INJECT = ("\x00", "$(", "`")
        if any(token in command for token in _INJECT):
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": "Command injection syntax is not allowed.",
            }
        cwd = get_session_workspace(state.get("session_id"))
        # Use shell=True if the command uses shell features (pipes, redirects, chaining)
        _SHELL_FEATURES = ("|", ">", "<", "&&", "||", ";", "2>&1")
        use_shell = any(token in command for token in _SHELL_FEATURES)
        process = subprocess.Popen(
            command if use_shell else shlex.split(command),
            shell=use_shell,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cancel_check = state.get("cancel_check")
        deadline = time.monotonic() + int(state.get("timeout_seconds") or 60)
        stdout = ""
        stderr = ""
        returncode = None
        while True:
            if callable(cancel_check) and cancel_check():
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                return {
                    "edit_applied": False,
                    "status": "cancelled",
                    "tool_output": {
                        "tool": state.get("edit_mode"),
                        "command": command,
                        "cwd": str(cwd),
                        "stdout": stdout[:4000],
                        "stderr": stderr[:4000],
                    },
                    "no_op": True,
                    "error": "Run cancelled.",
                }
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                return {
                    "edit_applied": False,
                    "status": "verification_failed" if state.get("edit_mode") in {"verify_command", "run_tests"} else "failed",
                    "tool_output": {
                        "tool": state.get("edit_mode"),
                        "command": command,
                        "cwd": str(cwd),
                        "stdout": stdout[:4000],
                        "stderr": stderr[:4000],
                    },
                    "no_op": True,
                    "error": f"Command timed out after {int(state.get('timeout_seconds') or 20)} seconds.",
                }
            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                break
            time.sleep(0.1)
        is_verification = state.get("edit_mode") in {"verify_command", "run_tests"}
        return {
            "edit_applied": False,
            "status": ("verified" if returncode == 0 else "verification_failed") if is_verification else ("observed" if returncode == 0 else "verification_failed"),
            "tool_output": {
                "tool": state.get("edit_mode"),
                "command": command,
                "cwd": str(cwd),
                "returncode": returncode,
                "stdout": stdout[:4000],
                "stderr": stderr[:4000],
            },
            "no_op": True,
            "error": None if returncode == 0 else f"Command failed with exit code {returncode}.",
        }

    if state.get("edit_mode") == "create_directory":
        target_path = state.get("target_path")
        if not target_path:
            return {"edit_applied": False, "status": "awaiting_edit_spec", "error": "Missing target path for create_directory."}
        directory = Path(target_path)
        if err := _check_bounds(directory, _get_allowed_roots(state), "create_directory target"):
            return {"edit_applied": False, "status": "edit_blocked", "error": err}
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "edit_applied": True,
            "status": "edited",
            "changed_files": [str(directory.resolve())],
            "target_path": str(directory.resolve()),
            "no_op": False,
        }

    if state.get("edit_mode") in {"move_file", "rename_file"}:
        proposal_summary = dict(state.get("proposal_summary", {}) or {})
        preplanned_action = dict(state.get("preplanned_action", {}) or {})
        source = state.get("source_path") or proposal_summary.get("source_path") or preplanned_action.get("source_path")
        destination = (
            state.get("destination_path")
            or proposal_summary.get("destination_path")
            or preplanned_action.get("destination_path")
        )
        if not source or not destination:
            return {"edit_applied": False, "status": "awaiting_edit_spec", "error": "Missing source or destination path."}
        src = Path(str(source))
        dst = Path(str(destination))
        allowed = _get_allowed_roots(state)
        if (err := _check_bounds(src, allowed, "move_file/rename_file source")) or \
           (err := _check_bounds(dst, allowed, "move_file/rename_file destination")):
            return {"edit_applied": False, "status": "edit_blocked", "error": err}
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {
            "edit_applied": True,
            "status": "edited",
            "changed_files": [str(dst.resolve())],
            "target_path": str(dst.resolve()),
            "no_op": False,
        }

    if state.get("edit_mode") == "search_and_replace":
        target_path = state.get("target_path")
        pattern = state.get("pattern") or state.get("anchor")
        replacement = state.get("replacement")
        if not target_path or pattern is None or replacement is None:
            return {"edit_applied": False, "status": "awaiting_edit_spec", "error": "Missing search-and-replace details."}
        file_path = Path(target_path)
        snapshot_path = snapshot_file(target_path)
        current_content = file_path.read_text(encoding="utf-8") if file_path.is_file() else ""
        updated_content = current_content.replace(str(pattern), str(replacement))
        file_path.write_text(updated_content, encoding="utf-8")
        changed = updated_content != current_content
        return {
            "edit_applied": changed,
            "edit_attempts": state.get("edit_attempts", 0) + 1,
            "snapshot_path": snapshot_path,
            "changed_files": [str(file_path.resolve())] if changed else [],
            "no_op": not changed,
            "status": "edited",
        }

    if state.get("edit_mode") == "named_function":
        target_path = state.get("target_path")
        function_name = state.get("context", {}).get("function_name")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1
        code_graph_status = dict(state.get("code_graph_status", {}))

        if not target_path or not function_name or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, function name, or replacement.",
            }

        snapshot_path = snapshot_file(target_path)
        try:
            apply_function_edit(target_path, function_name, replacement)
        except FunctionEditError as exc:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "snapshot_path": snapshot_path,
                "status": "edit_blocked",
                "error": str(exc),
            }

        index_state = dict(code_graph_status.get("index_state", {}))
        index_state["stale"] = True
        code_graph_status["index_state"] = index_state
        code_graph_status["refresh_required"] = True
        code_graph_status["reason"] = "Graph-backed function edit succeeded; re-index to refresh graph state."

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "code_graph_status": code_graph_status,
            "status": "edited",
        }

    if state.get("edit_mode") == "delete_file":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1

        if not target_path:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path for delete mode.",
            }

        file_path = Path(target_path)
        if not file_path.exists():
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "edit_blocked",
                "error": "Target file was not found for deletion.",
            }

        snapshot_path = snapshot_file(target_path)
        file_path.unlink()
        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "status": "edited",
        }

    if state.get("edit_mode") == "copy_file":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1
        copy_count = state.get("copy_count")

        if not target_path or copy_count is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path or copy count for copy mode.",
            }

        source_path = Path(target_path)
        if not source_path.exists():
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "edit_blocked",
                "error": "Target file was not found for copy mode.",
            }

        count = int(copy_count)
        created_files = []
        snapshot_path = snapshot_file(target_path)
        for index in range(1, count + 1):
            copy_path = _build_copy_path(source_path, index)
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, copy_path)
            created_files.append(str(copy_path.resolve()))

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "changed_files": created_files,
            "status": "edited",
        }

    if state.get("edit_mode") == "create_files":
        target_path = state.get("target_path")
        edit_attempts = state.get("edit_attempts", 0) + 1
        quantity = state.get("quantity")
        replacement = state.get("replacement")

        if not target_path or quantity is None or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, quantity, or content for create_files mode.",
            }

        count = int(quantity)
        first_path = Path(target_path)
        first_path.parent.mkdir(parents=True, exist_ok=True)
        created_files = []
        focused_index = _extract_numbered_file_index(first_path)

        for index in range(1, count + 1):
            file_path = _build_numbered_batch_path(first_path, index) if focused_index is not None else (
                first_path if index == 1 else _build_copy_path(first_path, index)
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = replacement if focused_index is None or index == focused_index else ""
            file_path.write_text(content, encoding="utf-8")
            created_files.append(str(file_path.resolve()))

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "changed_files": created_files,
            "target_existed_before_edit": False,
            "status": "edited",
        }

    if state.get("edit_mode") == "scaffold_files":
        edit_attempts = state.get("edit_attempts", 0) + 1
        files = state.get("files")

        if not isinstance(files, list) or not files:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing files for scaffold_files mode.",
            }

        created_files = []
        for file_spec in files:
            raw_path = str(file_spec.get("path") or "")
            if not raw_path:
                continue
            # Use sandbox to resolve relative paths to workspace
            resolved_path = _sandbox_target_path(raw_path, state) or raw_path
            file_path = Path(resolved_path)
            if file_path.exists():
                return {
                    "edit_applied": False,
                    "edit_attempts": edit_attempts,
                    "status": "edit_blocked",
                    "error": f"scaffold_files would overwrite existing file: {file_path.name}",
                }
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(str(file_spec.get("content", "")), encoding="utf-8")
            created_files.append(str(file_path.resolve()))

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "changed_files": created_files,
            "target_existed_before_edit": False,
            "status": "edited",
        }

    if state.get("edit_mode") == "rename_symbol":
        target_path = state.get("target_path")
        anchor = state.get("anchor")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1

        if not target_path or not anchor or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path, symbol, or replacement for rename mode.",
            }

        snapshot_path = snapshot_file(target_path)
        try:
            apply_symbol_rename(target_path, anchor, replacement)
        except AnchorEditError as exc:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "snapshot_path": snapshot_path,
                "status": "edit_blocked",
                "error": str(exc),
            }

        return {
            "edit_applied": True,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "status": "edited",
        }

    if state.get("edit_mode") == "rename_symbol_global":
        old_name = state.get("anchor")
        new_name = state.get("replacement")
        root_path = state.get("target_path") or str(get_session_workspace(state.get("session_id")))
        if not old_name or new_name is None:
            return {"edit_applied": False, "status": "awaiting_edit_spec",
                    "error": "rename_symbol_global requires anchor (old name) and replacement (new name)."}
        root = Path(str(root_path))
        if not state.get("wide_impact_approved"):
            affected = _count_affected_files(root, str(old_name))
            if affected > _WIDE_IMPACT_THRESHOLD:
                gate = make_human_gate(
                    reason=f"rename_symbol_global would modify {affected} files.",
                    action="approve_wide_impact",
                    details={"file_count": affected, "old_name": old_name, "new_name": new_name},
                )
                return {"status": "needs_approval", "human_gate": gate, "no_op": True, "edit_applied": False}
        # Try rope for scope-aware renaming; fall back to text-walk on failure
        from .tools.symbol_tools import rename_symbol_rope
        rope_result = rename_symbol_rope(str(root), str(state.get("target_path") or root), str(old_name), str(new_name))
        if rope_result["error"] is None:
            changed = rope_result["changed_files"]
            return {
                "edit_applied": bool(changed),
                "status": "edited" if changed else "observed",
                "changed_files": changed,
                "no_op": not changed,
                "tool_output": {"tool": "rename_symbol_global", "method": "rope", "old_name": old_name, "new_name": new_name, "changed_files": changed},
            }
        # rope failed — fall back to text-walk
        changed = []
        snapshot_paths: list[str] = []
        for file_path in sorted(root.rglob("*.py")):
            if any(part in {".git", ".venv", "__pycache__"} for part in file_path.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if str(old_name) not in content:
                continue
            snap = snapshot_file(str(file_path))
            snapshot_paths.append(snap)
            try:
                apply_symbol_rename(str(file_path), str(old_name), str(new_name))
                changed.append(str(file_path.resolve()))
            except AnchorEditError:
                pass
        return {
            "edit_applied": bool(changed),
            "status": "edited" if changed else "observed",
            "changed_files": changed,
            "no_op": not changed,
            "tool_output": {"tool": "rename_symbol_global", "method": "text_walk", "rope_error": rope_result["error"], "old_name": old_name, "new_name": new_name, "changed_files": changed},
        }

    if state.get("edit_mode") == "update_imports":
        old_import = state.get("anchor")
        new_import = state.get("replacement")
        root_path = state.get("target_path") or str(get_session_workspace(state.get("session_id")))
        if not old_import or new_import is None:
            return {"edit_applied": False, "status": "awaiting_edit_spec",
                    "error": "update_imports requires anchor (old import path) and replacement (new import path)."}
        root = Path(str(root_path))
        if not state.get("wide_impact_approved"):
            affected = _count_affected_files(root, str(old_import))
            if affected > _WIDE_IMPACT_THRESHOLD:
                gate = make_human_gate(
                    reason=f"update_imports would modify {affected} files.",
                    action="approve_wide_impact",
                    details={"file_count": affected, "old_import": old_import, "new_import": new_import},
                )
                return {"status": "needs_approval", "human_gate": gate, "no_op": True, "edit_applied": False}
        changed: list[str] = []
        for file_path in sorted(root.rglob("*.py")):
            if any(part in {".git", ".venv", "__pycache__"} for part in file_path.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if str(old_import) not in content:
                continue
            updated = content.replace(str(old_import), str(new_import))
            if updated == content:
                continue
            snapshot_file(str(file_path))
            file_path.write_text(updated, encoding="utf-8")
            changed.append(str(file_path.resolve()))
        return {
            "edit_applied": bool(changed),
            "status": "edited" if changed else "observed",
            "changed_files": changed,
            "no_op": not changed,
            "tool_output": {"tool": "update_imports", "old_import": old_import, "new_import": new_import, "changed_files": changed},
        }

    if state.get("edit_mode") == "search_then_edit":
        from .action_planner import plan_edits_for_matches
        target_path = state.get("target_path") or str(get_session_workspace(state.get("session_id")))
        pattern = state.get("pattern")
        if not pattern:
            return {"edit_applied": False, "status": "awaiting_edit_spec",
                    "error": "search_then_edit requires a pattern."}
        root = Path(str(target_path))
        matches: list[dict] = []
        for file_path in sorted(root.rglob("*")):
            if any(part in {".git", ".venv", "__pycache__"} for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_num, line in enumerate(content.splitlines(), start=1):
                if str(pattern) in line:
                    matches.append({"path": str(file_path), "line": line_num, "text": line[:200]})
            if len(matches) >= 200:
                break
        if not matches:
            return {
                "edit_applied": False, "status": "observed", "no_op": True,
                "tool_output": {"tool": "search_then_edit", "pattern": pattern, "matches": []},
            }
        # Ask the LLM to generate one edit per matched file, inject as sub-actions
        sub_actions = plan_edits_for_matches(state, matches)
        return {
            "edit_applied": False, "status": "observed", "no_op": True,
            "expand_to": sub_actions,
            "tool_output": {"tool": "search_then_edit", "pattern": pattern, "matches": matches, "expanded_actions": len(sub_actions)},
        }

    if state.get("edit_mode") == "synthesize_tool":
        from .tool_registry import get_registry, validate_tool_source
        tool_name = str(state.get("tool_name") or "").strip()
        tool_source = str(state.get("tool_source") or "").strip()
        if not tool_name or not tool_source:
            return {
                "edit_applied": False,
                "status": "awaiting_edit_spec",
                "error": "synthesize_tool requires tool_name and tool_source.",
            }
        # Layer 1: static analysis before touching disk
        errors = validate_tool_source(tool_name, tool_source)
        if errors:
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": f"Tool source failed validation: {'; '.join(errors)}",
                "no_op": True,
            }
        # Layer 4: human gate — always require approval before writing new executable code
        if not state.get("wide_impact_approved"):
            return {
                "edit_applied": False,
                "status": "needs_approval",
                "no_op": True,
            }
        result = get_registry().synthesize(tool_name, tool_source)
        if not result["ok"]:
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": f"Tool synthesis failed: {'; '.join(result.get('errors', []))}",
                "no_op": True,
            }
        return {
            "edit_applied": True,
            "status": "edited",
            "changed_files": [result["path"]],
            "tool_output": {"tool": "synthesize_tool", "tool_name": tool_name, "path": result["path"]},
            "no_op": False,
        }

    if state.get("edit_mode") == "invoke_tool":
        from .tool_registry import get_registry
        tool_name = str(state.get("tool_name") or "").strip()
        tool_args = state.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}
        if not tool_name:
            return {
                "edit_applied": False,
                "status": "awaiting_edit_spec",
                "error": "invoke_tool requires tool_name.",
            }
        timeout = float(state.get("timeout_seconds") or 10)
        # Layers 2+3: inspect-only contract + thread timeout enforced inside invoke()
        result = get_registry().invoke(tool_name, tool_args, timeout=timeout)
        if result.get("error"):
            return {
                "edit_applied": False,
                "status": "edit_blocked",
                "error": f"Tool {tool_name!r} error: {result['error']}",
                "tool_output": {"tool": "invoke_tool", "tool_name": tool_name, "args": tool_args, **result},
                "no_op": True,
            }
        return {
            "edit_applied": False,
            "status": "observed",
            "tool_result": result,
            "tool_output": {"tool": "invoke_tool", "tool_name": tool_name, "args": tool_args, **result},
            "no_op": True,
        }

    if state.get("edit_mode") in {"write_file", "append", "prepend"}:
        target_path = state.get("target_path")
        replacement = state.get("replacement")
        edit_attempts = state.get("edit_attempts", 0) + 1
        target_existed = state.get("target_existed_before_edit", True)

        if not target_path or replacement is None:
            return {
                "edit_applied": False,
                "edit_attempts": edit_attempts,
                "status": "awaiting_edit_spec",
                "error": "Missing target path or content for the requested edit mode.",
            }

        snapshot_path = snapshot_file(target_path)
        file_path = Path(target_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        current_content = file_path.read_text(encoding="utf-8") if file_path.is_file() else ""
        updated_content = current_content

        if state.get("edit_mode") == "write_file":
            # Safety: block write_file if it would lose >30% of existing content.
            # This catches the LLM regenerating a stripped-down version of the file.
            if current_content and len(replacement) < len(current_content) * 0.7:
                lost_pct = round((1 - len(replacement) / len(current_content)) * 100)
                return {
                    "edit_applied": False,
                    "edit_attempts": edit_attempts,
                    "status": "edit_blocked",
                    "error": (
                        f"write_file would lose ~{lost_pct}% of existing content "
                        f"({len(current_content)} → {len(replacement)} chars). "
                        "Use search_and_replace to make targeted changes instead."
                    ),
                }
            updated_content = replacement
        elif state.get("edit_mode") == "append":
            updated_content = current_content + replacement
        elif state.get("edit_mode") == "prepend":
            updated_content = replacement + current_content

        file_path.write_text(updated_content, encoding="utf-8")
        changed = updated_content != current_content
        return {
            "edit_applied": changed,
            "edit_attempts": edit_attempts,
            "snapshot_path": snapshot_path,
            "target_existed_before_edit": target_existed,
            "changed_files": [str(file_path.resolve())] if changed else [],
            "no_op": not changed,
            "status": "edited",
        }

    target_path = state.get("target_path")
    anchor = state.get("anchor")
    replacement = state.get("replacement")
    pointers = state.get("pointers")
    occurrence_selector = state.get("occurrence_selector") or parse_occurrence_selector(
        state.get("instruction", "")
    )
    file_before = state.get("file_before", "")
    edit_attempts = state.get("edit_attempts", 0) + 1

    if not target_path or replacement is None or (not anchor and not pointers):
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "awaiting_edit_spec",
            "error": "Missing target path, anchor or pointers, or replacement.",
        }

    try:
        if pointers:
            validate_pointer_edits(file_before, pointers, anchor)
        else:
            validate_anchor_edit(file_before, anchor, occurrence_selector)
    except AnchorEditError as exc:
        if pointers and anchor:
            exact_pointers = find_anchor_pointers(file_before, anchor)
            if len(exact_pointers) == 1:
                snapshot_path = snapshot_file(target_path)
                updated = apply_pointer_edits(
                    target_path,
                    exact_pointers,
                    replacement,
                    anchor,
                )
                changed = updated != file_before
                return {
                    "edit_applied": changed,
                    "edit_attempts": edit_attempts,
                    "snapshot_path": snapshot_path,
                    "changed_files": [str(Path(target_path).resolve())] if changed else [],
                    "no_op": not changed,
                    "status": "edited",
                    "pointers": exact_pointers,
                    "proposal_summary": {
                        **dict(state.get("proposal_summary", {}) or {}),
                        "pointers": exact_pointers,
                        "repair_reason": "Auto-corrected a single exact anchor span after invalid pointers.",
                    },
                }
        if not pointers and "multiple locations" in str(exc):
            repaired = _repair_anchor_with_pointers(state, str(exc))
            if repaired:
                repaired_state = {
                    **state,
                    **repaired,
                }
                try:
                    validate_pointer_edits(
                        file_before,
                        repaired_state.get("pointers", []),
                        repaired_state.get("anchor"),
                    )
                    snapshot_path = snapshot_file(target_path)
                    updated = apply_pointer_edits(
                        target_path,
                        repaired_state.get("pointers", []),
                        repaired_state.get("replacement"),
                        repaired_state.get("anchor"),
                    )
                    changed = updated != file_before
                    return {
                        "edit_applied": changed,
                        "edit_attempts": edit_attempts,
                        "snapshot_path": snapshot_path,
                        "changed_files": [str(Path(target_path).resolve())] if changed else [],
                        "no_op": not changed,
                        "status": "edited",
                        "pointers": repaired_state.get("pointers"),
                        "proposal_summary": repaired.get("proposal_summary"),
                        "context": repaired.get("context"),
                    }
                except AnchorEditError:
                    exact_pointers = find_anchor_pointers(file_before, repaired_state.get("anchor") or "")
                    if len(exact_pointers) == 1:
                        snapshot_path = snapshot_file(target_path)
                        updated = apply_pointer_edits(
                            target_path,
                            exact_pointers,
                            repaired_state.get("replacement"),
                            repaired_state.get("anchor"),
                        )
                        changed = updated != file_before
                        return {
                            "edit_applied": changed,
                            "edit_attempts": edit_attempts,
                            "snapshot_path": snapshot_path,
                            "changed_files": [str(Path(target_path).resolve())] if changed else [],
                            "no_op": not changed,
                            "status": "edited",
                            "pointers": exact_pointers,
                            "proposal_summary": {
                                **dict(state.get("proposal_summary", {}) or {}),
                                "pointers": exact_pointers,
                                "repair_reason": "Auto-corrected a single exact anchor span after invalid pointer repair.",
                            },
                            "context": repaired.get("context"),
                        }
        replanned = _replan_mutate_step_from_current_file(state, str(exc))
        if replanned:
            repaired_state = {
                **state,
                **replanned,
            }
            repaired_mode = repaired_state.get("edit_mode")
            repaired_anchor = repaired_state.get("anchor")
            repaired_replacement = repaired_state.get("replacement")
            repaired_pointers = repaired_state.get("pointers")
            repaired_file_before = repaired_state.get("file_before", file_before)
            try:
                if repaired_mode == "write_file" and repaired_replacement is not None:
                    snapshot_path = snapshot_file(target_path)
                    file_path = Path(str(target_path))
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(str(repaired_replacement), encoding="utf-8")
                    updated = str(repaired_replacement)
                    changed = updated != repaired_file_before
                    return {
                        "edit_applied": changed,
                        "edit_attempts": edit_attempts,
                        "snapshot_path": snapshot_path,
                        "changed_files": [str(file_path.resolve())] if changed else [],
                        "no_op": not changed,
                        "status": "edited",
                        "proposal_summary": replanned.get("proposal_summary"),
                        "context": replanned.get("context"),
                    }
                if repaired_mode == "rename_symbol" and repaired_anchor and repaired_replacement is not None:
                    snapshot_path = snapshot_file(target_path)
                    apply_symbol_rename(target_path, repaired_anchor, repaired_replacement)
                    updated = read_file(target_path)
                    changed = updated != repaired_file_before
                    return {
                        "edit_applied": changed,
                        "edit_attempts": edit_attempts,
                        "snapshot_path": snapshot_path,
                        "changed_files": [str(Path(target_path).resolve())] if changed else [],
                        "no_op": not changed,
                        "status": "edited",
                        "proposal_summary": replanned.get("proposal_summary"),
                        "context": replanned.get("context"),
                    }
                if repaired_pointers:
                    validate_pointer_edits(repaired_file_before, repaired_pointers, repaired_anchor)
                    snapshot_path = snapshot_file(target_path)
                    updated = apply_pointer_edits(
                        target_path,
                        repaired_pointers,
                        repaired_replacement,
                        repaired_anchor,
                    )
                    changed = updated != repaired_file_before
                    return {
                        "edit_applied": changed,
                        "edit_attempts": edit_attempts,
                        "snapshot_path": snapshot_path,
                        "changed_files": [str(Path(target_path).resolve())] if changed else [],
                        "no_op": not changed,
                        "status": "edited",
                        "pointers": repaired_pointers,
                        "proposal_summary": replanned.get("proposal_summary"),
                        "context": replanned.get("context"),
                    }
                if repaired_anchor and repaired_replacement is not None:
                    validate_anchor_edit(repaired_file_before, repaired_anchor, repaired_state.get("occurrence_selector"))
                    snapshot_path = snapshot_file(target_path)
                    updated = apply_anchor_edit(
                        target_path,
                        repaired_anchor,
                        repaired_replacement,
                        repaired_state.get("occurrence_selector"),
                    )
                    changed = updated != repaired_file_before
                    return {
                        "edit_applied": changed,
                        "edit_attempts": edit_attempts,
                        "snapshot_path": snapshot_path,
                        "changed_files": [str(Path(target_path).resolve())] if changed else [],
                        "no_op": not changed,
                        "status": "edited",
                        "proposal_summary": replanned.get("proposal_summary"),
                        "context": replanned.get("context"),
                    }
            except AnchorEditError:
                pass
        return {
            "edit_applied": False,
            "edit_attempts": edit_attempts,
            "status": "edit_blocked",
            "error": str(exc),
        }

    snapshot_path = snapshot_file(target_path)
    if pointers:
        updated = apply_pointer_edits(target_path, pointers, replacement, anchor)
    else:
        updated = apply_anchor_edit(target_path, anchor, replacement, occurrence_selector)
    changed = updated != file_before

    return {
        "edit_applied": changed,
        "edit_attempts": edit_attempts,
        "snapshot_path": snapshot_path,
        "changed_files": [str(Path(target_path).resolve())] if changed else [],
        "no_op": not changed,
        "status": "edited",
    }


_MUTATION_EDIT_MODES = {
    "anchor", "write_file", "search_and_replace", "named_function",
    "append", "prepend", "create_files", "scaffold_files",
    "rename_symbol_global", "update_imports",
}


def _detect_auto_verification_commands(state: ShipyardState) -> list[str]:
    """
    Auto-detect a fast syntax check after a mutation to a .py file.

    Only runs `ruff check` — fast and non-flaky.  pytest is intentionally
    excluded: it can take tens of seconds, pick up pre-existing failures
    unrelated to the current edit, and trigger unnecessary retry loops.
    Users who want test verification should supply explicit
    `verification_commands` in their request.
    """
    edit_mode = state.get("edit_mode") or ""
    if edit_mode not in _MUTATION_EDIT_MODES:
        return []
    target = state.get("target_path") or ""
    if not target or not Path(str(target)).exists():
        return []
    target_str = str(target)
    if target_str.endswith(".py"):
        if shutil.which("ruff"):
            return [f"ruff check {shlex.quote(target_str)} --select=E,F,W --quiet"]
    elif target_str.endswith(".js") or target_str.endswith(".mjs") or target_str.endswith(".cjs"):
        if shutil.which("node"):
            return [f"node --check {shlex.quote(target_str)}"]
    elif target_str.endswith(".ts") or target_str.endswith(".tsx") or target_str.endswith(".mts"):
        if shutil.which("tsc"):
            return [
                f"tsc --noEmit --allowJs --skipLibCheck --isolatedModules --noResolve {shlex.quote(target_str)}"
            ]
    return []


def verify_edit(state: ShipyardState) -> dict:
    # Skip verification when apply_edit was a no-op (content unchanged).
    # Running syntax checks on unchanged files produces misleading "passed" results.
    if state.get("no_op") and not state.get("edit_applied"):
        return {
            "verification_results": [],
            "verification_retry_count": int(state.get("verification_retry_count") or 0),
            "status": state.get("status", "edited"),
        }
    commands = list(state.get("verification_commands") or [])
    if not commands:
        commands = _detect_auto_verification_commands(state)
    if not commands:
        return {
            "verification_results": [],
            "verification_retry_count": int(state.get("verification_retry_count") or 0),
            "status": state.get("status", "edited"),
        }

    results = run_verification(commands)
    has_failure = any(result["returncode"] != 0 for result in results)
    retry_count = int(state.get("verification_retry_count") or 0)

    return {
        "verification_results": results,
        "verification_retry_count": retry_count,
        "status": "verification_failed" if has_failure else "verified",
    }


def recover_or_finish(state: ShipyardState) -> dict:
    if state.get("status") != "verification_failed":
        return {}

    # Revert only the CURRENT step's edit using its own snapshot.
    # Prior steps' file_transactions must NOT be used here — they accumulate
    # across the whole run, and reverting them would undo earlier successful
    # edits every time a later step fails verification.
    target_path = state.get("target_path")
    snapshot_path = state.get("snapshot_path")
    target_existed = state.get("target_existed_before_edit", True)

    reverted_files: list[str] = []
    if target_path and snapshot_path:
        target = Path(str(target_path))
        if not target_existed and target.exists():
            target.unlink()
            reverted_files.append(str(target.resolve()))
        elif Path(str(snapshot_path)).exists():
            revert_file(str(target_path), str(snapshot_path))
            reverted_files.append(str(target.resolve()))

    attempts = state.get("edit_attempts", 0)
    max_attempts = state.get("max_edit_attempts", 2)
    verification_results = state.get("verification_results", [])
    failure_messages = [
        result.get("stderr") or result.get("stdout") or ""
        for result in verification_results
        if result.get("returncode") != 0
    ]
    failure_text = "\n".join(message for message in failure_messages if message).strip()

    if attempts < max_attempts:
        context = dict(state.get("context", {}))
        if failure_text:
            context["test_failure"] = failure_text
            context["helper_notes"] = (
                f"Verification failed previously. Use the exact current files and fix the issue shown here:\n{failure_text}"
            )
        return {
            "context": context,
            "reverted_to_snapshot": True,
            "reverted_files": reverted_files,
            "revert_count": len(reverted_files),
            "status": "retry_ready",
            "error": "Verification failed. File reverted to latest snapshot.",
        }

    failure_summary = failure_text[:300] if failure_text else "Verification failed after maximum retry attempts."
    return {
        "reverted_to_snapshot": True,
        "reverted_files": reverted_files,
        "revert_count": len(reverted_files),
        "status": "failed_after_retries",
        "error": failure_summary,
    }


def should_retry(state: ShipyardState) -> str:
    if state.get("status") == "retry_ready":
        return "retry"
    return "done"


def should_continue_after_readiness(state: ShipyardState) -> str:
    if state.get("status") == "ready_for_file_read":
        return "continue"
    return "done"


def should_continue_after_proposal_validation(state: ShipyardState) -> str:
    if state.get("status") == "proposal_valid":
        return "continue"
    return "done"


def _build_copy_path(source_path: Path, index: int) -> Path:
    suffix = source_path.suffix
    stem = source_path.stem
    return source_path.with_name(f"{stem}_copy_{index}{suffix}")


def _extract_numbered_file_index(path: Path) -> int | None:
    match = re.search(r"^file(?P<index>\d+)$", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group("index"))


def _build_numbered_batch_path(source_path: Path, index: int) -> Path:
    return source_path.with_name(f"file{index}{source_path.suffix}")


def build_graph():
    graph = StateGraph(ShipyardState)
    graph.add_node("seed_defaults", seed_defaults)
    # fetch_step_context reads the target file + context_files BEFORE plan_edit so the
    # LLM generates anchors/pointers against live file content, not a blind guess.
    # It also runs on retry so the LLM sees post-revert state.
    graph.add_node("fetch_step_context", fetch_step_context)
    graph.add_node("prepare_prompt", prepare_prompt)
    graph.add_node("consult_helper_agent", consult_helper_agent)
    graph.add_node("plan_edit", plan_edit)
    graph.add_node("validate_proposal", validate_proposal)
    graph.add_node("check_edit_readiness", check_edit_readiness)
    graph.add_node("collect_edit_context", collect_edit_context)
    graph.add_node("apply_edit", apply_edit)
    graph.add_node("verify_edit", verify_edit)
    graph.add_node("recover_or_finish", recover_or_finish)
    graph.add_edge(START, "seed_defaults")
    graph.add_edge("seed_defaults", "fetch_step_context")
    graph.add_edge("fetch_step_context", "prepare_prompt")
    graph.add_edge("prepare_prompt", "consult_helper_agent")
    graph.add_edge("consult_helper_agent", "plan_edit")
    graph.add_edge("plan_edit", "validate_proposal")
    graph.add_conditional_edges(
        "validate_proposal",
        should_continue_after_proposal_validation,
        {
            "continue": "check_edit_readiness",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "check_edit_readiness",
        should_continue_after_readiness,
        {
            "continue": "collect_edit_context",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "collect_edit_context",
        should_continue_after_readiness,
        {
            # read_target_file removed: file is already loaded in fetch_step_context
            "continue": "apply_edit",
            "done": END,
        },
    )
    graph.add_edge("apply_edit", "verify_edit")
    graph.add_edge("verify_edit", "recover_or_finish")
    graph.add_conditional_edges(
        "recover_or_finish",
        should_retry,
        {
            # Loop back to fetch_step_context so the LLM re-reads post-revert file state
            # before regenerating the edit — not back to prepare_prompt which loses context.
            "retry": "fetch_step_context",
            "done": END,
        },
    )
    return graph.compile()
