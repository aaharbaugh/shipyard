"""Supervisor agent: decomposes high-level instructions into scoped sub-tasks.

The supervisor does NOT generate code or file edits — it only decides *what*
needs to happen and *where*. Each sub-task is then executed by an independent
worker agent with its own context window.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .action_planner import (
    _extract_response_text,
    _get_nano_model,
    _get_primary_model,
    _log_openai_call,
    _openai_headers,
    PlanningCancelledError,
)
from .context_explorer import build_broad_context
from .state import ShipyardState


def should_use_supervisor(state: ShipyardState) -> bool:
    """Decide if the instruction is complex enough for multi-agent execution.

    Returns True when the task likely involves multiple files/packages that
    can be worked on independently. Returns False for simple single-file tasks.
    """
    instruction = (state.get("instruction") or "").lower()
    broad_context = state.get("broad_context") or {}
    file_tree = broad_context.get("file_tree") or []

    # Explicit multi-agent request always honored
    if "multi-agent" in instruction or "parallel" in instruction:
        return True

    # Don't decompose on empty/near-empty workspaces — nothing to parallelize
    if len(file_tree) < 5:
        return False

    # Don't decompose scaffold/create operations — they're one atomic operation
    _create_signals = ("scaffold", "create a", "set up a", "initialize", "init a", "bootstrap")
    if any(instruction.startswith(s) or f" {s}" in instruction for s in _create_signals):
        return False

    # Simple/short instructions → single agent
    if len(instruction) < 40:
        return False

    # Count distinct directories mentioned or targeted
    discovered_docs = broad_context.get("discovered_docs") or []
    project_stack = broad_context.get("project_stack") or {}

    # Monorepo signals: multiple top-level packages
    top_dirs = {p.split("/")[0] for p in file_tree if "/" in p}
    package_dirs = top_dirs & {"api", "web", "shared", "packages", "apps", "libs", "services", "src"}
    if len(package_dirs) >= 2:
        # Instruction mentions multiple areas
        areas_mentioned = sum(1 for d in package_dirs if d in instruction)
        if areas_mentioned >= 2:
            return True

    # High-level instructions that need decomposition
    _decomp_signals = (
        "refactor", "redesign", "implement", "migrate", "upgrade",
        "across", "all files", "entire", "whole", "everywhere",
    )
    if any(signal in instruction for signal in _decomp_signals):
        return True

    # If docs were discovered and instruction is vague (no specific file), supervisor helps
    if discovered_docs and not state.get("target_path"):
        words = instruction.split()
        if len(words) > 8:
            return True

    return False


def plan_subtasks(
    state: ShipyardState,
    *,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Ask the LLM to decompose an instruction into independent sub-tasks.

    Returns:
        {
            "subtasks": [
                {
                    "id": "w1",
                    "instruction": "...",
                    "scope": "api/src/routes",  # directory or file pattern
                    "files": ["api/src/routes/auth.ts", ...],  # specific files if known
                    "depends_on": [],  # other subtask ids
                },
                ...
            ],
            "reasoning": "...",
            "provider": "openai",
        }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"subtasks": [], "error": "No API key configured."}

    model = _get_primary_model(state)
    prompt = _build_supervisor_prompt(state, max_workers)

    try:
        import httpx
        t0 = time.monotonic()
        with httpx.Client(timeout=45.0) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers=_openai_headers(api_key),
                json={
                    "model": model,
                    "input": prompt,
                    "text": {"format": {"type": "json_object"}},
                },
            )
            response.raise_for_status()
            _log_openai_call(
                {"model": model}, response, time.monotonic() - t0
            )
    except Exception as exc:
        return {"subtasks": [], "error": f"Supervisor planning failed: {exc}"}

    body = response.json()
    text = _extract_response_text(body)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"subtasks": [], "error": "Supervisor returned invalid JSON."}

    subtasks = parsed.get("subtasks") or parsed.get("tasks") or []
    if not isinstance(subtasks, list):
        subtasks = []

    # Normalize subtask fields
    normalized = []
    for i, task in enumerate(subtasks[:max_workers]):
        if not isinstance(task, dict):
            continue
        normalized.append({
            "id": str(task.get("id") or f"w{i + 1}"),
            "instruction": str(task.get("instruction") or "").strip(),
            "scope": str(task.get("scope") or task.get("directory") or "."),
            "files": list(task.get("files") or []),
            "depends_on": list(task.get("depends_on") or []),
        })

    return {
        "subtasks": normalized,
        "reasoning": str(parsed.get("reasoning") or ""),
        "provider": "openai",
        "model": model,
    }


def _build_supervisor_prompt(state: ShipyardState, max_workers: int) -> str:
    """Build a lightweight prompt for task decomposition — no file contents."""
    instruction = state.get("instruction", "").strip()
    broad_context = state.get("broad_context") or {}
    file_tree = broad_context.get("file_tree") or []
    project_stack = broad_context.get("project_stack") or {}
    discovered_docs = broad_context.get("discovered_docs") or []

    lines = [
        "You are a supervisor agent. Decompose this instruction into independent sub-tasks.",
        "Each sub-task will be executed by a separate worker agent in parallel.",
        "Workers cannot see each other's changes during execution.",
        "",
        "Return JSON with keys: subtasks, reasoning.",
        "Each subtask: {id, instruction, scope, files, depends_on}.",
        "- id: unique string (w1, w2, ...)",
        "- instruction: specific actionable instruction for the worker",
        "- scope: directory or file path the worker should focus on",
        "- files: list of specific files the worker needs to read/edit (if known)",
        "- depends_on: list of subtask ids that must complete first (use sparingly — prefer independent tasks)",
        "",
        "Rules:",
        f"- Maximum {max_workers} sub-tasks.",
        "- Each sub-task should target DIFFERENT files. Two workers must NOT edit the same file.",
        "- If tasks are truly sequential (B needs A's output), use depends_on. Otherwise keep them independent.",
        "- If the instruction is simple enough for one agent, return a single subtask.",
        "- Include read/inspect steps in each worker's instruction — workers start with no file context.",
        "",
        f"Instruction: {instruction}",
    ]

    if project_stack:
        parts = []
        for key in ("language", "framework", "package_manager"):
            if project_stack.get(key):
                parts.append(f"{key}={project_stack[key]}")
        if parts:
            lines.append(f"Project stack: {', '.join(parts)}")

    if file_tree:
        lines.append(f"File tree ({len(file_tree)} files):")
        lines.append(", ".join(file_tree[:300]))

    if discovered_docs:
        lines.append("Planning docs found:")
        lines.extend(f"  - {doc}" for doc in discovered_docs[:15])

    return "\n".join(lines)
