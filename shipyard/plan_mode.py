"""Plan mode: the agent reads, thinks, debates, and proposes before building.

Instead of immediately executing, plan mode:
1. Reads the reference codebase and docs
2. Analyzes the architecture
3. Proposes improvements with trade-off reasoning
4. Outputs a structured build plan
5. Waits for approval before executing

The plan is stored as a markdown spec that becomes part of the submission.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .action_planner import (
    _extract_response_text,
    _get_primary_model,
    _log_openai_call,
    _openai_headers,
)
from .context_explorer import build_broad_context
from .state import ShipyardState
from .workspaces import get_session_workspace


def generate_plan(
    state: ShipyardState,
    reference_path: str | None = None,
) -> dict[str, Any]:
    """Read the reference app, analyze it, and propose a rebuild plan.

    Returns a structured plan with:
    - analysis: what the reference app does
    - critique: what could be better
    - proposal: the agent's alternative architecture
    - trade_offs: honest assessment of each decision
    - build_phases: ordered list of implementation steps
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "No API key configured."}

    # Gather context from reference app
    ref_context = ""
    if reference_path:
        ref_context = _read_reference_context(reference_path)
    elif state.get("broad_context"):
        bc = state["broad_context"]
        parts = []
        if bc.get("file_tree"):
            parts.append("File tree:\n" + ", ".join(bc["file_tree"][:300]))
        if bc.get("project_stack"):
            parts.append(f"Stack: {bc['project_stack']}")
        if bc.get("discovered_docs"):
            parts.append("Docs:\n" + "\n".join(f"  - {d}" for d in bc["discovered_docs"][:20]))
        if bc.get("sampled_files"):
            for path, content in list(bc["sampled_files"].items())[:6]:
                parts.append(f"--- {path} ---\n{content[:4000]}")
        ref_context = "\n\n".join(parts)

    model = _get_primary_model(state)
    instruction = state.get("instruction", "").strip()

    prompt = f"""You are an expert software architect. You've been asked to rebuild an application, but you should make it BETTER, not just copy it.

## Your Task
{instruction}

## Reference Application Context
{ref_context[:30000]}

## What To Do

Analyze the reference app and produce a structured rebuild plan. You are NOT copying the original — you're building a better version informed by what you learned from reading it.

Return a JSON object with these keys:

### analysis
What does the reference app do? What's its core value proposition? List the main features and how they're implemented.

### critique
What's wrong with the reference architecture? What would you change? Be specific — name files, patterns, decisions that are suboptimal. Consider:
- Over-engineering vs under-engineering
- Dependency choices
- Code organization
- Missing features that should exist
- Features that add complexity without value

### proposal
Your alternative architecture. For each major component, explain:
- What you'd build differently and WHY
- What you'd keep the same and WHY
- Any new ideas the original doesn't have

### trade_offs
For each major decision in your proposal, honestly assess:
- What you gain
- What you lose
- Why the trade-off is worth it

### build_phases
Ordered list of implementation phases. Each phase should be:
- Small enough for one agent run
- Self-contained (can be verified independently)
- Building on the previous phase

Format each phase as: {{"phase": "name", "instruction": "detailed instruction", "files": ["list of files to create/modify"], "depends_on": ["prior phases"]}}

### tech_stack
Your chosen stack with justification for each choice.

Return ONLY valid JSON."""

    try:
        import httpx
        t0 = time.monotonic()
        with httpx.Client(timeout=60.0) as client:
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
            _log_openai_call({"model": model}, response, time.monotonic() - t0)
    except Exception as exc:
        return {"error": f"Plan generation failed: {exc}"}

    body = response.json()
    text = _extract_response_text(body)
    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Plan response was not valid JSON.", "raw": text[:2000]}

    # Save plan to workspace
    plan["model"] = model
    plan["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    plan["instruction"] = instruction

    try:
        workspace = get_session_workspace(state.get("session_id"))
        plan_path = workspace / "REBUILD_PLAN.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(_plan_to_markdown(plan), encoding="utf-8")
        plan["plan_path"] = str(plan_path)
    except Exception:
        pass

    return plan


def _read_reference_context(ref_path: str) -> str:
    """Read key files from the reference app for analysis."""
    root = Path(ref_path).resolve()
    if not root.exists():
        return f"Reference path not found: {ref_path}"

    parts: list[str] = []

    # File tree
    files = []
    ignored = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".next"}
    for p in sorted(root.rglob("*")):
        if any(part in ignored for part in p.relative_to(root).parts):
            continue
        if p.is_file():
            files.append(str(p.relative_to(root)))
        if len(files) >= 500:
            break
    parts.append(f"File tree ({len(files)} files):\n" + "\n".join(files[:200]))

    # Key files to read
    key_files = [
        "package.json", "tsconfig.json", "README.md",
        "api/package.json", "web/package.json", "shared/package.json",
        "api/src/index.ts", "web/src/App.tsx", "web/src/main.tsx",
        "shared/src/types/index.ts",
        "docs/application-architecture.md",
        "CODEBASE_ORIENTATION.md",
        "docker-compose.yml",
    ]
    for rel in key_files:
        full = root / rel
        if full.exists() and full.is_file():
            try:
                content = full.read_text(encoding="utf-8", errors="replace")[:6000]
                parts.append(f"--- {rel} ---\n{content}")
            except Exception:
                pass

    return "\n\n".join(parts)


def _plan_to_markdown(plan: dict[str, Any]) -> str:
    """Convert the structured plan to a readable markdown document."""
    lines = [
        "# Ship Rebuild Plan",
        f"*Generated: {plan.get('generated_at', 'unknown')}*",
        f"*Model: {plan.get('model', 'unknown')}*",
        "",
    ]

    if plan.get("instruction"):
        lines.extend(["## Instruction", plan["instruction"], ""])

    if plan.get("analysis"):
        lines.extend(["## Analysis", str(plan["analysis"]), ""])

    if plan.get("critique"):
        lines.extend(["## Critique", str(plan["critique"]), ""])

    if plan.get("proposal"):
        lines.extend(["## Proposed Architecture", str(plan["proposal"]), ""])

    if plan.get("trade_offs"):
        lines.extend(["## Trade-offs", str(plan["trade_offs"]), ""])

    if plan.get("tech_stack"):
        lines.extend(["## Tech Stack", str(plan["tech_stack"]), ""])

    if plan.get("build_phases"):
        lines.append("## Build Phases")
        phases = plan["build_phases"]
        if isinstance(phases, list):
            for i, phase in enumerate(phases, 1):
                if isinstance(phase, dict):
                    lines.append(f"### Phase {i}: {phase.get('phase', 'unnamed')}")
                    lines.append(phase.get("instruction", ""))
                    if phase.get("files"):
                        lines.append("Files: " + ", ".join(phase["files"]))
                    if phase.get("depends_on"):
                        lines.append("Depends on: " + ", ".join(phase["depends_on"]))
                    lines.append("")
                else:
                    lines.append(f"### Phase {i}")
                    lines.append(str(phase))
                    lines.append("")

    return "\n".join(lines)
