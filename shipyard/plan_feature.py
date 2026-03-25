from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .specify import classify_request
from .storage_paths import DATA_ROOT, ensure_dir


def generate_spec_bundle(session_id: str | None, instruction: str) -> dict[str, Any]:
    classification = classify_request(instruction)
    if classification.mode != "feature":
        return {
            "mode": classification.mode,
            "reason": classification.reason,
            "created": False,
        }

    safe_session = (session_id or "default").strip() or "default"
    base_dir = ensure_dir(DATA_ROOT / "specs" / safe_session)
    spec_path = base_dir / "spec.json"
    architecture_path = base_dir / "architecture.md"
    tasks_path = base_dir / "tasks.json"

    spec_path.write_text(
        json.dumps(
            {
                "session_id": safe_session,
                "instruction": instruction,
                "status": "draft",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    architecture_path.write_text(
        "# Architecture Draft\n\n"
        f"Instruction: {instruction}\n\n"
        "- Break the request into bounded tasks.\n"
        "- Resolve target files and graph dependencies before execution.\n",
        encoding="utf-8",
    )
    tasks_path.write_text(
        "[\n"
        '  {"id": "task-1", "title": "Clarify scope", "status": "pending"},\n'
        '  {"id": "task-2", "title": "Implement change", "status": "pending"},\n'
        '  {"id": "task-3", "title": "Verify and summarize", "status": "pending"}\n'
        "]\n",
        encoding="utf-8",
    )

    return {
        "mode": classification.mode,
        "reason": classification.reason,
        "created": True,
        "paths": {
            "spec": str(spec_path),
            "architecture": str(architecture_path),
            "tasks": str(tasks_path),
        },
    }
