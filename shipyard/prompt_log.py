from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage_paths import LOGS_ROOT, ensure_dir
from .state import ShipyardState


class PromptLog:
    def __init__(self, path: str | None = None) -> None:
        default_path = ensure_dir(LOGS_ROOT) / "prompt_log.jsonl"
        self.path = Path(path) if path is not None else default_path
        ensure_dir(self.path.parent)

    def append(self, state: ShipyardState) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": state.get("session_id"),
            "instruction": state.get("instruction"),
            "target_path": state.get("target_path"),
            "edit_mode": state.get("edit_mode"),
            "proposal_mode": state.get("proposal_mode"),
            "context": state.get("context", {}),
            "verification_commands": state.get("verification_commands", []),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return str(self.path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
