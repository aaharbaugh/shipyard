from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import ShipyardState


class SessionStore:
    def __init__(self, root: str = ".shipyard/sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_run(self, state: ShipyardState) -> str:
        session_id = state.get("session_id", "unknown")
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        history_path = session_dir / "history.jsonl"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "instruction": state.get("instruction"),
            "status": state.get("status"),
            "target_path": state.get("target_path"),
            "error": state.get("error"),
            "trace_path": state.get("trace_path"),
            "snapshot_path": state.get("snapshot_path"),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        latest_path = session_dir / "latest_state.json"
        latest_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return str(session_dir)

    def load_latest_state(self, session_id: str) -> ShipyardState | None:
        latest_path = self.root / session_id / "latest_state.json"
        if not latest_path.exists():
            return None
        return json.loads(latest_path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for session_dir in sorted(self.root.iterdir()):
            if not session_dir.is_dir():
                continue
            latest_path = session_dir / "latest_state.json"
            if latest_path.exists():
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "session_id": session_dir.name,
                        "status": latest.get("status"),
                        "instruction": latest.get("instruction"),
                    }
                )
        return sessions
