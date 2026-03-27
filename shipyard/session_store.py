from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .runtime_state import enrich_state_sections
from .storage_paths import SESSIONS_ROOT, ensure_dir
from .state import ShipyardState


class SessionStore:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root) if root is not None else SESSIONS_ROOT
        ensure_dir(self.root)

    def append_run(self, state: ShipyardState) -> str:
        public_state = state if {"request", "plan", "execution", "artifacts"}.issubset(state.keys()) else enrich_state_sections(state)
        session_id = state.get("session_id", "unknown")
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        history_path = session_dir / "history.jsonl"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "instruction": public_state.get("request", {}).get("instruction"),
            "status": public_state.get("status"),
            "changed_files": public_state.get("execution", {}).get("changed_files", []),
            "content_hash": public_state.get("execution", {}).get("content_hash"),
            "error": public_state.get("execution", {}).get("error"),
            "trace_path": public_state.get("artifacts", {}).get("trace_path"),
            "snapshot_path": public_state.get("artifacts", {}).get("snapshot_path"),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        latest_path = session_dir / "latest_state.json"
        latest_path.write_text(json.dumps(public_state, indent=2, sort_keys=True), encoding="utf-8")
        return str(session_dir)

    def load_latest_state(self, session_id: str) -> ShipyardState | None:
        latest_path = self.root / session_id / "latest_state.json"
        if not latest_path.exists():
            return None
        return json.loads(latest_path.read_text(encoding="utf-8"))

    def load_history(self, session_id: str) -> list[dict[str, Any]]:
        history_path = self.root / session_id / "history.jsonl"
        if not history_path.exists():
            return []
        with history_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

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
                        "instruction": latest.get("request", {}).get("instruction"),
                        "changed_files": latest.get("execution", {}).get("changed_files", []),
                        "content_hash": latest.get("execution", {}).get("content_hash"),
                    }
                )
        return sessions
