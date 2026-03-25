from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage_paths import SESSIONS_ROOT, ensure_dir
from .state import ShipyardState


class SessionStore:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root) if root is not None else SESSIONS_ROOT
        ensure_dir(self.root)

    def append_run(self, state: ShipyardState) -> str:
        session_id = state.get("session_id", "unknown")
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        history_path = session_dir / "history.jsonl"
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "instruction": state.get("instruction"),
            "status": state.get("status"),
            "edit_mode": state.get("edit_mode"),
            "proposal_provider": state.get("proposal_summary", {}).get("provider"),
            "proposal_valid": state.get("proposal_summary", {}).get("is_valid"),
            "target_path": state.get("target_path"),
            "changed_files": state.get("changed_files", []),
            "content_hash": state.get("content_hash"),
            "error": state.get("error"),
            "trace_path": state.get("trace_path"),
            "snapshot_path": state.get("snapshot_path"),
            "code_graph_ready": state.get("code_graph_status", {}).get("ready"),
            "code_graph_refresh_required": state.get("code_graph_status", {}).get("refresh_required"),
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
                        "instruction": latest.get("instruction"),
                        "edit_mode": latest.get("edit_mode"),
                        "changed_files": latest.get("changed_files", []),
                        "content_hash": latest.get("content_hash"),
                        "proposal_provider": latest.get("proposal_summary", {}).get("provider"),
                        "proposal_valid": latest.get("proposal_summary", {}).get("is_valid"),
                        "code_graph_refresh_required": latest.get("code_graph_status", {}).get("refresh_required"),
                    }
                )
        return sessions
