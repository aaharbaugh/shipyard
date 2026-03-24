from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .state import ShipyardState


def write_trace(state: ShipyardState) -> str:
    trace_dir = Path(".shipyard") / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = state.get("session_id", "unknown")
    trace_path = trace_dir / f"{stamp}-{session_id}.json"

    trace_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(trace_path)
