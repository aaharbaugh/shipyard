from __future__ import annotations

from datetime import datetime
from pathlib import Path


def snapshot_file(path: str) -> str:
    source = Path(path)
    snapshot_dir = Path(".shipyard") / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_path = snapshot_dir / f"{stamp}-{source.name}"
    snapshot_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(snapshot_path)
