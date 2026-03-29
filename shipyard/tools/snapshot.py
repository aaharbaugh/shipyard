from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..storage_paths import SNAPSHOTS_ROOT, ensure_dir


def snapshot_file(path: str) -> str:
    source = Path(path)
    snapshot_dir = ensure_dir(SNAPSHOTS_ROOT)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_path = snapshot_dir / f"{stamp}-{source.name}"
    snapshot_contents = ""
    if source.is_file():
        snapshot_contents = source.read_text(encoding="utf-8")
    snapshot_path.write_text(snapshot_contents, encoding="utf-8")
    return str(snapshot_path)
