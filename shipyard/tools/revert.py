from __future__ import annotations

from pathlib import Path


def revert_file(target_path: str, snapshot_path: str) -> None:
    target = Path(target_path)
    snapshot = Path(snapshot_path)
    target.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")
