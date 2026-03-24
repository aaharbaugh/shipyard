from __future__ import annotations

from pathlib import Path


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
