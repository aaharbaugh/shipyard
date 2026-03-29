from __future__ import annotations

from pathlib import Path

# Files larger than this are truncated rather than read whole into memory.
_MAX_READ_BYTES = 1_000_000  # 1 MB


def read_file(path: str, max_bytes: int = _MAX_READ_BYTES) -> str:
    p = Path(path)
    size = p.stat().st_size
    if size <= max_bytes:
        return p.read_text(encoding="utf-8", errors="replace")
    # Partial read for very large files.
    with p.open("rb") as fh:
        raw = fh.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    return text + f"\n\n[... file truncated at {max_bytes // 1024}KB of {size // 1024}KB ...]"
