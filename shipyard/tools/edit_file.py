from __future__ import annotations

from pathlib import Path


class AnchorEditError(ValueError):
    """Raised when an anchor-based edit cannot be applied safely."""


def validate_anchor_edit(content: str, anchor: str) -> None:
    match_count = content.count(anchor)

    if match_count == 0:
        raise AnchorEditError("Anchor was not found in the target file.")
    if match_count > 1:
        raise AnchorEditError("Anchor matched multiple locations in the target file.")


def apply_anchor_edit(path: str, anchor: str, replacement: str) -> str:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    validate_anchor_edit(content, anchor)

    updated = content.replace(anchor, replacement, 1)
    file_path.write_text(updated, encoding="utf-8")
    return updated
