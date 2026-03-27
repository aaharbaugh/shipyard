from __future__ import annotations
from pathlib import Path
import re


class AnchorEditError(ValueError):
    """Raised when an anchor-based edit cannot be applied safely."""


def _find_occurrences(content: str, anchor: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        index = content.find(anchor, start)
        if index == -1:
            return positions
        positions.append(index)
        start = index + len(anchor)


def find_anchor_pointers(content: str, anchor: str) -> list[dict[str, int]]:
    return [{"start": start, "end": start + len(anchor)} for start in _find_occurrences(content, anchor)]


def _find_occurrences_casefold(content: str, anchor: str) -> list[int]:
    if not anchor:
        return []

    lowered_content = content.casefold()
    lowered_anchor = anchor.casefold()
    positions: list[int] = []
    start = 0
    while True:
        index = lowered_content.find(lowered_anchor, start)
        if index == -1:
            return positions
        positions.append(index)
        start = index + len(lowered_anchor)


def _resolve_occurrence_index(match_count: int, occurrence_selector: str | None) -> int | None:
    if occurrence_selector is None:
        return None
    if occurrence_selector == "all":
        return None

    mapping = {
        "first": 0,
        "second": 1,
        "third": 2,
        "last": match_count - 1,
        "middle": match_count // 2,
    }
    index = mapping.get(occurrence_selector)
    if index is None or index < 0 or index >= match_count:
        raise AnchorEditError(f"Could not resolve `{occurrence_selector}` occurrence for the target text.")
    return index


def validate_anchor_edit(content: str, anchor: str, occurrence_selector: str | None = None) -> None:
    match_positions = _find_occurrences(content, anchor)
    match_count = len(match_positions)

    if match_count == 0:
        fallback_positions = _find_occurrences_casefold(content, anchor)
        fallback_count = len(fallback_positions)
        if fallback_count == 0:
            raise AnchorEditError("Anchor was not found in the target file.")
        if occurrence_selector is not None:
            _resolve_occurrence_index(fallback_count, occurrence_selector)
            return
        if fallback_count > 1:
            raise AnchorEditError("Anchor was not found exactly, and case-insensitive matching found multiple locations.")
        return
    if occurrence_selector is not None:
        _resolve_occurrence_index(match_count, occurrence_selector)
        return
    if match_count > 1:
        raise AnchorEditError("Anchor matched multiple locations in the target file.")


def apply_anchor_edit(
    path: str,
    anchor: str,
    replacement: str,
    occurrence_selector: str | None = None,
) -> str:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    validate_anchor_edit(content, anchor, occurrence_selector)

    positions = _find_occurrences(content, anchor)
    anchor_length = len(anchor)
    if not positions:
        positions = _find_occurrences_casefold(content, anchor)
        if positions:
            anchor_length = len(anchor)

    if occurrence_selector == "all":
        updated = content
        for start in reversed(positions):
            updated = updated[:start] + replacement + updated[start + anchor_length:]
        file_path.write_text(updated, encoding="utf-8")
        return updated

    if occurrence_selector is None:
        start = positions[0]
    else:
        occurrence_index = _resolve_occurrence_index(len(positions), occurrence_selector)
        start = positions[occurrence_index]

    updated = content[:start] + replacement + content[start + anchor_length:]
    file_path.write_text(updated, encoding="utf-8")
    return updated


def validate_pointer_edits(content: str, pointers: list[dict[str, int]], anchor: str | None = None) -> None:
    if not isinstance(pointers, list) or not pointers:
        raise AnchorEditError("Pointers were not provided for pointer-based edit.")

    last_end = -1
    for index, pointer in enumerate(sorted(pointers, key=lambda item: item["start"]), start=1):
        start = pointer.get("start")
        end = pointer.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise AnchorEditError(f"Pointer {index} requires integer start and end.")
        if start < 0 or end < start or end > len(content):
            raise AnchorEditError(f"Pointer {index} is outside the file bounds.")
        if start < last_end:
            raise AnchorEditError("Pointers overlap in the target file.")
        if anchor is not None and content[start:end] != anchor:
            raise AnchorEditError(f"Pointer {index} does not match the requested anchor text.")
        last_end = end


def apply_pointer_edits(
    path: str,
    pointers: list[dict[str, int]],
    replacement: str,
    anchor: str | None = None,
) -> str:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    validate_pointer_edits(content, pointers, anchor)

    updated = content
    for pointer in sorted(pointers, key=lambda item: item["start"], reverse=True):
        start = pointer["start"]
        end = pointer["end"]
        updated = updated[:start] + replacement + updated[end:]

    file_path.write_text(updated, encoding="utf-8")
    return updated


def apply_symbol_rename(path: str, old_symbol: str, new_symbol: str) -> str:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"\b{re.escape(old_symbol)}\b")
    if not pattern.search(content):
        raise AnchorEditError("Symbol was not found in the target file.")
    updated = pattern.sub(new_symbol, content)
    file_path.write_text(updated, encoding="utf-8")
    return updated
