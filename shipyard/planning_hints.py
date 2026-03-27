from __future__ import annotations

import re
from typing import Any


FILENAME_PATTERN = re.compile(r"\b(?P<name>[\w.-]+\.[A-Za-z0-9]+)\b")
COUNT_PATTERN = re.compile(r"\b(?P<count>\d+|some)\b", flags=re.IGNORECASE)
FILE_INDEX_PATTERN = re.compile(r"\bfile\s+(?P<index>\d+)\b", flags=re.IGNORECASE)


def infer_target_path_from_instruction(instruction: str, context: dict[str, Any]) -> str | None:
    if context.get("function_name"):
        return None
    match = FILENAME_PATTERN.search(instruction or "")
    if not match:
        return None
    return match.group("name")


def extract_explicit_filenames(instruction: str) -> list[str]:
    seen: set[str] = set()
    filenames: list[str] = []
    for match in FILENAME_PATTERN.finditer(instruction or ""):
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        filenames.append(name)
    return filenames


def resolve_requested_target_hint(
    state: dict[str, Any],
    context: dict[str, Any],
    inferred_target: str | None,
) -> str | None:
    explicit_target = state.get("target_path")
    file_hint = context.get("file_hint")

    if inferred_target and is_stale_scratch_target(explicit_target):
        return inferred_target
    if explicit_target:
        return str(explicit_target)

    if inferred_target and is_stale_scratch_target(file_hint):
        return inferred_target
    if file_hint:
        return str(file_hint)

    return inferred_target


def is_stale_scratch_target(value: Any) -> bool:
    if not value:
        return False
    name = str(value).strip().split("/")[-1]
    return bool(
        re.fullmatch(r"(?:scratch|file)(?:-[0-9a-f]{6})?\.[A-Za-z0-9]+", name)
    )


def infer_copy_count(instruction: str) -> int | None:
    text = instruction or ""
    if not re.search(r"\b(?:copy|duplicate|copies?)\b", text, flags=re.IGNORECASE):
        return None
    match = COUNT_PATTERN.search(text)
    if not match:
        return None
    return _normalize_count(match.group("count"))


def infer_create_quantity(instruction: str) -> int | None:
    text = instruction or ""
    if not re.search(r"\b(?:make|create)\b", text, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bfiles?\b", text, flags=re.IGNORECASE):
        return None
    match = COUNT_PATTERN.search(text)
    if not match:
        return None
    return _normalize_count(match.group("count"))


def infer_batch_target_path(instruction: str) -> str | None:
    match = FILE_INDEX_PATTERN.search(instruction or "")
    if not match:
        return None
    index = match.group("index")
    extension = ".py" if "python" in (instruction or "").lower() else ".txt"
    return f"file{index}{extension}"


def infer_batch_content(instruction: str) -> str | None:
    match = re.search(r"\bwrite\s+(?P<content>.+)$", (instruction or "").strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("content").strip().rstrip(".")


def _normalize_count(value: str) -> int:
    return 3 if value.lower() == "some" else max(1, int(value))
