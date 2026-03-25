from __future__ import annotations

import re
from typing import Any


REPLACE_PATTERNS = (
    r'replace\s+"(?P<old>.+?)"\s+with\s+"(?P<new>.+?)"',
    r"replace\s+'(?P<old>.+?)'\s+with\s+'(?P<new>.+?)'",
    r'change\s+"(?P<old>.+?)"\s+to\s+"(?P<new>.+?)"',
    r"change\s+'(?P<old>.+?)'\s+to\s+'(?P<new>.+?)'",
)

RENAME_SYMBOL_PATTERNS = (
    r"replace\s+(?P<old>[A-Za-z_][A-Za-z0-9_]*)\s+with\s+(?P<new>[A-Za-z_][A-Za-z0-9_]*)",
    r"(?:change|rename)\s+all\s+instances\s+of\s+(?P<old>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+(?P<new>[A-Za-z_][A-Za-z0-9_]*)",
    r"(?:change|rename)\s+all\s+occurrences\s+of\s+(?P<old>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+(?P<new>[A-Za-z_][A-Za-z0-9_]*)",
    r"(?:rename|change)\s+(?P<old>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+(?P<new>[A-Za-z_][A-Za-z0-9_]*)",
    r"change\s+(?P<old>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+(?P<new>[A-Za-z_][A-Za-z0-9_]*)\s*,?\s+and\s+update\s+other\s+places\s+where\s+(?P=old)\s+appears",
)

WRITE_PATTERNS = (
    r'write\s+"(?P<content>.+?)"\s+to\s+(?:the\s+)?file',
    r"write\s+'(?P<content>.+?)'\s+to\s+(?:the\s+)?file",
    r"write\s+(?P<content>.+?)\s+to\s+(?:a\s+|the\s+|new\s+)?file",
    r"write\s+(?P<content>.+)$",
    r"(?:make|create)\s+(?:(?:a|the|new)\s+)*file\s+with\s+(?P<content>.+)",
    r'fill\s+(?:the\s+)?file\s+with\s+"(?P<content>.+?)"',
    r"fill\s+(?:the\s+)?file\s+with\s+'(?P<content>.+?)'",
    r"fill\s+(?:a\s+|the\s+|new\s+)?file\s+with\s+(?P<content>.+)",
)

BLANK_FILE_PATTERNS = (
    r"(?:make|create)\s+(?:a\s+|the\s+)?(?:new\s+)?file$",
    r"(?:make|create)\s+(?:a\s+|the\s+)?(?:blank|empty)\s+file$",
    r"(?:make|create)\s+(?:a\s+|the\s+)?(?:new\s+)?[\w.-]+\.[A-Za-z0-9]+$",
)

TYPED_BLANK_FILE_PATTERNS = (
    r"(?:make|create)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:blank\s+|empty\s+)?(javascript|typescript|python|html|css|json|markdown|bash|sql)\s+file$",
)

APPEND_PATTERNS = (
    r'append\s+"(?P<content>.+?)"(?:\s+to\s+(?:the\s+)?file)?',
    r"append\s+'(?P<content>.+?)'(?:\s+to\s+(?:the\s+)?file)?",
    r"append\s+(?P<content>.+?)(?:\s+to\s+(?:a\s+|the\s+|new\s+)?file)?$",
)

PREPEND_PATTERNS = (
    r'prepend\s+"(?P<content>.+?)"(?:\s+to\s+(?:the\s+)?file)?',
    r"prepend\s+'(?P<content>.+?)'(?:\s+to\s+(?:the\s+)?file)?",
    r"prepend\s+(?P<content>.+?)(?:\s+to\s+(?:a\s+|the\s+|new\s+)?file)?$",
)

DELETE_PATTERNS = (
    r"(?:delete|remove)\s+(?P<target>[\w.-]+\.[A-Za-z0-9]+)",
)

POSITIONAL_REPLACE_PATTERNS = (
    r"\b(?P<which>first|second|third|middle|last)\s+(?P<old>\S+?)\s+(?:with|to|into)\s+(?:a\s+|an\s+)?(?P<new>\S+)",
    r"\bmake\s+the\s+(?P<which>first|second|third|middle|last)\s+(?P<old>\S+?)\s+(?:with|to|into)\s+(?:a\s+|an\s+)?(?P<new>\S+)",
    r"\bchange\s+the\s+(?P<which>first|second|third|middle|last)\s+(?P<old>\S+?)\s+(?:with|to|into)\s+(?:a\s+|an\s+)?(?P<new>\S+)",
)

COPY_PATTERNS = (
    r"(?:make|create)\s+(?P<count>\d+|some)\s+copies?\s+of\s+(?P<target>[\w.-]+\.[A-Za-z0-9]+)",
    r"(?:copy|duplicate)\s+(?P<target>[\w.-]+\.[A-Za-z0-9]+)\s+(?P<count>\d+|some)\s+times?",
)

CREATE_FILES_PATTERNS = (
    r"(?:make|create)\s+(?P<count>\d+|some)\s+(?:(?:new|random)\s+)?files?",
    r"(?:make|create)\s+(?P<count>\d+|some)\s+(?:blank|empty)\s+files?",
    r"(?:make|create)\s+(?P<count>\d+|some)\s+(?:(?:new|random)\s+)?(javascript|typescript|python|html|css|json|markdown|bash|sql)\s+files?",
)

STEP_STARTERS = "create|make|write|append|prepend|delete|remove|copy|duplicate|replace|change|rename|fill|edit"

GENERATION_CUE_PATTERN = re.compile(
    r"\b("
    r"algorithm|function|class|script|module|program|code|implementation|"
    r"component|handler|endpoint|query|migration|test|tests|unit test|"
    r"python|javascript|typescript|html|css|json|sql|bash|shell"
    r")\b",
    flags=re.IGNORECASE,
)

GENERATION_VERB_PATTERN = re.compile(
    r"\b(add|write|generate|create|insert|put|implement|draft)\b",
    flags=re.IGNORECASE,
)


def infer_edit_mode(state: dict[str, Any]) -> str:
    context = state.get("context", {})
    explicit_mode = state.get("edit_mode")
    if explicit_mode:
        return explicit_mode
    if context.get("function_name"):
        return "named_function"
    return "anchor"


def derive_edit_spec(state: dict[str, Any]) -> tuple[str, str | None, str | None, list[str]]:
    context = state.get("context", {})
    anchor = state.get("anchor")
    replacement = state.get("replacement")
    notes: list[str] = []
    edit_mode = infer_edit_mode(state)

    if edit_mode in {"write_file", "append", "prepend"} and replacement is not None:
        notes.append("Non-anchor edit content already provided by caller.")
        return edit_mode, anchor, replacement, notes

    if anchor and replacement is not None:
        notes.append("Edit spec already provided by caller.")
        return edit_mode, anchor, replacement, notes

    if context.get("write_text"):
        notes.append("Derived whole-file content from injected context.")
        return "write_file", None, context["write_text"], notes
    if context.get("append_text"):
        notes.append("Derived append content from injected context.")
        return "append", None, context["append_text"], notes
    if context.get("prepend_text"):
        notes.append("Derived prepend content from injected context.")
        return "prepend", None, context["prepend_text"], notes
    if context.get("search_text") and context.get("replace_text"):
        notes.append("Derived anchor and replacement from injected context.")
        return edit_mode, context["search_text"], context["replace_text"], notes

    parsed = parse_instruction(state.get("instruction", ""))
    if parsed:
        parsed_mode, parsed_anchor, parsed_replacement = parsed
        notes.append(f"Derived {parsed_mode} edit spec from instruction text.")
        return parsed_mode, parsed_anchor, parsed_replacement, notes

    if edit_mode == "named_function":
        notes.append("Using named-function edit mode from injected context.")

    return edit_mode, anchor, replacement, notes


def parse_instruction(instruction: str) -> tuple[str, str | None, str] | None:
    text = _normalize_instruction_text(instruction)

    for pattern in CREATE_FILES_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "create_files", None, _normalize_count_token(match.group("count"))

    for pattern in RENAME_SYMBOL_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "rename_symbol", match.group("old"), match.group("new")

    for pattern in COPY_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "copy_file", match.group("target"), _normalize_count_token(match.group("count"))

    for pattern in TYPED_BLANK_FILE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "write_file", None, ""

    for pattern in BLANK_FILE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "write_file", None, ""

    for pattern in DELETE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "delete_file", None, ""

    for pattern in POSITIONAL_REPLACE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "anchor", match.group("old").strip("\"'.,!?"), match.group("new").strip("\"'.,!?")

    for pattern in WRITE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "write_file", None, match.group("content")

    for pattern in APPEND_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "append", None, match.group("content")

    for pattern in PREPEND_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "prepend", None, match.group("content")

    for pattern in REPLACE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return "anchor", match.group("old"), match.group("new")
    return None


def parse_occurrence_selector(instruction: str) -> str | None:
    text = instruction.strip()
    match = re.search(r"\b(first|second|third|middle|last)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def is_generation_request(instruction: str) -> bool:
    text = _normalize_instruction_text(instruction)
    if not text:
        return False
    return bool(GENERATION_VERB_PATTERN.search(text) and GENERATION_CUE_PATTERN.search(text))


def prefers_append_for_generation(instruction: str) -> bool:
    text = _normalize_instruction_text(instruction)
    if not text:
        return False
    return bool(re.search(r"\b(add|insert|put)\b", text, flags=re.IGNORECASE) and GENERATION_CUE_PATTERN.search(text))


def _normalize_count_token(value: str) -> str:
    return "3" if value.lower() == "some" else value


def split_instruction_steps(instruction: str) -> list[str]:
    text = " ".join((instruction or "").strip().split())
    if not text:
        return []

    pattern = re.compile(
        rf"(?:,?\s+and\s+|\s+then\s+|;\s*)(?=(?:in\s+file\s+\d+\s*,?\s*)?(?:i\s+want\s+you\s+to\s+)?(?:{STEP_STARTERS})\b)",
        flags=re.IGNORECASE,
    )
    steps = [segment.strip(" ,") for segment in pattern.split(text) if segment.strip(" ,")]
    return steps or [text]


def _normalize_instruction_text(instruction: str) -> str:
    text = " ".join((instruction or "").strip().split())
    text = re.sub(r"^\s*i\s+want\s+you\s+to\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+file\s+\d+\s*,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+[\w.-]+\s*,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+file\s+[\w.-]+\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+[\w.-]+\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+file\s+\d+\s*,\s*i\s+want\s+you\s+to\s+", lambda m: re.sub(r"\bi\s+want\s+you\s+to\s+", "", m.group(0), flags=re.IGNORECASE), text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+[\w.-]+\s*,\s*i\s+want\s+you\s+to\s+", lambda m: re.sub(r"\bi\s+want\s+you\s+to\s+", "", m.group(0), flags=re.IGNORECASE), text, flags=re.IGNORECASE)
    return text
