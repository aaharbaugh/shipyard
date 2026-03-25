from __future__ import annotations

from dataclasses import dataclass


DIRECT_EDIT_PREFIXES = ("replace ", "write ", "append ", "prepend ", "change ", "fill ")
FEATURE_KEYWORDS = ("feature", "workflow", "support", "refactor", "cleanup", "clean up", "implement", "build")


@dataclass(frozen=True)
class RequestClassification:
    mode: str
    reason: str


def classify_request(instruction: str) -> RequestClassification:
    text = instruction.strip().lower()
    if not text:
        return RequestClassification(mode="empty", reason="No instruction provided.")
    if text.startswith(DIRECT_EDIT_PREFIXES):
        return RequestClassification(mode="direct_edit", reason="Instruction looks like a direct file edit.")
    if any(keyword in text for keyword in FEATURE_KEYWORDS):
        return RequestClassification(mode="feature", reason="Instruction looks like a larger feature/refactor request.")
    if len(text.split()) > 8:
        return RequestClassification(mode="feature", reason="Instruction is broad enough to benefit from spec/task scaffolding.")
    return RequestClassification(mode="direct_edit", reason="Instruction can be handled as a direct edit.")
