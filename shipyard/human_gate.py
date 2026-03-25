from __future__ import annotations

from typing import Any


def make_human_gate(
    reason: str,
    *,
    action: str,
    prompt: str,
    status: str = "needs_input",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "action": action,
        "prompt": prompt,
        "details": details or {},
    }
