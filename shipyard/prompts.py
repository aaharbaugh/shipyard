from __future__ import annotations

from .state import RuntimeContext, ShipyardState


def build_runtime_prompt(state: ShipyardState) -> str:
    context = state.get("context", {})
    sections = [
        "You are the Shipyard MVP coding agent.",
        f"Instruction: {state.get('instruction', '').strip()}",
    ]

    if state.get("target_path"):
        sections.append(f"Target path: {state['target_path']}")

    if context:
        sections.append("Injected context:")
        sections.extend(_format_context(context))

    edit_mode = state.get("edit_mode")
    if edit_mode == "named_function":
        function_name = context.get("function_name") or "unknown"
        sections.append(f"Editing mode: named-function replacement for {function_name}.")
    elif state.get("anchor"):
        sections.append("Editing mode: anchor-based replacement.")

    return "\n".join(sections).strip()


def build_proposal_prompt(state: ShipyardState) -> str:
    context = state.get("context", {})
    lines = [
        "Return only JSON with keys: target_path, anchor, replacement.",
        "You are proposing a surgical anchor-based edit for a coding agent.",
        f"Instruction: {state.get('instruction', '').strip()}",
        f"Current target path: {state.get('target_path', '')}",
    ]

    if context:
        lines.append("Injected context:")
        lines.extend(_format_context(context))

    lines.append("If file_hint exists, prefer it as target_path.")
    lines.append("If the instruction is a simple replace request, preserve exact text.")
    return "\n".join(lines).strip()


def _format_context(context: RuntimeContext) -> list[str]:
    lines: list[str] = []
    for key in (
        "spec_note",
        "test_failure",
        "file_hint",
        "search_text",
        "replace_text",
        "helper_notes",
        "function_name",
    ):
        value = context.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    return lines
