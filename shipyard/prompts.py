from __future__ import annotations

from .state import RuntimeContext, ShipyardState


def build_runtime_prompt(state: ShipyardState) -> str:
    context = state.get("context", {})
    visible_context = dict(context)
    if state.get("target_path") and visible_context.get("file_hint"):
        visible_context.pop("file_hint", None)
    sections = [
        "You are the Shipyard MVP coding agent.",
        f"Instruction: {state.get('instruction', '').strip()}",
    ]

    if state.get("target_path"):
        sections.append(f"Target path: {state['target_path']}")

    if visible_context:
        sections.append("Injected context:")
        sections.extend(_format_context(visible_context))

    edit_mode = state.get("edit_mode")
    if edit_mode == "named_function":
        function_name = context.get("function_name") or "unknown"
        sections.append(f"Editing mode: named-function replacement for {function_name}.")
    elif edit_mode == "write_file":
        sections.append("Editing mode: replace the entire file contents.")
    elif edit_mode == "append":
        sections.append("Editing mode: append content to the file.")
    elif edit_mode == "prepend":
        sections.append("Editing mode: prepend content to the file.")
    elif edit_mode == "copy_file":
        sections.append("Editing mode: create copies of the source file.")
    elif edit_mode == "create_files":
        sections.append("Editing mode: create multiple new files.")
    elif edit_mode == "rename_symbol":
        sections.append("Editing mode: rename a symbol everywhere it appears in the file.")
    elif state.get("anchor"):
        sections.append("Editing mode: anchor-based replacement.")

    return "\n".join(sections).strip()


def build_proposal_prompt(state: ShipyardState) -> str:
    context = state.get("context", {})
    helper_output = state.get("helper_output", {})
    code_graph_status = state.get("code_graph_status", {})
    lines = [
        "Return only JSON with keys: target_path, anchor, replacement, edit_mode, copy_count, quantity.",
        "Supported edit_mode values are: anchor, named_function, write_file, append, prepend, delete_file, copy_file, create_files, rename_symbol.",
        "You are proposing a precise file edit for a coding agent.",
        f"Instruction: {state.get('instruction', '').strip()}",
        f"Current target path: {state.get('target_path', '')}",
    ]

    if context:
        lines.append("Injected context:")
        lines.extend(_format_context(context))

    helper_agent = helper_output.get("helper_agent")
    if helper_agent:
        lines.append("Helper-agent recommendation:")
        lines.append(f"- task_type: {helper_agent.get('task_type', '')}")
        lines.append(f"- recommendation: {helper_agent.get('recommendation', '')}")
        lines.append(f"- notes: {helper_agent.get('notes', '')}")

    edit_context = helper_output.get("edit_context")
    if edit_context:
        lines.append("Collected edit context:")
        for key in ("mode", "function_name", "line_count", "query_mode", "reason", "status"):
            value = edit_context.get(key)
            if value not in (None, ""):
                lines.append(f"- {key}: {value}")
        if edit_context.get("current_source"):
            lines.append("Current function source:")
            lines.append(edit_context["current_source"])

    if state.get("current_function_source") and not (edit_context and edit_context.get("current_source")):
        lines.append("Current function source:")
        lines.append(state["current_function_source"])

    if code_graph_status:
        lines.append("Code graph status:")
        for key in ("ready", "query_mode", "refresh_required", "context_collected", "reason"):
            value = code_graph_status.get(key)
            if value not in (None, ""):
                lines.append(f"- {key}: {value}")

    lines.append("If file_hint exists, prefer it as target_path.")
    lines.append("Use anchor for localized replacements, rename_symbol for file-local whole-word renames, write_file for full file replacement, append for adding to the end, prepend for adding to the beginning, delete_file for deleting a file, and copy_file for creating sibling copies of an existing file.")
    lines.append("For named_function mode, return the full replacement function body in replacement.")
    lines.append("For write_file, append, and prepend, put the full content in replacement and leave anchor null.")
    lines.append("For copy_file, set copy_count to the number of copies and leave anchor and replacement null.")
    lines.append("For create_files, set quantity to the number of new files to create, use target_path as the first file path, and leave anchor null. Use replacement as the initial file contents, or an empty string for blank files.")
    lines.append("For rename_symbol, set anchor to the old symbol name and replacement to the new symbol name.")
    lines.append("If the user is replacing an identifier-like token in a file, prefer rename_symbol over anchor.")
    lines.append("Use anchor only when the user is clearly targeting a specific literal snippet or block.")
    return "\n".join(lines).strip()


def _format_context(context: RuntimeContext) -> list[str]:
    lines: list[str] = []
    for key in (
        "spec_note",
        "test_failure",
        "file_hint",
        "search_text",
        "replace_text",
        "write_text",
        "append_text",
        "prepend_text",
        "helper_notes",
        "function_name",
    ):
        value = context.get(key)
        if value:
            lines.append(f"- {key}: {value}")
    return lines
