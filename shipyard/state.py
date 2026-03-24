from __future__ import annotations

from typing import Any, TypedDict


class RuntimeContext(TypedDict, total=False):
    spec_note: str
    test_failure: str
    file_hint: str
    search_text: str
    replace_text: str
    helper_notes: str
    function_name: str


class ShipyardState(TypedDict, total=False):
    session_id: str
    instruction: str
    target_path: str
    anchor: str
    replacement: str
    proposal_mode: str
    proposal_model: str
    edit_mode: str
    context: RuntimeContext
    verification_commands: list[str]
    prompt: str
    helper_output: dict[str, Any]
    code_graph_status: dict[str, Any]
    file_before: str
    snapshot_path: str
    reverted_to_snapshot: bool
    edit_applied: bool
    edit_attempts: int
    max_edit_attempts: int
    verification_results: list[dict[str, Any]]
    status: str
    error: str
    trace_path: str
