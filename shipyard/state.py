from __future__ import annotations

from typing import Any, TypedDict


class RuntimeContext(TypedDict, total=False):
    spec_note: str
    test_failure: str
    file_hint: str
    search_text: str
    replace_text: str
    write_text: str
    append_text: str
    prepend_text: str
    helper_notes: str
    function_name: str
    occurrence_selector: str


class RequestSection(TypedDict, total=False):
    session_id: str
    instruction: str
    target_path: str
    proposal_mode: str
    proposal_model: str
    context: RuntimeContext
    verification_commands: list[str]


class PlanSection(TypedDict, total=False):
    target_path: str
    target_path_source: str
    edit_mode: str
    anchor: str
    replacement: str
    quantity: int
    copy_count: int
    occurrence_selector: str
    provider: str
    provider_reason: str
    valid: bool
    validation_errors: list[str]


class ExecutionSection(TypedDict, total=False):
    status: str
    edit_applied: bool
    edit_attempts: int
    max_edit_attempts: int
    snapshot_path: str
    reverted_to_snapshot: bool
    target_existed_before_edit: bool
    error: str
    changed_files: list[str]
    file_preview: str
    file_preview_truncated: bool
    content_hash: str


class VerificationSection(TypedDict, total=False):
    commands: list[str]
    results: list[dict[str, Any]]


class GraphSection(TypedDict, total=False):
    status: dict[str, Any]
    sync: dict[str, Any]


class HumanGateSection(TypedDict, total=False):
    status: str
    reason: str
    action: str
    prompt: str
    details: dict[str, Any]


class ArtifactSection(TypedDict, total=False):
    trace_path: str
    prompt: str
    spec_bundle: dict[str, Any]


class ActionPlanSection(TypedDict, total=False):
    actions: list[dict[str, Any]]
    provider: str
    provider_reason: str


class ShipyardState(TypedDict, total=False):
    session_id: str
    instruction: str
    target_path: str
    anchor: str
    replacement: str
    quantity: int
    copy_count: int
    occurrence_selector: str
    proposal_mode: str
    proposal_model: str
    edit_mode: str
    context: RuntimeContext
    verification_commands: list[str]
    prompt: str
    helper_output: dict[str, Any]
    proposal_summary: dict[str, Any]
    code_graph_status: dict[str, Any]
    current_function_source: str
    file_before: str
    snapshot_path: str
    target_existed_before_edit: bool
    reverted_to_snapshot: bool
    edit_applied: bool
    edit_attempts: int
    max_edit_attempts: int
    verification_results: list[dict[str, Any]]
    status: str
    error: str
    trace_path: str
    changed_files: list[str]
    file_preview: str
    file_preview_truncated: bool
    content_hash: str
    graph_sync: dict[str, Any]
    spec_bundle: dict[str, Any]
    action_plan: ActionPlanSection
    instruction_steps: list[str]
    human_gate: HumanGateSection
    request: RequestSection
    plan: PlanSection
    execution: ExecutionSection
    verification: VerificationSection
    graph: GraphSection
    artifacts: ArtifactSection
