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
    pointers: list[dict[str, int]]
    tool_outputs: list[dict[str, Any]]


class StepSection(TypedDict, total=False):
    id: str
    instruction: str
    action_class: str
    edit_mode: str
    target_path: str
    anchor: str
    pattern: str
    command: str
    pointers: list[dict[str, int]]
    replacement_preview: str
    depends_on: list[str]
    inputs_from: list[str]
    status: str
    changed_files: list[str]
    no_op: bool
    retry_count: int
    timeout_seconds: int
    max_retries: int
    tool_name: str
    tool_source: str
    tool_args: dict[str, Any]


class TaskSection(TypedDict, total=False):
    task_id: str
    role: str
    agent_type: str
    parent_task_id: str
    child_task_ids: list[str]
    goal: str
    allowed_actions: list[str]
    status: str
    result: dict[str, Any]
    artifacts: dict[str, Any]
    depends_on: list[str]
    inputs_from: list[str]


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
    files: list[dict[str, Any]]
    pattern: str
    command: str
    source_path: str
    destination_path: str
    paths: list[str]
    pointers: list[dict[str, int]]
    occurrence_selector: str
    action_class: str
    timeout_seconds: int
    max_retries: int
    provider: str
    provider_reason: str
    valid: bool
    validation_errors: list[str]
    task_count: int


class ExecutionSection(TypedDict, total=False):
    status: str
    edit_applied: bool
    edit_attempts: int
    max_edit_attempts: int
    snapshot_path: str
    reverted_to_snapshot: bool
    reverted_files: list[str]
    revert_count: int
    target_existed_before_edit: bool
    error: str
    changed_files: list[str]
    file_preview: str
    file_preview_truncated: bool
    content_hash: str
    no_op: bool
    tool_output: dict[str, Any]
    verification_results: list[dict[str, Any]]
    verification_retry_count: int


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
    tasks: list[TaskSection]


class ShipyardState(TypedDict, total=False):
    session_id: str
    instruction: str
    # Broad repo context built once before planning (file tree + key file samples)
    broad_context: dict[str, Any]
    # Per-step context files loaded by fetch_step_context (path -> content)
    live_file_context: dict[str, str]
    # Last N completed runs for this session, injected before planning
    session_journal: list[dict[str, Any]]
    # Set to True to bypass the wide-impact gate for rename_symbol_global / update_imports
    wide_impact_approved: bool
    target_path: str
    anchor: str
    replacement: str
    quantity: int
    copy_count: int
    files: list[dict[str, Any]]
    pattern: str
    command: str
    pointers: list[dict[str, int]]
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
    file_transactions: list[dict[str, Any]]
    target_existed_before_edit: bool
    reverted_to_snapshot: bool
    reverted_files: list[str]
    revert_count: int
    edit_applied: bool
    edit_attempts: int
    max_edit_attempts: int
    verification_results: list[dict[str, Any]]
    verification_retry_count: int
    status: str
    error: str
    trace_path: str
    changed_files: list[str]
    file_preview: str
    file_preview_truncated: bool
    content_hash: str
    no_op: bool
    tool_output: dict[str, Any]
    tool_outputs: list[dict[str, Any]]
    tool_name: str
    tool_source: str
    tool_args: dict[str, Any]
    tool_result: dict[str, Any]
    graph_sync: dict[str, Any]
    spec_bundle: dict[str, Any]
    action_plan: ActionPlanSection
    preplanned_action: dict[str, Any]
    instruction_steps: list[str]
    action_steps: list[dict[str, Any]]
    tasks: list[TaskSection]
    human_gate: HumanGateSection
    request: RequestSection
    plan: PlanSection
    steps: list[StepSection]
    execution: ExecutionSection
    verification: VerificationSection
    graph: GraphSection
    artifacts: ArtifactSection
