from __future__ import annotations

from .state import ShipyardState


def enrich_state_sections(state: ShipyardState) -> ShipyardState:
    enriched = dict(state)
    proposal_summary = enriched.get("proposal_summary", {})
    graph_status = enriched.get("code_graph_status", {})
    graph_sync = enriched.get("graph_sync", {})
    spec_bundle = enriched.get("spec_bundle", {})

    enriched["request"] = {
        "session_id": enriched.get("session_id"),
        "instruction": enriched.get("instruction"),
        "target_path": enriched.get("target_path"),
        "quantity": enriched.get("quantity"),
        "copy_count": enriched.get("copy_count"),
        "proposal_mode": enriched.get("proposal_mode"),
        "proposal_model": enriched.get("proposal_model"),
        "context": enriched.get("context", {}),
        "verification_commands": enriched.get("verification_commands", []),
    }
    enriched["plan"] = {
        "target_path": enriched.get("target_path"),
        "target_path_source": proposal_summary.get("target_path_source"),
        "edit_mode": proposal_summary.get("edit_mode") or enriched.get("edit_mode"),
        "anchor": enriched.get("anchor"),
        "replacement": enriched.get("replacement"),
        "quantity": enriched.get("quantity"),
        "copy_count": enriched.get("copy_count"),
        "occurrence_selector": enriched.get("occurrence_selector"),
        "provider": proposal_summary.get("provider"),
        "provider_reason": proposal_summary.get("provider_reason"),
        "valid": proposal_summary.get("is_valid"),
        "validation_errors": proposal_summary.get("validation_errors", []),
    }
    enriched["execution"] = {
        "status": enriched.get("status"),
        "edit_applied": enriched.get("edit_applied"),
        "edit_attempts": enriched.get("edit_attempts"),
        "max_edit_attempts": enriched.get("max_edit_attempts"),
        "snapshot_path": enriched.get("snapshot_path"),
        "reverted_to_snapshot": enriched.get("reverted_to_snapshot"),
        "target_existed_before_edit": enriched.get("target_existed_before_edit"),
        "error": enriched.get("error"),
        "changed_files": enriched.get("changed_files", []),
        "file_preview": enriched.get("file_preview"),
        "file_preview_truncated": enriched.get("file_preview_truncated"),
        "content_hash": enriched.get("content_hash"),
        "human_gate": enriched.get("human_gate", {}),
    }
    enriched["verification"] = {
        "commands": enriched.get("verification_commands", []),
        "results": enriched.get("verification_results", []),
    }
    enriched["graph"] = {
        "status": graph_status,
        "sync": graph_sync,
        "ready": graph_status.get("ready"),
        "available": graph_status.get("available"),
        "reason": graph_status.get("reason"),
        "live_graph_populated": graph_status.get("live_graph_state", {}).get("populated"),
        "index_stale": graph_status.get("index_state", {}).get("stale"),
        "sync_attempted": graph_sync.get("attempted"),
        "sync_ok": graph_sync.get("ok"),
    }
    enriched["artifacts"] = {
        "trace_path": enriched.get("trace_path"),
        "prompt": enriched.get("prompt"),
        "spec_bundle": spec_bundle,
        "spec_created": spec_bundle.get("created"),
    }
    if enriched.get("action_plan"):
        enriched["artifacts"]["action_plan"] = enriched.get("action_plan")
    return enriched
