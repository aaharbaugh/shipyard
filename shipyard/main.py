from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

from .graph import build_graph
from .session_store import SessionStore
from .state import ShipyardState
from .tracing import write_trace


def parse_user_input(raw: str) -> ShipyardState:
    text = raw.strip()
    if not text:
        return {"instruction": ""}

    if text.startswith("{"):
        payload = json.loads(text)
        return _normalize_payload(payload)

    return {"instruction": text, "context": {}}


def read_user_input() -> str:
    first_line = input("> ")
    stripped = first_line.strip()

    if stripped.lower() in {"exit", "quit"}:
        return stripped

    if not stripped.startswith("{"):
        return first_line

    lines = [first_line]
    while True:
        try:
            json.loads("\n".join(lines).strip())
            return "\n".join(lines)
        except json.JSONDecodeError:
            next_line = input("... ")
            if not next_line.strip():
                return "\n".join(lines)
            lines.append(next_line)


def _normalize_payload(payload: dict[str, Any]) -> ShipyardState:
    return {
        "session_id": str(payload.get("session_id", "")).strip() or None,
        "instruction": str(payload.get("instruction", "")).strip(),
        "target_path": _coerce_optional_str(payload.get("target_path")),
        "anchor": _coerce_optional_str(payload.get("anchor")),
        "replacement": _coerce_optional_str(payload.get("replacement")),
        "proposal_mode": _coerce_optional_str(payload.get("proposal_mode")),
        "proposal_model": _coerce_optional_str(payload.get("proposal_model")),
        "edit_mode": _coerce_optional_str(payload.get("edit_mode")),
        "context": payload.get("context", {}) or {},
        "edit_attempts": int(payload.get("edit_attempts", 0) or 0),
        "max_edit_attempts": int(payload.get("max_edit_attempts", 2) or 2),
        "verification_commands": list(payload.get("verification_commands", []) or []),
    }


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def run_once(app, session_store: SessionStore, state: ShipyardState) -> ShipyardState:
    result = app.invoke(state)
    trace_path = write_trace(result)
    result["trace_path"] = trace_path
    session_store.append_run(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Shipyard MVP agent.")
    parser.add_argument("--session-id", help="Reuse an existing session id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load and display the latest saved state for the session id at startup.",
    )
    args = parser.parse_args()

    app = build_graph()
    session_store = SessionStore()
    session_id = args.session_id or uuid.uuid4().hex[:8]

    print("Shipyard MVP runner")
    print(f"session_id={session_id}")
    print("Enter plain text or JSON. Type 'exit' to stop.")

    if args.resume:
        resumed = session_store.load_latest_state(session_id)
        if resumed:
            print(f"resumed_status={resumed.get('status', 'unknown')}")
            print(f"resumed_instruction={resumed.get('instruction', '')}")
        else:
            print("resume_requested_but_no_saved_state_found")

    while True:
        raw = read_user_input()
        if raw.strip().lower() in {"exit", "quit"}:
            print("Stopping Shipyard MVP runner.")
            return

        try:
            state = parse_user_input(raw)
        except json.JSONDecodeError as exc:
            print(f"input_error=Invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}")
            print("hint=Paste the full JSON block, then press Enter on a blank line.")
            continue

        state["session_id"] = state.get("session_id") or session_id
        result = run_once(app, session_store, state)
        _print_result(result)
        print(f"trace={result.get('trace_path')}")


def _print_result(result: ShipyardState) -> None:
    print(f"status={result.get('status', 'unknown')}")

    prompt = result.get("prompt")
    if prompt:
        print("prompt:")
        print(prompt)

    snapshot_path = result.get("snapshot_path")
    if snapshot_path:
        print(f"snapshot={snapshot_path}")

    helper_output = result.get("helper_output")
    if helper_output:
        print("helper_output:")
        print(json.dumps(helper_output, indent=2))

    if result.get("reverted_to_snapshot"):
        print("reverted_to_snapshot=true")

    error = result.get("error")
    if error:
        print(f"error={error}")

    verification_results = result.get("verification_results", [])
    if verification_results:
        print("verification:")
        print(json.dumps(verification_results, indent=2))


if __name__ == "__main__":
    main()
