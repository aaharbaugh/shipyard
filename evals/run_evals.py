from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shipyard.graph import build_graph


def run_eval_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        success_file = root / "success_demo.py"
        success_file.write_text('print("before")\n', encoding="utf-8")
        success_result = build_graph().invoke(
            {
                "instruction": 'replace "before" with "after"',
                "proposal_mode": "heuristic",
                "context": {"file_hint": str(success_file)},
                "verification_commands": [f"python3 -m py_compile {success_file}"],
            }
        )
        cases.append(
            {
                "name": "successful_edit",
                "expected_status": "verified",
                "actual_status": success_result["status"],
                "passed": success_result["status"] == "verified"
                and success_file.read_text(encoding="utf-8") == 'print("after")\n',
            }
        )

        blocked_file = root / "blocked_demo.py"
        blocked_file.write_text('print("fixed")\n', encoding="utf-8")
        blocked_result = build_graph().invoke(
            {
                "instruction": "blocked",
                "proposal_mode": "heuristic",
                "target_path": str(blocked_file),
                "anchor": 'print("missing")',
                "replacement": 'print("never")',
            }
        )
        cases.append(
            {
                "name": "blocked_edit",
                "expected_status": "edit_blocked",
                "actual_status": blocked_result["status"],
                "passed": blocked_result["status"] == "edit_blocked"
                and blocked_file.read_text(encoding="utf-8") == 'print("fixed")\n',
            }
        )

        rollback_file = root / "rollback_demo.py"
        rollback_file.write_text('print("stable")\n', encoding="utf-8")
        rollback_result = build_graph().invoke(
            {
                "instruction": "rollback",
                "proposal_mode": "heuristic",
                "target_path": str(rollback_file),
                "anchor": 'print("stable")',
                "replacement": 'print("unterminated)',
                "verification_commands": [f"python3 -m py_compile {rollback_file}"],
                "max_edit_attempts": 1,
            }
        )
        cases.append(
            {
                "name": "rollback_after_failed_verification",
                "expected_status": "failed_after_retries",
                "actual_status": rollback_result["status"],
                "passed": rollback_result["status"] == "failed_after_retries"
                and rollback_result["reverted_to_snapshot"] is True
                and rollback_file.read_text(encoding="utf-8") == 'print("stable")\n',
            }
        )

    return cases


def main() -> None:
    results = run_eval_cases()
    summary = {
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
