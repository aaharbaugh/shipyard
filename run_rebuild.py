#!/usr/bin/env python3
"""Feed rebuild plan instructions to Shipyard one at a time.

Usage:
    python run_rebuild.py                    # run all phases
    python run_rebuild.py --phase scaffold   # run specific phase
    python run_rebuild.py --from api-auth    # resume from specific phase
    python run_rebuild.py --dry-run          # show plan without executing
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure shipyard is importable
sys.path.insert(0, str(Path(__file__).parent))

from shipyard.main import run_once
from shipyard.graph import build_graph
from shipyard.session_store import SessionStore
from shipyard.workspaces import get_session_workspace


PLAN_PATH = Path(__file__).parent / "rebuild_plan.json"
SESSION_ID = "ship-rebuild"
RESULTS_PATH = Path(".shipyard/data/rebuild_results.json")


def load_plan() -> list[dict]:
    return json.loads(PLAN_PATH.read_text())


def run_phase(phase: dict, app, session_store: SessionStore) -> dict:
    """Run a single phase instruction."""
    phase_name = phase["phase"]
    instruction = phase["instruction"]

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*60}")
    print(f"Instruction: {instruction[:100]}...")
    print()

    state = {
        "session_id": SESSION_ID,
        "instruction": instruction,
    }

    t0 = time.monotonic()
    result = run_once(app, session_store, state)
    elapsed = time.monotonic() - t0

    status = result.get("status", "unknown")
    changed = result.get("changed_files") or []
    error = result.get("error") or ""
    steps = result.get("action_steps") or []

    print(f"\nResult: {status} ({elapsed:.1f}s)")
    print(f"Changed: {len(changed)} file(s)")
    if changed:
        for f in changed:
            print(f"  - {Path(f).name}")
    if error:
        print(f"Error: {error[:200]}")
    print(f"Steps: {len(steps)}")

    return {
        "phase": phase_name,
        "status": status,
        "elapsed": round(elapsed, 1),
        "changed_count": len(changed),
        "changed_files": [Path(f).name for f in changed],
        "step_count": len(steps),
        "error": error[:200] if error else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Shipyard rebuild plan")
    parser.add_argument("--phase", help="Run only this phase")
    parser.add_argument("--from", dest="from_phase", help="Resume from this phase")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    plan = load_plan()
    phases = [p["phase"] for p in plan]

    print(f"Rebuild plan: {len(plan)} phases")
    for i, p in enumerate(plan, 1):
        print(f"  {i}. {p['phase']}: {p['instruction'][:60]}...")
    print()

    # Filter phases
    if args.phase:
        plan = [p for p in plan if p["phase"] == args.phase]
        if not plan:
            print(f"Phase '{args.phase}' not found. Available: {', '.join(phases)}")
            return
    elif args.from_phase:
        try:
            start = phases.index(args.from_phase)
            plan = plan[start:]
        except ValueError:
            print(f"Phase '{args.from_phase}' not found. Available: {', '.join(phases)}")
            return

    if args.dry_run:
        print("DRY RUN — no execution")
        return

    # Clear workspace for fresh rebuild
    workspace = get_session_workspace(SESSION_ID)
    if workspace.exists() and not args.phase and not args.from_phase:
        print(f"Workspace: {workspace}")
        resp = input("Clear workspace for fresh rebuild? [y/N] ")
        if resp.lower() == "y":
            import shutil
            for item in workspace.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("Workspace cleared.")
        print()

    app = build_graph()
    session_store = SessionStore()
    results = []

    for phase in plan:
        result = run_phase(phase, app, session_store)
        results.append(result)

        if result["status"] in ("failed", "invalid_action_plan"):
            print(f"\n*** Phase '{phase['phase']}' failed. Stopping. ***")
            print("Fix the issue and resume with:")
            print(f"  python run_rebuild.py --from {phase['phase']}")
            break

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("REBUILD SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r["status"] in ("edited", "verified", "observed") else "✗"
        print(f"  {icon} {r['phase']}: {r['status']} ({r['elapsed']}s, {r['changed_count']} files)")
    passed = sum(1 for r in results if r["status"] in ("edited", "verified", "observed"))
    print(f"\n{passed}/{len(results)} phases completed")


if __name__ == "__main__":
    main()
