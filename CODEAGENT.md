# CODEAGENT.md

## Agent Architecture

Shipyard is an LLM-powered coding agent built on LangGraph. It takes natural-language instructions, plans a sequence of actions, and executes them against a workspace — either a managed sandbox or a real repository.

### System Flow

```
User instruction
  → build_broad_context (file tree, project stack, doc discovery)
  → plan_actions (LLM generates ordered action steps)
  → should_use_supervisor?
    → YES: plan_subtasks → parallel worker execution
    → NO: sequential execution
  → auto_branch (create feature branch from main)
  → for each action step:
      → fetch_step_context (sandbox paths, read target file, cache)
      → plan_edit (refine preplanned action with live file content)
      → validate_proposal
      → apply_edit (search_and_replace, write_file, append, etc.)
      → verify_edit (auto syntax check)
      → recover_or_finish (retry or move on)
  → auto_test (discover and run project test suite)
  → collect_diffs (unified diffs of all changes)
  → auto_rollback (revert if edits failed, keep if verify failed)
  → persist result + trace
```

### Core Components

| Component | File | Purpose |
|---|---|---|
| Run loop | `main.py` | Persistent loop, run_once orchestration, auto-branch/test/rollback |
| Action planner | `action_planner.py` | LLM-based multi-step plan generation with inspect-first pattern |
| Graph executor | `graph.py` | LangGraph state machine — per-step edit cycle |
| Supervisor | `supervisor.py` | Task decomposition for multi-agent execution |
| Worker orchestrator | `worker_orchestrator.py` | Parallel worker spawning, conflict detection, result merge |
| Proposal engine | `proposal.py` + `prompts.py` | LLM proposal generation for individual edits |
| Validation | `proposal_validation.py` + `action_plan_validation.py` | Structural validation of plans and proposals |
| Context explorer | `context_explorer.py` | File tree scanning, project stack detection, doc discovery |
| API + UI | `api.py` | FastAPI server with embedded workbench UI |
| Queue | `run_queue.py` | Job queue with task hierarchy tracking |
| Workspaces | `workspaces.py` | Session workspace management, external repo binding |

### Error Recovery

- **Plan invalid**: LLM repair call (nano model), then re-validate
- **Step fails**: adaptive replan of remaining steps (up to 2 replans)
- **Verify fails**: revert via snapshot, retry (up to 2 attempts)
- **Content loss**: write_file blocked if replacement loses >30% of existing content
- **Run fails**: auto-rollback reverts all changes if no edit step succeeded
- **Job vanishes**: frontend stops polling, returns terminal status

## File Editing Strategy

Shipyard uses **search_and_replace** as the primary surgical edit mode. The agent reads the file first (inspect-first pattern), then replaces only the exact text that needs to change.

### How It Works

1. **Inspect**: `read_file` loads the target file into the LLM's context
2. **Plan**: LLM generates `anchor` (exact text to find) and `replacement` (new text)
3. **Apply**: `str.replace(anchor, replacement)` — only the matched text changes
4. **Verify**: auto syntax check (`node --check`, `python -m py_compile`, `ruff`)
5. **Snapshot**: pre-edit snapshot enables rollback on verification failure

### Why search_and_replace

- **Cannot nuke content**: only changes what it matches, everything else preserved
- **Auditable**: the anchor shows exactly what was found, the replacement shows what changed
- **Robust**: works on any file type, no AST dependency, no line number drift

### Fallback Modes

| Mode | When Used |
|---|---|
| `write_file` | Broken files with syntax errors, new files, files <10 lines |
| `append` | Auto-downgrade from write_file when instruction is additive on large valid files |
| `anchor` | Legacy mode, still supported for exact literal matches |

### Safety Guards

- write_file blocked if it would lose >30% of existing content
- Anchor edits blocked on files with pre-existing syntax errors
- All writes sandboxed to session workspace (relative paths resolved)
- Snapshot taken before every mutation for rollback

### What happens when it gets the location wrong

1. `_should_refine_preplanned_action` detects stale anchors (0 or >1 matches)
2. `_refine_preplanned_action` re-reads the file and re-prompts the LLM with live content
3. If refinement fails, `_replan_mutate_step_from_current_file` tries a full replan
4. If all retries fail, the step is marked failed and remaining steps continue or replan

## Multi-Agent Design

### Architecture

```
Supervisor (LLM call)
  → Decomposes instruction into scoped sub-tasks
  → Detects file/directory scope per worker
  → Identifies dependencies between sub-tasks

Worker Orchestrator (ThreadPoolExecutor)
  → Groups sub-tasks into execution waves
  → Wave N: all tasks with no unmet dependencies run in parallel
  → Each worker gets:
    - Own instruction (scoped sub-task)
    - Own broad_context (filtered to its scope)
    - Own action plan (generated independently)
    - Shared LangGraph (thread-safe, stateless)
  → Results merged after each wave

Conflict Detector
  → Checks if any two workers edited the same file
  → Reports conflicts with severity
```

### When Multi-Agent Activates

`should_use_supervisor()` returns True when:
- Instruction contains decomposition signals: refactor, redesign, implement, migrate, across
- Multiple monorepo packages mentioned (api + web + shared)
- Explicit "multi-agent" or "parallel" in instruction

Falls back to single-agent for simple/short instructions.

### Communication Model

Workers do NOT communicate during execution. Each worker:
1. Plans its own actions via `plan_actions()`
2. Executes via `_run_action_plan()` (same as single-agent)
3. Returns changed_files, action_steps, diffs

The supervisor merges results post-execution. Conflicts are detected but not auto-resolved — the last writer wins, and conflicts are reported.

### Dependency Waves

```
Wave 1 (parallel): w1(shared/types), w2(api/routes) — no deps
Wave 2 (sequential): w3(web/components) — depends_on: [w1]
```

Tasks with `depends_on` wait for their dependencies to complete before starting.

## Trace Links

- Trace 1 (normal multi-file edit): _to be filled after ship rebuild_
- Trace 2 (error recovery path): _to be filled after ship rebuild_

Local traces are written to `.shipyard/data/traces/` as JSON files (550+ traces captured during development). LangSmith integration is configured via `LANGSMITH_API_KEY` environment variable.

## Architecture Decisions

### 1. search_and_replace over unified diff
**Considered**: unified diff (git-style patches), AST-based editing, line-range replacement, anchor-based replacement.
**Chose**: search_and_replace — exact text match + replace. The LLM copies the anchor verbatim from the file and provides the replacement.
**Why**: Unified diffs require the LLM to produce well-formed patch syntax (fragile). AST is language-specific. Line numbers drift. search_and_replace is simple, auditable, and physically cannot delete content it doesn't match.
**Trade-off**: Requires the anchor to appear exactly once. Handled by `_should_refine_preplanned_action` which detects 0 or >1 matches and re-prompts.

### 2. Inspect-first over blind planning
**Considered**: generate edits from instruction alone, generate edits with file samples.
**Chose**: mandatory inspect step before any edit — `read_file` or `run_command` must precede mutations.
**Why**: The LLM generates better anchors when it sees the actual file content, not a 2KB sample. Eliminates stale anchor failures.
**Trade-off**: Adds 1 LLM call per file (the read). Worth it for reliability.

### 3. Model tiering (primary + nano)
**Chose**: gpt-5.4-mini for planning/code generation, gpt-4.1-nano for repair/replan/exploration.
**Why**: Repair and exploration don't generate code — they fix JSON structure or pick filenames. Nano is ~10x cheaper and faster for these tasks.

### 4. Parallel batch execution over pure sequential
**Chose**: detect independent actions (different files, no cross-deps) and run them concurrently.
**Why**: Multi-file edits (3 reads, then 3 writes) complete in ~2 steps instead of 6. No correctness risk since files don't overlap.

### 5. Content loss guard over prompt engineering
**Considered**: telling the LLM to "preserve all content" in write_file mode.
**Chose**: hard runtime guard — write_file blocked if replacement loses >30% of existing content.
**Why**: Prompt engineering doesn't work. The LLM will drop content it considers irrelevant no matter how many times you tell it not to. The guard makes it structurally impossible.

## Ship Rebuild Log

_Running log populated during rebuild — see `.shipyard/data/rebuild_log.jsonl`_

## Comparative Analysis

_To be completed after ship rebuild._

## Cost Analysis

_To be completed after ship rebuild._
