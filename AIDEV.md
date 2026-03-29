# AI Development Log

## Architecture Overview

Shipyard is an LLM-powered coding agent built on LangGraph. It takes natural-language instructions,
plans a sequence of inspect/edit/verify actions, and executes them against a sandboxed workspace.

### Core Pipeline

```
instruction -> plan_actions -> [for each step: fetch_step_context -> plan_edit -> validate -> apply_edit -> verify -> recover_or_finish]
```

### Key Design Decisions

#### Inspect-First Pattern (2026-03)
The agent enforces an **inspect-first** approach: before editing any existing file, the plan must
include a `read_file` or `run_command` step to see the current file state. This prevents the LLM
from generating stale anchors or blind rewrites.

- **Why**: Anchor edits generated without seeing the file fail when the file has duplicates, syntax
  errors, or has changed since the instruction was written.
- **Trace**: `action_planner.py:_build_action_plan_prompt` contains the INSPECT-FIRST RULE section.

#### Simplified Edit Modes (2026-03)
Primary edit modes are:
1. `write_file` -- full file rewrite, best for small/broken files
2. `search_and_replace` -- exact text find-and-replace, best for targeted changes
3. `append` / `prepend` -- add content to file boundaries

Legacy modes (`anchor` with pointers, `named_function`) are still supported but no longer
recommended in planning prompts. The `named_function` mode auto-degrades to `write_file`.

- **Why**: Anchor-with-pointers is fragile (character offsets drift). Named-function requires
  Code Graph RAG (Memgraph) which is often unavailable.
- **Trace**: `prompts.py:build_proposal_prompt`, `graph.py:check_edit_readiness`

#### Code Graph RAG Removed as Hard Dependency (2026-03)
The agent no longer requires Memgraph/Code Graph RAG for `named_function` edits. Instead:
- `check_edit_readiness` degrades `named_function` to `write_file` automatically
- `collect_edit_context` attempts grep-based function extraction, falls back to full file contents
- The LLM uses the file contents (loaded by `fetch_step_context`) to generate edits

This follows the "grep + cat" principle: simple tools are more robust than complex RAG pipelines.

- **Trace**: `graph.py:check_edit_readiness`, `graph.py:collect_edit_context`

#### Pre-Planning Syntax Detection (2026-03)
Before `plan_actions` runs, the runtime scans workspace files for syntax errors and injects a
WARNING into the planning context. This ensures the LLM knows to use `write_file` for broken files.

- **Trace**: `main.py:_detect_workspace_syntax_errors`

#### Workspace Path Sandboxing (2026-03)
Relative paths like `app.js` from the LLM are resolved against the session workspace
(`.shipyard/data/workspace/default/`) instead of CWD. This prevents writes to the wrong directory.

- **Why**: The LLM returns bare filenames; without sandboxing, `Path("app.js")` resolves to the
  project root instead of the agent's managed workspace.
- **Trace**: `graph.py:_sandbox_target_path`, applied in `fetch_step_context` and `apply_edit`

#### Parallel Batch Execution (2026-03)
Independent actions targeting different files are detected and executed concurrently using
`ThreadPoolExecutor`. The batch detector groups consecutive actions that share the same
`action_class`, target different files, and have no cross-dependencies.

- **Why**: Multi-file edits (e.g., editing `app.js`, `index.html`, `styles.css` simultaneously)
  complete faster when parallelized. The LLM planner generates inspect-first plans with
  3 reads followed by 3 writes — both groups run as parallel batches.
- **Trace**: `main.py:_find_parallel_batch`, `main.py:_execute_batch_parallel`

### Trace Links

| Component | File | Key Function |
|-----------|------|-------------|
| Action planning | `shipyard/action_planner.py` | `plan_actions`, `_build_action_plan_prompt` |
| Proposal generation | `shipyard/prompts.py` | `build_proposal_prompt`, `build_runtime_prompt` |
| Graph nodes | `shipyard/graph.py` | `check_edit_readiness`, `collect_edit_context`, `apply_edit`, `recover_or_finish` |
| Proposal validation | `shipyard/proposal_validation.py` | `validate_proposal` |
| Syntax detection | `shipyard/main.py` | `_detect_workspace_syntax_errors` |
| Workspace management | `shipyard/workspaces.py` | `get_session_workspace` |
| Context exploration | `shipyard/context_explorer.py` | `build_broad_context`, `detect_project_stack` |
| Path sandboxing | `shipyard/graph.py` | `_sandbox_target_path` |
| Parallel execution | `shipyard/main.py` | `_find_parallel_batch`, `_execute_batch_parallel` |
| Inspect-first check | `shipyard/action_plan_validation.py` | `check_inspect_first` |

### Testing

Run all tests:
```bash
python -m pytest tests/ -x -q
```

Tests cover: action planning, graph flow, proposal validation, pathing, workspaces, and the main run loop.
