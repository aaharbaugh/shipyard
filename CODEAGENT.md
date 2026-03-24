# CODEAGENT.md

## Agent Architecture (MVP)

Shipyard MVP uses a LangGraph-based local coding agent built around a persistent
lead-agent loop. The system runs as a long-lived Python process so it can accept
multiple instructions without restarting. The implementation prioritizes the
hard-gate MVP requirements from the PRD over broader platform ambitions.

### MVP Loop

1. Start local runtime with a session/thread id
2. Wait for user instruction from stdin
3. Accept optional injected context payload at runtime
4. Build the next prompt from:
   - user instruction
   - injected context
   - persistent project context files
   - recent tool results
5. Run the lead agent through a LangGraph flow
6. Call a helper-planning step to derive or confirm the edit spec
7. Read target file(s)
8. Validate and apply the edit
9. Run verification tools
10. Revert from snapshot if verification fails
11. Persist latest session state and run history locally
12. Log the run and expose a trace
13. Return to waiting state for the next instruction

### State Management

- LangGraph manages node execution and transition state
- Session state is keyed by thread id
- Runtime context includes:
  - active instruction
  - injected context payload
  - target file path(s)
  - edit attempt count
  - verification output
  - trace metadata
- Session history is stored locally under `.shipyard/sessions/<session_id>/`
- The same core run path is exposed through stdin and a FastAPI endpoint

### Entry Conditions

- Process is started locally
- A new instruction is received
- Optional runtime context is supplied with the instruction

### Exit Conditions

- Successful edit and verification
- Safe failure after retry limit is reached
- Human stop command

### Error Branch

For named-function edits, the current implementation first performs a Code-Graph-RAG
readiness check and records graph/index metadata in state. If the graph is unavailable,
the run stops cleanly with `graph_unavailable` rather than silently falling back. When
the graph is available and a replacement function body is provided, Shipyard uses the
Code-Graph-RAG file editor primitives for surgical replacement. If verification fails
after a write, the failure output is injected back into the next attempt so the agent
can repair the edit or pause for human intervention. The MVP also reverts the file
from the latest snapshot before returning a failed status.

## File Editing Strategy (MVP)

The preferred MVP editing path is Code-Graph-RAG-targeted function replacement.
For named functions, the runtime checks graph readiness, uses Code-Graph-RAG file
editor support to locate the current function source, and replaces only that function
body. Anchor-based replacement remains the fallback for non-function targets such as
config blocks, class variables, or inline logic with no function wrapper.

### Step-by-Step Mechanism

1. Helper-planner contributes scoped notes for the current task
2. For named functions, the runtime performs a Code-Graph-RAG readiness check
3. The lead agent proposes a full replacement for the named function or anchor target
4. System snapshots the file before editing
5. For named functions, Shipyard performs surgical replacement through the
   Code-Graph-RAG file editor; otherwise it uses anchor replacement
6. Verification runs immediately after the write
7. If verification fails, the failure output is fed back into the next attempt
8. If the graph-backed path writes successfully, Shipyard marks the graph index stale
   and records that a refresh is required

### How the Correct Block Is Located

For named functions, location is determined through the Code-Graph-RAG parsing/editor
stack rather than brittle line numbers. For non-function targets, Shipyard still
requires a unique anchor match before writing.

### What Happens When Location Is Wrong

- Function not found or ambiguous: block the write and return a clear edit error
- Verification failure after write: use snapshot for rollback if needed, re-read the
  function, and retry with error context
- Non-function target: use the fallback edit path only when AST function replacement
  does not apply

## Multi-Agent Design (MVP)

The MVP design now supports one lead agent and one helper agent in sequence. This is
the simplest version that still satisfies the requirement to support more than one
agent without introducing parallel merge complexity too early.

### Orchestration Model

- Lead agent owns the main loop and all file writes
- Helper agent receives a narrow subtask such as:
  - identify whether the run is a named-function edit or anchor edit
  - contribute scoped helper notes before the lead agent finalizes the edit plan
  - summarize verification failure for the next retry
- Helper agent returns structured output to the lead agent
- Lead agent decides whether to apply, retry, or reject the suggestion

### Communication

- Shared task summary
- Relevant file excerpt
- Optional verification output
- Structured helper response with proposed change or recommendation

### Merge Strategy

Only the lead agent writes files in MVP. This avoids write conflicts and makes tracing
much easier.

## Trace Links (MVP)

- Trace 1 (normal run): https://smith.langchain.com/o/25524cca-82d8-4066-bd77-36d34114ef63/projects/p/9e3795fc-e683-4858-b1c8-739795bd9a82/r/019d1cd3-caa2-7721-a287-da815bb43f18?trace_id=019d1cd3-caa2-7721-a287-da815bb43f18&start_time=2026-03-23T22:32:15.266775
- Trace 2 (blocked edit): https://smith.langchain.com/o/25524cca-82d8-4066-bd77-36d34114ef63/projects/p/9e3795fc-e683-4858-b1c8-739795bd9a82/r/019d1cd5-c970-7b71-966c-91fa97652b4f?trace_id=019d1cd5-c970-7b71-966c-91fa97652b4f&start_time=2026-03-23T22:34:26.032408

## Architecture Decisions (Final Submission)

- LangGraph remains the orchestration framework because the project needed explicit
  state transitions, retries, and a persistent local loop more than autonomous breadth.
- The stdin runner stayed as the primary local entrypoint, with FastAPI added in
  parallel for testability and programmatic control.
- Snapshot-before-write and rollback remained mandatory. This preserved a narrow safety
  contract even as the function-edit path became more capable.
- Non-function edits continue to use anchor replacement because it is deterministic and
  easy to reason about in an MVP.
- Named-function edits now go through a graph-readiness gate and the Code-Graph-RAG
  file editor path. This gives Shipyard a real surgical function-edit mode while still
  surfacing graph operational failures clearly.
- Code-graph operations are treated as a first-class subsystem. The runtime now tracks
  index artifacts, stale graph state, and refresh requirements after graph-backed edits.
- The multi-agent requirement is satisfied through a sequential helper-planner step.
  Only the lead agent writes files, which keeps merge behavior and trace interpretation
  simple.

## Ship Rebuild Log (Final Submission)

The rebuild progressed in small, safety-first stages:

1. Establish the baseline loop, session persistence, trace writing, verification, and
   rollback behavior.
2. Add provider-backed proposal generation with a deterministic heuristic fallback so
   the agent could accept either explicit edit specs or inferred replacements.
3. Expose the same core run path through FastAPI for easier automation and inspection.
4. Introduce git automation helpers with clear repo-safety failures outside git repos.
5. Add Code-Graph-RAG readiness checks so named-function requests stop cleanly when the
   graph runtime is unavailable.
6. Wire a real named-function replacement path through the Code-Graph-RAG file editor.
7. Add graph operational helpers for status, indexing, local artifact inspection, and
   stale detection.
8. Mark graph refresh requirements after graph-backed edits so the runtime records when
   index maintenance is needed.
9. Improve the interactive runner to accept pasted multi-line JSON and handle malformed
   JSON without crashing the process.
10. Add the sequential helper-planner step so helper-agent participation is represented
    explicitly in the graph, saved state, and traces.

## Comparative Analysis (Final Submission)

Compared with the systems documented in `PRESEARCH.md`, Shipyard deliberately chooses a
smaller and more inspectable implementation:

- Compared with Claude Code:
  Shipyard mirrors the snapshot-before-edit safety instinct and the repo-local context
  file approach, but it keeps its workflow explicit in LangGraph state rather than
  hiding behavior behind a larger product runtime.
- Compared with OpenCode:
  Shipyard avoids unified-diff-first editing for the MVP because anchor replacement and
  named-function replacement are easier to make deterministic in a narrow scaffold.
  It also keeps verification feedback in the retry loop, which was one of the most
  valuable ideas from the research.
- Compared with LangChain Open Engineer:
  Shipyard adopts the graph/state-machine mindset and now includes a minimal helper
  agent, but it intentionally avoids full supervisor complexity, parallelism, and
  broad tool routing.

The project’s main tradeoff is depth versus breadth. It is narrower than the reference
systems, but much easier to understand, test, and run locally inside one repository.

## Cost Analysis (Final Submission)

The current MVP is inexpensive by default because most of the runtime is local:

- The default planning path is heuristic and deterministic, so local testing and most
  scripted runs can avoid external model cost entirely.
- Verification, snapshots, traces, sessions, and graph artifact inspection are local
  filesystem/subprocess operations.
- The optional OpenAI-backed proposal path is the only built-in feature that introduces
  model API cost, and it is scoped to edit proposal generation rather than the entire
  orchestration loop.
- The Code-Graph-RAG dependency introduces operational cost mainly through local graph
  infrastructure and indexing time rather than per-request API usage.

For the MVP, the practical cost profile is:

- low cost for local deterministic runs
- moderate local operational overhead when using graph indexing and Memgraph
- optional incremental API cost when enabling model-backed edit proposals
