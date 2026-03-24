# MVP Execution Plan

## Goal

Build the smallest LangGraph-based coding agent that passes the MVP hard gate from the
Shipyard PRD:

- persistent loop
- surgical file editing
- runtime context injection
- tracing with two distinct runs
- `PRESEARCH.md`
- `CODEAGENT.md`
- local GitHub-runnable setup

## Guiding Principle

A narrow, reliable agent is better than a broad, fragile one. The PRD explicitly favors
a focused agent that edits surgically and runs continuously over a feature-rich agent
that crashes or rewrites whole files.

## Scope For MVP

### In Scope

- LangGraph local runtime
- one persistent lead-agent loop
- stdin instruction intake
- Code-Graph-RAG-first surgical editing for named functions
- Code-Graph-RAG indexing and graph-availability checks
- snapshot-before-write safety
- runtime context injection
- verification step with local subprocesses
- traceable runs
- minimal sequential helper-agent design
- required MVP documentation

### Out of Scope For First Pass

- GitHub REST automation
- branch creation automation
- full Ship rebuild
- advanced conflict resolution
- parallel multi-agent writes
- deployment

## Recommended Build Order

### Phase 1: Skeleton Runtime

Deliverable: a process that stays alive and accepts multiple instructions.

Tasks:
- create project structure
- set up Python environment and dependencies
- create a LangGraph state schema
- implement a simple stdin instruction loop
- support thread/session id for trace grouping

Success check:
- process accepts at least two instructions without restarting

### Phase 2: Surgical Editing

Deliverable: one safe edit path that changes a specific block without rewriting the
entire file.

Tasks:
- stand up the required Code-Graph-RAG runtime dependencies
- document and verify how the local repo gets indexed before edits are allowed
- add a graph-readiness check so function edits do not run against an unindexed repo
- integrate Code-Graph-RAG query flow for named functions
- implement `query_codebase()` or `semantic_search()` before edits
- implement `get_function_code(file, function)`
- implement `replace_code(file, function, new_code)`
- implement `snapshot_file`
- keep anchor fallback only for non-function targets

Success check:
- agent can replace a known function surgically while leaving the rest of the file untouched
- agent refuses function edits clearly when the code graph is unavailable or stale

### Phase 3: Context Injection

Deliverable: agent accepts external runtime context and uses it in the next action.

Tasks:
- define a small instruction payload shape
- support optional context fields such as:
  - `spec_note`
  - `test_failure`
  - `file_hint`
- inject these values into graph state and prompt construction

Success check:
- two runs on the same task behave differently when different context is injected

### Phase 3.5: Code-Graph-RAG Operational Wiring

Deliverable: Code-Graph-RAG is treated as a real subsystem, not just a library import.

Tasks:
- use local `cgr` CLI orchestration as the chosen integration path
- use `cgr doctor` as the first dependency/configuration health check
- use `cgr index` to build the graph/index artifacts before function edits are allowed
- define when `cgr start`, `cgr stats`, or `cgr export` are used in development workflows
- document the Memgraph dependency and startup sequence
- define how graph refresh happens after file edits
- define how the agent detects and handles stale graph state

Success check:
- another engineer can boot the graph-backed edit path from the repo docs
- the agent can explain when it is using Code-Graph-RAG versus the fallback path

### Phase 4: Verification And Recovery

Deliverable: agent can detect bad edits and retry safely.

Tasks:
- add local verification commands
- capture stdout and stderr
- feed failures back into the next attempt
- set a small retry limit
- pause cleanly after repeated failure

Success check:
- one trace shows a successful path
- one trace shows a failure or retry path

### Phase 5: Minimal Multi-Agent Support

Deliverable: lead agent can delegate one narrow subtask and merge the result safely.

Tasks:
- define helper-agent prompt contract
- call helper agent for one bounded task
- keep file writes owned by lead agent only
- log delegation in traces

Success check:
- a trace shows helper-agent participation in sequence

### Phase 6: Documentation And Submission Readiness

Deliverable: required MVP docs are complete and aligned with the actual implementation.

Tasks:
- keep `PRESEARCH.md` updated
- fill MVP sections in `CODEAGENT.md`
- add setup and run instructions
- collect two trace links

Success check:
- another engineer can clone, run, and understand the architecture without asking for clarification

## Suggested Project Structure

```text
shipyard/
  PRESEARCH.md
  CODEAGENT.md
  MVP_EXECUTION_PLAN.md
  README.md
  requirements.txt
  shipyard/
    __init__.py
    main.py
    graph.py
    state.py
    prompts.py
    tools/
      read_file.py
      edit_file.py
      snapshot.py
      verify.py
```

## MVP Technical Choices

- Framework: LangGraph
- Language: Python
- Loop entry: stdin first, FastAPI later if needed
- Editing strategy: Code-Graph-RAG AST replacement for named functions, fallback path for non-function edits only
- Code graph backend: Code-Graph-RAG via the `cgr` CLI plus its graph/indexing runtime
- Verification: local subprocess commands such as `pytest` or `ruff`
- Tracing: LangSmith if available, otherwise structured local logging while wiring the graph

## Definition Of Done For MVP

The MVP is done when all of the following are true:

1. The agent process stays alive and handles multiple instructions.
2. The agent performs a targeted edit without rewriting the whole file.
3. Runtime context can be injected and affects the next action.
4. Two traceable runs exist with different execution paths.
5. `PRESEARCH.md` and `CODEAGENT.md` are present and aligned with the implementation.

## Immediate Next Steps

1. Create the Python project skeleton.
2. Implement the persistent LangGraph loop with stdin intake.
3. Use `cgr` as the fixed Code-Graph-RAG operating mode for this repo.
4. Wire the `cgr`-backed function edit path and keep non-function fallback isolated.
5. Add graph-readiness checks, context payload support, and tracing around the `cgr` path.
