# CODEAGENT.md

## Agent Architecture

Shipyard should be understood as an LLM-first coding agent with a constrained runtime.

The intended architecture is:

1. accept a user instruction
2. provide context and available tools to the model
3. let the model plan structured actions
4. execute those actions through controlled runtime tools
5. verify, persist, and trace the result

This is intentionally different from a parser-led command runner. The long-term system
should not depend on adding more regex patterns for every new phrasing. Parser logic
exists only as a fallback for simple and obvious cases.

### Core Runtime Components

- `api.py`
  Web workbench and HTTP endpoints
- `main.py`
  Main run loop and LangGraph invocation path
- `action_planner.py`
  Multi-action planning layer
- `proposal.py`
  Per-step planning and normalization
- `graph.py`
  Execution graph and edit application
- `run_queue.py`
  Queue state and run progression
- `session_store.py`
  Local session persistence
- `langsmith_config.py`
  LangGraph/LangSmith run config

### Planning Model

The primary planner is the LLM.

When OpenAI is configured, Shipyard should prefer:

- model-generated action plans
- model-generated edit proposals
- model-guided decomposition of multi-step requests

Heuristics should only be used when:

- the model is unavailable
- heuristic mode is explicitly requested

### Action Model

The runtime should converge on a stable action set rather than a growing list of
parser-specific verbs. Examples:

- `write_file`
- `append`
- `prepend`
- `replace_text`
- `rename_symbol`
- `copy_file`
- `create_files`
- `delete_file`
- `named_function_edit`
- `run_command`

The model should decide which action to use. The executor should validate and run it.

### Queue and Session Model

Shipyard is a persistent loop, not a fire-and-forget script.

- prompts are accepted into a queue
- each run has a session id and run/job state
- the UI should restore the active run and last session state cleanly
- work should stay inside one session workspace unless the user explicitly overrides it

### Graph Layer

Shipyard uses Code-Graph-RAG with Memgraph for graph-backed repository context.

- `code-graph-rag` is the graph/index/query toolchain
- `Memgraph` is the graph database backend

This graph layer is useful for:

- repository context
- graph-aware code lookup
- named-function workflows

It should not become a dependency for every trivial file operation.

### Tracing

Local traces are still written to disk, but the preferred trace path is LangSmith via
LangGraph runnable config. When LangSmith env vars are configured, runs should appear
as real remote traces instead of only local JSON artifacts.

## File Editing Strategy

Shipyard should prefer the narrowest edit that satisfies the instruction.

### Preferred Order

1. named function edit
2. file-local symbol rename
3. localized text replacement
4. append/prepend
5. whole-file rewrite

The runtime should avoid rewriting an entire file when a narrower operation is
available and clearly supported by the request.

### Editing Philosophy

- the model interprets intent
- the executor applies the smallest safe edit
- verification catches regressions
- snapshots enable rollback

### Current and Intended Behavior

- If the user asks to create files, Shipyard creates files in the managed workspace.
- If the user asks to write code into a named file, the planner should produce a
  concrete code-writing action instead of writing the raw instruction text.
- If the user asks to rename an identifier in a file, the planner should prefer a
  symbol-aware rename path rather than a brittle anchor match.
- If the user asks to add code to a file, the planner should preserve existing content
  unless the prompt clearly requests a full rewrite.

### Safety Boundaries

- all file writes stay within allowed workspace roots unless explicitly overridden
- destructive actions should be gated
- shell commands should be explicit tool actions, not the model’s default write path

## Multi-Agent Strategy

The current system includes queue state and helper-agent scaffolding, but the next
clean iteration should treat multi-agent behavior as scoped delegation, not parallel
chaos.

Recommended pattern:

- lead agent owns the run
- helper agents receive narrow subtasks
- helpers do not directly write files
- lead agent merges and executes

Examples of good helper subtasks:

- locate target file
- summarize blast radius
- propose verification steps
- propose alternate implementation

## Direction for the Next Phase

The next major refactor should push Shipyard toward:

- model-first command interpretation
- tool-schema-driven execution
- less parser logic
- clearer session/run boundaries
- stronger LangSmith visibility

The right mental model is:

- the LLM decides what to do
- Shipyard decides what is allowed and how it is executed
