# PRESEARCH.md — Project Shipyard

## Goal

Build Shipyard into a local coding agent that can:

- accept natural language instructions
- inspect and edit code safely
- run commands in the working directory when appropriate
- persist session state across runs
- use graph-backed repository context when useful
- rely on the LLM for planning rather than parser-heavy command logic

The main correction to the original MVP direction is this:

- Shipyard should be **LLM-first**
- parser logic should be **non-primary fallback only when the model is unavailable**

---

## External Research Summary

### OpenAI Tool Calling

Official OpenAI guidance strongly supports a pattern where:

1. the model is given a set of tools
2. the model decides which tool to call
3. the application executes the tool
4. the tool output is returned to the model
5. the model continues until the task is complete

This is a much better fit for Shipyard than trying to hardcode every possible command
shape in regex.

Why it matters for Shipyard:

- the model can interpret varied user phrasing
- the runtime still controls safety and actual execution
- the system remains inspectable and traceable

Sources:
- https://platform.openai.com/docs/guides/function-calling
- https://help.openai.com/en/articles/8555517-function-calling-updates

### Anthropic Agent Guidance

Anthropic’s guidance for effective agents matches the same broad pattern:

- simple composable tools
- clear tool descriptions
- constrained execution
- avoid overcomplicated routing and hidden agent behavior

This reinforces the idea that Shipyard should not keep growing parser complexity to
cover every phrase. The model should understand the request; the runtime should execute
it safely.

Sources:
- https://www.anthropic.com/news/building-effective-agents
- https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use

### LangGraph

LangGraph is still a strong fit for Shipyard because it provides:

- persistent state
- resumable execution
- queue-friendly orchestration
- human-in-the-loop interrupts
- LangSmith integration

Shipyard should keep LangGraph as the runtime/orchestration layer while letting the
LLM own more of the planning responsibility.

Sources:
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/human-in-the-loop

### Code-Graph-RAG + Memgraph

Code-Graph-RAG and Memgraph are useful for repository context, not as the universal
control path for every edit.

Best role inside Shipyard:

- repository lookup
- graph-aware code context
- named-function workflows
- blast-radius reasoning

Not the best role:

- handling every trivial file write
- blocking simple local file operations

Sources:
- https://docs.code-graph-rag.com/

---

## Architecture Decision

### Chosen Direction

Shipyard should use:

- **LLM-first planning**
- **tool-driven execution**
- **LangGraph runtime orchestration**
- **graph-backed context as a subsystem**

This means:

- the model plans what to do
- Shipyard decides what tools exist
- Shipyard validates and executes the resulting actions

### Rejected Direction

Shipyard should not keep evolving into:

- a regex-driven command interpreter
- a parser that needs a new rule for every verb
- a shell-command agent with weak safety boundaries

That path is too brittle and too expensive to maintain.

---

## Proposed Runtime Model

### Layer 1: Interface

The workbench remains a chat-style UI with:

- prompt entry
- activity stream
- queue status
- session history
- graph status
- raw details when needed

The UI should reflect the runtime truth, not invent workflow logic on its own.

### Layer 2: Planner

The planner should be model-first.

Inputs:

- user instruction
- session state
- current workspace target context
- optional graph context
- tool definitions

Outputs:

- structured action list
- target files
- content or rename instructions
- command actions when explicitly appropriate

### Layer 3: Executor

The executor should run a stable set of tools.

Recommended tool set:

- `list_files`
- `read_file`
- `search_files`
- `write_file`
- `append_to_file`
- `prepend_to_file`
- `replace_in_file`
- `rename_symbol`
- `create_files`
- `copy_file`
- `delete_file`
- `run_command`
- `git_status`
- `git_diff`
- `graph_query`

The model chooses tools.
The runtime executes tools.

### Layer 4: Verification

After edits:

- run verification commands when configured
- capture errors
- retry with error context if appropriate
- revert from snapshot on failure

### Layer 5: Persistence and Tracing

Shipyard should keep:

- local session history
- run queue state
- snapshots
- prompt log
- local JSON traces
- LangSmith traces when configured

---

## Editing Strategy

### Guiding Rule

Apply the narrowest edit that satisfies the request.

Preferred order:

1. named function replacement
2. file-local symbol rename
3. localized text replacement
4. append/prepend
5. full rewrite

### Important Change from Earlier MVP Thinking

Earlier versions of Shipyard leaned too heavily on explicit parser logic to decide
editing mode. The better approach is:

- let the model decide the edit strategy
- normalize and validate the output
- execute through deterministic tools

Parser logic should only rescue obvious trivial cases when the model is unavailable or
heuristic mode is explicitly selected.

### Whole-Codebase Rewrite Support

Shipyard should eventually support repository-scale work through a two-pass approach:

1. planning pass
   - identify affected files
   - summarize intended changes
   - ask for approval if the blast radius is large
2. execution pass
   - rewrite files in batches
   - run verification after each batch

---

## Queue and Session Design

Shipyard needs one clear workspace model:

- one session id
- one session workspace
- one run queue

The system should not drift between:

- `default`
- `web-*`
- `run-*`

unless the user explicitly requests a different root.

The UI should always show:

- current session
- current run
- current task
- changed files

---

## Multi-Agent Design

Shipyard does not need uncontrolled parallel agents.

Best next-step model:

- one lead agent owns the run
- helper agents get narrow subtasks
- helper agents do not directly write files
- lead agent merges outputs and executes the final action plan

Good helper tasks:

- locate file or function
- summarize graph context
- propose verification
- summarize a failure

---

## Graph Design

Shipyard should keep the graph subsystem, but treat it as optional context unless a
workflow truly requires it.

Recommended graph role:

- richer repo understanding
- function-aware retrieval
- code relationship lookup

Not recommended:

- routing all trivial edits through graph-dependent logic

Shipyard should clearly surface graph state:

- unavailable
- indexed
- populated
- stale

---

## Tracing Design

LangSmith should be the preferred external trace surface.

LangGraph runs should send:

- `thread_id`
- run metadata
- step metadata
- tags

Local JSON traces should remain as a fallback and debugging artifact.

---

## Concrete Refactor Plan

### Phase 1

- make LLM planning the default path everywhere
- keep heuristics as fallback only
- reduce direct parser-led execution paths

### Phase 2

- formalize the action schema
- make every prompt resolve to `actions[]`
- keep the executor deterministic

### Phase 3

- improve command execution support through an explicit `run_command` tool
- gate risky commands behind approval or strict allowlists

### Phase 4

- strengthen multi-agent delegation using scoped helper tasks
- improve graph-aware planning for function and repo-level work

### Phase 5

- add safer repo-wide rewrite workflows
- require approval for large blast-radius rewrites

---

## Final Recommendation

Shipyard should be built as:

- a model-first planning system
- a tool-constrained execution engine
- a LangGraph runtime with persistence
- a graph-aware coding assistant, not a parser-heavy command router

That is the cleanest path to a system that can:

- run commands in a directory
- analyze files
- rewrite files or whole code paths
- remain debuggable and safe

*PRESEARCH.md — updated for the next Shipyard phase*
