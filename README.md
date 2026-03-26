# Shipyard

Shipyard is a local coding agent with:

- an LLM-first planner
- a LangGraph execution loop
- a managed workspace for file operations
- queue/session state for ongoing runs
- optional Code-Graph-RAG + Memgraph repository context
- optional LangSmith tracing

The intended control flow is:

1. user prompt
2. model plans structured actions
3. Shipyard executes those actions through constrained tools
4. results are verified, persisted, and traced

## Current Direction

Shipyard is moving away from parser-led behavior and toward:

- model-first action planning
- deterministic tool execution
- minimal heuristic fallback

Regex and local parsing still exist, but they are fallback behavior. The long-term
architecture is not "add more verbs." The long-term architecture is:

- `LLM planner -> structured actions -> executor -> verification`

## Architecture

### 1. Workbench

The FastAPI workbench provides:

- a chat-style activity stream
- side-panel status for queue, sessions, graph, and raw output
- a persistent local session id

Main file:
- [shipyard/api.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/api.py)

### 2. Planning

The model is the primary planner when OpenAI is configured. It is expected to turn
natural language into a structured action plan.

Main files:
- [shipyard/action_planner.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/action_planner.py)
- [shipyard/proposal.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/proposal.py)
- [shipyard/prompts.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/prompts.py)

### 3. Execution

Shipyard executes planned actions through a constrained runtime instead of letting the
model directly mutate files. Supported operations currently include:

- file creation
- whole-file writes
- append/prepend
- copy/delete
- file-local rename operations
- anchor replacements
- named-function edits

Main file:
- [shipyard/graph.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/graph.py)

### 4. Runtime

The runtime keeps:

- session state
- queue state
- traces
- snapshots
- prompt logs
- workspace files

Main files:
- [shipyard/main.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/main.py)
- [shipyard/run_queue.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/run_queue.py)
- [shipyard/session_store.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/session_store.py)
- [shipyard/runtime_state.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/runtime_state.py)

### 5. Graph Context

Shipyard can use Code-Graph-RAG and Memgraph for repository context.

- `code-graph-rag` provides the graph/index/query toolchain
- `Memgraph` stores the live graph

This graph layer is useful for graph-aware code operations, but it should remain a
context subsystem, not the primary control path for every edit.

Main file:
- [shipyard/tools/code_graph.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/tools/code_graph.py)

## Setup

Create and activate the venv, then install requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
```

Suggested `.env` values:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=shipyard-mvp
```

## Run

Start the workbench:

```bash
source .venv/bin/activate
./scripts/run_workbench.sh
```

Open:

```text
http://127.0.0.1:8000/workbench
```

If you want graph-backed context too:

```bash
./scripts/start_memgraph.sh
./scripts/index_graph.sh
```

## LangSmith

LangGraph runs now pass a real runnable config with:

- `thread_id`
- run metadata
- tags

When LangSmith env vars are set, runs should appear in LangSmith for inspection.

Main file:
- [shipyard/langsmith_config.py](/home/aaron/projects/gauntlet/ship/shipyard/shipyard/langsmith_config.py)

## Demo Flow

Example demo:

1. `create 30 random python files`
2. `write a random python algorithm in file scratch_copy_3.py`
3. `in file scratch_copy_3.py replace total with totality`

## Near-Term Refactor Goal

The next phase should reduce parser logic further and make the LLM the clear owner of
command interpretation.

Target design:

- prompt goes to model
- model emits structured actions
- executor validates and runs actions
- human approval gates risky operations
- heuristics only rescue obvious fallback cases

## Tests

```bash
./.venv/bin/python -m unittest discover -s tests
```
