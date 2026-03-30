# Shipyard

Shipyard is an autonomous coding agent that takes natural-language instructions, plans structured edits, and executes them against a codebase. Built on LangGraph with OpenAI as the planning model.

## How It Works

1. User submits an instruction (via workbench UI or CLI)
2. Agent builds broad context (file tree, project stack, doc discovery)
3. LLM generates an ordered action plan
4. Each step executes through constrained tools: read, search_and_replace, write, append, delete
5. Results are verified, snapshots enable rollback on failure
6. Traces and logs are persisted for every run

## Key Features

- **Surgical editing** — search_and_replace as primary edit mode; only changes what it matches
- **Inspect-first pattern** — reads target files before editing, so anchors are never stale
- **Multi-agent execution** — supervisor decomposes large tasks, workers run in parallel with conflict detection
- **Self-healing loop** — detects syntax errors after edits, feeds them back, retries automatically
- **Auto-branching** — creates a feature branch before mutations
- **Model tiering** — gpt-5.4-mini for planning/code gen, gpt-4.1-nano for repair/replan
- **Cost tracking** — every OpenAI call logged with token counts and cost

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM planning |
| `OPENAI_MODEL` | No | Model to use (default: `gpt-5.4-mini`) |
| `LANGSMITH_API_KEY` | No | LangSmith key for tracing |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |

## Run

Start the workbench:

```bash
./scripts/run_workbench.sh
```

Open http://127.0.0.1:8000/workbench

You can also enter your OpenAI API key directly in the workbench UI.

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Run loop | `shipyard/main.py` | Orchestrates runs: plan, execute, verify, self-heal |
| Action planner | `shipyard/action_planner.py` | LLM-based multi-step plan generation |
| Graph executor | `shipyard/graph.py` | LangGraph state machine for per-step edit cycle |
| Supervisor | `shipyard/supervisor.py` | Task decomposition for multi-agent execution |
| Worker orchestrator | `shipyard/worker_orchestrator.py` | Parallel worker spawning and conflict detection |
| Context explorer | `shipyard/context_explorer.py` | File tree scanning, project stack detection |
| API + UI | `shipyard/api.py` | FastAPI server with embedded workbench |
| Queue | `shipyard/run_queue.py` | Job queue with task hierarchy tracking |
| Workspaces | `shipyard/workspaces.py` | Session workspace management |

## Tests

```bash
.venv/bin/pytest tests/ -q --ignore=tests/test_agent.py
```

## Data

All run data is stored in `.shipyard/data/`:

- `rebuild_log.jsonl` — structured log of every agent run
- `openai_costs.jsonl` — token and cost tracking per API call
- `traces/` — LangGraph trace files (JSON)
- `sessions/` — session state snapshots

## Docs

- [CODEAGENT.md](CODEAGENT.md) — architecture, editing strategy, multi-agent design, comparative analysis, cost analysis
- [AIDEV.md](AIDEV.md) — AI development log, effective prompts, strengths and limitations
