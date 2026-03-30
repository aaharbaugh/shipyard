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

- Full trace dataset: https://smith.langchain.com/public/9d546ecc-3825-489d-bcda-b8ca14e04e5c/d
- Trace 1 (normal run — scaffold + edit): Select any `shipyard.run_step` trace with status `edited`
- Trace 2 (error recovery — verification failure + retry): Select any trace showing `verification_failed` → `retry_ready` → `edited`

804+ local traces also written to `.shipyard/data/traces/` as JSON files.

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

307 runs captured in `.shipyard/data/rebuild_log.jsonl`. 804 traces in `.shipyard/data/traces/`.

Key interventions during rebuild:
1. **package.json dependencies** — agent created all source files but failed to populate package.json with dependencies. Manual `pnpm add` required for api and web packages.
2. **Import path mismatches** — agent wrote `../db/client` instead of `../../db/client` in module files. Fixed by sed across all modules.
3. **Content duplication** — write_file mode caused the LLM to echo existing content plus new content, doubling files. Led to architectural change: force search_and_replace on all existing files.
4. **Session not persisting** — agent removed `setSessionUser` call during a "fix" run, breaking login. Session was never saved because `req.session.userId` was never set.
5. **Content-Type header deleted** — api.ts `request` function had `headers.delete('Content-Type')` which prevented JSON body parsing. Agent re-introduced this bug multiple times during fix attempts.
6. **Verify step hangs** — subprocess execution in LangGraph thread didn't properly kill child processes on timeout. Fixed with `os.setsid` + `os.killpg` for process group management.
7. **Target path contamination** — parallel batch execution caused all steps to write to the same file. Fixed by removing `_locked_target_path` mechanism and sandboxing per-step.
8. **Schema migration** — `pool.end()` in migrate.ts closed the database connection before the API could use it. Removed the call.
9. **CORS origin** — cookies not sent because CORS `origin: true` doesn't work for credentialed requests cross-port. Changed to explicit `origin: 'http://localhost:3000'`.

## Comparative Analysis

### Executive Summary

Shipyard is an LLM-powered coding agent built on LangGraph that was used to rebuild the Ship workspace app from scratch. The agent analyzed the original 146K-line Ship codebase, critiqued its architecture, proposed a simpler alternative (Spaces + Pages + Tasks instead of "everything is a document"), and generated a 3,400-line rebuild with 33 TypeScript source files. The rebuild covers auth, workspace management, pages, tasks, search, and a React frontend — approximately 2.4% of the original's line count while preserving the core product functionality. The rebuild required 307 agent runs and 9 documented manual interventions.

### Architectural Comparison

| Aspect | Original Ship | Agent-Built Ship |
|--------|--------------|-----------------|
| Source files | 603 | 33 |
| Lines of code | 145,925 | 3,446 |
| API dependencies | 30 | 9 |
| Web dependencies | 44 | 4 |
| Domain entities | 15+ (documents, issues, projects, programs, iterations, weeks, standups, etc.) | 4 (Spaces, Pages, Tasks, Invitations) |
| React providers | 10+ global contexts | 1 (QueryClientProvider) |
| Real-time collaboration | Yjs + WebSocket + TipTap | Not implemented |
| AI features | Bedrock/OpenAI integration, FleetGraph | Not implemented |
| Database | PostgreSQL with 40+ tables | PostgreSQL with 8 tables |
| Auth | OpenID Connect + multiple providers | bcrypt + express-session |

**What the agent chose differently than a human would:**
- Collapsed all document types into a single `Pages` table instead of maintaining the polymorphic document model
- Removed all collaboration infrastructure (Yjs, WebSocket) — a human would likely keep at least basic collaboration
- Used inline styles in React components instead of a design system — a human would use Tailwind classes consistently
- No test files — a human would write tests alongside implementation
- Flat module structure with barrel exports — a human might use feature folders with co-located tests

### Performance Benchmarks

| Metric | Original | Rebuilt |
|--------|----------|--------|
| `vite build` time | ~3.2s (estimated for 600+ modules) | 0.8s (84 modules) |
| API startup | ~2s (migrations + service init) | <1s |
| Bundle size (gzip) | ~180KB+ | 67KB |
| DB tables | 40+ | 8 |
| API routes | 50+ endpoints | 22 endpoints |
| npm install time | ~45s | ~8s |

The rebuilt app is significantly faster to build, start, and deploy due to its reduced scope. Whether this is a meaningful comparison depends on whether you consider the dropped features (collaboration, AI, iteration planning) as scope reduction or missing functionality.

### Shortcomings

1. **Content duplication bug** — The agent's write_file mode consistently doubled file content. Every edit risk making files worse. Required architectural change (force search_and_replace) to fix, which then made the agent unable to fill in stub files efficiently.

2. **No cross-file coordination** — The agent edits files independently. When it writes auth routes that export `authRouter`, it doesn't verify the importing file uses the same name. This caused repeated import/export mismatches.

3. **Session management broken by fixes** — The agent removed the `setSessionUser` call while fixing an unrelated session issue, breaking login entirely. The agent has no concept of "this line is critical, don't touch it."

4. **Package.json never populated** — Despite explicit instructions listing every dependency, the agent created package.json files without dependencies. This happened on every attempt.

5. **Verification commands wrong** — The agent consistently planned `node -e "import('./file.ts')"` which can't run TypeScript. It should use `npx tsx` but the LLM doesn't know the project's toolchain.

6. **No tests generated** — Despite the build spec requesting tests, the agent never created test files.

7. **Stub detection too late** — The scaffold phase creates 45 stub files, but the self-heal loop takes multiple iterations to fill each one, making the process extremely slow.

8. **Target path contamination** — Parallel execution caused files to be written to wrong paths. Required multiple architectural fixes (locked paths, then removing locked paths, then sandboxing).

9. **Infinite loop on verification** — Subprocess management didn't properly kill child processes on timeout, causing the entire server to hang or self-kill via `os.killpg(0)`.

### Advances

1. **Architecture analysis** — The plan mode generated a genuinely insightful critique of the original Ship codebase. The "too many nouns" observation and the proposed simplification to Spaces/Pages/Tasks is a defensible architectural choice that a senior engineer might make.

2. **Scaffold speed** — Creating 45 files with correct monorepo structure, TypeScript configs, database schema, and Express route shells took ~20 seconds. A human would take hours to set up the same boilerplate.

3. **Parallel execution** — Independent file reads and writes run concurrently, completing 3-file batches in the time of 1.

4. **Auto-recovery on verification** — When the self-heal loop works, it catches syntax errors and feeds them back automatically. The loop fixed the `headers.delete('Content-Type')` bug without human intervention (before re-introducing it in a later run).

5. **Doc discovery** — The agent found and read REBUILD_PLAN.md, BUILD_SPEC.md, and reference docs automatically without being told where they were.

### Trade-off Analysis

| Decision | Right Call? | What I'd Change |
|----------|-------------|-----------------|
| search_and_replace as primary edit mode | **Yes** — prevented content duplication | Would add AST-aware editing for import/export coordination |
| Forcing search_and_replace on ALL existing files | **Partially** — prevented duplication but made stub filling slow | Would allow write_file on files < 15 lines (stubs) but block on larger files |
| Removing all validation gates | **Mixed** — unblocked execution but removed safety nets | Would keep type-level validation (missing edit_mode) but remove content validation |
| Parallel batch execution | **Yes** — 3x faster for multi-file operations | Would add better state isolation between workers |
| gpt-5.4-mini for code generation | **Adequate** — generates reasonable code | Would test Claude for better instruction following on surgical edits |
| Process group kill for timeouts | **Yes** — finally fixed the hanging issue | Should have been the approach from day 1, not the Popen polling loop |
| Self-heal loop | **Good concept, poor execution** — detects errors but often can't fix them | Would separate "detect" from "fix" — let the LLM see ALL errors before planning fixes |

### If You Built It Again

1. **Start with search_and_replace only** — never implement write_file for existing files. The duplication bug cost more debugging time than any other issue.

2. **Test the edit strategy on 200+ line files early** — the PRD explicitly warns about this. We discovered the duplication bug late.

3. **Use file hashing** — before and after each edit, hash the file. If the hash didn't change, the edit was a no-op. If the file grew by >50%, the edit duplicated content. Reject it automatically.

4. **Separate scaffold from implementation** — scaffold creates the file tree with stubs. A second pass fills in each file one at a time with full context. Don't try to do both in one scaffold_files call.

5. **Pin the session — don't let the agent edit critical infrastructure** — auth middleware, session setup, CORS config, and database connection should be marked as "do not modify" after initial creation.

6. **Verify after EVERY edit, not after the full run** — the current self-heal loop runs build checks after the entire action plan completes. It should verify after each individual file edit.

7. **Use a real test runner** — instead of trying to run `node --check` or `tsc`, run `vite build` which actually works and catches real errors.

## Cost Analysis

| Item | Amount |
|------|--------|
| LangSmith tracked traces | 4,063 |
| LangSmith tracked tokens | 1,938 |
| LangSmith tracked cost | $0.01 |
| OpenAI API calls (direct httpx) | ~900 (307 runs × ~3 calls avg: plan + refine + repair) |
| OpenAI estimated input tokens | ~7.2M (900 calls × ~8K avg input) |
| OpenAI estimated output tokens | ~2.7M (900 calls × ~3K avg output) |
| Total development spend | ~$8-15 estimated (OpenAI usage not instrumented through LangSmith) |

Note: LangSmith tracks the LangGraph step orchestration but not the direct OpenAI API calls made via httpx in `action_planner.py` and `proposal.py`. The actual token usage is on the OpenAI dashboard under the API key owner's account.

### Production Cost Projections

Assumptions:
- Average agent invocations per user per day: 10
- Average tokens per invocation: 12,000 input / 4,000 output
- gpt-5.4-mini: ~$0.15/1M input, ~$0.60/1M output
- gpt-4.1-nano: ~$0.01/1M input, ~$0.04/1M output (used for repair/replan)
- Cost per invocation: ~$0.004

| Scale | Monthly Cost |
|-------|-------------|
| 100 Users | ~$120/month |
| 1,000 Users | ~$1,200/month |
| 10,000 Users | ~$12,000/month |
