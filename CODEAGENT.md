# CODEAGENT.md

## Agent Architecture (MVP)

Shipyard MVP will use a LangGraph-based local coding agent built around a persistent
lead-agent loop. The system runs as a long-lived Python process so it can accept
multiple instructions without restarting. The first version should prioritize the
hard-gate MVP requirements from the PRD over ambitious platform features.

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

For named-function edits, the agent first queries Code-Graph-RAG for blast radius and
retrieves the current function body before proposing a replacement. If the target
function cannot be resolved uniquely, the graph does not write to disk. It re-reads the
function, retries with corrected context, and stops after a small retry limit. If
verification fails after a write, the failure output is injected back into the next
attempt so the agent can repair the edit or pause for human intervention. The current
MVP also reverts the file from the latest snapshot before returning a failed status.

## File Editing Strategy (MVP)

The preferred MVP editing path is Code-Graph-RAG AST-targeted function replacement.
For named functions, the agent should query the code graph first, retrieve the current
function source, and replace only that function body. Anchor-based replacement is now
strictly a fallback for non-function targets such as config blocks, class variables, or
inline logic with no function wrapper.

### Step-by-Step Mechanism

1. Agent calls `query_codebase()` or `semantic_search()` to understand callers,
   imports, and blast radius before editing
2. Agent calls `get_function_code(file, function)` to read the current implementation
3. Agent proposes a full replacement for that function
4. System snapshots the file before editing
5. System calls `replace_code(file, function, new_code)` to surgically replace the
   named function via AST-aware targeting
6. Verification runs immediately after the write
7. If verification fails, the failure output is fed back into the next attempt after
   re-reading the function with `get_function_code`
8. If the target is not a named function, fall back to the non-function edit path

### How the Correct Block Is Located

For named functions, location is determined by the function identity in the parsed code
graph rather than string anchors or brittle line numbers. Blast-radius context is
retrieved before the replacement is proposed.

### What Happens When Location Is Wrong

- Function not found or ambiguous: block the write, re-query, re-read the function, and
  retry with more specific context
- Verification failure after write: use snapshot for rollback if needed, re-read the
  function, and retry with error context
- Non-function target: use the fallback edit path only when AST function replacement
  does not apply

## Multi-Agent Design (MVP)

The MVP design supports one lead agent and one helper agent in sequence. This is the
simplest version that still satisfies the requirement to support more than one agent
without introducing parallel merge complexity too early.

### Orchestration Model

- Lead agent owns the main loop and all file writes
- Helper agent receives a narrow subtask such as:
  - identify the function to change and gather blast-radius context
  - prepare a candidate `replace_code()` body from the current function source
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

To be completed during the build.

## Ship Rebuild Log (Final Submission)

To be completed during the rebuild.

## Comparative Analysis (Final Submission)

To be completed after the rebuild.

## Cost Analysis (Final Submission)

To be completed with actual development usage and projected production assumptions.
