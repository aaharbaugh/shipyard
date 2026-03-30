# AI Development Log

## Tools & Workflow

**Primary AI tools used:**
- **Claude Code (CLI)** — primary development partner for all Shipyard code. Used for architecture design, implementation, debugging, and documentation. Every file in the agent was written or reviewed through Claude Code conversations.
- **OpenAI GPT-5.4-mini** — the LLM powering the agent itself. Handles action planning, edit proposal generation, and task decomposition at runtime.
- **OpenAI GPT-4.1-nano** — cheaper model used at runtime for low-stakes calls: plan repair, adaptive replan, file exploration.
- **LangSmith** — tracing and observability for LangGraph execution (configured via environment variables).

**Workflow:**
1. Describe the desired behavior or bug to Claude Code
2. Claude reads the codebase, proposes changes, implements them
3. Run `python -m pytest tests/ -x -q` to verify
4. Test end-to-end with `echo '{"instruction": "...", "session_id": "..."}' | python -m shipyard.main`
5. Iterate on failures — Claude reads traces, diagnoses, fixes

## Effective Prompts

### Prompt 1: Architecture simplification
```
The editing system has too many modes and the Code Graph RAG dependency adds
complexity without proportional value. Simplify to an inspect-first pattern:
read the file, then apply search_and_replace or write_file. Replace the graph
layer with grep and cat. Add trace links and development log documentation.
```
Drove the biggest architectural change — removing the Code Graph RAG dependency, collapsing edit modes, and establishing the inspect-first pattern that became the core editing strategy.

### Prompt 2: Content preservation
```
The agent is completing targeted edits but dropping unrelated content in the
same file. Edits to one function shouldn't affect the rest of the file.
```
Led to the discovery that write_file fundamentally can't preserve content on large files. Resulted in the search_and_replace-first strategy and the 30% content loss guard.

### Prompt 3: Autonomous execution
```
The agent needs to handle open-ended tasks autonomously. Guard against
malicious code execution, but otherwise remove approval gates and let it
recover from errors on its own.
```
Triggered removal of 17 human_gate blocks, relaxation of shell command restrictions, and the shift from "ask permission" to "fail and move on" error handling.

### Prompt 4: Multi-agent coordination
```
Implement multi-agent execution: a supervisor that decomposes instructions
into scoped sub-tasks, workers that execute in parallel, and a conflict
detector that flags when two workers edit the same file.
```
Led to the supervisor + worker orchestrator architecture with dependency waves and post-execution conflict detection.

### Prompt 5: Real-world integration test
```
Simulate running the agent against the full Ship codebase — a 146K-line
TypeScript monorepo. Identify what breaks when targeting a real repo instead
of a managed sandbox.
```
Uncovered the workspace binding bug, path sandboxing gaps, filename false positives, and the workspace clobbering issue. Validated that the agent could operate on external repositories.

## Code Analysis

**AI-generated vs hand-written code:**
- ~99.9% AI-generated (Claude Code wrote all implementation code)
- ~0.1% hand-directed (specific architectural decisions, prompt wording, test scenarios)

The human role was primarily: defining what to build, identifying failure modes, testing end-to-end, and pushing back when the agent over-engineered solutions.

## Strengths & Limitations

**Where AI excelled:**
- Rapid prototyping: entire supervisor + worker orchestrator (~536 lines) built in one session
- Bug tracing: Claude could read trace files, identify root causes (e.g., `list("s1")` → `["s", "1"]`), and fix them immediately
- Test generation: 231 tests covering edge cases the human wouldn't have thought of
- Cross-file consistency: updating prompts, validation, graph nodes, and tests in one pass

**Where AI fell short:**
- Over-engineering: initial designs had too many modes, too many gates, too many abstractions. Required explicit "simplify" pushback.
- Content preservation: the LLM consistently drops content when rewriting files. No amount of prompt engineering fixed this — needed a structural guard (content loss check).
- Path resolution: the sandbox logic had subtle bugs that only surfaced when testing against a real external repo. AI didn't anticipate these edge cases.
- False confidence: "verify: 1 passed" was reported even when the file hadn't actually changed, because the verification ran on unchanged content.

## Key Learnings

1. **Structural guards beat prompt engineering.** Telling the LLM "preserve all content" doesn't work. Blocking write_file when it would lose >30% content does.

2. **The LLM is smarter than the validator.** We removed multiple validation rules (step count heuristic, strict enum checking, human gates) because they blocked reasonable LLM output. The agent got more reliable by removing code, not adding it.

3. **Test against real repos early.** The todo app sandbox hid path resolution bugs, workspace binding issues, and timeout problems that only surfaced on the 116K-line ship monorepo.

4. **Model tiering saves money without losing quality.** Using nano for repair/replan/exploration cut costs ~70% on those calls with no quality impact.

5. **Multi-agent is mostly an orchestration problem.** The hard part isn't running workers in parallel — it's deciding what to parallelize and merging results without conflicts.
