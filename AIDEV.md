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
The professor says the agent is over-engineered. Fix three things:
1. Simplify the editing approach — use inspect-first (grep/cat/run_command),
   then write_file or search_and_replace. Remove anchor-with-pointers complexity.
2. Replace Code Graph RAG with simple grep+cat.
3. Add required docs (trace links, AI Development Log).
```
This prompt drove the biggest architectural change — removing Code Graph RAG dependency, simplifying edit modes, and establishing the inspect-first pattern.

### Prompt 2: Content preservation debugging
```
I'm giving it specific tasks and it's removing several other features in the
pass through.
```
Short but effective — led to the discovery that write_file fundamentally can't preserve content on large files, which led to the search_and_replace-first strategy and the 30% content loss guard.

### Prompt 3: Reliability audit
```
I need you to think that you might get very open ended tasks. We just need to
guard from malicious code but other than that you need to go ham on this bitch.
```
Triggered removal of 17 human_gate blocks, relaxation of shell command restrictions, and the shift from "ask permission" to "fail and move on" philosophy.

### Prompt 4: Multi-agent design
```
I want a solid version of multi-agent. Supervisor that decomposes, workers that
execute in parallel, conflict detection.
```
Led to the supervisor + worker orchestrator architecture with dependency waves and conflict detection.

### Prompt 5: Ship repo integration
```
Run through the scenario where I actually run this on the SHIP app.
```
Uncovered the workspace binding bug, path sandboxing gaps, filename false positives (console.log), and the workspace clobbering issue.

## Code Analysis

**AI-generated vs hand-written code:**
- ~95% AI-generated (Claude Code wrote all implementation code)
- ~5% hand-directed (specific architectural decisions, prompt wording, test scenarios)

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
