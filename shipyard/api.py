from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .graph import build_graph
from .intent_parser import parse_instruction
from .main import _normalize_payload, run_once
from .runtime_cleanup import cleanup_runtime_data
from .workspaces import get_session_workspace, get_workspace_status
from .session_store import SessionStore
from .proposal import get_planner_status
from .run_queue import RunQueue
from .tools.code_graph import index_code_graph, inspect_code_graph_status, sync_live_code_graph
from .tools.git_tools import GitAutomation, GitAutomationError


class InstructionRequest(BaseModel):
    session_id: str | None = None
    instruction: str = ""
    target_path: str | None = None
    anchor: str | None = None
    replacement: str | None = None
    proposal_mode: str | None = None
    proposal_model: str | None = None
    edit_mode: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    verification_commands: list[str] = Field(default_factory=list)
    edit_attempts: int = 0
    max_edit_attempts: int = 2


class GitBranchRequest(BaseModel):
    branch_name: str


class GitCommitRequest(BaseModel):
    message: str
    paths: list[str] = Field(default_factory=list)


class GraphIndexRequest(BaseModel):
    workdir: str | None = None
    output_dir: str | None = None
    clean: bool = False


class WorkspaceCreateRequest(BaseModel):
    prefix: str = "run"
    session_id: str | None = None


class CleanupRequest(BaseModel):
    keep_traces: int = 20
    keep_snapshots: int = 20
    keep_sessions: int = 20
    keep_logs: int = 20
    remove_empty_workspaces: bool = True
    remove_empty_spec_dirs: bool = True


app = FastAPI(title="Shipyard MVP API")
graph_app = build_graph()
session_store = SessionStore()
git_automation = GitAutomation()
run_queue = RunQueue(
    lambda state, progress_callback: run_once(
        graph_app,
        session_store,
        state,
        progress_callback=progress_callback,
    )
)


def _should_run_direct(request: InstructionRequest) -> bool:
    if os.getenv("OPENAI_API_KEY") and (request.proposal_mode or "").strip().lower() != "heuristic":
        return False
    parsed = parse_instruction(request.instruction or "")
    if not parsed:
        return False
    mode = parsed[0]
    testing_mode = bool((request.context or {}).get("testing_mode"))
    return testing_mode and mode in {"write_file", "append", "prepend", "delete_file", "copy_file", "create_files", "rename_symbol"}


WORKBENCH_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Shipyard Workbench</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: rgba(255, 255, 255, 0.92);
      --panel-strong: #ffffff;
      --ink: #152033;
      --muted: #66758a;
      --accent: #1565d8;
      --line: #dbe3ef;
      --soft: #f6f8fc;
      --good: #157347;
      --bad: #b02a37;
    }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(21, 101, 216, 0.12), transparent 28%),
        linear-gradient(180deg, #f8fbff, var(--bg));
      color: var(--ink);
    }
    .app-shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px 20px 40px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 18px;
      align-items: start;
      min-height: 100vh;
      box-sizing: border-box;
    }
    .main-column {
      display: grid;
      gap: 18px;
      min-height: 0;
    }
    .hero, .panel, .side-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 48px rgba(21, 32, 51, 0.08);
      backdrop-filter: blur(10px);
    }
    .hero, .panel, .side-panel {
      padding: 20px;
    }
    .terminal-panel {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: calc(100vh - 64px);
      max-height: calc(100vh - 64px);
      overflow: hidden;
      background: linear-gradient(180deg, rgba(13, 20, 32, 0.96), rgba(19, 29, 46, 0.96));
      color: #e8eef8;
      border-color: rgba(148, 163, 184, 0.18);
    }
    .terminal-panel .panel-title,
    .terminal-panel .muted,
    .terminal-panel .message-label,
    .terminal-panel .meta-line {
      color: rgba(226, 232, 240, 0.72);
    }
    .terminal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }
    .terminal-dots {
      display: flex;
      gap: 8px;
    }
    .terminal-dots span {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(248, 250, 252, 0.26);
    }
    .terminal-header h1 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
    }
    .hero-top {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 2.1rem;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 60ch;
      line-height: 1.45;
    }
    .panel-title {
      margin: 0 0 12px;
      font-size: 1rem;
    }
    label {
      display: block;
      margin: 10px 0 6px;
      font-size: 0.92rem;
      color: var(--muted);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    input, select, textarea, button {
      width: 100%;
      box-sizing: border-box;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      background: var(--panel-strong);
      color: var(--ink);
    }
    textarea {
      min-height: 160px;
      resize: vertical;
    }
    .compact-textarea {
      min-height: 96px;
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .actions button, .hero-top button, .tab-row button {
      cursor: pointer;
      width: auto;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border: none;
    }
    button.secondary {
      background: var(--soft);
      color: var(--ink);
    }
    .muted {
      color: var(--muted);
      font-size: 0.9rem;
    }
    .status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-strong);
    }
    .status strong {
      display: block;
      margin-bottom: 4px;
      font-size: 1rem;
    }
    .status span {
      color: var(--muted);
      font-size: 0.93rem;
    }
    .pill {
      white-space: nowrap;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.88rem;
      border: 1px solid var(--line);
      background: var(--soft);
    }
    .pill.good {
      color: var(--good);
      border-color: rgba(31, 122, 82, 0.25);
      background: rgba(31, 122, 82, 0.08);
    }
    .pill.bad {
      color: var(--bad);
      border-color: rgba(159, 47, 47, 0.2);
      background: rgba(159, 47, 47, 0.08);
    }
    .pill.neutral {
      color: #0f172a;
      border-color: rgba(148, 163, 184, 0.28);
      background: rgba(241, 245, 249, 0.94);
    }
    .result-grid, .graph-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .graph-grid {
      grid-template-columns: 1fr;
    }
    .summary-list {
      display: grid;
      gap: 8px;
    }
    .summary-item {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
    }
    .summary-item strong {
      display: block;
      font-size: 0.88rem;
      margin: 0;
    }
    .summary-item span {
      color: var(--muted);
      font-size: 0.84rem;
    }
    .summary-value {
      font-size: 0.92rem;
      text-align: right;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 14px;
    }
    .card h3 {
      margin: 0 0 10px;
      font-size: 0.98rem;
    }
    .queue-live {
      display: grid;
      gap: 10px;
    }
    .queue-live-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }
    .queue-live-line strong {
      font-size: 0.88rem;
    }
    .queue-status-text {
      color: var(--muted);
      font-size: 0.88rem;
    }
    .typing-dots {
      display: inline-flex;
      gap: 4px;
      margin-left: 8px;
      vertical-align: middle;
    }
    .typing-dots span {
      width: 5px;
      height: 5px;
      border-radius: 999px;
      background: currentColor;
      opacity: 0.28;
      animation: shipyardPulse 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) {
      animation-delay: 0.2s;
    }
    .typing-dots span:nth-child(3) {
      animation-delay: 0.4s;
    }
    .route-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .queue-timeline {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .timeline-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-strong);
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .timeline-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .timeline-title {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--ink);
    }
    .timeline-subtitle {
      font-size: 0.8rem;
      color: var(--muted);
    }
    .timeline-events {
      display: grid;
      gap: 8px;
    }
    .timeline-event {
      display: grid;
      grid-template-columns: 78px 1fr;
      gap: 10px;
      align-items: start;
      font-size: 0.84rem;
    }
    .timeline-event-time {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .timeline-event-body {
      display: grid;
      gap: 3px;
    }
    .timeline-event-label {
      font-weight: 600;
      color: var(--ink);
    }
    .timeline-event-meta {
      color: var(--muted);
      line-height: 1.35;
      word-break: break-word;
    }
    .route-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.24);
      background: rgba(21, 101, 216, 0.1);
      font-size: 0.78rem;
      color: #0f4fb5;
      font-weight: 600;
    }
    @keyframes shipyardPulse {
      0%, 80%, 100% { opacity: 0.28; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-1px); }
    }
    .card dl {
      margin: 0;
      display: grid;
      gap: 8px;
    }
    .card dt {
      font-size: 0.82rem;
      color: var(--muted);
    }
    .card dd {
      margin: 0;
      font-size: 0.95rem;
      word-break: break-word;
    }
    .graph-files {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }
    .side-panel {
      position: sticky;
      top: 20px;
      display: grid;
      gap: 14px;
      max-height: calc(100vh - 40px);
      overflow: auto;
    }
    .side-panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .side-panel-header h2 {
      margin: 0;
      font-size: 1rem;
    }
    .tab-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .tab-row button {
      border-radius: 999px;
      background: var(--soft);
      color: var(--muted);
      border: 1px solid var(--line);
    }
    .tab-row button.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .tab-panel {
      display: none;
      overflow: auto;
      padding-right: 2px;
    }
    .tab-panel.active {
      display: grid;
      gap: 12px;
    }
    ul {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 8px;
    }
    li button {
      text-align: left;
      background: #fff;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      min-height: 140px;
      overflow: auto;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 0 14px;
    }
    details summary {
      cursor: pointer;
      padding: 14px 0;
      color: var(--muted);
    }
    details[open] summary {
      border-bottom: 1px solid var(--line);
      margin-bottom: 12px;
    }
    .hidden-desktop {
      display: none;
    }
    .hero-note {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      color: var(--muted);
      font-size: 0.9rem;
      white-space: nowrap;
    }
    .summary-inline {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .summary-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      font-size: 0.88rem;
    }
    .summary-chip strong {
      font-size: 0.8rem;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .activity-stream {
      display: grid;
      gap: 12px;
      align-content: start;
      padding: 6px 2px 2px;
    }
    .activity-scroll {
      overflow: auto;
      padding: 10px 2px 14px;
      min-height: 0;
    }
    .message {
      display: grid;
      gap: 8px;
      max-width: min(720px, 100%);
    }
    .message.user {
      justify-self: end;
    }
    .message.assistant {
      justify-self: start;
    }
    .message-label {
      font-size: 0.76rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      padding: 0 4px;
    }
    .message-bubble {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      box-shadow: 0 10px 30px rgba(21, 32, 51, 0.05);
    }
    .message.user .message-bubble {
      background: rgba(21, 101, 216, 0.9);
      color: white;
      border-color: rgba(21, 101, 216, 0.25);
    }
    .message.assistant .message-bubble {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(148, 163, 184, 0.18);
      color: #e8eef8;
    }
    .message.user .message-bubble .meta-line {
      color: rgba(255, 255, 255, 0.84);
    }
    .message-text {
      font-size: 0.98rem;
      line-height: 1.45;
      word-break: break-word;
    }
    .message-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 0.9rem;
    }
    .meta-line {
      color: var(--muted);
    }
    .composer {
      display: grid;
      gap: 10px;
      padding-top: 14px;
      border-top: 1px solid rgba(148, 163, 184, 0.16);
    }
    .composer textarea {
      min-height: 84px;
      max-height: 220px;
      border-radius: 16px;
      border-color: rgba(148, 163, 184, 0.18);
      background: rgba(255, 255, 255, 0.08);
      color: #f8fafc;
      resize: none;
      overflow: hidden;
    }
    .composer textarea::placeholder {
      color: rgba(226, 232, 240, 0.52);
    }
    .composer-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .composer-hint {
      font-size: 0.84rem;
      color: rgba(226, 232, 240, 0.68);
    }
    .ghost-button {
      width: auto;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.14);
      border-color: rgba(148, 163, 184, 0.34);
      color: #ffffff;
      font-weight: 600;
    }
    .ghost-button:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    .ghost-button:disabled,
    .composer textarea:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    @media (max-width: 1060px) {
      .app-shell {
        grid-template-columns: 1fr;
        min-height: auto;
      }
      .terminal-panel {
        height: calc(100vh - 40px);
        max-height: calc(100vh - 40px);
      }
      .side-panel {
        position: static;
        max-height: none;
      }
    }
    @media (max-width: 720px) {
      .row, .result-grid {
        grid-template-columns: 1fr;
      }
      .hero-top, .actions {
        flex-direction: column;
      }
      .actions button, .hero-top button {
        width: 100%;
      }
      .hidden-desktop {
        display: inline-flex;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <main class="main-column">
      <section class="panel terminal-panel">
        <div class="terminal-header">
          <div>
            <h1>Shipyard Terminal</h1>
            <div class="muted" id="workspace_hint">Testing mode writes to Shipyard's managed workspace by default.</div>
          </div>
          <div class="terminal-dots" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
          <button class="ghost-button hidden-desktop" id="panel_toggle">Panel</button>
        </div>
        <div class="activity-scroll">
          <div class="activity-stream" id="activity_stream"></div>
        </div>
        <div class="composer">
          <textarea id="instruction" placeholder='Type a prompt and press Enter.

Shift+Enter adds a new line.'></textarea>
          <div class="composer-bar">
            <div class="composer-hint">Enter to run. Shift+Enter for a new line.</div>
            <div class="composer-actions">
              <button class="ghost-button" id="clear_button">Clear</button>
              <button class="ghost-button" id="workspace_button">New Temp Workspace</button>
            </div>
          </div>
        </div>
      </section>
    </main>

    <aside class="side-panel" id="side_panel">
      <div class="side-panel-header">
        <h2>Side Panel</h2>
        <button class="secondary hidden-desktop" id="panel_close">Close</button>
      </div>
      <div class="tab-row">
        <button class="active" data-tab="details">Details</button>
        <button data-tab="graph">Graph</button>
        <button data-tab="sessions">Sessions</button>
        <button data-tab="raw">Raw</button>
      </div>

      <section class="tab-panel active" id="tab_details">
        <div class="status">
          <div>
            <strong id="planner_title">Checking planner...</strong>
            <span id="planner_subtitle">Please wait.</span>
          </div>
          <div id="planner_pill" class="pill">...</div>
        </div>
        <div class="card">
          <h3>Workspace</h3>
          <p class="muted" id="workspace_details">Testing writes default to the managed workspace unless you override the target path below.</p>
        </div>
        <details>
          <summary>Advanced Options</summary>
          <div class="row">
            <div>
              <label for="session_id">Session ID</label>
              <input id="session_id" placeholder="optional-session-id" />
            </div>
            <div>
              <label for="function_name">Function Name</label>
              <input id="function_name" placeholder="boot_system" />
            </div>
          </div>
          <label for="target_path">Optional Target Path</label>
          <input id="target_path" placeholder="/tmp/demo.py" />
          <div class="row">
            <div>
              <label for="edit_mode">Edit Mode</label>
              <select id="edit_mode">
                <option value="">auto</option>
                <option value="anchor">anchor</option>
                <option value="named_function">named_function</option>
                <option value="write_file">write_file</option>
                <option value="append">append</option>
                <option value="prepend">prepend</option>
                <option value="delete_file">delete_file</option>
                <option value="copy_file">copy_file</option>
              </select>
            </div>
            <div>
              <label for="proposal_mode">Proposal Mode</label>
              <select id="proposal_mode">
                <option value="">auto</option>
                <option value="openai">openai</option>
                <option value="heuristic">heuristic</option>
              </select>
            </div>
          </div>
          <label for="verification_commands">Verification Commands</label>
          <textarea class="compact-textarea" id="verification_commands" placeholder="python3 -m py_compile /tmp/demo.py"></textarea>
          <label for="context_json">Extra Context JSON</label>
          <textarea class="compact-textarea" id="context_json" placeholder='{"file_hint":"/tmp/demo.py"}'></textarea>
        </details>
        <div class="actions">
          <button class="secondary" id="cleanup_button">Clean Runtime</button>
        </div>
        <div class="card">
          <h3>Queue</h3>
          <div class="queue-live" id="queue_status"></div>
          <div class="queue-timeline" id="queue_timeline"></div>
        </div>
      </section>

      <section class="tab-panel" id="tab_graph">
        <div class="status">
          <div>
            <strong id="graph_title">Checking graph status...</strong>
            <span id="graph_subtitle">Please wait.</span>
          </div>
          <div id="graph_pill" class="pill">...</div>
        </div>
        <div class="actions">
          <button class="secondary" id="reindex_button">Rebuild Graph</button>
        </div>
        <section class="card">
          <h3>Graph Summary</h3>
          <div class="summary-inline" id="graph_summary"></div>
        </section>
        <details>
          <summary>More Graph Details</summary>
          <div class="graph-grid">
            <section class="card">
              <h3>Connectivity</h3>
              <dl id="graph_connectivity"></dl>
            </section>
            <section class="card">
              <h3>Index</h3>
              <dl id="graph_index"></dl>
            </section>
            <section class="card">
              <h3>Live Graph</h3>
              <dl id="graph_live"></dl>
            </section>
          </div>
        </details>
      </section>

      <section class="tab-panel" id="tab_sessions">
        <div class="actions" style="margin-top: 0;">
          <button class="secondary" id="sessions_button">Refresh Sessions</button>
        </div>
        <ul id="session_list"></ul>
        <details>
          <summary>Selected Session History</summary>
          <pre id="history_output">[]</pre>
        </details>
      </section>

      <section class="tab-panel" id="tab_raw">
        <div class="card">
          <h3>Raw Result</h3>
          <pre id="result">{}</pre>
        </div>
        <div class="card">
          <h3>Raw Graph Output</h3>
          <pre id="graph_status">{}</pre>
        </div>
      </section>
    </aside>
  </div>
  <script>
    const resultEl = document.getElementById("result");
    const activityStreamEl = document.getElementById("activity_stream");
    const graphEl = document.getElementById("graph_status");
    const graphTitleEl = document.getElementById("graph_title");
    const graphSubtitleEl = document.getElementById("graph_subtitle");
    const graphPillEl = document.getElementById("graph_pill");
    const graphSummaryEl = document.getElementById("graph_summary");
    const graphConnectivityEl = document.getElementById("graph_connectivity");
    const graphIndexEl = document.getElementById("graph_index");
    const graphLiveEl = document.getElementById("graph_live");
    const plannerTitleEl = document.getElementById("planner_title");
    const plannerSubtitleEl = document.getElementById("planner_subtitle");
    const plannerPillEl = document.getElementById("planner_pill");
    const workspaceHintEl = document.getElementById("workspace_hint");
    const workspaceDetailsEl = document.getElementById("workspace_details");
    const historyEl = document.getElementById("history_output");
    const sessionListEl = document.getElementById("session_list");
    const sidePanelEl = document.getElementById("side_panel");
    const queueStatusEl = document.getElementById("queue_status");
    const queueTimelineEl = document.getElementById("queue_timeline");
    const instructionEl = document.getElementById("instruction");
    const STORAGE_KEY = "shipyard.workbench.v2";
    let uiState = loadState();
    let submitInFlight = false;

    function loadState() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
      } catch {
        return {};
      }
    }

    function saveState(patch) {
      uiState = {
        ...uiState,
        ...patch,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(uiState));
    }

    function currentFormState() {
      return {
        instruction: document.getElementById("instruction").value,
        session_id: document.getElementById("session_id").value,
        function_name: document.getElementById("function_name").value,
        target_path: document.getElementById("target_path").value,
        edit_mode: document.getElementById("edit_mode").value,
        proposal_mode: document.getElementById("proposal_mode").value,
        verification_commands: document.getElementById("verification_commands").value,
        context_json: document.getElementById("context_json").value,
      };
    }

    function setComposerBusy(isBusy) {
      submitInFlight = !!isBusy;
      document.getElementById("clear_button").disabled = !!isBusy;
    }

    function clearActivity() {
      document.getElementById("session_id").value = "";
      document.getElementById("target_path").value = "";
      document.getElementById("context_json").value = "";
      saveState({
        activityMessages: [],
        lastResult: null,
        pending: null,
        pendingInstruction: null,
        activeSessionId: null,
        activeJobId: null,
        lastCompletedJobId: null,
        queueStatus: null,
        form: currentFormState(),
      });
      renderActivityStream([]);
      queueStatusEl.innerHTML = "";
      queueTimelineEl.innerHTML = "";
      resultEl.textContent = pretty({});
      instructionEl.focus();
    }

    function sanitizeActivityMessages(messages) {
      if (!Array.isArray(messages)) return [];
      return messages.filter((message) => {
        if (message?.text === "No instruction yet.") return false;
        if (message?.text === "Write an instruction and Shipyard will handle it here.") return false;
        return true;
      });
    }

    function applyFormState(form) {
      if (!form) return;
      for (const [id, value] of Object.entries(form)) {
        const element = document.getElementById(id);
        if (element && typeof value === "string") {
          element.value = value;
        }
      }
    }

    function inferFilenameFromInstruction(instruction) {
      const match = String(instruction || "").match(/\b([\w.-]+\.[A-Za-z0-9]+)\b/);
      return match ? match[1] : null;
    }

    function isStaleScratchTarget(value) {
      const raw = String(value || "").trim();
      if (!raw) return false;
      const name = raw.split("/").pop();
      return /^scratch(?:-[0-9a-f]{6})?\.[A-Za-z0-9]+$/i.test(name);
    }

    function pretty(value) {
      return JSON.stringify(value, null, 2);
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const raw = await response.text();
      let data;
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error(`${response.status} ${response.statusText}: ${raw.slice(0, 160)}`);
      }
      if (!response.ok) {
        const detail = data?.detail;
        const message = typeof detail === "string"
          ? detail
          : detail?.error || detail?.reason || data?.error || `${response.status} ${response.statusText}`;
        throw new Error(message);
      }
      return data;
    }

    function renderDefinitionList(target, entries) {
      target.innerHTML = entries.map(([label, value]) => `
        <div>
          <dt>${label}</dt>
          <dd>${value}</dd>
        </div>
      `).join("");
    }

    function renderQueueStatus(data) {
      const active = data?.active;
      const session = data?.session;
      const dots = `<span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>`;
      const badges = (routing) => {
        if (!routing) return "";
        const relation = routing?.relation_to_previous?.label;
        const actionability = routing?.actionability?.label;
        const urgency = routing?.urgency?.label;
        const values = [relation, actionability, urgency].filter(Boolean);
        if (!values.length) return "";
        return `<div class="route-badges">${values.map((value) => `<span class="route-badge">${value}</span>`).join("")}</div>`;
      };
      const lines = [
        {
          label: "Active Run",
          value: active
            ? `${active.session_id} · ${active.current_task || active.status}${active.status === "running" ? dots : ""}`
            : "—",
          extra: active ? badges(active.routing) : "",
        },
        {
          label: "Current Session",
          value: session
            ? `${session.status}${session.result_status ? ` · ${session.result_status}` : ""}${session.current_task ? ` · ${session.current_task}` : ""}`
            : "—",
          extra: session ? badges(session.routing) : "",
        },
        {
          label: "Queued",
          value: String((data?.queued || []).length),
          extra: "",
        },
      ];
      queueStatusEl.innerHTML = lines.map((line) => `
        <div class="queue-live-line">
          <div>
            <strong>${line.label}</strong>
            ${line.extra || ""}
          </div>
          <span class="queue-status-text">${line.value}</span>
        </div>
      `).join("");
      renderQueueTimeline(data);
    }

    function renderQueuedRun(job, instruction) {
      if (!job?.job_id) return;
      appendActivityMessages([
        {
          id: `user-${job.job_id}`,
          role: "user",
          label: "You",
          text: instruction || "Submitted instruction",
          meta: [],
        },
        {
          id: `assistant-${job.job_id}`,
          role: "assistant",
          label: "Shipyard",
          text: job.status === "queued" ? "Queued." : "Working.",
          badge: job.status === "queued" ? "Queued" : "Working",
          badgeTone: "neutral",
          meta: [job.current_task ? `Task: ${job.current_task}` : null].filter(Boolean),
        },
      ]);
    }

    function summarizeEventPayload(payload) {
      if (!payload || typeof payload !== "object") return "";
      return (
        payload.reason ||
        payload.status ||
        payload.provider_reason ||
        payload.current_task ||
        payload.error ||
        ""
      );
    }

    function renderQueueTimeline(data) {
      const candidates = [data?.session, data?.active, ...(data?.queued || [])]
        .filter(Boolean)
        .filter((job, index, items) => items.findIndex((item) => item.job_id === job.job_id) === index)
        .slice(0, 3);

      if (!candidates.length) {
        queueTimelineEl.innerHTML = "";
        return;
      }

      queueTimelineEl.innerHTML = candidates.map((job) => {
        const events = Array.isArray(job.task_events) ? job.task_events.slice(-6).reverse() : [];
        const sessionLabel = job.session_id || job.job_id || "run";
        return `
          <div class="timeline-card">
            <div class="timeline-head">
              <div>
                <div class="timeline-title">${sessionLabel}</div>
                <div class="timeline-subtitle">${job.status || "unknown"}${job.current_task ? ` · ${job.current_task}` : ""}</div>
              </div>
              <span class="route-badge">${job.result_status || job.status || "pending"}</span>
            </div>
            ${events.length ? `
              <div class="timeline-events">
                ${events.map((event) => `
                  <div class="timeline-event">
                    <div class="timeline-event-time">${(event.timestamp || "").split("T")[1] || "—"}</div>
                    <div class="timeline-event-body">
                      <div class="timeline-event-label">${event.label || event.event || "Event"}</div>
                      <div class="timeline-event-meta">${summarizeEventPayload(event.payload) || "No extra details."}</div>
                    </div>
                  </div>
                `).join("")}
              </div>
            ` : `<div class="timeline-event-meta">No task events yet.</div>`}
          </div>
        `;
      }).join("");
    }

    function renderActivityStream(messages) {
      activityStreamEl.innerHTML = messages.map((message) => `
        <div class="message ${message.role}">
          <div class="message-label">${message.label}</div>
          <div class="message-bubble">
            <div class="message-text">${message.text}</div>
            ${message.badge ? `<div><span class="pill${message.badgeTone ? ` ${message.badgeTone}` : " neutral"}">${message.badge}</span></div>` : ""}
            ${message.meta && message.meta.length ? `<div class="message-meta">${message.meta.map((item) => `<span class="meta-line">${item}</span>`).join("")}</div>` : ""}
          </div>
        </div>
      `).join("");
      activityStreamEl.parentElement.scrollTop = activityStreamEl.parentElement.scrollHeight;
    }

    function appendActivityMessages(messages) {
      const existing = Array.isArray(uiState.activityMessages) ? [...uiState.activityMessages] : [];
      for (const message of messages) {
        if (!message?.id) continue;
        const index = existing.findIndex((item) => item.id === message.id);
        if (index >= 0) {
          existing[index] = message;
        } else {
          existing.push(message);
        }
      }
      saveState({activityMessages: existing});
      renderActivityStream(existing);
    }

    function yesNo(value) {
      return value ? "Yes" : "No";
    }

    function textOrDash(value) {
      if (value === null || value === undefined || value === "") return "—";
      return String(value);
    }

    function selectTab(name) {
      document.querySelectorAll(".tab-row button").forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === name);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab_${name}`);
      });
      saveState({activeTab: name});
    }

    function generateSessionId() {
      return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    }

    function ensureSessionId() {
      const field = document.getElementById("session_id");
      const existing = field.value.trim() || uiState.activeSessionId || uiState.form?.session_id || "";
      const sessionId = existing || generateSessionId();
      field.value = sessionId;
      saveState({
        activeSessionId: sessionId,
        form: currentFormState(),
      });
      return sessionId;
    }

    function getActiveSessionId(data = null) {
      return (
        data?.request?.session_id ||
        data?.session_id ||
        uiState.activeSessionId ||
        uiState.form?.session_id ||
        document.getElementById("session_id").value.trim() ||
        null
      );
    }

    let sessionPollHandle = null;

    function stopSessionPolling() {
      if (sessionPollHandle) {
        clearInterval(sessionPollHandle);
        sessionPollHandle = null;
      }
    }

    function openPanel(tabName = "details") {
      selectTab(tabName);
      sidePanelEl.scrollIntoView({behavior: "smooth", block: "start"});
    }

    function renderGraphDetails(data) {
      const indexFiles = data?.index_state?.files || [];
      const nextStep = data?.ready
        ? (data?.index_state?.stale ? "Sync again after more edits if you need graph-aware function work." : "No action needed.")
        : (!data?.index_state?.has_index || !data?.live_graph_state?.populated ? "Sync the live graph." : "Review graph status.");
      graphSummaryEl.innerHTML = [
        ["State", data?.ready ? "Ready" : "Needs attention"],
        ["Counts", `${textOrDash(data?.live_graph_state?.node_count)} nodes / ${textOrDash(data?.live_graph_state?.relationship_count)} rel`],
        ["Stale", yesNo(data?.index_state?.stale)],
        ["Next", nextStep],
      ].map(([label, value]) => `
        <div class="summary-chip">
          <strong>${label}</strong>
          <span>${value}</span>
        </div>
      `).join("");

      renderDefinitionList(graphConnectivityEl, [
        ["Available", yesNo(data?.available)],
        ["Reason", textOrDash(data?.reason)]
      ]);

      renderDefinitionList(graphIndexEl, [
        ["Has Index File", yesNo(data?.index_state?.has_index)],
        ["Stale", yesNo(data?.index_state?.stale)],
        ["Files", indexFiles.length ? `<ul class="graph-files">${indexFiles.map((file) => `<li>${file}</li>`).join("")}</ul>` : "—"]
      ]);

      renderDefinitionList(graphLiveEl, [
        ["Populated", yesNo(data?.live_graph_state?.populated)],
        ["Node Count", textOrDash(data?.live_graph_state?.node_count)],
        ["Relationship Count", textOrDash(data?.live_graph_state?.relationship_count)]
      ]);
    }

    function summarizeResult(data) {
      const status = data?.status || "unknown";
      const error = data?.error;
      if (error) {
        return {title: `Run finished with ${status}.`, subtitle: error, pill: status, tone: "bad"};
      }
      if (status === "verified" || status === "edited") {
        return {
          title: `Run finished with ${status}.`,
          subtitle: "The latest instruction completed without a recorded error.",
          pill: status,
          tone: "good"
        };
      }
      return {
        title: `Run finished with ${status}.`,
        subtitle: "Use the side panel if you need more detail.",
        pill: status,
        tone: "bad"
      };
    }

    function resultValue(data, path, fallback = "—") {
      const value = path.reduce((current, key) => current?.[key], data);
      return textOrDash(value ?? fallback);
    }

    function renderResultDetails(data) {
      const summary = summarizeResult(data);
      const humanGate = data?.execution?.human_gate || data?.human_gate || {};
      const requestInstruction = resultValue(data, ["request", "instruction"], data?.instruction);
      const targetPath = resultValue(data, ["plan", "target_path"], data?.target_path);
      const nextStep = textOrDash(humanGate?.action);
      const changedFiles = data?.execution?.changed_files || data?.changed_files || [];
      const preview = data?.execution?.file_preview || data?.file_preview;
      const contentHash = data?.execution?.content_hash || data?.content_hash;
      const previewSuffix = data?.execution?.file_preview_truncated || data?.file_preview_truncated ? "..." : "";
      const changedSummary = changedFiles.length
        ? (changedFiles.length === 1 ? `Changed: ${changedFiles[0]}` : `Changed ${changedFiles.length} files`)
        : null;
      const sessionId = getActiveSessionId(data) || "session";
      const resultKey = uiState.activeJobId || uiState.lastCompletedJobId || data?.job_id || data?.trace_path || data?.content_hash || `${sessionId}-${resultValue(data, ["execution", "status"], data?.status)}`;
      appendActivityMessages([
        {
          id: `user-${resultKey}`,
          role: "user",
          label: "You",
          text: requestInstruction === "—" ? "Submitted instruction" : requestInstruction,
          meta: [],
        },
        {
          id: `assistant-${resultKey}`,
          role: "assistant",
          label: "Shipyard",
          text: data?.status === "idle" ? "Write an instruction and Shipyard will handle it here." : (humanGate?.prompt || summary.subtitle),
          badge: summary.pill,
          badgeTone: humanGate?.prompt ? "bad" : summary.tone,
        meta: [
          targetPath !== "—" ? `Target: ${targetPath}` : null,
          changedSummary,
          preview ? `Preview: ${preview}${previewSuffix}` : null,
          nextStep !== "—" && nextStep !== "clarify_request" ? `Next: ${nextStep}` : null,
          contentHash ? `Hash: ${contentHash}` : null,
        ].filter((line) => typeof line === "string"),
        },
      ]);
    }

    function renderPendingState(pending) {
      if (!pending?.kind) return;
      appendActivityMessages([
        {
          id: `pending-${uiState.activeSessionId || "session"}`,
          role: "assistant",
          label: "Shipyard",
          text: pending.title || "Working.",
          badge: pending.kind === "queued" ? "Queued" : "Working",
          badgeTone: "neutral",
          meta: [pending.subtitle || "This page was refreshed while Shipyard was working."],
        },
      ]);
    }

    function summarizeGraphStatus(data) {
      const hasIndex = !!data?.index_state?.has_index;
      const stale = !!data?.index_state?.stale;
      const populated = !!data?.live_graph_state?.populated;
      if (data?.ready) {
        return {
          title: stale ? "Graph is available but stale." : "Graph is ready.",
          subtitle: stale ? "A refresh is recommended after recent edits." : "Named-function edits can use the graph-backed path.",
          pill: stale ? "Refresh Soon" : "Ready",
          tone: stale ? "bad" : "good"
        };
      }
      if (data?.available && hasIndex && !populated) {
        return {
          title: "Graph is connected but empty.",
          subtitle: "Memgraph is reachable, but the live graph still needs data.",
          pill: "Empty Graph",
          tone: "bad"
        };
      }
      if (!hasIndex) {
        return {
          title: "Graph index is missing.",
          subtitle: "Named-function edits stay blocked until the graph is synced.",
          pill: "Not Ready",
          tone: "bad"
        };
      }
      return {
        title: "Graph is not available.",
        subtitle: data?.reason || "Shipyard could not load graph statistics.",
        pill: "Unavailable",
        tone: "bad"
      };
    }

    function summarizePlannerStatus(data) {
      if (data?.default_mode === "openai") {
        return {
          title: "LLM planning is active.",
          subtitle: data?.summary || "Shipyard will ask the model to plan edits first.",
          pill: data?.proposal_model || "OpenAI",
          tone: "good"
        };
      }
      return {
        title: "Heuristic fallback is active.",
        subtitle: data?.summary || "Configure OpenAI to get natural-language planning.",
        pill: "Local Only",
        tone: "bad"
      };
    }

    function parseContext() {
      const raw = document.getElementById("context_json").value.trim();
      if (!raw) return {};
      return JSON.parse(raw);
    }

    function verificationCommands() {
      return document.getElementById("verification_commands").value
        .split("\\n")
        .map((line) => line.trim())
        .filter(Boolean);
    }

    async function loadGraphStatus() {
      const data = await fetchJson("/graph/status");
      const summary = summarizeGraphStatus(data);
      graphTitleEl.textContent = summary.title;
      graphSubtitleEl.textContent = summary.subtitle;
      graphPillEl.textContent = summary.pill;
      graphPillEl.className = `pill ${summary.tone}`;
      renderGraphDetails(data);
      graphEl.textContent = pretty(data);
      saveState({graphStatus: data});
    }

    async function hydrateSessionState(sessionId, {quiet = false} = {}) {
      if (!sessionId) return false;
      try {
        const data = await fetchJson(`/sessions/${sessionId}`);
        renderResultDetails(data);
        resultEl.textContent = pretty(data);
        saveState({lastResult: data, activeSessionId: sessionId});
        if (!quiet) {
          await loadHistory(sessionId);
        }
        return true;
      } catch {
        return false;
      }
    }

    function startSessionPolling(sessionId) {
      if (!sessionId) return;
      stopSessionPolling();
      sessionPollHandle = window.setInterval(async () => {
        const queueData = await loadQueueStatus(sessionId);
        if (uiState.activeJobId) {
          const terminal = await hydrateActiveRun(uiState.activeJobId, sessionId);
          if (terminal) {
            saveState({pending: null, pendingInstruction: null});
            await loadGraphStatus();
            await loadWorkspaceStatus();
            await loadSessions();
            stopSessionPolling();
            return;
          }
        }
        const recovered = !uiState.activeJobId ? await hydrateSessionState(sessionId, {quiet: true}) : false;
        if (recovered && uiState.pending) {
          saveState({pending: null, pendingInstruction: null});
          await loadGraphStatus();
          await loadWorkspaceStatus();
          await loadSessions();
          stopSessionPolling();
          return;
        }
        if (uiState.pending && queueData?.session) {
          if (!uiState.activeJobId) {
            renderPendingState({
              kind: queueData.session.status,
              title: queueData.session.status === "queued" ? "Queued." : "Working.",
              subtitle: queueData.session.status === "queued"
                ? `Waiting behind ${(queueData.queued || []).length} queued run(s).`
                : "Shipyard is processing this run now.",
            });
          }
        }
      }, 2000);
    }

    async function loadQueueJob(jobId) {
      if (!jobId) return null;
      try {
        return await fetchJson(`/queue/job/${encodeURIComponent(jobId)}`);
      } catch {
        return null;
      }
    }

    async function hydrateActiveRun(jobId, sessionId = null) {
      if (!jobId) return false;
      const job = await loadQueueJob(jobId);
      if (!job) return false;

      const terminal = job.status === "completed" || job.status === "failed";
      if (terminal) {
        saveState({activeJobId: null, lastCompletedJobId: job.job_id, pending: null, pendingInstruction: null});
        if (sessionId || job.session_id) {
          await hydrateSessionState(sessionId || job.session_id, {quiet: true});
        }
        return true;
      }

      renderQueuedRun(job, uiState.pendingInstruction || uiState.form?.instruction || "");

      saveState({
        activeJobId: job.job_id,
        activeSessionId: sessionId || job.session_id || uiState.activeSessionId,
      });
      return false;
    }

    async function syncGraph() {
      try {
        openPanel("graph");
        saveState({
          pending: {
            kind: "graph_rebuild",
            title: "Rebuilding graph...",
            subtitle: "Refreshing the live graph and waiting for the latest counts.",
          },
          form: currentFormState(),
        });
        renderResultDetails({status: "syncing_graph"});
        resultEl.textContent = pretty({status: "Syncing live graph..."});
        const data = await fetchJson("/graph/sync", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({clean: true})
        });
        renderResultDetails(data);
        resultEl.textContent = pretty(data);
        saveState({lastResult: data, pending: null, form: currentFormState()});
        await loadGraphStatus();
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        resultEl.textContent = pretty({error: String(error)});
        saveState({pending: null, lastResult: {status: "error", error: String(error)}, form: currentFormState()});
      }
    }

    async function cleanRuntime() {
      try {
        openPanel("details");
        saveState({
          pending: {
            kind: "runtime_cleanup",
            title: "Cleaning runtime data...",
            subtitle: "Removing old local runtime artifacts.",
          },
          form: currentFormState(),
        });
        renderResultDetails({status: "cleaning_runtime"});
        resultEl.textContent = pretty({status: "Cleaning runtime data..."});
        const data = await fetchJson("/runtime/cleanup", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({})
        });
        renderResultDetails({status: "runtime_cleaned"});
        resultEl.textContent = pretty(data);
        saveState({lastResult: {status: "runtime_cleaned", ...data}, pending: null, form: currentFormState()});
        await loadSessions();
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        resultEl.textContent = pretty({error: String(error)});
        saveState({pending: null, lastResult: {status: "error", error: String(error)}, form: currentFormState()});
      }
    }

    async function loadPlannerStatus() {
      const data = await fetchJson("/planner/status");
      const summary = summarizePlannerStatus(data);
      plannerTitleEl.textContent = summary.title;
      plannerSubtitleEl.textContent = summary.subtitle;
      plannerPillEl.textContent = summary.pill;
      plannerPillEl.className = `pill ${summary.tone}`;
      saveState({plannerStatus: data});
    }

    async function loadQueueStatus(sessionId = null) {
      const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
      const data = await fetchJson(`/queue/status${query}`);
      renderQueueStatus(data);
      saveState({queueStatus: data});
      return data;
    }

    async function loadSessions() {
      const data = await fetchJson("/sessions");
      sessionListEl.innerHTML = "";
      for (const session of data.sessions) {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.textContent = `${session.session_id} · ${session.status} · ${session.edit_mode || "auto"}`;
        button.addEventListener("click", () => {
          openPanel("sessions");
          loadHistory(session.session_id);
        });
        item.appendChild(button);
        sessionListEl.appendChild(item);
      }
      if (!data.sessions.length) {
        const empty = document.createElement("li");
        empty.textContent = "No sessions yet.";
        sessionListEl.appendChild(empty);
      }
    }

    async function loadWorkspaceStatus() {
      const data = await fetchJson("/workspace/status");
      workspaceHintEl.textContent = "Testing mode writes to Shipyard's managed workspace by default.";
      workspaceDetailsEl.textContent = "Testing mode writes to Shipyard's managed workspace by default. Use an override path only when you need one.";
      saveState({workspaceStatus: data});
    }

    async function createWorkspace() {
      try {
        const sessionId = uiState.activeSessionId || ensureSessionId();
        const data = await fetchJson("/workspace/temp", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({session_id: sessionId})
        });
        const target = `${data.path}/scratch.py`;
        document.getElementById("target_path").value = target;
        document.getElementById("session_id").value = sessionId;
        const context = parseContext();
        context.file_hint = target;
        document.getElementById("context_json").value = pretty(context);
        renderResultDetails({
          status: "workspace_ready",
          target_path: target,
          proposal_summary: {target_path_source: "workspace_button"}
        });
        resultEl.textContent = pretty({
          workspace_created: data.path,
          suggested_target_path: target
        });
        saveState({
          lastResult: {
            status: "workspace_ready",
            target_path: target,
            proposal_summary: {target_path_source: "workspace_button"},
            workspace_created: data.path,
          },
          activeSessionId: sessionId,
          form: currentFormState(),
        });
        await loadWorkspaceStatus();
        openPanel("details");
      } catch (error) {
        resultEl.textContent = pretty({error: String(error)});
      }
    }

    async function loadHistory(sessionId) {
      const data = await fetchJson(`/sessions/${sessionId}/history`);
      historyEl.textContent = pretty(data.history);
    }

    async function runInstruction() {
      if (submitInFlight) return;
      try {
        setComposerBusy(true);
        const context = parseContext();
        context.testing_mode = true;
        const instructionText = instructionEl.value.trim();
        if (!instructionText) {
          setComposerBusy(false);
          return;
        }
        const inferredFilename = inferFilenameFromInstruction(instructionText);
      let targetPath = document.getElementById("target_path").value.trim();
      if (inferredFilename && isStaleScratchTarget(targetPath)) {
        targetPath = "";
        document.getElementById("target_path").value = "";
      }
        const functionName = document.getElementById("function_name").value.trim();
        const sessionField = document.getElementById("session_id");
        let sessionId = sessionField.value.trim();
        if (!sessionId) {
          sessionId = generateSessionId();
          sessionField.value = sessionId;
        }
        if (!targetPath && context.file_hint) {
          delete context.file_hint;
        }
        if (targetPath && !context.file_hint) context.file_hint = targetPath;
        if (functionName) context.function_name = functionName;

        const payload = {
          session_id: sessionId,
          instruction: instructionText,
          target_path: targetPath || null,
          edit_mode: document.getElementById("edit_mode").value || null,
          proposal_mode: document.getElementById("proposal_mode").value || null,
          context,
          verification_commands: verificationCommands()
        };

        saveState({
          pending: {
            kind: "queued",
            title: "Run queued...",
            subtitle: "Shipyard accepted the run and will process it shortly.",
          },
          form: currentFormState(),
          activeSessionId: sessionId,
          pendingInstruction: payload.instruction,
        });
        appendActivityMessages([
          {
            id: `user-pending-${sessionId}`,
            role: "user",
            label: "You",
            text: payload.instruction,
            meta: [],
          },
          {
            id: `assistant-pending-${sessionId}`,
            role: "assistant",
            label: "Shipyard",
            text: "Queued.",
            badge: "Queued",
            badgeTone: "neutral",
            meta: [],
          },
        ]);

        const data = await fetchJson("/queue/instruct", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
      const resolvedSessionId = data?.session_id || sessionId;
      const jobId = data?.job_id || data?.queue_job?.job_id || null;
      document.getElementById("session_id").value = resolvedSessionId;
      appendActivityMessages([
        {
          id: `user-${jobId || resolvedSessionId}`,
          role: "user",
          label: "You",
          text: payload.instruction,
          meta: [],
        },
      ]);
      if (jobId) {
          const queueJob = data?.queue_job || {
            job_id: jobId,
            session_id: resolvedSessionId,
            status: data?.status || "queued",
            current_task: data?.current_task || "Waiting",
          };
          renderQueuedRun(queueJob, payload.instruction);
        } else {
          renderResultDetails(data);
        }
        resultEl.textContent = pretty(data);
        saveState({
          lastResult: data,
          form: currentFormState(),
          activeSessionId: resolvedSessionId,
          activeJobId: jobId,
          lastCompletedJobId: null,
          pendingInstruction: payload.instruction,
        });
        saveState({form: currentFormState()});
        if (jobId) {
          startSessionPolling(resolvedSessionId);
          await loadQueueStatus(resolvedSessionId);
        } else {
          saveState({pending: null, pendingInstruction: null, lastCompletedJobId: null});
          stopSessionPolling();
          await loadQueueStatus(resolvedSessionId);
        }
        await loadSessions();
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        resultEl.textContent = pretty({error: String(error)});
        saveState({pending: null, lastResult: {status: "error", error: String(error)}, form: currentFormState()});
        stopSessionPolling();
      } finally {
        const activeSession = document.getElementById("session_id").value.trim() || uiState.activeSessionId || "";
        const pendingId = activeSession || "session";
        saveState({
          activityMessages: (uiState.activityMessages || []).filter((message) => {
            return message?.id !== `user-pending-${pendingId}` && message?.id !== `assistant-pending-${pendingId}`;
          }),
        });
        setComposerBusy(false);
      }
    }

    document.getElementById("workspace_button").addEventListener("click", createWorkspace);
    document.getElementById("clear_button").addEventListener("click", clearActivity);
    document.getElementById("reindex_button").addEventListener("click", syncGraph);
    document.getElementById("cleanup_button").addEventListener("click", cleanRuntime);
    document.getElementById("sessions_button").addEventListener("click", () => {
      openPanel("sessions");
      loadSessions();
    });
    document.getElementById("panel_toggle").addEventListener("click", () => openPanel("details"));
    document.getElementById("panel_close").addEventListener("click", () => window.scrollTo({top: 0, behavior: "smooth"}));
    document.querySelectorAll(".tab-row button").forEach((button) => {
      button.addEventListener("click", () => selectTab(button.dataset.tab));
    });
    document.querySelectorAll("input, textarea, select").forEach((element) => {
      element.addEventListener("input", () => saveState({form: currentFormState()}));
      element.addEventListener("change", () => saveState({form: currentFormState()}));
    });
    instructionEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        runInstruction();
      }
    });

    applyFormState(uiState.form);
    uiState.activityMessages = sanitizeActivityMessages(uiState.activityMessages);
    saveState({activityMessages: uiState.activityMessages});
    if (Array.isArray(uiState.activityMessages) && uiState.activityMessages.length) {
      renderActivityStream(uiState.activityMessages);
    }
    if (uiState.lastResult) {
      resultEl.textContent = pretty(uiState.lastResult);
    } else {
      resultEl.textContent = pretty({});
    }
    if (uiState.pending) {
      renderPendingState(uiState.pending);
    }
    if (uiState.activeTab) {
      selectTab(uiState.activeTab);
    }
    async function initializeWorkbench() {
      loadPlannerStatus();
      loadWorkspaceStatus();
      loadGraphStatus();
      loadSessions();
      const restoredSessionId = uiState.activeSessionId || getActiveSessionId(uiState.lastResult);
      const restoredJobId = uiState.activeJobId || null;
      await loadQueueStatus(restoredSessionId);
      if (restoredJobId) {
        const terminal = await hydrateActiveRun(restoredJobId, restoredSessionId);
        if (!terminal) {
          startSessionPolling(restoredSessionId);
          return;
        }
      }
      if (restoredSessionId) {
        hydrateSessionState(restoredSessionId, {quiet: true});
      }
      if (uiState.pending && restoredSessionId) {
        startSessionPolling(restoredSessionId);
      }
    }

    initializeWorkbench();
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workbench", response_class=HTMLResponse)
def workbench() -> HTMLResponse:
    return HTMLResponse(WORKBENCH_HTML)


@app.get("/sessions")
def list_sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": session_store.list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    state = session_store.load_latest_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return state


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str) -> dict[str, list[dict[str, Any]]]:
    history = session_store.load_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session history not found.")
    return {"history": history}


@app.post("/instruct")
def instruct(request: InstructionRequest) -> dict[str, Any]:
    state = _normalize_payload(request.model_dump())
    return run_once(graph_app, session_store, state)


@app.post("/queue/instruct")
def queue_instruct(request: InstructionRequest) -> dict[str, Any]:
    if _should_run_direct(request):
        state = _normalize_payload(request.model_dump())
        result = run_once(graph_app, session_store, state)
        result["queue_job"] = run_queue.record_direct_run(state, result)
        return result
    state = _normalize_payload(request.model_dump())
    if not state.get("session_id"):
        from .main import _ensure_session_id

        state["session_id"] = _ensure_session_id(None)
    return run_queue.enqueue(state)


@app.get("/queue/status")
def queue_status(session_id: str | None = None) -> dict[str, Any]:
    return run_queue.get_status(session_id)


@app.get("/queue/job/{job_id}")
def queue_job(job_id: str) -> dict[str, Any]:
    job = run_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Queued job not found.")
    return job


@app.get("/graph/status")
def graph_status() -> dict[str, Any]:
    return inspect_code_graph_status()


@app.get("/planner/status")
def planner_status() -> dict[str, Any]:
    return get_planner_status()


@app.get("/workspace/status")
def workspace_status() -> dict[str, str | bool]:
    return get_workspace_status()


@app.post("/workspace/temp")
def workspace_temp(request: WorkspaceCreateRequest) -> dict[str, str]:
    workspace = get_session_workspace(request.session_id)
    return {"path": str(workspace.resolve())}


@app.post("/runtime/cleanup")
def runtime_cleanup(request: CleanupRequest) -> dict[str, Any]:
    return cleanup_runtime_data(
        keep_traces=request.keep_traces,
        keep_snapshots=request.keep_snapshots,
        keep_sessions=request.keep_sessions,
        keep_logs=request.keep_logs,
        remove_empty_workspaces=request.remove_empty_workspaces,
        remove_empty_spec_dirs=request.remove_empty_spec_dirs,
    )


@app.post("/graph/index")
def graph_index(request: GraphIndexRequest) -> dict[str, Any]:
    result = index_code_graph(request.workdir, request.output_dir)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/graph/sync")
def graph_sync(request: GraphIndexRequest) -> dict[str, Any]:
    result = sync_live_code_graph(request.workdir, request.clean)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return {
        "status": "graph_rebuilt",
        "ok": result.get("ok"),
        "reason": result.get("reason"),
        "code_graph_status": {
            "ready": result.get("ready"),
            "index_state": result.get("index_state", {}),
            "live_graph_state": result.get("live_graph_state", {}),
        },
        "graph_sync": result,
    }


@app.get("/git/status")
def git_status() -> dict[str, Any]:
    try:
        return git_automation.get_status()
    except GitAutomationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/git/branch")
def git_branch(request: GitBranchRequest) -> dict[str, str]:
    try:
        return git_automation.create_branch(request.branch_name)
    except GitAutomationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/git/commit")
def git_commit(request: GitCommitRequest) -> dict[str, str]:
    try:
        return git_automation.commit(request.message, request.paths or None)
    except GitAutomationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
