from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .graph import build_graph
from .main import _normalize_payload, run_once
from .runtime_cleanup import cleanup_runtime_data
from .storage_paths import LOGS_ROOT, ensure_dir
from .workspaces import (
    get_managed_workspace,
    get_session_workspace,
    get_session_workspace_selection,
    get_workspace_status,
    list_repo_workspace_folders,
    normalize_repo_workspace_path,
    set_session_workspace,
)
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
    wide_impact_approved: bool = False


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


class WorkspaceSelectRequest(BaseModel):
    session_id: str
    workspace_path: str | None = None


class CleanupRequest(BaseModel):
    keep_traces: int = 20
    keep_snapshots: int = 20
    keep_sessions: int = 20
    keep_logs: int = 20
    remove_empty_workspaces: bool = True
    remove_empty_spec_dirs: bool = True


class QueueCancelRequest(BaseModel):
    job_id: str


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
      padding: 3px 8px;
      font-size: 0.74rem;
      font-weight: 600;
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
    .task-panel {
      display: grid;
      gap: 12px;
    }
    .task-panel-empty {
      border: 1px dashed var(--line);
      border-radius: 14px;
      padding: 14px;
      color: var(--muted);
      background: var(--soft);
    }
    .task-panel-section {
      display: grid;
      gap: 10px;
    }
    .task-panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .task-panel-header h3 {
      margin: 0;
      font-size: 0.96rem;
    }
    .task-panel-meta {
      color: var(--muted);
      font-size: 0.84rem;
    }
    .session-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 10px;
      color: var(--ink);
    }
    .session-item:hover {
      background: rgba(255,255,255,0.1);
    }
    .session-excerpt {
      font-size: 0.88rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
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
      gap: 6px;
      align-content: start;
      padding: 4px 0;
    }
    .activity-scroll {
      overflow: auto;
      padding: 8px 2px 10px;
      min-height: 0;
    }
    .message {
      display: grid;
      gap: 2px;
      max-width: 100%;
    }
    .message.user {
      justify-self: end;
      max-width: 85%;
    }
    .message.assistant {
      justify-self: start;
    }
    .message-label {
      font-size: 0.7rem;
      color: rgba(226, 232, 240, 0.5);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
      padding: 0 2px;
    }
    .message-bubble {
      display: grid;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      background: transparent;
    }
    .message.user .message-bubble {
      background: rgba(21, 101, 216, 0.18);
      color: #c4daf8;
      border-color: rgba(21, 101, 216, 0.25);
    }
    .message.assistant .message-bubble {
      background: transparent;
      border-color: rgba(148, 163, 184, 0.1);
      color: #c8d4e4;
    }
    .message.user .message-bubble .meta-line {
      color: rgba(196, 218, 248, 0.7);
    }
    .message-text {
      font-size: 0.88rem;
      line-height: 1.4;
      word-break: break-word;
    }
    .message-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 0.78rem;
      color: rgba(226, 232, 240, 0.5);
    }
    .message-meta .meta-line:not(:last-child)::after {
      content: "·";
      margin-left: 8px;
      color: rgba(226, 232, 240, 0.3);
    }
    .message-details {
      display: grid;
      gap: 4px;
    }
    .task-block {
      display: grid;
      gap: 4px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.1);
      background: rgba(15, 23, 42, 0.2);
      font-size: 0.82rem;
    }
    .task-block-title {
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: rgba(226, 232, 240, 0.5);
      font-weight: 600;
    }
    .task-grid {
      display: grid;
      gap: 8px;
    }
    .task-card {
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(255, 255, 255, 0.04);
    }
    .task-card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .task-card-id {
      font-size: 0.8rem;
      color: rgba(226, 232, 240, 0.68);
      font-weight: 700;
    }
    .task-card-status {
      font-size: 0.76rem;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      color: rgba(248, 250, 252, 0.9);
      background: rgba(255, 255, 255, 0.06);
    }
    .task-card-goal {
      font-size: 0.92rem;
      line-height: 1.4;
      color: #f8fafc;
    }
    .task-card-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 0.8rem;
      color: rgba(226, 232, 240, 0.7);
    }
    .task-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(255, 255, 255, 0.05);
    }
    .meta-line {
      color: var(--muted);
    }
    .composer {
      display: grid;
      gap: 6px;
      padding-top: 10px;
      border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
    .composer textarea {
      min-height: 40px;
      max-height: 160px;
      padding: 10px 14px;
      border-radius: 12px;
      border-color: rgba(148, 163, 184, 0.15);
      background: rgba(255, 255, 255, 0.06);
      color: #f8fafc;
      resize: none;
      overflow: hidden;
      font-size: 0.9rem;
      line-height: 1.4;
    }
    .composer textarea:focus {
      border-color: rgba(21, 101, 216, 0.4);
      outline: none;
      background: rgba(255, 255, 255, 0.08);
    }
    .composer textarea::placeholder {
      color: rgba(226, 232, 240, 0.4);
    }
    .composer-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .composer-hint {
      font-size: 0.76rem;
      color: rgba(226, 232, 240, 0.4);
    }
    .ghost-button {
      width: auto;
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(148, 163, 184, 0.2);
      color: rgba(255, 255, 255, 0.7);
      font-weight: 600;
      font-size: 0.78rem;
    }
    .ghost-button:hover {
      background: rgba(255, 255, 255, 0.14);
      color: #fff;
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
    .git-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(255, 255, 255, 0.07);
      color: rgba(226, 232, 240, 0.82);
      font-size: 0.8rem;
      font-weight: 600;
      white-space: nowrap;
    }
    .git-badge.clean {
      border-color: rgba(31, 180, 100, 0.28);
      background: rgba(31, 180, 100, 0.1);
      color: #6ee7b7;
    }
    .git-badge.dirty {
      border-color: rgba(251, 191, 36, 0.28);
      background: rgba(251, 191, 36, 0.08);
      color: #fcd34d;
    }
    .verify-block {
      display: grid;
      gap: 3px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid;
      font-size: 0.78rem;
    }
    .verify-block.pass {
      border-color: rgba(31, 180, 100, 0.2);
      background: rgba(31, 180, 100, 0.06);
      color: #6ee7b7;
    }
    .verify-block.fail {
      border-color: rgba(220, 60, 60, 0.2);
      background: rgba(220, 60, 60, 0.06);
      color: #fca5a5;
    }
    .verify-block-label {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      opacity: 0.7;
    }
    .verify-line {
      opacity: 0.8;
      word-break: break-word;
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 0.76rem;
    }
    .gate-block {
      display: grid;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid rgba(251, 191, 36, 0.2);
      background: rgba(251, 191, 36, 0.05);
    }
    .gate-block-label {
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #fcd34d;
    }
    .gate-block-text {
      font-size: 0.9rem;
      color: rgba(226, 232, 240, 0.9);
      line-height: 1.4;
    }
    .gate-approve-btn {
      width: auto;
      align-self: start;
      padding: 8px 16px;
      border-radius: 999px;
      border: 1px solid rgba(251, 191, 36, 0.4);
      background: rgba(251, 191, 36, 0.15);
      color: #fcd34d;
      font-weight: 700;
      font-size: 0.88rem;
      cursor: pointer;
    }
    .gate-approve-btn:hover {
      background: rgba(251, 191, 36, 0.25);
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <main class="main-column">
      <section class="panel terminal-panel">
        <div class="terminal-header">
          <div>
            <h1>Shipyard</h1>
            <div class="muted" id="workspace_hint">Testing mode is attached to the managed workspace until you pick a repo folder.</div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span id="git_badge" class="git-badge" style="display:none;"></span>
            <div class="terminal-dots" aria-hidden="true">
              <span></span><span></span><span></span>
            </div>
            <button class="ghost-button hidden-desktop" id="panel_toggle">Panel</button>
          </div>
        </div>
        <div class="activity-scroll">
          <div class="activity-stream" id="activity_stream"></div>
        </div>
        <div class="composer">
          <textarea id="instruction" placeholder='What should Shipyard do?'></textarea>
          <div class="composer-bar">
            <div class="composer-hint">Enter to run. Shift+Enter for a new line.</div>
            <div class="composer-actions">
              <button class="ghost-button" id="clear_button">Clear</button>
              <button class="ghost-button" id="workspace_button">Use Managed Workspace</button>
            </div>
          </div>
        </div>
      </section>
    </main>

    <aside class="side-panel" id="side_panel">
      <div class="side-panel-header">
        <h2>Shipyard</h2>
        <button class="secondary hidden-desktop" id="panel_close">Close</button>
      </div>
      <div class="tab-row">
        <button class="active" data-tab="details">Status</button>
        <button data-tab="tasks">Steps</button>
        <button data-tab="graph">Graph</button>
        <button data-tab="sessions">History</button>
        <button data-tab="raw">Debug</button>
      </div>

      <section class="tab-panel active" id="tab_details">
        <div class="card" style="border-color: rgba(21,101,216,0.22); background: rgba(21,101,216,0.04);">
          <h3 style="color: var(--accent);">Queue</h3>
          <div class="queue-live" id="queue_status"></div>
          <div class="queue-timeline" id="queue_timeline"></div>
        </div>
        <div class="status" style="padding: 10px 14px;">
          <div>
            <strong id="planner_title" style="font-size:0.93rem;">Checking planner...</strong>
            <span id="planner_subtitle" style="font-size:0.82rem; display:block; margin-top:2px;"></span>
          </div>
          <div id="planner_pill" class="pill" style="font-size:0.8rem; padding:4px 10px;">...</div>
        </div>
        <details>
          <summary>Settings &amp; Workspace</summary>
          <p class="muted" id="workspace_details" style="font-size:0.84rem; margin: 8px 0 4px;">Attach this session to any folder — type a path or pick from the list.</p>
          <label for="workspace_select">Workspace Path</label>
          <input id="workspace_select" placeholder="/home/aaron/projects/gauntlet/ship/ship-rebuild" list="workspace_options" />
          <datalist id="workspace_options"></datalist>
          <div class="actions" style="margin-top:8px;">
            <button class="secondary" id="workspace_select_button">Use This Folder</button>
            <button class="secondary" id="workspace_refresh_button">Refresh List</button>
          </div>
          <div class="row" style="margin-top:12px;">
            <div>
              <label for="session_id">Session ID</label>
              <input id="session_id" placeholder="auto-generated" />
            </div>
            <div>
              <label for="function_name">Function Name</label>
              <input id="function_name" placeholder="boot_system" />
            </div>
          </div>
          <label for="target_path">Target Path</label>
          <input id="target_path" placeholder="/path/to/file.py" />
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
          <textarea class="compact-textarea" id="verification_commands" placeholder="python3 -m py_compile /path/to/file.py"></textarea>
          <label for="context_json">Extra Context JSON</label>
          <textarea class="compact-textarea" id="context_json" placeholder='{"file_hint":"/path/to/file.py"}'></textarea>
          <div class="actions" style="margin-top:12px;">
            <button class="secondary" id="cleanup_button">Clean Runtime Data</button>
          </div>
        </details>
      </section>

      <section class="tab-panel" id="tab_tasks">
        <div class="task-panel">
          <div class="status">
            <div>
              <strong id="tasks_title">No active task graph.</strong>
              <span id="tasks_subtitle" class="muted" style="font-size:0.84rem;">Run a prompt to see the execution plan.</span>
            </div>
            <div id="tasks_pill" class="pill neutral">Idle</div>
          </div>
          <div class="task-panel-section">
            <div class="task-panel-header">
              <h3>Tasks</h3>
              <span class="task-panel-meta" id="tasks_summary">—</span>
            </div>
            <div id="tasks_block" class="task-panel-empty">Nothing yet.</div>
          </div>
          <div class="task-panel-section">
            <div class="task-panel-header">
              <h3>Queue Events</h3>
              <span class="task-panel-meta" id="tasks_queue_summary">—</span>
            </div>
            <div id="tasks_queue_block" class="task-panel-empty">Nothing yet.</div>
          </div>
        </div>
      </section>

      <section class="tab-panel" id="tab_graph">
        <div class="status">
          <div>
            <strong id="graph_title">Checking graph status...</strong>
            <span id="graph_subtitle" class="muted" style="font-size:0.84rem;">Please wait.</span>
          </div>
          <div id="graph_pill" class="pill">...</div>
        </div>
        <div class="summary-inline" id="graph_summary" style="margin: 4px 0;"></div>
        <div class="actions" style="margin-top: 8px;">
          <button class="secondary" id="reindex_button">Rebuild Graph</button>
        </div>
        <details style="margin-top: 8px;">
          <summary>Details</summary>
          <dl id="graph_details" style="margin-top: 8px;"></dl>
        </details>
      </section>

      <section class="tab-panel" id="tab_sessions">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px;">
          <span class="muted" style="font-size:0.84rem;">Click a session to load its history.</span>
          <button class="secondary" id="sessions_button" style="width:auto; padding:6px 12px; font-size:0.84rem;">Refresh</button>
        </div>
        <ul id="session_list" style="list-style:none; padding:0; margin:0; display:grid; gap:6px;"></ul>
        <details style="margin-top:12px;">
          <summary style="font-size:0.84rem;">Raw History JSON</summary>
          <div style="position:relative; margin-top:6px;">
            <button class="secondary" onclick="navigator.clipboard.writeText(document.getElementById('history_output').textContent)" style="position:absolute;top:8px;right:8px;width:auto;padding:4px 10px;font-size:0.78rem;z-index:1;">Copy</button>
            <pre id="history_output" style="max-height:320px; overflow-y:auto; padding-top:36px;">[]</pre>
          </div>
        </details>
      </section>

      <section class="tab-panel" id="tab_raw">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
          <span class="muted" style="font-size:0.84rem;">Raw JSON for last run and graph state.</span>
          <button class="secondary" id="debug_refresh_button" style="width:auto; padding:6px 12px; font-size:0.84rem;">Refresh</button>
        </div>
        <div class="card" style="padding:10px 12px;">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <strong style="font-size:0.84rem;">Last Run</strong>
            <button class="secondary" onclick="navigator.clipboard.writeText(document.getElementById('result').textContent)" style="width:auto; padding:3px 10px; font-size:0.78rem;">Copy</button>
          </div>
          <pre id="result" style="max-height:300px; overflow-y:auto; margin:0;">{}</pre>
        </div>
        <div class="card" style="padding:10px 12px; margin-top:10px;">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <strong style="font-size:0.84rem;">Graph State</strong>
            <button class="secondary" onclick="navigator.clipboard.writeText(document.getElementById('graph_status').textContent)" style="width:auto; padding:3px 10px; font-size:0.78rem;">Copy</button>
          </div>
          <pre id="graph_status" style="max-height:300px; overflow-y:auto; margin:0;">{}</pre>
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
    const graphDetailsEl = document.getElementById("graph_details");
    const tasksTitleEl = document.getElementById("tasks_title");
    const tasksSubtitleEl = document.getElementById("tasks_subtitle");
    const tasksPillEl = document.getElementById("tasks_pill");
    const tasksSummaryEl = document.getElementById("tasks_summary");
    const tasksQueueSummaryEl = document.getElementById("tasks_queue_summary");
    const tasksBlockEl = document.getElementById("tasks_block");
    const tasksQueueBlockEl = document.getElementById("tasks_queue_block");
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
    let showInternalHelpers = false;

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

    function getRunMessageMap() {
      const value = uiState.runMessageMap;
      return value && typeof value === "object" ? value : {};
    }

    function saveRunMessageMap(map) {
      saveState({runMessageMap: map});
    }

    function rememberRunMessageIds(keys, ids) {
      const map = {...getRunMessageMap()};
      for (const key of keys || []) {
        if (!key) continue;
        map[String(key)] = ids;
      }
      saveRunMessageMap(map);
    }

    function lookupRunMessageIds(keys) {
      const map = getRunMessageMap();
      for (const key of keys || []) {
        if (!key) continue;
        const value = map[String(key)];
        if (value?.userId && value?.assistantId) return value;
      }
      return null;
    }

    function getSubmittedInstructions() {
      const value = uiState.submittedInstructions;
      return value && typeof value === "object" ? value : {};
    }

    function rememberSubmittedInstruction(keys, text) {
      const submitted = {...getSubmittedInstructions()};
      for (const key of keys || []) {
        if (!key) continue;
        submitted[String(key)] = text;
      }
      saveState({submittedInstructions: submitted});
    }

    function lookupSubmittedInstruction(keys) {
      const submitted = getSubmittedInstructions();
      for (const key of keys || []) {
        if (!key) continue;
        const value = submitted[String(key)];
        if (typeof value === "string" && value.trim()) return value;
      }
      return null;
    }

    function findActivityMessage(id) {
      return (Array.isArray(uiState.activityMessages) ? uiState.activityMessages : []).find((message) => message?.id === id) || null;
    }

    function currentFormState() {
      return {
        instruction: document.getElementById("instruction").value,
        session_id: document.getElementById("session_id").value,
        workspace_select: document.getElementById("workspace_select").value,
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
      document.getElementById("workspace_select").value = "";
      document.getElementById("target_path").value = "";
      document.getElementById("context_json").value = "";
      saveState({
        activityMessages: [],
        lastResult: null,
        pending: null,
        pendingInstruction: null,
        runMessageMap: {},
        activeSessionId: null,
        activeJobId: null,
        lastCompletedJobId: null,
        queueStatus: null,
        submittedInstructions: {},
        form: currentFormState(),
      });
      renderActivityStream([]);
      queueStatusEl.innerHTML = "";
      queueTimelineEl.innerHTML = "";
      resultEl.textContent = pretty({});
      renderTasksPanel(null, null);
      instructionEl.focus();
    }

    function clearPendingState() {
      const pendingId = uiState.activeSessionId || "session";
      const messages = Array.isArray(uiState.activityMessages) ? uiState.activityMessages : [];
      const filtered = messages.filter((message) => message?.id !== `pending-${pendingId}`);
      saveState({pending: null, pendingInstruction: null, activityMessages: filtered});
      renderActivityStream(filtered);
    }

    function clearTransientRunMessages() {
      const messages = Array.isArray(uiState.activityMessages) ? uiState.activityMessages : [];
      const filtered = messages.filter((message) => {
        if (message?.role !== "assistant") return true;
        if (message?.badge === "Queued" || message?.badge === "Working") return false;
        if (message?.text === "Queued." || message?.text === "Working.") return false;
        return true;
      });
      saveState({activityMessages: filtered});
      renderActivityStream(filtered);
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

    function extractExplicitFilenames(instruction) {
      const matches = String(instruction || "").match(/\b[\w.-]+\.[A-Za-z0-9]+\b/g) || [];
      return [...new Set(matches)];
    }

    function isExplicitScaffoldPrompt(instruction) {
      return extractExplicitFilenames(instruction).length > 1;
    }

    function isStaleScratchTarget(value) {
      const raw = String(value || "").trim();
      if (!raw) return false;
      const name = raw.split("/").pop();
      return /^(?:scratch|file)(?:-[0-9a-f]{6})?\.[A-Za-z0-9]+$/i.test(name);
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
      const queueMeta = (job) => job?.queue || {};
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
            ? `${active.session_id} · ${queueMeta(active).state || active.status}${(queueMeta(active).state || active.status) === "running" ? dots : ""}${queueMeta(active).current_task ? ` · ${queueMeta(active).current_task}` : ""}`
            : "—",
          extra: active ? badges(queueMeta(active).routing) : "",
        },
        {
          label: "Current Session",
          value: session
            ? `${queueMeta(session).state || session.status}${queueMeta(session).result_status ? ` · ${queueMeta(session).result_status}` : ""}${queueMeta(session).current_task ? ` · ${queueMeta(session).current_task}` : ""}`
            : "—",
          extra: session ? badges(queueMeta(session).routing) : "",
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
      const activeJobId = queueMeta(active).job_id || queueMeta(session).job_id || null;
      const activeState = queueMeta(active).state || queueMeta(session).state || null;
      if (activeJobId && isLiveQueueState(activeState)) {
        queueStatusEl.innerHTML += `<div class="actions"><button id="cancel_run_button" class="secondary">Cancel Run</button></div>`;
        const button = document.getElementById("cancel_run_button");
        if (button) button.addEventListener("click", () => cancelRun(activeJobId));
      }
      renderQueueTimeline(data);
      renderTasksPanel(null, active || session || null);
    }

    function renderQueuedRun(job, instruction) {
      const queue = job?.queue || job || {};
      if (!queue?.job_id) return;
      const state = queue.state || job.status;
      const isQueued = state === "queued";
      const currentTask = queue.current_task && queue.current_task !== "Waiting" ? queue.current_task : null;
      const tasks = Array.isArray(queue.task_events)
        ? queue.task_events
            .filter((event) => event?.payload?.instruction)
            .slice(-3)
            .map((event, index) => ({
              task_id: `live-${index + 1}`,
              role: "lead-agent",
              agent_type: "primary",
              goal: event.payload.instruction,
              status: event.label || event.event || state,
              allowed_actions: [],
            }))
        : [];
      appendActivityMessages([
        {
          id: `assistant-${queue.job_id}`,
          role: "assistant",
          label: "Shipyard",
          text: isQueued ? "Queued." : (state === "verifying" ? "Verifying." : "Working."),
          detailsHtml: renderTaskBlock(tasks, []),
          badge: isQueued ? null : state,
          badgeTone: "neutral",
          meta: [currentTask ? `Task: ${currentTask}` : null, queue.state ? `State: ${queue.state}` : null].filter(Boolean),
        },
      ]);
      renderTasksPanel(null, job);
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
        .filter((job, index, items) => {
          const jobId = job?.queue?.job_id || job?.job_id;
          return items.findIndex((item) => (item?.queue?.job_id || item?.job_id) === jobId) === index;
        })
        .slice(0, 3);

      if (!candidates.length) {
        queueTimelineEl.innerHTML = "";
        renderTasksPanel(uiState.lastResult || null, null);
        return;
      }

      queueTimelineEl.innerHTML = candidates.map((job) => {
        const queue = job?.queue || {};
        const events = Array.isArray(queue.task_events) ? queue.task_events.slice(-6).reverse() : [];
        const sessionLabel = job.session_id || queue.job_id || "run";
        return `
          <div class="timeline-card">
            <div class="timeline-head">
              <div>
                <div class="timeline-title">${sessionLabel}</div>
                <div class="timeline-subtitle">${job.status || "unknown"}${queue.current_task ? ` · ${queue.current_task}` : ""}</div>
              </div>
              <span class="route-badge">${queue.result_status || job.status || "pending"}</span>
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

    function _renderMessageHtml(message) {
      return `
        <div class="message ${message.role}" data-msg-id="${message.id || ''}">
          <div class="message-label">${message.label || ''}</div>
          <div class="message-bubble">
            <div class="message-text">${message.text || ''}</div>
            ${message.detailsHtml ? `<div class="message-details">${message.detailsHtml}</div>` : ""}
            ${message.badge ? `<span class="pill${message.badgeTone ? ` ${message.badgeTone}` : " neutral"}">${message.badge}</span>` : ""}
            ${message.meta && message.meta.length ? `<div class="message-meta">${message.meta.map((item) => `<span class="meta-line">${item}</span>`).join("")}</div>` : ""}
          </div>
        </div>`;
    }

    function renderActivityStream(messages) {
      // Incremental update: only re-render changed messages to avoid flicker
      const existingEls = activityStreamEl.querySelectorAll("[data-msg-id]");
      const existingIds = new Set([...existingEls].map(el => el.dataset.msgId));
      const newIds = new Set(messages.map(m => m.id || ''));

      // Remove messages no longer present
      existingEls.forEach(el => {
        if (!newIds.has(el.dataset.msgId)) el.remove();
      });

      // Add or update messages
      for (const message of messages) {
        const id = message.id || '';
        const existing = activityStreamEl.querySelector(`[data-msg-id="${CSS.escape(id)}"]`);
        if (existing) {
          // Update in place — only touch innerHTML if content changed
          const html = _renderMessageHtml(message);
          const temp = document.createElement('div');
          temp.innerHTML = html;
          const newEl = temp.firstElementChild;
          if (existing.innerHTML !== newEl.innerHTML) {
            existing.innerHTML = newEl.innerHTML;
            existing.className = newEl.className;
          }
        } else {
          activityStreamEl.insertAdjacentHTML('beforeend', _renderMessageHtml(message));
        }
      }
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

    function replaceActivityMessageIds(replacements) {
      const existing = Array.isArray(uiState.activityMessages) ? [...uiState.activityMessages] : [];
      for (const replacement of replacements) {
        const fromId = replacement?.fromId;
        const toMessage = replacement?.toMessage;
        if (!fromId || !toMessage?.id) continue;
        const index = existing.findIndex((item) => item.id === fromId);
        if (index >= 0) {
          existing[index] = toMessage;
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

    function isTerminalStatus(status) {
      return ["completed", "failed", "blocked", "cancelled", "not_found"].includes(String(status || ""));
    }

    function isLiveQueueState(status) {
      return ["queued", "planning", "running", "verifying"].includes(String(status || ""));
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function isHelperTask(task) {
      const role = String(task?.role || "");
      return role.includes("helper");
    }

    function classifyPhase(item) {
      const actionClass = String(item?.action_class || "").toLowerCase();
      const role = String(item?.role || "").toLowerCase();
      if (isHelperTask(item)) return "Internal Helpers";
      if (actionClass === "inspect" || role.includes("inspector")) return "Inspect";
      if (actionClass === "verify" || role.includes("verifier")) return "Verify";
      if (role.includes("orchestrator") || role.includes("supervisor")) return "Run";
      return "Edit";
    }

    function dedupeTasks(tasks) {
      const merged = new Map();
      for (const task of Array.isArray(tasks) ? tasks : []) {
        const taskId = String(task?.task_id || "");
        if (!taskId) continue;
        const current = merged.get(taskId) || {};
        const children = [...new Set([...(current.child_task_ids || []), ...(task.child_task_ids || [])])];
        const allowedActions = [...new Set([...(current.allowed_actions || []), ...(task.allowed_actions || [])])];
        merged.set(taskId, {
          ...current,
          ...task,
          child_task_ids: children,
          allowed_actions: allowedActions,
          retry_count: Math.max(Number(current.retry_count || 0), Number(task.retry_count || 0)) || undefined,
        });
      }
      return Array.from(merged.values());
    }

    function groupByPhase(items, classifier) {
      const phases = new Map();
      for (const item of items) {
        const phase = classifier(item);
        if (!phases.has(phase)) phases.set(phase, []);
        phases.get(phase).push(item);
      }
      return phases;
    }

    function renderTaskCards(taskList, {includeHelpers = false} = {}) {
      return taskList
        .filter((task) => includeHelpers || !isHelperTask(task))
        .map((task, index) => {
        const taskId = task?.task_id || `task-${index + 1}`;
        const goal = task?.goal || taskId;
        const status = task?.status || "planned";
        const bits = [];
        if (task?.role) bits.push(`<span class="task-chip">role: ${escapeHtml(task.role)}</span>`);
        if (task?.agent_type) bits.push(`<span class="task-chip">agent: ${escapeHtml(task.agent_type)}</span>`);
        if (Array.isArray(task?.allowed_actions) && task.allowed_actions.length) {
          bits.push(`<span class="task-chip">actions: ${escapeHtml(task.allowed_actions.join(", "))}</span>`);
        }
        if (Array.isArray(task?.depends_on) && task.depends_on.length) {
          bits.push(`<span class="task-chip">deps: ${escapeHtml(task.depends_on.join(", "))}</span>`);
        }
        if (Array.isArray(task?.inputs_from) && task.inputs_from.length) {
          bits.push(`<span class="task-chip">inputs: ${escapeHtml(task.inputs_from.join(", "))}</span>`);
        }
        if (task?.parent_task_id) {
          bits.push(`<span class="task-chip">parent: ${escapeHtml(task.parent_task_id)}</span>`);
        }
        if (Array.isArray(task?.child_task_ids) && task.child_task_ids.length) {
          bits.push(`<span class="task-chip">children: ${escapeHtml(task.child_task_ids.join(", "))}</span>`);
        }
        if (task?.retry_count) {
          bits.push(`<span class="task-chip">retries: ${escapeHtml(task.retry_count)}</span>`);
        }
        return `
          <div class="task-card">
            <div class="task-card-top">
              <div class="task-card-id">${escapeHtml(taskId)}</div>
              <div class="task-card-status">${escapeHtml(status)}</div>
            </div>
            <div class="task-card-goal">${escapeHtml(goal)}</div>
            ${bits.length ? `<div class="task-card-meta">${bits.join("")}</div>` : ""}
          </div>
        `;
      }).join("");
    }

    function renderQueueEvents(events) {
      const eventList = Array.isArray(events) ? events.slice(-8).reverse() : [];
      if (!eventList.length) return "";
      return `<div class="task-block"><div class="task-block-title">Queue Activity</div><div class="task-grid">${
        eventList.map((event, index) => `
          <div class="task-card">
            <div class="task-card-top">
              <div class="task-card-id">${escapeHtml(event?.label || event?.event || `event-${index + 1}`)}</div>
              <div class="task-card-status">${escapeHtml(String(event?.timestamp || "").split("T")[1] || "")}</div>
            </div>
            <div class="task-card-goal">${escapeHtml(event?.payload?.instruction || event?.payload?.status || "Queue update")}</div>
          </div>
        `).join("")
      }</div></div>`;
    }

    function renderStepCards(stepList) {
      return stepList.map((step, index) => {
        const taskId = step?.id || `step-${index + 1}`;
        const goal = step?.instruction || taskId;
        const status = step?.status || step?.edit_mode || "planned";
        const bits = [];
        if (step?.action_class) bits.push(`<span class="task-chip">class: ${escapeHtml(step.action_class)}</span>`);
        if (step?.edit_mode) bits.push(`<span class="task-chip">mode: ${escapeHtml(step.edit_mode)}</span>`);
        if (step?.target_path) bits.push(`<span class="task-chip">target: ${escapeHtml(String(step.target_path).split("/").pop())}</span>`);
        if (Array.isArray(step?.depends_on) && step.depends_on.length) {
          bits.push(`<span class="task-chip">deps: ${escapeHtml(step.depends_on.join(", "))}</span>`);
        }
        if (Array.isArray(step?.inputs_from) && step.inputs_from.length) {
          bits.push(`<span class="task-chip">inputs: ${escapeHtml(step.inputs_from.join(", "))}</span>`);
        }
        if (step?.retry_count) {
          bits.push(`<span class="task-chip">retries: ${escapeHtml(step.retry_count)}</span>`);
        }
        if (step?.no_op) {
          bits.push(`<span class="task-chip">no-op</span>`);
        }
        return `
          <div class="task-card">
            <div class="task-card-top">
              <div class="task-card-id">${escapeHtml(taskId)}</div>
              <div class="task-card-status">${escapeHtml(status)}</div>
            </div>
            <div class="task-card-goal">${escapeHtml(goal)}</div>
            ${bits.length ? `<div class="task-card-meta">${bits.join("")}</div>` : ""}
          </div>
        `;
      }).join("");
    }

    function renderPhaseSections(title, items, renderer, {includeHelpers = false} = {}) {
      const grouped = groupByPhase(items, classifyPhase);
      const order = ["Run", "Inspect", "Edit", "Verify", "Internal Helpers"];
      const sections = [];
      for (const phase of order) {
        const phaseItems = grouped.get(phase) || [];
        const visibleItems = includeHelpers ? phaseItems : phaseItems.filter((item) => !isHelperTask(item));
        if (!visibleItems.length) continue;
        const cards = renderer(visibleItems, {includeHelpers});
        if (!cards) continue;
        sections.push(`<div class="task-block"><div class="task-block-title">${escapeHtml(phase)}</div><div class="task-grid">${cards}</div></div>`);
      }
      if (!sections.length) return "";
      return `<div class="task-block"><div class="task-block-title">${escapeHtml(title)}</div>${sections.join("")}</div>`;
    }

    function renderTaskBlock(tasks, steps) {
      const taskList = dedupeTasks(tasks);
      const stepList = Array.isArray(steps) ? steps : [];
      if (!taskList.length && !stepList.length) return "";
      const stepIds = new Set(stepList.map((step) => String(step?.id || "")).filter(Boolean));
      const logicalTasks = taskList.filter((task) => {
        const taskId = String(task?.task_id || "");
        if (!taskId) return false;
        return !stepIds.has(taskId);
      });
      const visibleHelperCount = taskList.filter((task) => isHelperTask(task)).length;
      const helperToggle = visibleHelperCount
        ? `<div class="actions" style="margin-top:0;"><button class="secondary" id="toggle_helpers_button">${showInternalHelpers ? "Hide internal helpers" : "Show internal helpers"}</button></div>`
        : "";
      const taskSections = renderPhaseSections("Run", logicalTasks, renderTaskCards, {includeHelpers: showInternalHelpers});
      const stepSections = renderPhaseSections("Lifecycle Steps", stepList, renderStepCards, {includeHelpers: true});
      return `
        ${helperToggle}
        ${taskSections}
        ${stepSections}
      `;
    }

    function renderTasksPanel(resultData = null, queueData = null) {
      const queue = queueData?.queue || queueData || {};
      const queueState = queue?.state || queueData?.status || resultData?.queue?.state || resultData?.status || "idle";
      const preferQueue = isLiveQueueState(queueState);
      const tasks = preferQueue
        ? (Array.isArray(queueData?.tasks) ? queueData.tasks : [])
        : (Array.isArray(resultData?.tasks) ? resultData.tasks : (Array.isArray(queueData?.tasks) ? queueData.tasks : []));
      const steps = preferQueue
        ? (Array.isArray(queueData?.steps) ? queueData.steps : [])
        : (Array.isArray(resultData?.steps) ? resultData.steps : (Array.isArray(queueData?.steps) ? queueData.steps : []));
      const queueEvents = Array.isArray(queue.task_events) ? queue.task_events : [];
      const activeCount = tasks.length || steps.length;
      tasksTitleEl.textContent = activeCount ? "Task graph loaded." : "No active task graph.";
      tasksSubtitleEl.textContent = activeCount
        ? "Inspect the canonical run lifecycle, logical steps, and internal helpers only when needed."
        : "Run a prompt to inspect lifecycle steps, dependencies, recovery, and internal helper work.";
      tasksPillEl.textContent = textOrDash(queueState);
      tasksPillEl.className = `pill ${["completed", "verified", "edited", "observed"].includes(queueState) ? "good" : (["failed", "blocked", "cancelled"].includes(queueState) ? "bad" : "neutral")}`;
      tasksSummaryEl.textContent = tasks.length || steps.length ? `${tasks.length} task${tasks.length === 1 ? "" : "s"} · ${steps.length} step${steps.length === 1 ? "" : "s"}` : "—";
      tasksQueueSummaryEl.textContent = queueEvents.length ? `${queueEvents.length} event${queueEvents.length === 1 ? "" : "s"}` : "—";
      const taskMarkup = renderTaskBlock(tasks, steps);
      tasksBlockEl.className = taskMarkup ? "" : "task-panel-empty";
      tasksBlockEl.innerHTML = taskMarkup || "No task data yet.";
      const toggleButton = document.getElementById("toggle_helpers_button");
      if (toggleButton) {
        toggleButton.addEventListener("click", () => {
          showInternalHelpers = !showInternalHelpers;
          renderTasksPanel(resultData, queueData);
        });
      }
      const queueMarkup = renderQueueEvents(queueEvents);
      tasksQueueBlockEl.className = queueMarkup ? "" : "task-panel-empty";
      tasksQueueBlockEl.innerHTML = queueMarkup || "No live queue activity.";
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
        ? (data?.index_state?.stale ? "Sync after more edits." : "No action needed.")
        : (!data?.index_state?.has_index || !data?.live_graph_state?.populated ? "Sync the live graph." : "Review graph status.");
      graphSummaryEl.innerHTML = [
        ["State", data?.ready ? "Ready" : "Needs attention"],
        ["Counts", `${textOrDash(data?.live_graph_state?.node_count)} nodes / ${textOrDash(data?.live_graph_state?.relationship_count)} rel`],
        ["Next", nextStep],
      ].map(([label, value]) => `
        <div class="summary-chip">
          <strong>${label}</strong>
          <span>${value}</span>
        </div>
      `).join("");

      renderDefinitionList(graphDetailsEl, [
        ["Available", yesNo(data?.available)],
        ["Reason", textOrDash(data?.reason)],
        ["Has Index", yesNo(data?.index_state?.has_index)],
        ["Stale", yesNo(data?.index_state?.stale)],
        ["Populated", yesNo(data?.live_graph_state?.populated)],
        ["Nodes", textOrDash(data?.live_graph_state?.node_count)],
        ["Relationships", textOrDash(data?.live_graph_state?.relationship_count)],
        ["Files", indexFiles.length ? indexFiles.join(", ") : "—"],
      ]);
    }

    function summarizeResult(data) {
      const status = data?.status || "unknown";
      const error = data?.error;
      const noOp = Boolean(data?.execution?.no_op ?? data?.no_op);
      if (error) {
        return {title: `Run finished with ${status}.`, subtitle: error, pill: status, tone: "bad"};
      }
      if (noOp && (status === "edited" || status === "observed" || status === "no_op")) {
        return {
          title: "Run finished without changing files.",
          subtitle: "The latest instruction completed, but it did not apply a material file change.",
          pill: "no_op",
          tone: "neutral"
        };
      }
      if (status === "verified" || status === "edited") {
        return {
          title: `Run finished with ${status}.`,
          subtitle: "The latest instruction completed without a recorded error.",
          pill: status,
          tone: "good"
        };
      }
      if (status === "observed") {
        return {
          title: "Observation completed.",
          subtitle: "The latest tool action completed without a recorded error.",
          pill: "observed",
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

    function summarizeToolOutput(toolOutput) {
      if (!toolOutput || typeof toolOutput !== "object") return [];
      const tool = toolOutput.tool;
      if (tool === "run_command" || tool === "verify_command" || tool === "run_tests") {
        const lines = [
          toolOutput.command ? `Command: ${toolOutput.command}` : null,
          Number.isInteger(toolOutput.returncode) ? `Exit: ${toolOutput.returncode}` : null,
        ];
        if (toolOutput.stdout) lines.push(`Stdout: ${String(toolOutput.stdout).trim().slice(0, 240)}`);
        if (toolOutput.stderr) lines.push(`Stderr: ${String(toolOutput.stderr).trim().slice(0, 240)}`);
        return lines.filter(Boolean);
      }
      if (tool === "read_file") {
        return [
          toolOutput.target_path ? `Read: ${toolOutput.target_path}` : null,
          toolOutput.content ? `Content: ${String(toolOutput.content).slice(0, 240)}` : null,
        ].filter(Boolean);
      }
      if (tool === "list_files") {
        const count = Array.isArray(toolOutput.files) ? toolOutput.files.length : 0;
        return [
          toolOutput.target_path ? `Listed: ${toolOutput.target_path}` : null,
          count ? `Files: ${count}` : null,
        ].filter(Boolean);
      }
      if (tool === "search_files") {
        const count = Array.isArray(toolOutput.matches) ? toolOutput.matches.length : 0;
        return [
          toolOutput.pattern ? `Pattern: ${toolOutput.pattern}` : null,
          count ? `Matches: ${count}` : "Matches: 0",
        ].filter(Boolean);
      }
      return [];
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
      const toolOutput = data?.execution?.tool_output || data?.tool_output;
      const verificationResults = data?.execution?.verification_results || data?.verification_results || [];
      const verificationRetryCount = data?.execution?.verification_retry_count ?? data?.verification_retry_count;
      const revertedFiles = data?.execution?.reverted_files || data?.reverted_files || [];
      const revertCount = data?.execution?.revert_count ?? data?.revert_count;
      const taskBlockHtml = renderTaskBlock(data?.tasks || [], data?.steps || []);
      const previewSuffix = data?.execution?.file_preview_truncated || data?.file_preview_truncated ? "..." : "";
      const changedSummary = changedFiles.length
        ? (changedFiles.length === 1 ? `Changed: ${changedFiles[0]}` : `Changed ${changedFiles.length} files`)
        : null;
      const toolSummary = summarizeToolOutput(toolOutput);
      const verificationSummary = [];
      const verifyBlockHtml = (() => {
        if (!Array.isArray(verificationResults) || !verificationResults.length) return "";
        const allPassed = verificationResults.every((r) => r?.returncode === 0);
        const lines = verificationResults.flatMap((result) => {
          if (!result || typeof result !== "object") return [];
          const out = [];
          if (result.command) out.push(`$ ${result.command}`);
          if (result.stdout) out.push(String(result.stdout).trim().slice(0, 300));
          if (result.stderr) out.push(String(result.stderr).trim().slice(0, 300));
          if (Number.isInteger(result.returncode)) out.push(`exit ${result.returncode}`);
          return out;
        });
        if (Number.isInteger(verificationRetryCount) && verificationRetryCount > 0) {
          lines.push(`Retries: ${verificationRetryCount}`);
        }
        if (revertedFiles.length) {
          lines.push(revertedFiles.length === 1 ? `Reverted: ${revertedFiles[0]}` : `Reverted: ${revertedFiles.length} files`);
        } else if (Number.isInteger(revertCount) && revertCount > 0) {
          lines.push(`Reverted: ${revertCount} files`);
        }
        const tone = allPassed ? "pass" : "fail";
        const label = allPassed ? "Verification passed" : "Verification failed";
        return `<div class="verify-block ${tone}"><div class="verify-block-label">${label}</div>${lines.map((l) => `<div class="verify-line">${escapeHtml(l)}</div>`).join("")}</div>`;
      })();
      const sessionId = getActiveSessionId(data) || "session";
      const resultKey = uiState.activeJobId || uiState.lastCompletedJobId || data?.job_id || data?.trace_path || data?.content_hash || `${sessionId}-${resultValue(data, ["execution", "status"], data?.status)}`;
      const runKeys = [uiState.activeJobId, uiState.lastCompletedJobId, data?.job_id, data?.session_id, sessionId, data?.trace_path];
      const rememberedIds = lookupRunMessageIds(runKeys);
      const userMessageId = rememberedIds?.userId || `user-${resultKey}`;
      const assistantMessageId = rememberedIds?.assistantId || `assistant-${resultKey}`;
      const submittedInstruction = lookupSubmittedInstruction([
        uiState.activeJobId,
        uiState.lastCompletedJobId,
        data?.job_id,
        data?.trace_path,
        data?.session_id,
        sessionId,
      ]);
      const existingUserMessage = findActivityMessage(userMessageId);
      const renderedInstruction = (
        existingUserMessage?.text ||
        submittedInstruction ||
        (requestInstruction === "—" ? "Submitted instruction" : requestInstruction)
      );
      rememberRunMessageIds(runKeys, {userId: userMessageId, assistantId: assistantMessageId});
      rememberSubmittedInstruction(
        [uiState.activeJobId, uiState.lastCompletedJobId, data?.job_id, data?.trace_path, data?.session_id, sessionId],
        renderedInstruction,
      );
      const gateBlockHtml = (() => {
        if (!humanGate?.prompt) return "";
        const action = humanGate?.action || "";
        const isWideImpact = action === "approve_wide_impact";
        const detailCount = humanGate?.details?.count;
        const detailText = detailCount ? ` (${detailCount} files affected)` : "";
        const approveBtn = isWideImpact
          ? `<button class="gate-approve-btn" onclick="approveWideImpact()">Approve${detailText}</button>`
          : "";
        return `<div class="gate-block"><div class="gate-block-label">Action required · ${escapeHtml(action)}</div><div class="gate-block-text">${escapeHtml(humanGate.prompt)}</div>${approveBtn}</div>`;
      })();
      if (humanGate?.action === "approve_wide_impact") {
        _pendingApproveInstruction = renderedInstruction;
      }
      const detailsHtml = [taskBlockHtml, verifyBlockHtml, gateBlockHtml].filter(Boolean).join("") || undefined;
      appendActivityMessages([
        {
          id: userMessageId,
          role: "user",
          label: "You",
          text: renderedInstruction,
          meta: [],
        },
        {
          id: assistantMessageId,
          role: "assistant",
          label: "Shipyard",
          text: data?.status === "idle" ? "Write an instruction and Shipyard will handle it here." : summary.subtitle,
          detailsHtml,
          badge: summary.pill,
          badgeTone: summary.tone,
          meta: [
            targetPath !== "—" ? `Target: ${targetPath}` : null,
            changedSummary,
            preview ? `Preview: ${preview}${previewSuffix}` : null,
            ...toolSummary,
            ...verificationSummary,
            contentHash ? `Hash: ${contentHash}` : null,
          ].filter((line) => typeof line === "string"),
        },
      ]);
    }

    function renderPendingState(pending) {
      if (!pending?.kind) return;
      const isQueued = pending.kind === "queued";
      appendActivityMessages([
        {
          id: `pending-${uiState.activeSessionId || "session"}`,
          role: "assistant",
          label: "Shipyard",
          text: pending.title || "Working.",
          badge: isQueued ? null : "Working",
          badgeTone: "neutral",
          meta: [!isQueued ? (pending.subtitle || "This page was refreshed while Shipyard was working.") : null].filter(Boolean),
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

    function syncWorkspaceContext(workspacePath) {
      const context = parseContext();
      if (workspacePath) {
        context.workspace_path = workspacePath;
      } else {
        delete context.workspace_path;
      }
      if (!document.getElementById("target_path").value.trim() && context.file_hint) {
        delete context.file_hint;
      }
      document.getElementById("context_json").value = Object.keys(context).length ? pretty(context) : "";
      return context;
    }

    function renderWorkspaceFolders(items, selectedPath) {
      const inputEl = document.getElementById("workspace_select");
      const datalistEl = document.getElementById("workspace_options");
      const options = [{path: "", label: "Managed workspace (default)"}].concat(Array.isArray(items) ? items : []);
      datalistEl.innerHTML = options.map((item) => {
        const value = escapeHtml(item.resolved_path || item.path || "");
        const label = escapeHtml(item.label || item.path || "");
        return `<option value="${value}">${label}</option>`;
      }).join("");
      if (selectedPath) {
        inputEl.value = selectedPath;
      }
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
        const workspacePath = data?.request?.context?.workspace_path || "";
        document.getElementById("workspace_select").value = workspacePath;
        syncWorkspaceContext(workspacePath || null);
        renderResultDetails(data);
        resultEl.textContent = pretty(data);
        saveState({lastResult: data, activeSessionId: sessionId});
        renderTasksPanel(data, data?.queue || null);
        loadWorkspaceStatus(sessionId);
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
        try {
          const queueData = await loadQueueStatus(sessionId);
          if (uiState.activeJobId) {
            const terminal = await hydrateActiveRun(uiState.activeJobId, sessionId);
            if (terminal) {
              saveState({pending: null, pendingInstruction: null});
              stopSessionPolling();
              Promise.allSettled([loadGraphStatus(), loadWorkspaceStatus(sessionId), loadSessions()]);
              return;
            }
          }
          const recovered = !uiState.activeJobId ? await hydrateSessionState(sessionId, {quiet: true}) : false;
          if (recovered && uiState.pending) {
            saveState({pending: null, pendingInstruction: null});
            stopSessionPolling();
            Promise.allSettled([loadGraphStatus(), loadWorkspaceStatus(sessionId), loadSessions()]);
            return;
          }
          if (queueData?.session && isTerminalStatus(queueData.session.status) && !uiState.activeJobId) {
            saveState({pending: null, pendingInstruction: null});
            stopSessionPolling();
            clearTransientRunMessages();
            renderResultDetails(queueData.session);
            resultEl.textContent = pretty(queueData.session);
            renderTasksPanel(queueData.session, queueData.session?.queue || null);
            saveState({lastResult: queueData.session, activeSessionId: sessionId});
            Promise.allSettled([loadHistory(sessionId), loadGraphStatus(), loadWorkspaceStatus(sessionId), loadSessions()]);
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
        } catch (error) {
          console.error("Shipyard polling failed", error);
          stopSessionPolling();
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
      if (!job) {
        // Job vanished (404) — clear active state and stop polling
        saveState({ activeJobId: null, pending: null, pendingInstruction: null });
        stopSessionPolling();
        return true;
      }

      const terminal = isTerminalStatus(job.status);
      if (terminal) {
        const resolvedJobId = job?.queue?.job_id || job?.job_id || null;
        const resolvedSessionId = sessionId || job.session_id || uiState.activeSessionId || null;
        saveState({
          activeJobId: null,
          lastCompletedJobId: resolvedJobId,
          pending: null,
          pendingInstruction: null,
          activeSessionId: resolvedSessionId,
          lastResult: job,
        });
        renderResultDetails(job);
        resultEl.textContent = pretty(job);
        renderTasksPanel(job, job?.queue || null);
        if (resolvedSessionId) loadHistory(resolvedSessionId);
        return true;
      }

      renderQueuedRun(job, uiState.pendingInstruction || uiState.form?.instruction || "");
      renderTasksPanel(null, job);

      saveState({
        activeJobId: job?.queue?.job_id || job?.job_id,
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
        renderTasksPanel(data, data?.queue || null);
        saveState({lastResult: data, pending: null, form: currentFormState()});
        await loadGraphStatus();
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        resultEl.textContent = pretty({error: String(error)});
        renderTasksPanel({status: "error", error: String(error)}, null);
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
        renderTasksPanel({status: "runtime_cleaned"}, null);
        saveState({lastResult: {status: "runtime_cleaned", ...data}, pending: null, form: currentFormState()});
        await loadSessions();
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        resultEl.textContent = pretty({error: String(error)});
        renderTasksPanel({status: "error", error: String(error)}, null);
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

    async function cancelRun(jobId) {
      if (!jobId) return;
      try {
        const data = await fetchJson("/queue/cancel", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({job_id: jobId})
        });
        saveState({lastResult: data, activeJobId: null, pending: null, pendingInstruction: null});
        resultEl.textContent = pretty(data);
        renderResultDetails(data);
        renderTasksPanel(data, data?.queue || null);
        stopSessionPolling();
        await loadQueueStatus(getActiveSessionId(data) || uiState.activeSessionId);
      } catch (error) {
        console.error("Cancel failed", error);
      }
    }

    async function loadSessions() {
      const data = await fetchJson("/sessions");
      sessionListEl.innerHTML = "";
      for (const session of data.sessions) {
        const item = document.createElement("li");
        const shortId = session.session_id.length > 16 ? "…" + session.session_id.slice(-12) : session.session_id;
        const excerpt = session.instruction ? session.instruction.slice(0, 60) + (session.instruction.length > 60 ? "…" : "") : shortId;
        const tone = ["edited", "verified", "observed"].includes(session.status) ? "good" : (["failed", "blocked"].includes(session.status) ? "bad" : "neutral");
        item.innerHTML = `
          <button class="session-item" data-sid="${session.session_id}">
            <span class="session-excerpt">${excerpt}</span>
            <span class="pill ${tone}" style="font-size:0.74rem; padding:2px 8px; flex-shrink:0;">${session.status || "—"}</span>
          </button>`;
        item.querySelector("button").addEventListener("click", () => {
          openPanel("sessions");
          loadHistory(session.session_id);
        });
        sessionListEl.appendChild(item);
      }
      if (!data.sessions.length) {
        const empty = document.createElement("li");
        empty.className = "muted";
        empty.style.fontSize = "0.84rem";
        empty.style.padding = "8px 2px";
        empty.textContent = "No sessions yet.";
        sessionListEl.appendChild(empty);
      }
    }

    async function loadWorkspaceStatus(sessionId = null) {
      const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
      const data = await fetchJson(`/workspace/status${query}`);
      const selection = data?.selection || {};
      const label = selection?.workspace_label || "default";
      if (selection?.mode === "repo_folder") {
        workspaceHintEl.textContent = `Testing mode is attached to repo folder ${label}.`;
        workspaceDetailsEl.textContent = `This session is writing into repo folder ${label}. Relative targets resolve inside that folder.`;
      } else {
        workspaceHintEl.textContent = "Testing mode is attached to the managed workspace until you pick a repo folder.";
        workspaceDetailsEl.textContent = "This session is using Shipyard's managed workspace. Pick a repo folder if you want edits attached to a real directory in this repo.";
      }
      renderWorkspaceFolders(data?.folders || [], selection?.workspace_path || "");
      saveState({workspaceStatus: data});
    }

    async function createWorkspace() {
      try {
        const sessionId = uiState.activeSessionId || ensureSessionId();
        const data = await fetchJson("/workspace/select", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({session_id: sessionId, workspace_path: null})
        });
        document.getElementById("session_id").value = sessionId;
        document.getElementById("workspace_select").value = "";
        syncWorkspaceContext(null);
        renderResultDetails({
          status: "workspace_ready",
          workspace: data,
          proposal_summary: {target_path_source: "workspace_button"}
        });
        resultEl.textContent = pretty({
          workspace_selected: data
        });
        saveState({
          lastResult: {
            status: "workspace_ready",
            workspace: data,
            proposal_summary: {target_path_source: "workspace_button"},
          },
          activeSessionId: sessionId,
          form: currentFormState(),
        });
        await loadWorkspaceStatus(sessionId);
        openPanel("details");
      } catch (error) {
        resultEl.textContent = pretty({error: String(error)});
      }
    }

    async function selectWorkspaceFolder() {
      try {
        const sessionId = uiState.activeSessionId || ensureSessionId();
        const selectedPath = document.getElementById("workspace_select").value || null;
        const data = await fetchJson("/workspace/select", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({session_id: sessionId, workspace_path: selectedPath})
        });
        document.getElementById("session_id").value = sessionId;
        syncWorkspaceContext(data?.workspace_path || null);
        renderResultDetails({
          status: "workspace_ready",
          workspace: data,
          proposal_summary: {target_path_source: "workspace_select"}
        });
        resultEl.textContent = pretty({workspace_selected: data});
        saveState({
          lastResult: {
            status: "workspace_ready",
            workspace: data,
            proposal_summary: {target_path_source: "workspace_select"},
          },
          activeSessionId: sessionId,
          form: currentFormState(),
        });
        await loadWorkspaceStatus(sessionId);
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
        const explicitScaffoldPrompt = isExplicitScaffoldPrompt(instructionText);
        let targetPath = document.getElementById("target_path").value.trim();
        if (explicitScaffoldPrompt) {
          targetPath = "";
          document.getElementById("target_path").value = "";
          delete context.file_hint;
        }
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
        const selectedWorkspace = document.getElementById("workspace_select").value.trim();
        if (selectedWorkspace) {
          context.workspace_path = selectedWorkspace;
        } else {
          delete context.workspace_path;
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
          verification_commands: verificationCommands(),
          wide_impact_approved: false,
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
        rememberSubmittedInstruction([sessionId, `pending-${sessionId}`], payload.instruction);
        rememberRunMessageIds(
          [sessionId, `pending-${sessionId}`],
          {userId: `user-pending-${sessionId}`, assistantId: `assistant-pending-${sessionId}`},
        );
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
        if (jobId) {
          rememberSubmittedInstruction([jobId, resolvedSessionId], payload.instruction);
          rememberRunMessageIds(
            [jobId, resolvedSessionId],
            {userId: `user-${jobId}`, assistantId: `assistant-${jobId}`},
          );
        } else {
          rememberSubmittedInstruction([resolvedSessionId], payload.instruction);
        }
        document.getElementById("session_id").value = resolvedSessionId;
        if (jobId) {
          replaceActivityMessageIds([
            {
              fromId: `user-pending-${sessionId}`,
              toMessage: {
                id: `user-${jobId}`,
                role: "user",
                label: "You",
                text: payload.instruction,
                meta: [],
              },
            },
            {
              fromId: `assistant-pending-${sessionId}`,
              toMessage: {
                id: `assistant-${jobId}`,
                role: "assistant",
                label: "Shipyard",
                text: "Queued.",
                badge: "Queued",
                badgeTone: "neutral",
                meta: [],
              },
            },
          ]);
          const queueJob = data?.queue_job || {
            job_id: jobId,
            session_id: resolvedSessionId,
            status: data?.status || "queued",
            current_task: data?.current_task || "Waiting",
          };
          renderQueuedRun(queueJob, payload.instruction);
          renderTasksPanel(null, queueJob);
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
        instructionEl.value = "";
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
        renderTasksPanel({status: "error", error: String(error)}, null);
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
    document.getElementById("workspace_select_button").addEventListener("click", selectWorkspaceFolder);
    document.getElementById("workspace_refresh_button").addEventListener("click", () => {
      loadWorkspaceStatus(getActiveSessionId(uiState.lastResult));
    });
    document.getElementById("clear_button").addEventListener("click", clearActivity);
    document.getElementById("reindex_button").addEventListener("click", syncGraph);
    document.getElementById("cleanup_button").addEventListener("click", cleanRuntime);
    document.getElementById("sessions_button").addEventListener("click", () => {
      openPanel("sessions");
      loadSessions();
    });
    document.getElementById("debug_refresh_button").addEventListener("click", () => {
      loadGraphStatus();
      const sid = getActiveSessionId();
      if (sid) hydrateSessionState(sid, {quiet: true});
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
      renderTasksPanel(uiState.lastResult, uiState.lastResult?.queue || null);
    } else {
      resultEl.textContent = pretty({});
      renderTasksPanel(null, null);
    }
    if (uiState.activeTab) {
      selectTab(uiState.activeTab);
    }
    async function loadGitBadge() {
      const badgeEl = document.getElementById("git_badge");
      if (!badgeEl) return;
      try {
        const data = await fetchJson("/git/status");
        const branch = data?.branch || "";
        if (!branch) { badgeEl.style.display = "none"; return; }
        const isClean = data?.is_clean !== false;
        const changedCount = Array.isArray(data?.status_lines) ? data.status_lines.length : 0;
        badgeEl.textContent = isClean ? `⎇ ${branch}` : `⎇ ${branch} · ${changedCount} changed`;
        badgeEl.className = `git-badge ${isClean ? "clean" : "dirty"}`;
        badgeEl.style.display = "";
      } catch {
        badgeEl.style.display = "none";
      }
    }

    let _pendingApproveInstruction = null;

    async function approveWideImpact() {
      const instruction = _pendingApproveInstruction || uiState.form?.instruction || "";
      if (!instruction) return;
      const context = parseContext();
      context.testing_mode = true;
      const sessionId = document.getElementById("session_id").value.trim() || uiState.activeSessionId || generateSessionId();
      document.getElementById("session_id").value = sessionId;
      const payload = {
        session_id: sessionId,
        instruction,
        target_path: document.getElementById("target_path").value.trim() || null,
        edit_mode: document.getElementById("edit_mode").value || null,
        proposal_mode: document.getElementById("proposal_mode").value || null,
        context,
        verification_commands: verificationCommands(),
        wide_impact_approved: true,
      };
      _pendingApproveInstruction = null;
      saveState({
        pending: {kind: "queued", title: "Run queued (approved)...", subtitle: "Wide-impact operation approved."},
        form: currentFormState(),
        activeSessionId: sessionId,
        pendingInstruction: payload.instruction,
      });
      try {
        const data = await fetchJson("/queue/instruct", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const resolvedSessionId = data?.session_id || sessionId;
        const jobId = data?.job_id || data?.queue_job?.job_id || null;
        saveState({lastResult: data, activeSessionId: resolvedSessionId, activeJobId: jobId, lastCompletedJobId: null});
        resultEl.textContent = pretty(data);
        if (jobId) {
          startSessionPolling(resolvedSessionId);
          await loadQueueStatus(resolvedSessionId);
        }
      } catch (error) {
        renderResultDetails({status: "error", error: String(error)});
        saveState({pending: null});
        stopSessionPolling();
      }
    }

    async function initializeWorkbench() {
      loadPlannerStatus();
      loadGraphStatus();
      loadSessions();
      loadGitBadge();
      const restoredSessionId = uiState.activeSessionId || getActiveSessionId(uiState.lastResult);
      loadWorkspaceStatus(restoredSessionId);
      const restoredJobId = uiState.activeJobId || null;
      const queueData = await loadQueueStatus(restoredSessionId);
      const hasLiveQueuedOrRunning = Boolean(
        (queueData?.active && isLiveQueueState(queueData.active?.queue?.state || queueData.active?.status)) ||
        ((queueData?.queued || []).some((job) => isLiveQueueState(job?.queue?.state || job?.status))) ||
        (queueData?.session && isLiveQueueState(queueData.session.status))
      );
      if (restoredJobId && !hasLiveQueuedOrRunning) {
        saveState({
          activeJobId: null,
          pending: null,
          pendingInstruction: null,
        });
        clearTransientRunMessages();
      }
      if (uiState.pending && hasLiveQueuedOrRunning) {
        renderPendingState(uiState.pending);
      } else if (uiState.pending) {
        clearPendingState();
      }
      if (!hasLiveQueuedOrRunning && !restoredJobId) {
        clearTransientRunMessages();
      }
      if (restoredJobId && hasLiveQueuedOrRunning) {
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


@app.post("/plan")
def plan(request: InstructionRequest) -> dict[str, Any]:
    """Generate a rebuild plan without executing. The agent reads, thinks, and proposes."""
    from .plan_mode import generate_plan
    from .context_explorer import build_broad_context
    state = _normalize_payload(request.model_dump())
    if not state.get("session_id"):
        from .main import _ensure_session_id
        state["session_id"] = _ensure_session_id(None)
    state["broad_context"] = build_broad_context(state.get("session_id"), state.get("instruction", ""))
    reference_path = state.get("context", {}).get("reference_path")
    return generate_plan(state, reference_path=reference_path)


@app.post("/queue/instruct")
def queue_instruct(request: InstructionRequest) -> dict[str, Any]:
    state = _normalize_payload(request.model_dump())
    if not state.get("session_id"):
        from .main import _ensure_session_id

        state["session_id"] = _ensure_session_id(None)
    _write_request_receipt(state)
    return run_queue.enqueue(state)


def _write_request_receipt(state: dict[str, Any]) -> None:
    session_id = str(state.get("session_id") or "unknown")
    if not session_id.startswith("web-"):
        return
    logs_dir = ensure_dir(LOGS_ROOT)
    payload = {
        "session_id": session_id,
        "instruction": str(state.get("instruction") or ""),
        "status": "accepted",
        "accepted_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = logs_dir / f"latest-{session_id}-request.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@app.get("/queue/status")
def queue_status(session_id: str | None = None) -> dict[str, Any]:
    return run_queue.get_status(session_id)


@app.get("/queue/job/{job_id}")
def queue_job(job_id: str) -> dict[str, Any]:
    job = run_queue.get_job(job_id)
    if job is None:
        # Return a terminal status instead of 404 so the frontend stops polling
        return {"job_id": job_id, "status": "not_found", "error": "Job expired or never existed."}
    return job


@app.post("/queue/cancel")
def queue_cancel(request: QueueCancelRequest) -> dict[str, Any]:
    job = run_queue.cancel(request.job_id)
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
def workspace_status(session_id: str | None = None) -> dict[str, Any]:
    return {
        **get_workspace_status(),
        "selection": get_session_workspace_selection(session_id),
        "folders": list_repo_workspace_folders(),
    }


@app.get("/workspace/folders")
def workspace_folders() -> dict[str, Any]:
    return {"folders": list_repo_workspace_folders()}


@app.post("/workspace/select")
def workspace_select(request: WorkspaceSelectRequest) -> dict[str, Any]:
    if request.workspace_path:
        normalized = normalize_repo_workspace_path(request.workspace_path)
        if normalized is None:
            raise HTTPException(status_code=400, detail="Workspace path must be an existing folder inside the current repo.")
    selection = set_session_workspace(request.session_id, request.workspace_path)
    return selection


@app.post("/workspace/temp")
def workspace_temp(request: WorkspaceCreateRequest) -> dict[str, str]:
    workspace = get_managed_workspace()
    if request.session_id:
        set_session_workspace(request.session_id, None)
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
