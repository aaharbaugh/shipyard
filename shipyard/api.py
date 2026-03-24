from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import build_graph
from .main import _normalize_payload, run_once
from .session_store import SessionStore
from .tools.code_graph import index_code_graph, inspect_code_graph_status
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


app = FastAPI(title="Shipyard MVP API")
graph_app = build_graph()
session_store = SessionStore()
git_automation = GitAutomation()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
def list_sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": session_store.list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    state = session_store.load_latest_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return state


@app.post("/instruct")
def instruct(request: InstructionRequest) -> dict[str, Any]:
    state = _normalize_payload(request.model_dump())
    return run_once(graph_app, session_store, state)


@app.get("/graph/status")
def graph_status() -> dict[str, Any]:
    return inspect_code_graph_status()


@app.post("/graph/index")
def graph_index(request: GraphIndexRequest) -> dict[str, Any]:
    result = index_code_graph(request.workdir, request.output_dir)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


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
