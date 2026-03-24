from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROTO_DIR = Path(".shipyard") / "code_graph"


def inspect_code_graph_status(workdir: str | None = None) -> dict[str, Any]:
    normalized_workdir = _normalize_workdir(workdir)
    if _truthy_env("SHIPYARD_ASSUME_CODE_GRAPH_READY"):
        return {
            "ready": True,
            "available": True,
            "source": "env_override",
            "reason": "SHIPYARD_ASSUME_CODE_GRAPH_READY is enabled.",
            "index_state": inspect_code_graph_artifacts(normalized_workdir),
        }

    executable = _find_cgr_executable()
    if executable is None:
        return {
            "ready": False,
            "available": False,
            "source": "cli_lookup",
            "reason": "Code-Graph-RAG CLI `cgr` was not found.",
            "index_state": inspect_code_graph_artifacts(normalized_workdir),
        }

    index_state = inspect_code_graph_artifacts(normalized_workdir)
    completed = subprocess.run(
        [executable, "stats"],
        cwd=normalized_workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if isinstance(part, str) and part.strip()
    ).strip()

    if completed.returncode == 0:
        return {
            "ready": True,
            "available": True,
            "source": "cgr_stats",
            "reason": "Code graph statistics are available.",
            "index_state": index_state,
            "details": combined_output,
        }

    return {
        "ready": False,
        "available": True,
        "source": "cgr_stats",
        "reason": _summarize_failure(combined_output),
        "index_state": index_state,
        "details": combined_output,
        "returncode": completed.returncode,
    }


def inspect_code_graph_artifacts(workdir: str | None = None) -> dict[str, Any]:
    root = _normalize_workdir(workdir)
    proto_dir = root / DEFAULT_PROTO_DIR
    file_paths = sorted(path for path in proto_dir.glob("*") if path.is_file())
    files = [str(path.relative_to(root)) for path in file_paths]
    latest_artifact_mtime = max((path.stat().st_mtime for path in file_paths), default=None)
    latest_source_mtime = _latest_source_mtime(root)
    stale = False
    if latest_artifact_mtime is not None and latest_source_mtime is not None:
        stale = latest_source_mtime > latest_artifact_mtime

    return {
        "proto_dir": str(proto_dir),
        "exists": proto_dir.exists(),
        "files": files,
        "has_index": any(path.endswith(".bin") or path.endswith(".pkl") for path in files),
        "latest_artifact_at": _isoformat_mtime(latest_artifact_mtime),
        "latest_source_at": _isoformat_mtime(latest_source_mtime),
        "stale": stale if file_paths else False,
    }


def index_code_graph(
    workdir: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    executable = _find_cgr_executable()
    root = _normalize_workdir(workdir)
    if executable is None:
        return {
            "ok": False,
            "reason": "Code-Graph-RAG CLI `cgr` was not found.",
            "index_state": inspect_code_graph_artifacts(str(root)),
        }

    proto_dir = Path(output_dir) if output_dir else root / DEFAULT_PROTO_DIR
    proto_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            executable,
            "index",
            "--repo-path",
            str(root),
            "--output-proto-dir",
            str(proto_dir),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if isinstance(part, str) and part.strip()
    ).strip()

    return {
        "ok": completed.returncode == 0,
        "reason": "Code graph index created." if completed.returncode == 0 else _summarize_failure(combined_output),
        "returncode": completed.returncode,
        "output": combined_output,
        "index_state": inspect_code_graph_artifacts(str(root)),
    }


def _find_cgr_executable() -> str | None:
    local_candidate = Path(sys.executable).resolve().parent / "cgr"
    if local_candidate.exists():
        return str(local_candidate)
    return shutil.which("cgr")


def _summarize_failure(output: str) -> str:
    lowered = output.lower()
    if "couldn't open socket" in lowered or "operation not permitted" in lowered:
        return "Memgraph is not reachable from the current environment."
    if "failed to get graph statistics" in lowered:
        return "Code graph statistics could not be loaded."
    if output:
        return output.splitlines()[-1].strip()
    return "Code-Graph-RAG readiness check failed."


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_workdir(workdir: str | None) -> Path:
    candidate = Path(workdir or ".").resolve()
    if candidate.is_file():
        return candidate.parent
    return candidate


def _latest_source_mtime(root: Path) -> float | None:
    latest: float | None = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_ignore_path(root, path):
            continue
        current = path.stat().st_mtime
        if latest is None or current > latest:
            latest = current
    return latest


def _should_ignore_path(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    first = relative.parts[0] if relative.parts else ""
    return first in {".git", ".venv", ".shipyard", "__pycache__"}


def _isoformat_mtime(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")
