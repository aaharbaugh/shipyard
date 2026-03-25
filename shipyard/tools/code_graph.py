from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..storage_paths import GRAPH_INDEX_ROOT
from ..workspaces import get_live_graph_runtime_root

DEFAULT_PROTO_DIR = GRAPH_INDEX_ROOT


def inspect_code_graph_status(workdir: str | None = None, include_debug: bool = False) -> dict[str, Any]:
    normalized_workdir = _normalize_workdir(workdir)
    if _truthy_env("SHIPYARD_ASSUME_CODE_GRAPH_READY"):
        return {
            "ready": True,
            "available": True,
            "source": "env_override",
            "reason": "SHIPYARD_ASSUME_CODE_GRAPH_READY is enabled.",
            "index_state": inspect_code_graph_artifacts(normalized_workdir),
            "live_graph_state": {
                "connected": True,
                "node_count": None,
                "relationship_count": None,
                "populated": True,
            },
        }

    executable = _find_cgr_executable()
    if executable is None:
        return {
            "ready": False,
            "available": False,
            "source": "cli_lookup",
            "reason": "Code-Graph-RAG CLI `cgr` was not found.",
            "index_state": inspect_code_graph_artifacts(normalized_workdir),
            "live_graph_state": {
                "connected": False,
                "node_count": None,
                "relationship_count": None,
                "populated": False,
            },
        }

    index_state = inspect_code_graph_artifacts(normalized_workdir)
    completed = subprocess.run(
        [executable, "stats"],
        cwd=_cgr_run_dir(),
        env=_cgr_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if isinstance(part, str) and part.strip()
    ).strip()

    live_graph_state = _parse_live_graph_state(combined_output, completed.returncode)

    if completed.returncode == 0 and live_graph_state["populated"]:
        result = {
            "ready": True,
            "available": True,
            "source": "cgr_stats",
            "reason": "Code graph statistics are available.",
            "index_state": index_state,
            "live_graph_state": live_graph_state,
        }
        if include_debug and combined_output:
            result["details"] = combined_output
        return result

    if completed.returncode == 0:
        result = {
            "ready": False,
            "available": True,
            "source": "cgr_stats",
            "reason": "Memgraph is reachable, but the live graph is still empty.",
            "index_state": index_state,
            "live_graph_state": live_graph_state,
        }
        if include_debug and combined_output:
            result["details"] = combined_output
        return result

    result = {
        "ready": False,
        "available": True,
        "source": "cgr_stats",
        "reason": _summarize_failure(combined_output),
        "index_state": index_state,
        "live_graph_state": live_graph_state,
        "returncode": completed.returncode,
    }
    if include_debug and combined_output:
        result["details"] = combined_output
    return result


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
    include_debug: bool = False,
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
        cwd=_cgr_run_dir(),
        env=_cgr_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if isinstance(part, str) and part.strip()
    ).strip()

    result = {
        "ok": completed.returncode == 0,
        "reason": "Code graph index created." if completed.returncode == 0 else _summarize_failure(combined_output),
        "returncode": completed.returncode,
        "index_state": inspect_code_graph_artifacts(str(root)),
    }
    if include_debug and combined_output:
        result["output"] = combined_output
    return result


def sync_live_code_graph(
    workdir: str | None = None,
    clean: bool = False,
    include_debug: bool = False,
) -> dict[str, Any]:
    executable = _find_cgr_executable()
    root = _normalize_workdir(workdir)
    if executable is None:
        return {
            "ok": False,
            "reason": "Code-Graph-RAG CLI `cgr` was not found.",
            "index_state": inspect_code_graph_artifacts(str(root)),
        }

    command = [
        executable,
        "start",
        "--repo-path",
        str(root),
        "--update-graph",
    ]
    if clean:
        command.append("--clean")

    completed = subprocess.run(
        command,
        cwd=_cgr_run_dir(),
        env=_cgr_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if isinstance(part, str) and part.strip()
    ).strip()
    status = inspect_code_graph_status(str(root), include_debug=include_debug)

    result = {
        "ok": completed.returncode == 0,
        "reason": "Live code graph synchronized." if completed.returncode == 0 else _summarize_failure(combined_output),
        "returncode": completed.returncode,
        "index_state": status.get("index_state", inspect_code_graph_artifacts(str(root))),
        "live_graph_state": status.get("live_graph_state", {}),
        "ready": status.get("ready", False),
    }
    if include_debug and combined_output:
        result["output"] = combined_output
    return result


def _find_cgr_executable() -> str | None:
    local_candidate = Path(sys.executable).resolve().parent / "cgr"
    if local_candidate.exists():
        return str(local_candidate)
    return shutil.which("cgr")


def _cgr_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_MODEL", None)
    return env


def _cgr_run_dir() -> str:
    return str(get_live_graph_runtime_root().resolve())


def _parse_live_graph_state(output: str, returncode: int) -> dict[str, Any]:
    clean_output = _strip_ansi(output)
    node_count = _extract_total_count(clean_output, "Total Nodes")
    relationship_count = _extract_total_count(clean_output, "Total Relationships")
    populated = False
    if node_count is not None:
        populated = node_count > 0
    elif relationship_count is not None:
        populated = relationship_count > 0

    return {
        "connected": returncode == 0,
        "node_count": node_count,
        "relationship_count": relationship_count,
        "populated": populated,
    }


def _extract_total_count(output: str, label: str) -> int | None:
    pattern = rf"{label}\s*[^\d]*(\d+)"
    match = re.search(pattern, output, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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
