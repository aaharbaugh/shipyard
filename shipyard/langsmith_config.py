from __future__ import annotations

import os
from typing import Any


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )


def build_langgraph_config(
    session_id: str | None,
    *,
    instruction: str | None = None,
    step_index: int | None = None,
    step_count: int | None = None,
) -> dict[str, Any]:
    sid = str(session_id or "default")
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": sid,
        }
    }

    if not langsmith_enabled():
        return config

    metadata: dict[str, Any] = {
        "session_id": sid,
    }
    if instruction:
        metadata["instruction"] = instruction
    if step_index is not None:
        metadata["step_index"] = step_index
    if step_count is not None:
        metadata["step_count"] = step_count

    config.update(
        {
            "run_name": "shipyard.run_step",
            "tags": ["shipyard", "langgraph", "workbench"],
            "metadata": metadata,
        }
    )
    return config
