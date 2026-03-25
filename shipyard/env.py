from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_path: str | Path | None = None) -> dict[str, str]:
    path = Path(env_path) if env_path is not None else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value

    return loaded
