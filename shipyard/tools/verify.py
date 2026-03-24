from __future__ import annotations

import subprocess
from typing import Any


def run_verification(commands: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for command in commands:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )

    return results
