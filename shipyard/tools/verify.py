from __future__ import annotations

import shlex
import subprocess
from typing import Any

# Only block true injection vectors — pipes and chaining are allowed via shell=True.
_BLOCKED_TOKENS = ("\x00", "$(", "`")

# Hard timeout per verification command (seconds).
_VERIFY_TIMEOUT = 120


def run_verification(commands: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for command in commands:
        cmd_str = str(command).strip()
        # Reject shell metacharacters before execution.
        if any(token in cmd_str for token in _BLOCKED_TOKENS):
            results.append(
                {
                    "command": cmd_str,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Unsafe shell syntax blocked in verification command: {cmd_str!r}",
                }
            )
            continue

        _SHELL_FEATURES = ("|", ">", "<", "&&", "||", ";", "2>&1")
        use_shell = any(token in cmd_str for token in _SHELL_FEATURES)

        if not use_shell:
            try:
                args = shlex.split(cmd_str)
            except ValueError as exc:
                results.append(
                    {
                        "command": cmd_str,
                        "returncode": -1,
                        "stdout": "",
                        "stderr": f"Could not parse verification command: {exc}",
                    }
                )
                continue
        else:
            args = cmd_str

        try:
            completed = subprocess.run(
                args,
                shell=use_shell,
                capture_output=True,
                text=True,
                check=False,
                timeout=_VERIFY_TIMEOUT,
            )
            results.append(
                {
                    "command": cmd_str,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "command": cmd_str,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Verification command timed out after {_VERIFY_TIMEOUT}s.",
                }
            )
        except FileNotFoundError:
            results.append(
                {
                    "command": cmd_str,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Command not found: {args[0]!r}",
                }
            )

    return results
