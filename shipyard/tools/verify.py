from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Any

_BLOCKED_TOKENS = ("\x00", "$(", "`")
_VERIFY_TIMEOUT = 10  # 10 seconds hard cap — enough to catch errors, not enough to hang


def run_verification(commands: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for command in commands:
        cmd_str = str(command).strip()
        if any(token in cmd_str for token in _BLOCKED_TOKENS):
            results.append({"command": cmd_str, "returncode": -1, "stdout": "", "stderr": "Blocked"})
            continue

        _SHELL_FEATURES = ("|", ">", "<", "&&", "||", ";", "2>&1")
        use_shell = any(token in cmd_str for token in _SHELL_FEATURES)
        args = cmd_str if use_shell else shlex.split(cmd_str)

        try:
            completed = subprocess.run(
                args,
                shell=use_shell,
                capture_output=True,
                text=True,
                check=False,
                timeout=_VERIFY_TIMEOUT,
                preexec_fn=os.setsid,
            )
            results.append({
                "command": cmd_str,
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "").strip()[:2000],
                "stderr": (completed.stderr or "").strip()[:2000],
            })
        except subprocess.TimeoutExpired:
            # Kill entire process group
            try:
                os.killpg(os.getpgid(0), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            results.append({"command": cmd_str, "returncode": -1, "stdout": "", "stderr": f"Timed out ({_VERIFY_TIMEOUT}s)"})
        except (FileNotFoundError, ValueError) as exc:
            results.append({"command": cmd_str, "returncode": -1, "stdout": "", "stderr": str(exc)[:200]})

    return results
