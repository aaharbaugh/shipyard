from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Any

_BLOCKED_TOKENS = ("\x00", "$(", "`")
_VERIFY_TIMEOUT = 10


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
            proc = subprocess.Popen(
                args,
                shell=use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )
            try:
                stdout, stderr = proc.communicate(timeout=_VERIFY_TIMEOUT)
                results.append({
                    "command": cmd_str,
                    "returncode": proc.returncode,
                    "stdout": (stdout or "").strip()[:2000],
                    "stderr": (stderr or "").strip()[:2000],
                })
            except subprocess.TimeoutExpired:
                # Kill the entire process group using the actual PID
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.wait()
                results.append({"command": cmd_str, "returncode": -1, "stdout": "", "stderr": f"Timed out ({_VERIFY_TIMEOUT}s)"})
        except (FileNotFoundError, ValueError, OSError) as exc:
            results.append({"command": cmd_str, "returncode": -1, "stdout": "", "stderr": str(exc)[:200]})

    return results
