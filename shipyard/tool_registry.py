from __future__ import annotations

import ast
import re
import sys
import threading
import types
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Safe import allowlist — only these stdlib modules may appear in tool code.
# Anything not in this set is rejected at synthesis time.
# ---------------------------------------------------------------------------
_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "ast",
        "re",
        "json",
        "pathlib",
        "math",
        "collections",
        "itertools",
        "functools",
        "typing",
        "datetime",
        "textwrap",
        "hashlib",
        "dataclasses",
        "enum",
        "string",
        "unicodedata",
        "difflib",
        "fnmatch",
        "glob",
        "pprint",
        "copy",
        "operator",
        "statistics",
        "decimal",
        "fractions",
        "io",
    }
)

# Builtins that must never appear as calls in tool code.
_BLOCKED_CALLS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "object",
    }
)

# The single function every dynamic tool must export.
_REQUIRED_ENTRYPOINT = "run"

# Hard ceiling on tool execution time.
_DEFAULT_TOOL_TIMEOUT = 10  # seconds

# Maximum size of synthesized tool source (chars).
_MAX_SOURCE_CHARS = 8_000

# Validated tool name: lowercase words joined by underscores.
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

def _collect_imports(tree: ast.Module) -> list[str]:
    """Return top-level module names used in import statements."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _collect_blocked_calls(tree: ast.Module) -> list[str]:
    """Return names of any blocked builtins that are called."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                found.append(node.func.id)
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_CALLS:
                found.append(node.func.attr)
    return found


def _has_required_entrypoint(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _REQUIRED_ENTRYPOINT:
            return True
    return False


def validate_tool_source(name: str, source: str) -> list[str]:
    """
    Return a list of human-readable error strings.
    Empty list means the source passes all static checks.
    """
    errors: list[str] = []

    if not _TOOL_NAME_RE.match(name):
        errors.append(
            f"Tool name {name!r} is invalid. Use lowercase letters, digits, and underscores (2-64 chars, start with letter)."
        )

    if len(source) > _MAX_SOURCE_CHARS:
        errors.append(f"Tool source exceeds {_MAX_SOURCE_CHARS} chars ({len(source)}).")

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        errors.append(f"Syntax error in tool source: {exc}")
        return errors  # Can't do further checks without a valid AST.

    if not _has_required_entrypoint(tree):
        errors.append(f"Tool must define a top-level function named '{_REQUIRED_ENTRYPOINT}(args: dict) -> dict'.")

    bad_imports = [m for m in _collect_imports(tree) if m not in _ALLOWED_IMPORTS]
    if bad_imports:
        errors.append(
            f"Blocked import(s): {', '.join(sorted(bad_imports))}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_IMPORTS))}."
        )

    bad_calls = _collect_blocked_calls(tree)
    if bad_calls:
        errors.append(f"Blocked call(s): {', '.join(sorted(set(bad_calls)))}.")

    return errors


# ---------------------------------------------------------------------------
# Execution with timeout
# ---------------------------------------------------------------------------

def _run_with_timeout(fn: Callable, args: dict, timeout: float) -> dict:
    result: dict = {}
    exc_holder: list[Exception] = []

    def _target() -> None:
        try:
            result.update(fn(args))
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return {"result": None, "error": f"Tool timed out after {timeout}s."}
    if exc_holder:
        return {"result": None, "error": str(exc_holder[0])}
    if not result:
        return {"result": None, "error": "Tool returned no output."}
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registry for dynamic (LLM-synthesised) tools.

    Tools are stored as Python source files in `dynamic_dir`.
    Each file must define ``run(args: dict) -> dict``.
    """

    def __init__(self, dynamic_dir: Path | None = None) -> None:
        if dynamic_dir is None:
            dynamic_dir = Path(__file__).parent / "tools" / "dynamic"
        self.dynamic_dir = Path(dynamic_dir)
        self.dynamic_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Callable] = {}
        # Version counter: bumped on every synthesize so _load gets a fresh module key.
        self._versions: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata for every registered dynamic tool."""
        tools: list[dict[str, Any]] = []
        for path in sorted(self.dynamic_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            name = path.stem
            description = self._read_description(path)
            tools.append({"name": name, "description": description, "source_path": str(path)})
        return tools

    def get_tool(self, name: str) -> Callable | None:
        """Load and return the ``run`` function for *name*, or None."""
        if name in self._cache:
            return self._cache[name]
        path = self.dynamic_dir / f"{name}.py"
        if not path.exists():
            return None
        fn = self._load(name, path)
        if fn is not None:
            self._cache[name] = fn
        return fn

    def synthesize(self, name: str, source: str) -> dict[str, Any]:
        """
        Validate *source* and write it to the dynamic dir.

        Returns ``{"ok": True, "path": "..."}`` on success,
        ``{"ok": False, "errors": [...]}`` on failure.
        """
        errors = validate_tool_source(name, source)
        if errors:
            return {"ok": False, "errors": errors}
        path = self.dynamic_dir / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        # Bump version so _load uses a unique module key, forcing a fresh exec.
        self._versions[name] = self._versions.get(name, 0) + 1
        # Evict old entry from cache and sys.modules.
        self._cache.pop(name, None)
        old_key = f"_shipyard_dynamic_{name}_v{self._versions[name] - 1}"
        sys.modules.pop(old_key, None)
        return {"ok": True, "path": str(path)}

    def invoke(self, name: str, args: dict, timeout: float = _DEFAULT_TOOL_TIMEOUT) -> dict[str, Any]:
        """
        Run the tool *name* with *args* under a thread timeout.

        Returns the tool's result dict, or ``{"result": None, "error": "..."}``
        on failure.
        """
        fn = self.get_tool(name)
        if fn is None:
            return {"result": None, "error": f"Tool {name!r} not found in registry."}
        return _run_with_timeout(fn, args, timeout)

    def delete(self, name: str) -> bool:
        """Remove a dynamic tool by name. Returns True if deleted."""
        path = self.dynamic_dir / f"{name}.py"
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            sys.modules.pop(f"_shipyard_dynamic_{name}", None)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, name: str, path: Path) -> Callable | None:
        version = self._versions.get(name, 0)
        module_key = f"_shipyard_dynamic_{name}_v{version}"
        try:
            source = path.read_text(encoding="utf-8")
            code = compile(source, str(path), "exec")
            module = types.ModuleType(module_key)
            module.__file__ = str(path)
            # exec is safe here: source was already validated by validate_tool_source
            exec(code, module.__dict__)  # noqa: S102
            sys.modules[module_key] = module
            fn = getattr(module, _REQUIRED_ENTRYPOINT, None)
            return fn if callable(fn) else None
        except Exception:  # noqa: BLE001
            return None

    def _read_description(self, path: Path) -> str:
        """Extract the docstring of the ``run`` function as description."""
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == _REQUIRED_ENTRYPOINT:
                    return ast.get_docstring(node) or ""
        except Exception:  # noqa: BLE001
            pass
        return ""


# Module-level singleton shared across the process.
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
