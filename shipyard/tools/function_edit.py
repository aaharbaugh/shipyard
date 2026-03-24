from __future__ import annotations

from pathlib import Path

from codebase_rag.tools.file_editor import FileEditor


class FunctionEditError(ValueError):
    """Raised when a named-function edit cannot be applied safely."""


def get_function_source(path: str, function_name: str) -> str:
    file_editor = FileEditor(project_root=_project_root_for(path))
    resolved_path = _resolved_path_for(path)
    source = file_editor.get_function_source_code(resolved_path, function_name)
    if not source:
        raise FunctionEditError(
            f"Function `{function_name}` was not found unambiguously in {path}."
        )
    return source


def apply_function_edit(path: str, function_name: str, replacement: str) -> str:
    file_editor = FileEditor(project_root=_project_root_for(path))
    resolved_path = _resolved_path_for(path)
    current_source = file_editor.get_function_source_code(resolved_path, function_name)
    if not current_source:
        raise FunctionEditError(
            f"Function `{function_name}` was not found unambiguously in {path}."
        )

    success = file_editor.replace_code_block(resolved_path, current_source, replacement)
    if not success:
        raise FunctionEditError(
            f"Failed to replace function `{function_name}` in {path}."
        )
    return current_source


def _project_root_for(path: str) -> str:
    return str(Path(path).resolve().parent)


def _resolved_path_for(path: str) -> str:
    return str(Path(path).resolve())
