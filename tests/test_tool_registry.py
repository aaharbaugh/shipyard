from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.tool_registry import (
    ToolRegistry,
    _ALLOWED_IMPORTS,
    validate_tool_source,
)


SIMPLE_TOOL = textwrap.dedent(
    """\
    def run(args):
        \"\"\"Return the length of a string.\"\"\"
        value = args.get("value", "")
        return {"length": len(str(value))}
    """
)

TOOL_WITH_ALLOWED_IMPORT = textwrap.dedent(
    """\
    import re

    def run(args):
        \"\"\"Count regex matches in a string.\"\"\"
        pattern = args.get("pattern", "")
        text = args.get("text", "")
        return {"count": len(re.findall(pattern, text))}
    """
)


class TestValidateToolSource(unittest.TestCase):
    def test_valid_tool_passes(self):
        errors = validate_tool_source("my_tool", SIMPLE_TOOL)
        self.assertEqual(errors, [])

    def test_valid_tool_with_allowed_import(self):
        errors = validate_tool_source("regex_tool", TOOL_WITH_ALLOWED_IMPORT)
        self.assertEqual(errors, [])

    def test_missing_run_function(self):
        source = "def helper(): pass\n"
        errors = validate_tool_source("no_run", source)
        self.assertTrue(any("run" in e for e in errors))

    def test_blocked_import_subprocess(self):
        source = "import subprocess\ndef run(args): return subprocess.run([])\n"
        errors = validate_tool_source("bad_tool", source)
        self.assertTrue(any("subprocess" in e for e in errors))

    def test_blocked_import_os(self):
        source = "import os\ndef run(args): return {'cwd': os.getcwd()}\n"
        errors = validate_tool_source("bad_os", source)
        self.assertTrue(any("os" in e for e in errors))

    def test_blocked_import_socket(self):
        source = "import socket\ndef run(args): return {}\n"
        errors = validate_tool_source("bad_socket", source)
        self.assertTrue(any("socket" in e for e in errors))

    def test_blocked_call_eval(self):
        source = "def run(args): return {'x': eval(args['code'])}\n"
        errors = validate_tool_source("eval_tool", source)
        self.assertTrue(any("eval" in e for e in errors))

    def test_blocked_call_exec(self):
        source = "def run(args): exec(args['code']); return {}\n"
        errors = validate_tool_source("exec_tool", source)
        self.assertTrue(any("exec" in e for e in errors))

    def test_invalid_tool_name_spaces(self):
        errors = validate_tool_source("my tool", SIMPLE_TOOL)
        self.assertTrue(any("invalid" in e.lower() for e in errors))

    def test_invalid_tool_name_leading_digit(self):
        errors = validate_tool_source("1bad", SIMPLE_TOOL)
        self.assertTrue(any("invalid" in e.lower() for e in errors))

    def test_syntax_error_in_source(self):
        source = "def run(args):\n  return {broken\n"
        errors = validate_tool_source("syntax_err", source)
        self.assertTrue(any("syntax" in e.lower() for e in errors))

    def test_source_too_large(self):
        big_source = "def run(args): return {}\n" + "# " + "x" * 9000
        errors = validate_tool_source("big_tool", big_source)
        self.assertTrue(any("exceeds" in e for e in errors))


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.registry = ToolRegistry(dynamic_dir=Path(self._tmpdir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_synthesize_and_invoke(self):
        result = self.registry.synthesize("len_tool", SIMPLE_TOOL)
        self.assertTrue(result["ok"])
        out = self.registry.invoke("len_tool", {"value": "hello"})
        self.assertIsNone(out.get("error"))
        self.assertEqual(out["length"], 5)

    def test_synthesize_blocked_import_rejected(self):
        source = "import subprocess\ndef run(args): return {}\n"
        result = self.registry.synthesize("blocked", source)
        self.assertFalse(result["ok"])
        self.assertTrue(any("subprocess" in e for e in result["errors"]))

    def test_invoke_missing_tool(self):
        out = self.registry.invoke("nonexistent", {})
        self.assertIsNotNone(out.get("error"))
        self.assertIn("not found", out["error"])

    def test_list_tools_empty(self):
        tools = self.registry.list_tools()
        self.assertEqual(tools, [])

    def test_list_tools_after_synthesize(self):
        self.registry.synthesize("len_tool", SIMPLE_TOOL)
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "len_tool")
        self.assertIn("length", tools[0]["description"])

    def test_delete_tool(self):
        self.registry.synthesize("len_tool", SIMPLE_TOOL)
        deleted = self.registry.delete("len_tool")
        self.assertTrue(deleted)
        tools = self.registry.list_tools()
        self.assertEqual(tools, [])

    def test_delete_nonexistent_tool(self):
        deleted = self.registry.delete("ghost")
        self.assertFalse(deleted)

    def test_tool_timeout(self):
        source = textwrap.dedent(
            """\
            import time
            def run(args):
                time.sleep(999)
                return {"done": True}
            """
        )
        # time is not in the allowlist so synthesis should be rejected
        result = self.registry.synthesize("slow_tool", source)
        self.assertFalse(result["ok"])

    def test_invoke_timeout_enforced(self):
        # Manually write a tool that bypasses synthesis validation (white-box)
        # to test that the runtime timeout actually fires.
        import time as _time
        tool_path = Path(self._tmpdir) / "infinite_tool.py"
        tool_path.write_text(
            "import time\ndef run(args):\n    time.sleep(999)\n    return {}\n",
            encoding="utf-8",
        )
        out = self.registry.invoke("infinite_tool", {}, timeout=0.2)
        self.assertIsNotNone(out.get("error"))
        self.assertIn("timed out", out["error"])

    def test_cache_invalidated_on_re_synthesize(self):
        source_v1 = "def run(args): return {'version': 1}\n"
        source_v2 = "def run(args): return {'version': 2}\n"
        self.registry.synthesize("versioned", source_v1)
        out1 = self.registry.invoke("versioned", {})
        self.assertEqual(out1["version"], 1)
        self.registry.synthesize("versioned", source_v2)
        out2 = self.registry.invoke("versioned", {})
        self.assertEqual(out2["version"], 2)

    def test_tool_exception_returns_error(self):
        source = "def run(args): raise ValueError('boom')\n"
        self.registry.synthesize("boom_tool", source)
        out = self.registry.invoke("boom_tool", {})
        self.assertIsNotNone(out.get("error"))
        self.assertIn("boom", out["error"])


if __name__ == "__main__":
    unittest.main()
