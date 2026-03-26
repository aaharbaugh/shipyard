from __future__ import annotations

import unittest

from shipyard.intent_parser import derive_edit_spec, infer_edit_mode, parse_instruction, parse_occurrence_selector, split_instruction_steps


class IntentParserFallbackTests(unittest.TestCase):
    def test_infer_edit_mode_prefers_function_context(self) -> None:
        result = infer_edit_mode({"context": {"function_name": "boot_system"}})
        self.assertEqual(result, "named_function")

    def test_derive_edit_spec_uses_context_before_instruction(self) -> None:
        mode, anchor, replacement, notes = derive_edit_spec(
            {
                "instruction": 'replace "old" with "new"',
                "context": {
                    "search_text": "one",
                    "replace_text": "two",
                },
            }
        )

        self.assertEqual(mode, "anchor")
        self.assertEqual(anchor, "one")
        self.assertEqual(replacement, "two")
        self.assertTrue(notes)

    def test_parse_instruction_handles_core_fallback_write(self) -> None:
        result = parse_instruction('write "hello" to file')
        self.assertEqual(result, ("write_file", None, "hello"))

    def test_parse_instruction_handles_core_fallback_symbol_rename(self) -> None:
        result = parse_instruction("replace total with totality")
        self.assertEqual(result, ("rename_symbol", "total", "totality"))

    def test_parse_instruction_handles_core_fallback_batch_create(self) -> None:
        result = parse_instruction("make 2 new files")
        self.assertEqual(result, ("create_files", None, "2"))

    def test_parse_occurrence_selector_detects_middle(self) -> None:
        self.assertEqual(parse_occurrence_selector("make the middle ha into ho"), "middle")

    def test_split_instruction_steps_handles_sequential_prompt(self) -> None:
        steps = split_instruction_steps("create 4 random files and in file 3, write hello world")
        self.assertEqual(steps, ["create 4 random files", "in file 3, write hello world"])


if __name__ == "__main__":
    unittest.main()
