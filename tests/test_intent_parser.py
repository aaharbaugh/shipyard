from __future__ import annotations

import unittest

from shipyard.intent_parser import derive_edit_spec, infer_edit_mode, parse_instruction, parse_occurrence_selector, split_instruction_steps


class IntentParserTests(unittest.TestCase):
    def test_parse_instruction_handles_write(self) -> None:
        result = parse_instruction('write "hello" to file')
        self.assertEqual(result, ("write_file", None, "hello"))

    def test_parse_instruction_handles_bare_write(self) -> None:
        result = parse_instruction("write hello world")
        self.assertEqual(result, ("write_file", None, "hello world"))

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

    def test_parse_instruction_handles_middle_occurrence_replace(self) -> None:
        result = parse_instruction("make the middle ha into ho")
        self.assertEqual(result, ("anchor", "ha", "ho"))

    def test_parse_occurrence_selector_detects_middle(self) -> None:
        self.assertEqual(parse_occurrence_selector("make the middle ha into ho"), "middle")

    def test_parse_instruction_handles_delete(self) -> None:
        result = parse_instruction("delete unknown-Copy.txt")
        self.assertEqual(result, ("delete_file", None, ""))

    def test_parse_instruction_handles_create_javascript_file(self) -> None:
        result = parse_instruction("make a javascript file")
        self.assertEqual(result, ("write_file", None, ""))

    def test_parse_instruction_handles_copy_request(self) -> None:
        result = parse_instruction("make some copies of pledge.txt")
        self.assertEqual(result, ("copy_file", "pledge.txt", "3"))

    def test_parse_instruction_handles_multiple_new_files(self) -> None:
        result = parse_instruction("make 2 new files")
        self.assertEqual(result, ("create_files", None, "2"))

    def test_parse_instruction_handles_symbol_rename_request(self) -> None:
        result = parse_instruction("change base to boos and update other places where base appears as boos")
        self.assertEqual(result, ("rename_symbol", "base", "boos"))

    def test_parse_instruction_handles_all_instances_symbol_rename_request(self) -> None:
        result = parse_instruction("change all instances of base to boos")
        self.assertEqual(result, ("rename_symbol", "base", "boos"))

    def test_parse_instruction_handles_replace_identifier_as_symbol_rename(self) -> None:
        result = parse_instruction("replace total with totality")
        self.assertEqual(result, ("rename_symbol", "total", "totality"))

    def test_parse_instruction_handles_blank_new_file(self) -> None:
        result = parse_instruction("make a new file")
        self.assertEqual(result, ("write_file", None, ""))

    def test_parse_instruction_handles_blank_named_file(self) -> None:
        result = parse_instruction("create pledge.txt")
        self.assertEqual(result, ("write_file", None, ""))

    def test_split_instruction_steps_handles_sequential_prompt(self) -> None:
        steps = split_instruction_steps("create 4 random files and in file 3, write hello world")
        self.assertEqual(steps, ["create 4 random files", "in file 3, write hello world"])


if __name__ == "__main__":
    unittest.main()
