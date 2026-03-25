from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from shipyard.prompts import build_proposal_prompt
from shipyard.proposal import propose_edit


class AgentPlanningTests(unittest.TestCase):
    def test_heuristic_proposal_uses_context_search_and_replace(self) -> None:
        result = propose_edit(
            {
                "instruction": "Update the file",
                "proposal_mode": "heuristic",
                "context": {
                    "file_hint": "demo.py",
                    "search_text": 'print("old")',
                    "replace_text": 'print("new")',
                },
            }
        )

        self.assertEqual(result["target_path"], "demo.py")
        self.assertEqual(result["anchor"], 'print("old")')
        self.assertEqual(result["replacement"], 'print("new")')
        self.assertEqual(result["edit_mode"], "anchor")
        self.assertEqual(result["provider"], "heuristic")
        self.assertTrue(result["is_valid"])

    def test_heuristic_proposal_parses_simple_replace_instruction(self) -> None:
        result = propose_edit(
            {
                "instruction": 'replace "before" with "after"',
                "proposal_mode": "heuristic",
                "context": {"file_hint": "demo.py"},
            }
        )

        self.assertEqual(result["target_path"], "demo.py")
        self.assertEqual(result["anchor"], "before")
        self.assertEqual(result["replacement"], "after")
        self.assertEqual(result["edit_mode"], "anchor")
        self.assertTrue(result["is_valid"])

    def test_openai_mode_without_key_falls_back_to_heuristic(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = propose_edit(
                {
                    "instruction": 'replace "one" with "two"',
                    "proposal_mode": "openai",
                    "context": {"file_hint": "demo.py"},
                }
            )

        self.assertEqual(result["provider"], "heuristic")
        self.assertIn("OPENAI_API_KEY", result["provider_reason"])

    def test_heuristic_proposal_switches_to_named_function_mode(self) -> None:
        result = propose_edit(
            {
                "instruction": "Update the startup routine",
                "proposal_mode": "heuristic",
                "context": {
                    "file_hint": "demo.py",
                    "function_name": "boot_system",
                },
            }
        )

        self.assertEqual(result["target_path"], "demo.py")
        self.assertEqual(result["edit_mode"], "named_function")
        self.assertIn("named-function edit mode", result["provider_reason"])

    def test_heuristic_proposal_parses_write_file_instruction(self) -> None:
        result = propose_edit(
            {
                "instruction": 'write "hello world" to file',
                "proposal_mode": "heuristic",
                "context": {"file_hint": "demo.txt"},
            }
        )

        self.assertEqual(result["target_path"], "demo.txt")
        self.assertEqual(result["edit_mode"], "write_file")
        self.assertEqual(result["replacement"], "hello world")

    def test_heuristic_proposal_parses_unquoted_write_instruction(self) -> None:
        result = propose_edit(
            {
                "instruction": "write hello world to new file",
                "proposal_mode": "heuristic",
                "context": {"file_hint": "demo.txt"},
            }
        )

        self.assertEqual(result["target_path"], "demo.txt")
        self.assertEqual(result["edit_mode"], "write_file")
        self.assertEqual(result["replacement"], "hello world")
        self.assertTrue(result["is_valid"])

    def test_heuristic_proposal_infers_filename_from_instruction_in_testing_mode(self) -> None:
        result = propose_edit(
            {
                "instruction": "edit unknown.txt in the session folder",
                "proposal_mode": "heuristic",
                "context": {"testing_mode": True},
                "session_id": "demo",
            }
        )

        self.assertIn("/.shipyard/data/workspace/demo/unknown.txt", result["target_path"])
        self.assertEqual(result["target_path_source"], "sandboxed_target_path")

    def test_heuristic_proposal_captures_middle_occurrence_selector(self) -> None:
        result = propose_edit(
            {
                "instruction": "make the middle ha into ho",
                "proposal_mode": "heuristic",
                "context": {"file_hint": "demo.txt"},
            }
        )

        self.assertEqual(result["edit_mode"], "anchor")
        self.assertEqual(result["anchor"], "ha")
        self.assertEqual(result["replacement"], "ho")
        self.assertEqual(result["occurrence_selector"], "middle")

    def test_heuristic_proposal_supports_delete_file(self) -> None:
        result = propose_edit(
            {
                "instruction": "delete unknown-Copy.txt",
                "proposal_mode": "heuristic",
                "context": {"testing_mode": True},
                "session_id": "demo",
            }
        )

        self.assertEqual(result["edit_mode"], "delete_file")
        self.assertIn("/.shipyard/data/workspace/demo/unknown-Copy.txt", result["target_path"])
        self.assertTrue(result["is_valid"])

    def test_heuristic_proposal_infers_javascript_filename_for_new_file(self) -> None:
        result = propose_edit(
            {
                "instruction": "make a new file with javascript code",
                "proposal_mode": "heuristic",
                "session_id": "demo",
            }
        )

        self.assertRegex(result["target_path"], r"scratch-[0-9a-f]{6}\.js$")
        self.assertEqual(result["edit_mode"], "write_file")

    def test_auto_mode_uses_fast_path_for_blank_new_file(self) -> None:
        result = propose_edit(
            {
                "instruction": "make a new file",
                "session_id": "demo",
            }
        )

        self.assertEqual(result["provider"], "heuristic")
        self.assertEqual(result["edit_mode"], "write_file")
        self.assertEqual(result["replacement"], "")

    def test_auto_mode_uses_openai_when_key_is_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal._openai_or_fallback",
            return_value={"provider": "openai"},
        ) as mocked:
            result = propose_edit(
                {
                    "instruction": "make a new file",
                    "session_id": "demo",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(result["provider"], "openai")

    def test_auto_mode_uses_fast_path_for_multiple_new_files(self) -> None:
        result = propose_edit(
            {
                "instruction": "make 2 new files",
                "session_id": "demo",
            }
        )

        self.assertEqual(result["provider"], "heuristic")
        self.assertEqual(result["edit_mode"], "create_files")
        self.assertEqual(result["quantity"], 2)
        self.assertEqual(result["replacement"], "")
        self.assertTrue(result["is_valid"])

    def test_create_files_with_embedded_write_request_uses_llm_path_when_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal._openai_or_fallback",
            return_value={"provider": "openai"},
        ) as mocked:
            result = propose_edit(
                {
                    "instruction": "create 45 random files and in file 32 write a python algorithm",
                    "session_id": "demo",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(result["provider"], "openai")

    def test_generative_write_to_named_file_uses_llm_path_when_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal._openai_or_fallback",
            return_value={"provider": "openai"},
        ) as mocked:
            result = propose_edit(
                {
                    "instruction": "write a random python algorithm in file scratch_copy_30.py",
                    "session_id": "demo",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(result["provider"], "openai")

    def test_replace_identifier_uses_llm_path_when_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal._openai_or_fallback",
            return_value={"provider": "openai"},
        ) as mocked:
            result = propose_edit(
                {
                    "instruction": "replace total with totality in scratch_copy_3.py",
                    "session_id": "demo",
                }
            )

        mocked.assert_called_once()
        self.assertEqual(result["provider"], "openai")

    def test_heuristic_proposal_uses_append_context(self) -> None:
        result = propose_edit(
            {
                "instruction": "Add more text",
                "proposal_mode": "heuristic",
                "context": {
                    "file_hint": "demo.txt",
                    "append_text": "\nsecond line",
                },
            }
        )

        self.assertEqual(result["target_path"], "demo.txt")
        self.assertEqual(result["edit_mode"], "append")
        self.assertEqual(result["replacement"], "\nsecond line")
        self.assertTrue(result["is_valid"])

    def test_build_proposal_prompt_includes_function_and_graph_context(self) -> None:
        prompt = build_proposal_prompt(
            {
                "instruction": "Update boot_system",
                "target_path": "demo.py",
                "context": {
                    "file_hint": "demo.py",
                    "function_name": "boot_system",
                    "helper_notes": "Focus on startup behavior.",
                },
                "current_function_source": "def boot_system():\n    return 'old'\n",
                "helper_output": {
                    "helper_agent": {
                        "task_type": "function_edit_planning",
                        "recommendation": "Prepare a full replacement function body.",
                        "notes": "Target the startup path.",
                    },
                    "edit_context": {
                        "mode": "named_function",
                        "function_name": "boot_system",
                        "query_mode": "function_source_only",
                    },
                },
                "code_graph_status": {
                    "ready": True,
                    "query_mode": "function_source_only",
                    "context_collected": True,
                },
            }
        )

        self.assertIn("Helper-agent recommendation:", prompt)
        self.assertIn("Current function source:", prompt)
        self.assertIn("Code graph status:", prompt)
        self.assertIn("boot_system", prompt)

    def test_openai_proposal_supports_content_field_for_named_function(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"edit_mode":"named_function","content":"def boot_system():\\n    return \\"new\\"\\n"}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal.httpx.Client",
            return_value=mock_client,
        ):
            result = propose_edit(
                {
                    "instruction": "Update boot_system",
                    "proposal_mode": "openai",
                    "context": {
                        "file_hint": "demo.py",
                        "function_name": "boot_system",
                    },
                }
            )

        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["edit_mode"], "named_function")
        self.assertIsNone(result["anchor"])
        self.assertIn('return "new"', result["replacement"])
        self.assertTrue(result["is_valid"])

    def test_openai_proposal_does_not_override_explicit_target_path(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"edit_mode":"write_file","target_path":"override.py","content":"hello"}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal.httpx.Client",
            return_value=mock_client,
        ):
            result = propose_edit(
                {
                    "instruction": "write hello to new file",
                    "proposal_mode": "openai",
                    "target_path": "/tmp/keep-me.py",
                    "context": {
                        "file_hint": "/tmp/other.py",
                    },
                }
            )

        self.assertEqual(result["target_path"], "/tmp/keep-me.py")

    def test_openai_proposal_does_not_override_filename_in_instruction(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"edit_mode":"write_file","target_path":"scratch.py","content":"hello"}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal.httpx.Client",
            return_value=mock_client,
        ):
            result = propose_edit(
                {
                    "instruction": "add a random algorithm to scratch_copy_32.py",
                    "proposal_mode": "openai",
                    "session_id": "demo",
                    "context": {"testing_mode": True},
                }
            )

        self.assertIn("scratch_copy_32.py", result["target_path"])

    def test_openai_anchor_identifier_replace_promotes_to_rename_symbol(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"edit_mode":"anchor","target_path":"scratch_copy_3.py","anchor":"total","replacement":"totality"}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.proposal.httpx.Client",
            return_value=mock_client,
        ):
            result = propose_edit(
                {
                    "instruction": "in file scratch_copy_3.py replace total with totality",
                    "proposal_mode": "openai",
                    "session_id": "demo",
                    "context": {"testing_mode": True},
                }
            )

        self.assertEqual(result["edit_mode"], "rename_symbol")
        self.assertIn("scratch_copy_3.py", result["target_path"])

    def test_heuristic_proposal_marks_missing_target_invalid(self) -> None:
        result = propose_edit(
            {
                "instruction": 'replace "hello" with "world"',
                "proposal_mode": "heuristic",
                "context": {},
            }
        )

        self.assertFalse(result["is_valid"])
        self.assertIn("Missing target_path.", result["validation_errors"])

    def test_heuristic_filename_in_prompt_overrides_stale_scratch_target(self) -> None:
        result = propose_edit(
            {
                "instruction": "add a random algorithm to scratch_copy_9.py",
                "proposal_mode": "heuristic",
                "target_path": "/tmp/scratch.py",
                "context": {"testing_mode": True},
                "session_id": "demo",
            }
        )

        self.assertIn("scratch_copy_9.py", result["target_path"])

    def test_heuristic_write_file_without_target_allocates_workspace_path(self) -> None:
        result = propose_edit(
            {
                "instruction": 'write "hello" to file',
                "proposal_mode": "heuristic",
                "session_id": "demo-session",
                "context": {},
            }
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["edit_mode"], "write_file")
        self.assertTrue(result["target_path"].endswith("scratch.py"))
        self.assertIn("/demo-session/", result["target_path"])


if __name__ == "__main__":
    unittest.main()
