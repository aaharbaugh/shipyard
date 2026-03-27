from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shipyard.action_planner import _build_action_plan_prompt, plan_actions, _sanitize_state_for_scaffold_planning


class ActionPlannerTests(unittest.TestCase):
    def test_action_plan_prompt_lists_explicit_files(self) -> None:
        prompt = _build_action_plan_prompt(
            {
                "instruction": "Create a tiny Python repo with main.py, math_utils.py, formatter.py, and config.json",
                "session_id": "demo",
            }
        )

        self.assertIn("Explicit files mentioned by the user:", prompt)
        self.assertIn("main.py", prompt)
        self.assertIn("math_utils.py", prompt)
        self.assertIn("formatter.py", prompt)
        self.assertIn("config.json", prompt)

    def test_action_plan_prompt_includes_existing_file_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "formatter.py").write_text(
                "def format_result(value, unit):\n    return f\"{value:.2f} {unit}\"\n",
                encoding="utf-8",
            )
            prompt = _build_action_plan_prompt(
                {
                    "instruction": "In formatter.py, update format_result so it prefixes Average latency",
                    "session_id": "demo",
                }
            )

        self.assertIn("Existing file context:", prompt)
        self.assertIn("Existing file: formatter.py", prompt)
        self.assertIn("def format_result", prompt)

    def test_sanitize_state_for_scaffold_planning_strips_stale_scratch_targets(self) -> None:
        sanitized = _sanitize_state_for_scaffold_planning(
            {
                "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                "target_path": "/tmp/scratch.py",
                "context": {"file_hint": "/tmp/scratch.py", "testing_mode": True},
            }
        )

        self.assertIsNone(sanitized.get("target_path"))
        self.assertNotIn("file_hint", sanitized["context"])

    def test_heuristic_action_plan_splits_into_multiple_actions(self) -> None:
        result = plan_actions(
            {
                "instruction": "create 4 random files and in file 3, write hello world",
                "proposal_mode": "heuristic",
                "session_id": "demo",
            }
        )

        self.assertEqual(len(result["actions"]), 2)
        self.assertEqual(result["actions"][0]["edit_mode"], "create_files")
        self.assertEqual(result["actions"][1]["edit_mode"], "write_file")

    def test_openai_action_plan_falls_back_to_heuristic_when_unavailable(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = plan_actions(
                {
                    "instruction": "make 2 new files and rename all instances of base to boos",
                }
            )

        self.assertEqual(result["provider"], "heuristic")
        self.assertGreaterEqual(len(result["actions"]), 2)

    def test_openai_action_plan_failure_returns_openai_invalid_action(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner.httpx.Client",
            side_effect=RuntimeError("boom"),
        ):
            result = plan_actions(
                {
                    "instruction": "replace total with totality in scratch_copy_3.py",
                }
            )

        self.assertEqual(result["provider"], "openai")
        self.assertEqual(len(result["actions"]), 1)
        self.assertFalse(result["actions"][0]["valid"])

    def test_single_simple_instruction_uses_openai_when_key_is_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner._openai_action_plan_or_fallback"
        ) as mocked:
            result = plan_actions(
                {
                    "instruction": "delete sample.txt",
                }
            )

        mocked.assert_called_once()

    def test_replace_instruction_uses_openai_when_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner._openai_action_plan_or_fallback",
            return_value={"provider": "openai", "actions": []},
        ) as mocked:
            plan_actions(
                {
                    "instruction": "replace total with totality in scratch_copy_3.py",
                }
            )

        mocked.assert_called_once()

    def test_generation_write_to_named_file_uses_openai_when_available(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner._openai_action_plan_or_fallback",
            return_value={"provider": "openai", "actions": []},
        ) as mocked:
            plan_actions(
                {
                    "instruction": "write a random python algorithm in file scratch_copy_30.py",
                }
            )

        mocked.assert_called_once()

    def test_openai_action_plan_preserves_filename_from_instruction(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"add a random algorithm to scratch_copy_11.py","edit_mode":"write_file","target_path":"scratch.py","replacement":"hello"}]}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner.httpx.Client",
            return_value=mock_client,
        ):
            result = plan_actions(
                {
                    "instruction": "add a random algorithm to scratch_copy_11.py",
                    "proposal_mode": "openai",
                    "session_id": "demo",
                    "context": {"testing_mode": True},
                }
            )

        self.assertEqual(len(result["actions"]), 1)
        self.assertIn("scratch_copy_11.py", result["actions"][0]["target_path"])

    def test_openai_action_plan_preserves_named_file_for_generated_code(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"add a random algorithm to scratch_copy_9.py","edit_mode":"write_file","target_path":"scratch.py","replacement":"def random_algorithm(data):\\n    return list(reversed(data))\\n"}]}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner.httpx.Client",
            return_value=mock_client,
        ):
            result = plan_actions(
                {
                    "instruction": "add a random algorithm to scratch_copy_9.py",
                    "proposal_mode": "openai",
                    "session_id": "demo",
                    "context": {"testing_mode": True},
                }
            )

        self.assertIn("scratch_copy_9.py", result["actions"][0]["target_path"])
        self.assertIn("random_algorithm", result["actions"][0]["replacement"])
        self.assertEqual(result["actions"][0]["edit_mode"], "write_file")

    def test_openai_action_plan_marks_explicit_file_scaffold_incomplete_when_files_are_missing(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"write main.py","edit_mode":"write_file","target_path":"main.py","replacement":"print(\\"hi\\")\\n"}]}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "shipyard.action_planner.httpx.Client",
            return_value=mock_client,
        ):
            result = plan_actions(
                {
                    "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                    "proposal_mode": "openai",
                    "session_id": "demo",
                    "context": {"testing_mode": True},
                }
            )

        self.assertFalse(result["is_valid"])
        self.assertTrue(any("math_utils.py" in error for error in result["validation_errors"]))

    def test_openai_action_plan_autofills_unique_anchor_pointers_before_validation(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Update main.py import usage","edit_mode":"anchor","target_path":"main.py","anchor":"formatted = formatter.format_result(result, unit)","replacement":"formatted = report.build_report(result, unit)"}]}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "main.py").write_text(
                "formatted = formatter.format_result(result, unit)\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Update main.py to use build_report instead of formatter.format_result",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["pointers"], [{"start": 0, "end": 49}])

    def test_openai_action_plan_marks_anchor_step_satisfied_when_replacement_already_present(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Update main.py to use build_report","edit_mode":"anchor","target_path":"main.py","anchor":"formatted = formatter.format_result(result, unit)","replacement":"formatted = report.build_report(result, unit)"}]}'
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "main.py").write_text(
                "formatted = report.build_report(result, unit)\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Update main.py to use build_report instead of formatter.format_result",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "read_file")

    def test_openai_action_plan_prefers_direct_scaffold_generation_for_explicit_multi_file_prompt(self) -> None:
        scaffold_response = Mock()
        scaffold_response.json.return_value = {
            "output_text": (
                '{"files":['
                '{"path":"main.py","content":"print(\\"hi\\")\\n"},'
                '{"path":"math_utils.py","content":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n"},'
                '{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n"},'
                '{"path":"config.json","content":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
                ']}'
            )
        }
        scaffold_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = scaffold_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")
        self.assertIn("generated scaffold files", result["provider_reason"])

    def test_openai_action_plan_does_not_force_scaffold_when_named_files_already_exist(self) -> None:
        action_response = Mock()
        action_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Update formatter.py","edit_mode":"anchor","target_path":"formatter.py","anchor":"return f\\"{value:.2f} {unit}\\"","replacement":"return f\\"Average latency: {value:.2f} {unit}\\"","full_file_rewrite":false},{"instruction":"Update main.py","edit_mode":"anchor","target_path":"main.py","anchor":"print(format_result(avg, config[\\"unit\\"]))","replacement":"print(format_result(avg, config[\\"unit\\"]))","full_file_rewrite":false}]}'
        }
        action_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "formatter.py").write_text("old\n", encoding="utf-8")
            (workspace / "main.py").write_text("old\n", encoding="utf-8")

            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = action_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "In formatter.py, update format_result. Then update main.py if needed.",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertNotEqual(result["actions"][0]["edit_mode"], "scaffold_files")

    def test_openai_action_plan_rejects_write_file_on_existing_file_without_full_rewrite(self) -> None:
        action_response = Mock()
        action_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Rewrite formatter.py","edit_mode":"write_file","target_path":"formatter.py","replacement":"new content","full_file_rewrite":false}]}'
        }
        action_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "formatter.py").write_text("old\n", encoding="utf-8")

            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = action_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Update formatter.py to change the output text.",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertFalse(result["is_valid"])
        self.assertTrue(
            any(
                "full_file_rewrite=true" in error
                for error in result["actions"][0]["validation_errors"]
            )
        )

    def test_openai_action_plan_rejects_tool_step_with_edit_content(self) -> None:
        action_response = Mock()
        action_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Read main.py","edit_mode":"read_file","target_path":"main.py","replacement":"print(\\"oops\\")"}]}'
        }
        action_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "main.py").write_text("print('hi')\n", encoding="utf-8")

            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = action_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Update main.py.",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertFalse(result["is_valid"])
        self.assertTrue(
            any(
                "read_file mode cannot include edit content." in error
                for error in result["actions"][0]["validation_errors"]
            )
        )

    def test_openai_action_plan_treats_empty_pointer_list_as_absent(self) -> None:
        action_response = Mock()
        action_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Add more runs to main.py","edit_mode":"anchor","target_path":"main.py","anchor":"print(\\"hi\\")\\n","replacement":"print(\\"hi\\")\\nprint(\\"bye\\")\\n","pointers":[]}]}'
        }
        action_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "main.py").write_text('print("hi")\n', encoding="utf-8")

            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = action_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "add some runs to main.py make sure you follow the existing structure",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["pointers"], [{"start": 0, "end": 12}])

    def test_openai_action_plan_accepts_dict_shaped_scaffold_output(self) -> None:
        scaffold_response = Mock()
        scaffold_response.json.return_value = {
            "output_text": (
                '{"main.py":"print(\\"hi\\")\\n",'
                '"math_utils.py":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n",'
                '"formatter.py":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n",'
                '"config.json":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
            )
        }
        scaffold_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = scaffold_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "target_path": "/tmp/scratch.py",
                        "context": {"testing_mode": True, "file_hint": "/tmp/scratch.py"},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")

    def test_openai_action_plan_reads_nested_output_text_for_scaffold_generation(self) -> None:
        scaffold_response = Mock()
        scaffold_response.json.return_value = {
            "output": [
                {
                    "content": [
                        {
                            "text": (
                                '{"files":['
                                '{"path":"main.py","content":"print(\\"hi\\")\\n"},'
                                '{"path":"math_utils.py","content":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n"},'
                                '{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n"},'
                                '{"path":"config.json","content":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
                                ']}'
                            )
                        }
                    ]
                }
            ]
        }
        scaffold_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = scaffold_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")

    def test_openai_action_plan_repairs_incomplete_scaffold_with_second_response(self) -> None:
        repaired_response = Mock()
        repaired_response.json.return_value = {
            "output_text": (
                '{"files":['
                '{"path":"main.py","content":"print(\\"hi\\")\\n"},'
                '{"path":"math_utils.py","content":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n"},'
                '{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n"},'
                '{"path":"config.json","content":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
                ']}'
            )
        }
        repaired_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = repaired_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["provider"], "openai")
        self.assertIn("generated scaffold files", result["provider_reason"])
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")
        self.assertEqual(
            [file_spec["path"] for file_spec in result["actions"][0]["files"]],
            ["main.py", "math_utils.py", "formatter.py", "config.json"],
        )

    def test_openai_action_plan_repairs_missing_actions_array_for_scaffold_prompt(self) -> None:
        first_response = Mock()
        first_response.json.return_value = {"output_text": '{"message":"I can help with that."}'}
        first_response.raise_for_status.return_value = None

        repaired_response = Mock()
        repaired_response.json.return_value = {
            "output_text": (
                '{"actions":[{"instruction":"Create tiny repo files","edit_mode":"scaffold_files","files":['
                '{"path":"main.py","content":"print(\\"hi\\")\\n"},'
                '{"path":"math_utils.py","content":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n"},'
                '{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n"},'
                '{"path":"config.json","content":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
                ']}]}'
            )
        }
        repaired_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.side_effect = [first_response, repaired_response]

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")

    def test_openai_action_plan_generates_scaffold_files_after_failed_repair(self) -> None:
        first_response = Mock()
        first_response.json.return_value = {"output_text": '{"message":"I can help with that."}'}
        first_response.raise_for_status.return_value = None

        repair_response = Mock()
        repair_response.json.return_value = {"output_text": '{"message":"still incomplete"}'}
        repair_response.raise_for_status.return_value = None

        scaffold_response = Mock()
        scaffold_response.json.return_value = {
            "output_text": (
                '{"files":['
                '{"path":"main.py","content":"print(\\"hi\\")\\n"},'
                '{"path":"math_utils.py","content":"def moving_average(values):\\n    return sum(values) / len(values) if values else 0\\n"},'
                '{"path":"formatter.py","content":"def format_result(value, unit):\\n    return f\\"{value} {unit}\\"\\n"},'
                '{"path":"config.json","content":"{\\"unit\\": \\"ms\\", \\"threshold\\": 10}\\n"}'
                ']}'
            )
        }
        scaffold_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.side_effect = [first_response, repair_response, scaffold_response]

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Create a tiny repo with main.py, math_utils.py, formatter.py, and config.json",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["actions"][0]["edit_mode"], "scaffold_files")
        self.assertIn("generated scaffold files", result["provider_reason"])

    def test_openai_tool_action_prefers_sanitized_workspace_target(self) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "output_text": '{"actions":[{"instruction":"Inspect the current tiny repo files before editing","edit_mode":"list_files","target_path":"shipyard"}]}'
        }
        mock_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.workspaces.WORKSPACE_ROOT",
            Path(tmpdir) / "workspaces",
        ):
            workspace = Path(tmpdir) / "workspaces" / "default"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "main.py").write_text("print('hi')\n", encoding="utf-8")

            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.post.return_value = mock_response

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "shipyard.action_planner.httpx.Client",
                return_value=mock_client,
            ):
                result = plan_actions(
                    {
                        "instruction": "Inspect the current tiny repo files before editing",
                        "proposal_mode": "openai",
                        "session_id": "demo",
                        "context": {"testing_mode": True},
                    }
                )

        self.assertEqual(result["actions"][0]["edit_mode"], "list_files")
        self.assertTrue(result["actions"][0]["target_path"].endswith("/workspaces/default"))
        self.assertTrue(result["is_valid"])


if __name__ == "__main__":
    unittest.main()
