from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from shipyard.action_planner import plan_actions


class ActionPlannerTests(unittest.TestCase):
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
        self.assertEqual(result["actions"][0]["edit_mode"], "append")


if __name__ == "__main__":
    unittest.main()
