from __future__ import annotations

import unittest
from unittest.mock import patch

from shipyard.langsmith_config import build_langgraph_config, langsmith_enabled


class LangSmithConfigTests(unittest.TestCase):
    def test_langsmith_disabled_without_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(langsmith_enabled())
            config = build_langgraph_config("demo")

        self.assertEqual(config["configurable"]["thread_id"], "demo")
        self.assertNotIn("run_name", config)

    def test_langsmith_enabled_adds_metadata(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "LANGSMITH_API_KEY": "test-key",
                "LANGSMITH_TRACING": "true",
            },
            clear=True,
        ):
            self.assertTrue(langsmith_enabled())
            config = build_langgraph_config(
                "demo",
                instruction="replace total with totality",
                step_index=1,
                step_count=2,
            )

        self.assertEqual(config["configurable"]["thread_id"], "demo")
        self.assertEqual(config["run_name"], "shipyard.run_step")
        self.assertIn("shipyard", config["tags"])
        self.assertEqual(config["metadata"]["instruction"], "replace total with totality")
        self.assertEqual(config["metadata"]["step_index"], 1)
        self.assertEqual(config["metadata"]["step_count"], 2)


if __name__ == "__main__":
    unittest.main()
