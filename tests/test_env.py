from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.env import load_dotenv


class EnvLoaderTests(unittest.TestCase):
    def test_load_dotenv_sets_values_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                'OPENAI_API_KEY="test-key"\nOPENAI_MODEL=gpt-5.4-mini\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                loaded = load_dotenv(env_path)

            self.assertEqual(loaded["OPENAI_API_KEY"], "test-key")
            self.assertEqual(loaded["OPENAI_MODEL"], "gpt-5.4-mini")

    def test_load_dotenv_does_not_override_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                loaded = load_dotenv(env_path)
                current = os.environ["OPENAI_API_KEY"]

            self.assertEqual(loaded, {})
            self.assertEqual(current, "shell-key")


if __name__ == "__main__":
    unittest.main()
