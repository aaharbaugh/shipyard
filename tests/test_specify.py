from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shipyard.plan_feature import generate_spec_bundle
from shipyard.specify import classify_request


class SpecifyTests(unittest.TestCase):
    def test_classify_request_distinguishes_feature_from_direct_edit(self) -> None:
        self.assertEqual(classify_request('replace "a" with "b"').mode, "direct_edit")
        self.assertEqual(classify_request("build a feature to clean up the workbench flow").mode, "feature")

    def test_generate_spec_bundle_creates_files_for_feature_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "shipyard.plan_feature.DATA_ROOT",
            Path(tmpdir) / "data",
        ):
            bundle = generate_spec_bundle("demo", "build a feature to clean up the workbench flow")

            self.assertTrue(bundle["created"])
            self.assertTrue(Path(bundle["paths"]["spec"]).exists())
            self.assertTrue(Path(bundle["paths"]["architecture"]).exists())
            self.assertTrue(Path(bundle["paths"]["tasks"]).exists())


if __name__ == "__main__":
    unittest.main()
