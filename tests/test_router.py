from __future__ import annotations

import unittest

from shipyard.router import route_message


class RouterTests(unittest.TestCase):
    def test_route_message_detects_correction_and_high_urgency(self) -> None:
        result = route_message("Actually fix that now")

        self.assertEqual(result["relation_to_previous"]["label"], "correction")
        self.assertEqual(result["urgency"]["label"], "high")
        self.assertEqual(result["actionability"]["label"], "act")
        self.assertIn("priority_score", result)


if __name__ == "__main__":
    unittest.main()
