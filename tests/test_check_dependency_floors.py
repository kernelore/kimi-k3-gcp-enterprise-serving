#!/usr/bin/env python3
"""Unit tests for check_dependency_floors.py workflow-content evaluation logic (PRO-4b)."""
import unittest
import check_dependency_floors

class TestCheckDependencyFloors(unittest.TestCase):
    def test_checkout_v4_fails_floor(self):
        content = "    steps:\n      - uses: actions/checkout@v4\n"
        self.assertFalse(
            check_dependency_floors.check_workflow_content(content, "actions/checkout", 7),
            "actions/checkout@v4 must return False for floor 7"
        )

    def test_checkout_v7_passes_floor(self):
        content = "    steps:\n      - uses: actions/checkout@v7\n"
        self.assertTrue(
            check_dependency_floors.check_workflow_content(content, "actions/checkout", 7),
            "actions/checkout@v7 must return True for floor 7"
        )

if __name__ == "__main__":
    unittest.main()
