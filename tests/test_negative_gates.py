#!/usr/bin/env python3
# ==============================================================================
# test_negative_gates.py - Negative Tests for BQ Audit and Dependency Floors
# ==============================================================================

import unittest
from scripts.check_bq import evaluate_bq_results
from tests.check_dependency_floors import check_workflow_content

class TestNegativeGates(unittest.TestCase):
    def test_check_bq_zero_rows(self):
        """Assert evaluate_bq_results returns 1 (FAIL) when total_rows == 0."""
        self.assertEqual(evaluate_bq_results(True, 0, "test.table"), 1)

    def test_check_bq_positive_rows(self):
        """Assert evaluate_bq_results returns 0 (PASS) when total_rows > 0."""
        self.assertEqual(evaluate_bq_results(True, 10, "test.table"), 0)

    def test_check_dependency_floors_checkout_v4(self):
        """Assert check_workflow_content returns False for actions/checkout@v4 below floor v7."""
        content = "      - uses: actions/checkout@v4"
        self.assertFalse(check_workflow_content(content, "actions/checkout", 7))

    def test_check_dependency_floors_checkout_v7(self):
        """Assert check_workflow_content returns True for actions/checkout@v7 meeting floor v7."""
        content = "      - uses: actions/checkout@v7"
        self.assertTrue(check_workflow_content(content, "actions/checkout", 7))

if __name__ == "__main__":
    unittest.main()

