#!/usr/bin/env python3
"""Unit tests for check_bq.py result-evaluation logic (PRO-4a)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import check_bq

class TestCheckBq(unittest.TestCase):
    def test_zero_rows_fails(self):
        code = check_bq.evaluate_bq_results(success=True, total_rows=0, table_ref="test_table")
        self.assertEqual(code, 1, "total_rows == 0 must return exit code 1 (FAIL)")

    def test_positive_rows_passes(self):
        code = check_bq.evaluate_bq_results(success=True, total_rows=5, table_ref="test_table")
        self.assertEqual(code, 0, "total_rows > 0 must return exit code 0 (PASS)")

    def test_failure_fails(self):
        code = check_bq.evaluate_bq_results(success=False, total_rows=0, table_ref="test_table")
        self.assertEqual(code, 1, "success=False must return exit code 1 (FAIL)")

if __name__ == "__main__":
    unittest.main()
