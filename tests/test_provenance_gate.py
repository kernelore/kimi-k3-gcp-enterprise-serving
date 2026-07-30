#!/usr/bin/env python3
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.generate_comparison import validate_provenance, get_suite_timestamps


class TestProvenanceGate(unittest.TestCase):

    def test_get_suite_timestamps_from_grid(self):
        data = {
            "grid": {
                "suite_start_ts": "2026-07-28T10:00:00Z",
                "suite_end_ts": "2026-07-28T10:05:00Z"
            }
        }
        start, end = get_suite_timestamps(data)
        self.assertEqual(start, "2026-07-28T10:00:00Z")
        self.assertEqual(end, "2026-07-28T10:05:00Z")

    def test_validate_provenance_rejects_overlapping_intervals(self):
        sglang_data = {
            "standard": {
                "engine": "sglang",
                "benchmark_config": {
                    "suite_start_ts": "2026-07-28T10:00:00Z",
                    "suite_end_ts": "2026-07-28T10:10:00Z"
                },
                "metadata": {"engine_version": "0.4.3", "image": "sglang-blackwell"}
            },
            "massive": {
                "engine": "sglang",
                "benchmark_config": {
                    "suite_start_ts": "2026-07-28T10:05:00Z",
                    "suite_end_ts": "2026-07-28T10:15:00Z"
                },
                "metadata": {"engine_version": "0.4.3", "image": "sglang-blackwell"}
            }
        }
        with self.assertRaises((ValueError, SystemExit)):
            validate_provenance(sglang_data, None)


if __name__ == "__main__":
    unittest.main()
