#!/usr/bin/env python3
"""Unit test for NCCL bus bandwidth log extraction logic (P4-1)."""
import subprocess
import unittest

CANNED_NCCL_OUTPUT = """
#                                           out-of-place                       in-place          
#       size         count      type   op    time  algbw  busbw #wrong     time  algbw  busbw #wrong
#        (B)    (elements)                   (us) (GB/s) (GB/s)          (us) (GB/s) (GB/s)
  1073741824     268435456     float  sum  20480  52.43 124.50      0    20490  52.40 124.45      0
# Out of bounds values : 0 fatal 0 warn 0 total
# Avg bus bandwidth    : 124.5
"""

class TestNcclBandwidthParse(unittest.TestCase):
    def test_nccl_busbw_extraction(self):
        cmd = "grep -i '# Avg bus bandwidth' | tail -n 1 | awk -F':' '{print $2}' | xargs"
        proc = subprocess.run(
            cmd,
            shell=True,
            input=CANNED_NCCL_OUTPUT,
            text=True,
            capture_output=True,
            check=True,
        )
        val_str = proc.stdout.strip()
        val = float(val_str)
        self.assertEqual(val, 124.5, f"Expected 124.5, got {val}")
        self.assertGreaterEqual(val, 100.0, f"Expected >= 100.0, got {val}")

if __name__ == "__main__":
    unittest.main()
