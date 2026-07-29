#!/usr/bin/env python3
"""Unit test for NCCL bus bandwidth log extraction logic (P4-1)."""
import subprocess

CANNED_NCCL_OUTPUT = """
#                                           out-of-place                       in-place          
#       size         count      type   op    time  algbw  busbw #wrong     time  algbw  busbw #wrong
#        (B)    (elements)                   (us) (GB/s) (GB/s)          (us) (GB/s) (GB/s)
  1073741824     268435456     float  sum  20480  52.43 124.50      0    20490  52.40 124.45      0
# Out of bounds values : 0 fatal 0 warn 0 total
# Avg bus bandwidth    : 124.5
"""

def test_nccl_busbw_extraction():
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
    assert val == 124.5, f"Expected 124.5, got {val}"
    assert val >= 100.0, f"Expected >= 100.0, got {val}"
    print(f"NCCL bus bandwidth unit test passed: extracted {val} GB/s >= 100.0 GB/s")

if __name__ == "__main__":
    test_nccl_busbw_extraction()
