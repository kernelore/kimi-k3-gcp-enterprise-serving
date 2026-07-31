#!/usr/bin/env python3
"""Unit test for NCCL bus bandwidth log extraction logic (P4-1) and PRO-1 gate hardening."""
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

def eval_nccl_gate(log_text):
    cmd = """
    MARKER_LINE=$(echo "$LOG_INPUT" | grep -E "^NCCL_GATE_RESULT " | tail -n 1 || true)
    if [ -z "$MARKER_LINE" ]; then exit 1; fi
    if echo "$MARKER_LINE" | grep -E -q "^NCCL_GATE_RESULT fail"; then exit 1; fi
    BUSBW_VAL=$(echo "$MARKER_LINE" | awk -F'busbw_gbps=' '{print $2}' | awk '{print $1}' | xargs || true)
    if [ -z "$BUSBW_VAL" ]; then exit 1; fi
    if ! echo "$BUSBW_VAL" | grep -E -q '^[0-9]+(\\.[0-9]+)?$'; then exit 1; fi
    awk -v val="$BUSBW_VAL" 'BEGIN {exit !(val >= 100.0)}' 2>/dev/null
    """
    proc = subprocess.run(
        cmd,
        shell=True,
        env={"LOG_INPUT": log_text},
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0

def eval_parity_gate(log_text):
    cmd = """
    PARITY_MARKER=$(echo "$LOG_INPUT" | grep -E "^NCCL_PARITY_RESULT " | tail -n 1 || true)
    if [ -z "$PARITY_MARKER" ]; then exit 1; fi
    if ! echo "$PARITY_MARKER" | grep -E -q "^NCCL_PARITY_RESULT pass$"; then exit 1; fi
    """
    proc = subprocess.run(
        cmd,
        shell=True,
        env={"LOG_INPUT": log_text},
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0

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

    def test_no_marker(self):
        self.assertFalse(eval_nccl_gate("Random log output without marker\n"))

    def test_fail_marker(self):
        self.assertFalse(eval_nccl_gate("NCCL_GATE_RESULT fail\n"))

    def test_empty_value(self):
        self.assertFalse(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=\n"))

    def test_non_numeric_value(self):
        self.assertFalse(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=100GB/s\n"))
        self.assertFalse(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=garbage\n"))

    def test_below_floor_99_9(self):
        self.assertFalse(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=99.9\n"))

    def test_at_floor_100_0(self):
        self.assertTrue(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=100.0\n"))

    def test_above_floor_124_5(self):
        self.assertTrue(eval_nccl_gate("NCCL_GATE_RESULT busbw_gbps=124.5\n"))

    def test_parity_pass_marker(self):
        self.assertTrue(eval_parity_gate("NCCL_PARITY_RESULT pass\n"))

    def test_parity_fail_marker(self):
        self.assertFalse(eval_parity_gate("NCCL_PARITY_RESULT fail\n"))

    def test_parity_no_marker(self):
        self.assertFalse(eval_parity_gate("Some parity check log without marker\n"))

    def test_parity_invalid_marker(self):
        self.assertFalse(eval_parity_gate("NCCL_PARITY_RESULT unknown\n"))

if __name__ == "__main__":
    unittest.main()
