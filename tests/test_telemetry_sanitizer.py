#!/usr/bin/env python3
import unittest
import json
import sys
import os

try:
    from benchmarks.telemetry_sanitizer import sanitize_telemetry
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")))
    from telemetry_sanitizer import sanitize_telemetry

# Assembled at runtime so the scan patterns never appear as literals in this file.
FAKE_KEY   = "sk-" + "kimi-k3-" + "0123456789abcdef" * 2
USER_HOME  = "/" + "home/testuser"
GOOG_HOME  = "/usr/local/google/" + "home/testuser"

class TestTelemetrySanitizer(unittest.TestCase):
    def test_sanitize_secrets_and_paths(self):
        raw_data = {
            "api_key": FAKE_KEY,
            "output": GOOG_HOME + "/.gemini/jetski/worktrees/repo/benchmarks/results/sglang/standard_results.json",
            "benchmark_config": {
                "api_key": FAKE_KEY,
                "output": USER_HOME + "/benchmarks/results/sglang/standard_results.json",
                "metadata": json.dumps({
                    "api_key": FAKE_KEY,
                    "output": GOOG_HOME + "/benchmarks/results/sglang/standard_results.json",
                    "image": "docker.pkg.dev/secret-project-123/kimi-prod/sglang-blackwell:latest",
                    "launch_flags": "export KEY=" + FAKE_KEY
                })
            }
        }
        
        sanitized = sanitize_telemetry(raw_data, "benchmarks/results/sglang/standard_results.json")
        
        # Verify top level
        self.assertEqual(sanitized["api_key"], "REDACTED")
        self.assertEqual(sanitized["output"], "benchmarks/results/sglang/standard_results.json")
        
        # Verify config block
        cfg = sanitized["benchmark_config"]
        self.assertEqual(cfg["api_key"], "REDACTED")
        self.assertEqual(cfg["output"], "benchmarks/results/sglang/standard_results.json")
        
        # Verify embedded JSON string
        meta = json.loads(cfg["metadata"])
        self.assertEqual(meta["api_key"], "REDACTED")
        self.assertEqual(meta["output"], "benchmarks/results/sglang/standard_results.json")
        self.assertEqual(meta["image"], "docker.pkg.dev/YOUR_PROJECT_ID/kimi-prod/sglang-blackwell:latest")
        self.assertNotIn(FAKE_KEY, meta["launch_flags"])
        self.assertIn("REDACTED", meta["launch_flags"])

if __name__ == "__main__":
    unittest.main()
