#!/usr/bin/env python3
"""
test_telemetry_sanitizer.py — Comprehensive Unit Tests for Telemetry Sanitizer
"""

import json
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

try:
    from google3.experimental.users.donina.KIMI3_GCPA4.benchmarks.telemetry_sanitizer import sanitize_telemetry, sanitize_string  # type: ignore
except ImportError:
    try:
        from experimental.users.donina.KIMI3_GCPA4.benchmarks.telemetry_sanitizer import sanitize_telemetry, sanitize_string  # type: ignore
    except ImportError:
        try:
            from benchmarks.telemetry_sanitizer import sanitize_telemetry, sanitize_string  # type: ignore
        except ImportError:
            from telemetry_sanitizer import sanitize_telemetry, sanitize_string  # type: ignore

class TestTelemetrySanitizer(unittest.TestCase):

    def test_litellm_and_master_key_redaction(self):
        data = {
            "api_key": "sk-1234567890abcdef12345",
            "master_key": "sk-kimi-k3-master-secret-key-change-me",
            "authorization": "Bearer sk-litellm-virtual-key-99999",
            "normal_field": "Response with key sk-abcdef1234567890 embedded inside text"
        }
        sanitized = sanitize_telemetry(data)
        self.assertEqual(sanitized["api_key"], "REDACTED")
        self.assertEqual(sanitized["master_key"], "REDACTED")
        self.assertEqual(sanitized["authorization"], "REDACTED")
        self.assertNotIn("sk-abcdef1234567890", sanitized["normal_field"])
        self.assertIn("REDACTED", sanitized["normal_field"])

    def test_unanchored_home_path_redaction(self):
        data = {
            "output": "/usr/local/google/home/donina/.gemini/jetski/worktrees/abc/def/benchmarks/results/standard_results.json",
            "message": "Logs stored at /home/user/project/logs.txt in system"
        }
        sanitized = sanitize_telemetry(data)
        self.assertEqual(sanitized["output"], "benchmarks/results/standard_results.json")
        self.assertNotIn("/home/user/", sanitized["message"])

    def test_nested_dictionary_and_list_preservation(self):
        data = {
            "execution_summary": {
                "total_requests": 16,
                "api_key": "sk-secretkey123456",
                "nested_level": {
                    "master_key": "sk-kimi-k3-master-secret-key-change-me",
                    "value": "keep_me"
                }
            },
            "results_list": [
                {"api_key": "sk-keyinlist123456"},
                "Path: /usr/local/google/home/testuser/file.py"
            ]
        }
        sanitized = sanitize_telemetry(data)
        self.assertEqual(sanitized["execution_summary"]["total_requests"], 16)
        self.assertEqual(sanitized["execution_summary"]["api_key"], "REDACTED")
        self.assertEqual(sanitized["execution_summary"]["nested_level"]["master_key"], "REDACTED")
        self.assertEqual(sanitized["execution_summary"]["nested_level"]["value"], "keep_me")
        self.assertEqual(sanitized["results_list"][0]["api_key"], "REDACTED")
        self.assertNotIn("/usr/local/google/home/testuser/", sanitized["results_list"][1])

    def test_json_string_metadata_sanitization(self):
        raw_meta = json.dumps({"api_key": "sk-insidejson12345", "engine": "sglang"})
        data = {"metadata": raw_meta}
        sanitized = sanitize_telemetry(data)
        meta_obj = json.loads(sanitized["metadata"])
        self.assertEqual(meta_obj["api_key"], "REDACTED")
        self.assertEqual(meta_obj["engine"], "sglang")

    def test_idempotency(self):
        data = {
            "api_key": "sk-1234567890abcdef12345",
            "nested": {"output": "/usr/local/google/home/donina/benchmarks/results/res.json"}
        }
        pass1 = sanitize_telemetry(data)
        pass2 = sanitize_telemetry(pass1)
        self.assertEqual(pass1, pass2)

if __name__ == "__main__":
    unittest.main()
