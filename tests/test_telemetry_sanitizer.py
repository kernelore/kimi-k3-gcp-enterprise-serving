#!/usr/bin/env python3
import unittest
import json
import sys
import os
import re
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.telemetry_sanitizer import sanitize_telemetry

# Assembled at runtime so the scan patterns never appear as literals in this file.
FAKE_KEY   = "sk-" + "kimi-k3-" + "0123456789abcdef" * 2
HF_TOKEN   = "hf_" + "0123456789abcdef" * 2
USER_HOME  = "/" + "home/testuser"
GOOG_HOME  = "/usr/local/google/" + "home/testuser"

class TestTelemetrySanitizer(unittest.TestCase):
    def test_hf_token_and_master_key_redaction(self):
        raw_data = {
            "master_key": FAKE_KEY,
            "authorization": "Bearer " + FAKE_KEY,
            "token": HF_TOKEN,
            "hf_token": HF_TOKEN
        }
        sanitized = sanitize_telemetry(raw_data)
        self.assertEqual(sanitized["master_key"], "REDACTED")
        self.assertEqual(sanitized["authorization"], "REDACTED")
        self.assertEqual(sanitized["token"], "REDACTED")
        self.assertEqual(sanitized["hf_token"], "REDACTED")

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


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Every harness that serialises a result artifact. The sanitizer being correct
# is worth nothing if a writer never calls it, and that is not hypothetical:
# run_prefix_reuse_bench and kv_accuracy_gate both shipped without the call and
# both wrote a real project ID to disk during the 2026-08-03 measurement window.
# A pre-publish grep caught it, which is to say a human caught it.
RESULT_WRITERS = (
    "benchmarks/run_realistic_sweep_kimi_k3.py",
    "benchmarks/run_prefix_reuse_bench.py",
    "benchmarks/kv_accuracy_gate.py",
)


class TestResultArtifactsArePublishable(unittest.TestCase):
    def test_every_result_writer_calls_the_sanitizer(self):
        for rel in RESULT_WRITERS:
            path = os.path.join(REPO_ROOT, rel)
            with open(path, encoding="utf-8") as handle:
                src = handle.read()
            self.assertIn(
                "sanitize_telemetry", src,
                f"{rel} writes a result artifact without importing the sanitizer",
            )
            # The import alone proves nothing; require a call as well.
            self.assertRegex(
                src, r"sanitize_telemetry\(",
                f"{rel} imports the sanitizer but never calls it",
            )

    def test_committed_result_artifacts_carry_no_project_id(self):
        """The artifacts, not the code path -- this is what actually ships.

        Scoped to git-tracked files on purpose. benchmarks/results/windows/ is
        the raw capture directory and is gitignored; it holds unsanitised output
        by design, and asserting over it would fail on any machine that has run
        a measurement window. What must be clean is what a clone can see.
        """
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "benchmarks/results"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        checked = 0
        for rel in tracked:
            if not rel.endswith(".json"):
                continue
            path = os.path.join(REPO_ROOT, rel)
            with open(path, encoding="utf-8") as handle:
                body = handle.read()
            checked += 1
            for ref in re.findall(r"docker\.pkg\.dev/([^/\"]+)", body):
                self.assertEqual(
                    ref, "YOUR_PROJECT_ID",
                    f"{rel} names a real registry project: {ref}",
                )
            self.assertNotRegex(
                body, r"sk-[a-zA-Z0-9_-]{12,}", f"{rel} contains an API key")
            self.assertNotRegex(
                body, r"hf_[A-Za-z0-9]{32,}", f"{rel} contains an HF token")
            self.assertNotIn(
                "/home/", body, f"{rel} leaks an absolute home directory")
        self.assertGreater(checked, 0, "expected at least one committed result artifact")


if __name__ == "__main__":
    unittest.main()
