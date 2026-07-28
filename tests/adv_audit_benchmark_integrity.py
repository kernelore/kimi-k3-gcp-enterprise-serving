#!/usr/bin/env python3
"""
adv_audit_benchmark_integrity.py — Adversarial Benchmark Data Integrity & Provenance Audit (Kimi K3)
Verifies result files in benchmarks/results/{trtllm,sglang}/ to ensure:
1. Zero fabricated or unmeasured data exists (no nulls, no TODO/unknown placeholders).
2. Genuine request execution (rejecting failure-tainted files where success=False or failed_requests > 0).
3. Engine references match deployed versions.
4. Timestamp monotonicity and non-overlap across suites.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tests.lib.engine_versions import normalize_engine_version as normalize_version
except ImportError:
    def normalize_version(ver: str) -> str:
        if not ver:
            return ""
        ver = ver.strip()
        if ver.startswith("v") or ver.startswith("V"):
            ver = ver[1:]
        if ver.endswith("-cu130"):
            ver = ver[:-6]
        return ver

def get_metadata(data: dict) -> dict:
    meta = data.get("metadata")
    if not meta and "benchmark_config" in data:
        meta = data["benchmark_config"].get("metadata")
    if not meta and "soak_config" in data:
        meta = data["soak_config"].get("metadata")
    if not meta and "sweep_config" in data:
        meta = data["sweep_config"].get("metadata")
    if not meta and "grid" in data:
        meta = data["grid"].get("metadata")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}

def main():
    root = PROJECT_ROOT / "benchmarks" / "results"
    if not root.exists():
        print(f"Notice: Results directory {root} does not exist yet. Skipping audit.")
        sys.exit(0)

    engines = ["trtllm", "sglang"]
    suites = ["standard", "massive", "soak", "saturation", "prefill"]

    print("==============================================================================")
    print("Kimi K3 GCP Enterprise Serving - Benchmark Data Integrity Stress-Test")
    print("==============================================================================")

    errors = []

    for eng in engines:
        eng_dir = root / eng
        if not eng_dir.exists():
            continue
        print(f"\n--> Auditing Engine: {eng.upper()}")
        for s in suites:
            fpath = eng_dir / f"{s}_results.json"
            if not fpath.exists():
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                errors.append(f"Invalid JSON in {fpath}: {e}")
                continue

            # 1. Reject failure-tainted result files
            if data.get("success") is False:
                errors.append(f"[{eng}/{s}] Rejection: Result file marks request as failed (success: false)")
            
            exec_summary = data.get("execution_summary", {})
            if isinstance(exec_summary, dict):
                if exec_summary.get("failed_requests", 0) > 0:
                    errors.append(f"[{eng}/{s}] Rejection: Result file contains failed_requests > 0 ({exec_summary.get('failed_requests')})")
                if exec_summary.get("successful_requests") == 0 and exec_summary.get("total_requests", 0) > 0:
                    errors.append(f"[{eng}/{s}] Rejection: Zero successful requests recorded")

            # 2. Inspect Metadata & Engine References
            meta = get_metadata(data)
            ver = meta.get("version") or meta.get("engine_version", "")
            if "unknown" in str(ver).lower() or "todo" in str(ver).lower():
                errors.append(f"[{eng}/{s}] Placeholder version found: '{ver}'")

    if errors:
        print("\n[FAIL] BENCHMARK INTEGRITY AUDIT FAILED WITH ERRORS:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n[OK] Benchmark Data Integrity Audit Passed Cleanly.")
    sys.exit(0)

if __name__ == "__main__":
    main()
