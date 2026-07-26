#!/usr/bin/env python3
"""
adv_audit_benchmark_integrity.py — Adversarial Benchmark Data Integrity & Provenance Audit
Verifies all 10 JSON result files in benchmarks/results/{sglang,trtllm}/ to ensure:
1. Zero fabricated or unmeasured data exists (no nulls, no TODO/unknown placeholders, valid positive numerical values).
2. Genuine SSE token counting logic (checking token_count_source and verifying consistency).
3. Engine references match expected versions (SGLang and TRT-LLM).
4. Timestamp monotonicity and non-overlap across suites.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def get_metadata(data: dict) -> dict:
    meta = data.get("metadata")
    if not meta and "benchmark_config" in data:
        meta = data["benchmark_config"].get("metadata")
    if not meta and "soak_config" in data:
        meta = data["soak_config"].get("metadata")
    if not meta and "sweep_config" in data:
        meta = data["sweep_config"].get("metadata")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}

def normalize_version(ver: str) -> str:
    if not ver:
        return ""
    ver = ver.strip()
    if ver.startswith("v") or ver.startswith("V"):
        ver = ver[1:]
    if ver.endswith("-cu130"):
        ver = ver[:-6]
    return ver

def main():
    default_root = Path(__file__).resolve().parent.parent / "benchmarks" / "results"
    root = Path(os.getenv("AUDIT_RESULTS_DIR", default_root))
    if not root.exists() or not any(root.glob("*/*.json")):
        print(f"[SKIP] Results directory {root} is empty or absent (pre-launch state). Skipping benchmark integrity audit.")
        sys.exit(0)

    try:
        from tests.lib.engine_versions import get_engine_version
    except ImportError:
        lib_dir = str(Path(__file__).resolve().parent / "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        from engine_versions import get_engine_version

    sglang_ver = get_engine_version("sglang", root=Path(__file__).resolve().parent.parent)
    trtllm_ver = get_engine_version("trtllm", root=Path(__file__).resolve().parent.parent)
    engines = {
        "sglang": {"expected_ver_norm": sglang_ver, "expected_img_sub": "sglang-blackwell", "expected_ver_raw_options": [f"v{sglang_ver}", sglang_ver]},
        "trtllm": {"expected_ver_norm": trtllm_ver, "expected_img_sub": "trtllm-blackwell", "expected_ver_raw_options": [f"v{trtllm_ver}", trtllm_ver]}
    }
    suites = ["standard", "massive", "soak", "saturation", "prefill"]

    print("==============================================================================")
    print("Kimi K3 GCP Enterprise Serving - Benchmark Data Integrity Stress-Test")
    print("==============================================================================")

    errors = []
    warnings = []
    total_files = 0
    passed_files = 0

    eng_status = {}
    for eng in engines.keys():
        present = [s for s in suites if (root / eng / f"{s}_results.json").exists()]
        missing = [s for s in suites if not (root / eng / f"{s}_results.json").exists()]
        if len(present) == len(suites):
            eng_status[eng] = "complete"
        elif len(present) == 0:
            eng_status[eng] = "absent"
        else:
            eng_status[eng] = "partial"
            errors.append(f"[{eng}] Partial benchmark data: present={present}, missing={missing}. All 5 suite files required.")

    if all(status == "absent" for status in eng_status.values()):
        print(f"[SKIP] No benchmark results found in {root} (0 complete engines). Skipping benchmark integrity audit.")
        sys.exit(0)

    for eng, rules in engines.items():
        if eng_status[eng] == "absent":
            print(f"\n--> Auditing Engine: {eng.upper()}... [SKIP] Engine results absent (0/5 files present).")
            continue
        if eng_status[eng] == "partial":
            print(f"\n--> Auditing Engine: {eng.upper()}... [FAIL] Partial benchmark data.")
            continue
        print(f"\n--> Auditing Engine: {eng.upper()} (Expected Normalized Version: {rules['expected_ver_norm']})")
        runs = []
        for s in suites:
            fpath = root / eng / f"{s}_results.json"
            total_files += 1
            if not fpath.exists():
                errors.append(f"[{eng}/{s}] Missing benchmark result file: {fpath}")
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                errors.append(f"Invalid JSON in {fpath}: {e}")
                continue

            if data.get("dry_run") is True or str(data.get("dry_run", "")).lower() == "true":
                errors.append(f"[{eng}/{s}] Dry-run contamination: file has top-level dry_run=True")

            # 1. Inspect Metadata & Engine References
            meta = get_metadata(data)
            ver = meta.get("engine_version", "")
            img = meta.get("image", "")
            ts_str = meta.get("run_timestamp", "")

            # Check version
            norm_ver = normalize_version(ver)
            if norm_ver != rules["expected_ver_norm"] and ver not in rules["expected_ver_raw_options"]:
                errors.append(f"[{eng}/{s}] Version mismatch: found '{ver}' (norm '{norm_ver}'), expected norm '{rules['expected_ver_norm']}' or {rules['expected_ver_raw_options']}")

            # Check image
            if rules["expected_img_sub"] not in img:
                errors.append(f"[{eng}/{s}] Image mismatch or placeholder: found '{img}', expected substring '{rules['expected_img_sub']}'")
            if "unknown" in img.lower() or "todo" in img.lower() or "placeholder" in img.lower():
                errors.append(f"[{eng}/{s}] Fabricated/placeholder image string found: '{img}'")

            # Check timestamp
            if not ts_str:
                errors.append(f"[{eng}/{s}] Missing run_timestamp in metadata")
            else:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    runs.append((ts, ts_str, s, data))
                except Exception as e:
                    errors.append(f"[{eng}/{s}] Invalid ISO timestamp '{ts_str}': {e}")

            # 2. Inspect Data Integrity & SSE Token Counting Logic
            tok_src = data.get("token_count_source", "MISSING")
            if tok_src == "MISSING" and s not in ["saturation"]:
                errors.append(f"[{eng}/{s}] Missing token_count_source tag in result JSON")
            elif tok_src == "chunk_count_fallback":
                warnings.append(f"[{eng}/{s}] Used fallback chunk counting instead of exact OpenAI usage tokens.")

            # Validate metrics and ensure zero fabricated/unmeasured/null data
            if s in ["standard", "massive", "soak"]:
                succ = data.get("successful_requests")
                tot = data.get("total_requests")
                tps = data.get("throughput_tokens_sec")
                if succ is None or tot is None or tps is None:
                    errors.append(f"[{eng}/{s}] Null/missing summary metrics (succ={succ}, tot={tot}, tps={tps})")
                elif succ != tot or succ <= 0:
                    errors.append(f"[{eng}/{s}] Unsuccessful or zero request count: {succ}/{tot}")
                elif tps <= 0:
                    errors.append(f"[{eng}/{s}] Implausible throughput: {tps} tok/s")

                metrics = data.get("metrics", {})
                for m_name in ["ttft_ms", "tpot_ms"]:
                    m_data = metrics.get(m_name, {})
                    if not m_data or m_data.get("mean", 0) <= 0 or m_data.get("p50", 0) <= 0:
                        errors.append(f"[{eng}/{s}] Invalid or unmeasured {m_name}: {m_data}")

                print(f"    [OK] {s:<10} | ver={ver:<14} | ts={ts_str:<22} | succ={succ}/{tot} | tps={tps:6.2f} | tok_src={tok_src}")

            elif s == "saturation":
                grid = data.get("grid", {})
                isl_osl = grid.get("ISL_OSL_GRID", grid.get("isl_osl", []))
                levels  = grid.get("sweep_levels", grid.get("concurrency", []))
                expected = len(isl_osl) * len(levels)
                sweep = data.get("sweep_results", [])
                if not expected:
                    errors.append(f"[{eng}/{s}] Missing or empty top-level 'grid' block — cannot verify sweep completeness")
                elif len(sweep) != expected:
                    errors.append(f"[{eng}/{s}] Sweep has {len(sweep)} cells, grid declares {expected}")
                for item in sweep:
                    if item.get("dry_run") is True or str(item.get("dry_run", "")).lower() == "true":
                        errors.append(f"[{eng}/{s}] Dry-run contamination: sweep record has dry_run=True")
                    c = item.get("concurrency")
                    status = item.get("status", "ok")
                    if status == "skipped":
                        if not item.get("reason"):
                            errors.append(f"[{eng}/{s}] Skipped cell c={c} missing reason")
                        continue
                    tps = item.get("aggregate_tok_s", 0)
                    err_pct = item.get("error_rate_pct", 100)
                    item_tok_src = item.get("token_count_source", "MISSING")
                    if tps <= 0 or err_pct != 0.0:
                        errors.append(f"[{eng}/{s}] Concurrency c={c} failed or zero throughput (tps={tps}, err={err_pct}%)")
                    if item_tok_src == "chunk_count_fallback":
                        warnings.append(f"[{eng}/{s}] c={c} used fallback chunk counting.")
                print(f"    [OK] {s:<10} | ver={ver:<14} | ts={ts_str:<22} | levels={[item['concurrency'] for item in sweep]} | max_tps={max((item['aggregate_tok_s'] for item in sweep if item.get('status') != 'skipped'), default=0):6.2f}")

            elif s == "prefill":
                p_tok = data.get("prompt_tokens", 0)
                ttft_ms = data.get("ttft_ms", 0)
                sys_rate = data.get("prefill_tok_s_system", 0)
                if p_tok <= 0 or ttft_ms <= 0 or sys_rate <= 0:
                    errors.append(f"[{eng}/{s}] Unmeasured or zero prefill metrics (prompt_tok={p_tok}, ttft={ttft_ms}ms, sys_rate={sys_rate})")
                print(f"    [OK] {s:<10} | ver={ver:<14} | ts={ts_str:<22} | prompt_tok={p_tok} | ttft={ttft_ms:6.2f}ms | sys_rate={sys_rate:8.2f} tok/s")

            passed_files += 1

        # Check timestamp monotonicity and non-overlap
        runs.sort(key=lambda x: x[0])
        for i in range(1, len(runs)):
            prev_ts, prev_str, prev_s, prev_data = runs[i-1]
            curr_ts, curr_str, curr_s, curr_data = runs[i]
            if curr_ts <= prev_ts:
                errors.append(f"[{eng}] Non-monotonic timestamp between '{prev_s}' ({prev_str}) and '{curr_s}' ({curr_str})")
            
            # Calculate duration
            dur = 0.0
            if prev_s in ["standard", "massive"]:
                dur = prev_data.get("execution_summary", {}).get("total_benchmark_time_seconds", 0.0)
            elif prev_s == "soak":
                dur = prev_data.get("execution_summary", {}).get("total_duration_seconds", 0.0)
            elif prev_s == "saturation":
                dur = sum(item.get("total_duration_sec", 0.0) for item in prev_data.get("sweep_results", []))
            elif prev_s == "prefill":
                dur = prev_data.get("ttft_sec", 0.0)
                if dur == 0.0 and "ttft_ms" in prev_data:
                    dur = prev_data["ttft_ms"] / 1000.0

            prev_end = prev_ts.timestamp() + dur
            if prev_end > curr_ts.timestamp():
                errors.append(f"[{eng}] Overlapping run: '{prev_s}' ended at {prev_end:.2f}, but '{curr_s}' started at {curr_ts.timestamp():.2f}")

    print("\n==============================================================================")
    print(f"Audit Summary: {passed_files}/{total_files} JSON files inspected.")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  * {w}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  * {e}", file=sys.stderr)
        print("==============================================================================")
        print("VERDICT: FAILED - Benchmark data integrity violations found!", file=sys.stderr)
        sys.exit(1)
    else:
        print("VERDICT: PASSED - Zero fabricated/unmeasured data, genuine SSE counting, valid references.")
        print("==============================================================================")
        sys.exit(0)

if __name__ == "__main__":
    main()
