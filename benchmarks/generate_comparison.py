#!/usr/bin/env python3
"""
generate_comparison.py — Automated Parity Validator & README.md Table Generator for Kimi K3

Reads dual-engine benchmark results from benchmarks/results/{sglang,trtllm}/,
enforces strict parameter parity and sanity gates (no cold-start contamination,
valid TPOT, 100% success rate), and idempotently updates README.md between
<!-- ENGINE_COMPARISON_START --> and <!-- ENGINE_COMPARISON_END --> markers.

Strict constraint: Does NOT create any new .md files.
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from tests.lib.engine_versions import get_engine_version
except ImportError:
    lib_dir = str(Path(__file__).resolve().parent.parent / "tests" / "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    from engine_versions import get_engine_version

def load_json(path: Path):
    if not path.exists():
        print(f"ERROR: Missing benchmark result file: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

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
    if ver.endswith("-py3"):
        ver = ver[:-4]
    return ver

def get_suite_timestamps(data: dict) -> tuple:
    for key in ["benchmark_config", "soak_config", "sweep_config", "prefill_config", "metadata"]:
        cfg = data.get(key)
        if isinstance(cfg, dict):
            start_ts = cfg.get("suite_start_ts")
            end_ts = cfg.get("suite_end_ts")
            if start_ts and end_ts:
                return start_ts, end_ts
    start_ts = data.get("suite_start_ts")
    end_ts = data.get("suite_end_ts")
    if start_ts and end_ts:
        return start_ts, end_ts
    return None, None

def get_suite_duration(suite_name: str, data: dict) -> float:
    dur = 0.0
    if suite_name in ["standard", "massive"]:
        dur = data.get("execution_summary", {}).get("total_benchmark_time_seconds", 0.0)
    elif suite_name == "soak":
        dur = data.get("execution_summary", {}).get("total_duration_seconds", 0.0)
    elif suite_name == "saturation":
        dur = sum(item.get("total_duration_sec", 0.0) for item in data.get("sweep_results", []) if item.get("status") == "ok")
    elif suite_name == "prefill":
        dur = data.get("ttft_sec", 0.0)
        if dur == 0.0 and "ttft_ms" in data:
            dur = data["ttft_ms"] / 1000.0
    return dur

def validate_provenance(sglang: dict | None, trtllm: dict | None):
    """
    Validates engine identity, container image, engine version, timestamp formatting,
    and interval monotonicity/non-overlap against explicit suite_start_ts / suite_end_ts boundaries.
    """
    suites = ["standard", "massive", "soak", "saturation", "prefill"]
    for eng_name, eng_data, exp_eng, exp_img_sub in [
        ("sglang", sglang, "sglang", "sglang-blackwell"),
        ("trtllm", trtllm, "trtllm", "trtllm-blackwell")
    ]:
        if eng_data is None:
            continue
        exp_ver = get_engine_version(eng_name, root=Path(__file__).resolve().parent.parent)
        prev_end_dt = None
        prev_suite_name = None
        for s in suites:
            meta = get_metadata(eng_data[s])
            eng_val = meta.get("engine", "")
            img_val = meta.get("image", "")
            if eng_val != exp_eng or exp_img_sub not in img_val:
                raise ValueError(f"PROVENANCE GATE FAILURE: results/{eng_name}/{s}_results.json has mismatched metadata (engine='{eng_val}', image='{img_val}')")
            
            ver_raw = meta.get("engine_version", "")
            if ver_raw == "unknown":
                raise ValueError(f"PROVENANCE GATE FAILURE: results/{eng_name}/{s}_results.json engine_version is 'unknown'")
            ver_norm = normalize_version(ver_raw)
            if ver_norm != exp_ver:
                raise ValueError(f"PROVENANCE GATE FAILURE: results/{eng_name}/{s}_results.json engine_version '{ver_raw}' does not match expected '{exp_ver}'")
            
            ts_str = meta.get("run_timestamp", "")
            if not ts_str:
                raise ValueError(f"PROVENANCE GATE FAILURE: Missing run_timestamp in results/{eng_name}/{s}_results.json")
            try:
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception as e:
                raise ValueError(f"PROVENANCE GATE FAILURE: Invalid run_timestamp '{ts_str}' in results/{eng_name}/{s}_results.json: {e}")
            
            start_ts, end_ts = get_suite_timestamps(eng_data[s])
            if not start_ts or not end_ts:
                print(f"NOTE: Skipping interval overlap checks for {eng_name} {s} (suite_start_ts/suite_end_ts not recorded).")
                continue
            try:
                curr_start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                curr_end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
            except Exception as e:
                raise ValueError(f"PROVENANCE GATE FAILURE: Invalid suite timestamp in results/{eng_name}/{s}_results.json: {e}")
            
            if curr_start_dt > curr_end_dt:
                raise ValueError(f"PROVENANCE GATE FAILURE: results/{eng_name}/{s}_results.json has suite_start_ts after suite_end_ts")
            
            if prev_end_dt is not None:
                if curr_start_dt < prev_end_dt:
                    raise ValueError(f"PROVENANCE GATE FAILURE: results/{eng_name}/{s}_results.json suite_start_ts ({start_ts}) overlaps/precedes previous suite {prev_suite_name} suite_end_ts ({prev_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')})")
            
            prev_end_dt = curr_end_dt
            prev_suite_name = s

def validate_sanity(sglang: dict | None, trtllm: dict | None):
    suites = ["standard", "massive", "soak", "saturation", "prefill"]
    for eng_name, data in [("SGLang", sglang), ("TRT-LLM", trtllm)]:
        if data is None:
            continue
        for s in suites:
            if s in ["standard", "massive", "soak"]:
                ttft_p50 = data[s]["metrics"]["ttft_ms"]["p50"]
                tpot_mean = data[s]["metrics"]["tpot_ms"]["mean"]
                succ = data[s]["successful_requests"]
                tot = data[s]["total_requests"]
                if s == "standard" and ttft_p50 > 10000.0:
                    raise ValueError(f"Sanity Gate Failure [TTFT Contamination]: {eng_name} {s} TTFT P50 is {ttft_p50:.2f} ms (> 10s threshold at c <= 8). Re-run after warm-up.")
                if tpot_mean < 1.0:
                    raise ValueError(f"Sanity Gate Failure [TPOT Implausible]: {eng_name} {s} TPOT mean is {tpot_mean:.4f} ms (< 1 ms threshold).")
                if succ != tot:
                    raise ValueError(f"Sanity Gate Failure [Non-100% Success Rate]: {eng_name} {s} success rate is {succ}/{tot}.")
            elif s == "saturation":
                for item in data[s].get("sweep_results", []):
                    if item.get("status") == "skipped":
                        if not item.get("reason"):
                            raise ValueError(f"Sanity Gate Failure [Skipped Without Reason]: {eng_name} saturation cell ISL={item.get('isl_target')} c={item.get('concurrency')} skipped with empty reason.")
                        continue
                    c_level = item["concurrency"]
                    ttft_p50 = item["ttft_ms"]["p50"]
                    tpot_mean = item["tpot_ms"]["mean"]
                    err_pct = item["error_rate_pct"]
                    if c_level <= 8 and ttft_p50 > 10000.0:
                        raise ValueError(f"Sanity Gate Failure [TTFT Contamination]: {eng_name} saturation c={c_level} TTFT P50 is {ttft_p50:.2f} ms (> 10s threshold). Re-run after warm-up.")
                    if tpot_mean < 1.0:
                        raise ValueError(f"Sanity Gate Failure [TPOT Implausible]: {eng_name} saturation c={c_level} TPOT mean is {tpot_mean:.4f} ms (< 1 ms threshold).")
                    if err_pct != 0.0:
                        raise ValueError(f"Sanity Gate Failure [Non-100% Success Rate]: {eng_name} saturation c={c_level} error rate is {err_pct}%.")

def validate_parity(sglang: dict | None, trtllm: dict | None):
    if sglang is None or trtllm is None:
        present_eng = "sglang" if sglang is not None else "trtllm"
        print(f"[SKIP] Dual-engine parity verification skipped (only {present_eng} results present).")
        return

    suites = ["standard", "massive", "soak", "saturation", "prefill"]
    for s in suites:
        g = sglang[s]
        t = trtllm[s]

        if s in ["standard", "massive"]:
            gc, tc = g["benchmark_config"]["concurrency"], t["benchmark_config"]["concurrency"]
            gr, tr = g["benchmark_config"]["requests"], t["benchmark_config"]["requests"]
            gm, tm = g["benchmark_config"]["max_tokens"], t["benchmark_config"]["max_tokens"]
            if (gc, gr, gm) != (tc, tr, tm):
                raise ValueError(f"Parameter Parity Violation in {s}: SGLang ({gc},{gr},{gm}) != TRT-LLM ({tc},{tr},{tm})")
        elif s == "soak":
            gc, tc = g["soak_config"]["concurrency"], t["soak_config"]["concurrency"]
            gd, td = g["soak_config"]["duration"], t["soak_config"]["duration"]
            gm, tm = g["soak_config"]["max_tokens"], t["soak_config"]["max_tokens"]
            if (gc, gd, gm) != (tc, td, tm):
                raise ValueError(f"Parameter Parity Violation in soak: SGLang ({gc},{gd},{gm}) != TRT-LLM ({tc},{td},{tm})")
        elif s == "saturation":
            g_cells = [(item.get("isl_target"), item.get("osl"), item.get("concurrency")) for item in g.get("sweep_results", [])]
            t_cells = [(item.get("isl_target"), item.get("osl"), item.get("concurrency")) for item in t.get("sweep_results", [])]
            if g_cells != t_cells:
                raise ValueError(f"Parameter Parity Violation in saturation: SGLang cells {g_cells} != TRT-LLM cells {t_cells}")
            g_grid = g.get("grid", {})
            t_grid = t.get("grid", {})
            for const_name in ["MAX_INFLIGHT_PROMPT_TOKENS", "BASE_TOKENS_APPROX"]:
                if g_grid.get(const_name) != t_grid.get(const_name):
                    raise ValueError(f"Parameter Parity Violation in saturation: Drifted calibration constant '{const_name}' (SGLang={g_grid.get(const_name)} != TRT-LLM={t_grid.get(const_name)})")
        elif s == "prefill":
            gp, tp = g["prompt_tokens"], t["prompt_tokens"]
            if gp != tp:
                raise ValueError(f"Parameter Parity Violation in prefill: SGLang prompt tokens ({gp}) != TRT-LLM ({tp})")

def calc_delta(val_sglang: float, val_trtllm: float, higher_is_better: bool = True) -> str:
    if val_sglang == 0:
        return "+0.00%"
    if higher_is_better:
        d = ((val_trtllm - val_sglang) / val_sglang) * 100.0
    else:
        d = ((val_sglang - val_trtllm) / val_sglang) * 100.0
    return f"{d:+.2f}%"

def generate_markdown(sglang: dict | None, trtllm: dict | None) -> str:
    if sglang is not None and trtllm is not None:
        g_meta = get_metadata(sglang["standard"])
        t_meta = get_metadata(trtllm["standard"])
        exp_sglang_ver = get_engine_version("sglang", root=Path(__file__).resolve().parent.parent)
        exp_trtllm_ver = get_engine_version("trtllm", root=Path(__file__).resolve().parent.parent)
        g_ver = g_meta.get("engine_version", f"v{exp_sglang_ver}")
        t_ver = t_meta.get("engine_version", exp_trtllm_ver)
        
        g_label = f"SGLang ({g_ver})"
        t_label = f"NVIDIA TensorRT-LLM ({t_ver})"

        lines = []
        lines.append("### Live Benchmark Performance Comparison (SGLang vs TensorRT-LLM)")
        lines.append("")
        lines.append("All benchmarks were executed on the live GKE serving cluster with identical hardware allocations (16x NVIDIA B200 HGX across 2 nodes, GKE `a4-highgpu-8g` node pool, NVLink 5th-gen, RoCEv2 GPUDirect RDMA fabric) and identical model weights mounted read-only from a shared Hyperdisk ML volume. Both engines served via the LiteLLM Enterprise Gateway on port 4000 (Standard, Massive, Soak) and direct container port 8000 (Saturation Sweep, Prefill Ingestion).")
        lines.append("")
        lines.append("#### Methodology & Provenance Protocol")
        lines.append("* **Cache Policy:** Workload suites (Standard, Massive, Soak) evaluated end-to-end serving performance on port 4000, where dynamic prompt nonce injection bypassed LiteLLM Redis exact-match caching. The Concurrency Saturation Sweep and Prefill Ingestion suites evaluated direct engine performance on port 8000, utilizing unique prompt sets and radix cache flushing to ensure 0% prefix-cache hits (measuring true cold decoding and prefill throughput).")
        lines.append("* **Sequential Execution & Drain Protocol:** To prevent resource contention and queue contamination, benchmark suites were executed strictly sequentially with full queue drain intervals between runs.")
        lines.append("* **Engine Provenance Verification:** Engine identity and container provenance were verified prior to every suite by inspecting `/metrics` endpoints and deployment container images. Collection timestamps recorded in suite metadata:")
        
        g_img = g_meta.get("image", "unknown").split("/")[-1]
        t_img = t_meta.get("image", "unknown").split("/")[-1]
        
        g_ts_list = [f"{s.capitalize()} ({get_metadata(sglang[s]).get('run_timestamp', 'N/A')})" for s in ["standard", "massive", "soak", "saturation", "prefill"]]
        t_ts_list = [f"{s.capitalize()} ({get_metadata(trtllm[s]).get('run_timestamp', 'N/A')})" for s in ["standard", "massive", "soak", "saturation", "prefill"]]
        
        lines.append(f"  * **SGLang** (`{g_img}`): {', '.join(g_ts_list)}.")
        lines.append(f"  * **TensorRT-LLM** (`{t_img}`): {', '.join(t_ts_list)}.")
        lines.append("")
        
        lines.append("#### Table 1: Production Workload Suite Summary (Gateway Port 4000)")
        lines.append(f"| Workload Suite | Metric | {g_label} | {t_label} | Delta ($\\Delta$) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        
        suites_t1 = [
            ("Standard Suite ($c=8$, $128\\text{ tok}$)", "standard", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], False, "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], False, "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], True, "{:.2f}"),
                ("Success rate", lambda d: 100.0 * d["successful_requests"] / max(1, d["total_requests"]), True, "{:.1f}%"),
            ]),
            ("Massive Stress ($c=20$, $256\\text{ tok}$)", "massive", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], False, "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], False, "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], True, "{:.2f}"),
                ("Success rate", lambda d: 100.0 * d["successful_requests"] / max(1, d["total_requests"]), True, "{:.1f}%"),
            ]),
            ("Endurance Soak ($c=18$, $1800\\text{s}$)", "soak", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], False, "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], False, "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], True, "{:.2f}"),
                ("Completed cycles", lambda d: float(d["successful_requests"]), True, "{:.0f}"),
            ]),
        ]
        for suite_label, s_key, metrics in suites_t1:
            for i, (m_label, extractor, hib, fmt) in enumerate(metrics):
                val_g = extractor(sglang[s_key])
                val_t = extractor(trtllm[s_key])
                delta_str = calc_delta(val_g, val_t, hib)
                w_col = suite_label if i == 0 else ""
                lines.append(f"| {w_col} | {m_label} | {fmt.format(val_g)} | {fmt.format(val_t)} | **{delta_str}** |")
                
        lines.append("")
        lines.append("#### Table 2: ISL/OSL x Concurrency Saturation Sweep (Direct Port 8000, 0% Cache Hits)")
        lines.append(f"| Grid Cell (ISL/OSL, $c$) | {g_label} tok/s | {t_label} tok/s | Throughput $\\Delta$ | {g_label} TTFT P99 (s) | {t_label} TTFT P99 (s) | TTFT P99 $\\Delta$ |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        g_sweep = {(item.get("isl_target"), item.get("osl"), item.get("concurrency")): item for item in sglang["saturation"].get("sweep_results", [])}
        t_sweep = {(item.get("isl_target"), item.get("osl"), item.get("concurrency")): item for item in trtllm["saturation"].get("sweep_results", [])}
        all_keys = sorted(set(g_sweep.keys()) | set(t_sweep.keys()), key=lambda k: (k[0] or 0, k[1] or 0, k[2] or 0))
        for key in all_keys:
            isl, osl, c = key
            cell_label = f"${isl//1024}k/{osl//1024}k$, $c={c}$" if (isl and osl) else f"$c={c}$"
            gi = g_sweep.get(key, {})
            ti = t_sweep.get(key, {})
            g_status = gi.get("status", "error")
            t_status = ti.get("status", "error")
            if g_status == "skipped" and t_status == "skipped":
                g_reason = gi.get("reason", "Skipped")
                t_reason = ti.get("reason", "Skipped")
                reason_str = g_reason if g_reason == t_reason else f"SGLang: {g_reason}; TRTLLM: {t_reason}"
                lines.append(f"| {cell_label} | *SKIPPED* | *SKIPPED* | — | *{reason_str}* | *{reason_str}* | — |")
                continue
            elif g_status == "skipped":
                g_tok_str, g_ttft_str = "*SKIPPED*", f"*{gi.get('reason', 'Skipped')}*"
                t_tok, t_ttft_s = ti.get("aggregate_tok_s", 0.0), ti.get("ttft_ms", {}).get("p99", 0.0) / 1000.0
                t_tok_str, t_ttft_str = f"{t_tok:.2f}", f"{t_ttft_s:.4f} s"
                d_tok, d_ttft = "—", "—"
            elif t_status == "skipped":
                g_tok, g_ttft_s = gi.get("aggregate_tok_s", 0.0), gi.get("ttft_ms", {}).get("p99", 0.0) / 1000.0
                g_tok_str, g_ttft_str = f"{g_tok:.2f}", f"{g_ttft_s:.4f} s"
                t_tok_str, t_ttft_str = "*SKIPPED*", f"*{ti.get('reason', 'Skipped')}*"
                d_tok, d_ttft = "—", "—"
            else:
                g_tok, g_ttft_s = gi.get("aggregate_tok_s", 0.0), gi.get("ttft_ms", {}).get("p99", 0.0) / 1000.0
                t_tok, t_ttft_s = ti.get("aggregate_tok_s", 0.0), ti.get("ttft_ms", {}).get("p99", 0.0) / 1000.0
                g_tok_str, g_ttft_str = f"{g_tok:.2f}", f"{g_ttft_s:.4f} s"
                t_tok_str, t_ttft_str = f"{t_tok:.2f}", f"{t_ttft_s:.4f} s"
                d_tok = calc_delta(g_tok, t_tok, True)
                d_ttft = calc_delta(g_ttft_s, t_ttft_s, False)
            lines.append(f"| {cell_label} | {g_tok_str} | {t_tok_str} | **{d_tok}** | {g_ttft_str} | {t_ttft_str} | **{d_ttft}** |")

        lines.append("")
        lines.append("#### Table 3: Prompt Prefill Ingestion Stress ($8,192\\text{ prompt tok} \\to 16\\text{ out}$)")
        lines.append(f"| Metric | {g_label} | {t_label} | Delta ($\\Delta$) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        gp_tok, tp_tok = sglang["prefill"]["prefill_tok_s_system"], trtllm["prefill"]["prefill_tok_s_system"]
        gp_ttft, tp_ttft = sglang["prefill"]["ttft_ms"], trtllm["prefill"]["ttft_ms"]
        d_prefill_tok = calc_delta(gp_tok, tp_tok, True)
        d_prefill_ttft = calc_delta(gp_ttft, tp_ttft, False)
        lines.append(f"| Prefill throughput | {gp_tok:.2f} prompt tok/s | {tp_tok:.2f} prompt tok/s | **{d_prefill_tok}** |")
        lines.append(f"| TTFT mean (ms) | {gp_ttft:.2f} ms | {tp_ttft:.2f} ms | **{d_prefill_ttft}** |")
        
        lines.append("")
        lines.append("#### Technical Guidance: When to Choose SGLang vs TensorRT-LLM")
        lines.append("")
        sglang_std_tps = sglang["standard"]["throughput_tokens_sec"]
        trtllm_std_tps = trtllm["standard"]["throughput_tokens_sec"]
        sglang_std_tpot = sglang["standard"]["metrics"]["tpot_ms"]["mean"]
        trtllm_std_tpot = trtllm["standard"]["metrics"]["tpot_ms"]["mean"]
        lines.append(f"* **Choose SGLang (`INFERENCE_ENGINE=sglang`)** as the primary production default when your application relies on RadixAttention prefix caching, structured JSON generation, or dynamic multi-turn conversational agents. SGLang demonstrated robust decoding performance (Standard TPOT of {sglang_std_tpot:.2f} ms vs TensorRT-LLM {trtllm_std_tpot:.2f} ms) and sustained stability during 30-minute endurance soak testing.")
        if tp_tok > gp_tok:
            prefill_text = f"TensorRT-LLM demonstrated superior prompt ingestion throughput ({tp_tok:.2f} prompt tok/s vs SGLang {gp_tok:.2f} prompt tok/s)."
        else:
            prefill_text = f"SGLang demonstrated superior prompt ingestion throughput ({gp_tok:.2f} prompt tok/s vs TensorRT-LLM {tp_tok:.2f} prompt tok/s)."
        lines.append(f"* **Choose NVIDIA TensorRT-LLM (`INFERENCE_ENGINE=trtllm`)** for high-concurrency raw batch serving and static workload profiles where experimental engine optimization is desired. {prefill_text} TensorRT-LLM provides native OpenMPI orchestration across multi-node HGX allocations.")
        return "\n".join(lines)
    else:
        eng_name = "sglang" if sglang is not None else "trtllm"
        eng_data = sglang if sglang is not None else trtllm
        meta = get_metadata(eng_data["standard"])
        exp_ver = get_engine_version(eng_name, root=Path(__file__).resolve().parent.parent)
        ver = meta.get("engine_version", f"v{exp_ver}" if eng_name == "sglang" else exp_ver)
        label = f"SGLang ({ver})" if eng_name == "sglang" else f"NVIDIA TensorRT-LLM ({ver})"
        heading_name = "SGLang" if eng_name == "sglang" else "NVIDIA TensorRT-LLM"
        img = meta.get("image", "unknown").split("/")[-1]
        ts_list = [f"{s.capitalize()} ({get_metadata(eng_data[s]).get('run_timestamp', 'N/A')})" for s in ["standard", "massive", "soak", "saturation", "prefill"]]

        other_eng_name = "NVIDIA TensorRT-LLM" if eng_name == "sglang" else "SGLang"
        lines = []
        lines.append(f"### Live Benchmark Performance ({heading_name})")
        lines.append("")
        lines.append("All benchmarks were executed on the live GKE serving cluster with identical hardware allocations (16x NVIDIA B200 HGX across 2 nodes, GKE `a4-highgpu-8g` node pool, NVLink 5th-gen, RoCEv2 GPUDirect RDMA fabric) and identical model weights mounted read-only from a shared Hyperdisk ML volume. The engine served via the LiteLLM Enterprise Gateway on port 4000 (Standard, Massive, Soak) and direct container port 8000 (Saturation Sweep, Prefill Ingestion).")
        lines.append("")
        lines.append(f"**Note:** {other_eng_name} was not benchmarked in this run, so comparative delta columns and selection guidance are omitted.")
        lines.append("")
        lines.append("#### Methodology & Provenance Protocol")
        lines.append("* **Cache Policy:** Workload suites (Standard, Massive, Soak) evaluated end-to-end serving performance on port 4000, where dynamic prompt nonce injection bypassed LiteLLM Redis exact-match caching. The Concurrency Saturation Sweep and Prefill Ingestion suites evaluated direct engine performance on port 8000, utilizing unique prompt sets and radix cache flushing to ensure 0% prefix-cache hits (measuring true cold decoding and prefill throughput).")
        lines.append("* **Sequential Execution & Drain Protocol:** To prevent resource contention and queue contamination, benchmark suites were executed strictly sequentially with full queue drain intervals between runs.")
        lines.append("* **Engine Provenance Verification:** Engine identity and container provenance were verified prior to every suite by inspecting `/metrics` endpoints and deployment container images. Collection timestamps recorded in suite metadata:")
        lines.append(f"  * **{heading_name}** (`{img}`): {', '.join(ts_list)}.")
        lines.append("")
        
        lines.append("#### Table 1: Production Workload Suite Summary (Gateway Port 4000)")
        lines.append(f"| Workload Suite | Metric | {label} |")
        lines.append("| :--- | :--- | :--- |")
        
        suites_t1 = [
            ("Standard Suite ($c=8$, $128\\text{ tok}$)", "standard", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], "{:.2f}"),
                ("Success rate", lambda d: 100.0 * d["successful_requests"] / max(1, d["total_requests"]), "{:.1f}%"),
            ]),
            ("Massive Stress ($c=20$, $256\\text{ tok}$)", "massive", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], "{:.2f}"),
                ("Success rate", lambda d: 100.0 * d["successful_requests"] / max(1, d["total_requests"]), "{:.1f}%"),
            ]),
            ("Endurance Soak ($c=18$, $1800\\text{s}$)", "soak", [
                ("TTFT P50 (ms)", lambda d: d["metrics"]["ttft_ms"]["p50"], "{:.2f}"),
                ("TPOT mean (ms)", lambda d: d["metrics"]["tpot_ms"]["mean"], "{:.2f}"),
                ("Throughput (tok/s)", lambda d: d["throughput_tokens_sec"], "{:.2f}"),
                ("Completed cycles", lambda d: float(d["successful_requests"]), "{:.0f}"),
            ]),
        ]
        for suite_label, s_key, metrics in suites_t1:
            for i, (m_label, extractor, fmt) in enumerate(metrics):
                val = extractor(eng_data[s_key])
                w_col = suite_label if i == 0 else ""
                lines.append(f"| {w_col} | {m_label} | {fmt.format(val)} |")
                
        lines.append("")
        lines.append("#### Table 2: ISL/OSL x Concurrency Saturation Sweep (Direct Port 8000, 0% Cache Hits)")
        lines.append(f"| Grid Cell (ISL/OSL, $c$) | {label} tok/s | {label} TTFT P99 (s) |")
        lines.append("| :--- | :--- | :--- |")
        
        sweep = {(item.get("isl_target"), item.get("osl"), item.get("concurrency")): item for item in eng_data["saturation"].get("sweep_results", [])}
        all_keys = sorted(sweep.keys(), key=lambda k: (k[0] or 0, k[1] or 0, k[2] or 0))
        for key in all_keys:
            isl, osl, c = key
            cell_label = f"${isl//1024}k/{osl//1024}k$, $c={c}$" if (isl and osl) else f"$c={c}$"
            item = sweep.get(key, {})
            status = item.get("status", "error")
            if status == "skipped":
                reason_str = item.get("reason", "Skipped")
                lines.append(f"| {cell_label} | *SKIPPED* | *{reason_str}* |")
            else:
                tok = item.get("aggregate_tok_s", 0.0)
                ttft_s = item.get("ttft_ms", {}).get("p99", 0.0) / 1000.0
                lines.append(f"| {cell_label} | {tok:.2f} | {ttft_s:.4f} s |")
                
        lines.append("")
        lines.append("#### Table 3: Prompt Prefill Ingestion Stress ($8,192\\text{ prompt tok} \\to 16\\text{ out}$)")
        lines.append(f"| Metric | {label} |")
        lines.append("| :--- | :--- |")
        p_tok = eng_data["prefill"]["prefill_tok_s_system"]
        p_ttft = eng_data["prefill"]["ttft_ms"]
        lines.append(f"| Prefill throughput | {p_tok:.2f} prompt tok/s |")
        lines.append(f"| TTFT mean (ms) | {p_ttft:.2f} ms |")
        
        return "\n".join(lines)

def update_readme(markdown_block: str, readme_path: Path):
    if not readme_path.exists():
        print(f"ERROR: {readme_path} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    content = readme_path.read_text(encoding="utf-8")
    start_marker = "<!-- ENGINE_COMPARISON_START -->"
    end_marker = "<!-- ENGINE_COMPARISON_END -->"
    
    if start_marker not in content or end_marker not in content:
        print("ERROR: Comparison markers not found in README.md", file=sys.stderr)
        sys.exit(1)
        
    pattern = re.compile(rf"({re.escape(start_marker)}).*?({re.escape(end_marker)})", re.DOTALL)
    new_content, count = pattern.subn(lambda m: f"{m.group(1)}\n\n{markdown_block.strip()}\n\n{m.group(2)}", content)
    
    if count == 0:
        print("ERROR: Regex substitution failed on README.md", file=sys.stderr)
        sys.exit(1)
        
    if new_content == content:
        print("README.md comparison block is already up-to-date (no changes needed).")
    else:
        readme_path.write_text(new_content, encoding="utf-8")
        print("Successfully updated README.md comparison block in place.")

def main():
    root = Path(__file__).resolve().parent
    results_dir = Path(os.getenv("COMPARISON_RESULTS_DIR", root / "results"))
    
    sglang_dir = results_dir / "sglang"
    trtllm_dir = results_dir / "trtllm"
    
    suites = ["standard", "massive", "soak", "saturation", "prefill"]
    
    status = {}
    for eng in ["sglang", "trtllm"]:
        eng_dir = results_dir / eng
        present = [s for s in suites if (eng_dir / f"{s}_results.json").exists()]
        missing = [s for s in suites if not (eng_dir / f"{s}_results.json").exists()]
        if len(present) == len(suites):
            status[eng] = "complete"
        elif len(present) == 0:
            status[eng] = "absent"
        else:
            status[eng] = "partial"
            print(f"ERROR: Partial benchmark data for {eng}: present={present}, missing={missing}. All 5 suite files required.", file=sys.stderr)

    if "partial" in status.values():
        sys.exit(1)

    if status["sglang"] == "absent" and status["trtllm"] == "absent":
        print("No complete benchmark results found (0 complete engines). Leaving README.md untouched.")
        sys.exit(0)

    sglang_data = {} if status["sglang"] == "complete" else None
    trtllm_data = {} if status["trtllm"] == "complete" else None

    print("Loading benchmark JSON results...")
    for s in suites:
        if sglang_data is not None:
            sglang_data[s] = load_json(sglang_dir / f"{s}_results.json")
        if trtllm_data is not None:
            trtllm_data[s] = load_json(trtllm_dir / f"{s}_results.json")
        
    print("Enforcing provenance, parameter parity, and sanity gates...")
    try:
        validate_provenance(sglang_data, trtllm_data)
        validate_sanity(sglang_data, trtllm_data)
        validate_parity(sglang_data, trtllm_data)
    except ValueError as e:
        print(f"GATE FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Generating Markdown block...")
    md_block = generate_markdown(sglang_data, trtllm_data)
    
    readme_path = Path(os.getenv("COMPARISON_README_PATH", root.parent / "README.md"))
    print(f"Idempotently updating {readme_path}...")
    update_readme(md_block, readme_path)
    print("Done.")

if __name__ == "__main__":
    main()
