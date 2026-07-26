#!/usr/bin/env python3
# ==============================================================================
# DAY-1 SWEEP MATRIX (Pre-launch Parameter Grid for Monday Benchmark Execution)
# ==============================================================================
# 1. Engine: sglang, trtllm
# 2. Parallelism: TP=16 / PP=1, TP=8 / PP=2
# 3. Concurrency Levels: 1, 8, 32, 128
# 4. ISL / OSL (Input/Output Token Pairs): 1k/1k, 8k/1k, 32k/2k, 128k/2k
# 5. Target Metrics per Cell:
#    - TTFT (p50, p95, p99)
#    - TPOT (p50, p95, p99)
#    - Aggregate output throughput (tok/s)
#    - Output tok/s per GPU (aggregate divided by 16.0)
#    - Peak KV-cache utilization
#    - Request success rate
#    - Leader GPU Memory Utilization Percentage
# 6. Parallelism Invariants Note: EP_SIZE must equal the total GPU count (16) in
#    every cell, and TP * PP must equal 16 — those are the real invariants.
# ==============================================================================
"""Kimi K3 (2.8T MXFP4) Saturation & Throughput-Ceiling Sweep

Measures peak GPU-generated aggregate throughput, per-user tok/s, TTFT/TPOT
percentiles, and the interactive SLA knee across concurrency levels c in [1, 8,
32, 128].
Connects directly to TensorRT-LLM on port 8000 with 16-character cryptographic
nonce injection to guarantee 0% cache hit rate.
Normalizes per-GPU throughput by dividing aggregate tokens/sec by 16.0 (16x B200
GPUs in 2-node pool).
"""

import argparse
import concurrent.futures
import json
import os
import secrets
import statistics
import string
import sys
import time
import urllib.error
import urllib.request

SYNTHETIC_BASE_1K = (
    "In large-scale distributed artificial intelligence deployments on"
    " sovereign cloud infrastructure, NVIDIA Blackwell B200 HGX 2-node spot"
    " pools provide 16 GPUs connected via GPUDirect RDMA over RoCEv2 fabric"
    " with 3.2 Tbps bandwidth per node and fifth-generation NVLink with 1.8"
    " TB/s bidirectional bandwidth per GPU. When serving MoE architectures with"
    " 2.8 trillion parameters and 896 experts such as Kimi K3 using MXFP4"
    " weight quantization and TensorRT-LLM MPI distributed inference across 16x"
    " B200 GPUs, expert routing decisions occur across both nodes with minimal"
    " latency. Furthermore, the 2 TB ReadOnlyMany Hyperdisk ML storage"
    " architecture enables horizontal pod scaling without redundant checkpoint"
    " downloads. "
) * 6  # ~1024 tokens (~6400 chars)


def generate_unique_prompt(idx, c):
  nonce = "".join(
      secrets.choice(string.ascii_letters + string.digits) for _ in range(16)
  )
  return f"[Sweep C={c} ReqId={idx} Nonce={nonce}] {SYNTHETIC_BASE_1K}"


def execute_single_request(
    req_idx, c, endpoint, model, max_tokens, temperature, api_key=""
):
  prompt = generate_unique_prompt(req_idx, c)
  payload = {
      "model": model,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "ignore_eos": True,
      "stream": True,
      "stream_options": {"include_usage": True},
  }
  if "/chat/completions" in endpoint:
    payload["messages"] = [{"role": "user", "content": prompt}]
  else:
    payload["prompt"] = prompt
  req_body = json.dumps(payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "User-Agent": f"KIMI3-Saturation-Sweep/1.0 (Req-{req_idx})",
  }
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  req = urllib.request.Request(
      endpoint,
      data=req_body,
      headers=headers,
      method="POST",
  )

  t_start = time.time()
  t_first_token = None
  token_timestamps = []
  generated_tokens = 0
  prompt_tokens = 0

  try:
    with urllib.request.urlopen(req, timeout=300) as resp:
      for line in resp:
        decoded = line.decode("utf-8").strip()
        if not decoded.startswith("data: "):
          continue
        data_str = decoded[6:]
        if data_str == "[DONE]":
          break
        try:
          chunk = json.loads(data_str)
          if (
              "choices" in chunk
              and chunk["choices"]
              and "text" in chunk["choices"][0]
          ):
            txt = chunk["choices"][0].get("text")
            if txt is not None and len(txt) > 0:
              now = time.time()
              if t_first_token is None:
                t_first_token = now
              token_timestamps.append(now)
              generated_tokens += 1
          elif "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            if usage.get("completion_tokens"):
              generated_tokens = usage["completion_tokens"]
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            Exception,
        ):
          pass
    t_end = time.time()
    ttft_ms = (
        (t_first_token - t_start) * 1000.0
        if t_first_token
        else (t_end - t_start) * 1000.0
    )
    tpots = []
    if len(token_timestamps) > 1:
      tpots = [
          (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
          for i in range(1, len(token_timestamps))
      ]
    mean_tpot = statistics.mean(tpots) if tpots else 0.0
    duration = t_end - t_start
    throughput = generated_tokens / duration if duration > 0 else 0.0
    return {
        "success": True,
        "ttft_ms": ttft_ms,
        "mean_tpot_ms": mean_tpot,
        "all_tpots_ms": tpots,
        "tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "duration": duration,
        "throughput": throughput,
    }
  except Exception as e:
    return {"success": False, "error": str(e)}


def run_sweep_concurrency(c, requests_per_c, endpoint, model, max_tokens, api_key=""):
  print(f"\n============================================================")
  print(
      f"Executing Saturation Sweep at Concurrency = {c} ({requests_per_c} total"
      " requests)..."
  )
  print(f"============================================================")

  results = []
  start_time = time.time()
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
    futures = [
        executor.submit(
            execute_single_request, i, c, endpoint, model, max_tokens, 0.2, api_key
        )
        for i in range(requests_per_c)
    ]
    for f in concurrent.futures.as_completed(futures):
      results.append(f.result())
  total_duration = time.time() - start_time

  successful = [r for r in results if r["success"]]
  failed = [r for r in results if not r["success"]]
  err_rate = (len(failed) / len(results)) * 100.0 if results else 100.0

  if successful:
    ttfts = sorted([r["ttft_ms"] for r in successful])
    tpots = []
    for r in successful:
      tpots.extend([val for val in r.get("all_tpots_ms", []) if val > 0])
    tpots.sort()
    total_tokens = sum(r["tokens"] for r in successful)
    agg_tok_s = total_tokens / total_duration
    per_gpu_tok_s = agg_tok_s / 16.0
    per_user_tok_s = agg_tok_s / c

    def pctl(arr, p):
      if not arr:
        return 0.0
      idx = int(len(arr) * p / 100.0)
      return arr[min(idx, len(arr) - 1)]

    try:
      ttft_stats = {
          "mean": statistics.mean(ttfts) if ttfts else 0.0,
          "p50": pctl(ttfts, 50),
          "p90": pctl(ttfts, 90),
          "p99": pctl(ttfts, 99),
      }
      tpot_stats = {
          "mean": statistics.mean(tpots) if tpots else 0.0,
          "p50": pctl(tpots, 50),
          "p90": pctl(tpots, 90),
          "p99": pctl(tpots, 99),
      }
      summary = {
          "concurrency": c,
          "requests": len(results),
          "successful": len(successful),
          "error_rate_pct": err_rate,
          "total_tokens": total_tokens,
          "total_duration_sec": total_duration,
          "aggregate_tok_s": agg_tok_s,
          "per_gpu_tok_s": per_gpu_tok_s,
          "per_user_tok_s": per_user_tok_s,
          "ttft_ms": ttft_stats,
          "tpot_ms": tpot_stats,
      }
    except (
        ZeroDivisionError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        Exception,
    ):
      ttft_stats = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
      tpot_stats = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
      summary = {
          "concurrency": c,
          "requests": len(results),
          "successful": 0,
          "error_rate_pct": 100.0,
          "total_tokens": 0,
          "total_duration_sec": 0.0,
          "aggregate_tok_s": 0.0,
          "per_gpu_tok_s": 0.0,
          "per_user_tok_s": 0.0,
          "ttft_ms": ttft_stats,
          "tpot_ms": tpot_stats,
      }
    print(
        f"  Agg Throughput:  {agg_tok_s:.2f} tok/s ({per_gpu_tok_s:.2f}"
        " tok/s/GPU across 16x B200)"
    )
    print(f"  Per-User TPS:    {per_user_tok_s:.2f} tok/s")
    print(
        f"  TTFT (P50/P90):  {ttft_stats['p50']:.1f} ms /"
        f" {ttft_stats['p90']:.1f} ms"
    )
    print(
        f"  TPOT (P50/P90):  {tpot_stats['p50']:.2f} ms /"
        f" {tpot_stats['p90']:.2f} ms"
    )
    print(f"  Error Rate:      {err_rate:.1f}%")
    return summary
  else:
    print(f"  All requests failed at Concurrency={c}!")
    return {"concurrency": c, "error_rate_pct": 100.0, "aggregate_tok_s": 0.0}


def parse_args():
  parser = argparse.ArgumentParser(
      description="Kimi K3 Saturation & Throughput-Ceiling Sweep"
  )
  parser.add_argument(
      "--dry-run",
      dest="dry_run",
      action="store_true",
      help="Validate harness syntax",
  )
  parser.add_argument(
      "--endpoint",
      default="http://localhost:8000/v1/completions",
      help="TensorRT-LLM completions endpoint URL",
  )
  parser.add_argument(
      "--model", default="kimi-k3-2.8t-mxfp4", help="Served model ID"
  )
  parser.add_argument(
      "--output",
      default="benchmarks/saturation_sweep_results_kimi_k3.json",
      help="Output JSON path",
  )
  parser.add_argument(
      "--engine", default="trtllm", help="Inference engine (trtllm or sglang)"
  )
  parser.add_argument(
      "--metadata", default="{}", help="JSON string of engine metadata"
  )
  parser.add_argument(
      "--api-key", default="", help="Optional API key for gateway authentication"
  )
  return parser.parse_args()


def main():
  args = parse_args()
  if args.dry_run:
    print("[SUCCESS] Kimi K3 saturation sweep harness syntax verified.")
    sys.exit(0)

  print(f"\n=== KIMI K3 SATURATION SWEEP (Engine: {args.engine}) ===")
  sweep_levels = [1, 8, 16, 32, 64]
  sweep_results = []

  for c in sweep_levels:
    requests = max(c * 2, 8)  # Run at least 2 full waves per concurrency level
    res = run_sweep_concurrency(
        c, requests, args.endpoint, args.model, max_tokens=1024, api_key=args.api_key
    )
    sweep_results.append(res)
    time.sleep(2)

  try:
    meta_dict = json.loads(args.metadata) if args.metadata else {}
  except (json.JSONDecodeError, TypeError, ValueError, Exception):
    meta_dict = {"raw": args.metadata}

  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as f:
    json.dump(
        {
            "engine": args.engine,
            "metadata": meta_dict,
            "sweep_results": sweep_results,
        },
        f,
        indent=2,
    )
  print(f"\nSaved full saturation sweep results to {args.output}")


if __name__ == "__main__":
  main()
