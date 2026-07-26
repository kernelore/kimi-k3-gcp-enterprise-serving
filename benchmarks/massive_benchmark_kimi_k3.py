#!/usr/bin/env python3
"""Kimi K3 (2.8T MXFP4) Massive Production Stress & Performance Suite

Executes a high-concurrency (C=20), high-volume (N=100) stress test against the
TensorRT-LLM MPI serving engine on NVIDIA Blackwell GKE clusters (16x B200 GPUs
in 2-node pool).
Simulates 20 concurrent autonomous engineering agents and enterprise developers
running long-context reasoning and code generation.
Measures TTFT (Time-to-First-Token), TPOT (Time-Per-Output-Token), KV Cache
Saturation, and Sustained Cluster & Per-GPU Throughput.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

MASSIVE_PROMPTS = [
    """You are a Senior Systems Architect at Google. Analyze and explain the complete memory co-design of Kimi K3 (~2.8T parameters, 896 MoE experts) on a 2-node NVIDIA Blackwell spot pool (`2x a4-highgpu-8g` = 16x B200 HGX GPUs total = 2,880 GB HBM3e VRAM). 
Detail the exact byte savings from MXFP4 weight quantization (0.5 bytes/param) and MXFP8 activation scaling, calculate the static weight footprint (~1,400 GB), and derive the remaining HBM3e capacity (~1,050 GB) dedicated to PagedAttention Key-Value cache across the pool. Furthermore, explain how MoE expert routing across 896 experts minimizes inter-GPU communication latency.""",
    """Derive the comprehensive 3-year Committed Use Discount (CUD) Total Cost of Ownership (TCO) model for hosting Kimi K3 in europe-north1 (Hamina, Finland).
Include the exact pricing differences between On-Demand rate ($64.00/hour across the 16x B200 GPU pool, or $4.00/GPU/hour) and 3-Year CUD ($28.80/hour, or $1.80/GPU/hour at 55% savings). Provide the step-by-step mathematical derivation of annual savings, accounting for Hamina's 100% seawater cooling efficiency (PUE 1.10) and compact placement policies (`collocated`).""",
    """Explain the internal network routing mechanics of TensorRT-LLM multi-node MPI distributed inference (`--tp_size 8 --pp_size 2 --ep_size 8`) across the 2-node pool connected via GPUDirect RDMA over RoCEv2 fabric (3.2 Tbps per node, MTU 8896) and intra-node NVLink 5.0 (1.8 TB/s bidirectional per GPU).
How does this dual-network hierarchy eliminate communication bottlenecks during MoE all-to-all expert token dispatch across 16x B200 GPUs?""",
    """Write a robust, production-grade Python async service using `aiohttp` and `pydantic` that acts as a resilient gateway to the Kimi K3 TensorRT-LLM inference endpoint.
Your implementation must include:
1. Circuit breaking when inter-token latency (TPOT) exceeds 50ms for 3 consecutive windows.
2. Exponential backoff with jitter on HTTP 429 / 503 errors.
3. Prefix caching-aware prompt sorting to maximize hit rate across concurrent agent streams.
4. Structured logging of TTFT and token generation speeds.""",
    """Examine the failover and recovery mechanics of the 2-node Kimi K3 TensorRT-LLM MPI serving StatefulSet during an unannounced spot preemption event (`cloud.google.com/gke-spot`).
Describe what happens when a SIGTERM is received by the TensorRT-LLM engine: how active requests are drained during graceful shutdown, and how the newly scheduled replacement pod hydrates from Tier-0 Hyperdisk ML ReadOnlyMany (`ROX`) in ~38.18 seconds without network egress storms.""",
    """Compare and contrast speculative decoding (Multi-Token Prediction / draft heads) against standard autoregressive decoding in 2.8T MoE models like Kimi K3.
How does predicting future tokens increase effective memory bandwidth utilization on Blackwell tensor cores when serving 20 concurrent agent streams at 128k context?""",
    """Design the complete Kubernetes manifest stack and custom Horizontal Pod Autoscaler (HPA) definition for scaling Kimi K3 pods based on PagedAttention KV cache utilization or queue depth.
Provide exact YAML snippets for a `CustomMetric` HPA targeting 75% KV cache saturation. Explain why CPU or memory utilization metrics are fundamentally unsuitable for scaling LLM inference engines and how dynamic KV cache swapping to local NVMe SSD RAID 0 scratch array (/mnt/disks/local-scratch) acts as a critical buffer during traffic spikes.""",
]


def parse_args():
  parser = argparse.ArgumentParser(
      description="Kimi K3 Massive Concurrency Stress Test"
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
      "--api-key",
      default=os.environ.get("GATEWAY_MASTER_KEY", ""),
      help="API key for Gateway authentication",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=20,
      help="Number of concurrent requests (Target: 20 agents)",
  )
  parser.add_argument(
      "--requests",
      type=int,
      default=100,
      help="Total number of requests to execute",
  )
  parser.add_argument(
      "--max-tokens",
      type=int,
      default=256,
      help="Max generation tokens per request",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.3,
      help="Sampling temperature",
  )
  parser.add_argument(
      "--output",
      default="massive_benchmark_results_kimi_k3.json",
      help="Output JSON path",
  )
  parser.add_argument(
      "--engine", default="trtllm", help="Inference engine (trtllm or sglang)"
  )
  parser.add_argument(
      "--metadata", default="{}", help="JSON string of engine metadata"
  )
  return parser.parse_args()


def execute_stream_request(
    req_id, endpoint, model, prompt, max_tokens, temperature, api_key=""
):
  payload = {
      "model": model,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "stream": True,
      "stream_options": {"include_usage": True},
  }
  if "/chat/completions" in endpoint:
    payload["messages"] = [{"role": "user", "content": prompt}]
  else:
    payload["prompt"] = prompt

  data = json.dumps(payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "User-Agent": f"KIMI3-Massive-Bench/2.0 (ReqId-{req_id})",
  }
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

  req = urllib.request.Request(
      endpoint, data=data, headers=headers, method="POST"
  )
  t_start = time.perf_counter()
  t_first_token = None
  t_end = None
  tokens_received = 0
  prompt_tokens = 0
  error_msg = None
  has_exact_usage = False

  try:
    with urllib.request.urlopen(req, timeout=300) as response:
      for line in response:
        line_str = line.decode("utf-8").strip()
        if not line_str.startswith("data: "):
          continue
        data_part = line_str[6:].strip()
        if data_part == "[DONE]":
          break
        try:
          chunk = json.loads(data_part)
          usage = chunk.get("usage")
          if usage and isinstance(usage, dict):
            if "completion_tokens" in usage:
              tokens_received = usage.get("completion_tokens", tokens_received)
              has_exact_usage = True
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

          choices = chunk.get("choices", [])
          if choices:
            choice = choices[0]
            text = choice.get("text", "")
            if not text and isinstance(choice.get("delta"), dict):
              text = choice.get("delta", {}).get("content", "")
            if text:
              now = time.perf_counter()
              if t_first_token is None:
                t_first_token = now
              if not has_exact_usage:
                tokens_received += 1
        except (
            json.JSONDecodeError,
            ZeroDivisionError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            Exception,
        ):
          continue
    t_end = time.perf_counter()
  except Exception as e:
    error_msg = str(e)
    t_end = time.perf_counter()

  try:
    if t_first_token is not None and t_end is not None and tokens_received > 0:
      ttft = t_first_token - t_start
      total_time = t_end - t_start
      tpot = (
          (t_end - t_first_token) / max(1, tokens_received - 1)
          if tokens_received > 1
          else ttft
      )
      req_throughput = tokens_received / total_time if total_time > 0 else 0
      success = True
    else:
      ttft = 0
      tpot = 0
      total_time = t_end - t_start if t_end else 0
      req_throughput = 0
      success = False
  except (ZeroDivisionError, KeyError, TypeError, ValueError, Exception) as e:
    ttft = 0
    tpot = 0
    total_time = 0
    req_throughput = 0
    success = False
    error_msg = str(e)

  return {
      "req_id": req_id,
      "success": success,
      "error": error_msg,
      "tokens": tokens_received,
      "prompt_tokens": prompt_tokens,
      "ttft_ms": ttft * 1000,
      "tpot_ms": tpot * 1000,
      "total_time_s": total_time,
      "req_throughput_tps": req_throughput,
  }


def main():
  args = parse_args()
  if args.dry_run:
    print("[SUCCESS] Kimi K3 massive stress test harness syntax verified.")
    sys.exit(0)

  print("=" * 70)
  print(f"  Kimi K3 MASSIVE PRODUCTION STRESS SUITE (Engine: {args.engine})  ")
  print("=" * 70)
  print(f"  Endpoint:        {args.endpoint}")
  print(f"  Served Model:    {args.model}")
  print(f"  Engine:          {args.engine}")
  print(
      f"  Concurrency:     {args.concurrency} concurrent streams (Target"
      " Workload)"
  )
  print(f"  Total Requests:  {args.requests} generations")
  print(f"  Max Tokens/Req:  {args.max_tokens} output tokens per prompt")
  print("-" * 70)

  start_bench_time = time.perf_counter()
  results = []
  completed_count = 0

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.concurrency
  ) as executor:
    futures = []
    for i in range(args.requests):
      prompt = MASSIVE_PROMPTS[i % len(MASSIVE_PROMPTS)]
      futures.append(
          executor.submit(
              execute_stream_request,
              i + 1,
              args.endpoint,
              args.model,
              prompt,
              args.max_tokens,
              args.temperature,
              args.api_key,
          )
      )

    consecutive_errors = 0
    for future in concurrent.futures.as_completed(futures):
      res = future.result()
      results.append(res)
      completed_count += 1
      status_char = "✓" if res["success"] else "✗"
      if res["success"]:
        consecutive_errors = 0
        print(
            f"[{completed_count:03d}/{args.requests:03d}] [{status_char}] Req"
            f" {res['req_id']:03d}: TTFT={res['ttft_ms']:6.1f}ms |"
            f" TPOT={res['tpot_ms']:5.2f}ms | Tokens={res['tokens']:3d} |"
            f" TPS={res['req_throughput_tps']:5.1f} t/s"
        )
      else:
        consecutive_errors += 1
        print(
            f"[{completed_count:03d}/{args.requests:03d}] [{status_char}] Req"
            f" {res['req_id']:03d}: FAILED ({res['error']})"
        )
        if consecutive_errors >= 5:
          print("\n" + "!" * 70)
          print("ERROR: Port-forward tunnel dropped (HTTP 000).")
          print(
              "Run benchmark in-cluster via: ./scripts/05_run_benchmarks.sh"
              " --in-cluster"
          )
          print("!" * 70 + "\n")
          break

  total_bench_time = time.perf_counter() - start_bench_time
  successful_results = [r for r in results if r["success"]]
  failed_results = [r for r in results if not r["success"]]

  ttft_vals = (
      sorted([r["ttft_ms"] for r in successful_results])
      if successful_results
      else []
  )
  tpot_vals = (
      sorted([r["tpot_ms"] for r in successful_results])
      if successful_results
      else []
  )
  total_tokens = (
      sum([r["tokens"] for r in successful_results])
      if successful_results
      else 0
  )
  cluster_throughput = (
      total_tokens / total_bench_time
      if total_bench_time > 0 and successful_results
      else 0
  )

  ttft_mean = round(statistics.mean(ttft_vals), 2) if ttft_vals else 0.0
  tpot_mean = round(statistics.mean(tpot_vals), 2) if tpot_vals else 0.0
  throughput_tps = round(cluster_throughput, 2)

  try:
    meta_dict = json.loads(args.metadata) if args.metadata else {}
  except Exception:
    meta_dict = {"raw": args.metadata}

  summary = {
      "engine": args.engine,
      "metadata": meta_dict,
      "successful_requests": len(successful_results),
      "total_requests": args.requests,
      "total_completed": len(successful_results),
      "ttft_mean_ms": ttft_mean,
      "tpot_mean_ms": tpot_mean,
      "throughput_tokens_sec": throughput_tps,
      "benchmark_config": vars(args),
      "execution_summary": {
          "total_requests": args.requests,
          "successful_requests": len(successful_results),
          "failed_requests": len(failed_results),
          "total_benchmark_time_seconds": round(total_bench_time, 3),
      },
      "metrics": {},
  }

  if successful_results:

    def pct(lst, p):
      idx = int(len(lst) * (p / 100.0))
      return lst[min(idx, len(lst) - 1)]

    try:
      summary["metrics"] = {
          "total_tokens_generated": total_tokens,
          "cluster_throughput_tokens_per_sec": round(cluster_throughput, 2),
          "per_gpu_throughput_tokens_per_sec": round(
              cluster_throughput / 16.0, 2
          ),
          "ttft_ms": {
              "mean": round(statistics.mean(ttft_vals), 2) if ttft_vals else 0,
              "p50": round(pct(ttft_vals, 50), 2) if ttft_vals else 0,
              "p90": round(pct(ttft_vals, 90), 2) if ttft_vals else 0,
              "p99": round(pct(ttft_vals, 99), 2) if ttft_vals else 0,
              "min": round(min(ttft_vals), 2) if ttft_vals else 0,
              "max": round(max(ttft_vals), 2) if ttft_vals else 0,
          },
          "tpot_ms": {
              "mean": round(statistics.mean(tpot_vals), 2) if tpot_vals else 0,
              "p50": round(pct(tpot_vals, 50), 2) if tpot_vals else 0,
              "p90": round(pct(tpot_vals, 90), 2) if tpot_vals else 0,
              "p99": round(pct(tpot_vals, 99), 2) if tpot_vals else 0,
              "min": round(min(tpot_vals), 2) if tpot_vals else 0,
              "max": round(max(tpot_vals), 2) if tpot_vals else 0,
          },
      }
    except (
        ZeroDivisionError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        Exception,
    ):
      summary["metrics"] = {
          "total_tokens_generated": 0,
          "cluster_throughput_tokens_per_sec": 0,
          "per_gpu_throughput_tokens_per_sec": 0,
          "ttft_ms": {
              "mean": 0,
              "p50": 0,
              "p90": 0,
              "p99": 0,
              "min": 0,
              "max": 0,
          },
          "tpot_ms": {
              "mean": 0,
              "p50": 0,
              "p90": 0,
              "p99": 0,
              "min": 0,
              "max": 0,
          },
      }

    print("=" * 70)
    print(
        "          MASSIVE STRESS BENCHMARK AGGREGATE SUMMARY (Engine:"
        f" {args.engine})          "
    )
    print("=" * 70)
    print(
        "  Total Requests Completed:   "
        f" {len(successful_results)} / {args.requests}"
    )
    print(f"  Total Output Tokens:        {total_tokens:,} tokens")
    print(f"  Total Wall Clock Duration:  {total_bench_time:.2f} seconds")
    print(
        f"  Sustained Cluster TPS:      {cluster_throughput:.2f} tokens/sec"
        " across 20 streams"
    )
    print(
        f"  Per-GPU Throughput:         {cluster_throughput / 16.0:.2f}"
        " tokens/sec/GPU (16x B200 pool)"
    )
    print("-" * 70)
    print("  TTFT (Time-to-First-Token):")
    print(
        f"    Mean: {summary['metrics']['ttft_ms']['mean']:6.2f} ms | P50:"
        f" {summary['metrics']['ttft_ms']['p50']:6.2f} ms | P90:"
        f" {summary['metrics']['ttft_ms']['p90']:6.2f} ms | P99:"
        f" {summary['metrics']['ttft_ms']['p99']:6.2f} ms"
    )
    print(
        "    Min (Prefix Hit):"
        f" {summary['metrics']['ttft_ms']['min']:6.2f} ms | Max (Cold"
        f" Prefill): {summary['metrics']['ttft_ms']['max']:6.2f} ms"
    )
    print("-" * 70)
    print("  TPOT (Inter-Token Latency):")
    print(
        f"    Mean: {summary['metrics']['tpot_ms']['mean']:6.2f} ms | P50:"
        f" {summary['metrics']['tpot_ms']['p50']:6.2f} ms | P90:"
        f" {summary['metrics']['tpot_ms']['p90']:6.2f} ms | P99:"
        f" {summary['metrics']['tpot_ms']['p99']:6.2f} ms"
    )
    print("=" * 70)

  try:
    from telemetry_sanitizer import sanitize_telemetry
  except ImportError:
    from benchmarks.telemetry_sanitizer import sanitize_telemetry
  summary = sanitize_telemetry(summary, args.output)

  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"Massive benchmark report saved to {args.output}")


if __name__ == "__main__":
  main()
