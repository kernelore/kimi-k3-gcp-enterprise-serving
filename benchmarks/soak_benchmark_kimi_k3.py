#!/usr/bin/env python3
"""Kimi K3 (2.8T MXFP4) 30-Minute Continuous Production Soak Test

Simulates 20 enterprise engineering developers across 18 concurrent active
streams (`--concurrency 18`) over 1,800 seconds (`--duration 1800`).
Executes continuous back-to-back prefill and generation cycles without pause
against TensorRT-LLM MPI serving endpoints on NVIDIA Blackwell GKE clusters (16x
B200 GPUs in 2-node pool).
Stresses PagedAttention KV cache, HBM3e thermal stability, and dual
RoCEv2/NVLink transport (`TP=8`).
Logs per-minute progress and exports aggregate 30-minute latency/throughput
percentiles and stability proof.
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

SOAK_PROMPTS = [
    (
        "Analyze the structural benefits of MXFP4 weight quantization and MXFP8"
        " activation scaling on NVIDIA Blackwell GPUs for Kimi K3 (2.8T"
        " parameters)."
    ),
    (
        "Derive the step-by-step Key-Value cache memory consumption formulas"
        " across 16x B200 HGX GPUs in a 2-node spot pool."
    ),
    (
        "Explain how GPUDirect RDMA over RoCEv2 fabric (3.2 Tbps per node, MTU"
        " 8896) eliminates inter-node bottlenecks during MoE routing across 896"
        " experts."
    ),
    (
        "Detail the mechanics of speculative decoding using Multi-Token"
        " Prediction (draft heads) and how it optimizes HBM3e bandwidth for"
        " Kimi K3."
    ),
    (
        "Write a production-grade Python async gateway with circuit breaking"
        " and exponential backoff for a Kimi K3 TensorRT-LLM inference cluster."
    ),
    (
        "Examine the graceful shutdown and failover behavior of 2-node"
        " TensorRT-LLM MPI serving pods during Kubernetes spot preemption"
        " events."
    ),
    (
        "Describe the performance speedups of mounting a 2,000 GB (2 TB)"
        " Hyperdisk ML volume in ReadOnlyMany (ROX) mode across GKE clusters."
    ),
    (
        "Explain why PagedAttention KV cache utilization percentage or queue"
        " depth is the ideal metric for custom HPA scaling of Kimi K3 pods."
    ),
    (
        "Compare the tensor core FLOPS and HBM3e memory bandwidth (2,880 GB"
        " total) of 16x B200 HGX Blackwell GPUs against H100 and A100"
        " architectures."
    ),
]


def parse_args():
  parser = argparse.ArgumentParser(
      description="Kimi K3 Continuous 30m Soak Test"
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
      "--model", default="moonshotai/Kimi-K3", help="Served model ID"
  )
  parser.add_argument(
      "--api-key",
      default=os.environ.get("GATEWAY_MASTER_KEY", ""),
      help="API key for Gateway authentication",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=18,
      help="Simulated active concurrent streams (Target: 18 streams / 20 devs)",
  )
  parser.add_argument(
      "--duration",
      type=int,
      default=1800,
      help="Total soak test duration in seconds (Target: 1800s / 30m)",
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
      default="soak_test_results_kimi_k3.json",
      help="Output JSON report path",
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
      "User-Agent": f"KIMI3-SoakTest/3.0 (StreamId-{req_id})",
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
      "token_count_source": "openai_usage" if has_exact_usage else "chunk_count_fallback",
      "prompt_tokens": prompt_tokens,
      "ttft_ms": ttft * 1000,
      "tpot_ms": tpot * 1000,
      "total_time_s": total_time,
      "req_throughput_tps": req_throughput,
      "completion_timestamp": time.time(),
  }


def main():
  args = parse_args()
  if args.dry_run:
    print("[SUCCESS] Kimi K3 soak test harness syntax verified.")
    sys.exit(0)

  print("=" * 75)
  print(
      "   Kimi K3 (2.8T MXFP4) 30-MINUTE CONTINUOUS SOAK TEST (Engine:"
      f" {args.engine})   "
  )
  print("=" * 75)
  print(f"  Endpoint:          {args.endpoint}")
  print(f"  Served Model:      {args.model}")
  print(f"  Engine:            {args.engine}")
  print(
      f"  Target Concurrency: {args.concurrency} simultaneous active streams"
      " (20 devs simulated)"
  )
  print(
      f"  Soak Duration:     {args.duration} seconds ({args.duration/60:.1f}"
      " minutes)"
  )
  print(f"  Max Tokens/Req:    {args.max_tokens} output tokens per generation")
  print("-" * 75)

  start_soak_time = time.perf_counter()
  start_dt = datetime.now(timezone.utc)
  suite_start_ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  results = []
  completed_requests = 0
  failed_requests = 0
  req_counter = 0
  last_log_minute = 0

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=args.concurrency
  ) as executor:
    active_futures = set()
    for _ in range(args.concurrency):
      req_counter += 1
      prompt = f"{SOAK_PROMPTS[req_counter % len(SOAK_PROMPTS)]} [Req={req_counter} Nonce={uuid.uuid4().hex[:12]}]"
      active_futures.add(
          executor.submit(
              execute_stream_request,
              req_counter,
              args.endpoint,
              args.model,
              prompt,
              args.max_tokens,
              args.temperature,
              args.api_key,
          )
      )

    consecutive_errors = 0
    while active_futures:
      done_set, active_futures = concurrent.futures.wait(
          active_futures, return_when=concurrent.futures.FIRST_COMPLETED
      )

      for future in done_set:
        res = future.result()
        results.append(res)
        if res["success"]:
          completed_requests += 1
          consecutive_errors = 0
        else:
          failed_requests += 1
          consecutive_errors += 1
          if consecutive_errors >= 5:
            print("\n" + "!" * 75)
            print(
                "ERROR: Detected port-forward tunnel collapse / consecutive"
                " HTTP 000 errors."
            )
            print(
                "Local workstation port-forward cannot sustain continuous soak"
                " load."
            )
            print(
                "RECOMMENDATION: Run benchmarks in-cluster via Kubernetes Job:"
            )
            print("  ./scripts/05_run_benchmarks.sh --in-cluster")
            print("!" * 75 + "\n")
            active_futures.clear()
            break

        elapsed = time.perf_counter() - start_soak_time

        if elapsed < args.duration and consecutive_errors < 5:
          req_counter += 1
          prompt = f"{SOAK_PROMPTS[req_counter % len(SOAK_PROMPTS)]} [Req={req_counter} Nonce={uuid.uuid4().hex[:12]}]"
          active_futures.add(
              executor.submit(
                  execute_stream_request,
                  req_counter,
                  args.endpoint,
                  args.model,
                  prompt,
                  args.max_tokens,
                  args.temperature,
                  args.api_key,
              )
          )

        current_minute = int(elapsed // 60)
        if current_minute > last_log_minute and current_minute <= (
            args.duration // 60
        ):
          last_log_minute = current_minute
          succ_results = [r for r in results if r["success"]]
          tot_toks = sum([r["tokens"] for r in succ_results])
          cum_tps = tot_toks / elapsed if elapsed > 0 else 0

          recent_window = [
              r
              for r in succ_results
              if r["completion_timestamp"] >= (time.time() - 65)
          ]
          if recent_window:
            rec_tps = sum([r["tokens"] for r in recent_window]) / 60.0
            rec_ttft = statistics.median([r["ttft_ms"] for r in recent_window])
            rec_tpot = statistics.median([r["tpot_ms"] for r in recent_window])
          else:
            rec_tps = cum_tps
            rec_ttft = 0
            rec_tpot = 0

          print(
              f"[Soak Minute {current_minute:02d}/{int(args.duration//60):02d}]"
              f" Elapsed: {elapsed:6.1f}s | Completed Reqs:"
              f" {completed_requests:4d} | Recent TPS: {rec_tps:6.1f} t/s |"
              f" Cumulative TPS: {cum_tps:6.1f} t/s | Recent TTFT P50:"
              f" {rec_ttft:5.1f}ms | TPOT P50: {rec_tpot:5.2f}ms | Errors:"
              f" {failed_requests}"
          )
          sys.stdout.flush()

  total_soak_time = time.perf_counter() - start_soak_time
  successful_results = [r for r in results if r["success"]]

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
      total_tokens / total_soak_time
      if total_soak_time > 0 and successful_results
      else 0
  )

  ttft_mean = round(statistics.mean(ttft_vals), 2) if ttft_vals else 0.0
  tpot_mean = round(statistics.mean(tpot_vals), 2) if tpot_vals else 0.0
  throughput_tps = round(cluster_throughput, 2)

  end_dt = datetime.now(timezone.utc)
  suite_end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  suite_duration_s = round((end_dt - start_dt).total_seconds(), 4)

  sources_used = sorted(list(set(r.get("token_count_source", "unknown") for r in successful_results)))
  if len(sources_used) == 1:
    agg_source = sources_used[0]
  elif len(sources_used) > 1:
    agg_source = "mixed: " + ", ".join(sources_used)
  else:
    agg_source = "none"

  try:
    meta_dict = json.loads(args.metadata) if args.metadata else {}
  except Exception:
    meta_dict = {"raw": args.metadata}

  summary = {
      "engine": args.engine,
      "metadata": meta_dict,
      "successful_requests": completed_requests,
      "total_requests": len(results),
      "total_completed": completed_requests,
      "token_count_source": agg_source,
      "ttft_mean_ms": ttft_mean,
      "tpot_mean_ms": tpot_mean,
      "throughput_tokens_sec": throughput_tps,
      "soak_config": {
          **vars(args),
          "suite_start_ts": suite_start_ts,
          "suite_end_ts": suite_end_ts,
          "suite_duration_s": suite_duration_s,
      },
      "execution_summary": {
          "total_duration_seconds": round(total_soak_time, 3),
          "total_requests_completed": completed_requests,
          "total_requests_failed": failed_requests,
          "success_rate_percent": round(
              100.0 * completed_requests / max(1, len(results)), 3
          ),
          "token_count_source": agg_source,
      },
      "metrics": {},
  }

  if successful_results:

    def pct(lst, p):
      idx = int(len(lst) * (p / 100.0))
      return lst[min(idx, len(lst) - 1)]

    try:
      summary["metrics"] = {
          "total_output_tokens": total_tokens,
          "sustained_cluster_tps": round(cluster_throughput, 2),
          "per_gpu_throughput_tps": round(cluster_throughput / 16.0, 2),
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
          "total_output_tokens": 0,
          "sustained_cluster_tps": 0,
          "per_gpu_throughput_tps": 0,
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

    print("=" * 75)
    print(
        "        30-MINUTE SOAK TEST AGGREGATE SUMMARY (Engine:"
        f" {args.engine})        "
    )
    print("=" * 75)
    print(
        f"  Total Duration:             {total_soak_time:.2f} seconds"
        f" ({total_soak_time/60:.2f} minutes)"
    )
    print(
        "  Total Requests Completed:   "
        f" {completed_requests:,} runs ({failed_requests} errors)"
    )
    print(f"  Total Output Tokens:        {total_tokens:,} tokens generated")
    print(
        f"  Sustained Cluster TPS:      {cluster_throughput:.2f} tokens/sec"
        " across 18 streams"
    )
    print(
        f"  Per-GPU Throughput:         {cluster_throughput / 16.0:.2f}"
        " tokens/sec/GPU (16x B200 pool)"
    )
    print("-" * 75)
    print("  TTFT (Time-to-First-Token) across 30m:")
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
    print("-" * 75)
    print("  TPOT (Inter-Token Latency) across 30m:")
    print(
        f"    Mean: {summary['metrics']['tpot_ms']['mean']:6.2f} ms | P50:"
        f" {summary['metrics']['tpot_ms']['p50']:6.2f} ms | P90:"
        f" {summary['metrics']['tpot_ms']['p90']:6.2f} ms | P99:"
        f" {summary['metrics']['tpot_ms']['p99']:6.2f} ms"
    )
    print("=" * 75)

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
  print(f"30-minute soak test report saved to {args.output}")


if __name__ == "__main__":
  main()
