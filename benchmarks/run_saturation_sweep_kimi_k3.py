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
Connects directly to the serving engine's OpenAI-compatible endpoint (default
port 8000) with 16-character cryptographic nonce injection to guarantee 0%
cache hit rate.
Normalizes per-GPU throughput by dividing aggregate tokens/sec by 16.0 (16x B200
GPUs in 2-node pool).
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import secrets
import statistics
import string
import sys
import threading
import time
import urllib.error
import urllib.request


def extract_chunk_text(chunk: dict) -> str | None:
  if "choices" in chunk and chunk["choices"]:
    choice = chunk["choices"][0]
    txt = choice.get("text")
    if txt is None and isinstance(choice.get("delta"), dict):
      txt = choice["delta"].get("content")
    return txt
  return None


# Approximate token count of SYNTHETIC_BASE_1K (authoritative count is reported by engine in usage block).
#
# Calibrated against the DAY-1 run rather than assumed. The passage was written to be
# "about 1024 tokens" and 1024 was used here, but Kimi K3's tokenizer packs it into
# roughly 889: the 8192-target cells issued 8 repetitions and measured 7,133 prompt
# tokens, the 32768-target cells issued 32 and measured 28,445 -- both solving to
# ~889/rep, so every cell undershot its target by 11-13%. The measurements were still
# sound (throughput is divided by the tokens the engine actually counted, and
# prompt_tokens_observed records them), but the grid labels overstated the load.
# At 889 the 8k and 32k targets land within ~2%. The 1024 target cannot: one
# repetition is the smallest prompt this construction can build, so that cell stays
# near 900 tokens regardless. Re-derive this constant if the passage or model changes.
BASE_TOKENS_APPROX = 889

# (input_tokens_target, output_tokens) — the documented DAY-1 ISL/OSL grid.
ISL_OSL_GRID = [
    (1024, 1024),
    (8192, 1024),
    (32768, 2048),
    (131072, 2048),
]

# Maximum simultaneous prompt tokens the 16x B200 KV cache is expected to hold.
# Cells above this are reported as skipped rather than run — see the printed
# SKIPPED lines and the "status" field in the results JSON.
MAX_INFLIGHT_PROMPT_TOKENS = 2_000_000

# Per-request context ceiling: the engine validates prompt + completion against its
# --context-length and rejects the whole request with HTTP 400 when the sum exceeds it.
# The DAY-1 grid's 131072/2048 row asks for 133,120 tokens against a 131,072-token
# window, so both of its runnable cells would have returned nothing but 400s and
# published a row of zeroes. Cells that cannot fit are skipped with a recorded reason
# instead, the same way the in-flight ceiling is handled.
MAX_CONTEXT_TOKENS = 131_072

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
) * 6  # ~889 tokens as measured on Kimi K3 (~4300 chars); see BASE_TOKENS_APPROX


def generate_unique_prompt(idx, c, isl_target):
  nonce = "".join(
      secrets.choice(string.ascii_letters + string.digits) for _ in range(16)
  )
  header = f"[Sweep C={c} ReqId={idx} ISL={isl_target} Nonce={nonce}] "
  reps = max(1, round(isl_target / BASE_TOKENS_APPROX))
  return header + (SYNTHETIC_BASE_1K * reps)


def execute_single_request(
    req_idx, c, isl_target, endpoint, model, max_tokens, temperature, api_key=""
):
  prompt = generate_unique_prompt(req_idx, c, isl_target)
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
  has_exact_usage = False

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
          txt = extract_chunk_text(chunk)
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
              has_exact_usage = True
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
        "token_count_source": "openai_usage" if has_exact_usage else "chunk_count_fallback",
        "prompt_tokens": prompt_tokens,
        "duration": duration,
        "throughput": throughput,
    }
  except Exception as e:
    return {"success": False, "error": str(e), "prompt_tokens": 0, "token_count_source": "none"}


class MetricsScraper:

  def __init__(self, endpoint, names_list, interval=2.0):
    self.endpoint = endpoint
    self.names_list = names_list
    self.interval = interval
    self.stop_event = threading.Event()
    self.peaks = {name: None for name in names_list}
    self.lock = threading.Lock()
    self.thread = None
    self.scrape_error_warned = False

  def _scrape_once(self):
    try:
      req = urllib.request.Request(
          self.endpoint,
          headers={"User-Agent": "KIMI3-Saturation-Sweep-Metrics/1.0"},
          method="GET",
      )
      with urllib.request.urlopen(req, timeout=5) as resp:
        content = resp.read().decode("utf-8", errors="replace")

      for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
          continue
        parts = line.split()
        if not parts:
          continue
        token = parts[0]
        m_name = token.split("{")[0]
        if m_name in self.peaks:
          try:
            val = float(parts[1])
            with self.lock:
              curr = self.peaks[m_name]
              if curr is None or val > curr:
                self.peaks[m_name] = val
          except (ValueError, IndexError):
            pass
    except Exception as e:
      if not self.scrape_error_warned:
        print(
            f"WARNING: Failed to scrape metrics endpoint {self.endpoint}: {e}",
            file=sys.stderr,
        )
        self.scrape_error_warned = True

  def _run_loop(self):
    while not self.stop_event.is_set():
      self._scrape_once()
      self.stop_event.wait(self.interval)

  def start(self):
    if not self.endpoint or not self.names_list:
      return
    self.thread = threading.Thread(target=self._run_loop, daemon=True)
    self.thread.start()

  def stop_and_get_peaks(self):
    if not self.endpoint or not self.names_list:
      return None
    self.stop_event.set()
    if self.thread and self.thread.is_alive():
      self.thread.join(timeout=3.0)
    self._scrape_once()

    with self.lock:
      result = dict(self.peaks)

    for name, val in result.items():
      if val is None:
        print(
            f"WARNING: Requested metric name '{name}' never appeared in scraped"
            f" output from {self.endpoint}; recording as null.",
            file=sys.stderr,
        )
    return result


def run_sweep_concurrency(
    c,
    isl_target,
    osl_target,
    requests_per_c,
    endpoint,
    model,
    api_key="",
    metrics_endpoint="",
    metrics_names=[],
):
  print(f"\n============================================================")
  print(
      f"Executing Saturation Sweep at ISL={isl_target}, OSL={osl_target},"
      f" Concurrency={c} ({requests_per_c} total requests)..."
  )
  print(f"============================================================")

  scraper = MetricsScraper(metrics_endpoint, metrics_names)
  scraper.start()

  results = []
  start_time = time.time()
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
    futures = [
        executor.submit(
            execute_single_request,
            i,
            c,
            isl_target,
            endpoint,
            model,
            osl_target,
            0.2,
            api_key,
        )
        for i in range(requests_per_c)
    ]
    for f in concurrent.futures.as_completed(futures):
      results.append(f.result())
  total_duration = time.time() - start_time

  peak_metrics = scraper.stop_and_get_peaks()

  successful = [r for r in results if r["success"]]
  failed = [r for r in results if not r["success"]]
  err_rate = (len(failed) / len(results)) * 100.0 if results else 100.0

  observed_prompts = [
      r["prompt_tokens"] for r in results if r.get("prompt_tokens", 0) > 0
  ]
  mean_prompt_tokens = (
      statistics.mean(observed_prompts) if observed_prompts else 0.0
  )

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
      sources_used = sorted(list(set(r.get("token_count_source", "unknown") for r in successful)))
      if len(sources_used) == 1:
        cell_source = sources_used[0]
      elif len(sources_used) > 1:
        cell_source = "mixed: " + ", ".join(sources_used)
      else:
        cell_source = "none"
      summary = {
          "isl_target": isl_target,
          "osl": osl_target,
          "concurrency": c,
          "prompt_tokens_observed": mean_prompt_tokens,
          "status": "ok",
          "requests": len(results),
          "successful": len(successful),
          "error_rate_pct": err_rate,
          "total_tokens": total_tokens,
          "token_count_source": cell_source,
          "total_duration_sec": total_duration,
          "aggregate_tok_s": agg_tok_s,
          "per_gpu_tok_s": per_gpu_tok_s,
          "per_user_tok_s": per_user_tok_s,
          "ttft_ms": ttft_stats,
          "tpot_ms": tpot_stats,
          "peak_metrics": peak_metrics,
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
          "isl_target": isl_target,
          "osl": osl_target,
          "concurrency": c,
          "prompt_tokens_observed": mean_prompt_tokens,
          "status": "error",
          "requests": len(results),
          "successful": 0,
          "error_rate_pct": 100.0,
          "total_tokens": 0,
          "token_count_source": "none",
          "total_duration_sec": 0.0,
          "aggregate_tok_s": 0.0,
          "per_gpu_tok_s": 0.0,
          "per_user_tok_s": 0.0,
          "ttft_ms": ttft_stats,
          "tpot_ms": tpot_stats,
          "peak_metrics": peak_metrics,
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
    return {
        "isl_target": isl_target,
        "osl": osl_target,
        "concurrency": c,
        "prompt_tokens_observed": mean_prompt_tokens,
        "status": "error",
        "requests": len(results),
        "successful": 0,
        "error_rate_pct": 100.0,
        "total_tokens": 0,
        "token_count_source": "none",
        "total_duration_sec": total_duration,
        "aggregate_tok_s": 0.0,
        "per_gpu_tok_s": 0.0,
        "per_user_tok_s": 0.0,
        "ttft_ms": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0},
        "tpot_ms": {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0},
        "peak_metrics": peak_metrics,
    }


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
      "--model", default="moonshotai/Kimi-K3", help="Served model ID"
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
  parser.add_argument(
      "--concurrency-levels",
      dest="concurrency_levels",
      default="1,8,32,128",
      type=lambda s: [int(x) for x in s.split(",") if x.strip()],
      help="Comma-separated concurrency levels to sweep",
  )
  parser.add_argument(
      "--max-inflight-prompt-tokens",
      dest="max_inflight_prompt_tokens",
      default=2_000_000,
      type=int,
      help="Maximum simultaneous in-flight prompt tokens before skipping cell",
  )
  parser.add_argument(
      "--max-context-tokens",
      dest="max_context_tokens",
      default=MAX_CONTEXT_TOKENS,
      type=int,
      help=(
          "Engine context window. Cells whose ISL+OSL exceeds it are skipped,"
          " because the engine rejects such requests outright with HTTP 400"
      ),
  )
  parser.add_argument(
      "--metrics-endpoint",
      default="",
      help="Optional Prometheus /metrics URL of the serving engine leader",
  )
  parser.add_argument(
      "--metrics-names",
      default="",
      help=(
          "Comma-separated exact metric names to sample (e.g. KV-cache usage,"
          " GPU memory)"
      ),
  )
  return parser.parse_args()


def main():
  args = parse_args()
  if args.dry_run:
    print("[INFO] Executing dry-run syntax and matrix validation...")

  print(f"\n=== KIMI K3 SATURATION SWEEP (Engine: {args.engine}) ===")
  start_bench_time = time.perf_counter()
  start_dt = datetime.now(timezone.utc)
  suite_start_ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  sweep_levels = args.concurrency_levels
  max_inflight = args.max_inflight_prompt_tokens
  max_context = args.max_context_tokens

  if not args.metrics_endpoint or not args.metrics_names:
    print(
        "NOTE: --metrics-endpoint/--metrics-names not supplied; peak KV-cache"
        " and GPU memory not captured for any cell."
    )
    metrics_names_list = []
  else:
    metrics_names_list = [
        x.strip() for x in args.metrics_names.split(",") if x.strip()
    ]

  sweep_results = []

  for isl, osl in ISL_OSL_GRID:
    for c in sweep_levels:
      if isl + osl > max_context:
        reason = (
            f"ISL+OSL={isl + osl:,} tokens exceeds the engine context window"
            f" MAX_CONTEXT_TOKENS={max_context:,}; the engine rejects such"
            " requests with HTTP 400 before any tokens are generated"
        )
        print(f"SKIPPED cell ISL={isl} OSL={osl} c={c} -> {reason}")
        sweep_results.append({
            "isl_target": isl,
            "osl": osl,
            "concurrency": c,
            "prompt_tokens_observed": 0,
            "status": "skipped",
            "reason": reason,
            "peak_metrics": None,
        })
        continue

      inflight_tokens = c * isl
      if inflight_tokens > max_inflight:
        reason = (
            f"{inflight_tokens:,} in-flight prompt tokens exceeds"
            f" MAX_INFLIGHT_PROMPT_TOKENS={max_inflight:,}"
        )
        print(f"SKIPPED cell ISL={isl} OSL={osl} c={c} -> {reason}")
        sweep_results.append({
            "isl_target": isl,
            "osl": osl,
            "concurrency": c,
            "prompt_tokens_observed": 0,
            "status": "skipped",
            "reason": reason,
            "peak_metrics": None,
        })
        continue

      print(
          f"RUN cell ISL={isl} OSL={osl} c={c} -> {inflight_tokens:,} in-flight"
          f" prompt tokens (within limit {max_inflight:,})"
      )
      if args.dry_run:
        sweep_results.append({
            "isl_target": isl,
            "osl": osl,
            "concurrency": c,
            "prompt_tokens_observed": isl,
            "status": "ok",
            "dry_run": True,
            "peak_metrics": None,
        })
        continue

      requests = max(c * 2, 8)  # Run at least 2 full waves per concurrency level
      res = run_sweep_concurrency(
          c,
          isl,
          osl,
          requests,
          args.endpoint,
          args.model,
          api_key=args.api_key,
          metrics_endpoint=args.metrics_endpoint,
          metrics_names=metrics_names_list,
      )
      sweep_results.append(res)
      time.sleep(2)

  if args.dry_run:
    run_count = sum(1 for r in sweep_results if r.get("status") != "skipped")
    skip_count = sum(1 for r in sweep_results if r.get("status") == "skipped")
    print(
        "\n[SUCCESS] Kimi K3 saturation sweep harness syntax and matrix verified"
        f" (RUN: {run_count}, SKIPPED: {skip_count})."
    )
    sys.exit(0)

  try:
    meta_dict = json.loads(args.metadata) if args.metadata else {}
  except (json.JSONDecodeError, TypeError, ValueError, Exception):
    meta_dict = {"raw": args.metadata}

  end_dt = datetime.now(timezone.utc)
  suite_end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  suite_duration_s = round((end_dt - start_dt).total_seconds(), 4)

  sources_used = sorted(list(set(r.get("token_count_source", "unknown") for r in sweep_results if r.get("status") == "ok")))
  if len(sources_used) == 1:
    agg_source = sources_used[0]
  elif len(sources_used) > 1:
    agg_source = "mixed: " + ", ".join(sources_used)
  else:
    agg_source = "none"

  grid_meta = {
      "ISL_OSL_GRID": ISL_OSL_GRID,
      "sweep_levels": sweep_levels,
      "MAX_INFLIGHT_PROMPT_TOKENS": max_inflight,
      "MAX_CONTEXT_TOKENS": max_context,
      "BASE_TOKENS_APPROX": BASE_TOKENS_APPROX,
      "suite_start_ts": suite_start_ts,
      "suite_end_ts": suite_end_ts,
      "suite_duration_s": suite_duration_s,
  }

  try:
    from telemetry_sanitizer import sanitize_telemetry
  except ImportError:
    from benchmarks.telemetry_sanitizer import sanitize_telemetry

  out_payload = {
      "engine": args.engine,
      "metadata": meta_dict,
      "token_count_source": agg_source,
      "grid": grid_meta,
      "sweep_results": sweep_results,
  }
  out_payload = sanitize_telemetry(out_payload, args.output)

  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as f:
    json.dump(
        out_payload,
        f,
        indent=2,
    )
  print(f"\nSaved full saturation sweep results to {args.output}")


if __name__ == "__main__":
  main()
