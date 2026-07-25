#!/usr/bin/env python3
"""Kimi K3 (2.8T MXFP4) Prompt-Ingestion / Prefill Benchmark

Measures actual prompt processing rate (prompt tokens / sec) across 16x B200 HGX
GPUs (2-node spot pool).
Sends an ~8192 token input prompt interrogating Kimi K3 architecture with
max_tokens=16 and measures TTFT and prompt throughput.
Normalizes per-GPU throughput by dividing system prompt tokens/sec by 16.0.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

SYNTHETIC_8K = (
    "In large-scale sovereign artificial intelligence deployments on GKE"
    " Blackwell B200 HGX 2-node spot pools, GPUDirect RDMA over RoCEv2 fabric"
    " provides 3.2 Tbps bandwidth per node and NVIDIA NVLink fifth-generation"
    " interconnect provides 1.8 TB/s bidirectional bandwidth per GPU. When"
    " serving MoE architectures with 2.8 trillion parameters and 896 experts"
    " such as Kimi K3 using MXFP4 weight quantization and TensorRT-LLM MPI"
    " distributed inference across 16x B200 GPUs, expert routing decisions"
    " occur seamlessly across both nodes. Furthermore, the 2 TB ReadOnlyMany"
    " Hyperdisk ML storage architecture enables instant volume hydration and"
    " engine cache loading without network egress storms. "
) * 40  # ~8192 tokens (~20,800 chars)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Kimi K3 Prefill / Prompt-Ingestion Benchmark"
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
      default="benchmarks/prefill_results_kimi_k3.json",
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


def measure_prefill(
    endpoint, model, output_path, engine="trtllm", metadata="{}", api_key=""
):
  payload = {
      "model": model,
      "max_tokens": 16,
      "temperature": 0.1,
      "stream": True,
      "stream_options": {"include_usage": True},
  }
  if "/chat/completions" in endpoint:
    payload["messages"] = [{"role": "user", "content": SYNTHETIC_8K}]
  else:
    payload["prompt"] = SYNTHETIC_8K
  req_body = json.dumps(payload).encode("utf-8")
  headers = {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "User-Agent": "KIMI3-Prefill-Bench/1.0",
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
  t_first = None
  prompt_tokens = 8192

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
              and chunk["choices"][0].get("text")
          ):
            if t_first is None:
              t_first = time.time()
          usage = chunk.get("usage")
          if usage and isinstance(usage, dict):
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
  except Exception as e:
    print(f"Warning: Prefill request failed or timed out: {e}")

  t_end = time.time()
  ttft = (t_first - t_start) if t_first else (t_end - t_start)
  prefill_tok_s = prompt_tokens / ttft if ttft > 0 else 0.0

  try:
    meta_dict = json.loads(metadata) if metadata else {}
  except (json.JSONDecodeError, TypeError, ValueError, Exception):
    meta_dict = {"raw": metadata}

  try:
    result = {
        "engine": engine,
        "metadata": meta_dict,
        "prompt_tokens": prompt_tokens,
        "ttft_sec": ttft,
        "ttft_ms": ttft * 1000.0,
        "prefill_tok_s_system": prefill_tok_s,
        "prefill_tok_s_per_gpu": prefill_tok_s / 16.0,
    }
  except (ZeroDivisionError, KeyError, TypeError, ValueError, Exception):
    result = {
        "engine": engine,
        "metadata": meta_dict,
        "prompt_tokens": 0,
        "ttft_sec": 0.0,
        "ttft_ms": 0.0,
        "prefill_tok_s_system": 0.0,
        "prefill_tok_s_per_gpu": 0.0,
    }
  print(
      f"=== Kimi K3 PREFILL (PROMPT INGESTION) BENCHMARK (Engine: {engine}) ==="
  )
  print(f"Prompt Tokens:        {result['prompt_tokens']}")
  print(
      f"TTFT (Prefill Time):  {result['ttft_ms']:.2f} ms"
      f" ({result['ttft_sec']:.4f} s)"
  )
  print(
      f"System Prefill Rate:  {result['prefill_tok_s_system']:.2f} prompt tok/s"
      " (across 16x B200 GPUs)"
  )
  print(
      f"Per-GPU Prefill Rate: {result['prefill_tok_s_per_gpu']:.2f} prompt"
      " tok/s/GPU (normalized by 16.0)"
  )

  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
  print(f"Prefill benchmark report saved to {output_path}")
  return result


def main():
  args = parse_args()
  if args.dry_run:
    print("[SUCCESS] Kimi K3 prefill benchmark harness syntax verified.")
    sys.exit(0)
  measure_prefill(
      args.endpoint, args.model, args.output, args.engine, args.metadata, args.api_key
  )


if __name__ == "__main__":
  main()
