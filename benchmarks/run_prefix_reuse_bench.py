#!/usr/bin/env python3
"""Kimi K3 prefix-reuse benchmark: does caching a prefix actually save anything?

Every committed benchmark in this repo injects a random nonce at the head of the
prompt, which forces a 0% prefix-cache hit rate by construction. That is the
right call for a throughput number and the wrong instrument for the question
this file exists to answer, so none of them can say whether the radix cache, the
hierarchical (NVMe) cache tier, or --schedule-policy lpm do anything at all on
this model.

The question is not rhetorical for Kimi K3. Of its 93 layers only 24 are
full-attention (MLA); the other 69 are linear-attention (KDA) and carry
recurrent state rather than a per-token KV cache. A prefix cache stores KV. If
the recurrent state cannot be reused across requests, then re-serving a known
prefix still has to push it through 69 of 93 layers, and the ceiling on what
prefix caching can save is far below what the phrase "cache hit" suggests.
A hit-rate counter cannot distinguish those worlds -- it will happily report
100% while the engine recomputes most of the network.

So the headline here is not a hit rate. It is the slope of TTFT against prefix
length:

    cold  -- prefix never seen before                (baseline slope)
    warm  -- prefix served immediately before        (best case, GPU-resident)
    evict -- prefix served, then displaced from the GPU pool by enough
             intervening traffic, then requested again  (the arm that either
             justifies the NVMe cache tier or does not)

If warm's slope matches cold's, prefix reuse is worth nothing here no matter
what the counters say. If it is flat, reuse is total. In between,

    reuse_efficiency = 1 - slope_warm / slope_cold

is directly comparable to the architectural bound: reuse limited to the
full-attention layers alone lands near 24/93 = 0.258. That comparison is the
point of the whole benchmark, and it is reported explicitly rather than left for
a reader to compute.

The evict arm is the one that speaks to Tier 2. If TTFT there tracks cold, the
prefix is being recomputed and hierarchical cache on local NVMe is buying
nothing; if it tracks warm, it is being fetched back and the tier pays for
itself. Nothing else in the repo can tell those apart.

A fourth arm, mixed, runs a concurrent stream at a configurable target hit rate.
That is what exercises --schedule-policy lpm, which has never been exercised,
and it reports request-level and token-level hit rates separately because they
are different numbers and conflating them overstates the benefit.

Prompts come from the same generated corpus as run_realistic_sweep_kimi_k3, for
a reason that matters: if the filler text repeated, the engine would find shared
prefixes this harness never intended to create and every arm would be
contaminated. That corpus measures at 0.004% repeated 8-gram positions.

Offline use:
  run_prefix_reuse_bench.py --dry-run      # plan the matrix, spend nothing
  run_prefix_reuse_bench.py --self-check   # prove prefixes are shared exactly
                                           # and suffixes share nothing
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
  from run_realistic_sweep_kimi_k3 import DEFAULT_CHARS_PER_TOKEN
  from run_realistic_sweep_kimi_k3 import GeneratedCorpus
  from run_realistic_sweep_kimi_k3 import max_shared_prompt_prefix
  from run_realistic_sweep_kimi_k3 import scrape_counter_sum
except ImportError:  # invoked as benchmarks.run_prefix_reuse_bench
  from benchmarks.run_realistic_sweep_kimi_k3 import DEFAULT_CHARS_PER_TOKEN
  from benchmarks.run_realistic_sweep_kimi_k3 import GeneratedCorpus
  from benchmarks.run_realistic_sweep_kimi_k3 import max_shared_prompt_prefix
  from benchmarks.run_realistic_sweep_kimi_k3 import scrape_counter_sum

DEFAULT_CORPUS_SEED = 20260803

ARM_COLD = "cold"
ARM_WARM = "warm"
ARM_EVICT = "evicted"
ARM_MIXED = "mixed"
SLOPE_ARMS = (ARM_COLD, ARM_WARM, ARM_EVICT)
ALL_ARMS = (ARM_COLD, ARM_WARM, ARM_EVICT, ARM_MIXED)

# Kimi K3 layer split. Defaults, not assertions -- both are CLI-overridable so
# the reported bound stays honest if the architecture summary is ever revised.
DEFAULT_TOTAL_LAYERS = 93
DEFAULT_FULL_ATTENTION_LAYERS = 24

MAX_CONTEXT_TOKENS = 131_072

# Flushing the GPU KV pool is the expensive part of this benchmark: it costs one
# full pool's worth of prefill per evict measurement. These ceilings exist so a
# mis-detected pool size cannot silently turn a 20 minute run into a 3 hour one.
DEFAULT_MAX_FLUSH_TOKENS = 8_000_000
DEFAULT_MAX_TOTAL_FLUSH_TOKENS = 24_000_000

DEFAULT_CACHED_TOKENS_METRIC = "sglang:cached_tokens_total"
DEFAULT_PROMPT_TOKENS_METRIC = "sglang:prompt_tokens_total"

REQUEST_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class PromptFactory:
  """Builds shared prefixes and never-shared suffixes from one seeded corpus.

  Prefixes must be byte-identical across the requests that are supposed to hit,
  and suffixes must diverge immediately, or the arms stop measuring what they
  claim to. Both properties are asserted by --self-check rather than assumed.
  """

  def __init__(self, seed, chars_per_token):
    self.corpus = GeneratedCorpus(seed, chars_per_token)
    self.seed = seed
    self.chars_per_token = chars_per_token
    self._prefixes = {}
    self._suffixes_issued = 0

  def prefix(self, prefix_id, prefix_tokens):
    """Return the shared prefix for prefix_id, generating it once."""
    key = (prefix_id, prefix_tokens)
    if key not in self._prefixes:
      body = self.corpus.build_prompt(prefix_tokens)
      # A stable, human-legible header makes it obvious in a server-side log
      # which prefix a request belongs to, and it is part of the shared bytes so
      # it cannot break the match.
      self._prefixes[key] = (
          f"[Prefix id={prefix_id} tokens~{prefix_tokens}]\n\n{body}"
      )
    return self._prefixes[key]

  def suffix(self, suffix_tokens):
    """Return a suffix that has never been issued before."""
    self._suffixes_issued += 1
    body = self.corpus.build_prompt(suffix_tokens)
    return (
        f"\n\n[Query {self._suffixes_issued}]\n\n{body}\n\nSummarise the above."
    )

  def describe(self):
    profile = self.corpus.describe()
    profile["distinct_prefixes_built"] = len(self._prefixes)
    profile["suffixes_issued"] = self._suffixes_issued
    return profile


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------
#
# Deliberately not run_saturation_sweep_kimi_k3.execute_single_request. That
# path discards usage.prompt_tokens_details.cached_tokens, which is the only
# per-request evidence the server offers that a hit occurred, and it wraps every
# request in a thread pool -- fine for throughput, fatal for a serial TTFT
# measurement where a queued request would be scored as a slow prefill.


def issue_request(endpoint, model, prompt, max_tokens, api_key, label):
  """Issue one streaming completion and time its first token."""
  payload = {
      "model": model,
      "max_tokens": max_tokens,
      "temperature": 0.0,
      "ignore_eos": True,
      "stream": True,
      "stream_options": {"include_usage": True},
  }
  if "/chat/completions" in endpoint:
    payload["messages"] = [{"role": "user", "content": prompt}]
  else:
    payload["prompt"] = prompt

  headers = {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "User-Agent": f"KIMI3-Prefix-Reuse/1.0 ({label})",
  }
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

  req = urllib.request.Request(
      endpoint,
      data=json.dumps(payload).encode("utf-8"),
      headers=headers,
      method="POST",
  )

  t_start = time.time()
  t_first = None
  generated = 0
  prompt_tokens = 0
  cached_tokens = None

  try:
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
      for line in resp:
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded.startswith("data: "):
          continue
        body = decoded[6:]
        if body == "[DONE]":
          break
        try:
          chunk = json.loads(body)
        except json.JSONDecodeError:
          continue
        text = None
        if chunk.get("choices"):
          choice = chunk["choices"][0]
          text = choice.get("text")
          if text is None and isinstance(choice.get("delta"), dict):
            text = choice["delta"].get("content")
        if text:
          if t_first is None:
            t_first = time.time()
          generated += 1
        usage = chunk.get("usage")
        if usage:
          prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
          if usage.get("completion_tokens"):
            generated = usage["completion_tokens"]
          details = usage.get("prompt_tokens_details") or {}
          if isinstance(details, dict) and details.get("cached_tokens") is not None:
            cached_tokens = int(details["cached_tokens"])
    t_end = time.time()
  except (urllib.error.URLError, TimeoutError, OSError) as exc:
    return {"success": False, "error": str(exc), "label": label}

  ttft_ms = ((t_first or t_end) - t_start) * 1000.0
  return {
      "success": True,
      "label": label,
      "ttft_ms": ttft_ms,
      "e2e_ms": (t_end - t_start) * 1000.0,
      "generated_tokens": generated,
      "prompt_tokens": prompt_tokens,
      "cached_tokens": cached_tokens,
      "prompt_chars": len(prompt),
  }


def fetch_server_info(base_endpoint, api_key):
  """Best-effort GET /get_server_info; returns {} if the engine has no such route."""
  parsed = urllib.parse.urlsplit(base_endpoint)
  url = urllib.parse.urlunsplit(
      (parsed.scheme, parsed.netloc, "/get_server_info", "", "")
  )
  headers = {"User-Agent": "KIMI3-Prefix-Reuse/1.0 (server-info)"}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  try:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
      return json.loads(resp.read().decode("utf-8", errors="replace"))
  except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
    return {}


def extract_pool_tokens(server_info):
  """Pull the KV pool capacity out of whatever shape /get_server_info returned."""
  if not isinstance(server_info, dict):
    return None
  for key in ("max_total_num_tokens", "max_total_tokens"):
    value = server_info.get(key)
    if isinstance(value, (int, float)) and value > 0:
      return int(value)
  nested = server_info.get("internal_states")
  if isinstance(nested, list) and nested:
    return extract_pool_tokens(nested[0])
  if isinstance(nested, dict):
    return extract_pool_tokens(nested)
  return None


# ---------------------------------------------------------------------------
# Matrix planning
# ---------------------------------------------------------------------------


def parse_int_list(raw):
  try:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
  except ValueError:
    raise argparse.ArgumentTypeError(f"expected comma-separated integers: {raw!r}")
  if not values:
    raise argparse.ArgumentTypeError(f"expected at least one integer: {raw!r}")
  return values


def parse_float_list(raw):
  try:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
  except ValueError:
    raise argparse.ArgumentTypeError(f"expected comma-separated floats: {raw!r}")
  if not values:
    raise argparse.ArgumentTypeError(f"expected at least one float: {raw!r}")
  for value in values:
    if not 0.0 <= value <= 1.0:
      raise argparse.ArgumentTypeError(f"hit rate must be in [0, 1]: {value}")
  return values


def plan_points(
    arms,
    prefix_tokens_list,
    suffix_tokens,
    osl,
    repeats,
    evict_repeats,
    max_context_tokens,
    pool_tokens,
    eviction_overshoot,
    max_flush_tokens,
    max_total_flush_tokens,
):
  """Enumerate every slope-arm measurement, recording why any is skipped.

  Skips carry their reason into the results file. A cell that silently vanishes
  reads afterwards as a cell that was measured and found uninteresting.
  """
  points = []
  flush_committed = 0
  for arm in arms:
    if arm == ARM_MIXED:
      continue
    for prefix_tokens in prefix_tokens_list:
      point = {
          "arm": arm,
          "prefix_tokens": prefix_tokens,
          "suffix_tokens": suffix_tokens,
          "osl": osl,
          "repeats": evict_repeats if arm == ARM_EVICT else repeats,
          "status": "run",
          "reason": "",
          "flush_tokens": 0,
      }
      total = prefix_tokens + suffix_tokens + osl
      if total > max_context_tokens:
        point["status"] = "skipped"
        point["reason"] = (
            f"prefix+suffix+osl {total} exceeds context ceiling"
            f" {max_context_tokens}"
        )
      elif arm == ARM_EVICT:
        if not pool_tokens:
          point["status"] = "skipped"
          point["reason"] = (
              "KV pool size unknown: /get_server_info reported no"
              " max_total_num_tokens and --kv-pool-tokens was not supplied, so"
              " eviction cannot be guaranteed"
          )
        else:
          # One flush per sample, not per point: after the first re-measure the
          # prefix is resident again, so samples 2..n would silently be warm.
          flush = int(pool_tokens * eviction_overshoot)
          cost = flush * point["repeats"]
          if flush > max_flush_tokens:
            point["status"] = "skipped"
            point["reason"] = (
                f"flushing {flush} tokens per sample exceeds"
                f" --max-flush-tokens {max_flush_tokens}"
            )
          elif flush_committed + cost > max_total_flush_tokens:
            point["status"] = "skipped"
            point["reason"] = (
                f"{cost} flush tokens would take the run past"
                f" --max-total-flush-tokens {max_total_flush_tokens}"
                f" ({flush_committed} already committed)"
            )
          else:
            point["flush_tokens"] = flush
            flush_committed += cost
      points.append(point)
  return points


def plan_mixed(target_hit_rates, requests, concurrency, prefix_pool, enabled):
  if not enabled:
    return []
  return [
      {
          "arm": ARM_MIXED,
          "target_hit_rate": rate,
          "requests": requests,
          "concurrency": concurrency,
          "prefix_pool": prefix_pool,
          "status": "run",
          "reason": "",
      }
      for rate in target_hit_rates
  ]


def hit_schedule(requests, target_hit_rate):
  """Which requests reuse a prefix, spread evenly, first one forced to miss.

  Deterministic on purpose: an RNG here would make the interleave -- and so the
  queue state each request meets -- unreproducible between runs.
  """
  wanted = int(round(requests * target_hit_rate))
  wanted = max(0, min(wanted, max(0, requests - 1)))
  schedule = [False] * requests
  if wanted == 0:
    return schedule
  # Bresenham over the positions after the first, so hits are evenly spaced
  # rather than clustered at either end.
  slots = requests - 1
  placed = 0
  accumulator = 0
  for i in range(1, requests):
    accumulator += wanted
    if accumulator >= slots:
      accumulator -= slots
      schedule[i] = True
      placed += 1
      if placed == wanted:
        break
  return schedule


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def least_squares_slope(xs, ys):
  """Slope and intercept of y = a*x + b, or (None, None) if underdetermined."""
  n = len(xs)
  if n < 2:
    return None, None
  mean_x = statistics.fmean(xs)
  mean_y = statistics.fmean(ys)
  denom = sum((x - mean_x) ** 2 for x in xs)
  if denom == 0:
    return None, None
  slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
  return slope, mean_y - slope * mean_x


def summarise_point(point, samples):
  ok = [s for s in samples if s.get("success")]
  ttfts = sorted(s["ttft_ms"] for s in ok)
  cached = [s["cached_tokens"] for s in ok if s.get("cached_tokens") is not None]
  prompt_tokens = [s["prompt_tokens"] for s in ok if s.get("prompt_tokens")]
  summary = dict(point)
  summary["samples"] = len(samples)
  summary["successful"] = len(ok)
  summary["ttft_ms"] = {
      "min": round(ttfts[0], 3) if ttfts else None,
      "median": round(statistics.median(ttfts), 3) if ttfts else None,
      "max": round(ttfts[-1], 3) if ttfts else None,
  }
  summary["prompt_tokens_observed"] = (
      round(statistics.fmean(prompt_tokens), 1) if prompt_tokens else None
  )
  summary["cached_tokens_reported"] = (
      round(statistics.fmean(cached), 1) if cached else None
  )
  # Median, not mean: a single spot-instance hiccup during a 4-sample point
  # would drag a mean far enough to invent a slope that is not there.
  if ttfts and prompt_tokens:
    median_ttft_s = statistics.median(ttfts) / 1000.0
    summary["effective_prefill_tok_s"] = (
        round(statistics.fmean(prompt_tokens) / median_ttft_s, 2)
        if median_ttft_s > 0
        else None
    )
  else:
    summary["effective_prefill_tok_s"] = None
  return summary


def analyse_slopes(point_summaries, total_layers, full_attention_layers):
  """Turn per-arm TTFT-vs-prefix-length slopes into a reuse-efficiency verdict."""
  by_arm = {}
  for summary in point_summaries:
    if summary.get("status") != "run" or not summary.get("successful"):
      continue
    median = summary["ttft_ms"]["median"]
    if median is None:
      continue
    by_arm.setdefault(summary["arm"], []).append(
        (summary["prefix_tokens"], median)
    )

  slopes = {}
  for arm, pairs in by_arm.items():
    pairs.sort()
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    slope, intercept = least_squares_slope(xs, ys)
    slopes[arm] = {
        "points": len(pairs),
        "ms_per_prefix_token": round(slope, 6) if slope is not None else None,
        "intercept_ms": round(intercept, 3) if intercept is not None else None,
        "note": (
            "" if slope is not None
            else "needs at least two prefix lengths to fit a slope"
        ),
    }

  verdict = {
      "slopes": slopes,
      "full_attention_layer_fraction": round(
          full_attention_layers / total_layers, 4
      ) if total_layers else None,
      "reuse_efficiency": {},
      "interpretation": [],
  }

  cold = slopes.get(ARM_COLD, {}).get("ms_per_prefix_token")
  if not cold or cold <= 0:
    verdict["interpretation"].append(
        "No usable cold-arm slope, so nothing here can be normalised. TTFT must"
        " grow with prefix length on cold prompts; if it did not, the prompts"
        " were being served from a cache the cold arm was supposed to miss."
    )
    return verdict

  bound = full_attention_layers / total_layers if total_layers else None
  for arm in (ARM_WARM, ARM_EVICT):
    warm = slopes.get(arm, {}).get("ms_per_prefix_token")
    if warm is None:
      continue
    efficiency = 1.0 - (warm / cold)
    verdict["reuse_efficiency"][arm] = round(efficiency, 4)
    if efficiency < 0.05:
      verdict["interpretation"].append(
          f"{arm}: TTFT still scales with prefix length at {warm:.6f} ms/token"
          f" against cold's {cold:.6f}. The prefix is being recomputed. Prefix"
          " caching is not paying for itself in this configuration."
      )
    elif bound is not None and efficiency < bound * 1.5:
      verdict["interpretation"].append(
          f"{arm}: reuse efficiency {efficiency:.3f} sits near the"
          f" full-attention-only bound of {bound:.3f}"
          f" ({full_attention_layers}/{total_layers} layers). Consistent with"
          " the KV cache being reused while the linear-attention layers"
          " recompute their recurrent state from the start of the prefix."
      )
    else:
      verdict["interpretation"].append(
          f"{arm}: reuse efficiency {efficiency:.3f} exceeds the"
          f" full-attention-only bound of {bound:.3f}, so more than the MLA KV"
          " is being carried across requests."
      )

  warm_eff = verdict["reuse_efficiency"].get(ARM_WARM)
  evict_eff = verdict["reuse_efficiency"].get(ARM_EVICT)
  if warm_eff is not None and evict_eff is not None:
    if warm_eff <= 0.05:
      verdict["interpretation"].append(
          "Tier 2 verdict: undecidable. The warm arm shows no reuse benefit at"
          " all, so there is no benefit for the NVMe tier to preserve across"
          " eviction and this run says nothing about hierarchical cache."
      )
    elif evict_eff >= warm_eff * 0.5:
      verdict["interpretation"].append(
          f"Tier 2 verdict: the evicted arm retains {evict_eff / warm_eff:.0%}"
          " of the warm arm's benefit after the GPU pool was flushed, so the"
          " prefix is being fetched back rather than recomputed. Hierarchical"
          " cache on local NVMe is doing work."
      )
    else:
      verdict["interpretation"].append(
          f"Tier 2 verdict: the evicted arm retains only"
          f" {evict_eff / warm_eff:.0%} of the warm arm's benefit, so a flushed"
          " prefix is largely recomputed. Hierarchical cache on local NVMe is"
          " not justified by this measurement."
      )
  return verdict


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def flush_pool(args, factory, flush_tokens):
  """Push distinct traffic through until the GPU KV pool has turned over."""
  issued = 0
  requests = 0
  chunk_tokens = max(1024, args.suffix_tokens)
  print(
      f"    flushing ~{flush_tokens} tokens of KV pool with distinct prompts...",
      flush=True,
  )
  while issued < flush_tokens:
    prompt = factory.prefix(f"flush-{requests}", chunk_tokens)
    result = issue_request(
        args.endpoint, args.model, prompt, 1, args.api_key, f"flush-{requests}"
    )
    requests += 1
    issued += result.get("prompt_tokens") or chunk_tokens
    if not result.get("success"):
      print(
          f"    WARNING: flush request {requests} failed: {result.get('error')}",
          file=sys.stderr,
      )
      if requests > 4 and issued == 0:
        print(
            "    ERROR: flush is making no progress; abandoning this point.",
            file=sys.stderr,
        )
        return None
  return {"flush_requests": requests, "flush_tokens_issued": issued}


def run_slope_point(args, factory, point):
  """Measure one (arm, prefix_length) point serially."""
  arm = point["arm"]
  prefix_tokens = point["prefix_tokens"]
  samples = []
  detail = {}

  for rep in range(point["repeats"]):
    prefix_id = (
        f"{arm}-{prefix_tokens}-{rep}"
        if arm == ARM_COLD
        else f"{arm}-{prefix_tokens}-shared"
    )
    prefix = factory.prefix(prefix_id, prefix_tokens)

    if arm in (ARM_WARM, ARM_EVICT):
      # Populate: the priming request is never measured, only its side effect.
      priming = issue_request(
          args.endpoint,
          args.model,
          prefix + factory.suffix(args.suffix_tokens),
          1,
          args.api_key,
          f"{arm}-prime-{prefix_tokens}-{rep}",
      )
      if not priming.get("success"):
        print(
            f"    WARNING: priming request failed: {priming.get('error')}",
            file=sys.stderr,
        )

    if arm == ARM_EVICT:
      flushed = flush_pool(args, factory, point["flush_tokens"])
      if flushed is None:
        return dict(point, status="failed", reason="flush made no progress")
      detail = flushed

    if arm == ARM_COLD:
      # A cold prefix must be genuinely unseen, so it is built fresh per rep and
      # never primed.
      pass

    samples.append(
        issue_request(
            args.endpoint,
            args.model,
            prefix + factory.suffix(args.suffix_tokens),
            args.osl,
            args.api_key,
            f"{arm}-measure-{prefix_tokens}-{rep}",
        )
    )

  summary = summarise_point(point, samples)
  summary.update(detail)
  return summary


def run_mixed_arm(args, factory, spec):
  """Concurrent stream at a target hit rate; exercises --schedule-policy lpm."""
  requests = spec["requests"]
  schedule = hit_schedule(requests, spec["target_hit_rate"])
  prefix_pool = spec["prefix_pool"]

  plan = []
  next_new = 0
  for i, is_hit in enumerate(schedule):
    if is_hit and next_new > 0:
      prefix_id = f"mixed-{spec['target_hit_rate']}-{i % min(next_new, prefix_pool)}"
      plan.append((i, prefix_id, True))
    else:
      prefix_id = f"mixed-{spec['target_hit_rate']}-{next_new}"
      next_new += 1
      plan.append((i, prefix_id, False))

  prefix_tokens = args.prefix_tokens_list[-1]

  # Prime the reuse pool serially first. Without this the stream can dispatch a
  # request that was planned as a hit before the request establishing its prefix
  # has finished, which shows up as an unexplained shortfall against the target
  # rate rather than as the scheduling artefact it is.
  reuse_ids = sorted({pid for _, pid, is_hit in plan if is_hit})
  for prefix_id in reuse_ids:
    issue_request(
        args.endpoint,
        args.model,
        factory.prefix(prefix_id, prefix_tokens)
        + factory.suffix(args.suffix_tokens),
        1,
        args.api_key,
        f"mixed-prime-{prefix_id}",
    )

  prompts = [
      (
          idx,
          factory.prefix(prefix_id, prefix_tokens)
          + factory.suffix(args.suffix_tokens),
          expected_hit,
      )
      for idx, prefix_id, expected_hit in plan
  ]

  metric_before = scrape_counter_sum(args.metrics_endpoint, args.cached_tokens_metric)
  prompt_before = scrape_counter_sum(args.metrics_endpoint, args.prompt_tokens_metric)

  started = time.time()
  results = []
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=spec["concurrency"]
  ) as pool:
    futures = {
        pool.submit(
            issue_request,
            args.endpoint,
            args.model,
            prompt,
            args.osl,
            args.api_key,
            f"mixed-{idx}",
        ): expected_hit
        for idx, prompt, expected_hit in prompts
    }
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      result["expected_hit"] = futures[future]
      results.append(result)
  duration = time.time() - started

  metric_after = scrape_counter_sum(args.metrics_endpoint, args.cached_tokens_metric)
  prompt_after = scrape_counter_sum(args.metrics_endpoint, args.prompt_tokens_metric)

  ok = [r for r in results if r.get("success")]
  ttfts = sorted(r["ttft_ms"] for r in ok)
  expected_hits = sum(1 for r in results if r.get("expected_hit"))
  total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok)
  reported_cached = [r["cached_tokens"] for r in ok if r.get("cached_tokens") is not None]

  observed_token_hit_rate = None
  if (
      metric_before is not None
      and metric_after is not None
      and prompt_before is not None
      and prompt_after is not None
      and prompt_after > prompt_before
  ):
    observed_token_hit_rate = round(
        (metric_after - metric_before) / (prompt_after - prompt_before), 4
    )
  elif reported_cached and total_prompt_tokens > 0:
    observed_token_hit_rate = round(sum(reported_cached) / total_prompt_tokens, 4)

  # Request-level and token-level hit rates are different numbers. A run where
  # 80% of requests hit but the shared prefix is a third of each prompt has a
  # token-level rate near 0.27, and it is the token-level one that bounds any
  # prefill saving.
  expected_token_hit_rate = None
  if requests:
    expected_token_hit_rate = round(
        (expected_hits * prefix_tokens)
        / (requests * (prefix_tokens + args.suffix_tokens)),
        4,
    )

  return {
      "arm": ARM_MIXED,
      "status": "run",
      "target_hit_rate": spec["target_hit_rate"],
      "concurrency": spec["concurrency"],
      "requests": requests,
      "successful": len(ok),
      "prefix_tokens": prefix_tokens,
      "suffix_tokens": args.suffix_tokens,
      "osl": args.osl,
      "duration_s": round(duration, 3),
      "request_level_hit_rate_planned": (
          round(expected_hits / requests, 4) if requests else None
      ),
      "token_level_hit_rate_expected": expected_token_hit_rate,
      "token_level_hit_rate_observed": observed_token_hit_rate,
      "hit_rate_source": (
          f"{args.cached_tokens_metric} / {args.prompt_tokens_metric} delta"
          if metric_after is not None and prompt_after is not None
          else (
              "usage.prompt_tokens_details.cached_tokens"
              if reported_cached
              else "not measured"
          )
      ),
      "ttft_ms": {
          "p50": round(statistics.median(ttfts), 3) if ttfts else None,
          "p99": round(ttfts[int(len(ttfts) * 0.99) - 1], 3) if ttfts else None,
          "max": round(ttfts[-1], 3) if ttfts else None,
      },
      "aggregate_output_tok_s": (
          round(sum(r.get("generated_tokens") or 0 for r in ok) / duration, 2)
          if duration > 0
          else None
      ),
  }


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------


def report_self_check(args):
  """Prove offline that the arms can measure what they claim to."""
  factory = PromptFactory(args.corpus_seed, args.chars_per_token)
  prefix_tokens = args.prefix_tokens_list[0]

  shared_a = factory.prefix("selfcheck-shared", prefix_tokens)
  shared_b = factory.prefix("selfcheck-shared", prefix_tokens)
  cold_a = factory.prefix("selfcheck-cold-0", prefix_tokens)
  cold_b = factory.prefix("selfcheck-cold-1", prefix_tokens)
  suffixes = [factory.suffix(args.suffix_tokens) for _ in range(8)]

  shared_identical = shared_a == shared_b
  cold_distinct = cold_a != cold_b
  cold_shared_words = max_shared_prompt_prefix([cold_a, cold_b])
  suffix_shared_words = max_shared_prompt_prefix(suffixes)

  full_prompts = [shared_a + suffix for suffix in suffixes]
  # startswith, not max_shared_prompt_prefix: that helper caps its comparison at
  # 64 words, which is shorter than any prefix worth testing, so using it here
  # would report a pass at 64 no matter how early the prompts diverged.
  prompts_carry_whole_prefix = all(p.startswith(shared_a) for p in full_prompts)
  prefix_words = len(shared_a.split())

  print("=== PREFIX-REUSE SELF-CHECK ===")
  print(f"  Prefix length target:          {prefix_tokens} tokens")
  print(f"  Prefix rendered:               {prefix_words} words,"
        f" {len(shared_a)} chars")
  print(f"  Shared prefix byte-identical:  {shared_identical}")
  print(f"  Cold prefixes distinct:        {cold_distinct}")
  print(f"  Cold prefixes share:           {cold_shared_words} leading words"
        f" (ceiling {args.max_cold_shared_words})")
  print(f"  Suffixes share:                {suffix_shared_words} leading words"
        f" (ceiling {args.max_cold_shared_words})")
  print(f"  Same-prefix prompts carry the whole {prefix_words}-word prefix:"
        f" {prompts_carry_whole_prefix}")

  failures = []
  if not shared_identical:
    failures.append("the shared prefix is not byte-identical across calls")
  if not cold_distinct:
    failures.append("two cold prefixes came out identical")
  if cold_shared_words > args.max_cold_shared_words:
    failures.append(
        f"cold prefixes share {cold_shared_words} leading words, so the cold arm"
        " would partially hit"
    )
  if suffix_shared_words > args.max_cold_shared_words:
    failures.append(
        f"suffixes share {suffix_shared_words} leading words, which would extend"
        " the effective shared prefix beyond the one under test"
    )
  if not prompts_carry_whole_prefix:
    failures.append(
        "same-prefix prompts do not carry the whole prefix, so the warm arm"
        " cannot hit on all of it"
    )

  if args.kv_pool_tokens:
    flush = int(args.kv_pool_tokens * args.eviction_overshoot)
    total = flush * args.evict_repeats * len(args.prefix_tokens_list)
    print(f"  Evict arm flush per sample:    ~{flush} tokens"
          f" (ceiling {args.max_flush_tokens})")
    print(f"  Evict arm flush per run:       ~{total} tokens"
          f" (ceiling {args.max_total_flush_tokens})")
    if flush > args.max_flush_tokens:
      failures.append(
          f"flushing {flush} tokens exceeds --max-flush-tokens"
          f" {args.max_flush_tokens}; the evicted arm would be skipped"
      )
    elif total > args.max_total_flush_tokens:
      failures.append(
          f"the evicted arm would flush {total} tokens over the run, past"
          f" --max-total-flush-tokens {args.max_total_flush_tokens}; later"
          " prefix lengths would be skipped"
      )
  else:
    print("  Evict arm flush per sample:    unknown until the server reports"
          " its KV pool size")

  if failures:
    print("\nVERDICT: self-check FAILED")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print("\nVERDICT: prefix and suffix construction is sound.")
  return 0


def report_dry_run(args, points, mixed_specs):
  print(f"=== KIMI K3 PREFIX-REUSE BENCH (Engine: {args.engine}) ===")
  for point in points:
    if point["status"] == "run":
      extra = (
          f", flush ~{point['flush_tokens']} tok"
          if point.get("flush_tokens")
          else ""
      )
      print(
          f"RUN  {point['arm']:<8} prefix={point['prefix_tokens']:>6}"
          f" suffix={point['suffix_tokens']} osl={point['osl']}"
          f" x{point['repeats']}{extra}"
      )
    else:
      print(
          f"SKIP {point['arm']:<8} prefix={point['prefix_tokens']:>6}"
          f" -> {point['reason']}"
      )
  for spec in mixed_specs:
    print(
        f"RUN  {ARM_MIXED:<8} hit_rate={spec['target_hit_rate']}"
        f" requests={spec['requests']} c={spec['concurrency']}"
    )

  run_points = [p for p in points if p["status"] == "run"]
  measured = sum(p["repeats"] for p in run_points)
  primed = sum(
      p["repeats"] for p in run_points if p["arm"] in (ARM_WARM, ARM_EVICT)
  )
  flush_tokens = sum(p["repeats"] * p["flush_tokens"] for p in run_points)
  mixed_requests = sum(s["requests"] for s in mixed_specs)
  print("")
  print(f"[INFO] Measured requests:  {measured}")
  print(f"[INFO] Priming requests:   {primed}")
  print(f"[INFO] Mixed-arm requests: {mixed_requests}")
  print(f"[INFO] Flush tokens total: {flush_tokens}")
  if flush_tokens:
    print("       Flushing is the dominant cost of this benchmark: it prefills"
          " a full KV pool per evicted-arm sample.")
  print("")
  print(
      f"[SUCCESS] Kimi K3 prefix-reuse harness syntax and matrix verified"
      f" (RUN: {len(run_points)}, SKIPPED: {len(points) - len(run_points)},"
      f" MIXED: {len(mixed_specs)})."
  )
  return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Kimi K3 prefix-reuse / hierarchical-cache benchmark"
  )
  parser.add_argument("--dry-run", action="store_true",
                      help="Plan the matrix and exit without contacting a server")
  parser.add_argument("--self-check", action="store_true",
                      help="Prove offline that prefixes are shared and suffixes are not")
  parser.add_argument("--endpoint",
                      default="http://localhost:8000/v1/completions",
                      help="Completions endpoint URL")
  parser.add_argument("--model", default="moonshotai/Kimi-K3",
                      help="Served model ID")
  parser.add_argument("--output",
                      default="benchmarks/prefix_reuse_results_kimi_k3.json",
                      help="Output JSON path")
  parser.add_argument("--engine", default="sglang",
                      help="Inference engine (sglang or trtllm)")
  parser.add_argument("--metadata", default="{}",
                      help="JSON string of engine metadata")
  parser.add_argument("--api-key", default="",
                      help="Optional API key for gateway authentication")

  parser.add_argument("--arms", default=",".join(ALL_ARMS),
                      help=f"Comma-separated subset of {','.join(ALL_ARMS)}")
  parser.add_argument("--prefix-tokens", dest="prefix_tokens", default="1024,4096,16384",
                      help="Comma-separated shared-prefix lengths; at least two"
                           " are needed to fit a slope")
  parser.add_argument("--suffix-tokens", type=int, default=256,
                      help="Per-request unique suffix length in tokens")
  parser.add_argument("--osl", type=int, default=32,
                      help="Output tokens per measured request; kept small so"
                           " TTFT dominates")
  parser.add_argument("--repeats", type=int, default=4,
                      help="Measured samples per slope point")
  parser.add_argument("--evict-repeats", type=int, default=2,
                      help="Measured samples per evicted-arm point; lower than"
                           " --repeats because each one costs a full KV-pool"
                           " flush")

  parser.add_argument("--target-hit-rate", dest="target_hit_rate", default="0.5,0.8",
                      help="Comma-separated request-level hit rates for the mixed arm")
  parser.add_argument("--mixed-requests", type=int, default=64,
                      help="Requests issued per mixed-arm hit rate")
  parser.add_argument("--mixed-concurrency", type=int, default=16,
                      help="In-flight requests during the mixed arm")
  parser.add_argument("--mixed-prefix-pool", type=int, default=8,
                      help="Distinct prefixes the mixed arm reuses from")

  parser.add_argument("--kv-pool-tokens", type=int, default=0,
                      help="GPU KV pool capacity in tokens; 0 autodetects via"
                           " /get_server_info. The evicted arm is skipped if"
                           " neither is available.")
  parser.add_argument("--eviction-overshoot", type=float, default=1.25,
                      help="Multiple of the KV pool to flush before re-measuring")
  parser.add_argument("--max-flush-tokens", type=int, default=DEFAULT_MAX_FLUSH_TOKENS,
                      help="Refuse an evicted-arm sample that would cost more"
                           " than this many flush tokens")
  parser.add_argument("--max-total-flush-tokens", type=int,
                      default=DEFAULT_MAX_TOTAL_FLUSH_TOKENS,
                      help="Whole-run flush budget. Flushing is prefill and"
                           " prefill is billed, so a mis-detected pool size is"
                           " a cost incident, not a slow test.")
  parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS,
                      help="Served context ceiling")

  parser.add_argument("--metrics-endpoint", default="",
                      help="Prometheus /metrics URL used to corroborate hit rates")
  parser.add_argument("--cached-tokens-metric", default=DEFAULT_CACHED_TOKENS_METRIC,
                      help="Counter of prompt tokens served from cache")
  parser.add_argument("--prompt-tokens-metric", default=DEFAULT_PROMPT_TOKENS_METRIC,
                      help="Counter of prompt tokens processed")

  parser.add_argument("--corpus-seed", type=int, default=DEFAULT_CORPUS_SEED,
                      help="Seed for the shared non-repetitive corpus")
  parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
                      help="Characters per token assumed when sizing prompts")
  parser.add_argument("--max-cold-shared-words", type=int, default=8,
                      help="Self-check ceiling on accidental sharing between"
                           " prompts that must not share")
  parser.add_argument("--total-layers", type=int, default=DEFAULT_TOTAL_LAYERS,
                      help="Model layer count, for the reuse-efficiency bound")
  parser.add_argument("--full-attention-layers", type=int,
                      default=DEFAULT_FULL_ATTENTION_LAYERS,
                      help="Layers holding a reusable KV cache; the rest are"
                           " linear-attention and hold recurrent state")

  args = parser.parse_args(argv)
  args.prefix_tokens_list = parse_int_list(args.prefix_tokens)
  args.prefix_tokens_list.sort()
  args.target_hit_rate_list = parse_float_list(args.target_hit_rate)
  args.arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
  unknown = [a for a in args.arm_list if a not in ALL_ARMS]
  if unknown:
    parser.error(f"unknown arm(s): {', '.join(unknown)}")
  if args.total_layers <= 0 or not 0 <= args.full_attention_layers <= args.total_layers:
    parser.error("--full-attention-layers must be within [0, --total-layers]")
  # --mixed-prefix-pool reaches a modulo in the mixed arm, so zero here is a
  # ZeroDivisionError partway through a paid run rather than a usage error now.
  for flag, value in (
      ("--mixed-prefix-pool", args.mixed_prefix_pool),
      ("--mixed-requests", args.mixed_requests),
      ("--mixed-concurrency", args.mixed_concurrency),
      ("--repeats", args.repeats),
      ("--evict-repeats", args.evict_repeats),
      ("--suffix-tokens", args.suffix_tokens),
  ):
    if value < 1:
      parser.error(f"{flag} must be at least 1")
  if args.eviction_overshoot <= 1.0:
    parser.error(
        "--eviction-overshoot must exceed 1.0, or the flush cannot be relied on"
        " to displace the primed prefix"
    )
  return args


def main(argv=None):
  args = parse_args(argv)

  if args.self_check:
    return report_self_check(args)

  server_info = {}
  pool_tokens = args.kv_pool_tokens
  if not args.dry_run:
    server_info = fetch_server_info(args.endpoint, args.api_key)
    if not pool_tokens:
      pool_tokens = extract_pool_tokens(server_info) or 0
      if pool_tokens:
        print(f"[INFO] KV pool reported as {pool_tokens} tokens.")
      else:
        print(
            "[WARN] Could not read the KV pool size from /get_server_info."
            " Pass --kv-pool-tokens to enable the evicted arm.",
            file=sys.stderr,
        )

  points = plan_points(
      args.arm_list,
      args.prefix_tokens_list,
      args.suffix_tokens,
      args.osl,
      args.repeats,
      args.evict_repeats,
      args.max_context_tokens,
      pool_tokens,
      args.eviction_overshoot,
      args.max_flush_tokens,
      args.max_total_flush_tokens,
  )
  mixed_specs = plan_mixed(
      args.target_hit_rate_list,
      args.mixed_requests,
      args.mixed_concurrency,
      args.mixed_prefix_pool,
      ARM_MIXED in args.arm_list,
  )

  if args.dry_run:
    return report_dry_run(args, points, mixed_specs)

  if len(args.prefix_tokens_list) < 2:
    print(
        "[WARN] Only one prefix length was requested, so no slope can be fitted"
        " and the reuse-efficiency verdict will be empty.",
        file=sys.stderr,
    )

  factory = PromptFactory(args.corpus_seed, args.chars_per_token)
  summaries = []
  for point in points:
    if point["status"] != "run":
      print(f"SKIP {point['arm']} prefix={point['prefix_tokens']}:"
            f" {point['reason']}")
      summaries.append(dict(point))
      continue
    print(f"--> {point['arm']} prefix={point['prefix_tokens']} ...", flush=True)
    # Every point already measured cost GPU minutes, and an evicted point cost a
    # full pool flush on top. Losing all of that to one bad response at the last
    # point is not an acceptable failure mode, so a point that raises is recorded
    # as failed and the run continues to the ones that can still be salvaged.
    try:
      summaries.append(run_slope_point(args, factory, point))
    except (ZeroDivisionError, KeyError, ValueError, OSError) as exc:
      print(f"    ERROR: {point['arm']} prefix={point['prefix_tokens']}"
            f" failed: {exc}", file=sys.stderr)
      summaries.append(dict(point, status="failed", reason=str(exc)))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"    ERROR: {point['arm']} prefix={point['prefix_tokens']} raised"
            f" {type(exc).__name__}: {exc}", file=sys.stderr)
      summaries.append(
          dict(point, status="failed", reason=f"{type(exc).__name__}: {exc}")
      )

  mixed_results = []
  for spec in mixed_specs:
    print(f"--> mixed hit_rate={spec['target_hit_rate']}"
          f" c={spec['concurrency']} ...", flush=True)
    try:
      mixed_results.append(run_mixed_arm(args, factory, spec))
    except (ZeroDivisionError, KeyError, ValueError, OSError) as exc:
      print(f"    ERROR: mixed arm at {spec['target_hit_rate']} failed: {exc}",
            file=sys.stderr)
      mixed_results.append(dict(spec, status="failed", reason=str(exc)))
    except Exception as exc:  # pylint: disable=broad-except
      print(f"    ERROR: mixed arm at {spec['target_hit_rate']} raised"
            f" {type(exc).__name__}: {exc}", file=sys.stderr)
      mixed_results.append(
          dict(spec, status="failed", reason=f"{type(exc).__name__}: {exc}")
      )

  verdict = analyse_slopes(
      summaries, args.total_layers, args.full_attention_layers
  )

  try:
    metadata = json.loads(args.metadata)
  except json.JSONDecodeError:
    metadata = {"raw_metadata_unparseable": True}

  payload = {
      "benchmark": "prefix_reuse",
      "engine": args.engine,
      "model": args.model,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "metadata": metadata,
      "shape": {
          "prefix_tokens": args.prefix_tokens_list,
          "suffix_tokens": args.suffix_tokens,
          "osl": args.osl,
          "repeats": args.repeats,
          "arms": args.arm_list,
          "kv_pool_tokens": pool_tokens or None,
          "eviction_overshoot": args.eviction_overshoot,
      },
      "server_info": {
          key: server_info.get(key)
          for key in (
              "schedule_policy",
              "enable_hierarchical_cache",
              "hicache_storage_backend",
              "hicache_ratio",
              "disable_radix_cache",
              "max_total_num_tokens",
              "kv_cache_dtype",
              "speculative_algorithm",
          )
          if key in server_info
      },
      "layer_split": {
          "total_layers": args.total_layers,
          "full_attention_layers": args.full_attention_layers,
          "linear_attention_layers": args.total_layers - args.full_attention_layers,
          "note": "Reuse limited to the full-attention layers alone would show a"
                  " reuse efficiency near full_attention_layers/total_layers.",
      },
      "corpus_profile": factory.describe(),
      "slope_points": summaries,
      "mixed_arm": mixed_results,
      "verdict": verdict,
  }

  # The sweep harness has always routed its payload through the sanitizer; this
  # file never did, and the metadata it captures from /get_server_info includes
  # the served model path -- which on GKE is the Artifact Registry image, which
  # carries the project ID. The 2026-08-03 window wrote two artifacts containing
  # a real project ID in plaintext, and they were only caught by a pre-publish
  # grep. A benchmark that cannot be committed is a benchmark that gets
  # committed by hand, badly, so the redaction belongs here rather than in the
  # operator's memory.
  try:
    from telemetry_sanitizer import sanitize_telemetry
  except ImportError:
    from benchmarks.telemetry_sanitizer import sanitize_telemetry
  payload = sanitize_telemetry(payload, args.output)

  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

  print("")
  print(f"[SUCCESS] Prefix-reuse results written to {args.output}")
  for arm, slope in verdict["slopes"].items():
    print(f"  {arm:<8} slope={slope['ms_per_prefix_token']} ms/prefix-token"
          f" over {slope['points']} point(s)")
  for arm, efficiency in verdict["reuse_efficiency"].items():
    print(f"  {arm:<8} reuse_efficiency={efficiency}")
  for line in verdict["interpretation"]:
    print(f"  - {line}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
