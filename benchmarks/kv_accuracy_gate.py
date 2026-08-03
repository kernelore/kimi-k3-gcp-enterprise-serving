#!/usr/bin/env python3
"""fp8 KV cache accuracy gate: does halving the KV bytes cost correctness?

--kv-cache-dtype fp8_e4m3 roughly halves KV bytes, which roughly doubles the
sequences that fit in the pool. That is the single largest concurrency lever
available, and it is also the one change in the plan that can degrade output
silently: throughput goes up, every dashboard looks better, and the model is
quietly worse at long context. So the rule is accuracy first, throughput second,
and this file is what makes that rule enforceable rather than aspirational.

The obvious test -- run the same prompts under bf16 and under fp8 and diff the
text -- does not work on its own, because a 2.8T MoE sharded over 16 GPUs is not
bit-deterministic. Expert routing and reduction order vary run to run, so two
bf16 runs already disagree. Diffing against a baseline without knowing that
floor measures nondeterminism and calls it quantisation error.

Two independent signals are collected instead.

**Retrieval accuracy** is the absolute one, and it needs no baseline at all. Each
probe plants a unique code at a known depth in a long context and asks for it
back. Recovering it depends on the KV of those specific tokens being faithful,
which is exactly what fp8 puts at risk, and the score is right or wrong rather
than same-or-different. Probes are spread across depths because KV quantisation
error accumulates with distance, so a gate that only probes the end of the
context would miss the failure it exists to catch. Retrieving the *wrong* code
is tracked separately from retrieving nothing: a confident wrong answer is the
signature of corrupted KV, an empty one is usually just a refusal.

**Output agreement** is the sensitive one, and it needs a floor. Run bf16 twice
to measure how much two identical configurations already disagree, then compare
fp8 against bf16 on the same scale. If the floor itself is poor, the gate says
so and returns inconclusive rather than passing fp8 on an instrument that could
not have detected a problem.

Capture and compare are separate subcommands because switching KV dtype is a
StatefulSet env change and a ~17 minute warm restart. Each run writes a capture
file; comparison happens offline afterwards and costs nothing.

    kv_accuracy_gate.py capture  --label bf16-a --output caps/bf16-a.json
    kv_accuracy_gate.py capture  --label bf16-b --output caps/bf16-b.json
    # ... flip SGLANG_KV_CACHE_DTYPE to fp8_e4m3, restart, wait Ready ...
    kv_accuracy_gate.py capture  --label fp8   --output caps/fp8.json
    kv_accuracy_gate.py compare  --baseline caps/bf16-a.json \
                                 --repeat   caps/bf16-b.json \
                                 --candidate caps/fp8.json

Offline: `--dry-run` plans the probe set, `--self-check` proves each needle is
present exactly once at the depth it claims.
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
  from run_realistic_sweep_kimi_k3 import DEFAULT_CHARS_PER_TOKEN
  from run_realistic_sweep_kimi_k3 import SentenceSource
except ImportError:  # invoked as benchmarks.kv_accuracy_gate
  from benchmarks.run_realistic_sweep_kimi_k3 import DEFAULT_CHARS_PER_TOKEN
  from benchmarks.run_realistic_sweep_kimi_k3 import SentenceSource

try:
  from telemetry_sanitizer import sanitize_telemetry as _sanitize_telemetry
except ImportError:  # invoked as benchmarks.kv_accuracy_gate
  from benchmarks.telemetry_sanitizer import sanitize_telemetry as _sanitize_telemetry


def _sanitize(payload, out_path):
  """Strip project-identifying strings before anything reaches disk.

  Both artifacts this file writes echo the served model path back from
  /get_server_info, and on GKE that is the Artifact Registry image -- which
  contains the project ID. The sweep harness has sanitised its payload since it
  was written; this one did not, and the 2026-08-03 window produced two result
  files with a real project ID in them. Nothing leaked, because a pre-publish
  grep caught it, but "a grep catches it" is not a property of the repo, it is a
  property of whoever remembered to run the grep.
  """
  return _sanitize_telemetry(payload, out_path)


DEFAULT_CORPUS_SEED = 20260803
DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_DEPTHS = "0.05,0.25,0.50,0.75,0.95"
DEFAULT_PROBES_PER_DEPTH = 10
DEFAULT_MAX_TOKENS = 256

# Pre-registered thresholds. Fixed here, before any number exists, so the gate
# cannot be argued down later against a result someone wants to ship.
#
# The retrieval thresholds are absolute because retrieval is an absolute
# measurement; the agreement threshold is relative to the measured
# self-consistency floor because agreement is not.
THRESHOLD_MAX_RETRIEVAL_DROP = 0.05      # percentage points of correct recall
THRESHOLD_MAX_WRONG_CODE_RATE = 0.02     # confidently wrong retrievals
THRESHOLD_MAX_EXACT_MATCH_DROP = 0.10    # vs the bf16-vs-bf16 floor
THRESHOLD_MIN_DIVERGENCE_RATIO = 0.50    # vs the floor's first-divergence depth
THRESHOLD_MIN_FLOOR_EXACT_MATCH = 0.20   # below this the instrument is blind
THRESHOLD_MAX_DEGENERATE_RATE = 0.02     # looping output

REQUEST_TIMEOUT_S = 600
CODE_RE = re.compile(r"\b([A-Z]{2}-[0-9A-F]{8})\b")


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def _token(probe_id, salt):
  digest = hashlib.blake2b(
      f"{probe_id}|{salt}".encode("utf-8"), digest_size=4
  ).hexdigest().upper()
  return digest


def make_unit_name(probe_id):
  return f"Array-{_token(probe_id, 'unit')[:4]}"


def make_code(probe_id):
  # Two letters plus eight hex, so CODE_RE can find any code the model emits and
  # a wrong answer is distinguishable from no answer.
  letters = _token(probe_id, "alpha")
  alpha = chr(65 + int(letters[0], 16) % 26) + chr(65 + int(letters[1], 16) % 26)
  return f"{alpha}-{_token(probe_id, 'code')}"


def build_probes(seed, context_tokens, depths, probes_per_depth, chars_per_token):
  """Build the probe set: a long context with one planted code per probe.

  Deterministic in (seed, context_tokens, depths, probes_per_depth) so a capture
  taken before a restart is comparable with one taken after it.
  """
  source = SentenceSource(seed)
  budget = int(round(context_tokens * chars_per_token))
  probes = []
  for depth in depths:
    for index in range(probes_per_depth):
      probe_id = f"d{depth:.2f}-n{index}"
      unit = make_unit_name(probe_id)
      code = make_code(probe_id)
      needle = (
          f"Operational note: the calibration code for {unit} is {code}."
      )

      sentences = []
      length = 0
      while length < budget:
        sentence = source.next_sentence()
        sentences.append(sentence)
        length += len(sentence) + 1

      position = min(
          len(sentences) - 1, max(0, int(round(len(sentences) * depth)))
      )
      sentences.insert(position, needle)
      context = " ".join(sentences)

      prompt = (
          "Read the following operations log, then answer the question at the"
          " end using only what the log says.\n\n"
          f"--- BEGIN LOG ---\n{context}\n--- END LOG ---\n\n"
          f"Question: what is the calibration code for {unit}?\n"
          "Answer with the code and nothing else.\nAnswer:"
      )
      probes.append({
          "probe_id": probe_id,
          "depth": depth,
          "unit": unit,
          "expected_code": code,
          "needle_sentence_index": position,
          "sentence_count": len(sentences),
          "prompt": prompt,
      })
  return probes


def describe_probes(probes):
  return {
      "count": len(probes),
      "depths": sorted({p["depth"] for p in probes}),
      "prompt_chars_mean": round(
          statistics.fmean(len(p["prompt"]) for p in probes), 1
      ) if probes else 0.0,
      "distinct_codes": len({p["expected_code"] for p in probes}),
  }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def longest_repeated_tail(text, min_len=24):
  """Length of the longest immediately-repeated substring at the tail.

  Degenerate looping is the classic way a damaged KV cache shows up, and it is
  invisible to an exact-match score because a loop is perfectly reproducible.
  """
  stripped = text.strip()
  if len(stripped) < min_len * 2:
    return 0
  for span in range(len(stripped) // 2, min_len - 1, -1):
    if stripped[-span:] == stripped[-2 * span:-span]:
      return span
  return 0


def score_output(probe, text):
  """Right answer, wrong answer, or no answer -- they are different failures."""
  found = CODE_RE.findall(text.upper())
  expected = probe["expected_code"]
  correct = expected in found
  wrong = [code for code in found if code != expected]
  return {
      "probe_id": probe["probe_id"],
      "depth": probe["depth"],
      "correct": correct,
      "wrong_code_returned": bool(wrong) and not correct,
      "codes_returned": found[:4],
      "output_chars": len(text),
      "degenerate_tail_chars": longest_repeated_tail(text),
  }


def first_divergence_fraction(a, b):
  """Where two outputs stop agreeing, as a fraction of the shorter one."""
  shorter = min(len(a), len(b))
  if shorter == 0:
    return 1.0 if a == b else 0.0
  index = 0
  while index < shorter and a[index] == b[index]:
    index += 1
  return index / shorter


def compare_captures(left, right):
  """Agreement between two capture files, keyed on probe id."""
  left_by_id = {r["probe_id"]: r for r in left["results"]}
  right_by_id = {r["probe_id"]: r for r in right["results"]}
  shared = sorted(set(left_by_id) & set(right_by_id))

  exact = 0
  divergences = []
  for probe_id in shared:
    a = left_by_id[probe_id].get("output_text") or ""
    b = right_by_id[probe_id].get("output_text") or ""
    if a == b:
      exact += 1
      divergences.append(1.0)
    else:
      divergences.append(first_divergence_fraction(a, b))

  return {
      "probes_compared": len(shared),
      "exact_match_rate": round(exact / len(shared), 4) if shared else None,
      "median_first_divergence_fraction": (
          round(statistics.median(divergences), 4) if divergences else None
      ),
      "mean_first_divergence_fraction": (
          round(statistics.fmean(divergences), 4) if divergences else None
      ),
  }


def summarise_capture(capture):
  results = [r for r in capture["results"] if r.get("success")]
  total = len(results)
  if not total:
    return {
        "probes_succeeded": 0,
        "retrieval_accuracy": None,
        "wrong_code_rate": None,
        "degenerate_rate": None,
        "retrieval_accuracy_by_depth": {},
    }
  by_depth = {}
  for result in results:
    by_depth.setdefault(result["depth"], []).append(result)
  return {
      "probes_succeeded": total,
      "retrieval_accuracy": round(
          sum(1 for r in results if r["correct"]) / total, 4
      ),
      "wrong_code_rate": round(
          sum(1 for r in results if r["wrong_code_returned"]) / total, 4
      ),
      "degenerate_rate": round(
          sum(1 for r in results if r["degenerate_tail_chars"] > 0) / total, 4
      ),
      "retrieval_accuracy_by_depth": {
          f"{depth:.2f}": round(
              sum(1 for r in group if r["correct"]) / len(group), 4
          )
          for depth, group in sorted(by_depth.items())
      },
  }


def apply_gate(baseline, repeat, candidate, thresholds):
  """The pre-registered decision. Returns a verdict of pass, fail, or inconclusive."""
  base_summary = summarise_capture(baseline)
  repeat_summary = summarise_capture(repeat) if repeat else None
  cand_summary = summarise_capture(candidate)

  floor = compare_captures(baseline, repeat) if repeat else None
  measured = compare_captures(baseline, candidate)

  reasons = []
  blocked = []

  # --- absolute checks: these need no floor and are never inconclusive -------
  base_acc = base_summary["retrieval_accuracy"]
  cand_acc = cand_summary["retrieval_accuracy"]
  if base_acc is None or cand_acc is None:
    blocked.append("a capture returned no successful probes")
  else:
    # Rounded to the precision the rates are themselves reported at. Without
    # this, 1.0 - 0.95 is 0.050000000000000044 and a candidate sitting exactly
    # on a pre-registered limit is rejected by float representation rather than
    # by the rule -- which would make the limit unstatable.
    drop = round(base_acc - cand_acc, 4)
    if drop > thresholds["max_retrieval_drop"]:
      reasons.append(
          f"retrieval accuracy fell {drop:.3f} (from {base_acc:.3f} to"
          f" {cand_acc:.3f}), past the {thresholds['max_retrieval_drop']:.3f}"
          " pre-registered limit"
      )

  if cand_summary["wrong_code_rate"] is not None:
    if cand_summary["wrong_code_rate"] > thresholds["max_wrong_code_rate"]:
      reasons.append(
          f"{cand_summary['wrong_code_rate']:.3f} of probes returned a"
          " confidently wrong code, past the"
          f" {thresholds['max_wrong_code_rate']:.3f} limit. A wrong code is"
          " corrupted KV, not a refusal."
      )

  if cand_summary["degenerate_rate"] is not None:
    if cand_summary["degenerate_rate"] > thresholds["max_degenerate_rate"]:
      reasons.append(
          f"{cand_summary['degenerate_rate']:.3f} of outputs ended in a"
          " repeating loop, past the"
          f" {thresholds['max_degenerate_rate']:.3f} limit"
      )

  # --- relative checks: only meaningful if the floor could detect anything ---
  agreement_verdict = "not evaluated"
  if floor is None:
    agreement_verdict = (
        "skipped: no --repeat capture, so there is no self-consistency floor and"
        " an exact-match comparison would be measuring nondeterminism"
    )
  elif floor["exact_match_rate"] is None:
    agreement_verdict = "skipped: the floor comparison shared no probes"
  elif measured["exact_match_rate"] is None:
    agreement_verdict = (
        "skipped: the candidate shared no probes with the baseline"
    )
  elif floor["exact_match_rate"] < thresholds["min_floor_exact_match"]:
    agreement_verdict = (
        f"inconclusive: two identical bf16 runs agreed exactly on only"
        f" {floor['exact_match_rate']:.3f} of probes, below the"
        f" {thresholds['min_floor_exact_match']:.3f} floor this test needs."
        " Output agreement cannot resolve a quantisation effect at that noise"
        " level; shorten --max-tokens or reduce batch nondeterminism first."
    )
  else:
    agreement_verdict = "evaluated"
    drop = round(floor["exact_match_rate"] - measured["exact_match_rate"], 4)
    if drop > thresholds["max_exact_match_drop"]:
      reasons.append(
          f"exact-match agreement fell {drop:.3f} below the self-consistency"
          f" floor ({measured['exact_match_rate']:.3f} against"
          f" {floor['exact_match_rate']:.3f}), past the"
          f" {thresholds['max_exact_match_drop']:.3f} limit"
      )
    floor_div = floor["median_first_divergence_fraction"]
    cand_div = measured["median_first_divergence_fraction"]
    if floor_div and cand_div is not None:
      ratio = round(cand_div / floor_div, 4)
      if ratio < thresholds["min_divergence_ratio"]:
        reasons.append(
            f"outputs diverge from the baseline {ratio:.2f}x as early as two"
            " identical runs do, past the"
            f" {thresholds['min_divergence_ratio']:.2f} limit"
        )

  if blocked:
    decision = "inconclusive"
  elif reasons:
    decision = "reject"
  elif agreement_verdict.startswith("inconclusive"):
    # The absolute checks passed, so this is not a rejection -- but it is not a
    # clean pass either, and calling it one would overstate what was measured.
    decision = "pass_with_caveat"
  else:
    decision = "pass"

  return {
      "decision": decision,
      "reasons": reasons or blocked,
      "agreement_check": agreement_verdict,
      "thresholds": thresholds,
      "baseline": base_summary,
      "baseline_repeat": repeat_summary,
      "candidate": cand_summary,
      "self_consistency_floor": floor,
      "candidate_vs_baseline": measured,
  }


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------


def issue_probe(endpoint, model, prompt, max_tokens, api_key, probe_id):
  """Greedy, non-streaming. Temperature 0 is the whole point of this file."""
  payload = {
      "model": model,
      "max_tokens": max_tokens,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": False,
  }
  if "/chat/completions" in endpoint:
    payload["messages"] = [{"role": "user", "content": prompt}]
  else:
    payload["prompt"] = prompt

  headers = {
      "Content-Type": "application/json",
      "User-Agent": f"KIMI3-KV-Accuracy-Gate/1.0 ({probe_id})",
  }
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

  req = urllib.request.Request(
      endpoint,
      data=json.dumps(payload).encode("utf-8"),
      headers=headers,
      method="POST",
  )
  started = time.time()
  try:
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
      body = json.loads(resp.read().decode("utf-8", errors="replace"))
  except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
    return {"success": False, "error": str(exc)}

  choice = (body.get("choices") or [{}])[0]
  text = choice.get("text")
  if text is None:
    text = (choice.get("message") or {}).get("content") or ""
  return {
      "success": True,
      "output_text": text,
      "finish_reason": choice.get("finish_reason"),
      "usage": body.get("usage") or {},
      "latency_ms": round((time.time() - started) * 1000.0, 2),
  }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def run_capture(args, probes):
  results = []
  for index, probe in enumerate(probes, start=1):
    print(f"  [{index}/{len(probes)}] {probe['probe_id']} ...", flush=True)
    try:
      response = issue_probe(
          args.endpoint,
          args.model,
          probe["prompt"],
          args.max_tokens,
          args.api_key,
          probe["probe_id"],
      )
    except (KeyError, ValueError, OSError) as exc:
      response = {"success": False, "error": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except
      # A capture costs a full pass over the probe set on a running cluster.
      # One bad response must not discard the probes already paid for.
      response = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    if not response.get("success"):
      print(f"      FAILED: {response.get('error')}", file=sys.stderr)
      results.append({
          "probe_id": probe["probe_id"],
          "depth": probe["depth"],
          "success": False,
          "error": response.get("error"),
      })
      continue

    text = response["output_text"]
    entry = score_output(probe, text)
    entry.update({
        "success": True,
        "output_text": text,
        "finish_reason": response.get("finish_reason"),
        "latency_ms": response.get("latency_ms"),
    })
    results.append(entry)

  capture = {
      "capture_label": args.label,
      "kv_cache_dtype_declared": args.kv_cache_dtype,
      "engine": args.engine,
      "model": args.model,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "probe_set": {
          "corpus_seed": args.corpus_seed,
          "context_tokens": args.context_tokens,
          "depths": args.depth_list,
          "probes_per_depth": args.probes_per_depth,
          "max_tokens": args.max_tokens,
          "chars_per_token": args.chars_per_token,
      },
      "results": results,
  }
  summary = summarise_capture(capture)
  capture["summary"] = summary
  capture = _sanitize(capture, args.output)

  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(capture, handle, indent=2)

  print("")
  print(f"[SUCCESS] Capture '{args.label}' written to {args.output}")
  print(f"  Probes succeeded:   {summary['probes_succeeded']}/{len(probes)}")
  print(f"  Retrieval accuracy: {summary['retrieval_accuracy']}")
  print(f"  Wrong-code rate:    {summary['wrong_code_rate']}")
  print(f"  Degenerate rate:    {summary['degenerate_rate']}")
  for depth, accuracy in summary["retrieval_accuracy_by_depth"].items():
    print(f"    depth {depth}: {accuracy}")
  if args.kv_cache_dtype == "unknown":
    print("  NOTE: --kv-cache-dtype was not supplied, so this capture does not"
          " record which configuration produced it.", file=sys.stderr)
  return 0


def load_capture(path):
  with open(path, encoding="utf-8") as handle:
    return json.load(handle)


def probe_sets_match(*captures):
  keys = [
      (
          c["probe_set"]["corpus_seed"],
          c["probe_set"]["context_tokens"],
          tuple(c["probe_set"]["depths"]),
          c["probe_set"]["probes_per_depth"],
          c["probe_set"]["max_tokens"],
      )
      for c in captures
  ]
  return len(set(keys)) == 1


def run_compare(args):
  baseline = load_capture(args.baseline)
  repeat = load_capture(args.repeat) if args.repeat else None
  candidate = load_capture(args.candidate)

  captures = [c for c in (baseline, repeat, candidate) if c]
  if not probe_sets_match(*captures):
    print(
        "ERROR: the captures were taken with different probe sets, so any"
        " comparison between them would be meaningless.",
        file=sys.stderr,
    )
    for capture in captures:
      print(f"  {capture['capture_label']}: {capture['probe_set']}", file=sys.stderr)
    return 2

  thresholds = {
      "max_retrieval_drop": args.max_retrieval_drop,
      "max_wrong_code_rate": args.max_wrong_code_rate,
      "max_exact_match_drop": args.max_exact_match_drop,
      "min_divergence_ratio": args.min_divergence_ratio,
      "min_floor_exact_match": args.min_floor_exact_match,
      "max_degenerate_rate": args.max_degenerate_rate,
  }
  verdict = apply_gate(baseline, repeat, candidate, thresholds)
  verdict["captures"] = {
      "baseline": {
          "label": baseline["capture_label"],
          "kv_cache_dtype": baseline.get("kv_cache_dtype_declared"),
      },
      "repeat": {
          "label": repeat["capture_label"],
          "kv_cache_dtype": repeat.get("kv_cache_dtype_declared"),
      } if repeat else None,
      "candidate": {
          "label": candidate["capture_label"],
          "kv_cache_dtype": candidate.get("kv_cache_dtype_declared"),
      },
  }
  verdict["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
  verdict = _sanitize(verdict, args.output)

  if args.output:
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
      json.dump(verdict, handle, indent=2)

  print("=== fp8 KV ACCURACY GATE ===")
  print(f"  Baseline  : {baseline['capture_label']}"
        f" ({baseline.get('kv_cache_dtype_declared')})")
  if repeat:
    print(f"  Repeat    : {repeat['capture_label']}"
          f" ({repeat.get('kv_cache_dtype_declared')})")
  print(f"  Candidate : {candidate['capture_label']}"
        f" ({candidate.get('kv_cache_dtype_declared')})")
  print("")
  print(f"  Retrieval accuracy: {verdict['baseline']['retrieval_accuracy']}"
        f" -> {verdict['candidate']['retrieval_accuracy']}")
  print(f"  Wrong-code rate:    {verdict['baseline']['wrong_code_rate']}"
        f" -> {verdict['candidate']['wrong_code_rate']}")
  print(f"  Degenerate rate:    {verdict['baseline']['degenerate_rate']}"
        f" -> {verdict['candidate']['degenerate_rate']}")
  if verdict["self_consistency_floor"]:
    print(f"  Self-consistency floor (bf16 vs bf16):"
          f" {verdict['self_consistency_floor']['exact_match_rate']} exact,"
          f" divergence at"
          f" {verdict['self_consistency_floor']['median_first_divergence_fraction']}")
  print(f"  Candidate vs baseline:"
        f" {verdict['candidate_vs_baseline']['exact_match_rate']} exact,"
        f" divergence at"
        f" {verdict['candidate_vs_baseline']['median_first_divergence_fraction']}")
  print(f"  Agreement check: {verdict['agreement_check']}")
  print("")
  print(f"  DECISION: {verdict['decision'].upper()}")
  for reason in verdict["reasons"]:
    print(f"    - {reason}")
  if args.output:
    print(f"\n  Written to {args.output}")

  return 0 if verdict["decision"] in ("pass", "pass_with_caveat") else 1


def run_self_check(args, probes):
  print("=== KV ACCURACY GATE SELF-CHECK ===")
  print(f"  Probes:              {len(probes)}")
  print(f"  Depths:              {args.depth_list}")
  print(f"  Context target:      {args.context_tokens} tokens")

  failures = []
  codes = [p["expected_code"] for p in probes]
  if len(set(codes)) != len(codes):
    failures.append("two probes were assigned the same code")

  for probe in probes:
    prompt = probe["prompt"]
    code = probe["expected_code"]
    # Exactly once, in the planted note. The question names the unit, not the
    # code -- a second occurrence would put the answer in the question and the
    # probe would measure nothing.
    if prompt.count(code) != 1:
      failures.append(
          f"{probe['probe_id']}: code appears {prompt.count(code)} times in the"
          " prompt, expected exactly 1"
      )
      break
    found = CODE_RE.findall(prompt.upper())
    if found.count(code) != 1:
      failures.append(
          f"{probe['probe_id']}: {found.count(code)} regex-visible occurrences"
          " of the code"
      )
      break
    if probe["unit"] not in prompt:
      failures.append(f"{probe['probe_id']}: unit name missing from the prompt")
      break
    actual_depth = probe["needle_sentence_index"] / probe["sentence_count"]
    # A needle cannot be placed more precisely than one sentence, so the
    # tolerance has to scale with how few sentences there are.
    tolerance = max(0.02, 1.0 / probe["sentence_count"])
    if abs(actual_depth - probe["depth"]) > tolerance:
      failures.append(
          f"{probe['probe_id']}: needle sits at depth {actual_depth:.3f},"
          f" not the {probe['depth']:.2f} it claims (tolerance"
          f" {tolerance:.3f} at {probe['sentence_count']} sentences)"
      )
      break

  distinct_prompts = len({p["prompt"] for p in probes})
  print(f"  Distinct prompts:    {distinct_prompts}/{len(probes)}")
  print(f"  Distinct codes:      {len(set(codes))}/{len(codes)}")
  print(f"  Mean prompt length:  "
        f"{describe_probes(probes)['prompt_chars_mean']:.0f} chars")
  if distinct_prompts != len(probes):
    failures.append("two probes share a prompt")

  # A scorer that cannot tell a right answer from a wrong one would pass
  # everything, so it is exercised here on known input rather than trusted.
  sample = probes[0]
  right = score_output(sample, f"The code is {sample['expected_code']}.")
  wrong = score_output(sample, "The code is ZZ-DEADBEEF.")
  empty = score_output(sample, "I could not find it in the log.")
  looped = score_output(sample, "spinning up the array. " * 40)
  print(f"  Scorer, right answer: correct={right['correct']}")
  print(f"  Scorer, wrong answer: correct={wrong['correct']}"
        f" wrong_code={wrong['wrong_code_returned']}")
  print(f"  Scorer, no answer:    correct={empty['correct']}"
        f" wrong_code={empty['wrong_code_returned']}")
  print(f"  Scorer, looping:      degenerate_tail="
        f"{looped['degenerate_tail_chars']} chars")
  if not right["correct"]:
    failures.append("the scorer missed a correct answer")
  if wrong["correct"] or not wrong["wrong_code_returned"]:
    failures.append("the scorer does not flag a confidently wrong answer")
  if empty["correct"] or empty["wrong_code_returned"]:
    failures.append("the scorer misreads a refusal as a wrong answer")
  if looped["degenerate_tail_chars"] <= 0:
    failures.append("the scorer does not detect looping output")

  if failures:
    print("\nVERDICT: self-check FAILED")
    for failure in failures:
      print(f"  - {failure}")
    return 1
  print("\nVERDICT: probe set and scorer are sound.")
  return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_depths(raw):
  try:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
  except ValueError:
    raise argparse.ArgumentTypeError(f"expected comma-separated floats: {raw!r}")
  if not values:
    raise argparse.ArgumentTypeError("at least one depth is required")
  for value in values:
    if not 0.0 <= value <= 1.0:
      raise argparse.ArgumentTypeError(f"depth must be in [0, 1]: {value}")
  return values


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="fp8 KV cache accuracy gate for Kimi K3"
  )
  sub = parser.add_subparsers(dest="command", required=True)

  def add_probe_args(target):
    target.add_argument("--corpus-seed", type=int, default=DEFAULT_CORPUS_SEED)
    target.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help="Log length per probe; KV error accumulates with it")
    target.add_argument("--depths", type=parse_depths, default=DEFAULT_DEPTHS,
                        help="Comma-separated needle depths as fractions")
    target.add_argument("--probes-per-depth", type=int,
                        default=DEFAULT_PROBES_PER_DEPTH)
    target.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    target.add_argument("--chars-per-token", type=float,
                        default=DEFAULT_CHARS_PER_TOKEN)
    target.add_argument("--dry-run", action="store_true",
                        help="Plan the probe set and exit")
    target.add_argument("--self-check", action="store_true",
                        help="Prove offline that each needle and the scorer work")

  capture = sub.add_parser("capture", help="Run the probe set against a server")
  capture.add_argument("--endpoint",
                       default="http://localhost:8000/v1/completions")
  capture.add_argument("--model", default="moonshotai/Kimi-K3")
  capture.add_argument("--api-key", default="")
  capture.add_argument("--engine", default="sglang")
  capture.add_argument("--label", required=True,
                       help="Name for this capture, e.g. bf16-a or fp8")
  capture.add_argument("--kv-cache-dtype", default="unknown",
                       help="The dtype this server is actually running, recorded"
                            " into the capture so a comparison cannot silently"
                            " compare a config against itself")
  capture.add_argument("--output", default="benchmarks/kv_accuracy_capture.json")
  add_probe_args(capture)

  compare = sub.add_parser("compare", help="Apply the gate to captured runs")
  compare.add_argument("--baseline", required=True,
                       help="bf16 capture, run 1")
  compare.add_argument("--repeat", default="",
                       help="bf16 capture, run 2. Without it the agreement check"
                            " is skipped, because there is no noise floor.")
  compare.add_argument("--candidate", required=True, help="fp8 capture")
  compare.add_argument("--output", default="")
  compare.add_argument("--max-retrieval-drop", type=float,
                       default=THRESHOLD_MAX_RETRIEVAL_DROP)
  compare.add_argument("--max-wrong-code-rate", type=float,
                       default=THRESHOLD_MAX_WRONG_CODE_RATE)
  compare.add_argument("--max-exact-match-drop", type=float,
                       default=THRESHOLD_MAX_EXACT_MATCH_DROP)
  compare.add_argument("--min-divergence-ratio", type=float,
                       default=THRESHOLD_MIN_DIVERGENCE_RATIO)
  compare.add_argument("--min-floor-exact-match", type=float,
                       default=THRESHOLD_MIN_FLOOR_EXACT_MATCH)
  compare.add_argument("--max-degenerate-rate", type=float,
                       default=THRESHOLD_MAX_DEGENERATE_RATE)

  args = parser.parse_args(argv)
  if args.command == "capture":
    args.depth_list = args.depths
    if args.probes_per_depth < 1:
      parser.error("--probes-per-depth must be at least 1")
    if args.context_tokens < 64:
      parser.error("--context-tokens must be at least 64 to place a needle")
    if args.chars_per_token <= 0:
      parser.error("--chars-per-token must be positive")
    if args.max_tokens < 1:
      parser.error("--max-tokens must be at least 1")
  return args


def main(argv=None):
  args = parse_args(argv)

  if args.command == "compare":
    return run_compare(args)

  probes = build_probes(
      args.corpus_seed,
      args.context_tokens,
      args.depth_list,
      args.probes_per_depth,
      args.chars_per_token,
  )

  if args.self_check:
    return run_self_check(args, probes)

  if args.dry_run:
    profile = describe_probes(probes)
    print("=== KIMI K3 fp8 KV ACCURACY GATE ===")
    print(f"  Probes:            {profile['count']}")
    print(f"  Depths:            {profile['depths']}")
    print(f"  Distinct codes:    {profile['distinct_codes']}")
    print(f"  Mean prompt chars: {profile['prompt_chars_mean']:.0f}"
          f" (~{profile['prompt_chars_mean'] / args.chars_per_token:.0f} tokens)")
    print(f"  Output tokens:     {args.max_tokens} per probe, temperature 0")
    print("")
    print("  This gate needs three captures: bf16 twice for the"
          " self-consistency floor, then fp8. Comparison is offline.")
    print("")
    print(f"[SUCCESS] Kimi K3 fp8 KV accuracy gate verified"
          f" (PROBES: {profile['count']}).")
    return 0

  return run_capture(args, probes)


if __name__ == "__main__":
  sys.exit(main())
