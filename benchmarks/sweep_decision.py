#!/usr/bin/env python3
"""Apply the pre-registered keep/back-out rule to one sweep variant.

The tuning plan fixes these rules before any measurement exists, for the stated
reason that they should not be invented at 2am with a spot clock running. This
file is where they stop being prose. It is separate from the driver that calls
it so the arithmetic can be tested offline against known input -- a rule that
only ever runs on a live cluster at $92/hour is a rule nobody has checked.

Two objectives, because the plan judges two kinds of change differently.

**throughput** -- kernels, memory, speculation. Keep the change if it gains at
least 2% aggregate output tok/s at c=16, with no worse than a 2% regression at
c=8 or c=32 and no worse than a 10% worsening of TTFT P99. Two independent runs
of the existing sweep agreed to within 0.2%, so 2% is ten times the measured
noise floor rather than a round number.

**scheduling** -- schedule policy, chunked prefill, admission caps. These are
judged on TTFT P99 instead, subject to aggregate not dropping more than 5%.
Aggregate throughput is *expected* to fall slightly here; that is the trade
being bought, and scoring these knobs on throughput would back out every one of
them for doing exactly what they were turned on to do.

Cells are matched on (isl_target, osl, concurrency) and only compared when both
sides ran. A variant is undecidable rather than accepted when the cells it
needed are missing: a spot preemption halfway through a sweep leaves a results
file that looks fine and covers half the grid, and treating a thin file as a
pass is how a regression gets stacked on top of.

Both results shapes are read. The realistic harness nests its cells under
``arm_results[arm]`` because it runs a repeated-passage arm alongside the
non-repetitive one; the saturation harness emits a flat ``sweep_results``. The
default arm is the non-repetitive one, since the plan's baseline is defined on
that harness and the repeated-passage arm exists to be the thing it is measured
*against*. A baseline and a candidate read from different shapes are refused
rather than compared -- the prompt distribution differs between them, so the
delta would be measuring the corpus rather than the variant.

Exit status is the interface the driver uses: 0 accept, 1 back out,
2 undecidable.
"""

import argparse
import json
import sys

OBJECTIVE_THROUGHPUT = "throughput"
OBJECTIVE_SCHEDULING = "scheduling"
OBJECTIVES = (OBJECTIVE_THROUGHPUT, OBJECTIVE_SCHEDULING)

DECISION_ACCEPT = "accept"
DECISION_BACK_OUT = "back_out"
DECISION_UNDECIDABLE = "undecidable"

EXIT_CODES = {
    DECISION_ACCEPT: 0,
    DECISION_BACK_OUT: 1,
    DECISION_UNDECIDABLE: 2,
}

# Percentage points. Named rather than inlined so the verdict can record which
# numbers it was judged against.
THROUGHPUT_MIN_GAIN_PCT = 2.0
THROUGHPUT_MAX_REGRESSION_PCT = 2.0
THROUGHPUT_MAX_TTFT_WORSENING_PCT = 10.0
SCHEDULING_MAX_AGGREGATE_DROP_PCT = 5.0

# The plan writes every throughput rule against c=16, which is why the realistic
# harness defaults to levels 8,16,32. The saturation harness runs 1,8,32,128 and
# has no c=16 cell at all, so pointing this at a saturation file without also
# moving the anchor yields undecidable on every variant. Overridable so that
# mismatch is a stated choice rather than a silent one.
PRIMARY_CONCURRENCY = 16
GUARD_CONCURRENCIES = (8, 32)
MIN_MATCHED_CELLS = 2

ARM_NON_REPETITIVE = "non_repetitive"


def load_cells(path, arm=ARM_NON_REPETITIVE):
  """Index the ok cells of a results file by grid position.

  Accepts either harness shape. Returns the source it read so a caller can
  refuse to compare a realistic arm against a saturation run.
  """
  with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
  arm_results = payload.get("arm_results")
  if isinstance(arm_results, dict) and arm_results:
    if arm not in arm_results:
      raise KeyError(
          f"{path} carries arms {sorted(arm_results)} but not {arm!r}"
      )
    rows = arm_results[arm]
    source = f"arm_results[{arm}]"
  else:
    rows = payload.get("sweep_results")
    source = "sweep_results"
  cells = {}
  for cell in rows or []:
    if cell.get("status") != "ok":
      continue
    key = (cell.get("isl_target"), cell.get("osl"), cell.get("concurrency"))
    cells[key] = cell
  return cells, payload, source


def pct_delta(before, after):
  if not before:
    return None
  return round((after - before) / before * 100.0, 3)


def pooled_throughput(cells, concurrency):
  """Total tokens over total duration for one concurrency band.

  Pooling by duration rather than averaging per-cell rates is what "aggregate
  output tok/s at c=N" means: it is the throughput of that band as a whole. A
  mean of rates would weight a 30-second 1k cell the same as a 20-minute 128k
  cell.
  """
  tokens = 0.0
  duration = 0.0
  count = 0
  for (_isl, _osl, conc), cell in cells.items():
    if conc != concurrency:
      continue
    cell_tokens = cell.get("total_tokens")
    cell_duration = cell.get("total_duration_sec")
    if not cell_tokens or not cell_duration:
      continue
    tokens += float(cell_tokens)
    duration += float(cell_duration)
    count += 1
  if not duration or not count:
    return None, count
  return tokens / duration, count


def band_comparison(base_cells, cand_cells, concurrency):
  shared = {
      key for key in base_cells
      if key in cand_cells and key[2] == concurrency
  }
  base_only = {k: v for k, v in base_cells.items() if k in shared}
  cand_only = {k: v for k, v in cand_cells.items() if k in shared}
  base_tps, _ = pooled_throughput(base_only, concurrency)
  cand_tps, _ = pooled_throughput(cand_only, concurrency)
  return {
      "concurrency": concurrency,
      "matched_cells": len(shared),
      "baseline_tok_s": round(base_tps, 3) if base_tps else None,
      "candidate_tok_s": round(cand_tps, 3) if cand_tps else None,
      "delta_pct": pct_delta(base_tps, cand_tps) if base_tps and cand_tps else None,
  }


def worst_ttft_change(base_cells, cand_cells):
  """The largest P99 worsening across every matched cell.

  Max rather than pooled: a P99 blowout confined to one shape is exactly the
  failure this guard exists to catch, and pooling would average it away.
  """
  worst = None
  worst_key = None
  best = None
  for key, base in base_cells.items():
    cand = cand_cells.get(key)
    if not cand:
      continue
    base_p99 = (base.get("ttft_ms") or {}).get("p99")
    cand_p99 = (cand.get("ttft_ms") or {}).get("p99")
    if not base_p99 or cand_p99 is None:
      continue
    delta = pct_delta(base_p99, cand_p99)
    if delta is None:
      continue
    if worst is None or delta > worst:
      worst, worst_key = delta, key
    if best is None or delta < best:
      best = delta
  return worst, worst_key, best


def decide(baseline_path, candidate_path, objective, variant="",
           arm=ARM_NON_REPETITIVE, primary=PRIMARY_CONCURRENCY,
           guards=GUARD_CONCURRENCIES):
  base_cells, base_payload, base_source = load_cells(baseline_path, arm)
  cand_cells, cand_payload, cand_source = load_cells(candidate_path, arm)

  shared = set(base_cells) & set(cand_cells)
  guards = tuple(c for c in guards if c != primary)
  bands = {
      conc: band_comparison(base_cells, cand_cells, conc)
      for conc in (primary,) + guards
  }
  worst_ttft, worst_ttft_cell, best_ttft = worst_ttft_change(base_cells, cand_cells)

  verdict = {
      "variant": variant,
      "objective": objective,
      "baseline_file": baseline_path,
      "candidate_file": candidate_path,
      "baseline_engine": base_payload.get("engine"),
      "candidate_engine": cand_payload.get("engine"),
      "baseline_source": base_source,
      "candidate_source": cand_source,
      "arm": arm,
      "primary_concurrency": primary,
      "guard_concurrencies": list(guards),
      "matched_cells_total": len(shared),
      "baseline_cells_ok": len(base_cells),
      "candidate_cells_ok": len(cand_cells),
      "bands": bands,
      "worst_ttft_p99_delta_pct": worst_ttft,
      "worst_ttft_p99_cell": (
          f"isl{worst_ttft_cell[0]}_osl{worst_ttft_cell[1]}_c{worst_ttft_cell[2]}"
          if worst_ttft_cell else None
      ),
      "best_ttft_p99_delta_pct": best_ttft,
      "thresholds": {
          "throughput_min_gain_pct": THROUGHPUT_MIN_GAIN_PCT,
          "throughput_max_regression_pct": THROUGHPUT_MAX_REGRESSION_PCT,
          "throughput_max_ttft_worsening_pct": THROUGHPUT_MAX_TTFT_WORSENING_PCT,
          "scheduling_max_aggregate_drop_pct": SCHEDULING_MAX_AGGREGATE_DROP_PCT,
      },
      "reasons": [],
  }

  primary_band = bands[primary]
  if base_source != cand_source:
    verdict["decision"] = DECISION_UNDECIDABLE
    verdict["reasons"].append(
        f"the baseline was read from {base_source} and the candidate from"
        f" {cand_source}. Those are different prompt distributions, so the"
        " delta between them would be measuring the corpus rather than the"
        " variant -- which is the specific error the non-repetitive harness"
        " exists to prevent."
    )
    return verdict
  if len(shared) < MIN_MATCHED_CELLS:
    verdict["decision"] = DECISION_UNDECIDABLE
    verdict["reasons"].append(
        f"only {len(shared)} grid cell(s) ran in both the baseline and the"
        f" candidate, below the {MIN_MATCHED_CELLS} needed to decide anything."
        " A half-finished sweep is what a spot preemption leaves behind."
    )
    return verdict
  if primary_band["delta_pct"] is None:
    verdict["decision"] = DECISION_UNDECIDABLE
    verdict["reasons"].append(
        f"no c={primary} cell ran in both, so the primary comparison the rule"
        " is written against does not exist. The realistic harness runs"
        " 8,16,32 by default; the saturation harness runs 1,8,32,128 and has"
        " no c=16 cell to compare."
    )
    return verdict

  if objective == OBJECTIVE_THROUGHPUT:
    _decide_throughput(verdict, bands, primary_band, worst_ttft, primary, guards)
  else:
    _decide_scheduling(verdict, primary_band, worst_ttft, best_ttft, primary)

  return verdict


def _decide_throughput(verdict, bands, primary_band, worst_ttft, primary, guards):
  failures = []
  if primary_band["delta_pct"] < THROUGHPUT_MIN_GAIN_PCT:
    failures.append(
        f"c={primary} aggregate moved {primary_band['delta_pct']:+.2f}%,"
        f" short of the +{THROUGHPUT_MIN_GAIN_PCT:.1f}% required to keep a"
        " stacked change"
    )
  for conc in guards:
    band = bands[conc]
    if band["delta_pct"] is None:
      verdict["reasons"].append(
          f"NOTE: no c={conc} cell ran in both, so that guard rail was not"
          " checked"
      )
      continue
    if band["delta_pct"] < -THROUGHPUT_MAX_REGRESSION_PCT:
      failures.append(
          f"c={conc} aggregate regressed {band['delta_pct']:+.2f}%, past the"
          f" {THROUGHPUT_MAX_REGRESSION_PCT:.1f}% limit"
      )
  if worst_ttft is not None and worst_ttft > THROUGHPUT_MAX_TTFT_WORSENING_PCT:
    failures.append(
        f"TTFT P99 worsened {worst_ttft:+.2f}% at {verdict['worst_ttft_p99_cell']},"
        f" past the {THROUGHPUT_MAX_TTFT_WORSENING_PCT:.1f}% limit"
    )

  if failures:
    verdict["decision"] = DECISION_BACK_OUT
    verdict["reasons"].extend(failures)
  else:
    verdict["decision"] = DECISION_ACCEPT
    verdict["reasons"].append(
        f"c={primary} aggregate gained"
        f" {primary_band['delta_pct']:+.2f}% with both guard rails intact"
    )


def _decide_scheduling(verdict, primary_band, worst_ttft, best_ttft, primary):
  failures = []
  if primary_band["delta_pct"] < -SCHEDULING_MAX_AGGREGATE_DROP_PCT:
    failures.append(
        f"c={primary} aggregate dropped {primary_band['delta_pct']:+.2f}%,"
        f" past the {SCHEDULING_MAX_AGGREGATE_DROP_PCT:.1f}% a scheduling knob"
        " is allowed to cost"
    )
  # The objective is TTFT P99, so it has to actually improve somewhere. Judging
  # a scheduling knob on throughput would back out every one of them for doing
  # what they were turned on to do -- but a knob that improves nothing is still
  # just a slower configuration.
  if best_ttft is None:
    failures.append("no matched cell reported TTFT P99, so there is nothing to minimise")
  elif best_ttft >= 0:
    failures.append(
        f"TTFT P99 did not improve in any matched cell (best was"
        f" {best_ttft:+.2f}%), and this knob is judged on TTFT"
    )

  if failures:
    verdict["decision"] = DECISION_BACK_OUT
    verdict["reasons"].extend(failures)
  else:
    verdict["decision"] = DECISION_ACCEPT
    verdict["reasons"].append(
        f"TTFT P99 improved by up to {abs(best_ttft):.2f}% for"
        f" {primary_band['delta_pct']:+.2f}% aggregate, inside the"
        f" {SCHEDULING_MAX_AGGREGATE_DROP_PCT:.1f}% allowance"
    )
  if worst_ttft is not None and worst_ttft > 0:
    verdict["reasons"].append(
        f"NOTE: TTFT P99 still worsened {worst_ttft:+.2f}% at"
        f" {verdict['worst_ttft_p99_cell']}"
    )


def render(verdict):
  primary = verdict["primary_concurrency"]
  lines = [
      f"=== SWEEP DECISION: {verdict['variant'] or '(unnamed variant)'} ===",
      f"  Objective:      {verdict['objective']}",
      f"  Read from:      {verdict['baseline_source']}",
      f"  Matched cells:  {verdict['matched_cells_total']}"
      f" (baseline {verdict['baseline_cells_ok']} ok,"
      f" candidate {verdict['candidate_cells_ok']} ok)",
  ]
  for conc in [primary] + list(verdict["guard_concurrencies"]):
    band = verdict["bands"][conc]
    marker = "*" if conc == primary else " "
    if band["delta_pct"] is None:
      lines.append(f"  {marker} c={conc:<3} not comparable ({band['matched_cells']} matched)")
    else:
      lines.append(
          f"  {marker} c={conc:<3} {band['baseline_tok_s']} ->"
          f" {band['candidate_tok_s']} tok/s ({band['delta_pct']:+.2f}%)"
      )
  if verdict["worst_ttft_p99_delta_pct"] is not None:
    lines.append(
        f"    TTFT P99 worst {verdict['worst_ttft_p99_delta_pct']:+.2f}% at"
        f" {verdict['worst_ttft_p99_cell']}, best"
        f" {verdict['best_ttft_p99_delta_pct']:+.2f}%"
    )
  lines.append(f"  DECISION: {verdict['decision'].upper()}")
  for reason in verdict["reasons"]:
    lines.append(f"    - {reason}")
  return "\n".join(lines)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Apply the pre-registered keep/back-out rule to a sweep variant"
  )
  parser.add_argument("--baseline", required=True,
                      help="Results file for the last accepted stack")
  parser.add_argument("--candidate", required=True,
                      help="Results file for this variant")
  parser.add_argument("--objective", choices=OBJECTIVES,
                      default=OBJECTIVE_THROUGHPUT)
  parser.add_argument("--variant", default="", help="Label for the report")
  parser.add_argument(
      "--arm", default=ARM_NON_REPETITIVE,
      help=(
          "Arm to read from a realistic-harness results file. Ignored for"
          " saturation files, which are not split by arm."
      ),
  )
  parser.add_argument(
      "--primary-concurrency", type=int, default=PRIMARY_CONCURRENCY,
      help=(
          "Concurrency the keep/back-out rule is anchored on. The plan writes"
          " every threshold against c=16, which the realistic harness runs by"
          " default and the saturation harness does not run at all."
      ),
  )
  parser.add_argument(
      "--guard-concurrencies", default=",".join(str(c) for c in GUARD_CONCURRENCIES),
      type=lambda s: tuple(int(x) for x in s.split(",") if x.strip()),
      help="Comma-separated bands checked for regressions either side.",
  )
  parser.add_argument("--output", default="", help="Write the verdict as JSON")
  parser.add_argument("--quiet", action="store_true")
  args = parser.parse_args(argv)
  if args.primary_concurrency < 1:
    parser.error("--primary-concurrency must be positive")
  return args


def main(argv=None):
  args = parse_args(argv)
  try:
    verdict = decide(
        args.baseline, args.candidate, args.objective, args.variant,
        arm=args.arm, primary=args.primary_concurrency,
        guards=args.guard_concurrencies,
    )
  except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    # A driver mid-sweep needs a decision it can act on, and "the file is not
    # there" is undecidable, not a rejection -- backing out a variant because
    # its results failed to upload would discard a good change.
    print(f"ERROR: could not decide: {exc}", file=sys.stderr)
    return EXIT_CODES[DECISION_UNDECIDABLE]
  except Exception as exc:  # pylint: disable=broad-except
    print(f"ERROR: could not decide: {type(exc).__name__}: {exc}", file=sys.stderr)
    return EXIT_CODES[DECISION_UNDECIDABLE]

  if args.output:
    with open(args.output, "w", encoding="utf-8") as handle:
      json.dump(verdict, handle, indent=2)
  if not args.quiet:
    print(render(verdict))
  return EXIT_CODES[verdict["decision"]]


if __name__ == "__main__":
  sys.exit(main())
