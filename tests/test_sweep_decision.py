"""Offline gates for the pre-registered sweep keep/back-out rule.

The rule exists so that nine variants get judged by arithmetic fixed in advance
rather than by whoever is awake when the numbers land. That only holds if the
arithmetic is right, and the sweep is the worst possible place to find out it
is not: each variant costs a 17-minute restart plus benchmark time on a $92/h
spot pair, so a rule that accepts a regression is paid for twice -- once in the
measurement and again in everything stacked on top of it.

So the decision function is exercised here on synthetic results where the right
verdict is known by construction. A gain over the threshold is accepted, a gain
under it is not, a guard-rail regression backs the variant out even when the
primary band improves, and a grid missing the band the rule is anchored on is
undecidable rather than a pass.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.sweep_decision import ARM_NON_REPETITIVE
from benchmarks.sweep_decision import DECISION_ACCEPT
from benchmarks.sweep_decision import DECISION_BACK_OUT
from benchmarks.sweep_decision import DECISION_UNDECIDABLE
from benchmarks.sweep_decision import EXIT_CODES
from benchmarks.sweep_decision import OBJECTIVE_SCHEDULING
from benchmarks.sweep_decision import OBJECTIVE_THROUGHPUT
from benchmarks.sweep_decision import SCHEDULING_MAX_AGGREGATE_DROP_PCT
from benchmarks.sweep_decision import THROUGHPUT_MAX_REGRESSION_PCT
from benchmarks.sweep_decision import THROUGHPUT_MAX_TTFT_WORSENING_PCT
from benchmarks.sweep_decision import THROUGHPUT_MIN_GAIN_PCT
from benchmarks.sweep_decision import band_comparison
from benchmarks.sweep_decision import decide
from benchmarks.sweep_decision import load_cells
from benchmarks.sweep_decision import main
from benchmarks.sweep_decision import parse_args
from benchmarks.sweep_decision import pct_delta
from benchmarks.sweep_decision import pooled_throughput
from benchmarks.sweep_decision import render
from benchmarks.sweep_decision import worst_ttft_change

# The realistic harness's default shape: two ISL/OSL pairs across the three
# concurrency levels the plan's tables were taken at.
GRID = ((1536, 1024), (8192, 1024))
LEVELS = (8, 16, 32)
BASE_TOKENS = 100000.0
BASE_DURATION = 100.0
BASE_P99 = 500.0


def make_cells(gain=None, ttft=None, levels=LEVELS, grid=GRID, skip=()):
  """Build a cell list where each band's throughput is a known multiple.

  Every cell runs for the same duration, so pooling a band multiplies its
  tokens: a gain of 1.05 at c=16 is exactly +5.00% there and nowhere else.
  """
  gain = gain or {}
  ttft = ttft or {}
  rows = []
  for conc in levels:
    for isl, osl in grid:
      key = (isl, osl, conc)
      rows.append({
          "isl_target": isl,
          "osl": osl,
          "concurrency": conc,
          "status": "skipped" if key in skip else "ok",
          "total_tokens": BASE_TOKENS * conc * gain.get(conc, 1.0),
          "total_duration_sec": BASE_DURATION,
          "ttft_ms": {"p99": BASE_P99 * ttft.get(key, ttft.get(conc, 1.0))},
      })
  return rows


def flat_payload(**kwargs):
  """The saturation harness shape."""
  return {"engine": "sglang", "sweep_results": make_cells(**kwargs)}


def arm_payload(arm=ARM_NON_REPETITIVE, **kwargs):
  """The realistic harness shape: cells nested per arm."""
  return {
      "engine": "sglang",
      "arm_results": {
          "repeated_passage": make_cells(gain={8: 3.0, 16: 3.0, 32: 3.0}),
          arm: make_cells(**kwargs),
      },
  }


class TempResults(unittest.TestCase):
  """Writes payloads to disk, because decide() takes paths."""

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)

  def write(self, name, payload):
    path = os.path.join(self._tmp.name, name)
    with open(path, "w", encoding="utf-8") as handle:
      json.dump(payload, handle)
    return path

  def judge(self, baseline, candidate, objective=OBJECTIVE_THROUGHPUT, **kwargs):
    return decide(
        self.write("baseline.json", baseline),
        self.write("candidate.json", candidate),
        objective,
        "TEST",
        **kwargs,
    )


class TestLoadCells(TempResults):

  def test_flat_sweep_results_are_read(self):
    cells, _payload, source = load_cells(self.write("f.json", flat_payload()))
    self.assertEqual(source, "sweep_results")
    self.assertEqual(len(cells), len(LEVELS) * len(GRID))
    self.assertIn((1536, 1024, 16), cells)

  def test_arm_results_default_to_the_non_repetitive_arm(self):
    cells, _payload, source = load_cells(self.write("a.json", arm_payload()))
    self.assertEqual(source, f"arm_results[{ARM_NON_REPETITIVE}]")
    # The repeated-passage arm is 3x faster by construction. Reading it by
    # mistake is the exact error the non-repetitive harness exists to prevent,
    # so assert we got the slow one.
    tps, _count = pooled_throughput(cells, 16)
    self.assertAlmostEqual(tps, BASE_TOKENS * 16 * 2 / (BASE_DURATION * 2))

  def test_an_explicit_arm_is_honoured(self):
    path = self.write("a.json", arm_payload())
    cells, _payload, source = load_cells(path, "repeated_passage")
    self.assertEqual(source, "arm_results[repeated_passage]")
    tps, _count = pooled_throughput(cells, 16)
    self.assertAlmostEqual(tps, BASE_TOKENS * 16 * 3 * 2 / (BASE_DURATION * 2))

  def test_a_missing_arm_raises_rather_than_reading_the_wrong_one(self):
    path = self.write("a.json", arm_payload())
    with self.assertRaises(KeyError) as ctx:
      load_cells(path, "no_such_arm")
    self.assertIn("no_such_arm", str(ctx.exception))

  def test_skipped_cells_are_excluded(self):
    payload = flat_payload(skip=((8192, 1024, 32),))
    cells, _payload, _source = load_cells(self.write("f.json", payload))
    self.assertNotIn((8192, 1024, 32), cells)
    self.assertEqual(len(cells), len(LEVELS) * len(GRID) - 1)

  def test_an_empty_file_yields_no_cells_rather_than_raising(self):
    cells, _payload, source = load_cells(self.write("e.json", {}))
    self.assertEqual(cells, {})
    self.assertEqual(source, "sweep_results")


class TestArithmetic(unittest.TestCase):

  def test_pct_delta(self):
    self.assertEqual(pct_delta(100.0, 105.0), 5.0)
    self.assertEqual(pct_delta(100.0, 95.0), -5.0)
    self.assertIsNone(pct_delta(0, 5.0))
    self.assertIsNone(pct_delta(None, 5.0))

  def test_pooled_throughput_weights_by_duration_not_by_cell(self):
    # A 30-second cell and a 300-second cell are not worth the same. Pooling
    # totals rather than averaging rates is what "aggregate at c=N" means.
    cells = {
        (1024, 1024, 8): {
            "total_tokens": 1000.0, "total_duration_sec": 10.0,
        },
        (8192, 1024, 8): {
            "total_tokens": 1000.0, "total_duration_sec": 90.0,
        },
    }
    tps, count = pooled_throughput(cells, 8)
    self.assertEqual(count, 2)
    self.assertAlmostEqual(tps, 2000.0 / 100.0)
    # The mean of the two rates would be 55.6, nearly triple the truth.
    self.assertNotAlmostEqual(tps, (100.0 + 11.111) / 2, places=1)

  def test_pooled_throughput_ignores_a_band_that_never_ran(self):
    tps, count = pooled_throughput({}, 16)
    self.assertIsNone(tps)
    self.assertEqual(count, 0)

  def test_band_comparison_uses_only_cells_present_on_both_sides(self):
    base_cells = {
        (1536, 1024, 16): {"total_tokens": 1000.0, "total_duration_sec": 10.0},
        (8192, 1024, 16): {"total_tokens": 5000.0, "total_duration_sec": 10.0},
    }
    cand_cells = {
        (1536, 1024, 16): {"total_tokens": 1100.0, "total_duration_sec": 10.0},
    }
    band = band_comparison(base_cells, cand_cells, 16)
    self.assertEqual(band["matched_cells"], 1)
    # The unmatched 8192 cell must not inflate the baseline it is absent from.
    self.assertEqual(band["delta_pct"], 10.0)

  def test_worst_ttft_change_takes_the_max_not_the_mean(self):
    base_cells = {
        (1536, 1024, 16): {"ttft_ms": {"p99": 500.0}},
        (8192, 1024, 16): {"ttft_ms": {"p99": 500.0}},
    }
    cand_cells = {
        (1536, 1024, 16): {"ttft_ms": {"p99": 250.0}},
        (8192, 1024, 16): {"ttft_ms": {"p99": 1000.0}},
    }
    worst, worst_key, best = worst_ttft_change(base_cells, cand_cells)
    # Pooling would report a wash. A P99 doubling confined to one shape is
    # exactly the failure the guard is for.
    self.assertEqual(worst, 100.0)
    self.assertEqual(worst_key, (8192, 1024, 16))
    self.assertEqual(best, -50.0)

  def test_worst_ttft_change_reports_nothing_when_no_cell_has_p99(self):
    worst, worst_key, best = worst_ttft_change({}, {})
    self.assertIsNone(worst)
    self.assertIsNone(worst_key)
    self.assertIsNone(best)


class TestThroughputObjective(TempResults):

  def test_a_gain_above_the_threshold_is_accepted(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.05}),
    )
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)
    self.assertEqual(verdict["bands"][16]["delta_pct"], 5.0)

  def test_a_gain_exactly_on_the_threshold_is_accepted(self):
    # A limit that cannot be hit is not a limit. The rule says "at least 2%".
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.02}),
    )
    self.assertEqual(verdict["bands"][16]["delta_pct"], THROUGHPUT_MIN_GAIN_PCT)
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)

  def test_a_gain_below_the_threshold_is_backed_out(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.015}),
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)
    self.assertTrue(any("short of" in r for r in verdict["reasons"]))

  def test_no_change_at_all_is_backed_out(self):
    # Stacking a change that buys nothing costs a restart on every later
    # variant and makes the final config harder to explain.
    verdict = self.judge(arm_payload(), arm_payload())
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)

  def test_a_guard_rail_regression_backs_out_a_winning_primary(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.10, 32: 0.90}),
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)
    self.assertEqual(verdict["bands"][16]["delta_pct"], 10.0)
    self.assertTrue(any("c=32 aggregate regressed" in r for r in verdict["reasons"]))

  def test_a_guard_rail_regression_inside_the_limit_is_tolerated(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.10, 8: 1.0 - THROUGHPUT_MAX_REGRESSION_PCT / 100.0}),
    )
    self.assertEqual(verdict["bands"][8]["delta_pct"], -THROUGHPUT_MAX_REGRESSION_PCT)
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)

  def test_a_ttft_blowout_backs_out_a_throughput_win(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.10}, ttft={(8192, 1024, 32): 1.5}),
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)
    self.assertEqual(verdict["worst_ttft_p99_cell"], "isl8192_osl1024_c32")
    self.assertTrue(any("TTFT P99 worsened" in r for r in verdict["reasons"]))

  def test_ttft_worsening_inside_the_limit_is_tolerated(self):
    ratio = 1.0 + THROUGHPUT_MAX_TTFT_WORSENING_PCT / 100.0
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 1.10}, ttft={(1536, 1024, 8): ratio}),
    )
    self.assertEqual(verdict["worst_ttft_p99_delta_pct"],
                     THROUGHPUT_MAX_TTFT_WORSENING_PCT)
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)

  def test_a_missing_guard_band_is_noted_but_does_not_block(self):
    # Losing c=32 to a preemption should not silently pass as "guard clean".
    verdict = self.judge(
        arm_payload(levels=(8, 16)),
        arm_payload(levels=(8, 16), gain={16: 1.05}),
    )
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)
    self.assertTrue(any("no c=32 cell ran in both" in r for r in verdict["reasons"]))


class TestSchedulingObjective(TempResults):

  def test_a_ttft_win_paid_for_in_throughput_is_accepted(self):
    # This is the trade a scheduling knob is turned on to make. Judging it on
    # throughput would back out every one of them for working as intended.
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={8: 0.97, 16: 0.97, 32: 0.97}, ttft={16: 0.7}),
        objective=OBJECTIVE_SCHEDULING,
    )
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)
    self.assertEqual(verdict["bands"][16]["delta_pct"], -3.0)

  def test_the_same_throughput_loss_backs_out_a_throughput_knob(self):
    # Same numbers, different objective: the objective is what decides.
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={8: 0.97, 16: 0.97, 32: 0.97}, ttft={16: 0.7}),
        objective=OBJECTIVE_THROUGHPUT,
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)

  def test_a_throughput_drop_past_the_allowance_is_backed_out(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={8: 0.90, 16: 0.90, 32: 0.90}, ttft={16: 0.5}),
        objective=OBJECTIVE_SCHEDULING,
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)
    self.assertTrue(any("a scheduling knob" in r for r in verdict["reasons"]))

  def test_a_drop_exactly_on_the_allowance_is_tolerated(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(
            gain={c: 1.0 - SCHEDULING_MAX_AGGREGATE_DROP_PCT / 100.0 for c in LEVELS},
            ttft={16: 0.9},
        ),
        objective=OBJECTIVE_SCHEDULING,
    )
    self.assertEqual(verdict["bands"][16]["delta_pct"],
                     -SCHEDULING_MAX_AGGREGATE_DROP_PCT)
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)

  def test_no_ttft_improvement_anywhere_is_backed_out(self):
    # A scheduling knob that improves no latency and costs throughput is just
    # a slower configuration.
    verdict = self.judge(
        arm_payload(),
        arm_payload(gain={16: 0.99}, ttft={8: 1.01, 16: 1.01, 32: 1.01}),
        objective=OBJECTIVE_SCHEDULING,
    )
    self.assertEqual(verdict["decision"], DECISION_BACK_OUT)
    self.assertTrue(any("did not improve" in r for r in verdict["reasons"]))

  def test_a_local_ttft_worsening_is_noted_on_an_accepted_knob(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(ttft={(1536, 1024, 8): 0.5, (8192, 1024, 32): 1.2}),
        objective=OBJECTIVE_SCHEDULING,
    )
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)
    self.assertTrue(any(r.startswith("NOTE: TTFT P99 still worsened")
                        for r in verdict["reasons"]))


class TestUndecidable(TempResults):

  def test_a_grid_without_the_anchor_band_is_undecidable(self):
    # The saturation harness runs 1,8,32,128. Judging it at c=16 compares
    # nothing, and reporting that as a pass would stack an unmeasured change.
    verdict = self.judge(
        flat_payload(levels=(1, 8, 32, 128)),
        flat_payload(levels=(1, 8, 32, 128), gain={32: 1.5}),
    )
    self.assertEqual(verdict["decision"], DECISION_UNDECIDABLE)
    self.assertTrue(any("no c=16 cell ran in both" in r for r in verdict["reasons"]))

  def test_moving_the_anchor_makes_that_same_grid_decidable(self):
    verdict = self.judge(
        flat_payload(levels=(1, 8, 32, 128)),
        flat_payload(levels=(1, 8, 32, 128), gain={32: 1.5}),
        primary=32,
        guards=(8, 128),
    )
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)
    self.assertEqual(verdict["primary_concurrency"], 32)

  def test_the_anchor_is_not_also_checked_as_its_own_guard_rail(self):
    # Default guards are 8 and 32. Anchoring on 32 must not make the rule
    # demand a >=2% gain and a >-2% regression from the same number.
    verdict = self.judge(
        flat_payload(levels=(8, 32)),
        flat_payload(levels=(8, 32), gain={32: 1.05}),
        primary=32,
    )
    self.assertEqual(verdict["guard_concurrencies"], [8])
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)

  def test_a_half_finished_sweep_is_undecidable(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(levels=(16,), grid=((1536, 1024),)),
    )
    self.assertEqual(verdict["decision"], DECISION_UNDECIDABLE)
    self.assertTrue(any("half-finished" in r for r in verdict["reasons"]))

  def test_a_candidate_with_no_ok_cells_is_undecidable(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(skip=tuple((i, o, c) for c in LEVELS for i, o in GRID)),
    )
    self.assertEqual(verdict["decision"], DECISION_UNDECIDABLE)
    self.assertEqual(verdict["candidate_cells_ok"], 0)

  def test_comparing_across_harness_shapes_is_refused(self):
    # A realistic non-repetitive arm against a saturation run differs in
    # prompt distribution, so the delta would measure the corpus.
    verdict = self.judge(arm_payload(), flat_payload(gain={16: 1.20}))
    self.assertEqual(verdict["decision"], DECISION_UNDECIDABLE)
    self.assertEqual(verdict["baseline_source"], f"arm_results[{ARM_NON_REPETITIVE}]")
    self.assertEqual(verdict["candidate_source"], "sweep_results")
    self.assertTrue(any("measuring the corpus" in r for r in verdict["reasons"]))

  def test_the_same_shape_on_both_sides_is_not_refused(self):
    verdict = self.judge(flat_payload(), flat_payload(gain={16: 1.05}))
    self.assertEqual(verdict["decision"], DECISION_ACCEPT)


class TestRender(TempResults):

  def test_render_marks_the_anchor_band_and_states_the_decision(self):
    verdict = self.judge(arm_payload(), arm_payload(gain={16: 1.05}))
    text = render(verdict)
    self.assertIn("DECISION: ACCEPT", text)
    self.assertIn("* c=16", text)
    self.assertIn("  c=8", text)
    self.assertIn(f"arm_results[{ARM_NON_REPETITIVE}]", text)

  def test_render_survives_an_undecidable_verdict_with_empty_bands(self):
    verdict = self.judge(
        arm_payload(),
        arm_payload(levels=(16,), grid=((1536, 1024),)),
    )
    text = render(verdict)
    self.assertIn("DECISION: UNDECIDABLE", text)
    self.assertIn("not comparable", text)


class TestExitCodes(TempResults):
  """The exit status is the driver's whole interface to this file."""

  def _run(self, baseline, candidate, extra=()):
    argv = [
        "--baseline", self.write("b.json", baseline),
        "--candidate", self.write("c.json", candidate),
        "--quiet",
    ]
    return main(argv + list(extra))

  def test_accept_exits_zero(self):
    self.assertEqual(self._run(arm_payload(), arm_payload(gain={16: 1.05})), 0)

  def test_back_out_exits_one(self):
    self.assertEqual(self._run(arm_payload(), arm_payload()), 1)

  def test_undecidable_exits_two(self):
    self.assertEqual(
        self._run(arm_payload(), arm_payload(levels=(16,), grid=((1536, 1024),))), 2)

  def test_a_missing_file_is_undecidable_not_a_rejection(self):
    # Backing out a good change because its results failed to upload would
    # discard real GPU time and be near-impossible to notice afterwards.
    code = main([
        "--baseline", os.path.join(self._tmp.name, "nope.json"),
        "--candidate", self.write("c.json", arm_payload()),
        "--quiet",
    ])
    self.assertEqual(code, EXIT_CODES[DECISION_UNDECIDABLE])

  def test_malformed_json_is_undecidable(self):
    path = os.path.join(self._tmp.name, "bad.json")
    with open(path, "w", encoding="utf-8") as handle:
      handle.write("{not json")
    code = main([
        "--baseline", path,
        "--candidate", self.write("c.json", arm_payload()),
        "--quiet",
    ])
    self.assertEqual(code, EXIT_CODES[DECISION_UNDECIDABLE])

  def test_a_missing_arm_is_undecidable_rather_than_a_crash(self):
    code = self._run(arm_payload(), arm_payload(), extra=("--arm", "ghost"))
    self.assertEqual(code, EXIT_CODES[DECISION_UNDECIDABLE])

  def test_the_verdict_can_be_written_as_json(self):
    out = os.path.join(self._tmp.name, "verdict.json")
    self._run(arm_payload(), arm_payload(gain={16: 1.05}), extra=("--output", out))
    with open(out, encoding="utf-8") as handle:
      written = json.load(handle)
    self.assertEqual(written["decision"], DECISION_ACCEPT)
    self.assertEqual(written["arm"], ARM_NON_REPETITIVE)
    self.assertEqual(written["primary_concurrency"], 16)
    # The thresholds travel with the verdict so a result read months later
    # says what it was judged against, not what the file says today.
    self.assertEqual(written["thresholds"]["throughput_min_gain_pct"],
                     THROUGHPUT_MIN_GAIN_PCT)


class TestArgParsing(unittest.TestCase):

  def test_defaults_match_the_plan(self):
    args = parse_args(["--baseline", "b.json", "--candidate", "c.json"])
    self.assertEqual(args.objective, OBJECTIVE_THROUGHPUT)
    self.assertEqual(args.arm, ARM_NON_REPETITIVE)
    self.assertEqual(args.primary_concurrency, 16)
    self.assertEqual(args.guard_concurrencies, (8, 32))

  def test_guard_concurrencies_parse_to_ints(self):
    args = parse_args([
        "--baseline", "b.json", "--candidate", "c.json",
        "--guard-concurrencies", "1,8,128",
    ])
    self.assertEqual(args.guard_concurrencies, (1, 8, 128))

  def test_an_unknown_objective_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args([
          "--baseline", "b.json", "--candidate", "c.json",
          "--objective", "vibes",
      ])

  def test_a_nonsense_anchor_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args([
          "--baseline", "b.json", "--candidate", "c.json",
          "--primary-concurrency", "0",
      ])


if __name__ == "__main__":
  unittest.main()
