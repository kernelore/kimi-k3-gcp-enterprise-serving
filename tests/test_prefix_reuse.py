"""Offline gates for the prefix-reuse benchmark.

The verdict this benchmark produces -- whether prefix caching, and by extension
the NVMe cache tier, is worth anything on a model that is 69/93 linear-attention
-- is a division of two fitted slopes. That makes the fitting and the planning
worth testing on synthetic input where the right answer is known, because on
real input there is nothing to check the answer against.
"""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_prefix_reuse_bench import ARM_COLD
from benchmarks.run_prefix_reuse_bench import ARM_EVICT
from benchmarks.run_prefix_reuse_bench import ARM_WARM
from benchmarks.run_prefix_reuse_bench import DEFAULT_CORPUS_SEED
from benchmarks.run_prefix_reuse_bench import PromptFactory
from benchmarks.run_prefix_reuse_bench import analyse_slopes
from benchmarks.run_prefix_reuse_bench import extract_pool_tokens
from benchmarks.run_prefix_reuse_bench import hit_schedule
from benchmarks.run_prefix_reuse_bench import least_squares_slope
from benchmarks.run_prefix_reuse_bench import parse_args
from benchmarks.run_prefix_reuse_bench import parse_float_list
from benchmarks.run_prefix_reuse_bench import parse_int_list
from benchmarks.run_prefix_reuse_bench import plan_points
from benchmarks.run_prefix_reuse_bench import summarise_point

TOTAL_LAYERS = 93
FULL_ATTENTION_LAYERS = 24


def slope_point(arm, prefix_tokens, median_ttft_ms):
  return {
      "arm": arm,
      "prefix_tokens": prefix_tokens,
      "status": "run",
      "successful": 4,
      "ttft_ms": {"median": median_ttft_ms},
  }


class TestPromptFactory(unittest.TestCase):

  def test_same_prefix_id_is_byte_identical(self):
    factory = PromptFactory(DEFAULT_CORPUS_SEED, 4.5)
    self.assertEqual(factory.prefix("shared", 512), factory.prefix("shared", 512))

  def test_distinct_prefix_ids_differ(self):
    factory = PromptFactory(DEFAULT_CORPUS_SEED, 4.5)
    self.assertNotEqual(factory.prefix("a", 512), factory.prefix("b", 512))

  def test_suffixes_are_never_repeated(self):
    factory = PromptFactory(DEFAULT_CORPUS_SEED, 4.5)
    suffixes = [factory.suffix(64) for _ in range(40)]
    self.assertEqual(len(suffixes), len(set(suffixes)))

  def test_same_seed_reproduces_the_same_prefix(self):
    a = PromptFactory(4242, 4.5).prefix("shared", 512)
    b = PromptFactory(4242, 4.5).prefix("shared", 512)
    self.assertEqual(a, b)

  def test_prompts_carry_the_whole_prefix(self):
    factory = PromptFactory(DEFAULT_CORPUS_SEED, 4.5)
    prefix = factory.prefix("shared", 512)
    for _ in range(8):
      self.assertTrue((prefix + factory.suffix(64)).startswith(prefix))


class TestHitSchedule(unittest.TestCase):

  def test_first_request_can_never_be_a_hit(self):
    # Nothing has been served yet, so a "hit" there would be a planning lie.
    for rate in (0.1, 0.5, 0.9, 1.0):
      self.assertFalse(hit_schedule(32, rate)[0], rate)

  def test_counts_match_the_target(self):
    self.assertEqual(sum(hit_schedule(64, 0.5)), 32)
    self.assertEqual(sum(hit_schedule(64, 0.8)), 51)
    self.assertEqual(sum(hit_schedule(64, 0.0)), 0)

  def test_rate_of_one_leaves_exactly_one_miss(self):
    schedule = hit_schedule(64, 1.0)
    self.assertEqual(sum(schedule), 63)

  def test_hits_are_spread_not_clustered(self):
    # A clustered schedule would let the engine serve every hit while the pool
    # is still hot, which measures something easier than the intended workload.
    schedule = hit_schedule(64, 0.5)
    first_half = sum(schedule[:32])
    second_half = sum(schedule[32:])
    self.assertLessEqual(abs(first_half - second_half), 2)

  def test_schedule_is_deterministic(self):
    self.assertEqual(hit_schedule(37, 0.6), hit_schedule(37, 0.6))


class TestPlanning(unittest.TestCase):

  def test_context_ceiling_skips_with_a_reason(self):
    points = plan_points(
        [ARM_COLD], [131072], 256, 32, 4, 2, 131072, 0, 1.25, 8_000_000, 24_000_000
    )
    self.assertEqual(points[0]["status"], "skipped")
    self.assertIn("context ceiling", points[0]["reason"])

  def test_evict_arm_needs_a_known_pool_size(self):
    points = plan_points(
        [ARM_EVICT], [1024], 256, 32, 4, 2, 131072, 0, 1.25, 8_000_000, 24_000_000
    )
    self.assertEqual(points[0]["status"], "skipped")
    self.assertIn("KV pool size unknown", points[0]["reason"])

  def test_per_sample_flush_ceiling(self):
    points = plan_points(
        [ARM_EVICT], [1024], 256, 32, 4, 2, 131072, 10_000_000, 1.25,
        8_000_000, 100_000_000,
    )
    self.assertEqual(points[0]["status"], "skipped")
    self.assertIn("--max-flush-tokens", points[0]["reason"])

  def test_total_flush_budget_stops_later_points(self):
    # 2M pool x 1.0 overshoot x 2 evict repeats = 4M per point, so a 9M budget
    # buys two points and refuses the third.
    points = plan_points(
        [ARM_EVICT], [1024, 4096, 16384], 256, 32, 4, 2, 131072,
        2_000_000, 1.0, 8_000_000, 9_000_000,
    )
    self.assertEqual(points[0]["status"], "run")
    self.assertEqual(points[1]["status"], "run")
    self.assertEqual(points[2]["status"], "skipped")
    self.assertIn("--max-total-flush-tokens", points[2]["reason"])

  def test_evict_arm_uses_its_own_repeat_count(self):
    points = plan_points(
        [ARM_COLD, ARM_EVICT], [1024], 256, 32, 4, 2, 131072,
        1_000_000, 1.25, 8_000_000, 24_000_000,
    )
    by_arm = {p["arm"]: p for p in points}
    self.assertEqual(by_arm[ARM_COLD]["repeats"], 4)
    self.assertEqual(by_arm[ARM_EVICT]["repeats"], 2)

  def test_mixed_arm_is_not_a_slope_point(self):
    points = plan_points(
        ["mixed"], [1024], 256, 32, 4, 2, 131072, 0, 1.25, 8_000_000, 24_000_000
    )
    self.assertEqual(points, [])


class TestSlopeFitting(unittest.TestCase):

  def test_recovers_a_known_line(self):
    xs = [1000, 2000, 4000]
    ys = [100 + 0.05 * x for x in xs]
    slope, intercept = least_squares_slope(xs, ys)
    self.assertAlmostEqual(slope, 0.05, places=9)
    self.assertAlmostEqual(intercept, 100.0, places=6)

  def test_single_point_is_underdetermined(self):
    self.assertEqual(least_squares_slope([1000], [50.0]), (None, None))

  def test_zero_variance_is_underdetermined(self):
    self.assertEqual(least_squares_slope([1000, 1000], [50.0, 60.0]), (None, None))


class TestVerdict(unittest.TestCase):

  def _verdict(self, points):
    return analyse_slopes(points, TOTAL_LAYERS, FULL_ATTENTION_LAYERS)

  def test_no_cold_slope_refuses_to_normalise(self):
    verdict = self._verdict([slope_point(ARM_WARM, 1024, 500.0)])
    self.assertEqual(verdict["reuse_efficiency"], {})
    self.assertIn("nothing here can be normalised", " ".join(verdict["interpretation"]))

  def test_warm_matching_cold_reports_no_reuse(self):
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        slope_point(ARM_WARM, 1024, 200.0),
        slope_point(ARM_WARM, 4096, 500.0),
    ]
    verdict = self._verdict(points)
    self.assertAlmostEqual(verdict["reuse_efficiency"][ARM_WARM], 0.0, places=6)
    self.assertIn("being recomputed", " ".join(verdict["interpretation"]))

  def test_flat_warm_slope_reports_total_reuse(self):
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        slope_point(ARM_WARM, 1024, 60.0),
        slope_point(ARM_WARM, 4096, 60.0),
    ]
    verdict = self._verdict(points)
    self.assertAlmostEqual(verdict["reuse_efficiency"][ARM_WARM], 1.0, places=6)
    self.assertIn("exceeds the", " ".join(verdict["interpretation"]))

  def test_full_attention_only_reuse_is_named_as_such(self):
    # The case the whole benchmark exists to detect: the KV of the 24 MLA layers
    # is reused and the 69 KDA layers recompute, so ~25.8% of the slope goes.
    bound = FULL_ATTENTION_LAYERS / TOTAL_LAYERS
    cold_slope = 0.1
    points = [
        slope_point(ARM_COLD, 1024, 100.0 + cold_slope * 1024),
        slope_point(ARM_COLD, 4096, 100.0 + cold_slope * 4096),
        slope_point(ARM_WARM, 1024, 100.0 + cold_slope * (1 - bound) * 1024),
        slope_point(ARM_WARM, 4096, 100.0 + cold_slope * (1 - bound) * 4096),
    ]
    verdict = self._verdict(points)
    self.assertAlmostEqual(verdict["reuse_efficiency"][ARM_WARM], bound, places=3)
    self.assertAlmostEqual(verdict["full_attention_layer_fraction"], 0.2581, places=4)
    self.assertIn("full-attention-only bound", " ".join(verdict["interpretation"]))

  def test_tier2_verdict_when_eviction_is_recovered(self):
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        slope_point(ARM_WARM, 1024, 100.0),
        slope_point(ARM_WARM, 4096, 130.0),
        slope_point(ARM_EVICT, 1024, 110.0),
        slope_point(ARM_EVICT, 4096, 150.0),
    ]
    text = " ".join(self._verdict(points)["interpretation"])
    self.assertIn("Hierarchical cache on local NVMe is doing work", text)

  def test_tier2_verdict_when_eviction_is_recomputed(self):
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        slope_point(ARM_WARM, 1024, 100.0),
        slope_point(ARM_WARM, 4096, 130.0),
        slope_point(ARM_EVICT, 1024, 200.0),
        slope_point(ARM_EVICT, 4096, 495.0),
    ]
    text = " ".join(self._verdict(points)["interpretation"])
    self.assertIn("not justified by this measurement", text)

  def test_tier2_is_undecidable_when_warm_shows_nothing(self):
    # If reuse buys nothing even from GPU memory, there is no benefit for a disk
    # tier to preserve, and saying anything about hicache here would be invented.
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        slope_point(ARM_WARM, 1024, 200.0),
        slope_point(ARM_WARM, 4096, 500.0),
        slope_point(ARM_EVICT, 1024, 201.0),
        slope_point(ARM_EVICT, 4096, 501.0),
    ]
    text = " ".join(self._verdict(points)["interpretation"])
    self.assertIn("Tier 2 verdict: undecidable", text)

  def test_skipped_points_do_not_enter_the_fit(self):
    points = [
        slope_point(ARM_COLD, 1024, 200.0),
        slope_point(ARM_COLD, 4096, 500.0),
        {"arm": ARM_COLD, "prefix_tokens": 16384, "status": "skipped",
         "reason": "context ceiling", "ttft_ms": {"median": None}},
    ]
    self.assertEqual(self._verdict(points)["slopes"][ARM_COLD]["points"], 2)


class TestSummarise(unittest.TestCase):

  def test_median_resists_a_single_outlier(self):
    point = {"arm": ARM_COLD, "prefix_tokens": 1024, "status": "run"}
    samples = [
        {"success": True, "ttft_ms": 100.0, "prompt_tokens": 1280},
        {"success": True, "ttft_ms": 102.0, "prompt_tokens": 1280},
        {"success": True, "ttft_ms": 104.0, "prompt_tokens": 1280},
        {"success": True, "ttft_ms": 9000.0, "prompt_tokens": 1280},
    ]
    summary = summarise_point(point, samples)
    self.assertEqual(summary["ttft_ms"]["median"], 103.0)
    self.assertEqual(summary["successful"], 4)

  def test_failures_are_counted_but_not_averaged(self):
    point = {"arm": ARM_COLD, "prefix_tokens": 1024, "status": "run"}
    samples = [
        {"success": True, "ttft_ms": 100.0, "prompt_tokens": 1280},
        {"success": False, "error": "timeout"},
    ]
    summary = summarise_point(point, samples)
    self.assertEqual(summary["samples"], 2)
    self.assertEqual(summary["successful"], 1)
    self.assertEqual(summary["ttft_ms"]["median"], 100.0)

  def test_cached_tokens_absent_is_null_not_zero(self):
    point = {"arm": ARM_WARM, "prefix_tokens": 1024, "status": "run"}
    samples = [{"success": True, "ttft_ms": 50.0, "prompt_tokens": 1280}]
    self.assertIsNone(summarise_point(point, samples)["cached_tokens_reported"])


class TestServerInfo(unittest.TestCase):

  def test_top_level_pool_size(self):
    self.assertEqual(extract_pool_tokens({"max_total_num_tokens": 2400000}), 2400000)

  def test_nested_in_internal_states_list(self):
    info = {"internal_states": [{"max_total_num_tokens": 1234}]}
    self.assertEqual(extract_pool_tokens(info), 1234)

  def test_nested_in_internal_states_dict(self):
    info = {"internal_states": {"max_total_tokens": 99}}
    self.assertEqual(extract_pool_tokens(info), 99)

  def test_absent_returns_none(self):
    self.assertIsNone(extract_pool_tokens({"schedule_policy": "lpm"}))
    self.assertIsNone(extract_pool_tokens({}))
    self.assertIsNone(extract_pool_tokens(None))


class TestArgParsing(unittest.TestCase):

  def test_int_list(self):
    self.assertEqual(parse_int_list("1024, 4096"), [1024, 4096])

  def test_int_list_rejects_junk(self):
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_int_list("1024,abc")
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_int_list("")

  def test_float_list_bounds(self):
    self.assertEqual(parse_float_list("0.5,0.8"), [0.5, 0.8])
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_float_list("1.5")
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_float_list("-0.1")

  def test_defaults_parse(self):
    args = parse_args([])
    self.assertEqual(args.prefix_tokens_list, [1024, 4096, 16384])
    self.assertEqual(args.target_hit_rate_list, [0.5, 0.8])

  def test_prefix_lengths_are_sorted(self):
    self.assertEqual(
        parse_args(["--prefix-tokens", "16384,1024,4096"]).prefix_tokens_list,
        [1024, 4096, 16384],
    )

  def test_unknown_arm_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args(["--arms", "cold,lukewarm"])

  def test_zero_prefix_pool_is_rejected_up_front(self):
    # It reaches a modulo inside the mixed arm, so accepting it here would mean
    # a ZeroDivisionError partway through a run that has already been paid for.
    with self.assertRaises(SystemExit):
      parse_args(["--mixed-prefix-pool", "0"])

  def test_overshoot_must_exceed_one(self):
    with self.assertRaises(SystemExit):
      parse_args(["--eviction-overshoot", "1.0"])

  def test_layer_split_must_be_coherent(self):
    with self.assertRaises(SystemExit):
      parse_args(["--full-attention-layers", "94", "--total-layers", "93"])


if __name__ == "__main__":
  unittest.main()
