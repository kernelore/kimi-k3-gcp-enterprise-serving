"""Offline gates for the fp8 KV accuracy gate.

This gate is what stands between a throughput win and a silent quality
regression, and it only has authority if it is known to fail when it should.
A gate that passes everything is worse than no gate, because it launders a
decision nobody actually checked. So the decision function is exercised here on
synthetic captures where the right verdict is known by construction: clean fp8
passes, degraded fp8 is rejected, and a noise floor too high to resolve anything
returns inconclusive rather than a pass.
"""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.kv_accuracy_gate import CODE_RE
from benchmarks.kv_accuracy_gate import THRESHOLD_MAX_DEGENERATE_RATE
from benchmarks.kv_accuracy_gate import THRESHOLD_MAX_EXACT_MATCH_DROP
from benchmarks.kv_accuracy_gate import THRESHOLD_MAX_RETRIEVAL_DROP
from benchmarks.kv_accuracy_gate import THRESHOLD_MAX_WRONG_CODE_RATE
from benchmarks.kv_accuracy_gate import THRESHOLD_MIN_DIVERGENCE_RATIO
from benchmarks.kv_accuracy_gate import THRESHOLD_MIN_FLOOR_EXACT_MATCH
from benchmarks.kv_accuracy_gate import apply_gate
from benchmarks.kv_accuracy_gate import build_probes
from benchmarks.kv_accuracy_gate import compare_captures
from benchmarks.kv_accuracy_gate import first_divergence_fraction
from benchmarks.kv_accuracy_gate import longest_repeated_tail
from benchmarks.kv_accuracy_gate import make_code
from benchmarks.kv_accuracy_gate import parse_args
from benchmarks.kv_accuracy_gate import parse_depths
from benchmarks.kv_accuracy_gate import probe_sets_match
from benchmarks.kv_accuracy_gate import score_output
from benchmarks.kv_accuracy_gate import summarise_capture

CHARS_PER_TOKEN = 4.5

DEFAULT_THRESHOLDS = {
    "max_retrieval_drop": THRESHOLD_MAX_RETRIEVAL_DROP,
    "max_wrong_code_rate": THRESHOLD_MAX_WRONG_CODE_RATE,
    "max_exact_match_drop": THRESHOLD_MAX_EXACT_MATCH_DROP,
    "min_divergence_ratio": THRESHOLD_MIN_DIVERGENCE_RATIO,
    "min_floor_exact_match": THRESHOLD_MIN_FLOOR_EXACT_MATCH,
    "max_degenerate_rate": THRESHOLD_MAX_DEGENERATE_RATE,
}

PROBE_SET = {
    "corpus_seed": 1,
    "context_tokens": 2048,
    "depths": [0.5],
    "probes_per_depth": 20,
    "max_tokens": 256,
}


def make_capture(label, outcomes, probe_set=None):
  """Synthesise a capture file.

  Each outcome is (correct, wrong_code, text). Everything else is filled in
  consistently so the functions under test see the shape they see in production.
  """
  results = []
  for index, (correct, wrong_code, text) in enumerate(outcomes):
    results.append({
        "probe_id": f"p{index}",
        "depth": 0.5,
        "success": True,
        "correct": correct,
        "wrong_code_returned": wrong_code,
        "codes_returned": [],
        "output_chars": len(text),
        "degenerate_tail_chars": longest_repeated_tail(text),
        "output_text": text,
    })
  return {
      "capture_label": label,
      "kv_cache_dtype_declared": label,
      "probe_set": dict(probe_set or PROBE_SET),
      "results": results,
  }


def uniform_capture(label, count=20, correct=True, text_fn=None):
  text_fn = text_fn or (lambda i: f"answer {i}")
  return make_capture(
      label, [(correct, False, text_fn(i)) for i in range(count)]
  )


class TestProbeConstruction(unittest.TestCase):

  def setUp(self):
    self.probes = build_probes(7, 512, [0.1, 0.9], 3, CHARS_PER_TOKEN)

  def test_probe_count_is_depths_times_per_depth(self):
    self.assertEqual(len(self.probes), 6)

  def test_probe_set_is_deterministic(self):
    again = build_probes(7, 512, [0.1, 0.9], 3, CHARS_PER_TOKEN)
    self.assertEqual(
        [p["prompt"] for p in self.probes], [p["prompt"] for p in again]
    )

  def test_different_seed_gives_different_context(self):
    other = build_probes(8, 512, [0.1, 0.9], 3, CHARS_PER_TOKEN)
    self.assertNotEqual(self.probes[0]["prompt"], other[0]["prompt"])

  def test_every_probe_has_a_distinct_code(self):
    codes = [p["expected_code"] for p in self.probes]
    self.assertEqual(len(set(codes)), len(codes))

  def test_code_appears_exactly_once_in_its_prompt(self):
    for probe in self.probes:
      self.assertEqual(probe["prompt"].count(probe["expected_code"]), 1)

  def test_no_other_probes_code_leaks_into_a_prompt(self):
    # If probe A's code were visible in probe B's prompt, a wrong-code answer
    # could be an honest read rather than corrupted KV.
    for probe in self.probes:
      for other in self.probes:
        if other is probe:
          continue
        self.assertNotIn(other["expected_code"], probe["prompt"])

  def test_needle_lands_at_the_requested_depth(self):
    for probe in self.probes:
      actual = probe["needle_sentence_index"] / probe["sentence_count"]
      tolerance = max(0.02, 1.0 / probe["sentence_count"])
      self.assertLessEqual(abs(actual - probe["depth"]), tolerance)

  def test_question_names_the_unit_not_the_code(self):
    probe = self.probes[0]
    question = probe["prompt"].split("--- END LOG ---")[1]
    self.assertIn(probe["unit"], question)
    self.assertNotIn(probe["expected_code"], question)

  def test_generated_codes_match_the_scoring_regex(self):
    for probe in self.probes:
      self.assertRegex(probe["expected_code"], CODE_RE)

  def test_context_length_tracks_the_token_budget(self):
    short = build_probes(7, 512, [0.5], 1, CHARS_PER_TOKEN)[0]
    long = build_probes(7, 4096, [0.5], 1, CHARS_PER_TOKEN)[0]
    self.assertGreater(len(long["prompt"]), 4 * len(short["prompt"]))


class TestScoring(unittest.TestCase):

  def setUp(self):
    self.probe = {"probe_id": "p0", "depth": 0.5, "expected_code": make_code("p0")}

  def test_correct_answer_scores_correct(self):
    result = score_output(self.probe, f"{self.probe['expected_code']}")
    self.assertTrue(result["correct"])
    self.assertFalse(result["wrong_code_returned"])

  def test_correct_answer_in_a_sentence_still_scores(self):
    result = score_output(
        self.probe, f"The calibration code is {self.probe['expected_code']}."
    )
    self.assertTrue(result["correct"])

  def test_lowercase_answer_still_scores(self):
    result = score_output(
        self.probe, self.probe["expected_code"].lower()
    )
    self.assertTrue(result["correct"])

  def test_wrong_code_is_distinguished_from_no_code(self):
    wrong = score_output(self.probe, "The code is ZZ-DEADBEEF.")
    missing = score_output(self.probe, "The log does not say.")
    self.assertTrue(wrong["wrong_code_returned"])
    self.assertFalse(missing["wrong_code_returned"])
    self.assertFalse(wrong["correct"])
    self.assertFalse(missing["correct"])

  def test_correct_plus_extra_code_is_not_marked_wrong(self):
    # Reasoning aloud and landing on the right answer is a pass, not a failure.
    text = f"Could be ZZ-DEADBEEF, but it is {self.probe['expected_code']}."
    result = score_output(self.probe, text)
    self.assertTrue(result["correct"])
    self.assertFalse(result["wrong_code_returned"])

  def test_looping_output_is_detected(self):
    self.assertGreater(longest_repeated_tail("the array spins up. " * 30), 0)

  def test_normal_prose_is_not_flagged_as_looping(self):
    prose = (
        "The calibration code for the array is recorded in the operations log"
        " and appears exactly once in the section describing the unit."
    )
    self.assertEqual(longest_repeated_tail(prose), 0)

  def test_empty_output_is_not_flagged_as_looping(self):
    self.assertEqual(longest_repeated_tail(""), 0)


class TestDivergence(unittest.TestCase):

  def test_identical_strings_diverge_at_one(self):
    self.assertEqual(first_divergence_fraction("abcdef", "abcdef"), 1.0)

  def test_immediate_divergence_is_zero(self):
    self.assertEqual(first_divergence_fraction("xbcdef", "abcdef"), 0.0)

  def test_half_way_divergence(self):
    self.assertAlmostEqual(first_divergence_fraction("abcdef", "abcxyz"), 0.5)

  def test_prefix_of_longer_string_scores_one(self):
    self.assertEqual(first_divergence_fraction("abc", "abcdef"), 1.0)

  def test_two_empty_strings_agree(self):
    self.assertEqual(first_divergence_fraction("", ""), 1.0)

  def test_one_empty_string_disagrees(self):
    self.assertEqual(first_divergence_fraction("", "abc"), 0.0)


class TestCompareCaptures(unittest.TestCase):

  def test_identical_captures_match_exactly(self):
    left = uniform_capture("a")
    right = uniform_capture("b")
    result = compare_captures(left, right)
    self.assertEqual(result["exact_match_rate"], 1.0)
    self.assertEqual(result["probes_compared"], 20)

  def test_half_differing_captures(self):
    left = uniform_capture("a")
    right = uniform_capture("b", text_fn=lambda i: f"answer {i}" if i % 2 else "x")
    result = compare_captures(left, right)
    self.assertEqual(result["exact_match_rate"], 0.5)

  def test_comparison_is_keyed_on_probe_id_not_position(self):
    left = uniform_capture("a")
    right = uniform_capture("b")
    right["results"].reverse()
    self.assertEqual(compare_captures(left, right)["exact_match_rate"], 1.0)

  def test_only_shared_probes_are_compared(self):
    left = uniform_capture("a", count=20)
    right = uniform_capture("b", count=10)
    self.assertEqual(compare_captures(left, right)["probes_compared"], 10)

  def test_no_shared_probes_reports_none(self):
    left = uniform_capture("a", count=2)
    right = uniform_capture("b", count=2)
    for result in right["results"]:
      result["probe_id"] = "z" + result["probe_id"]
    self.assertIsNone(compare_captures(left, right)["exact_match_rate"])


class TestSummariseCapture(unittest.TestCase):

  def test_retrieval_accuracy(self):
    capture = make_capture(
        "x", [(i < 8, False, "t") for i in range(10)]
    )
    self.assertEqual(summarise_capture(capture)["retrieval_accuracy"], 0.8)

  def test_failed_probes_are_excluded_from_the_denominator(self):
    # A network error is not a wrong answer, and counting it as one would let a
    # flaky capture masquerade as a quality regression.
    capture = make_capture("x", [(True, False, "t") for _ in range(4)])
    capture["results"].append(
        {"probe_id": "p9", "depth": 0.5, "success": False, "error": "timeout"}
    )
    summary = summarise_capture(capture)
    self.assertEqual(summary["probes_succeeded"], 4)
    self.assertEqual(summary["retrieval_accuracy"], 1.0)

  def test_capture_with_no_successes_reports_none(self):
    capture = make_capture("x", [])
    self.assertIsNone(summarise_capture(capture)["retrieval_accuracy"])
    self.assertEqual(summarise_capture(capture)["probes_succeeded"], 0)

  def test_accuracy_is_broken_out_by_depth(self):
    capture = make_capture("x", [(True, False, "t") for _ in range(4)])
    for index, result in enumerate(capture["results"]):
      result["depth"] = 0.05 if index < 2 else 0.95
      result["correct"] = index != 3
    summary = summarise_capture(capture)
    self.assertEqual(summary["retrieval_accuracy_by_depth"]["0.05"], 1.0)
    self.assertEqual(summary["retrieval_accuracy_by_depth"]["0.95"], 0.5)


class TestGate(unittest.TestCase):

  def gate(self, baseline, repeat, candidate, **overrides):
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(overrides)
    return apply_gate(baseline, repeat, candidate, thresholds)

  def test_clean_fp8_passes(self):
    verdict = self.gate(
        uniform_capture("bf16-a"),
        uniform_capture("bf16-b"),
        uniform_capture("fp8"),
    )
    self.assertEqual(verdict["decision"], "pass")
    self.assertEqual(verdict["reasons"], [])

  def test_retrieval_regression_is_rejected(self):
    candidate = make_capture(
        "fp8", [(i < 14, False, f"answer {i}") for i in range(20)]
    )
    verdict = self.gate(
        uniform_capture("bf16-a"), uniform_capture("bf16-b"), candidate
    )
    self.assertEqual(verdict["decision"], "reject")
    self.assertTrue(any("retrieval accuracy" in r for r in verdict["reasons"]))

  def test_retrieval_drop_inside_the_threshold_passes(self):
    # 19/20 against 20/20 is a 0.05 drop, exactly at the pre-registered limit.
    candidate = make_capture(
        "fp8", [(i < 19, False, f"answer {i}") for i in range(20)]
    )
    verdict = self.gate(
        uniform_capture("bf16-a"), uniform_capture("bf16-b"), candidate
    )
    self.assertEqual(verdict["decision"], "pass")

  def test_confidently_wrong_answers_are_rejected(self):
    candidate = make_capture(
        "fp8",
        [(i < 18, i >= 18, f"answer {i}") for i in range(20)],
    )
    verdict = self.gate(
        uniform_capture("bf16-a"), uniform_capture("bf16-b"), candidate
    )
    self.assertEqual(verdict["decision"], "reject")
    self.assertTrue(any("wrong code" in r for r in verdict["reasons"]))

  def test_looping_output_is_rejected(self):
    candidate = make_capture(
        "fp8",
        [(True, False, "spin up the array. " * 30 if i >= 18 else f"answer {i}")
         for i in range(20)],
    )
    verdict = self.gate(
        uniform_capture("bf16-a"), uniform_capture("bf16-b"), candidate
    )
    self.assertEqual(verdict["decision"], "reject")
    self.assertTrue(any("repeating loop" in r for r in verdict["reasons"]))

  def test_agreement_collapse_is_rejected(self):
    baseline = uniform_capture("bf16-a")
    repeat = uniform_capture("bf16-b")
    candidate = uniform_capture("fp8", text_fn=lambda i: f"different {i}")
    verdict = self.gate(baseline, repeat, candidate)
    self.assertEqual(verdict["decision"], "reject")
    self.assertTrue(any("exact-match" in r for r in verdict["reasons"]))

  def test_a_noisy_floor_makes_the_agreement_check_inconclusive(self):
    # Two identical bf16 runs that agree on nothing mean the instrument cannot
    # resolve a quantisation effect. The absolute checks still stand, so this is
    # not a rejection -- but it is not a clean pass either.
    baseline = uniform_capture("bf16-a")
    repeat = uniform_capture("bf16-b", text_fn=lambda i: f"noise {i}")
    candidate = uniform_capture("fp8", text_fn=lambda i: f"other {i}")
    verdict = self.gate(baseline, repeat, candidate)
    self.assertEqual(verdict["decision"], "pass_with_caveat")
    self.assertTrue(verdict["agreement_check"].startswith("inconclusive"))

  def test_a_noisy_floor_does_not_rescue_a_retrieval_regression(self):
    baseline = uniform_capture("bf16-a")
    repeat = uniform_capture("bf16-b", text_fn=lambda i: f"noise {i}")
    candidate = make_capture(
        "fp8", [(i < 10, False, f"other {i}") for i in range(20)]
    )
    verdict = self.gate(baseline, repeat, candidate)
    self.assertEqual(verdict["decision"], "reject")

  def test_missing_repeat_skips_the_agreement_check(self):
    verdict = self.gate(
        uniform_capture("bf16-a"), None, uniform_capture("fp8")
    )
    self.assertEqual(verdict["decision"], "pass")
    self.assertTrue(verdict["agreement_check"].startswith("skipped"))
    self.assertIsNone(verdict["self_consistency_floor"])

  def test_missing_repeat_still_rejects_on_retrieval(self):
    candidate = make_capture(
        "fp8", [(i < 10, False, f"answer {i}") for i in range(20)]
    )
    verdict = self.gate(uniform_capture("bf16-a"), None, candidate)
    self.assertEqual(verdict["decision"], "reject")

  def test_an_empty_candidate_capture_is_inconclusive_not_a_pass(self):
    verdict = self.gate(
        uniform_capture("bf16-a"),
        uniform_capture("bf16-b"),
        make_capture("fp8", []),
    )
    self.assertEqual(verdict["decision"], "inconclusive")

  def test_early_divergence_is_rejected(self):
    # Exact-match rate is held identical to the floor (5/20 in both) while
    # divergence moves from half way through the output to the first character,
    # so only the divergence rule can catch this one. The identical probes are
    # kept to a quarter of the set: any more and they dominate the median, which
    # is what makes this rule easy to write a test that never fires.
    baseline = make_capture(
        "bf16-a", [(True, False, "abcdefghijklmnop") for _ in range(20)]
    )
    repeat = make_capture(
        "bf16-b",
        [(True, False, "abcdefghijklmnop" if i < 5 else "abcdefghXXXXXXXX")
         for i in range(20)],
    )
    candidate = make_capture(
        "fp8",
        [(True, False, "abcdefghijklmnop" if i < 5 else "aXXXXXXXXXXXXXXX")
         for i in range(20)],
    )
    verdict = self.gate(baseline, repeat, candidate)
    self.assertEqual(verdict["self_consistency_floor"]["exact_match_rate"],
                     verdict["candidate_vs_baseline"]["exact_match_rate"])
    self.assertEqual(verdict["decision"], "reject")
    self.assertTrue(any("diverge" in r for r in verdict["reasons"]))

  def test_thresholds_are_recorded_in_the_verdict(self):
    verdict = self.gate(
        uniform_capture("bf16-a"),
        uniform_capture("bf16-b"),
        uniform_capture("fp8"),
    )
    self.assertEqual(
        verdict["thresholds"]["max_retrieval_drop"],
        THRESHOLD_MAX_RETRIEVAL_DROP,
    )


class TestProbeSetMatch(unittest.TestCase):

  def test_matching_probe_sets(self):
    self.assertTrue(
        probe_sets_match(uniform_capture("a"), uniform_capture("b"))
    )

  def test_a_different_context_length_is_not_comparable(self):
    other = dict(PROBE_SET, context_tokens=4096)
    left = uniform_capture("a")
    right = make_capture("b", [(True, False, "t")], probe_set=other)
    self.assertFalse(probe_sets_match(left, right))

  def test_a_different_seed_is_not_comparable(self):
    other = dict(PROBE_SET, corpus_seed=999)
    left = uniform_capture("a")
    right = make_capture("b", [(True, False, "t")], probe_set=other)
    self.assertFalse(probe_sets_match(left, right))


class TestArgParsing(unittest.TestCase):

  def test_depths_parse_to_floats(self):
    self.assertEqual(parse_depths("0.1, 0.5,0.9"), [0.1, 0.5, 0.9])

  def test_depth_above_one_is_rejected(self):
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_depths("0.5,1.5")

  def test_empty_depths_are_rejected(self):
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_depths(" , ")

  def test_non_numeric_depths_are_rejected(self):
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_depths("shallow")

  def test_capture_defaults_give_fifty_probes(self):
    args = parse_args(["capture", "--label", "bf16-a"])
    self.assertEqual(len(args.depth_list) * args.probes_per_depth, 50)
    self.assertEqual(args.max_tokens, 256)

  def test_capture_requires_a_label(self):
    with self.assertRaises(SystemExit):
      parse_args(["capture"])

  def test_zero_probes_per_depth_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args(["capture", "--label", "x", "--probes-per-depth", "0"])

  def test_zero_chars_per_token_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args(["capture", "--label", "x", "--chars-per-token", "0"])

  def test_context_too_short_to_hold_a_needle_is_rejected(self):
    with self.assertRaises(SystemExit):
      parse_args(["capture", "--label", "x", "--context-tokens", "8"])

  def test_compare_requires_baseline_and_candidate(self):
    with self.assertRaises(SystemExit):
      parse_args(["compare", "--baseline", "a.json"])

  def test_compare_repeat_is_optional(self):
    args = parse_args(["compare", "--baseline", "a.json", "--candidate", "c.json"])
    self.assertEqual(args.repeat, "")

  def test_a_subcommand_is_required(self):
    with self.assertRaises(SystemExit):
      parse_args([])


if __name__ == "__main__":
  unittest.main()
