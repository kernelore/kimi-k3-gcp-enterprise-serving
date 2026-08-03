"""Offline gates for the non-repetitive corpus used by the A1 sweep.

The point of run_realistic_sweep_kimi_k3 is to measure what speculative
decoding does when prompts stop being predictable. That measurement is only
worth publishing if the corpus really is unpredictable and really is
reproducible, so both properties are asserted here rather than asserted in a
docstring.
"""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_realistic_sweep_kimi_k3 import DEFAULT_CORPUS_SEED
from benchmarks.run_realistic_sweep_kimi_k3 import GeneratedCorpus
from benchmarks.run_realistic_sweep_kimi_k3 import SentenceSource
from benchmarks.run_realistic_sweep_kimi_k3 import analyse_corpus
from benchmarks.run_realistic_sweep_kimi_k3 import longest_repeated_ngram
from benchmarks.run_realistic_sweep_kimi_k3 import max_shared_prompt_prefix
from benchmarks.run_realistic_sweep_kimi_k3 import parse_grid
from benchmarks.run_realistic_sweep_kimi_k3 import plan_cells

# The committed seed must produce the committed text on every interpreter this
# repo is ever run on. SentenceSource draws only through getrandbits precisely
# so that this assertion can hold; if a CPython upgrade ever breaks it, the
# published corpus is no longer reproducible and this test is the warning.
FIRST_SENTENCE_AT_DEFAULT_SEED = (
    "Although the sullen windlass tilts, the restless cornerstone barely"
    " rivals the tangled forge."
)


class TestSentenceSource(unittest.TestCase):

  def test_committed_seed_is_stable(self):
    self.assertEqual(
        SentenceSource(DEFAULT_CORPUS_SEED).next_sentence(),
        FIRST_SENTENCE_AT_DEFAULT_SEED,
    )

  def test_same_seed_reproduces_same_stream(self):
    first = [SentenceSource(4242).next_sentence() for _ in range(1)]
    source_a = SentenceSource(4242)
    source_b = SentenceSource(4242)
    self.assertEqual(
        [source_a.next_sentence() for _ in range(50)],
        [source_b.next_sentence() for _ in range(50)],
    )
    self.assertEqual(first[0], SentenceSource(4242).next_sentence())

  def test_different_seed_diverges(self):
    self.assertNotEqual(
        SentenceSource(1).next_sentence(), SentenceSource(2).next_sentence()
    )

  def test_sentences_are_unique(self):
    source = SentenceSource(7)
    sentences = [source.next_sentence() for _ in range(3000)]
    self.assertEqual(len(sentences), len(set(sentences)))
    self.assertEqual(source.emitted, 3000)

  def test_sentences_are_well_formed(self):
    source = SentenceSource(11)
    for _ in range(300):
      sentence = source.next_sentence()
      self.assertTrue(sentence[0].isupper(), sentence)
      self.assertTrue(sentence.endswith("."), sentence)
      self.assertNotIn("<", sentence)
      self.assertNotIn("  ", sentence)


class TestCorpusRepetition(unittest.TestCase):
  """The gate the earlier clause-pool corpus would have failed."""

  @classmethod
  def setUpClass(cls):
    corpus = GeneratedCorpus(DEFAULT_CORPUS_SEED, 4.5)
    cls.prompts = [corpus.build_prompt(1536) for _ in range(24)]
    cls.profile = analyse_corpus(cls.prompts, 400_000, 32)

  def test_no_duplicate_sentences(self):
    self.assertEqual(self.profile["duplicate_sentences"], 0)

  def test_repeated_8gram_rate_is_negligible(self):
    # A block-8 speculative decoder feeds on recurring 8-grams. The clause-pool
    # corpus this replaced sat at 62%.
    self.assertLess(self.profile["duplicate_8gram_fraction"] * 100.0, 2.0)

  def test_longest_repeated_span_is_bounded(self):
    self.assertLessEqual(self.profile["longest_repeated_ngram_words"], 24)

  def test_prompts_share_no_meaningful_prefix(self):
    # Prefix length is what a radix cache keys on, so this is the number that
    # decides whether the two arms get comparable cache behaviour.
    self.assertLessEqual(self.profile["max_shared_prompt_prefix_words"], 8)

  def test_prompts_are_sized_to_the_isl_target(self):
    for prompt in self.prompts:
      self.assertGreaterEqual(len(prompt), 1536 * 4.5)
      self.assertLess(len(prompt), 1536 * 4.5 + 400)


class TestRepetitionMetrics(unittest.TestCase):

  def test_longest_repeated_ngram_finds_a_planted_span(self):
    words = ("alpha beta gamma delta epsilon zeta".split()
             + "one two three".split()
             + "alpha beta gamma delta epsilon zeta".split())
    self.assertEqual(longest_repeated_ngram(words, 32), 6)

  def test_longest_repeated_ngram_on_unique_text(self):
    words = [f"w{i}" for i in range(200)]
    self.assertEqual(longest_repeated_ngram(words, 32), 0)

  def test_max_shared_prompt_prefix(self):
    prompts = [
        "the quick brown fox jumps",
        "the quick brown dog sleeps",
        "a wholly different opening",
    ]
    self.assertEqual(max_shared_prompt_prefix(prompts), 3)
    self.assertEqual(max_shared_prompt_prefix(["only one"]), 0)


class TestMatrixPlanning(unittest.TestCase):

  def test_parse_grid(self):
    self.assertEqual(parse_grid("1536:1024"), [(1536, 1024)])
    self.assertEqual(
        parse_grid("1024:1024, 8192:1024"), [(1024, 1024), (8192, 1024)]
    )

  def test_parse_grid_rejects_junk(self):
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_grid("not-a-pair")
    with self.assertRaises(argparse.ArgumentTypeError):
      parse_grid("")

  def test_context_ceiling_skips_with_a_reason(self):
    cells = plan_cells([(131072, 2048)], [8], 131072, 2_000_000, "pool")
    self.assertEqual(cells[0]["status"], "skipped")
    self.assertIn("context", cells[0]["reason"])

  def test_inflight_ceiling_skips_with_a_reason(self):
    cells = plan_cells([(32768, 2048)], [32], 131072, 265_000, "pool")
    self.assertEqual(cells[0]["status"], "skipped")
    self.assertIn("MAX_INFLIGHT_PROMPT_TOKENS", cells[0]["reason"])

  def test_issuance_controls_request_count(self):
    pooled = plan_cells([(1536, 1024)], [16], 131072, 2_000_000, "pool")
    burst = plan_cells([(1536, 1024)], [16], 131072, 2_000_000, "burst")
    self.assertEqual(pooled[0]["requests"], 32)
    self.assertEqual(burst[0]["requests"], 16)

  def test_small_concurrency_still_runs_two_waves(self):
    cells = plan_cells([(1536, 1024)], [1], 131072, 2_000_000, "pool")
    self.assertEqual(cells[0]["requests"], 8)


if __name__ == "__main__":
  unittest.main()
