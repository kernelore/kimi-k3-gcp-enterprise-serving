#!/usr/bin/env python3
"""Kimi K3 prompt-predictability sweep over a non-repetitive corpus.

Why this exists
---------------
`benchmarks/results/sglang/dspark_prompt_sensitivity_c16.json` records that
accepted tokens per verify step collapses from 6.29-6.41 to 2.20-2.28 when
prompts stop being a repeated passage. That number is the single most important
qualifier on every DSPARK speedup in the README, and it is currently
unreproducible: the file records the results but not the prompts, and no
committed script regenerates them.

This harness regenerates both arms from a committed seed:

  repeated_passage  the existing saturation builder -- a 16-char nonce header
                    followed by SYNTHETIC_BASE_1K repeated to the ISL target.
                    Imported from run_saturation_sweep_kimi_k3 so the two
                    harnesses cannot drift apart.
  non_repetitive    fluent English assembled from committed word pools through
                    nine sentence frames, with a hard guarantee that no
                    sentence is emitted twice -- not within a prompt, not
                    across prompts, not across cells.

Both arms run the same ISL/OSL shapes, the same concurrency levels, the same
issuance pattern and the same request path, so the delta between them is
attributable to prompt content alone. The recorded probe could not claim that:
its two arms were 917.6 and 1523.0 prompt tokens, a 66% difference that rides
along with the acceptance difference. Here both arms are built to one ISL
target.

What the corpus is, and is not
------------------------------
The text is grammatical, fluent and novel -- every sentence is unique and the
longest span repeated anywhere in a default run is around a dozen words. It is
not, however, a document: slots are filled independently, so clauses are not
thematically connected. That makes this arm a *floor* rather than a sample of
production traffic. Real traffic sits somewhere between the repeated-passage
ceiling and this floor, and `--corpus-file` exists so an operator can measure
their own text with the same machinery and the same self-check.

`--self-check` measures the corpus rather than asserting anything about it: the
longest repeated word span, the share of 8-gram positions that recur, and the
longest prefix any two prompts share -- prefix length being what a radix cache
actually keys on. Those measurements are written into the results file next to
the numbers they qualify.

No GPU, no network and no cluster are needed for `--dry-run` or `--self-check`.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

try:
  from run_saturation_sweep_kimi_k3 import generate_unique_prompt
  from run_saturation_sweep_kimi_k3 import run_sweep_concurrency
except ImportError:
  from benchmarks.run_saturation_sweep_kimi_k3 import generate_unique_prompt
  from benchmarks.run_saturation_sweep_kimi_k3 import run_sweep_concurrency

# ==============================================================================
# Committed corpus vocabulary.
#
# Sentences are built by filling nine frames from seven word pools. Every slot
# occurrence draws independently, so an 8-word window typically spans four or
# five independent draws -- roughly 10^8 possibilities -- which is what keeps
# the repeated-8-gram rate near zero. An earlier version of this file drew
# whole clauses from five pools of ~40 phrases each; it read better but 62% of
# its 8-gram positions recurred within a single run, which is not a corpus you
# can use to measure a speculative decoder's acceptance floor.
#
# The vocabulary deliberately shares almost no content words with
# SYNTHETIC_BASE_1K, so a repeated span found by --self-check cannot be an
# artefact of the two arms describing the same subject matter.
# ==============================================================================

ADJECTIVES = [
    "brittle", "vacant", "shallow", "patient", "restless", "crooked", "faded",
    "stubborn", "narrow", "damp", "hollow", "spare", "tidy", "unruly", "quiet",
    "blunt", "weathered", "modest", "jagged", "idle", "sober", "tangled",
    "plain", "sunken", "brisk", "forgotten", "slender", "coarse", "humid",
    "splendid", "awkward", "orderly", "reluctant", "sullen", "luminous",
    "frayed", "sturdy", "obscure", "tepid", "wayward", "gaunt", "courteous",
    "lopsided",
]

NOUNS = [
    "ledger", "orchard", "harbour", "lantern", "carriage", "foreman",
    "terrace", "meadow", "cellar", "spire", "ferry", "thicket", "quarry",
    "chapel", "bridge", "kettle", "verandah", "milestone", "hedgerow",
    "cistern", "courtyard", "granary", "footbridge", "weathervane", "saddler",
    "watchman", "embankment", "boathouse", "apiary", "turnstile", "causeway",
    "dovecote", "millpond", "gatehouse", "ropewalk", "sundial", "paddock",
    "coppice", "brickyard", "tollhouse", "windlass", "sluice", "barrow",
    "stairwell", "pantry", "forge", "lighthouse", "almanac", "satchel",
    "compass", "inkwell", "cornerstone", "rooftop", "vestibule", "warden",
    "draughtsman", "apprentice", "registrar", "stonemason", "timekeeper",
    "wheelwright",
]

ADVERBS = [
    "quietly", "slowly", "abruptly", "patiently", "needlessly", "reliably",
    "faintly", "stubbornly", "briskly", "plainly", "steadily", "warily",
    "cheerfully", "grudgingly", "openly", "gently", "sharply", "loosely",
    "constantly", "barely", "gradually", "awkwardly", "firmly", "repeatedly",
    "hastily", "tidily", "wearily", "boldly", "idly", "neatly", "keenly",
    "dimly", "soundly", "promptly", "crookedly", "doggedly", "mildly",
]

TRANSITIVE_VERBS = [
    "outlives", "overshadows", "conceals", "mirrors", "unsettles", "flanks",
    "predates", "obscures", "dwarfs", "shelters", "borders", "crowds",
    "steadies", "dislodges", "encircles", "outlasts", "upstages", "shadows",
    "anchors", "undercuts", "warms", "blocks", "frames", "rivals", "supports",
    "nudges", "crosses", "disturbs", "hides", "marks", "guards", "outnumbers",
    "echoes", "muffles", "straddles", "grazes", "balances", "tempers",
    "skirts", "harbours", "unseats",
]

INTRANSITIVE_VERBS = [
    "hesitates", "settles", "creaks", "lingers", "falters", "drifts",
    "endures", "subsides", "glistens", "rattles", "persists", "tilts", "sags",
    "shudders", "widens", "dwindles", "revives", "flickers", "warps",
    "hardens", "recedes", "sways", "collapses", "stiffens", "thaws",
    "crumbles", "brightens", "leans", "wavers", "rusts", "slackens",
]

PREPOSITIONS = [
    "beyond", "beneath", "opposite", "behind", "alongside", "above", "below",
    "near", "past", "within", "outside", "beside", "under", "across from",
    "in front of", "downhill from", "upstream of", "north of", "south of",
    "east of", "west of", "adjacent to", "well clear of",
]

# Mid-sentence, after a comma, coordinators and subordinators both work.
CONJUNCTIONS = [
    "though", "while", "whereas", "and", "but", "yet", "so", "although",
    "because", "unless", "until", "wherever", "whenever",
]

# Sentence-initial, only subordinators work: "And the ledger tilts, the spire
# rivals the forge" is a comma splice, "Although the ledger tilts, ..." is not.
SUBORDINATORS = [
    "though", "while", "whereas", "although", "because", "unless", "until",
    "wherever", "whenever", "if", "after", "before", "once", "since",
]

SLOT_POOLS = {
    "adj": ADJECTIVES,
    "noun": NOUNS,
    "adv": ADVERBS,
    "vt": TRANSITIVE_VERBS,
    "vi": INTRANSITIVE_VERBS,
    "prep": PREPOSITIONS,
    "conj": CONJUNCTIONS,
    "sub": SUBORDINATORS,
}

# Nine frames from 10 to 30 words. Varying the skeleton as well as the fill
# stops the function words from landing at fixed offsets in every sentence.
# Determiners are always "the", "one", "every" or "no" so no slot can produce
# an a/an agreement error, and every verb slot is third-person singular.
FRAMES = [
    "the <adj> <noun> <adv> <vt> the <adj> <noun> <prep> the <adj> <noun>,"
    " <conj> the <adj> <noun> <vi> <adv>.",
    "<prep> the <adj> <noun> stands one <adj> <noun>, which the <adj> <noun>"
    " <adv> <vt>.",
    "<sub> the <adj> <noun> <vi>, the <adj> <noun> <adv> <vt> the <adj>"
    " <noun>.",
    "the <adj> <noun>, like the <adj> <noun>, <vi> <adv> <prep> the <adj>"
    " <noun>.",
    "no <adj> <noun> <vt> the <adj> <noun> as <adv> as the <adj> <noun> <vi>.",
    "the <adj> <noun> <vt> whatever the <adj> <noun> <vi> <prep>, <conj>"
    " nothing <adv> <vi>.",
    "every <adj> <noun> <prep> the <adj> <noun> <vi> <adv>.",
    "the <adj> <noun> <adv> <vt> the <adj> <noun>, and the <adj> <noun> <vt>"
    " the <adj> <noun> <prep> the <adj> <noun>.",
    "<sub> the <adj> <noun> <adv> <vi>, one <adj> <noun> <vt> the <adj>"
    " <noun> <prep> the <adj> <noun>, <conj> the <adj> <noun> <vi>.",
]

_SLOT_RE = re.compile(r"<([a-z]+)>")

# Sentences per paragraph. Paragraph breaks cost a token each and make the
# prompt legible when it is dumped for inspection; they carry no other meaning.
SENTENCES_PER_PARAGRAPH = 5

# Default seed. Any integer reproduces its own corpus exactly; this one is the
# value used for the committed numbers.
DEFAULT_CORPUS_SEED = 20260803

# English prose against the Kimi K3 tokenizer sits near 4.5 characters per
# token, versus 4.84 for the dense technical passage the repeated arm uses (see
# BASE_TOKENS_APPROX in run_saturation_sweep_kimi_k3). Only the char budget
# depends on this; prompt_tokens_observed always reports what the engine
# actually counted, so recalibrate from a real run rather than trusting it.
DEFAULT_CHARS_PER_TOKEN = 4.5

MAX_CONTEXT_TOKENS = 131_072
MAX_INFLIGHT_PROMPT_TOKENS = 2_000_000

DEFAULT_SPEC_VERIFY_METRIC = "sglang:spec_verify_calls_total"

ARM_REPEATED = "repeated_passage"
ARM_NON_REPETITIVE = "non_repetitive"
KNOWN_ARMS = (ARM_REPEATED, ARM_NON_REPETITIVE)


class SentenceSource:
  """Deterministic, non-repeating sentence generator.

  Draws are made only through random.Random.getrandbits, whose Mersenne
  Twister stream is specified and stable across CPython versions and
  platforms. randrange, choice and sample are not guaranteed to be, and a
  benchmark corpus that changes when the interpreter is upgraded is not a
  reproducible corpus.

  Uniqueness is enforced, not assumed: every emitted (frame, draws) key is
  recorded and re-drawn on collision, so no sentence can appear twice in a run.
  """

  def __init__(self, seed):
    self._rng = random.Random(seed)
    self._seen = set()
    self._collisions = 0
    self._frame_slots = [_SLOT_RE.findall(frame) for frame in FRAMES]

  def _draw_below(self, n):
    bits = max(1, (n - 1).bit_length())
    while True:
      val = self._rng.getrandbits(bits)
      if val < n:
        return val

  def next_sentence(self):
    while True:
      frame_idx = self._draw_below(len(FRAMES))
      slots = self._frame_slots[frame_idx]
      draws = tuple(self._draw_below(len(SLOT_POOLS[s])) for s in slots)
      key = (frame_idx, draws)
      if key in self._seen:
        self._collisions += 1
        continue
      self._seen.add(key)
      break
    values = iter(
        SLOT_POOLS[slot][idx] for slot, idx in zip(slots, draws)
    )
    sentence = _SLOT_RE.sub(lambda _m: next(values), FRAMES[frame_idx])
    return sentence[0].upper() + sentence[1:]

  @property
  def emitted(self):
    return len(self._seen)

  @property
  def collisions(self):
    return self._collisions

  def space_size(self):
    """Distinct sentences this grammar can produce."""
    total = 0
    for slots in self._frame_slots:
      product = 1
      for slot in slots:
        product *= len(SLOT_POOLS[slot])
      total += product
    return total


class GeneratedCorpus:
  """Builds prompts to a character budget from a SentenceSource."""

  def __init__(self, seed, chars_per_token):
    self.source = SentenceSource(seed)
    self.chars_per_token = chars_per_token
    self.seed = seed

  def build_prompt(self, isl_target):
    budget = int(round(isl_target * self.chars_per_token))
    sentences = []
    length = 0
    while length < budget:
      sentence = self.source.next_sentence()
      sentences.append(sentence)
      length += len(sentence) + 1
    paragraphs = [
        " ".join(sentences[i:i + SENTENCES_PER_PARAGRAPH])
        for i in range(0, len(sentences), SENTENCES_PER_PARAGRAPH)
    ]
    return "\n\n".join(paragraphs)

  def describe(self):
    return {
        "kind": "generated",
        "seed": self.seed,
        "chars_per_token_assumed": self.chars_per_token,
        "frames": len(FRAMES),
        "pool_sizes": {k: len(v) for k, v in sorted(SLOT_POOLS.items())},
        "sentence_space": self.source.space_size(),
        "sentences_emitted": self.source.emitted,
        "draw_collisions_resolved": self.source.collisions,
        "sentence_uniqueness": "enforced",
    }


class FileCorpus:
  """Slices a text file into non-overlapping word windows, one per prompt."""

  def __init__(self, path, chars_per_token):
    with open(path, "r", encoding="utf-8") as handle:
      self._words = handle.read().split()
    if not self._words:
      raise ValueError(f"Corpus file {path} contains no words")
    self._cursor = 0
    self.chars_per_token = chars_per_token
    self.path = path

  def build_prompt(self, isl_target):
    budget = int(round(isl_target * self.chars_per_token))
    start = self._cursor
    length = 0
    while length < budget and self._cursor < len(self._words):
      length += len(self._words[self._cursor]) + 1
      self._cursor += 1
    if length < budget:
      raise ValueError(
          f"Corpus file {self.path} exhausted after {start} words: needed"
          f" ~{budget} more characters for one prompt at ISL={isl_target}."
          " Supply a longer file or reduce the grid."
      )
    return " ".join(self._words[start:self._cursor])

  def describe(self):
    return {
        "kind": "file",
        "path": os.path.basename(self.path),
        "chars_per_token_assumed": self.chars_per_token,
        "total_words": len(self._words),
        "words_consumed": self._cursor,
        "sentence_uniqueness": "windows are non-overlapping",
    }


def _gram_digest(words, start, length):
  joined = " ".join(words[start:start + length]).encode("utf-8")
  return hashlib.blake2b(joined, digest_size=8).digest()


def _has_repeated_ngram(words, length):
  seen = set()
  for i in range(len(words) - length + 1):
    digest = _gram_digest(words, i, length)
    if digest in seen:
      return True
    seen.add(digest)
  return False


def longest_repeated_ngram(words, cap):
  """Longest word n-gram (up to cap) occurring more than once.

  Uses 64-bit digests rather than the tuples themselves to keep memory flat on
  long corpora. A digest collision can only overstate the answer, which is the
  safe direction for a gate.
  """
  if len(words) < 2:
    return 0
  lo, hi = 0, min(cap, len(words) - 1)
  while lo < hi:
    mid = (lo + hi + 1) // 2
    if _has_repeated_ngram(words, mid):
      lo = mid
    else:
      hi = mid - 1
  return lo


def duplicate_ngram_fraction(words, length):
  """Fraction of n-gram positions whose n-gram appears more than once."""
  positions = len(words) - length + 1
  if positions <= 0:
    return 0.0
  counts = {}
  for i in range(positions):
    digest = _gram_digest(words, i, length)
    counts[digest] = counts.get(digest, 0) + 1
  repeated = sum(n for n in counts.values() if n > 1)
  return repeated / positions


def max_shared_prompt_prefix(prompts, cap=64):
  """Longest word prefix shared by any two prompts."""
  if len(prompts) < 2:
    return 0
  heads = sorted(tuple(p.split()[:cap]) for p in prompts)
  best = 0
  for first, second in zip(heads, heads[1:]):
    shared = 0
    for word_a, word_b in zip(first, second):
      if word_a != word_b:
        break
      shared += 1
    best = max(best, shared)
  return best


def split_sentences(prompts):
  sentences = []
  for prompt in prompts:
    for chunk in prompt.replace("\n", " ").split(". "):
      chunk = chunk.strip().rstrip(".")
      if chunk:
        sentences.append(chunk)
  return sentences


def analyse_corpus(prompts, max_words, ngram_cap):
  """Repetition profile of a prompt set. Pure CPU, no network."""
  words = []
  truncated = False
  for prompt in prompts:
    if len(words) >= max_words:
      truncated = True
      break
    words.extend(prompt.split())
  if len(words) > max_words:
    words = words[:max_words]
    truncated = True

  sentences = split_sentences(prompts)

  return {
      "prompts": len(prompts),
      "words_analysed": len(words),
      "analysis_truncated": truncated,
      "sentences": len(sentences),
      "duplicate_sentences": len(sentences) - len(set(sentences)),
      "longest_repeated_ngram_words": longest_repeated_ngram(words, ngram_cap),
      "ngram_cap": ngram_cap,
      "duplicate_8gram_fraction": round(duplicate_ngram_fraction(words, 8), 6),
      "max_shared_prompt_prefix_words": max_shared_prompt_prefix(prompts),
  }


def scrape_counter_sum(metrics_endpoint, metric_name):
  """Sum every Prometheus series named metric_name, or None if unavailable."""
  if not metrics_endpoint or not metric_name:
    return None
  try:
    req = urllib.request.Request(
        metrics_endpoint,
        headers={"User-Agent": "KIMI3-Prompt-Sensitivity-Metrics/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
      content = resp.read().decode("utf-8", errors="replace")
  except Exception as exc:
    print(
        f"WARNING: could not read {metric_name} from {metrics_endpoint}: {exc}",
        file=sys.stderr,
    )
    return None

  total = None
  for line in content.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    parts = line.split()
    if len(parts) < 2:
      continue
    if parts[0].split("{")[0] != metric_name:
      continue
    try:
      total = (total or 0.0) + float(parts[1])
    except ValueError:
      continue
  if total is None:
    print(
        f"WARNING: metric '{metric_name}' absent from {metrics_endpoint};"
        " acceptance per verify step will be recorded as null.",
        file=sys.stderr,
    )
  return total


def build_prompt_set(arm, corpus, count, c, isl_target):
  """Pre-render every prompt for a cell, single-threaded.

  The generated corpus is stateful: which sentences land in which prompt
  depends on draw order. Rendering up front rather than inside the worker
  threads keeps that order independent of thread scheduling, which is what
  makes a seed reproduce a run.
  """
  if arm == ARM_REPEATED:
    return [generate_unique_prompt(i, c, isl_target) for i in range(count)]
  return [corpus.build_prompt(isl_target) for _ in range(count)]


def parse_grid(spec):
  grid = []
  for pair in spec.split(","):
    pair = pair.strip()
    if not pair:
      continue
    isl, _, osl = pair.partition(":")
    try:
      grid.append((int(isl), int(osl)))
    except ValueError:
      raise argparse.ArgumentTypeError(
          f"--isl-osl-grid entry '{pair}' is not an ISL:OSL integer pair"
      )
  if not grid:
    raise argparse.ArgumentTypeError(
        "--isl-osl-grid must contain at least one ISL:OSL pair"
    )
  return grid


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Kimi K3 prompt-predictability sweep -- repeated passage vs"
          " non-repetitive corpus at a matched ISL target"
      )
  )
  parser.add_argument(
      "--dry-run",
      dest="dry_run",
      action="store_true",
      help="Build the corpus and validate the matrix without contacting an engine",
  )
  parser.add_argument(
      "--self-check",
      dest="self_check",
      action="store_true",
      help=(
          "Build the corpus, measure its repetition profile against the"
          " thresholds and exit non-zero if it regresses. Offline."
      ),
  )
  parser.add_argument(
      "--endpoint",
      default="http://localhost:8000/v1/completions",
      help="OpenAI-compatible completions endpoint URL",
  )
  parser.add_argument(
      "--model", default="moonshotai/Kimi-K3", help="Served model ID"
  )
  parser.add_argument(
      "--output",
      default="benchmarks/realistic_sweep_results_kimi_k3.json",
      help="Output JSON path",
  )
  parser.add_argument(
      "--engine", default="sglang", help="Inference engine (sglang or trtllm)"
  )
  parser.add_argument(
      "--metadata", default="{}", help="JSON string of engine metadata"
  )
  parser.add_argument(
      "--api-key", default="", help="Optional API key for gateway authentication"
  )
  parser.add_argument(
      "--arms",
      default=",".join(KNOWN_ARMS),
      help=(
          "Comma-separated arms to run, in order. Both by default so the"
          " comparison is controlled within a single run."
      ),
  )
  parser.add_argument(
      "--concurrency-levels",
      dest="concurrency_levels",
      default="8,16,32",
      type=lambda s: [int(x) for x in s.split(",") if x.strip()],
      help=(
          "Comma-separated concurrency levels. Defaults to the levels the"
          " recorded DSPARK prompt-sensitivity probe used."
      ),
  )
  parser.add_argument(
      "--isl-osl-grid",
      dest="isl_osl_grid",
      default="1536:1024",
      type=parse_grid,
      help=(
          "Comma-separated ISL:OSL pairs. The default matches the recorded"
          " probe's ~1523-token non-repetitive arm at OSL 1024. Pass"
          " '1024:1024,8192:1024,32768:2048' to align with the saturation grid."
      ),
  )
  parser.add_argument(
      "--issuance",
      choices=["pool", "burst"],
      default="pool",
      help=(
          "'pool' issues 2c requests through a c-wide pool, matching the"
          " saturation sweep. 'burst' issues exactly c requests at once,"
          " matching the recorded probe, which pays more batch drain."
      ),
  )
  parser.add_argument(
      "--corpus-seed",
      dest="corpus_seed",
      default=DEFAULT_CORPUS_SEED,
      type=int,
      help="Seed for the generated corpus",
  )
  parser.add_argument(
      "--corpus-file",
      dest="corpus_file",
      default="",
      help=(
          "Optional UTF-8 text file to use instead of the generated corpus."
          " Sliced into non-overlapping windows, one per request."
      ),
  )
  parser.add_argument(
      "--chars-per-token",
      dest="chars_per_token",
      default=DEFAULT_CHARS_PER_TOKEN,
      type=float,
      help="Characters per token used to size prompts to the ISL target",
  )
  parser.add_argument(
      "--max-inflight-prompt-tokens",
      dest="max_inflight_prompt_tokens",
      default=MAX_INFLIGHT_PROMPT_TOKENS,
      type=int,
      help="Maximum simultaneous in-flight prompt tokens before skipping a cell",
  )
  parser.add_argument(
      "--max-context-tokens",
      dest="max_context_tokens",
      default=MAX_CONTEXT_TOKENS,
      type=int,
      help="Engine context window; cells whose ISL+OSL exceeds it are skipped",
  )
  parser.add_argument(
      "--metrics-endpoint",
      default="",
      help="Optional Prometheus /metrics URL of the serving engine leader",
  )
  parser.add_argument(
      "--metrics-names",
      default="",
      help="Comma-separated exact metric names to sample for peak values",
  )
  parser.add_argument(
      "--spec-verify-metric",
      dest="spec_verify_metric",
      default=DEFAULT_SPEC_VERIFY_METRIC,
      help=(
          "Counter whose per-cell delta divides output tokens to give accepted"
          " tokens per verify step. Requires --metrics-endpoint."
      ),
  )
  parser.add_argument(
      "--max-repeat-ngram",
      dest="max_repeat_ngram",
      default=24,
      type=int,
      help=(
          "Self-check ceiling on the longest repeated word span. A default run"
          " lands well under this; the headroom is for larger grids, where"
          " more sentences mean more chances for two windows to coincide."
      ),
  )
  parser.add_argument(
      "--max-duplicate-8gram-pct",
      dest="max_duplicate_8gram_pct",
      default=2.0,
      type=float,
      help=(
          "Self-check ceiling on the percentage of 8-gram positions that"
          " recur. This is the number that matters for a block-8 speculative"
          " decoder, and the one that failed the earlier clause-pool corpus."
      ),
  )
  parser.add_argument(
      "--max-shared-prefix",
      dest="max_shared_prefix",
      default=8,
      type=int,
      help=(
          "Self-check ceiling on the longest word prefix shared by two"
          " prompts. Prefix sharing is what a radix cache actually keys on."
      ),
  )
  parser.add_argument(
      "--self-check-max-words",
      dest="self_check_max_words",
      default=400_000,
      type=int,
      help="Word ceiling for the repetition analysis; truncation is reported",
  )
  return parser.parse_args()


def plan_cells(grid, levels, max_context, max_inflight, issuance):
  """Expand the matrix into runnable and skipped cells, mirroring the sweep."""
  cells = []
  for isl, osl in grid:
    for c in levels:
      if isl + osl > max_context:
        cells.append({
            "isl_target": isl,
            "osl": osl,
            "concurrency": c,
            "status": "skipped",
            "reason": (
                f"ISL+OSL={isl + osl:,} tokens exceeds the engine context"
                f" window MAX_CONTEXT_TOKENS={max_context:,}; the engine"
                " rejects such requests with HTTP 400 before any tokens are"
                " generated"
            ),
        })
        continue
      inflight = c * isl
      if inflight > max_inflight:
        cells.append({
            "isl_target": isl,
            "osl": osl,
            "concurrency": c,
            "status": "skipped",
            "reason": (
                f"{inflight:,} in-flight prompt tokens exceeds"
                f" MAX_INFLIGHT_PROMPT_TOKENS={max_inflight:,}"
            ),
        })
        continue
      cells.append({
          "isl_target": isl,
          "osl": osl,
          "concurrency": c,
          "status": "planned",
          "requests": c if issuance == "burst" else max(c * 2, 8),
      })
  return cells


def make_corpus(args):
  if args.corpus_file:
    return FileCorpus(args.corpus_file, args.chars_per_token)
  return GeneratedCorpus(args.corpus_seed, args.chars_per_token)


def render_all_prompts(arms, cells, corpus):
  """Render every prompt the run will send, arm by arm, cell by cell."""
  rendered = {}
  for arm in arms:
    for cell in cells:
      if cell["status"] != "planned":
        continue
      key = (arm, cell["isl_target"], cell["osl"], cell["concurrency"])
      rendered[key] = build_prompt_set(
          arm,
          corpus,
          cell["requests"],
          cell["concurrency"],
          cell["isl_target"],
      )
  return rendered


def collect_arm_prompts(rendered, arm):
  return [
      prompt
      for key, prompts in sorted(rendered.items(), key=lambda kv: str(kv[0]))
      if key[0] == arm
      for prompt in prompts
  ]


def report_self_check(args, corpus, rendered):
  """Print the repetition profile and return (profile, ok)."""
  prompts = collect_arm_prompts(rendered, ARM_NON_REPETITIVE)
  if not prompts:
    print(
        f"[SKIP] No '{ARM_NON_REPETITIVE}' arm in --arms; nothing to"
        " self-check.",
        file=sys.stderr,
    )
    return None, True

  profile = analyse_corpus(
      prompts, args.self_check_max_words, args.max_repeat_ngram + 8
  )
  duplicate_pct = profile["duplicate_8gram_fraction"] * 100.0

  print("\n=== NON-REPETITIVE CORPUS PROFILE ===")
  print(f"  Prompts rendered:             {profile['prompts']}")
  print(f"  Sentences:                    {profile['sentences']}")
  print(f"  Duplicate sentences:          {profile['duplicate_sentences']}")
  print(f"  Words analysed:               {profile['words_analysed']:,}")
  if profile["analysis_truncated"]:
    print(
        "  NOTE: analysis truncated at --self-check-max-words="
        f"{args.self_check_max_words:,}; spans beyond that point were not"
        " examined."
    )
  print(
      f"  Longest repeated span:        "
      f"{profile['longest_repeated_ngram_words']} words"
      f" (ceiling {args.max_repeat_ngram})"
  )
  print(
      f"  Repeated 8-gram positions:    {duplicate_pct:.3f}%"
      f" (ceiling {args.max_duplicate_8gram_pct}%)"
  )
  print(
      f"  Longest shared prompt prefix: "
      f"{profile['max_shared_prompt_prefix_words']} words"
      f" (ceiling {args.max_shared_prefix})"
  )

  failures = []
  if profile["duplicate_sentences"] != 0:
    failures.append(
        f"{profile['duplicate_sentences']} duplicate sentences (expected 0)"
    )
  if profile["longest_repeated_ngram_words"] > args.max_repeat_ngram:
    failures.append(
        f"longest repeated span {profile['longest_repeated_ngram_words']} words"
        f" exceeds --max-repeat-ngram={args.max_repeat_ngram}"
    )
  if duplicate_pct > args.max_duplicate_8gram_pct:
    failures.append(
        f"repeated 8-gram positions {duplicate_pct:.3f}% exceeds"
        f" --max-duplicate-8gram-pct={args.max_duplicate_8gram_pct}"
    )
  if profile["max_shared_prompt_prefix_words"] > args.max_shared_prefix:
    failures.append(
        f"shared prompt prefix {profile['max_shared_prompt_prefix_words']}"
        f" words exceeds --max-shared-prefix={args.max_shared_prefix}"
    )

  if failures:
    print("\nCORPUS SELF-CHECK FAILED:", file=sys.stderr)
    for failure in failures:
      print(f"  * {failure}", file=sys.stderr)
    return profile, False

  print("  VERDICT: corpus is within all repetition thresholds.")
  return profile, True


def summarise_arms(arm_results):
  """Pair the arms cell by cell so the delta is readable without a spreadsheet."""
  by_cell = {}
  for arm, cells in arm_results.items():
    for cell in cells:
      if cell.get("status") != "ok":
        continue
      key = (cell["isl_target"], cell["osl"], cell["concurrency"])
      by_cell.setdefault(key, {})[arm] = cell

  comparison = []
  for (isl, osl, c), arms in sorted(by_cell.items()):
    rep = arms.get(ARM_REPEATED)
    novel = arms.get(ARM_NON_REPETITIVE)
    if not rep or not novel:
      continue
    entry = {
        "isl_target": isl,
        "osl": osl,
        "concurrency": c,
        ARM_REPEATED: {
            "prompt_tokens_observed": rep.get("prompt_tokens_observed"),
            "aggregate_tok_s": rep.get("aggregate_tok_s"),
            "accepted_tok_per_step": rep.get("accepted_tok_per_step"),
        },
        ARM_NON_REPETITIVE: {
            "prompt_tokens_observed": novel.get("prompt_tokens_observed"),
            "aggregate_tok_s": novel.get("aggregate_tok_s"),
            "accepted_tok_per_step": novel.get("accepted_tok_per_step"),
        },
    }
    rep_tps = rep.get("aggregate_tok_s") or 0.0
    if rep_tps > 0:
      entry["non_repetitive_over_repeated_tok_s"] = round(
          (novel.get("aggregate_tok_s") or 0.0) / rep_tps, 4
      )
    rep_acc = rep.get("accepted_tok_per_step")
    novel_acc = novel.get("accepted_tok_per_step")
    if rep_acc and novel_acc:
      entry["non_repetitive_over_repeated_acceptance"] = round(
          novel_acc / rep_acc, 4
      )
    rep_pt = rep.get("prompt_tokens_observed") or 0.0
    novel_pt = novel.get("prompt_tokens_observed") or 0.0
    if rep_pt > 0:
      entry["prompt_token_ratio"] = round(novel_pt / rep_pt, 4)
    comparison.append(entry)
  return comparison


def run_one_cell(args, arm, cell, prompts, metrics_names_list):
  """Run a single arm/cell and attach its speculative-acceptance delta."""

  def prompt_builder(req_idx, _c, _isl):
    return prompts[req_idx % len(prompts)]

  verify_before = scrape_counter_sum(
      args.metrics_endpoint, args.spec_verify_metric
  )
  result = run_sweep_concurrency(
      cell["concurrency"],
      cell["isl_target"],
      cell["osl"],
      cell["requests"],
      args.endpoint,
      args.model,
      api_key=args.api_key,
      metrics_endpoint=args.metrics_endpoint,
      metrics_names=metrics_names_list,
      prompt_builder=prompt_builder,
      label=f"Prompt-Sensitivity Sweep [{arm}]",
  )
  verify_after = scrape_counter_sum(
      args.metrics_endpoint, args.spec_verify_metric
  )

  result["arm"] = arm
  result["issuance"] = args.issuance
  result["spec_verify_calls"] = None
  result["accepted_tok_per_step"] = None
  if verify_before is not None and verify_after is not None:
    delta = verify_after - verify_before
    result["spec_verify_calls"] = delta
    if delta > 0:
      result["accepted_tok_per_step"] = round(
          result.get("total_tokens", 0) / delta, 4
      )
  return result


def main():
  args = parse_args()
  arms = [a.strip() for a in args.arms.split(",") if a.strip()]
  unknown = [a for a in arms if a not in KNOWN_ARMS]
  if unknown:
    print(
        f"ERROR: unknown arm(s): {', '.join(unknown)}."
        f" Known arms: {', '.join(KNOWN_ARMS)}",
        file=sys.stderr,
    )
    sys.exit(2)

  print(f"\n=== KIMI K3 PROMPT-SENSITIVITY SWEEP (Engine: {args.engine}) ===")
  start_dt = datetime.now(timezone.utc)
  suite_start_ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  cells = plan_cells(
      args.isl_osl_grid,
      args.concurrency_levels,
      args.max_context_tokens,
      args.max_inflight_prompt_tokens,
      args.issuance,
  )
  for cell in cells:
    if cell["status"] == "skipped":
      print(
          f"SKIPPED cell ISL={cell['isl_target']} OSL={cell['osl']}"
          f" c={cell['concurrency']} -> {cell['reason']}"
      )
    else:
      print(
          f"RUN cell ISL={cell['isl_target']} OSL={cell['osl']}"
          f" c={cell['concurrency']} -> {cell['requests']} requests per arm"
          f" ({args.issuance} issuance)"
      )

  corpus = make_corpus(args)
  rendered = render_all_prompts(arms, cells, corpus)

  if args.self_check:
    _, ok = report_self_check(args, corpus, rendered)
    sys.exit(0 if ok else 1)

  if args.dry_run:
    run_count = sum(1 for cell in cells if cell["status"] == "planned")
    skip_count = sum(1 for cell in cells if cell["status"] == "skipped")
    described = corpus.describe()
    print("\n[INFO] Corpus:")
    for key in sorted(described):
      print(f"  {key}: {described[key]}")
    print(
        "\n[SUCCESS] Kimi K3 prompt-sensitivity harness syntax and matrix"
        f" verified (ARMS: {len(arms)}, RUN: {run_count},"
        f" SKIPPED: {skip_count})."
    )
    sys.exit(0)

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

  if not args.metrics_endpoint:
    print(
        "NOTE: without --metrics-endpoint the speculative-acceptance delta"
        " cannot be read, so accepted_tok_per_step will be null and the"
        " headline finding of this sweep will not be reproduced.",
        file=sys.stderr,
    )
    acceptance_source = "not measured (no --metrics-endpoint)"
  else:
    acceptance_source = f"output tokens / {args.spec_verify_metric} delta"

  arm_results = {}
  for arm in arms:
    arm_cells = []
    for cell in cells:
      if cell["status"] == "skipped":
        arm_cells.append({
            "isl_target": cell["isl_target"],
            "osl": cell["osl"],
            "concurrency": cell["concurrency"],
            "prompt_tokens_observed": 0,
            "status": "skipped",
            "reason": cell["reason"],
            "arm": arm,
            "peak_metrics": None,
        })
        continue
      key = (arm, cell["isl_target"], cell["osl"], cell["concurrency"])
      arm_cells.append(
          run_one_cell(args, arm, cell, rendered[key], metrics_names_list)
      )
      time.sleep(2)
    arm_results[arm] = arm_cells

  try:
    meta_dict = json.loads(args.metadata) if args.metadata else {}
  except (json.JSONDecodeError, TypeError, ValueError):
    meta_dict = {"raw": args.metadata}

  end_dt = datetime.now(timezone.utc)
  suite_end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

  all_cells = [cell for cells_ in arm_results.values() for cell in cells_]
  sources_used = sorted({
      cell.get("token_count_source", "unknown")
      for cell in all_cells
      if cell.get("status") == "ok"
  })
  if len(sources_used) == 1:
    agg_source = sources_used[0]
  elif len(sources_used) > 1:
    agg_source = "mixed: " + ", ".join(sources_used)
  else:
    agg_source = "none"

  novel_prompts = collect_arm_prompts(rendered, ARM_NON_REPETITIVE)
  corpus_profile = (
      analyse_corpus(
          novel_prompts,
          args.self_check_max_words,
          args.max_repeat_ngram + 8,
      )
      if novel_prompts
      else None
  )

  try:
    from telemetry_sanitizer import sanitize_telemetry
  except ImportError:
    from benchmarks.telemetry_sanitizer import sanitize_telemetry

  out_payload = {
      "engine": args.engine,
      "metadata": meta_dict,
      "token_count_source": agg_source,
      "acceptance_source": acceptance_source,
      "grid": {
          "ISL_OSL_GRID": [list(pair) for pair in args.isl_osl_grid],
          "sweep_levels": args.concurrency_levels,
          "arms": arms,
          "issuance": args.issuance,
          "MAX_INFLIGHT_PROMPT_TOKENS": args.max_inflight_prompt_tokens,
          "MAX_CONTEXT_TOKENS": args.max_context_tokens,
          "suite_start_ts": suite_start_ts,
          "suite_end_ts": suite_end_ts,
          "suite_duration_s": round((end_dt - start_dt).total_seconds(), 4),
      },
      "corpus": corpus.describe(),
      "corpus_profile": corpus_profile,
      "arm_results": arm_results,
      "arm_comparison": summarise_arms(arm_results),
  }
  out_payload = sanitize_telemetry(out_payload, args.output)

  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as handle:
    json.dump(out_payload, handle, indent=2)
  print(f"\nSaved prompt-sensitivity sweep results to {args.output}")

  for entry in out_payload["arm_comparison"]:
    print(
        f"  c={entry['concurrency']:<4} ISL={entry['isl_target']}"
        f"  repeated={entry[ARM_REPEATED]['aggregate_tok_s']}"
        f"  non_repetitive={entry[ARM_NON_REPETITIVE]['aggregate_tok_s']}"
        f"  tok_s_ratio={entry.get('non_repetitive_over_repeated_tok_s')}"
        "  acceptance_ratio="
        f"{entry.get('non_repetitive_over_repeated_acceptance')}"
    )


if __name__ == "__main__":
  main()
