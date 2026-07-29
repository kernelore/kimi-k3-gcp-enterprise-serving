import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_prefill_benchmark_kimi_k3 import extract_chunk_text as prefill_extract
from benchmarks.run_saturation_sweep_kimi_k3 import extract_chunk_text as sweep_extract


class TestSseParsing(unittest.TestCase):

  def test_completions_text_chunk(self):
    chunk = {"choices": [{"text": "hello"}]}
    self.assertEqual(prefill_extract(chunk), "hello")
    self.assertEqual(sweep_extract(chunk), "hello")

  def test_chat_completions_delta_chunk(self):
    chunk = {"choices": [{"delta": {"content": "world"}}]}
    self.assertEqual(prefill_extract(chunk), "world")
    self.assertEqual(sweep_extract(chunk), "world")

  def test_empty_or_invalid_chunk(self):
    self.assertIsNone(prefill_extract({}))
    self.assertIsNone(prefill_extract({"choices": []}))
    self.assertIsNone(prefill_extract({"choices": [{"delta": {}}]}))
    self.assertIsNone(sweep_extract({}))


if __name__ == "__main__":
  unittest.main()
