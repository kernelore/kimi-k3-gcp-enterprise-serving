#!/usr/bin/env python3
"""Unit tests for SSE stream parsing logic across benchmark harnesses."""

import json
import unittest

def parse_sse_chunk(line_bytes: bytes) -> str:
    """Utility function replicating dual-path SSE chunk parsing."""
    line = line_bytes.decode('utf-8').strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        return ""
    try:
        data = json.loads(line[6:])
        choices = data.get('choices', [])
        if not choices:
            return ""
        choice = choices[0]
        # Dual-path extraction: handle completions ('text') and chat completions ('delta.content')
        if 'text' in choice and choice['text'] is not None:
            return choice['text']
        delta = choice.get('delta', {})
        if 'content' in delta and delta['content'] is not None:
            return delta['content']
    except Exception:
        pass
    return ""


class TestSSEParsing(unittest.TestCase):

    def test_legacy_completions_stream(self):
        line = b'data: {"choices": [{"text": "Hello world"}]}'
        self.assertEqual(parse_sse_chunk(line), "Hello world")

    def test_chat_completions_stream(self):
        line = b'data: {"choices": [{"delta": {"content": "Hello chat"}}]}'
        self.assertEqual(parse_sse_chunk(line), "Hello chat")

    def test_role_only_delta(self):
        line = b'data: {"choices": [{"delta": {"role": "assistant"}}]}'
        self.assertEqual(parse_sse_chunk(line), "")

    def test_empty_choices(self):
        line = b'data: {"choices": []}'
        self.assertEqual(parse_sse_chunk(line), "")

    def test_done_marker(self):
        line = b'data: [DONE]'
        self.assertEqual(parse_sse_chunk(line), "")


if __name__ == '__main__':
    unittest.main()
