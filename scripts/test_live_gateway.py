#!/usr/bin/env python3
# ==============================================================================
# test_live_gateway.py - Zero-Dependency Live Gateway Testing Script (Kimi K3)
# ==============================================================================
# Tests live chat completion routing through the Enterprise AI Gateway on port 4000
# using standard urllib without third-party dependencies (PEP 668 compliance).
# ==============================================================================

import json
import os
import sys
import urllib.request

def send_chat_completion():
    port = os.environ.get("GATEWAY_PORT", "4000")
    master_key = os.environ.get("GATEWAY_MASTER_KEY", "sk-kimi-k3-master-secret-key-change-me")
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "kimi-k3-2.8t-mxfp4",
        "messages": [{"role": "user", "content": "Hello Kimi K3, verify your MXFP4 routing."}],
        "temperature": 0.2,
        "max_tokens": 100
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            print(f"[OK] Live Chat completion output (Kimi K3 MXFP4):\n{content.strip()}")
            return 0
    except Exception as e:
        print(f"[FAIL] Request error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(send_chat_completion())
