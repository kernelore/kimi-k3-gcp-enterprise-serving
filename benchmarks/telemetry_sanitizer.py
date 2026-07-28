#!/usr/bin/env python3
"""
telemetry_sanitizer.py — Telemetry & Benchmark Result Data Redactor

Sanitizes benchmark telemetry data before output/publication.
Redacts API keys, master secrets, local filesystem paths, project IDs,
and container registry locations across nested dictionary and list structures.
"""

import json
import re

from typing import Optional

RE_LITELLM_KEY = re.compile(r"sk-[a-zA-Z0-9_-]{12,}")
RE_MASTER_KEY = re.compile(r"sk-kimi-k3-master-secret-key-change-me")
RE_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{32,}")
RE_HOME_PATH = re.compile(r"/(?:usr/local/google/home|home)/[^/\s\"']+(?:/\.gemini/jetski/worktrees/[^/\s\"']+)?[/\s\"']?")
RE_DOCKER_REG = re.compile(r"docker\.pkg\.dev/[^/\s\"']+")

SECRET_KEYS = {
    "api_key", "raw", "master_key", "authorization",
    "GATEWAY_MASTER_KEY", "secret", "password"
}

def sanitize_string(s: str, out_path: Optional[str] = None) -> str:
    if not isinstance(s, str):
        return s
    
    # 1. Redact API keys and secret tokens
    s = RE_LITELLM_KEY.sub("REDACTED", s)
    s = RE_MASTER_KEY.sub("REDACTED", s)
    s = RE_HF_TOKEN.sub("REDACTED", s)

    # 2. Redact container registry project paths
    s = RE_DOCKER_REG.sub("docker.pkg.dev/YOUR_PROJECT_ID", s)

    # 3. Unanchored path matching — redact local home paths while preserving relative path structure
    if "benchmarks/results/" in s:
        s = "benchmarks/results/" + s.split("benchmarks/results/")[-1]
    else:
        # Match home paths unanchored and strip out the leading user home directory
        s = re.sub(r"/(?:usr/local/google/home|home)/[^/\s\"']+/\.gemini/jetski/worktrees/[^/\s\"']+/([^/\s\"']+)/", r"\1/", s)
        s = re.sub(r"/(?:usr/local/google/home|home)/[^/\s\"']+/", "", s)

    return s

def sanitize_telemetry(data, out_path: Optional[str] = None):
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if k.lower() in {sk.lower() for sk in SECRET_KEYS}:
                sanitized[k] = "REDACTED"
            elif k == "output" and isinstance(v, str):
                p = out_path if out_path else v
                p_norm = p.replace("\\", "/")
                if "benchmarks/results/" in p_norm:
                    sanitized[k] = "benchmarks/results/" + p_norm.split("benchmarks/results/")[-1]
                else:
                    sanitized[k] = sanitize_string(p_norm, out_path)
            elif isinstance(v, (dict, list)):
                sanitized[k] = sanitize_telemetry(v, out_path)
            elif isinstance(v, str):
                # Check if string is serialized JSON dict/list
                if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
                    try:
                        parsed = json.loads(v)
                        sanitized_parsed = sanitize_telemetry(parsed, out_path)
                        sanitized[k] = json.dumps(sanitized_parsed)
                    except Exception:
                        sanitized[k] = sanitize_string(v, out_path)
                else:
                    sanitized[k] = sanitize_string(v, out_path)
            else:
                sanitized[k] = v
        return sanitized
    elif isinstance(data, list):
        return [sanitize_telemetry(item, out_path) for item in data]
    elif isinstance(data, str):
        return sanitize_string(data, out_path)
    else:
        return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            d = json.load(f)
        cleaned = sanitize_telemetry(d)
        print(json.dumps(cleaned, indent=2))
