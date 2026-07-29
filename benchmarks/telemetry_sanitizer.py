#!/usr/bin/env python3
import json
import re

SENSITIVE_KEYS = {"api_key", "raw", "master_key", "authorization", "hf_token", "token"}

def _sanitize_str(val):
    if not isinstance(val, str):
        return val
    val = re.sub(r"sk-[a-zA-Z0-9_-]{12,}", "REDACTED", val)
    val = re.sub(r"hf_[A-Za-z0-9]{32,}", "REDACTED", val)
    val = re.sub(r"docker\.pkg\.dev/[^/]+", "docker.pkg.dev/YOUR_PROJECT_ID", val)
    val = re.sub(r"(?:/usr/local/google/home/[^/]+/\.gemini/[^/\s]+/worktrees/[^/\s]+/[^/\s]+/|/home/[^/\s]+/|/Users/[^/\s]+/)", "", val)
    return val

def sanitize_telemetry(data, out_path=None):
    if not isinstance(data, dict):
        return data
    for k in list(data.keys()):
        if k in SENSITIVE_KEYS and isinstance(data[k], str):
            data[k] = "REDACTED"
            continue
        if k == "output" and isinstance(data["output"], str):
            p = out_path.replace("\\", "/") if out_path else data["output"].replace("\\", "/")
            if "benchmarks/results/" in p:
                data["output"] = "benchmarks/results/" + p.split("benchmarks/results/")[-1]
            else:
                data["output"] = _sanitize_str(p)
            continue
        val = data[k]
        if isinstance(val, dict):
            sanitize_telemetry(val, out_path)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    sanitize_telemetry(item, out_path)
                elif isinstance(item, str):
                    val[i] = _sanitize_str(item)
        elif isinstance(val, str):
            # Check if it's an embedded JSON string (like GLM config/metadata strings)
            if k in ["benchmark_config", "soak_config", "sweep_config", "prefill_config", "metadata", "raw"]:
                try:
                    m = json.loads(val)
                    if isinstance(m, dict):
                        sanitize_telemetry(m, out_path)
                        data[k] = json.dumps(m)
                        continue
                except Exception:
                    pass
            data[k] = _sanitize_str(val)
    return data
