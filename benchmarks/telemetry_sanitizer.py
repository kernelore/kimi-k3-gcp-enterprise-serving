#!/usr/bin/env python3
import json
import re

def _sanitize_str(val):
    if not isinstance(val, str):
        return val
    val = re.sub(r"sk-kimi-k3-[a-fA-F0-9]{16,}", "REDACTED", val)
    val = re.sub(r"docker\.pkg\.dev/[^/]+", "docker.pkg.dev/YOUR_PROJECT_ID", val)
    val = re.sub(r"^(.*?)(/home/[^/]+/|/usr/local/google/home/[^/]+/.gemini/jetski/worktrees/[^/]+/[^/]+/)", "", val)
    return val

def sanitize_telemetry(data, out_path=None):
    if not isinstance(data, dict):
        return data
    if "api_key" in data:
        data["api_key"] = "REDACTED"
    if "output" in data and isinstance(data["output"], str):
        p = out_path.replace("\\", "/") if out_path else data["output"].replace("\\", "/")
        if "benchmarks/results/" in p:
            data["output"] = "benchmarks/results/" + p.split("benchmarks/results/")[-1]
        else:
            data["output"] = _sanitize_str(p)

    for k in list(data.keys()):
        if k == "api_key" or k == "output":
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
