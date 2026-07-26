"""Shared helper to extract and normalize engine versions from Dockerfiles."""

from pathlib import Path

def normalize_version(ver: str) -> str:
    if not ver:
        return ""
    ver = ver.strip()
    if ver.startswith("v") or ver.startswith("V"):
        ver = ver[1:]
    if ver.endswith("-cu130"):
        ver = ver[:-6]
    if ver.endswith("-py3"):
        ver = ver[:-4]
    return ver

def get_engine_version(engine: str, root: Path | None = None) -> str:
    if root is None:
        root = Path(__file__).resolve().parent.parent.parent
    if engine == "sglang":
        dockerfile = root / "docker" / "Dockerfile.sglang"
    elif engine == "trtllm":
        dockerfile = root / "docker" / "Dockerfile"
    else:
        raise ValueError(f"Unknown engine: '{engine}'")
    
    if not dockerfile.is_file():
        raise FileNotFoundError(f"Dockerfile not found at {dockerfile}")

    content = dockerfile.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("FROM "):
            parts = line.split(":")
            if len(parts) >= 2:
                raw_tag = parts[1].split()[0]
                return normalize_version(raw_tag)
    raise ValueError(f"Could not extract FROM tag in {dockerfile}")
