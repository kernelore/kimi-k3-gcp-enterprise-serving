#!/usr/bin/env python3
"""Library for normalizing engine image versions and tags."""

import re

def normalize_engine_version(image_ref: str) -> str:
    if not image_ref:
        return "unknown"
    if "@" in image_ref:
        image_ref = image_ref.split("@")[0]
    if ":" in image_ref:
        return image_ref.split(":")[-1]
    return "latest"
