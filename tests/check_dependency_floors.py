#!/usr/bin/env python3
"""
Dependency Floor Gate

Enforces that dependencies in Terraform, requirements.txt, CI workflows, and init containers
are at or above declared floor versions. Prevents accidental downgrades
from silent reverts or older file restores.
"""

import os
import re
import sys

FLOORS = {
    "terraform/main.tf": {
        "hashicorp/google": "7.41",
        "hashicorp/google-beta": "7.41",
    },
    "scripts/requirements.txt": {
        "google-cloud-storage": "3.13.0",
        "google-cloud-bigquery": "3.42.2",
    },
    ".github/workflows/ci.yml": {
        "actions/checkout": 7,
        "hashicorp/setup-terraform": 4,
        "actions/setup-python": 7,
    },
    "terraform/manifests/templates/00-local-nvme-raid.yaml.template": {
        "google/cloud-sdk": "500.0.0",
    },
    "terraform/manifests/templates/02-hydrate-weights-gcs.yaml.template": {
        "google/cloud-sdk": "500.0.0",
    },
    "scripts/03_deploy_workloads.sh": {
        "google/cloud-sdk": "500.0.0",
    },
}

def parse_version_tuple(v_str):
    m = re.search(r'([0-9]+(?:\.[0-9]+)*)', str(v_str))
    if not m:
        raise ValueError(f"Could not parse numeric version from '{v_str}'")
    parts = [int(x) for x in m.group(1).split('.')]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)

def check_workflow_content(content: str, dep: str, floor_val: int) -> bool:
    pattern = r'uses:\s*' + re.escape(dep) + r'@v([0-9]+)'
    matches = re.findall(pattern, content)
    if not matches:
        return False
    for actual_str in matches:
        if int(actual_str) < int(floor_val):
            return False
    return True

def check_floors():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    failed = False
    
    print("Checking dependency floor constraints...")
    
    for rel_path, constraints in FLOORS.items():
        file_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(file_path):
            print(f"ERROR: Target file not found: {rel_path}", file=sys.stderr)
            failed = True
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        for dep, floor_val in constraints.items():
            if rel_path == "terraform/main.tf":
                pattern = r'source\s*=\s*"' + re.escape(dep) + r'"\s*version\s*=\s*"[^0-9]*([0-9.]+)"'
                m = re.search(pattern, content, re.DOTALL)
                if not m:
                    print(f"ERROR: Could not find version pin for '{dep}' in {rel_path}", file=sys.stderr)
                    failed = True
                    continue
                actual_str = m.group(1)
                if parse_version_tuple(actual_str) < parse_version_tuple(floor_val):
                    print(f"ERROR: Floor violation in {rel_path}: '{dep}' is pinned to {actual_str}, which is below floor {floor_val}!", file=sys.stderr)
                    failed = True
                else:
                    print(f"    [OK] {rel_path}: {dep} ({actual_str} >= floor {floor_val})")
                    
            elif rel_path == "scripts/requirements.txt":
                pattern = r'^' + re.escape(dep) + r'[:<=>~]+([0-9]+(?:\.[0-9]+)*)'
                m = re.search(pattern, content, re.MULTILINE)
                if not m:
                    print(f"ERROR: Could not find version pin for '{dep}' in {rel_path}", file=sys.stderr)
                    failed = True
                    continue
                actual_str = m.group(1)
                if parse_version_tuple(actual_str) < parse_version_tuple(floor_val):
                    print(f"ERROR: Floor violation in {rel_path}: '{dep}' is pinned to {actual_str}, which is below floor {floor_val}!", file=sys.stderr)
                    failed = True
                else:
                    print(f"    [OK] {rel_path}: {dep} ({actual_str} >= floor {floor_val})")
                    
            elif rel_path == ".github/workflows/ci.yml":
                pattern = r'uses:\s*' + re.escape(dep) + r'@v([0-9]+)'
                matches = re.findall(pattern, content)
                if not matches:
                    print(f"ERROR: Could not find action pin for '{dep}' in {rel_path}", file=sys.stderr)
                    failed = True
                    continue
                if not check_workflow_content(content, dep, floor_val):
                    for actual_str in matches:
                        if int(actual_str) < int(floor_val):
                            print(f"ERROR: Floor violation in {rel_path}: '{dep}@v{actual_str}' is below floor v{floor_val}!", file=sys.stderr)
                    failed = True
                else:
                    for actual_str in matches:
                        print(f"    [OK] {rel_path}: {dep}@v{actual_str} (>= floor v{floor_val})")
            
            elif "google/cloud-sdk" in dep:
                pattern = r'google/cloud-sdk:([0-9]+(?:\.[0-9]+)*)-slim'
                matches = re.findall(pattern, content)
                if not matches:
                    print(f"ERROR: Could not find release pin for '{dep}' in {rel_path}", file=sys.stderr)
                    failed = True
                    continue
                for actual_str in matches:
                    if parse_version_tuple(actual_str) < parse_version_tuple(floor_val):
                        print(f"ERROR: Floor violation in {rel_path}: '{dep}:{actual_str}-slim' is below floor {floor_val}!", file=sys.stderr)
                        failed = True
                    else:
                        print(f"    [OK] {rel_path}: {dep}:{actual_str}-slim (>= floor {floor_val})")

    if failed:
        print("\nDEPENDENCY FLOOR GATE FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nALL DEPENDENCIES MEET OR EXCEED FLOOR CONSTRAINTS.")
        sys.exit(0)

if __name__ == "__main__":
    check_floors()
