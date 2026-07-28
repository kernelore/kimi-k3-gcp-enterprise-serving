#!/usr/bin/env bash
# Secret scan: single source of truth, invoked by both CI and local suite.
# Scans git-tracked content only. benchmarks/telemetry_sanitizer.py is excluded.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
EXCLUDE=':(exclude)benchmarks/telemetry_sanitizer.py'
FAIL=0

echo "Scanning git-tracked files for credentials, home paths and project IDs..."

if git grep -nE 'sk-[a-zA-Z0-9-]{16,}' -- "$EXCLUDE" 2>/dev/null \
   | grep -vE 'sk-kimi-k3-master-secret-key-change-me|sk-kimi-k3-quota-test-'; then
  echo "ERROR: Found potential exposed API keys!" >&2; FAIL=1
fi

if git grep -nE 'hf_[A-Za-z0-9]{34,}' -- "$EXCLUDE" 2>/dev/null; then
  echo "ERROR: Found potential exposed Hugging Face API token in git-tracked files!" >&2; FAIL=1
fi

if git grep -nE '/usr/local/google/home/[a-zA-Z0-9_-]|/home/[a-zA-Z0-9_-]' -- "$EXCLUDE" 2>/dev/null \
   | grep -vE '/home/(kubernetes|runner)/'; then
  echo "ERROR: Found local filesystem home paths in repository!" >&2; FAIL=1
fi

if git grep -nE '[a-z][a-z0-9-]{3,26}-[0-9]{6}([^0-9]|$)' -- "$EXCLUDE" 2>/dev/null; then
  echo "ERROR: Found a GCP-project-ID-shaped literal!" >&2; FAIL=1
fi

if [ -n "${FORBIDDEN_PROJECT_ID:-}" ]; then
  if git grep -nF -e "${FORBIDDEN_PROJECT_ID}" -- "$EXCLUDE" 2>/dev/null; then
    echo "ERROR: Found hardcoded GCP project ID!" >&2; FAIL=1
  fi
fi

# Fixed commit range scan calculation
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo 1)
  if [ "$COMMIT_COUNT" -gt 1 ]; then
    SCAN_DEPTH=$(( COMMIT_COUNT > 5 ? 5 : COMMIT_COUNT - 1 ))
    COMMIT_RANGE="HEAD~${SCAN_DEPTH}..HEAD"
    echo "Scanning commit history (${COMMIT_RANGE})..."
    if git log -p "${COMMIT_RANGE}" -- "$EXCLUDE" 2>/dev/null \
       | grep -nE 'sk-[a-zA-Z0-9-]{16,}' \
       | grep -vE 'sk-kimi-k3-master-secret-key-change-me|sk-kimi-k3-quota-test-'; then
      echo "ERROR: Found potential exposed API keys in commit history!" >&2; FAIL=1
    fi
  fi
fi

[ "$FAIL" -eq 0 ] || { echo "SECRET SCAN FAILED." >&2; exit 1; }
echo "Secret scan passed cleanly."
