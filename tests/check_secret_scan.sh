#!/usr/bin/env bash
# Secret scan: single source of truth, invoked by both CI and the local
# remediation suite so the two can never drift apart.
#
# Scans git-tracked content and commit range. benchmarks/telemetry_sanitizer.py and
# tests/check_secret_scan.sh are the two file-level exclusions: they must carry these
# patterns as regex literals, and their behaviour is covered by unit/suite tests.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
EXCLUDE_SANITIZER=':(exclude)benchmarks/telemetry_sanitizer.py'
EXCLUDE_SCANNER=':(exclude)tests/check_secret_scan.sh'
EXCLUDE=( "${EXCLUDE_SANITIZER}" "${EXCLUDE_SCANNER}" )
FAIL=0

echo "Scanning git-tracked files for credentials, home paths and project IDs..."

if git grep -nE 'sk-[a-zA-Z0-9-]{16,}' -- "${EXCLUDE[@]}" \
   | grep -vE 'sk-kimi-k3-master-secret-key-change-me|sk-kimi-k3-master-change-me|sk-kimi-k3-test-dev-|sk-kimi-k3-quota-test-|sk-kimi-k3-m5-check1'; then
  echo "ERROR: Found potential exposed API keys in working tree!" >&2; FAIL=1
fi

if git grep -nE '/usr/local/google/home/[a-zA-Z0-9_-]|/home/[a-zA-Z0-9_-]' -- "${EXCLUDE[@]}" \
   | grep -vE '/home/(kubernetes|runner)/'; then
  echo "ERROR: Found local filesystem home paths in repository working tree!" >&2; FAIL=1
fi

if git grep -nE '[a-z][a-z0-9-]{3,26}-[0-9]{6}([^0-9]|$)' -- "${EXCLUDE[@]}"; then
  echo "ERROR: Found a GCP-project-ID-shaped literal in working tree!" >&2; FAIL=1
fi

if git grep -niE 'minitel|corp\.google\.com|google3|/users/[a-z]+' -- "${EXCLUDE[@]}" \
   | grep -vE 'google/cloud-sdk|cloudbuild\.googleapis\.com|googleapis\.com'; then
  echo "ERROR: Found internal Google/company references in working tree!" >&2; FAIL=1
fi

# Determine commit range for history scanning
if [ -n "${CI:-}" ] && git rev-parse --verify origin/main >/dev/null 2>&1; then
  RANGE="origin/main..HEAD"
elif git rev-parse --verify HEAD~5 >/dev/null 2>&1; then
  RANGE="HEAD~5..HEAD"
elif git rev-parse --verify HEAD >/dev/null 2>&1; then
  RANGE="HEAD^..HEAD" 2>/dev/null || RANGE="HEAD"
else
  RANGE=""
fi

if [ -n "${RANGE}" ] && git rev-parse --verify "${RANGE}" >/dev/null 2>&1; then
  echo "Scanning commit range ${RANGE} for leaked credentials and references..."
  if git log "${RANGE}" -p -- "${EXCLUDE[@]}" | grep -E 'sk-[a-zA-Z0-9-]{16,}' \
     | grep -vE 'sk-kimi-k3-master-secret-key-change-me|sk-kimi-k3-master-change-me|sk-kimi-k3-test-dev-|sk-kimi-k3-quota-test-|sk-kimi-k3-m5-check1'; then
    echo "ERROR: Found potential exposed API keys in commit range ${RANGE}!" >&2; FAIL=1
  fi
  if git log "${RANGE}" -p -- "${EXCLUDE[@]}" | grep -iE 'minitel|corp\.google\.com|google3|/users/[a-z]+' \
     | grep -vE 'google/cloud-sdk|cloudbuild\.googleapis\.com|googleapis\.com'; then
    echo "ERROR: Found internal Google/company references in commit range ${RANGE}!" >&2; FAIL=1
  fi
fi

if [ -n "${FORBIDDEN_PROJECT_ID:-}" ]; then
  if git grep -nF -e "${FORBIDDEN_PROJECT_ID}" -- "${EXCLUDE[@]}"; then
    echo "ERROR: Found hardcoded GCP project ID!" >&2; FAIL=1
  fi
else
  echo "NOTE: FORBIDDEN_PROJECT_ID unset; exact-value check skipped (shape check still enforced)."
fi

[ "$FAIL" -eq 0 ] || { echo "SECRET SCAN FAILED." >&2; exit 1; }
echo "Secret scan passed cleanly."
