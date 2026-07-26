#!/usr/bin/env bash
# Verifies that unrendered ${...} variables in generated manifests match exactly
# the allow-list in terraform/manifests/.render-exceptions.
# Shared implementation called by both CI (.github/workflows/ci.yml) and the local
# remediation suite (tests/adv_test_serving_remediation.sh Check 12).
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

if [ ! -d "terraform/manifests/generated" ] || [ -z "$(ls -A terraform/manifests/generated/*.yaml 2>/dev/null)" ]; then
  echo "ERROR: No generated manifests found in terraform/manifests/generated/. Must run ./scripts/03_deploy_workloads.sh --render-only first." >&2
  exit 1
fi

grep -rhoE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' terraform/manifests/generated/ \
  | tr -d '{}$' | sort -u > /tmp/survivors.txt

if ! diff -u terraform/manifests/.render-exceptions /tmp/survivors.txt; then
  echo "ERROR: Unrendered variables in generated manifests do not match .render-exceptions allow-list!" >&2
  rm -f /tmp/survivors.txt
  exit 1
fi

rm -f /tmp/survivors.txt
echo "Rendered manifest variable allow-list check passed cleanly."
