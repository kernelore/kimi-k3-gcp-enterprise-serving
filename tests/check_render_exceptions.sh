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

grep -rhoE '\$\{[A-Za-z_][A-Za-z0-9_-]*(:-[^}]*)?\}' terraform/manifests/generated/ \
  | sed 's/^\${//;s/:-.*//;s/}$//' | sort -u > /tmp/survivors.txt

if ! diff -u terraform/manifests/.render-exceptions /tmp/survivors.txt; then
  echo "ERROR: Unrendered variables in generated manifests do not match .render-exceptions allow-list!" >&2
  rm -f /tmp/survivors.txt
  exit 1
fi

rm -f /tmp/survivors.txt

if [ -f "terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml" ]; then
  if ! grep -q -- "--moe-runner-backend flashinfer_mxfp4 --decode-attention-backend flashmla --kv-cache-dtype fp8_e4m3" terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml; then
    echo "ERROR: SGLang performance flags not found on reachable runtime path in generated manifest!" >&2
    exit 1
  fi
fi

echo "Rendered manifest variable allow-list check passed cleanly."
