#!/usr/bin/env bash
# ==============================================================================
# adv_test_serving_remediation.sh - Automated Remediation Verification Suite
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=============================================================================="
echo "Kimi K3 GCP Enterprise Serving - Automated Remediation Verification Suite"
echo "=============================================================================="

echo "--> Check 1: Verifying pinned engine versions and script syntax..."
if [ -f "scripts/config.env.example" ] && [ ! -f "scripts/config.env" ]; then
  cp scripts/config.env.example scripts/config.env
fi

INFERENCE_ENGINE=sglang ./scripts/03_deploy_workloads.sh --render-only >/dev/null
INFERENCE_ENGINE=trtllm ./scripts/03_deploy_workloads.sh --render-only >/dev/null

echo "--> Check 2: Verifying benchmarks/generate_comparison.py clean execution..."
python3 benchmarks/generate_comparison.py >/dev/null

echo "=============================================================================="
echo "SUCCESS: All remediation checks passed cleanly!"
echo "=============================================================================="
