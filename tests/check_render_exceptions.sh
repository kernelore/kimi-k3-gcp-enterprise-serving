#!/usr/bin/env bash
# Verification check for unrendered template variables and SGLang performance flags.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATED_DIR="${PROJECT_ROOT}/terraform/manifests/generated"

if [ ! -d "${GENERATED_DIR}" ]; then
  echo "ERROR: Generated manifests directory ${GENERATED_DIR} does not exist." >&2
  exit 1
fi

echo "Checking for unrendered variables in ${GENERATED_DIR}..."

# Check for surviving unrendered template variables, excluding known container runtime bash variables
# Allowed runtime bash vars: IP0, IP1, NODE_RANK, LEADER_HOST, LEADER_ADDR, LOG_FILE, BUSBW, POD_INDEX, HOSTNAME, EXTRA_ARGS, bw, count, sum, v, NCCL_XARGS
SURVIVING_VARS=$(grep -rn -E '\$\{([A-Za-z0-9_]+)(:-[^}]*)?\}' "${GENERATED_DIR}"/*.yaml | \
  grep -v -E '\$\{(IP[01]|NODE_RANK|LEADER_HOST|LEADER_ADDR|LOG_FILE|BUSBW|POD_INDEX|HOSTNAME|EXTRA_ARGS|bw|count|sum|v|NCCL_XARGS)(:-[^}]*)?\}' || true)

if [ -n "${SURVIVING_VARS}" ]; then
  echo "ERROR: Found surviving unrendered template variables in generated manifests:" >&2
  echo "${SURVIVING_VARS}" >&2
  exit 1
fi

echo "Checking SGLang performance flags in 09-kimi-k3-sglang-mpi.yaml..."
SGLANG_MANIFEST="${GENERATED_DIR}/09-kimi-k3-sglang-mpi.yaml"

if [ -f "${SGLANG_MANIFEST}" ]; then
  if ! grep -q -- "--moe-runner-backend" "${SGLANG_MANIFEST}"; then
    echo "ERROR: --moe-runner-backend missing from executable sglang server invocation in ${SGLANG_MANIFEST}" >&2
    exit 1
  fi
  if ! grep -q -- "--decode-attention-backend" "${SGLANG_MANIFEST}"; then
    echo "ERROR: --decode-attention-backend missing from executable sglang server invocation in ${SGLANG_MANIFEST}" >&2
    exit 1
  fi
  if ! grep -q -- "--kv-cache-dtype" "${SGLANG_MANIFEST}"; then
    echo "ERROR: --kv-cache-dtype missing from executable sglang server invocation in ${SGLANG_MANIFEST}" >&2
    exit 1
  fi
fi

echo "[OK] Render exception check passed cleanly."
