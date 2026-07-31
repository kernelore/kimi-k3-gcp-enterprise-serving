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

SGLANG_MANIFEST="terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
if [ -f "${SGLANG_MANIFEST}" ]; then
  if ! grep -q -- "--moe-runner-backend flashinfer_mxfp4 --kv-cache-dtype fp8_e4m3" "${SGLANG_MANIFEST}"; then
    echo "ERROR: SGLang performance flags not found on reachable runtime path in generated manifest!" >&2
    exit 1
  fi

  # Attention-backend drift guard. Kimi K3 is a KDA/MLA hybrid on Blackwell:
  #   flashmla     - Hopper-only kernel
  #   trtllm_mla   - not the dispatch path SGLang selects for this family
  #   trtllm_mha   - the SM100 default for MHA architectures; K3 is AttentionArch.MLA
  #   --attention-backend  - sets the base for BOTH phases, silently pinning decode
  # Decode must stay unpinned so SGLang's own SM100 dispatch (flashinfer) applies.
  # Comment lines are exempt - they document precisely why these are forbidden.
  grep -vE '^[[:space:]]*#' "${SGLANG_MANIFEST}" > /tmp/sglang_code.txt
  for FORBIDDEN in "flashmla" "trtllm_mla" "trtllm_mha" "--attention-backend " "--disable-cuda-graph" "--cuda-graph-backend-decode"; do
    if grep -q -- "${FORBIDDEN}" /tmp/sglang_code.txt; then
      echo "ERROR: forbidden SGLang flag '${FORBIDDEN}' present in generated manifest!" >&2
      rm -f /tmp/sglang_code.txt
      exit 1
    fi
  done
  rm -f /tmp/sglang_code.txt

  # A '#' comment on a backslash-continued line silently truncates argv: the shell joins
  # the lines, the '#' starts a comment, and every remaining flag is discarded. This has
  # regressed once already inside the sglang.launch_server invocation.
  if awk '
    prev ~ /\\[[:space:]]*$/ && $0 ~ /^[[:space:]]*#/ { print NR; found=1 }
    { prev = $0 }
    END { exit(found ? 0 : 1) }
  ' "${SGLANG_MANIFEST}" > /tmp/contbug.txt; then
    echo "ERROR: comment on a backslash-continued line in generated manifest (lines: $(tr '\n' ' ' < /tmp/contbug.txt))- this truncates the command's arguments!" >&2
    rm -f /tmp/contbug.txt
    exit 1
  fi
  rm -f /tmp/contbug.txt
fi

echo "Rendered manifest variable allow-list check passed cleanly."
