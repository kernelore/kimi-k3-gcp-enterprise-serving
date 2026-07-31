#!/bin/bash
# shellcheck disable=SC1091
# ==============================================================================
# test_e2e_kimi_k3.sh - Master E2E Test Suite Runner for Kimi K3 Architecture
# ==============================================================================
# Opaque-box requirement-driven test runner executing Tiers 1-4 (73 test cases).
# ==============================================================================
set -euo pipefail

# Define project root and source helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

# Source test helpers
# shellcheck source=./test_helpers.sh
source "${SCRIPT_DIR}/test_helpers.sh"

# Source test modules
# shellcheck source=./test_cases_t1.sh
if [ -f "${SCRIPT_DIR}/test_cases_t1.sh" ]; then
  source "${SCRIPT_DIR}/test_cases_t1.sh"
fi
# shellcheck source=./test_cases_t2.sh
if [ -f "${SCRIPT_DIR}/test_cases_t2.sh" ]; then
  source "${SCRIPT_DIR}/test_cases_t2.sh"
fi
# shellcheck source=./test_cases_t3.sh
if [ -f "${SCRIPT_DIR}/test_cases_t3.sh" ]; then
  source "${SCRIPT_DIR}/test_cases_t3.sh"
fi
# shellcheck source=./test_cases_t4.sh
if [ -f "${SCRIPT_DIR}/test_cases_t4.sh" ]; then
  source "${SCRIPT_DIR}/test_cases_t4.sh"
fi
# shellcheck source=./test_cases_t5.sh
if [ -f "${SCRIPT_DIR}/test_cases_t5.sh" ]; then
  source "${SCRIPT_DIR}/test_cases_t5.sh"
fi

# Tier 1/3/4/5 cases re-render terraform/manifests/generated/ under forced engines and
# synthetic values (INFERENCE_ENGINE=trtllm, PROJECT_ID=test-proj, CLUSTER_NAME=test-cluster).
# Whatever ran last would otherwise be left on disk, so a later `kubectl apply -f` or manual
# diff against the generated tree would read manifests built for the wrong engine and a
# nonexistent project. Re-render for the configured engine before reporting the summary.
restore_configured_render() {
  if [ -f "${PROJECT_ROOT}/scripts/config.env" ]; then
    (cd "${PROJECT_ROOT}" && ./scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1) || true
  fi
}

# Preserve the real exit code across the restore: on_exit reads $? and must not see the
# render's status. `(exit N)` re-arms $? for the call that follows.
trap 'E2E_RC=$?; restore_configured_render; (exit ${E2E_RC}); on_exit' EXIT

# Default tier selections
RUN_TIER_1=0
RUN_TIER_2=0
RUN_TIER_3=0
RUN_TIER_4=0
RUN_TIER_5=0

if [ $# -eq 0 ]; then
  RUN_TIER_1=1
  RUN_TIER_2=1
  RUN_TIER_3=1
  RUN_TIER_4=1
  RUN_TIER_5=1
else
  while [ $# -gt 0 ]; do
    case "$1" in
      --tier)
        shift
        if [ "$1" = "1" ]; then RUN_TIER_1=1; fi
        if [ "$1" = "2" ]; then RUN_TIER_2=1; fi
        if [ "$1" = "3" ]; then RUN_TIER_3=1; fi
        if [ "$1" = "4" ]; then RUN_TIER_4=1; fi
        if [ "$1" = "5" ]; then RUN_TIER_5=1; fi
        shift
        ;;
      --all)
        RUN_TIER_1=1
        RUN_TIER_2=1
        RUN_TIER_3=1
        RUN_TIER_4=1
        RUN_TIER_5=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
  done
fi

log_info "=============================================================================="
log_info "Kimi K3 Sovereign Enterprise Architecture - Master E2E Test Suite"
log_info "=============================================================================="
log_info "Selected Tiers: T1=${RUN_TIER_1} | T2=${RUN_TIER_2} | T3=${RUN_TIER_3} | T4=${RUN_TIER_4} | T5=${RUN_TIER_5} | Dry-Run=${DRY_RUN}"
log_info "=============================================================================="

if [ "${RUN_TIER_1}" -eq 1 ]; then
  log_info "Starting Tier 1: Feature Coverage (Happy Path / Conformance)..."
  run_tier_1_tests
fi

if [ "${RUN_TIER_2}" -eq 1 ]; then
  log_info "Starting Tier 2: Boundary & Corner Cases (Negative / Error Handling)..."
  run_tier_2_tests
fi

if [ "${RUN_TIER_3}" -eq 1 ]; then
  log_info "Starting Tier 3: Cross-Feature Combinations (Pairwise Combinatorial)..."
  run_tier_3_tests
fi

if [ "${RUN_TIER_4}" -eq 1 ]; then
  log_info "Starting Tier 4: Real-World Application Scenarios (Workload Simulation)..."
  run_tier_4_tests
fi

if [ "${RUN_TIER_5}" -eq 1 ]; then
  log_info "Starting Tier 5: Adversarial White-Box Coverage Hardening Suite..."
  run_tier_5_tests
fi

log_info "All selected test tiers executed successfully!"
exit 0
