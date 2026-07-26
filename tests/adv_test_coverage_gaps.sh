#!/bin/bash
# shellcheck disable=SC1091
# ==============================================================================
# adv_test_coverage_gaps.sh - Wrapper for Tier 5 Adversarial Test Suite
# ==============================================================================
# Restored to eliminate VCS missing file anomaly and execute test_cases_t5.sh.
# ==============================================================================
set -euo pipefail

# Define project root and source helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

# shellcheck source=./test_helpers.sh
source "${SCRIPT_DIR}/test_helpers.sh"
# shellcheck source=./test_cases_t5.sh
source "${SCRIPT_DIR}/test_cases_t5.sh"

trap on_exit EXIT

log_info "Starting Standalone Tier 5 Adversarial Verification Suite..."
run_tier_5_tests
log_info "Standalone Tier 5 verification completed successfully."
exit 0
