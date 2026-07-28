#!/bin/bash
# ==============================================================================
# test_helpers.sh - Common Logging & Assertion Library for Kimi K3 E2E Tests
# ==============================================================================
set -euo pipefail

# Color codes for terminal logging
if [ -t 1 ] && [ "${NO_COLOR:-}" != "1" ]; then
  GREEN=$'\033[0;32m'
  RED=$'\033[0;31m'
  YELLOW=$'\033[0;33m'
  BLUE=$'\033[0;34m'
  CYAN=$'\033[0;36m'
  RESET=$'\033[0m'
else
  GREEN="" RED="" YELLOW="" BLUE="" CYAN="" RESET=""
fi

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
DRY_RUN=0
CURRENT_TEST_ID=""

log_timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  echo -e "${CYAN}[INFO]  $(log_timestamp)${RESET} $*"
}

log_pass() {
  echo -e "${GREEN}[PASS]  $(log_timestamp)${RESET} $*"
}

log_fail() {
  echo -e "${RED}[FAIL]  $(log_timestamp)${RESET} $*" >&2
}

log_warn() {
  echo -e "${YELLOW}[WARN]  $(log_timestamp)${RESET} $*"
}

log_skip() {
  echo -e "${BLUE}[SKIP]  $(log_timestamp)${RESET} $*"
}

on_exit() {
  local ec=$?
  local summary_file="${PROJECT_ROOT}/TEST_SUMMARY.log"
  {
    echo "=============================================================================="
    echo "Kimi K3 Sovereign Enterprise E2E Test Suite Summary"
    echo "Timestamp: $(log_timestamp)"
    echo "Exit Code: ${ec}"
    echo "Total Executed: ${TESTS_RUN} | Passed: ${TESTS_PASSED} | Failed: ${TESTS_FAILED} | Skipped: ${TESTS_SKIPPED}"
    if [ ${ec} -eq 0 ]; then
      echo "Status: SUCCESS (All tests passed without compliance violations)"
    else
      echo "Status: FAILURE (Test suite aborted on failure or violation)"
    fi
    echo "=============================================================================="
  } | tee "${summary_file}"
  if [ ${ec} -ne 0 ]; then
    log_fail "Test suite terminated with exit code ${ec}. See ${summary_file} for details."
  fi
}

# Assertions
assert_true() {
  local cond="$1"
  local msg="${2:-Assertion failed: expected true condition}"
  if ! eval "${cond}"; then
    log_fail "[${CURRENT_TEST_ID}] ${msg}"
    return 1
  fi
}

assert_equals() {
  local expected="$1"
  local actual="$2"
  local msg="${3:-Assertion failed: expected \"${expected}\", got \"${actual}\"}"
  if [ "${expected}" != "${actual}" ]; then
    log_fail "[${CURRENT_TEST_ID}] ${msg}"
    return 1
  fi
}

assert_file_exists() {
  local filepath="$1"
  local msg="${2:-Assertion failed: file \"${filepath}\" does not exist}"
  if [ ! -e "${filepath}" ]; then
    log_fail "[${CURRENT_TEST_ID}] ${msg}"
    return 1
  fi
}

assert_match() {
  local regex="$1"
  local target="$2"
  local msg="${3:-Assertion failed: regex \"${regex}\" did not match target}"
  if [ -d "${target}" ]; then
    if ! grep -rE --exclude-dir=".*" -q -- "${regex}" "${target}" 2>/dev/null; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in directory ${target}"
      return 1
    fi
  elif [ -f "${target}" ]; then
    if ! grep -E -q -- "${regex}" "${target}"; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in file ${target}"
      return 1
    fi
  else
    if ! echo "${target}" | grep -E -q -- "${regex}"; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in string"
      return 1
    fi
  fi
}

assert_no_match() {
  local regex="$1"
  local target="$2"
  local msg="${3:-Assertion failed: prohibited regex \"${regex}\" matched target}"
  if [ -d "${target}" ]; then
    if grep -rE --exclude-dir=".*" -q -- "${regex}" "${target}" 2>/dev/null; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in directory ${target}"
      return 1
    fi
  elif [ -f "${target}" ]; then
    if grep -E -q -- "${regex}" "${target}"; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in file ${target}"
      return 1
    fi
  else
    if echo "${target}" | grep -E -q -- "${regex}"; then
      log_fail "[${CURRENT_TEST_ID}] ${msg} in string"
      return 1
    fi
  fi
}

assert_cmd_success() {
  local cmd="$1"
  local msg="${2:-Assertion failed: command \"${cmd}\" did not succeed}"
  if ! eval "${cmd}" >/dev/null 2>&1; then
    log_fail "[${CURRENT_TEST_ID}] ${msg}"
    return 1
  fi
}

assert_cmd_fails() {
  local cmd="$1"
  local msg="${2:-Assertion failed: command \"${cmd}\" succeeded but expected failure}"
  if eval "${cmd}" >/dev/null 2>&1; then
    log_fail "[${CURRENT_TEST_ID}] ${msg}"
    return 1
  fi
}

run_test_case() {
  local test_id="$1"
  local func_name="$2"
  CURRENT_TEST_ID="${test_id}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    log_skip "(Dry Run) ${test_id}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    return 0
  fi
  log_info "Executing ${test_id}..."
  TESTS_RUN=$((TESTS_RUN + 1))
  
  set +e
  ( set -e; eval "${func_name}" )
  local ec=$?
  set -e
  
  if [ ${ec} -eq 0 ]; then
    log_pass "${test_id}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    log_fail "${test_id} failed!"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
}
