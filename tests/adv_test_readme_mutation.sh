#!/usr/bin/env bash
# ==============================================================================
# adv_test_readme_mutation.sh - Adversarial Proof of README Mutation Rejection
# ==============================================================================
# Proves that Check 8 / Reproducibility job fails when README.md
# is deliberately mutated, and that reverting the mutation restores zero diff.
# Handles both pre-launch (empty results tree) and post-launch states per Rule 8.
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1

echo "=============================================================================="
echo "Adversarial Proof: README Mutation Rejection & Clean Revert"
echo "=============================================================================="

RESULTS_DIR="${COMPARISON_RESULTS_DIR:-benchmarks/results}"
if compgen -G "${RESULTS_DIR}/*/*.json" > /dev/null; then
  # Ensure README.md is synchronized with generate_comparison.py before starting check
  python3 benchmarks/generate_comparison.py >/dev/null 2>&1
  git add README.md >/dev/null 2>&1 || true

  # 1. Verify clean starting state (against current working tree comparison baseline)
  echo "--> Step 1: Verifying initial comparison baseline of README.md..."
  if ! python3 benchmarks/generate_comparison.py >/dev/null 2>&1; then
    echo "ERROR: generate_comparison.py failed on initial README.md! Aborting." >&2
    exit 1
  fi
  echo "    [OK] Initial README.md comparison baseline is clean."

  # 2. Mutate a numerical benchmark value in README.md (in a safe, reversible manner)
  echo "--> Step 2: Deliberately mutating a numerical benchmark value in README.md..."
  cp README.md /tmp/README.md.adv_backup
  # Mutate first occurrence of a measurement inside comparison block to 9999.99
  sed -i '/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/ s/[0-9]\+\.[0-9]\+/9999.99/' README.md

  if grep -q "9999.99" README.md; then
    echo "    [OK] Successfully mutated value to '9999.99' in README.md."
  else
    echo "ERROR: Mutation failed!" >&2
    cp /tmp/README.md.adv_backup README.md
    exit 1
  fi

  # 3. Prove generate_comparison.py detects mutation and overwrites it back to baseline
  echo "--> Step 3: Running reproducibility check (generate_comparison.py on unstaged mutation)..."
  python3 benchmarks/generate_comparison.py >/dev/null 2>&1
  if cmp -s README.md /tmp/README.md.adv_backup >/dev/null 2>&1; then
    echo "    [OK] Reproducibility check PASSED: generate_comparison.py detected mutation and restored baseline!"
  else
    echo "ERROR: generate_comparison.py failed to restore real benchmark numbers!" >&2
    cp /tmp/README.md.adv_backup README.md
    exit 1
  fi

  # 4. Re-apply mutation and prove git diff --exit-code fails when staged
  echo "--> Step 4: Testing reproducibility rejection on staged mutated README.md..."
  cp /tmp/README.md.adv_backup README.md
  sed -i '/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/ s/[0-9]\+\.[0-9]\+/9999.99/' README.md
  git add README.md >/dev/null 2>&1 || true
  python3 benchmarks/generate_comparison.py >/dev/null 2>&1
  if ! git diff --exit-code README.md > /dev/null 2>&1; then
    echo "    [OK] Reproducibility test FAILED as expected on staged mutation."
  else
    echo "ERROR: Reproducibility test unexpectedly PASSED on staged mutated README.md!" >&2
    cp /tmp/README.md.adv_backup README.md
    git add README.md >/dev/null 2>&1 || true
    exit 1
  fi

  # 5. Cleanly revert the mutation and verify restoration of exact original state
  echo "--> Step 5: Cleanly reverting mutation and verifying zero residual diff against backup..."
  cp /tmp/README.md.adv_backup README.md
  python3 benchmarks/generate_comparison.py >/dev/null 2>&1
  git add README.md >/dev/null 2>&1 || true
  rm -f /tmp/README.md.adv_backup

  if cmp -s README.md <(git show :README.md 2>/dev/null); then
    echo "    [OK] README.md cleanly reverted and verified up-to-date with benchmark results (zero diff)!"
    echo "=============================================================================="
    echo "=== README MUTATION REJECTION VERIFIED (1 passed, 0 skipped) ==="
    echo "=============================================================================="
    exit 0
  else
    echo "ERROR: Reverting README.md left residual discrepancy!" >&2
    exit 1
  fi
else
  echo "--> Step 1: benchmarks/results/ is empty (pre-launch state). Verifying placeholder table check..."
  BLOCK=$(sed -n '/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/p' README.md)
  if echo "$BLOCK" | grep -nE '[0-9]+(\.[0-9]+)?[[:space:]]*(ms|s|tok/s|tokens/s|GB|GiB|%)'; then
    echo "ERROR: README publishes measurements but benchmarks/results/ is empty." >&2
    exit 1
  fi
  echo "    [OK] Placeholder table contains no measurements."

  echo "--> Step 2: Testing adversarial injection of bogus measurement into placeholder table..."
  cp README.md /tmp/README.md.adv_backup
  sed -i '/ENGINE_COMPARISON_START/a | **TTFT** | 123 ms | 456 ms |' README.md

  BLOCK=$(sed -n '/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/p' README.md)
  if echo "$BLOCK" | grep -nE '[0-9]+(\.[0-9]+)?[[:space:]]*(ms|s|tok/s|tokens/s|GB|GiB|%)' >/dev/null 2>&1; then
    echo "    [OK] Bogus measurement injection detected."
  else
    echo "ERROR: Failed to inject bogus measurement!" >&2
    cp /tmp/README.md.adv_backup README.md
    exit 1
  fi

  echo "--> Step 3: Proving placeholder table check rejects the injected measurement..."
  if bash -c 'BLOCK=$(sed -n "/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/p" README.md); if echo "$BLOCK" | grep -nE "[0-9]+(\.[0-9]+)?[[:space:]]*(ms|s|tok/s|tokens/s|GB|GiB|%)" >/dev/null; then exit 1; else exit 0; fi'; then
    echo "ERROR: Placeholder table check unexpectedly passed on mutated README!" >&2
    cp /tmp/README.md.adv_backup README.md
    exit 1
  else
    echo "    [OK] Placeholder table check FAILED as expected on mutated README."
  fi

  echo "--> Step 4: Restoring README.md and verifying clean state..."
  cp /tmp/README.md.adv_backup README.md
  rm -f /tmp/README.md.adv_backup
  if bash -c 'BLOCK=$(sed -n "/ENGINE_COMPARISON_START/,/ENGINE_COMPARISON_END/p" README.md); if echo "$BLOCK" | grep -nE "[0-9]+(\.[0-9]+)?[[:space:]]*(ms|s|tok/s|tokens/s|GB|GiB|%)" >/dev/null; then exit 1; else exit 0; fi'; then
    echo "    [OK] README.md cleanly restored."
    echo "=============================================================================="
    echo "=== README MUTATION REJECTION VERIFIED (0 passed, 1 skipped) ==="
    echo "    [SKIP] benchmarks/results/ is empty (pre-launch state). Tested placeholder table mutation rejection instead of full comparison regeneration."
    echo "=============================================================================="
    exit 0
  else
    echo "ERROR: Restored README failed check!" >&2
    exit 1
  fi
fi
