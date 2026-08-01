#!/usr/bin/env bash
# ==============================================================================
# adv_test_serving_remediation.sh - Automated Remediation Verification Suite
# ==============================================================================
# Verifies zero legacy reference matches, pinned container
# images and adapter manifests, schema-validated Kubernetes manifests, clean
# benchmark comparison generation, and adversarial provenance gate rejection.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Several checks below re-render terraform/manifests/generated/ under a forced
# INFERENCE_ENGINE, and the engine loops end on trtllm. Left that way, the generated
# 09-kimi-k3-sglang-mpi.yaml carries the TRT-LLM container image, and anything that
# subsequently reads the generated tree -- a manual `kubectl apply -f`, a diff, a
# hand-built patch -- picks up a manifest that was never meant to be deployed.
# Always hand the tree back rendered for the engine configured in scripts/config.env.
restore_configured_render() {
  local rc=$?
  if [ -f "${PROJECT_ROOT}/scripts/config.env" ]; then
    (cd "${PROJECT_ROOT}" && ./scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1) || true
  fi
  return "${rc}"
}
trap restore_configured_render EXIT

echo "=============================================================================="
echo "Kimi K3 GCP Enterprise Serving - Automated Remediation Verification Suite"
echo "=============================================================================="

PASSED=0
SKIPPED=0

# ------------------------------------------------------------------------------
# Check 1: Zero matches for glm52, glm-5, vllm (case-insensitive)
# ------------------------------------------------------------------------------
echo "--> Check 1: Verifying zero matches for glm52, glm-5, vllm across repository..."
if grep -rnwiE 'glm52|glm-5|vllm' --exclude-dir={.agents,.git,.venv,.terraform,__pycache__,tests} . ; then
  echo "ERROR: Check 1 failed: Found forbidden legacy model/engine terms in repository!" >&2
  exit 1
fi
echo "    [OK] Check 1 passed: Zero legacy model/engine matches found."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 2: Pinned image tags across all templates (no unpinned :slim or :latest)
# ------------------------------------------------------------------------------
echo "--> Check 2: Verifying terraform/manifests/templates/*.template use pinned image tags..."
if grep -rnE 'image:[[:space:]]*[^[:space:]]+(:slim|:latest)([[:space:]]|$|"'\'')' terraform/manifests/templates/ \
   | grep -vE 'python:[0-9]+\.[0-9]+-slim|\$\{SERVING_IMAGE\}'; then
  echo "ERROR: Check 2 failed: Found unpinned ':slim' or ':latest' tag in templates!" >&2
  exit 1
fi
echo "    [OK] Check 2 passed: All templates use pinned image tags."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 3: Pinned engine Dockerfiles and deploy script references
# ------------------------------------------------------------------------------
echo "--> Check 3: Verifying pinned engine Dockerfiles and sdk pin in scripts/03_deploy_workloads.sh..."
# shellcheck source=tests/lib/engine_versions.sh disable=SC1091
source tests/lib/engine_versions.sh
for eng in sglang trtllm; do
  ver="$(get_engine_version "${eng}")"
  dockerfile="docker/Dockerfile.${eng}"
  if [ "${eng}" = "trtllm" ]; then dockerfile="docker/Dockerfile"; fi
  from_line=$(grep -E '^FROM[[:space:]]+' "${dockerfile}" | head -n 1)
  has_digest=0
  if echo "${from_line}" | grep -qE '@sha256:[a-fA-F0-9]{64}'; then
    has_digest=1
  fi
  if [ -z "${ver}" ] || [[ "${ver}" =~ ^(latest|slim|main|master|dev|nightly)$ ]]; then
    echo "ERROR: Check 3 failed: Dockerfile for ${eng} is unpinned or uses forbidden tag (got '${ver}')!" >&2
    exit 1
  fi
  is_semver=0
  if [[ "${ver}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)? ]]; then
    is_semver=1
  fi
  if [ "${is_semver}" = "0" ] && [ "${has_digest}" = "0" ]; then
    echo "ERROR: Check 3 failed: Dockerfile for ${eng} is non-semver ('${ver}') and NOT digest-pinned!" >&2
    exit 1
  fi
done
if grep -rnE '^FROM[[:space:]]+[^[:space:]]+(:latest|:slim)([[:space:]]|$)' docker/; then
  echo "ERROR: Check 3 failed: Found unpinned ':latest' or ':slim' base image in docker/!" >&2
  exit 1
fi
if ! grep -q "500.0.0-slim" scripts/03_deploy_workloads.sh || ! grep -q "sglang-blackwell" scripts/03_deploy_workloads.sh || ! grep -q "trtllm-blackwell" scripts/03_deploy_workloads.sh; then
  echo "ERROR: Check 3 failed: SDK and engine image names not properly configured in scripts/03_deploy_workloads.sh!" >&2
  exit 1
fi
echo "    [OK] Check 3 passed: Dockerfile base images and deploy script references are properly pinned."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 4: Kubeconform validation of rendered YAML manifests
# ------------------------------------------------------------------------------
echo "--> Check 4: Verifying rendered YAML manifests with kubeconform..."
if [ ! -f "scripts/config.env" ] && [ -f "scripts/config.env.example" ]; then
  cp scripts/config.env.example scripts/config.env
  sed -i 's|export PROJECT_ID="YOUR_PROJECT_ID"|export PROJECT_ID="ci-test"|' scripts/config.env
  CLEANUP_CONFIG_ENV=true
else
  CLEANUP_CONFIG_ENV=false
fi
for eng in sglang trtllm; do
  INFERENCE_ENGINE=$eng ./scripts/03_deploy_workloads.sh --render-only >/dev/null
  kubeconform -strict -summary -schema-location default -schema-location 'terraform/manifests/schemas/{{ .ResourceKind }}_{{ .ResourceAPIVersion }}.json' -skip GKENetworkParamSet,Network terraform/manifests/generated/*.yaml >/dev/null
done
if [ "${CLEANUP_CONFIG_ENV}" = "true" ]; then
  rm -f scripts/config.env
fi
echo "    [OK] Check 4 passed: All rendered manifests passed kubeconform schema validation."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 5: Clean execution of benchmarks/generate_comparison.py & zero diff
# ------------------------------------------------------------------------------
echo "--> Check 5: Verifying benchmarks/generate_comparison.py cleanly executes with zero README.md diff..."
if compgen -G "benchmarks/results/*/*.json" > /dev/null; then
  python3 benchmarks/generate_comparison.py >/dev/null
  if ! git diff --exit-code README.md >/dev/null; then
    echo "ERROR: Check 5 failed: benchmarks/generate_comparison.py modified README.md!" >&2
    git diff README.md >&2
    exit 1
  fi
  echo "    [OK] Check 5 passed: Benchmark comparison generated cleanly with zero diff against README.md."
  PASSED=$((PASSED + 1))
else
  echo "    [SKIP] Check 5 skipped: benchmarks/results/ is empty (pre-launch state)."
  SKIPPED=$((SKIPPED + 1))
fi

# ------------------------------------------------------------------------------
# Check 6: Adversarial Proof of Provenance Gate
# ------------------------------------------------------------------------------
echo "--> Check 6: Adversarial Proof of Provenance Gate (testing invalid inputs)..."
if compgen -G "benchmarks/results/*/*.json" > /dev/null; then
  python3 -c '
import copy
import sys
from pathlib import Path

sys.path.insert(0, "benchmarks")
from generate_comparison import validate_provenance, load_json

SUITES = ["standard", "massive", "soak", "saturation", "prefill"]
root = Path("benchmarks/results")


def load_engine(name):
    """Load an engine only when all five of its suites are present.

    validate_provenance takes None for an absent engine and the generator publishes
    single-engine result sets, so this check has to do the same. It used to load both
    engines unconditionally and died on the missing file the moment one engine was
    published on its own.
    """
    paths = [root / name / (s + "_results.json") for s in SUITES]
    if not all(p.exists() for p in paths):
        return None
    return {s: load_json(p) for s, p in zip(SUITES, paths)}


sglang_data = load_engine("sglang")
trtllm_data = load_engine("trtllm")
if sglang_data is None and trtllm_data is None:
    print("    [SKIP] No engine has a complete five-suite result set to mutate.")
    sys.exit(0)

if sglang_data is not None:
    base, other, first = sglang_data, trtllm_data, True
else:
    base, other, first = trtllm_data, None, False


def call(data):
    return validate_provenance(data, other) if first else validate_provenance(None, data)


def interval_key(doc):
    for k in ["benchmark_config", "soak_config", "sweep_config", "grid", "prefill_config"]:
        if isinstance(doc.get(k), dict) and "suite_start_ts" in doc[k]:
            return k
    return None


# An unmutated set must pass, otherwise every rejection below proves nothing.
call(copy.deepcopy(base))
print("    [OK] Unmutated result set passes the gate")


def expect_reject(label, mutate):
    data = copy.deepcopy(base)
    mutate(data)
    try:
        call(data)
    except ValueError:
        print("    [OK] Rejected: " + label)
        return
    print("ERROR: validate_provenance() accepted " + label + "!", file=sys.stderr)
    sys.exit(1)


# Only properties the gate enforces unconditionally are asserted here. Equality between
# the recorded engine_version and the pinned image tag is deliberately NOT one of them:
# the tag is a build name rather than a version, so that branch never runs, and a test
# claiming otherwise passed only because this whole check was being skipped.
expect_reject("placeholder engine_version", lambda d: d["standard"]["metadata"].update(engine_version="unknown"))
expect_reject("empty engine_version", lambda d: d["standard"]["metadata"].update(engine_version=""))
expect_reject("engine identity not matching its results directory", lambda d: d["standard"]["metadata"].update(engine="somethingelse"))
expect_reject("container image belonging to a different engine", lambda d: d["standard"]["metadata"].update(image="registry/x/unrelated-image:latest"))
expect_reject("malformed run_timestamp", lambda d: d["soak"]["metadata"].update(run_timestamp="invalid-date-format"))
expect_reject("missing run_timestamp", lambda d: d["soak"]["metadata"].update(run_timestamp=""))

key = interval_key(base["soak"])
if key:
    expect_reject(
        "soak interval moved before the suite preceding it",
        lambda d: d["soak"][key].update(suite_start_ts="2020-01-01T00:00:00Z", suite_end_ts="2020-01-01T00:01:00Z"),
    )
    expect_reject(
        "suite_start_ts later than suite_end_ts",
        lambda d: d["soak"][key].update(suite_start_ts="2020-01-02T00:00:00Z", suite_end_ts="2020-01-01T00:00:00Z"),
    )
else:
    print("    [SKIP] soak records no suite interval; ordering mutations not exercised.")
'
  echo "    [OK] Check 6 passed: Provenance Gate rejected every adversarial input."
  PASSED=$((PASSED + 1))
else
  echo "    [SKIP] Check 6 skipped: benchmarks/results/ is empty (pre-launch state)."
  SKIPPED=$((SKIPPED + 1))
fi

# ------------------------------------------------------------------------------
# Check 7: Unit test suite for telemetry sanitizer
# ------------------------------------------------------------------------------
echo "--> Check 7: Verifying unit test suite for telemetry sanitizer..."
python3 -m unittest discover -s tests -p "test_*.py" >/dev/null
echo "    [OK] Check 7 passed: Telemetry sanitizer unit tests passed cleanly."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 8: Verifying dependency floor constraints
# ------------------------------------------------------------------------------
echo "--> Check 8: Verifying dependency floor constraints..."
python3 tests/check_dependency_floors.py >/dev/null
echo "    [OK] Check 8 passed: All dependencies satisfy or exceed floor constraints."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 9: Verifying self-contained secret scan
# ------------------------------------------------------------------------------
echo "--> Check 9: Verifying self-contained secret scan..."
bash tests/check_secret_scan.sh >/dev/null
echo "    [OK] Check 9 passed: Secret scan passed cleanly."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 10: Adversarial Proof of Suite Timestamp Gate
# ------------------------------------------------------------------------------
echo "--> Check 10: Adversarial Proof of Suite Timestamp Gate (testing skip and overlap rejection)..."
if compgen -G "benchmarks/results/*/*.json" > /dev/null; then
  python3 -c '
import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, "benchmarks")
from generate_comparison import validate_provenance, load_json

SUITES = ["standard", "massive", "soak", "saturation", "prefill"]
root = Path("benchmarks/results")


def load_engine(name):
    paths = [root / name / (s + "_results.json") for s in SUITES]
    if not all(p.exists() for p in paths):
        return None
    return {s: load_json(p) for s, p in zip(SUITES, paths)}


sglang_data = load_engine("sglang")
trtllm_data = load_engine("trtllm")
if sglang_data is None and trtllm_data is None:
    print("    [SKIP] No engine has a complete five-suite result set to mutate.")
    sys.exit(0)

if sglang_data is not None:
    base, other, name, first = sglang_data, trtllm_data, "sglang", True
else:
    base, other, name, first = trtllm_data, None, "trtllm", False


def call(data):
    return validate_provenance(data, other) if first else validate_provenance(None, data)


def interval_key(doc):
    for k in ["benchmark_config", "soak_config", "sweep_config", "grid", "prefill_config"]:
        if isinstance(doc.get(k), dict) and "suite_start_ts" in doc[k]:
            return k
    return None


# Skip path. Construct the missing-interval condition rather than relying on the committed
# results to exhibit it: this used to assert that standard_results.json had no
# suite_start_ts, which was true only of the pre-launch placeholder files and became false
# the moment a real run was published.
stripped = copy.deepcopy(base)
key = interval_key(stripped["standard"])
if key:
    stripped["standard"][key].pop("suite_start_ts", None)
    stripped["standard"][key].pop("suite_end_ts", None)
buf = io.StringIO()
with redirect_stdout(buf):
    call(stripped)
if "Skipping interval overlap checks for " + name + " standard" not in buf.getvalue():
    print("ERROR: validate_provenance() did not print the skip notice when suite_start_ts was absent!", file=sys.stderr)
    sys.exit(1)
print("    [OK] Skip path verified: notice printed and no error raised when suite_start_ts is absent.")

# Rejection path: make massive start before standard finishes.
overlap = copy.deepcopy(base)
s_key = interval_key(overlap["standard"]) or "benchmark_config"
m_key = interval_key(overlap["massive"]) or "benchmark_config"
overlap["standard"].setdefault(s_key, {}).update(
    suite_start_ts="2026-07-26T10:00:00Z", suite_end_ts="2026-07-26T10:10:00Z"
)
overlap["massive"].setdefault(m_key, {}).update(
    suite_start_ts="2026-07-26T10:05:00Z", suite_end_ts="2026-07-26T10:20:00Z"
)
try:
    call(overlap)
    print("ERROR: validate_provenance() failed to catch overlapping suite timestamps!", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print("    [OK] Caught expected interval overlap error: " + str(e))
'
  echo "    [OK] Check 10 passed: Suite timestamp gate correctly handles skip and overlap rejection paths."
  PASSED=$((PASSED + 1))
else
  echo "    [SKIP] Check 10 skipped: benchmarks/results/ is empty (pre-launch state)."
  SKIPPED=$((SKIPPED + 1))
fi

# ------------------------------------------------------------------------------
# Check 11: NCCL fabric environment parity and RDMA network definitions
# ------------------------------------------------------------------------------
echo "--> Check 11: Verifying NCCL fabric environment parity and network definitions..."
GET_NCCL_KV() {
  local file="$1"
  awk '/- name: NCCL_/ { key=$3; getline; val=$2; gsub(/["'\'' ]/, "", val); print key "=" val }' "$file" || true
  (grep -oE "NCCL_[A-Za-z0-9_]+=[^[:space:]\"'+]+" "$file" 2>/dev/null || true) | tr -d " \"'"
}
REF_KV="$(GET_NCCL_KV terraform/manifests/templates/00c-nccl-test-job.yaml.template | sort -u)"
for tmpl in 09-kimi-k3-sglang-mpi.yaml.template 09-kimi-k3-trtllm-mpi.yaml.template; do
  TARGET_KV="$(GET_NCCL_KV "terraform/manifests/templates/${tmpl}" | sort -u)"
  DIFF_OUT="$(comm -23 <(echo "${REF_KV}") <(echo "${TARGET_KV}"))" || true
  if [ -n "${DIFF_OUT}" ]; then
    echo "ERROR: Check 11 failed: ${tmpl} is missing or has differing values for NCCL keys from 00c:" >&2
    echo "${DIFF_OUT}" >&2
    exit 1
  fi
  if grep -q "2>/dev/null || true" "terraform/manifests/templates/${tmpl}"; then
    echo "ERROR: Check 11 failed: ${tmpl} swallows NCCL preamble with '2>/dev/null || true'!" >&2
    exit 1
  fi
done
PARAMSET_COUNT="$(grep -c '^kind: GKENetworkParamSet' terraform/manifests/templates/00b-rdma-networks.yaml.template || true)"
NETWORK_COUNT="$(grep -c '^kind: Network' terraform/manifests/templates/00b-rdma-networks.yaml.template || true)"
if [ "${PARAMSET_COUNT}" -ne 8 ] || [ "${NETWORK_COUNT}" -ne 8 ]; then
  echo "ERROR: Check 11 failed: 00b declared ${PARAMSET_COUNT} GKENetworkParamSet and ${NETWORK_COUNT} Network resources (expected 8 each)!" >&2
  exit 1
fi
if (grep -i "mtu" terraform/manifests/templates/00b-rdma-networks.yaml.template 2>/dev/null || true) | grep -qv "8896"; then
  echo "ERROR: Check 11 failed: Found non-8896 MTU in 00b!" >&2
  exit 1
fi
echo "    [OK] Check 11 passed: NCCL fabric environment parity and RDMA network resources verified."
PASSED=$((PASSED + 1))

# ------------------------------------------------------------------------------
# Check 12: Shared render allow-list gate
# ------------------------------------------------------------------------------
echo "--> Check 12: Verifying manifest survivor allow-list parity..."
for eng in sglang trtllm; do
  INFERENCE_ENGINE=$eng ./scripts/03_deploy_workloads.sh --render-only >/dev/null
  bash tests/check_render_exceptions.sh >/dev/null
done
echo "    [OK] Check 12 passed: Unrendered variables match .render-exceptions allow-list."
PASSED=$((PASSED + 1))

echo "=============================================================================="
echo "=== SERVING REMEDIATION SUITE: ${PASSED} passed, ${SKIPPED} skipped ==="
echo "=============================================================================="
