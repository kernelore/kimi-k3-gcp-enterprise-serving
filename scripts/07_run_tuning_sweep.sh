#!/usr/bin/env bash
# ==============================================================================
# 07_run_tuning_sweep.sh - Unattended, resumable Kimi K3 tuning sweep driver
# ==============================================================================
# Walks the Phase B/C variant stack from PERF_TUNING_PLAN.md. For each variant:
# apply its env delta on top of the accepted stack, re-render the serving
# StatefulSet, wait for the rollout, run the benchmark, apply the pre-registered
# keep/back-out rule, checkpoint, and move on.
#
# Three facts drive the whole design.
#
# A config change here is a StatefulSet env change and a **17 min 03 s** warm
# restart. A twelve-cell sweep is about 3.4 hours of restarts before a single
# token is generated, so a restart that did not need to happen is not an
# inefficiency, it is roughly $26 of spot B200 time. Hence: apply first, and
# only force a rollout when the apply changed nothing but the caller asked for
# one anyway.
#
# The pool is spot. A preemption mid-sweep is expected, not exceptional, and
# losing four completed variants to it would cost more than the sweep. Hence a
# checkpoint written after every state transition and --resume as the default.
#
# The stack is cumulative. Each variant is measured as its marginal contribution
# given the changes already accepted, so a backed-out variant must not leak its
# env into the next one, and the next comparison baseline is the last *accepted*
# result rather than the last one run.
#
# Offline: --dry-run prints the full plan, the env deltas and the restart budget
# without touching a cluster.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"
TF_DIR="${PROJECT_ROOT}/terraform"
TEMPLATE_DIR="${TF_DIR}/manifests/templates"
GENERATED_DIR="${TF_DIR}/manifests/generated"

NAMESPACE="llm-serving"
STATEFULSET="kimi-k3-serving"
# shellcheck source=scripts/lib/serving_render_defaults.sh disable=SC1091
source "${SCRIPT_DIR}/lib/serving_render_defaults.sh"

VARIANTS=""
# The realistic harness, not the saturation one. Two reasons, and both of them
# invalidate the sweep if ignored. Its non-repetitive arm is the controlled
# baseline the plan defines every delta against -- the saturation numbers were
# taken on repeated-passage prompts, which a prefix cache serves far too well to
# be a measurement of anything. And its default concurrency levels are 8,16,32,
# so c=16 exists; saturation runs 1,8,32,128 and would make every variant
# undecidable for want of the one cell the rule is anchored on.
BENCH_MODE="realistic"
SWEEP_ARM="non_repetitive"
PRIMARY_CONCURRENCY="16"
BENCH_TARGET="serving"
STATE_DIR="${PROJECT_ROOT}/benchmarks/results/sweep"
RESULTS_BUCKET=""
DRY_RUN="false"
RESUME="true"
FORCE_RESTART="false"
READY_TIMEOUT="1800"
SETTLE_SECONDS="60"
STOP_ON_UNDECIDABLE="false"

# ------------------------------------------------------------------------------
# The variant stack.
#
# Each row is: name | objective | env delta (space-separated KEY=VALUE)
#
# The objective column decides which pre-registered rule applies. Kernel and
# memory changes are judged on throughput; scheduling knobs are judged on TTFT
# P99, because they are expected to cost a little aggregate and scoring them on
# throughput would back out every one of them for doing their job.
#
# B0 carries no delta on purpose: it is the controlled baseline on the A1
# non-repetitive harness, the thing every later number is measured against. The
# README numbers cannot serve as that baseline because they were taken on a
# nonce harness that forces a 0% prefix-cache hit rate.
#
# C0 is absent by design. fp8 KV is gated on benchmarks/kv_accuracy_gate.py, not
# on throughput, and the plan is explicit that accuracy runs first with no
# exceptions. Wiring it in here as one more throughput cell would route around
# that rule.
#
# C3 (hicache on NVMe) is absent for the same kind of reason: it is measured
# with the A2 prefix-reuse bench, since this sweep's harness defeats prefix
# reuse by construction and would show nothing either way.
# ------------------------------------------------------------------------------
VARIANT_TABLE=(
  "B0|throughput|"
  "B1|throughput|SGLANG_LINEAR_ATTN_PREFILL_BACKEND=cutedsl"
  "B2|throughput|SGLANG_MOE_RUNNER_BACKEND=flashinfer_mxfp4"
  "B3|throughput|SGLANG_PREFILL_ATTENTION_BACKEND=fa4"
  "B4|scheduling|SGLANG_CHUNKED_PREFILL_SIZE=4096"
  "B5|scheduling|SGLANG_MAX_RUNNING_REQUESTS=AUTO"
  "C1|throughput|SGLANG_MEM_FRACTION_STATIC=0.88"
  "C2a|throughput|SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE=4"
  "C2b|throughput|SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE=3"
)

# Env keys this driver is allowed to set. A typo in the table would otherwise
# render a manifest missing the var entirely and the variant would silently
# measure the default a second time.
TUNABLE_KEYS=(
  SGLANG_LINEAR_ATTN_PREFILL_BACKEND
  SGLANG_MOE_RUNNER_BACKEND
  SGLANG_PREFILL_ATTENTION_BACKEND
  SGLANG_DECODE_ATTENTION_BACKEND
  SGLANG_CHUNKED_PREFILL_SIZE
  SGLANG_MAX_RUNNING_REQUESTS
  SGLANG_MEM_FRACTION_STATIC
  SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE
  SGLANG_SCHEDULE_POLICY
  SGLANG_KV_CACHE_DTYPE
  SGLANG_ENABLE_HIERARCHICAL_CACHE
  SGLANG_HICACHE_STORAGE_BACKEND
  SGLANG_HICACHE_RATIO
  SGLANG_HICACHE_SIZE
  SGLANG_HICACHE_WRITE_POLICY
  SGLANG_HICACHE_IO_BACKEND
)

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Unattended, resumable driver for the Kimi K3 Phase B/C tuning sweep.

Options:
  --variants <list>       Comma-separated variant names, or 'all' (default: all)
                          Known: $(printf '%s ' "${VARIANT_TABLE[@]%%|*}")
  --bench-mode <realistic|saturation>
                          Benchmark suite per variant (default: realistic, whose
                          non-repetitive arm is the plan's controlled baseline
                          and whose default levels 8,16,32 include the c=16 the
                          decision rule is anchored on)
  --arm <name>            Arm to judge from a realistic results file
                          (default: non_repetitive)
  --primary-concurrency <n>
                          Concurrency the keep/back-out rule is anchored on
                          (default: 16). Checked against the levels the sweep
                          will actually run before any GPU time is spent.
  --target <gateway|serving>  Benchmark target (default: serving, to measure the
                          engine rather than the proxy in front of it)
  --state-dir <path>      Checkpoint and results directory
                          (default: benchmarks/results/sweep)
  --results-bucket <uri>  gs:// prefix to copy results to after each variant.
                          Preemption insurance: a checkpoint on a node that is
                          about to be reclaimed is not a checkpoint.
  --ready-timeout <sec>   Rollout wait per variant (default: 1800; the measured
                          warm restart is 17 min 03 s)
  --settle <sec>          Idle time after Ready before benchmarking (default: 60)
  --no-resume             Ignore the checkpoint and run every named variant
  --force-restart         Roll the StatefulSet even when the manifest is unchanged
  --stop-on-undecidable   Halt if a variant cannot be judged (default: continue,
                          leaving the stack at the last accepted state)
  --dry-run               Print the plan, the env deltas and the restart budget,
                          then exit. Touches nothing, costs nothing.
  -h, --help              Show this usage guide and exit

Resume:
  The checkpoint is <state-dir>/checkpoint.tsv, one line per variant, rewritten
  after every state change. Re-running the same command picks up at the first
  variant that is not already terminal and rebuilds the accepted stack from the
  variants marked accepted.

Examples:
  ./scripts/07_run_tuning_sweep.sh --dry-run
  ./scripts/07_run_tuning_sweep.sh --variants B0,B1,B2,B3 --results-bucket gs://my-bucket/sweep
  ./scripts/07_run_tuning_sweep.sh            # resumes wherever the last run stopped
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variants)        VARIANTS="$2"; shift 2 ;;
    --bench-mode)      BENCH_MODE="$2"; shift 2 ;;
    --arm)             SWEEP_ARM="$2"; shift 2 ;;
    --primary-concurrency) PRIMARY_CONCURRENCY="$2"; shift 2 ;;
    --target)          BENCH_TARGET="$2"; shift 2 ;;
    --state-dir)       STATE_DIR="$2"; shift 2 ;;
    --results-bucket)  RESULTS_BUCKET="$2"; shift 2 ;;
    --ready-timeout)   READY_TIMEOUT="$2"; shift 2 ;;
    --settle)          SETTLE_SECONDS="$2"; shift 2 ;;
    --no-resume)       RESUME="false"; shift ;;
    --force-restart)   FORCE_RESTART="true"; shift ;;
    --stop-on-undecidable) STOP_ON_UNDECIDABLE="true"; shift ;;
    --dry-run)         DRY_RUN="true"; shift ;;
    -h|--help)         show_usage; exit 0 ;;
    *)
      echo "ERROR: Unknown option '$1'" >&2
      show_usage >&2
      exit 1
      ;;
  esac
done

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ${CONFIG_FILE} not found. Please run ./scripts/01_setup_and_check.sh first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [ "${RESULTS_BUCKET}" != "" ] && [[ "${RESULTS_BUCKET}" != gs://* ]]; then
  echo "ERROR: --results-bucket must be a gs:// URI, got '${RESULTS_BUCKET}'" >&2
  exit 1
fi
case "${BENCH_TARGET}" in
  gateway|serving) ;;
  *) echo "ERROR: --target must be 'gateway' or 'serving', got '${BENCH_TARGET}'" >&2; exit 1 ;;
esac
case "${BENCH_MODE}" in
  realistic|saturation) ;;
  *)
    echo "ERROR: --bench-mode must be 'realistic' or 'saturation', got '${BENCH_MODE}'." >&2
    echo "       Those are the two modes that emit a per-concurrency grid the" >&2
    echo "       decision rule can read." >&2
    exit 1
    ;;
esac
if ! [[ "${PRIMARY_CONCURRENCY}" =~ ^[0-9]+$ ]] || [ "${PRIMARY_CONCURRENCY}" -lt 1 ]; then
  echo "ERROR: --primary-concurrency must be a positive integer" >&2
  exit 1
fi

# The rule is anchored on one concurrency band. If the sweep never runs that
# band, every variant comes back undecidable -- after the GPU time is already
# spent. Roughly 25 minutes per variant on the spot pair, so nine variants is
# about $234 of measurement that answers nothing. Fail here instead.
EFFECTIVE_LEVELS="${SWEEP_CONCURRENCY_LEVELS:-}"
if [ -z "${EFFECTIVE_LEVELS}" ]; then
  case "${BENCH_MODE}" in
    realistic)  EFFECTIVE_LEVELS="8,16,32" ;;
    saturation) EFFECTIVE_LEVELS="1,8,32,128" ;;
  esac
fi
if [[ ",${EFFECTIVE_LEVELS}," != *",${PRIMARY_CONCURRENCY},"* ]]; then
  echo "ERROR: the decision rule is anchored on c=${PRIMARY_CONCURRENCY}, but the" >&2
  echo "       ${BENCH_MODE} sweep will run levels ${EFFECTIVE_LEVELS}." >&2
  echo "       Every variant would come back undecidable after the GPU time was" >&2
  echo "       already spent. Either set SWEEP_CONCURRENCY_LEVELS to include" >&2
  echo "       ${PRIMARY_CONCURRENCY}, or move the anchor with --primary-concurrency." >&2
  exit 1
fi
if ! [[ "${READY_TIMEOUT}" =~ ^[0-9]+$ ]] || [ "${READY_TIMEOUT}" -lt 60 ]; then
  echo "ERROR: --ready-timeout must be an integer of at least 60 seconds" >&2
  exit 1
fi
if ! [[ "${SETTLE_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --settle must be a non-negative integer" >&2
  exit 1
fi

CHECKPOINT="${STATE_DIR}/checkpoint.tsv"

# ------------------------------------------------------------------------------
# Variant table helpers
# ------------------------------------------------------------------------------

variant_field() {
  local name="$1" field="$2" row
  for row in "${VARIANT_TABLE[@]}"; do
    if [ "${row%%|*}" = "${name}" ]; then
      case "${field}" in
        objective) printf '%s' "$(echo "${row}" | cut -d'|' -f2)" ;;
        delta)     printf '%s' "$(echo "${row}" | cut -d'|' -f3-)" ;;
      esac
      return 0
    fi
  done
  return 1
}

all_variant_names() {
  local row
  for row in "${VARIANT_TABLE[@]}"; do
    printf '%s\n' "${row%%|*}"
  done
}

resolve_variants() {
  local requested="${1}"
  if [ -z "${requested}" ] || [ "${requested}" = "all" ]; then
    all_variant_names
    return 0
  fi
  local name
  local -a wanted=()
  IFS=',' read -r -a wanted <<< "${requested}"
  for name in "${wanted[@]}"; do
    name="$(echo "${name}" | tr -d '[:space:]')"
    [ -z "${name}" ] && continue
    if ! variant_field "${name}" objective >/dev/null; then
      echo "ERROR: unknown variant '${name}'. Known: $(all_variant_names | tr '\n' ' ')" >&2
      exit 1
    fi
    printf '%s\n' "${name}"
  done
}

validate_table() {
  local row name key kv delta
  for row in "${VARIANT_TABLE[@]}"; do
    name="${row%%|*}"
    delta="$(echo "${row}" | cut -d'|' -f3-)"
    for kv in ${delta}; do
      key="${kv%%=*}"
      if [[ "${kv}" != *=* ]]; then
        echo "ERROR: variant ${name} has malformed delta entry '${kv}' (expected KEY=VALUE)" >&2
        exit 1
      fi
      if ! printf '%s\n' "${TUNABLE_KEYS[@]}" | grep -qx "${key}"; then
        # A key not in the render's allowed list renders as empty, so the
        # variant would quietly re-measure the default and report it as a
        # result. Fail here instead, where it is free.
        echo "ERROR: variant ${name} sets '${key}', which is not a known tunable." >&2
        echo "       Add it to TUNABLE_KEYS and to BASE_ALLOWED_VARS in 03_deploy_workloads.sh first." >&2
        exit 1
      fi
    done
  done
}

# ------------------------------------------------------------------------------
# Checkpoint
#
# Tab-separated, one line per variant: name, status, objective, results file,
# timestamp. Rewritten in full on every transition rather than appended, so a
# resumed run reads one authoritative line per variant instead of replaying a
# log and inferring the latest.
# ------------------------------------------------------------------------------

declare -A CP_STATUS=()
declare -A CP_RESULTS=()

load_checkpoint() {
  CP_STATUS=()
  CP_RESULTS=()
  [ -f "${CHECKPOINT}" ] || return 0
  local name status objective results stamp
  while IFS=$'\t' read -r name status objective results stamp; do
    [ -z "${name}" ] && continue
    [ "${name}" = "#variant" ] && continue
    : "${objective}" "${stamp}"
    CP_STATUS["${name}"]="${status}"
    CP_RESULTS["${name}"]="${results}"
  done < "${CHECKPOINT}"
}

write_checkpoint() {
  local tmp="${CHECKPOINT}.tmp"
  {
    printf '#variant\tstatus\tobjective\tresults\ttimestamp\n'
    local name
    while read -r name; do
      [ -n "${CP_STATUS[${name}]:-}" ] || continue
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${CP_STATUS[${name}]}" \
        "$(variant_field "${name}" objective || echo unknown)" \
        "${CP_RESULTS[${name}]:-}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    done < <(all_variant_names)
  } > "${tmp}"
  mv -f "${tmp}" "${CHECKPOINT}"
}

set_status() {
  CP_STATUS["$1"]="$2"
  [ -n "${3:-}" ] && CP_RESULTS["$1"]="$3"
  write_checkpoint
}

is_terminal() {
  case "${1}" in
    accepted|backed_out|failed|skipped) return 0 ;;
    *) return 1 ;;
  esac
}

# The stack is whatever the accepted variants set, applied in table order so a
# later variant overriding an earlier one (C2a then C2b) resolves the way the
# table reads.
accepted_stack_env() {
  local name kv
  while read -r name; do
    [ "${CP_STATUS[${name}]:-}" = "accepted" ] || continue
    for kv in $(variant_field "${name}" delta); do
      printf '%s\n' "${kv}"
    done
  done < <(all_variant_names)
}

# ------------------------------------------------------------------------------
# Cluster actions
# ------------------------------------------------------------------------------

# shellcheck disable=SC2016
safe_envsubst() {
  python3 -c '
import os, sys, re
allowed = set()
for arg in sys.argv[1:]:
    for var in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", arg):
        allowed.add(var)

content = sys.stdin.read()
def replace_var(match):
    var_name = match.group(1) or match.group(2)
    if not allowed or var_name in allowed:
        return os.environ.get(var_name, "")
    return match.group(0)

output = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, content)
sys.stdout.write(output)
' "$@"
}

serving_template() {
  local engine="${INFERENCE_ENGINE:-sglang}"
  printf '%s/09-kimi-k3-%s-mpi.yaml.template' "${TEMPLATE_DIR}" "${engine}"
}

# MAX_RUNNING_REQUESTS is the one value the plan computes rather than fixes:
# floor(max_total_num_tokens / (ISL+OSL)), reading max_total_num_tokens from the
# engine's own startup log. AUTO defers that to run time, because the number is
# a property of the pool the engine actually built and guessing it in a table
# would make B5 measure an arbitrary cap.
resolve_auto_value() {
  local key="$1"
  case "${key}" in
    SGLANG_MAX_RUNNING_REQUESTS)
      local total isl_osl
      total="$(kubectl logs -n "${NAMESPACE}" "${STATEFULSET}-0" --tail=-1 2>/dev/null \
               | grep -oE 'max_total_num_tokens=[0-9]+' | tail -1 | cut -d= -f2 || true)"
      isl_osl="${SWEEP_AUTO_ISL_OSL:-2048}"
      if [ -z "${total}" ]; then
        echo "" ; return 1
      fi
      echo $(( total / isl_osl ))
      ;;
    *)
      echo "" ; return 1 ;;
  esac
}

render_serving_manifest() {
  local out="$1"
  local template
  template="$(serving_template)"
  if [ ! -f "${template}" ]; then
    echo "ERROR: serving template ${template} not found" >&2
    return 1
  fi
  # Same allowed-var list the deploy script uses. Kept as a single source by
  # reading it out of 03 rather than restating it, so a var added there does not
  # silently render empty here.
  local allowed
  allowed="$(grep -m1 "^BASE_ALLOWED_VARS=" "${SCRIPT_DIR}/03_deploy_workloads.sh" \
             | sed "s/^BASE_ALLOWED_VARS='//; s/'$//")"
  if [ -z "${allowed}" ]; then
    echo "ERROR: could not read BASE_ALLOWED_VARS from 03_deploy_workloads.sh" >&2
    return 1
  fi

  # The allow-list is only half of what sharing with 03 needs to be: it controls which
  # names may be substituted and says nothing about their values. See the header of
  # lib/serving_render_defaults.sh -- this is what failed all nine variants of the first
  # live sweep, on a cluster that was up and answering inference correctly.
  ensure_serving_render_env || return 1

  mkdir -p "$(dirname "${out}")"
  safe_envsubst "${allowed}" < "${template}" > "${out}"
  assert_manifest_valid "${out}" || return 1
}

wait_for_ready() {
  local timeout="$1"
  echo "    Waiting up to ${timeout}s for ${STATEFULSET} rollout (measured warm restart: 17 min 03 s)..."
  if ! kubectl rollout status "statefulset/${STATEFULSET}" -n "${NAMESPACE}" --timeout="${timeout}s"; then
    echo "ERROR: rollout did not complete within ${timeout}s" >&2
    kubectl get pods -n "${NAMESPACE}" -l "app=${STATEFULSET}" >&2 || true
    return 1
  fi
  local ready
  ready="$(kubectl get statefulset "${STATEFULSET}" -n "${NAMESPACE}" \
           -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)"
  if [ "${ready:-0}" -lt 1 ]; then
    echo "ERROR: rollout reported complete but readyReplicas is ${ready:-0}" >&2
    return 1
  fi
  return 0
}

push_results() {
  local variant="$1" path="$2"
  [ -n "${RESULTS_BUCKET}" ] || return 0
  [ -s "${path}" ] || return 0
  echo "    Copying ${path} to ${RESULTS_BUCKET}/${variant}/"
  if ! gcloud storage cp "${path}" "${RESULTS_BUCKET}/${variant}/" 2>&1; then
    # Not fatal. The local checkpoint is still valid and the run should keep
    # going; losing the upload is cheaper than losing the remaining variants.
    echo "WARNING: failed to copy results for ${variant} to ${RESULTS_BUCKET}" >&2
  fi
}

# ------------------------------------------------------------------------------
# Plan
# ------------------------------------------------------------------------------

mkdir -p "${STATE_DIR}"
validate_table
load_checkpoint

mapfile -t PLANNED < <(resolve_variants "${VARIANTS}")
if [ "${#PLANNED[@]}" -eq 0 ]; then
  echo "ERROR: no variants selected" >&2
  exit 1
fi

TO_RUN=()
for variant in "${PLANNED[@]}"; do
  status="${CP_STATUS[${variant}]:-pending}"
  if [ "${RESUME}" = "true" ] && is_terminal "${status}"; then
    continue
  fi
  TO_RUN+=("${variant}")
done

echo "=============================================================================="
echo "Kimi K3 Tuning Sweep Driver"
echo "=============================================================================="
echo "    Engine:        ${INFERENCE_ENGINE:-sglang}"
echo "    Bench mode:    ${BENCH_MODE} (target: ${BENCH_TARGET})"
echo "    Judged on:     arm '${SWEEP_ARM}' at c=${PRIMARY_CONCURRENCY}, levels ${EFFECTIVE_LEVELS}"
echo "    State dir:     ${STATE_DIR}"
echo "    Results bucket:${RESULTS_BUCKET:-<none: results stay on this node only>}"
echo "    Planned:       ${PLANNED[*]}"
if [ "${RESUME}" = "true" ] && [ "${#TO_RUN[@]}" -ne "${#PLANNED[@]}" ]; then
  echo "    Resuming:      $(( ${#PLANNED[@]} - ${#TO_RUN[@]} )) variant(s) already terminal in ${CHECKPOINT}"
fi
echo "    To run:        ${TO_RUN[*]:-<nothing, all selected variants are terminal>}"
echo ""

if [ "${#TO_RUN[@]}" -eq 0 ]; then
  echo "Nothing to do. Use --no-resume to re-run terminal variants."
  exit 0
fi

if [ "${DRY_RUN}" = "true" ]; then
  echo "--- DRY RUN: no cluster is contacted and nothing is rendered ---"
  echo ""
  declare -A preview_value=()
  preview_order=()
  for variant in "${PLANNED[@]}"; do
    objective="$(variant_field "${variant}" objective)"
    delta="$(variant_field "${variant}" delta)"
    status="${CP_STATUS[${variant}]:-pending}"
    printf '  %-5s objective=%-10s status=%-10s delta=%s\n' \
      "${variant}" "${objective}" "${status}" "${delta:-<none: controlled baseline>}"
    # Show the stack as it would stand if every variant were accepted, which is
    # the worst case for restart count and the only one worth budgeting for.
    # Resolved per key rather than listed per variant: C2a and C2b both set the
    # DSPARK block size, and printing both would show a stack that never exists.
    for kv in ${delta}; do
      key="${kv%%=*}"
      if [ -z "${preview_value[${key}]:-}" ]; then
        preview_order+=("${key}")
      fi
      preview_value["${key}"]="${kv#*=}"
    done
  done
  echo ""
  # On a resume this is the one that matters: it is what the next variant will
  # actually stack on, with backed-out variants already excluded.
  echo "  Accepted stack as of the current checkpoint:"
  if [ -z "$(accepted_stack_env)" ]; then
    echo "    <nothing accepted yet>"
  else
    accepted_stack_env | sed 's/^/    /'
  fi
  echo ""
  echo "  Cumulative env if every remaining variant is also accepted (last setter of a key wins):"
  if [ "${#preview_order[@]}" -eq 0 ]; then
    echo "    <none>"
  else
    for key in "${preview_order[@]}"; do
      printf '    %s=%s\n' "${key}" "${preview_value[${key}]}"
    done
  fi
  echo ""
  restarts="${#TO_RUN[@]}"
  restart_minutes=$(( restarts * 1023 / 60 ))
  echo "  Restart budget: ${restarts} variant(s) x 17 min 03 s = ~${restart_minutes} min of restart alone,"
  echo "                  before any benchmark time. At ~\$92/h that is ~\$$(( restart_minutes * 92 / 60 )) of restarts."
  echo ""
  echo "  Decision rules that will be applied:"
  echo "    throughput: keep on >=2% aggregate gain at c=16, no >2% regression at c=8/c=32,"
  echo "                no >10% TTFT P99 worsening"
  echo "    scheduling: keep on any TTFT P99 improvement, aggregate not down more than 5%"
  echo ""
  echo "  Not driven from here, by design:"
  echo "    C0 (fp8 KV)     -> benchmarks/kv_accuracy_gate.py; accuracy gates before throughput"
  echo "    C3 (hicache)    -> benchmarks/run_prefix_reuse_bench.py; this harness defeats prefix reuse"
  echo "    C4 (final)      -> scripts/05_run_benchmarks.sh --mode all, as separate sequential Jobs"
  exit 0
fi

# ------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl not found." >&2
  exit 1
fi
if ! kubectl get statefulset "${STATEFULSET}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "ERROR: statefulset/${STATEFULSET} not found in namespace ${NAMESPACE}." >&2
  echo "       Deploy with ./scripts/03_deploy_workloads.sh before sweeping." >&2
  exit 1
fi

INTERRUPTED="false"
# shellcheck disable=SC2317  # invoked via trap, which shellcheck cannot see
on_interrupt() {
  INTERRUPTED="true"
  echo ""
  echo "--> Interrupted. The checkpoint at ${CHECKPOINT} is current; re-run to resume." >&2
}
trap on_interrupt INT TERM

BASELINE_RESULTS=""
for variant in "${PLANNED[@]}"; do
  if [ "${CP_STATUS[${variant}]:-}" = "accepted" ] && [ -n "${CP_RESULTS[${variant}]:-}" ]; then
    BASELINE_RESULTS="${CP_RESULTS[${variant}]}"
  fi
done

for variant in "${TO_RUN[@]}"; do
  [ "${INTERRUPTED}" = "true" ] && break

  objective="$(variant_field "${variant}" objective)"
  delta="$(variant_field "${variant}" delta)"
  results_file="${STATE_DIR}/${variant}_${BENCH_MODE}_results.json"

  echo ""
  echo "------------------------------------------------------------------------------"
  echo "--> Variant ${variant} (objective: ${objective})"
  echo "------------------------------------------------------------------------------"
  echo "    Delta:    ${delta:-<none: controlled baseline>}"
  set_status "${variant}" "running"

  # Rebuild the environment from the accepted stack every time rather than
  # mutating in place, so a backed-out variant cannot leak into its successor.
  for key in "${TUNABLE_KEYS[@]}"; do
    unset "${key}" || true
  done
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  while read -r kv; do
    [ -z "${kv}" ] && continue
    export "${kv?}"
  done < <(accepted_stack_env)

  variant_failed="false"
  for kv in ${delta}; do
    key="${kv%%=*}"
    value="${kv#*=}"
    if [ "${value}" = "AUTO" ]; then
      if ! value="$(resolve_auto_value "${key}")" || [ -z "${value}" ]; then
        echo "ERROR: could not resolve AUTO for ${key} from the engine startup log." >&2
        echo "       Set SWEEP_AUTO_ISL_OSL, or pin the value in VARIANT_TABLE." >&2
        variant_failed="true"
        break
      fi
      echo "    Resolved ${key}=AUTO to ${value} from the engine's own max_total_num_tokens"
    fi
    export "${key}=${value}"
    echo "    Set ${key}=${value}"
  done
  if [ "${variant_failed}" = "true" ]; then
    set_status "${variant}" "failed"
    [ "${STOP_ON_UNDECIDABLE}" = "true" ] && break
    continue
  fi

  rendered="${GENERATED_DIR}/09-kimi-k3-${INFERENCE_ENGINE:-sglang}-mpi.yaml"
  previous_hash=""
  [ -f "${rendered}" ] && previous_hash="$(sha256sum "${rendered}" | cut -d' ' -f1)"
  if ! render_serving_manifest "${rendered}"; then
    set_status "${variant}" "failed"
    [ "${STOP_ON_UNDECIDABLE}" = "true" ] && break
    continue
  fi
  new_hash="$(sha256sum "${rendered}" | cut -d' ' -f1)"

  if ! kubectl apply -f "${rendered}"; then
    echo "ERROR: kubectl apply failed for ${variant}" >&2
    set_status "${variant}" "failed"
    [ "${STOP_ON_UNDECIDABLE}" = "true" ] && break
    continue
  fi

  # apply already rolls the StatefulSet when the pod spec changed. Adding a
  # rollout restart on top would roll it twice: 34 minutes and roughly $52 of
  # spot time to reach the same state.
  if [ "${previous_hash}" = "${new_hash}" ]; then
    if [ "${FORCE_RESTART}" = "true" ]; then
      echo "    Manifest unchanged; forcing a rollout because --force-restart was given."
      kubectl rollout restart "statefulset/${STATEFULSET}" -n "${NAMESPACE}"
    else
      echo "    Manifest unchanged, so no restart is needed (saving ~17 min and ~\$26)."
    fi
  else
    echo "    Manifest changed; the apply above rolls the StatefulSet."
  fi

  if ! wait_for_ready "${READY_TIMEOUT}"; then
    set_status "${variant}" "failed"
    [ "${STOP_ON_UNDECIDABLE}" = "true" ] && break
    continue
  fi

  if [ "${SETTLE_SECONDS}" -gt 0 ]; then
    echo "    Settling for ${SETTLE_SECONDS}s before measuring..."
    sleep "${SETTLE_SECONDS}"
  fi

  echo "    Running ${BENCH_MODE} benchmark..."
  if ! "${SCRIPT_DIR}/05_run_benchmarks.sh" --mode "${BENCH_MODE}" --target "${BENCH_TARGET}"; then
    echo "WARNING: benchmark reported a non-zero exit for ${variant}" >&2
  fi

  produced="${PROJECT_ROOT}/benchmarks/results/${INFERENCE_ENGINE:-sglang}/${BENCH_MODE}_results.json"
  if [ -s "${produced}" ]; then
    cp "${produced}" "${results_file}"
  else
    echo "ERROR: no results at ${produced} for ${variant}" >&2
    set_status "${variant}" "failed"
    [ "${STOP_ON_UNDECIDABLE}" = "true" ] && break
    continue
  fi
  push_results "${variant}" "${results_file}"

  # B0 is the baseline itself; there is nothing before it to compare against.
  if [ -z "${BASELINE_RESULTS}" ]; then
    echo "    First completed variant: adopting ${variant} as the comparison baseline."
    set_status "${variant}" "accepted" "${results_file}"
    BASELINE_RESULTS="${results_file}"
    continue
  fi

  verdict_file="${STATE_DIR}/${variant}_verdict.json"
  set +e
  python3 "${PROJECT_ROOT}/benchmarks/sweep_decision.py" \
    --baseline "${BASELINE_RESULTS}" \
    --candidate "${results_file}" \
    --objective "${objective}" \
    --variant "${variant}" \
    --arm "${SWEEP_ARM}" \
    --primary-concurrency "${PRIMARY_CONCURRENCY}" \
    --output "${verdict_file}"
  decision_rc=$?
  set -e
  push_results "${variant}" "${verdict_file}"

  case "${decision_rc}" in
    0)
      echo "    ${variant} ACCEPTED; it stays in the stack and becomes the new baseline."
      set_status "${variant}" "accepted" "${results_file}"
      BASELINE_RESULTS="${results_file}"
      ;;
    1)
      echo "    ${variant} BACKED OUT; the stack stays where it was and the next variant builds on it."
      set_status "${variant}" "backed_out" "${results_file}"
      ;;
    *)
      echo "    ${variant} UNDECIDABLE; not accepted, so the stack is unchanged." >&2
      set_status "${variant}" "undecided" "${results_file}"
      if [ "${STOP_ON_UNDECIDABLE}" = "true" ]; then
        echo "    Halting because --stop-on-undecidable was given." >&2
        break
      fi
      ;;
  esac
done

trap - INT TERM

echo ""
echo "=============================================================================="
echo "Sweep Summary"
echo "=============================================================================="
printf '  %-6s %-12s %s\n' "VARIANT" "STATUS" "RESULTS"
while read -r name; do
  [ -n "${CP_STATUS[${name}]:-}" ] || continue
  printf '  %-6s %-12s %s\n' "${name}" "${CP_STATUS[${name}]}" "${CP_RESULTS[${name}]:-}"
done < <(all_variant_names)
echo ""
echo "  Accepted stack:"
if [ -z "$(accepted_stack_env)" ]; then
  echo "    <shipped defaults; nothing cleared the pre-registered bar>"
else
  accepted_stack_env | sed 's/^/    /'
fi
echo ""
echo "  Checkpoint: ${CHECKPOINT}"
if [ "${INTERRUPTED}" = "true" ]; then
  echo "  Run was interrupted; re-run the same command to resume."
  exit 130
fi

# accepted, backed_out and undecided are all outcomes of the decision rule: the variant
# ran, the benchmark produced numbers, and the rule read them. failed is not an outcome.
# It means the apply was rejected, the rollout never became ready, or the benchmark never
# ran -- an infrastructure defect wearing a verdict's clothing.
#
# This exists because the first live sweep marked all nine variants failed in seven
# seconds and still exited 0, so the measurement window recorded sweep.done and moved on
# to spend the rest of an eight-hour budget on a cluster whose sweep had produced nothing.
# A resume then skipped the sweep entirely, because .done said it was finished.
measured=0
failed_count=0
for name in "${PLANNED[@]}"; do
  case "${CP_STATUS[${name}]:-pending}" in
    accepted|backed_out|undecided) measured=$(( measured + 1 )) ;;
    failed)                        failed_count=$(( failed_count + 1 )) ;;
  esac
done
if [ "${failed_count}" -gt 0 ]; then
  echo ""
  echo "  WARNING: ${failed_count} of ${#PLANNED[@]} variant(s) failed without producing a measurement." >&2
fi
if [ "${measured}" -eq 0 ]; then
  echo "  Nothing was measured. Exiting nonzero so the caller does not record this sweep as" >&2
  echo "  complete; a partial sweep is still worth keeping, an empty one is a bug to fix." >&2
  exit 1
fi

echo "  Still to do by hand: C0 fp8 accuracy gate, C3 hicache via the prefix-reuse bench,"
echo "  and C4 final validation as separate sequential Jobs."
