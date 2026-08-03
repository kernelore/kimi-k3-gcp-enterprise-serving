#!/usr/bin/env bash
# ==============================================================================
# 08_run_measurement_window.sh - one unattended, self-terminating measurement window
# ==============================================================================
# Deploys the stack, runs every measurement that needs GPUs, pulls the results
# back, and destroys everything. The destroy is not a final step that a failure
# can skip past -- it is an EXIT trap plus a detached watchdog, because the
# failure mode that matters here is not a wrong number, it is 16 B200s left
# running at roughly $92/h because something died at 03:00 and nobody noticed.
#
# Three independent things have to go wrong before the meter runs unbounded:
#
#   1. The EXIT trap. Fires on success, on error, on Ctrl-C, on SIGTERM.
#   2. The watchdog. A detached process holding a wall-clock deadline and this
#      script's PID. If the deadline passes, or this script disappears without
#      its trap having cleaned up, the watchdog destroys the stack itself. It
#      survives kill -9 of the runner and the death of the terminal.
#   3. The residual: if this VM itself dies, neither survives. That is the one
#      gap, and it is stated rather than papered over -- check the console.
#
# Ordering is deliberate. terraform apply brings up a cluster whose GPU pool has
# total_min_node_count = 0, and the container image build runs before any GPU
# pod is scheduled, so cluster creation and the Cloud Build cost no GPU time.
# The meter starts when the serving StatefulSet is applied, and not before.
#
# Never sets POPULATE_WEIGHTS_CACHE, PURGE_WEIGHTS_BACKUP or FORCE_WEIGHT_JOB.
# It forces all three false: unattended teardown needs FORCE_DESTROY=true, and
# that disarms one of the two locks on the 1.45 TiB weights backup, leaving
# PURGE_WEIGHTS_BACKUP as the only thing standing between an automated run and
# a multi-hour re-download.
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

BUDGET_HOURS="8"
STAGES="deploy,sweep,kv,prefix,collect"
RESULTS_BUCKET=""
DRY_RUN="false"
KEEP_CLUSTER="false"
READY_TIMEOUT="1800"
RUN_LABEL=""

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

One unattended measurement window: deploy, measure, collect, destroy.

Options:
  --budget-hours <n>    Wall-clock ceiling before the watchdog forces teardown
                        (default: 8). At ~\$92/h this is also the cost ceiling.
  --stages <list>       Comma-separated subset of: deploy,sweep,kv,prefix,collect
                        (default: all of them, in that order)
  --results-bucket <uri>  gs:// prefix to mirror results to as they land.
                        Spot nodes get reclaimed; a result only on a reclaimed
                        node is not a result.
  --ready-timeout <sec> Rollout wait per restart (default: 1800; measured warm
                        restart is 17 min 03 s)
  --run-label <name>    Log/state subdirectory name (default: derived from the
                        start time)
  --keep-cluster        Skip the teardown. Refuses unless CONFIRM_KEEP_CLUSTER=1
                        is also set, because this is how you get a surprise bill.
  --dry-run             Print the plan and the cost ceiling, touch nothing.
  -h, --help            Show this usage guide and exit

The window is resumable. Re-running with the same --run-label picks up at the
first stage that did not complete; terraform apply is idempotent and the sweep
driver keeps its own per-variant checkpoint.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget-hours)   BUDGET_HOURS="$2"; shift 2 ;;
    --stages)         STAGES="$2"; shift 2 ;;
    --results-bucket) RESULTS_BUCKET="$2"; shift 2 ;;
    --ready-timeout)  READY_TIMEOUT="$2"; shift 2 ;;
    --run-label)      RUN_LABEL="$2"; shift 2 ;;
    --keep-cluster)   KEEP_CLUSTER="true"; shift ;;
    --dry-run)        DRY_RUN="true"; shift ;;
    -h|--help)        show_usage; exit 0 ;;
    *)
      echo "ERROR: Unknown option '$1'" >&2
      show_usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${BUDGET_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: --budget-hours must be a number, got '${BUDGET_HOURS}'" >&2
  exit 1
fi
if [ -n "${RESULTS_BUCKET}" ] && [[ "${RESULTS_BUCKET}" != gs://* ]]; then
  echo "ERROR: --results-bucket must be a gs:// URI, got '${RESULTS_BUCKET}'" >&2
  exit 1
fi
if [ "${KEEP_CLUSTER}" = "true" ] && [ "${CONFIRM_KEEP_CLUSTER:-}" != "1" ]; then
  echo "ERROR: --keep-cluster leaves 16 B200s running at roughly \$92/h with no" >&2
  echo "       watchdog behind them. Set CONFIRM_KEEP_CLUSTER=1 to mean it." >&2
  exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ${CONFIG_FILE} not found. Run ./scripts/01_setup_and_check.sh first." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Cost guardrails. Checked before anything is created, so a config left in a
# dangerous state fails at second zero rather than three hours in.
# ------------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

for danger in POPULATE_WEIGHTS_CACHE PURGE_WEIGHTS_BACKUP FORCE_WEIGHT_JOB; do
  if [ "${!danger:-false}" = "true" ] || [ "${!danger:-false}" = "1" ]; then
    echo "ERROR: ${danger} is true in ${CONFIG_FILE}." >&2
    case "${danger}" in
      POPULATE_WEIGHTS_CACHE)
        echo "       That bypasses GCS hydration and forces a multi-hour HuggingFace" >&2
        echo "       re-download of a 1.45 TiB checkpoint, on the clock." >&2 ;;
      PURGE_WEIGHTS_BACKUP)
        echo "       Combined with the FORCE_DESTROY=true this script needs for an" >&2
        echo "       unattended teardown, that deletes the weights backup bucket." >&2 ;;
      FORCE_WEIGHT_JOB)
        echo "       That recreates the ROX disk in ReadWrite mode and restages the" >&2
        echo "       whole checkpoint instead of doing a warm hydrate." >&2 ;;
    esac
    exit 1
  fi
done
export POPULATE_WEIGHTS_CACHE="false"
export PURGE_WEIGHTS_BACKUP="false"
export FORCE_WEIGHT_JOB="false"

if [ -z "${CLUSTER_NAME:-}" ]; then
  echo "ERROR: CLUSTER_NAME is not set in ${CONFIG_FILE}; the watchdog would have" >&2
  echo "       nothing to check for and could not confirm a teardown." >&2
  exit 1
fi

START_EPOCH="$(date -u +%s)"
if [ -z "${RUN_LABEL}" ]; then
  RUN_LABEL="window-$(date -u -d "@${START_EPOCH}" +%Y%m%dT%H%M%SZ)"
fi
DEADLINE_EPOCH="$(python3 -c "print(int(${START_EPOCH} + ${BUDGET_HOURS} * 3600))")"
STATE_ROOT="${PROJECT_ROOT}/benchmarks/results/windows/${RUN_LABEL}"
LOG_DIR="${STATE_ROOT}/logs"
SWEEP_STATE="${STATE_ROOT}/sweep"
STAGE_DIR="${STATE_ROOT}/stages"
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results/${INFERENCE_ENGINE:-sglang}"

stage_wanted() { [[ ",${STAGES}," == *",$1,"* ]]; }
stage_done()   { [ -f "${STAGE_DIR}/$1.done" ]; }
mark_done()    { touch "${STAGE_DIR}/$1.done"; }
ts()           { date -u +%Y-%m-%dT%H:%M:%SZ; }
say()          { echo "[$(ts)] $*"; }

elapsed_note() {
  local now remain
  now="$(date -u +%s)"
  remain=$(( (DEADLINE_EPOCH - now) / 60 ))
  say "elapsed $(( (now - START_EPOCH) / 60 )) min, ${remain} min left on the budget"
}

# ------------------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------------------
if [ "${DRY_RUN}" = "true" ]; then
  cat << EOF
==============================================================================
Kimi K3 Measurement Window - DRY RUN
==============================================================================
    Run label:      ${RUN_LABEL}
    Cluster:        ${CLUSTER_NAME} in ${REGION:-?}/${ZONE:-?}
    Engine:         ${INFERENCE_ENGINE:-sglang}
    Stages:         ${STAGES}
    Budget:         ${BUDGET_HOURS} h wall clock -> teardown forced at $(date -u -d "@${DEADLINE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)
    Cost ceiling:   ~\$$(python3 -c "print(int(${BUDGET_HOURS} * 92))") at ~\$92/h once GPU nodes are up
    Results bucket: ${RESULTS_BUCKET:-<none: results stay on this VM>}
    State dir:      ${STATE_ROOT}

    Guardrails asserted: POPULATE_WEIGHTS_CACHE=false PURGE_WEIGHTS_BACKUP=false
                         FORCE_WEIGHT_JOB=false
    Teardown:       EXIT trap + detached watchdog (deadline and PID based)

    Nothing was created. Drop --dry-run to run it.
==============================================================================
EOF
  exit 0
fi

mkdir -p "${LOG_DIR}" "${SWEEP_STATE}" "${STAGE_DIR}"

# ------------------------------------------------------------------------------
# Teardown, and the watchdog that runs it when this script cannot
# ------------------------------------------------------------------------------
TEARDOWN_DONE="false"

do_teardown() {
  local reason="$1"
  if [ "${TEARDOWN_DONE}" = "true" ]; then
    return 0
  fi
  TEARDOWN_DONE="true"
  if [ "${KEEP_CLUSTER}" = "true" ]; then
    say "TEARDOWN SKIPPED (--keep-cluster). The cluster is still running and"
    say "still billing. Destroy it with: FORCE_DESTROY=true ./scripts/06_destroy_all.sh"
    return 0
  fi
  say "TEARDOWN (${reason})"
  local attempt
  for attempt in 1 2 3; do
    if FORCE_DESTROY=true PURGE_WEIGHTS_BACKUP=false \
       bash "${SCRIPT_DIR}/06_destroy_all.sh" >> "${LOG_DIR}/teardown.log" 2>&1; then
      say "teardown succeeded on attempt ${attempt}"
      return 0
    fi
    say "teardown attempt ${attempt} failed; see ${LOG_DIR}/teardown.log"
    sleep 30
  done
  say "TEARDOWN FAILED THREE TIMES. The cluster may still be billing."
  say "Check now: gcloud container clusters list"
  return 1
}

on_exit() {
  local rc=$?
  do_teardown "exit rc=${rc}" || true
  elapsed_note
  say "window ${RUN_LABEL} finished with rc=${rc}; logs in ${LOG_DIR}"
  exit "${rc}"
}
trap on_exit EXIT
trap 'echo; say "SIGINT received"; exit 130' INT
trap 'say "SIGTERM received"; exit 143' TERM

arm_watchdog() {
  local wd="${LOG_DIR}/watchdog.sh"
  cat > "${wd}" << 'WATCHDOG'
#!/usr/bin/env bash
# Detached dead-man's switch. Destroys the stack if the runner dies without
# cleaning up, or if the wall-clock deadline passes. Deliberately dependency
# free and deliberately not set -e: it must survive its own errors.
runner_pid="$1"; deadline="$2"; root="$3"; logdir="$4"; cluster="$5"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog: $*" >> "${logdir}/watchdog.log"; }

cluster_up() {
  gcloud container clusters list --filter="name=${cluster}" \
    --format='value(name)' 2>/dev/null | grep -q .
}

force_destroy() {
  log "forcing teardown"
  ( cd "${root}" && FORCE_DESTROY=true PURGE_WEIGHTS_BACKUP=false \
    bash scripts/06_destroy_all.sh >> "${logdir}/watchdog-teardown.log" 2>&1 )
  log "teardown exited rc=$?"
}

log "armed: runner pid ${runner_pid}, deadline $(date -u -d "@${deadline}" +%Y-%m-%dT%H:%M:%SZ), cluster ${cluster}"
while true; do
  sleep 60
  now="$(date -u +%s)"
  if [ "${now}" -ge "${deadline}" ]; then
    log "wall-clock budget exhausted"
    kill -TERM "${runner_pid}" 2>/dev/null
    sleep 120
    kill -KILL "${runner_pid}" 2>/dev/null
    if cluster_up; then force_destroy; else log "cluster already gone"; fi
    exit 0
  fi
  if ! kill -0 "${runner_pid}" 2>/dev/null; then
    log "runner gone; allowing 300s for its own trap to finish"
    sleep 300
    if cluster_up; then
      log "cluster still up after grace period -- the runner's trap did not clean up"
      force_destroy
    else
      log "cluster gone; nothing to do"
    fi
    exit 0
  fi
done
WATCHDOG
  chmod +x "${wd}"
  setsid nohup bash "${wd}" "$$" "${DEADLINE_EPOCH}" "${PROJECT_ROOT}" \
    "${LOG_DIR}" "${CLUSTER_NAME}" < /dev/null > /dev/null 2>&1 &
  WATCHDOG_PID="$!"
  say "watchdog armed (pid ${WATCHDOG_PID}), deadline $(date -u -d "@${DEADLINE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)"
}

# ------------------------------------------------------------------------------
# Restart helper for the two legs the sweep driver does not own
# ------------------------------------------------------------------------------
# Renders the serving manifest from the currently exported environment and
# applies it. kubectl apply rolls the StatefulSet by itself when the pod spec
# changed; adding a rollout restart on top would pay 17 minutes twice to reach
# the same state.
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
    name = match.group(1) or match.group(2)
    if not allowed or name in allowed:
        return os.environ.get(name, "")
    return match.group(0)
sys.stdout.write(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, content))
' "$@"
}

restart_with_env() {
  local label="$1"
  local engine="${INFERENCE_ENGINE:-sglang}"
  local template="${TEMPLATE_DIR}/09-kimi-k3-${engine}-mpi.yaml.template"
  local rendered="${GENERATED_DIR}/09-kimi-k3-${engine}-mpi.yaml"
  local allowed previous_hash="" new_hash

  [ -f "${template}" ] || { say "ERROR: ${template} not found"; return 1; }
  allowed="$(grep -m1 "^BASE_ALLOWED_VARS=" "${SCRIPT_DIR}/03_deploy_workloads.sh" \
             | sed "s/^BASE_ALLOWED_VARS='//; s/'$//")"
  [ -n "${allowed}" ] || { say "ERROR: could not read BASE_ALLOWED_VARS from 03"; return 1; }

  [ -f "${rendered}" ] && previous_hash="$(sha256sum "${rendered}" | cut -d' ' -f1)"
  mkdir -p "${GENERATED_DIR}"
  safe_envsubst "${allowed}" < "${template}" > "${rendered}"
  new_hash="$(sha256sum "${rendered}" | cut -d' ' -f1)"

  if [ "${previous_hash}" = "${new_hash}" ]; then
    say "${label}: manifest unchanged, no restart needed"
    return 0
  fi
  say "${label}: applying manifest and waiting for the rollout"
  kubectl apply -f "${rendered}" >> "${LOG_DIR}/${label}.log" 2>&1
  if ! kubectl rollout status "statefulset/${STATEFULSET}" -n "${NAMESPACE}" \
       --timeout="${READY_TIMEOUT}s" >> "${LOG_DIR}/${label}.log" 2>&1; then
    say "ERROR: ${label} rollout did not complete within ${READY_TIMEOUT}s"
    return 1
  fi
  sleep 60
  return 0
}

mirror() {
  local path="$1"
  [ -n "${RESULTS_BUCKET}" ] || return 0
  [ -s "${path}" ] || return 0
  gcloud storage cp "${path}" "${RESULTS_BUCKET}/${RUN_LABEL}/" >/dev/null 2>&1 \
    || say "WARNING: could not mirror $(basename "${path}") to ${RESULTS_BUCKET}"
}

# ------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------
stage_deploy() {
  say "STAGE deploy: cluster, image, weights, serving StatefulSet"
  say "  (GPU pool min is 0, so nothing bills until the StatefulSet lands)"
  bash "${SCRIPT_DIR}/01_setup_and_check.sh"      >> "${LOG_DIR}/01_setup.log" 2>&1
  AUTO_APPROVE=true \
  bash "${SCRIPT_DIR}/02_deploy_infra.sh"         >> "${LOG_DIR}/02_infra.log" 2>&1
  say "  infra up; GPU meter starts now"
  bash "${SCRIPT_DIR}/03_deploy_workloads.sh"     >> "${LOG_DIR}/03_workloads.log" 2>&1
  bash "${SCRIPT_DIR}/04_verify_cluster.sh"       >> "${LOG_DIR}/04_verify.log" 2>&1
  say "  cluster verified"
}

stage_sweep() {
  say "STAGE sweep: 9 variants through the pre-registered decision rule"
  local args=(
    --variants all
    --bench-mode realistic
    --primary-concurrency 16
    --state-dir "${SWEEP_STATE}"
    --ready-timeout "${READY_TIMEOUT}"
  )
  [ -n "${RESULTS_BUCKET}" ] && args+=(--results-bucket "${RESULTS_BUCKET}/${RUN_LABEL}/sweep")
  bash "${SCRIPT_DIR}/07_run_tuning_sweep.sh" "${args[@]}" 2>&1 \
    | tee -a "${LOG_DIR}/07_sweep.log"
}

# Accuracy before throughput, which is why this is not a sweep variant. Two
# bf16 captures establish the self-consistency floor -- the spread between two
# runs of the same config -- and fp8 is only meaningful measured against it.
# The two bf16 captures share one engine, so this leg costs a single restart.
stage_kv() {
  say "STAGE kv: fp8 KV accuracy, 3 captures, 1 restart"
  local label
  for label in bf16-a bf16-b; do
    if [ -e "${RESULTS_DIR}/kv_accuracy_${label}.json" ]; then
      say "  ${label} already captured, skipping"
      continue
    fi
    say "  capturing ${label}"
    KV_ACCURACY_LABEL="${label}" KV_ACCURACY_DTYPE="auto" \
      bash "${SCRIPT_DIR}/05_run_benchmarks.sh" --mode kv-accuracy --target serving \
      >> "${LOG_DIR}/kv_${label}.log" 2>&1
    mirror "${RESULTS_DIR}/kv_accuracy_${label}.json"
  done

  if [ ! -e "${RESULTS_DIR}/kv_accuracy_fp8.json" ]; then
    export SGLANG_KV_CACHE_DTYPE="fp8_e4m3"
    restart_with_env "kv-fp8"
    say "  capturing fp8"
    KV_ACCURACY_LABEL="fp8" KV_ACCURACY_DTYPE="fp8_e4m3" \
      bash "${SCRIPT_DIR}/05_run_benchmarks.sh" --mode kv-accuracy --target serving \
      >> "${LOG_DIR}/kv_fp8.log" 2>&1
    mirror "${RESULTS_DIR}/kv_accuracy_fp8.json"
    unset SGLANG_KV_CACHE_DTYPE
    restart_with_env "kv-restore"
  fi

  say "  reaching a verdict offline"
  python3 "${PROJECT_ROOT}/benchmarks/kv_accuracy_gate.py" compare \
    --baseline  "${RESULTS_DIR}/kv_accuracy_bf16-a.json" \
    --repeat    "${RESULTS_DIR}/kv_accuracy_bf16-b.json" \
    --candidate "${RESULTS_DIR}/kv_accuracy_fp8.json" \
    --output    "${STATE_ROOT}/kv_accuracy_verdict.json" \
    2>&1 | tee -a "${LOG_DIR}/kv_verdict.log" || true
  mirror "${STATE_ROOT}/kv_accuracy_verdict.json"
}

# C3. The open question: whether a third cache tier on NVMe is worth anything
# on a model where 69 of 93 layers are linear-attention and hold recurrent
# state rather than reusable KV. Two arms, one restart between them, and the
# results files are moved apart because 05 writes both to the same name.
stage_prefix() {
  say "STAGE prefix: hicache off vs on, 2 arms, 1 restart"
  local produced="${RESULTS_DIR}/prefix_reuse_results.json"

  if [ ! -e "${STATE_ROOT}/prefix_reuse_hicache-off.json" ]; then
    say "  arm 1/2: hicache off"
    export SGLANG_ENABLE_HIERARCHICAL_CACHE="false"
    restart_with_env "prefix-off"
    bash "${SCRIPT_DIR}/05_run_benchmarks.sh" --mode prefix-reuse --target serving \
      >> "${LOG_DIR}/prefix_off.log" 2>&1
    [ -s "${produced}" ] && mv "${produced}" "${STATE_ROOT}/prefix_reuse_hicache-off.json"
    mirror "${STATE_ROOT}/prefix_reuse_hicache-off.json"
  fi

  if [ ! -e "${STATE_ROOT}/prefix_reuse_hicache-on.json" ]; then
    say "  arm 2/2: hicache on, file backend on the NVMe array"
    export SGLANG_ENABLE_HIERARCHICAL_CACHE="true"
    export SGLANG_HICACHE_STORAGE_BACKEND="file"
    export SGLANG_HICACHE_IO_BACKEND="${SGLANG_HICACHE_IO_BACKEND:-kernel}"
    export SGLANG_HICACHE_RATIO="${SGLANG_HICACHE_RATIO:-2}"
    restart_with_env "prefix-on"
    bash "${SCRIPT_DIR}/05_run_benchmarks.sh" --mode prefix-reuse --target serving \
      >> "${LOG_DIR}/prefix_on.log" 2>&1
    [ -s "${produced}" ] && mv "${produced}" "${STATE_ROOT}/prefix_reuse_hicache-on.json"
    mirror "${STATE_ROOT}/prefix_reuse_hicache-on.json"
  fi
}

stage_collect() {
  say "STAGE collect: gathering everything before the cluster goes away"
  {
    echo "run_label: ${RUN_LABEL}"
    echo "started:   $(date -u -d "@${START_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)"
    echo "ended:     $(ts)"
    echo "engine:    ${INFERENCE_ENGINE:-sglang}"
    echo "stages:    ${STAGES}"
    echo
    echo "== sweep checkpoint =="
    cat "${SWEEP_STATE}/checkpoint.tsv" 2>/dev/null || echo "(none)"
    echo
    echo "== artifacts =="
    find "${STATE_ROOT}" "${RESULTS_DIR}" -maxdepth 1 -name '*.json' -newermt "@${START_EPOCH}" \
      -printf '%p\n' 2>/dev/null | sort || true
  } > "${STATE_ROOT}/MANIFEST.txt"
  cat "${STATE_ROOT}/MANIFEST.txt"
  if [ -n "${RESULTS_BUCKET}" ]; then
    gcloud storage cp -r "${STATE_ROOT}" "${RESULTS_BUCKET}/" >/dev/null 2>&1 \
      || say "WARNING: final mirror to ${RESULTS_BUCKET} failed; results are still on this VM"
  fi
}

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
echo "=============================================================================="
echo "Kimi K3 Measurement Window"
echo "=============================================================================="
say "run label     ${RUN_LABEL}"
say "cluster       ${CLUSTER_NAME} (${REGION:-?}/${ZONE:-?})"
say "stages        ${STAGES}"
say "budget        ${BUDGET_HOURS} h -> forced teardown at $(date -u -d "@${DEADLINE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)"
say "state         ${STATE_ROOT}"
echo "=============================================================================="

arm_watchdog

for stage in deploy sweep kv prefix collect; do
  stage_wanted "${stage}" || { say "skipping ${stage} (not in --stages)"; continue; }
  stage_done   "${stage}" && { say "skipping ${stage} (already complete)"; continue; }
  elapsed_note
  "stage_${stage}"
  mark_done "${stage}"
done

say "all requested stages complete"
# The EXIT trap tears down from here.
