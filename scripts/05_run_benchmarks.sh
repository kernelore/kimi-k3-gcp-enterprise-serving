#!/usr/bin/env bash
# ==============================================================================
# 05_run_benchmarks.sh - Execute Kimi K3 Sovereign Enterprise Inference Benchmarks
# ==============================================================================
# Executes standard, massive stress, and/or soak performance benchmarks against the
# Enterprise AI Gateway (port 4000) or directly against the TensorRT-LLM engine (port 8000).
# Automatically establishes a secure kubectl port-forward if running externally outside
# the private RoCEv2 VPC network. Supports in-cluster execution via Kubernetes Job.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"
TF_DIR="${PROJECT_ROOT}/terraform"
TEMPLATE_DIR="${TF_DIR}/manifests/templates"
GENERATED_DIR="${TF_DIR}/manifests/generated"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ${CONFIG_FILE} not found. Please run ./scripts/01_setup_and_check.sh first."
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

MODE="all"
TARGET="gateway"
CONCURRENCY=""
REQUESTS=""
IN_CLUSTER="false"

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

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Execute Kimi K3 performance benchmarks against the GKE serving stack.

Options:
  --mode <standard|massive|soak|prefill|saturation|realistic|prefix-reuse|kv-accuracy|failover|all>  Benchmark suite to run (default: all)
                                   - standard:   8 concurrent requests, 128 tokens
                                   - massive:    20 concurrent requests, 256 tokens (stress test)
                                   - soak:       30-minute continuous stability endurance test
                                   - prefill:    8192 prompt tokens ingestion stress test
                                   - saturation: ISL/OSL x concurrency sweep (1k-128k input, c=1..128) throughput ceiling
                                   - realistic:  repeated-passage vs non-repetitive corpus at a matched ISL,
                                                 to measure how much of the speculative-decoding gain survives
                                                 unpredictable prompts. Not part of 'all': it is a diagnostic,
                                                 its output is not one of the five audited suite files, and it
                                                 needs --metrics-endpoint to report acceptance per verify step.
                                   - prefix-reuse: shared-prefix TTFT against prefix length, cold vs warm vs
                                                 evicted-and-refetched. The only suite that can say whether the
                                                 radix cache, the NVMe cache tier or --schedule-policy lpm buy
                                                 anything on a model where 69 of 93 layers are linear-attention
                                                 and hold no reusable KV. Also a diagnostic, also not in 'all'.
                                   - kv-accuracy: one capture of the fp8 KV accuracy probe set. Answers whether
                                                 --kv-cache-dtype fp8_e4m3 costs correctness, which is settled
                                                 before its throughput is looked at. Reaching a verdict needs
                                                 three captures -- bf16 twice for the self-consistency floor,
                                                 then fp8 -- so set KV_ACCURACY_LABEL and KV_ACCURACY_DTYPE per
                                                 run and compare them offline afterwards with
                                                 'kv_accuracy_gate.py compare'. Also a diagnostic, not in 'all'.
                                   - failover:   kills a serving replica's rank 0 under steady load and measures
                                                 what the client saw: blackout, residual loss once service
                                                 returns, degraded p99, and how long the replica took to rejoin.
                                                 Needs a second replica to fail over to -- at DP=1 it measures
                                                 restart time, not failover, and will report a blackout in the
                                                 thousands of seconds. Destructive by design and not in 'all'.
                                   - all:        Run the 5 published benchmark suites sequentially
  --target <gateway|serving>     Target endpoint for benchmarking (default: gateway)
                                   - gateway: LiteLLM Enterprise Proxy (port 4000) with virtual keys & Redis auth
                                   - serving: Direct TensorRT-LLM Engine backend (port 8000) bypassing gateway
  --in-cluster                   Run benchmark as an in-cluster Kubernetes Job (Recommended for sustained/soak loads)
  --concurrency <N>              Override concurrency level (optional)
  --requests <N>                 Override total requests count (optional)
  -h, --help                     Show this usage guide and exit

Examples:
  ./scripts/05_run_benchmarks.sh --mode standard --target gateway
  ./scripts/05_run_benchmarks.sh --mode soak --in-cluster
  ./scripts/05_run_benchmarks.sh --mode massive --target serving --concurrency 16 --requests 64
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --in-cluster)
      IN_CLUSTER="true"
      shift 1
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --requests)
      REQUESTS="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      show_usage
      exit 1
      ;;
  esac
done

export SERVING_MODEL_NAME="${SERVING_MODEL_NAME:-moonshotai/Kimi-K3}"

echo "=============================================================================="
echo "Kimi K3 Sovereign Enterprise Inference - Benchmark Execution Suite"
echo "=============================================================================="
echo "Cluster:        ${CLUSTER_NAME} (${ZONE})"
echo "Mode:           ${MODE}"
echo "Target Layer:   ${TARGET}"
echo "Target Model:   ${SERVING_MODEL_NAME}"
echo "=============================================================================="

# 1. Verify prerequisites
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to run the benchmark scripts."
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl is required to verify cluster endpoints."
  exit 1
fi

# 2. Determine target service details
if [ "${TARGET}" = "gateway" ]; then
  SERVICE_NAME="kimi-k3-gateway-svc"
  REMOTE_PORT="4000"
  LOCAL_PORT="4000"
  ENDPOINT_PATH="/v1/chat/completions"
  DEV_KEY="${GATEWAY_MASTER_KEY:-sk-kimi-k3-master-secret-key-change-me}"
else
  SERVICE_NAME="kimi-k3-serving-svc"
  REMOTE_PORT="8000"
  LOCAL_PORT="8000"
  ENDPOINT_PATH="/v1/chat/completions"
  DEV_KEY="EMPTY"
fi

echo "--> 1. Resolving service VIP and connectivity for ${SERVICE_NAME} (port ${REMOTE_PORT})..."
VIP=$(kubectl get svc "${SERVICE_NAME}" -n llm-serving -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)

PF_PID=""
cleanup_port_forward() {
  if [ -n "${PF_PID}" ] && kill -0 "${PF_PID}" 2>/dev/null; then
    echo "    Cleaning up background kubectl port-forward (PID: ${PF_PID})..."
    kill "${PF_PID}" 2>/dev/null || true
  fi
}
trap cleanup_port_forward EXIT

if [ -n "${VIP}" ] && curl --connect-timeout 2 -s "http://${VIP}:${REMOTE_PORT}/health/liveliness" >/dev/null 2>&1; then
  echo "    [OK] Direct private VPC connection established to ${SERVICE_NAME} (${VIP}:${REMOTE_PORT})."
  BASE_URL="http://${VIP}:${REMOTE_PORT}"
elif [ -n "${VIP}" ] && curl --connect-timeout 2 -s "http://${VIP}:${REMOTE_PORT}/health" >/dev/null 2>&1; then
  echo "    [OK] Direct private VPC connection established to ${SERVICE_NAME} (${VIP}:${REMOTE_PORT})."
  BASE_URL="http://${VIP}:${REMOTE_PORT}"
else
  echo "    NOTE: Direct connection to private VIP (${VIP:-Unassigned}) not reachable from local workstation."
  echo "    --> Establishing automated kubectl port-forward (${LOCAL_PORT}:${REMOTE_PORT}) across private fabric..."
  kubectl port-forward -n llm-serving "svc/${SERVICE_NAME}" "${LOCAL_PORT}:${REMOTE_PORT}" >/dev/null 2>&1 &
  PF_PID=$!
  sleep 3
  if ! kill -0 "${PF_PID}" 2>/dev/null; then
    echo "ERROR: Failed to establish kubectl port-forward to svc/${SERVICE_NAME}. Please verify pod health."
    exit 1
  fi
  BASE_URL="http://localhost:${LOCAL_PORT}"
  echo "    [OK] Port-forward active on ${BASE_URL} (bypassing external network restrictions)."
fi

TARGET_URL="${BASE_URL}${ENDPOINT_PATH}"
echo "    Target Benchmark Endpoint: ${TARGET_URL}"

ENGINE="${INFERENCE_ENGINE:-sglang}"
ENGINE_VERSION="unknown"
# Probe the engine version in every mode. This used to be skipped whenever --in-cluster was
# set, which left engine_version at the literal string "unknown" on exactly those runs -- and
# benchmarks/generate_comparison.py rejects "unknown" at the provenance gate as a placeholder.
# The probe is a host-side `kubectl exec` and has nothing to do with where the benchmark
# traffic originates, so gating it on IN_CLUSTER only ever discarded valid provenance.
SERVING_POD=$(kubectl get pod -n llm-serving -l app=kimi-k3-serving -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "${SERVING_POD}" ]; then
  if [ "${ENGINE}" = "sglang" ]; then
    ENGINE_VERSION=$(kubectl exec -n llm-serving "${SERVING_POD}" -c sglang-mpi-node -- python3 -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
  else
    ENGINE_VERSION=$(kubectl exec -n llm-serving "${SERVING_POD}" -c trtllm-mpi-node -- python3 -c "import tensorrt_llm; print(getattr(tensorrt_llm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
  fi
fi
if [ -z "${ENGINE_VERSION}" ]; then
  ENGINE_VERSION="unknown"
fi

if [ -z "${SERVING_IMAGE:-}" ]; then
  SERVING_IMAGE=$(kubectl get pod -n llm-serving -l app=kimi-k3-serving -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null || echo "unknown")
  if [ -z "${SERVING_IMAGE}" ]; then SERVING_IMAGE="unknown"; fi
fi
get_metadata_json() {
  local ts tp pp ep
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  if [ "${ENGINE}" = "trtllm" ]; then
    tp=${TRTLLM_TP_SIZE:-8}
    pp=${TRTLLM_PP_SIZE:-2}
    ep=${TRTLLM_EP_SIZE:-8}
  else
    tp=${SGLANG_TP_SIZE:-16}
    pp=${SGLANG_PP_SIZE:-1}
    ep=${SGLANG_EP_SIZE:-16}
  fi
  echo "{\"engine\": \"${ENGINE}\", \"version\": \"${ENGINE_VERSION}\", \"engine_version\": \"${ENGINE_VERSION}\", \"image\": \"${SERVING_IMAGE}\", \"run_timestamp\": \"${ts}\", \"tp\": ${tp}, \"pp\": ${pp}, \"ep\": ${ep}, \"nodes\": 2, \"gpus\": 16}"
}

RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results/${ENGINE}"
mkdir -p "${RESULTS_DIR}"
echo "    Active Engine: ${ENGINE} (version: ${ENGINE_VERSION})"
echo "    Results Dir:   ${RESULTS_DIR}"

# Check for In-Cluster execution mode
if [ "${IN_CLUSTER}" = "true" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  echo "--> Executing In-Cluster Benchmark via Kubernetes Job & ConfigMap..."
  echo "------------------------------------------------------------------------------"

  if [ "${MODE}" = "all" ]; then
    echo "ERROR: '--mode all' is not supported with '--in-cluster'. Please specify a single benchmark mode (e.g., '--mode soak', '--mode standard', '--mode massive', '--mode prefill', or '--mode saturation')."
    exit 1
  fi

  # BENCH_LABEL only reaches the banner the Job echoes before it starts. The template is
  # written for soak and every other mode is patched into it below, so without this the
  # log of a saturation or prefill run opened with "Soak Endurance Test" and read as the
  # wrong suite to anyone tailing it.
  BENCH_SCRIPT=""
  BENCH_LABEL=""
  case "${MODE}" in
    soak)       BENCH_SCRIPT="soak_benchmark_kimi_k3.py";         BENCH_LABEL="Soak Endurance Test" ;;
    standard)   BENCH_SCRIPT="benchmark_kimi_k3.py";              BENCH_LABEL="Standard Suite" ;;
    massive)    BENCH_SCRIPT="massive_benchmark_kimi_k3.py";      BENCH_LABEL="Massive Stress Suite" ;;
    prefill)    BENCH_SCRIPT="run_prefill_benchmark_kimi_k3.py";  BENCH_LABEL="Prefill Ingestion Suite" ;;
    saturation) BENCH_SCRIPT="run_saturation_sweep_kimi_k3.py";   BENCH_LABEL="Saturation Sweep" ;;
    realistic)  BENCH_SCRIPT="run_realistic_sweep_kimi_k3.py";    BENCH_LABEL="Prompt Sensitivity Sweep" ;;
    prefix-reuse) BENCH_SCRIPT="run_prefix_reuse_bench.py";       BENCH_LABEL="Prefix Reuse Bench" ;;
    kv-accuracy)
      # The Job template invokes `python3 <script> --endpoint=... --output=...`
      # and this mode's CLI takes a `capture` subcommand before its flags, so
      # the sed substitution above would emit a Job that fails on startup.
      # Saying so is better than shipping one that dies after the pull.
      echo "ERROR: '--mode kv-accuracy' cannot run in-cluster: the Job template has no slot for the 'capture' subcommand." >&2
      echo "       Run it from the host against the gateway or serving endpoint instead." >&2
      exit 1
      ;;
    failover)
      # Two reasons, either one sufficient. The Job template has no slot for this mode's
      # 'measure' subcommand, same as kv-accuracy above; and the benchmark ServiceAccount
      # has no permission to delete pods, which is the entire point of the mode. A Job
      # that fails on an RBAC denial after pulling a multi-gigabyte image is a worse
      # outcome than refusing here.
      echo "ERROR: '--mode failover' cannot run in-cluster: the Job template has no slot for the 'measure'" >&2
      echo "       subcommand, and the benchmark ServiceAccount cannot delete pods." >&2
      echo "       Run it from the host, where kubectl already has the credentials it needs." >&2
      exit 1
      ;;
    *)
      echo "ERROR: Unsupported mode '${MODE}' for --in-cluster benchmark."
      exit 1
      ;;
  esac

  echo "    Creating/Updating benchmark ConfigMap with local test scripts..."
  
  FROM_FILES=()
  for py_file in "${PROJECT_ROOT}"/benchmarks/*.py; do
    if [ -f "${py_file}" ]; then
      FROM_FILES+=("--from-file=${py_file}")
    fi
  done

  if [ "${#FROM_FILES[@]}" -eq 0 ]; then
    echo "ERROR: No python benchmark scripts found in ${PROJECT_ROOT}/benchmarks/."
    exit 1
  fi

  kubectl create configmap kimi-k3-benchmark-scripts \
    "${FROM_FILES[@]}" \
    -n llm-serving --dry-run=client -o yaml | kubectl apply -f -

  echo "    Rendering in-cluster benchmark Job manifest..."
  mkdir -p "${GENERATED_DIR}"
  export ENV_LABEL="${ENV_LABEL:-kimi-k3-prod}"
  export OWNER_LABEL="${OWNER_LABEL:-opensource-user}"
  export GATEWAY_MASTER_KEY="${GATEWAY_MASTER_KEY:-sk-kimi-k3-master-secret-key-change-me}"
  
  # Stamp the same provenance block the host-side runs pass via --metadata. Without this the
  # Job leaves the flag at its "{}" default, every in-cluster result lands with an empty
  # metadata object, and benchmarks/generate_comparison.py rejects the run at the provenance
  # gate -- so the fallback this script recommends when the port-forward tunnel drops could
  # never actually produce the README comparison table.
  BENCHMARK_METADATA_JSON="$(get_metadata_json)"
  export BENCHMARK_METADATA_JSON

  # Honour --target in the in-cluster path too. The Job used to hardcode the gateway VIP, so
  # `--target serving` silently benchmarked the gateway instead of the engine -- and the
  # methodology note benchmarks/generate_comparison.py writes into the README states that the
  # Saturation Sweep and Prefill Ingestion suites measure the engine directly on port 8000.
  # In-cluster traffic resolves the ClusterIP by service name, so no port-forward is involved.
  if [ "${TARGET}" = "gateway" ]; then
    BENCHMARK_ENDPOINT="http://kimi-k3-gateway-svc:4000/v1/completions"
  else
    BENCHMARK_ENDPOINT="http://kimi-k3-serving-svc:8000/v1/completions"
  fi
  export BENCHMARK_ENDPOINT
  # Every harness defaults --engine to "trtllm". Left unset, an SGLang run wrote
  # "engine": "trtllm" into results/sglang/*.json, contradicting its own metadata block.
  export BENCHMARK_ENGINE="${ENGINE}"

  if [ -f "${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template" ]; then
    # shellcheck disable=SC2016
    safe_envsubst '${ENV_LABEL} ${OWNER_LABEL} ${SERVING_MODEL_NAME} ${BENCHMARK_METADATA_JSON} ${BENCHMARK_ENDPOINT} ${BENCHMARK_ENGINE}' < "${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template" > "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
    if [ "${MODE}" != "soak" ]; then
      echo "    Configuring in-cluster job for mode '${MODE}' (${BENCH_SCRIPT})..."
      sed -i "s/soak_benchmark_kimi_k3\.py/${BENCH_SCRIPT}/g" "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
      sed -i "s/incluster_soak_results\.json/incluster_${MODE}_results.json/g" "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
      sed -i "s/Soak Endurance Test/${BENCH_LABEL}/g" "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
    fi
    echo "    Applying in-cluster benchmark Job (${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml)..."
    kubectl delete job kimi-k3-incluster-benchmark -n llm-serving --ignore-not-found=true
    kubectl apply -f "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
  else
    echo "WARNING: ${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template not found."
    exit 1
  fi

  # Keep these in step with activeDeadlineSeconds in the Job template. The saturation sweep
  # runs 13 grid cells -- the c=1 cells alone issue 8 strictly sequential requests of up to
  # 2048 output tokens on top of a 128k-token prefill -- and comfortably outruns the 1200s
  # that used to apply to every mode except soak. When `kubectl wait` gave up, this script
  # exited non-zero and the next invocation deleted the still-running Job, so the sweep could
  # never complete no matter how long it was left alone.
  case "${MODE}" in
    soak)       TIMEOUT="3600s" ;;
    saturation) TIMEOUT="10800s" ;;
    # Two arms over the same grid, so budget like a small saturation sweep.
    # 10800s is the Job template's activeDeadlineSeconds, which is the real
    # ceiling either way -- there is no point waiting past it.
    realistic)  TIMEOUT="10800s" ;;
    # The evicted arm prefills a full KV pool per sample, so this one is bounded
    # by flush cost rather than by request count.
    prefix-reuse) TIMEOUT="10800s" ;;
    *)          TIMEOUT="1200s" ;;
  esac
  echo "    Waiting for in-cluster benchmark Job to complete (timeout: ${TIMEOUT})..."
  if ! kubectl wait --for=condition=complete job/kimi-k3-incluster-benchmark -n llm-serving --timeout="${TIMEOUT}"; then
    echo "ERROR: In-cluster benchmark Job failed or timed out!" >&2
    kubectl logs -n llm-serving -l app=kimi-k3-benchmark --tail=50 >&2 || true
    exit 1
  fi

  echo "    Retrieving in-cluster benchmark results..."
  BENCH_POD=$(kubectl get pod -n llm-serving -l app=kimi-k3-benchmark -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  mkdir -p "${RESULTS_DIR}"
  RESULT_FILE="${RESULTS_DIR}/incluster_${MODE}_results.json"
  if [ -n "${BENCH_POD}" ]; then
    if ! kubectl cp -n llm-serving "${BENCH_POD}:/tmp/results/" "${RESULTS_DIR}/" 2>/dev/null; then
      echo "    [INFO] kubectl cp failed; extracting JSON result from pod logs..."
      kubectl logs -n llm-serving "${BENCH_POD}" | awk '/=== JSON_RESULT_START ===/{flag=1; next} /=== JSON_RESULT_END ===/{flag=0} flag' > "${RESULTS_DIR}/incluster_${MODE}_results.json" || true
      if [ ! -s "${RESULTS_DIR}/incluster_${MODE}_results.json" ]; then
        echo "ERROR: Extracted benchmark JSON from pod logs is empty." >&2
        rm -f "${RESULTS_DIR}/incluster_${MODE}_results.json"
        exit 1
      fi
    fi
    res_file="${RESULTS_DIR}/incluster_${MODE}_results.json"
    if [ -f "${res_file}" ]; then
      if ! python3 -m json.tool "${res_file}" >/dev/null 2>&1; then
        echo "ERROR: Benchmark result file ${res_file} contains invalid JSON." >&2
        exit 1
      fi
    fi
    if [ ! -s "${RESULT_FILE}" ]; then
      echo "ERROR: Retrieved benchmark result file ${RESULT_FILE} is missing or empty." >&2
      exit 1
    fi
    if ! python3 -m json.tool "${RESULT_FILE}" >/dev/null 2>&1; then
      echo "ERROR: Benchmark result file ${RESULT_FILE} is not valid JSON!" >&2
      exit 1
    fi

    # Publish under the canonical suite name as well. generate_comparison.py globs for
    # "<suite>_results.json" and is blind to the "incluster_" prefix, so an in-cluster run
    # used to leave the results directory looking empty to the comparison generator no matter
    # how many suites had actually been executed.
    CANONICAL_RESULT_FILE="${RESULTS_DIR}/${MODE}_results.json"
    cp -f "${RESULT_FILE}" "${CANONICAL_RESULT_FILE}"
    echo "    Published canonical result: ${CANONICAL_RESULT_FILE}"
  fi
  echo "    [OK] In-cluster benchmark job execution finished and results retrieved."
  exit 0
fi

# 3. Execute Standard Benchmark Suite
if [ "${MODE}" = "standard" ] || [ "${MODE}" = "all" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 2. Executing Standard Enterprise Benchmark Suite (concurrency=8, requests=16)..."
  echo "------------------------------------------------------------------------------"
  sleep 1
  
  STD_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/standard_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  if [ -n "${CONCURRENCY}" ]; then STD_ARGS+=("--concurrency=${CONCURRENCY}"); fi
  if [ -n "${REQUESTS}" ]; then STD_ARGS+=("--requests=${REQUESTS}"); fi
  
  if [ -f "${PROJECT_ROOT}/benchmarks/benchmark_kimi_k3.py" ]; then
    python3 "${PROJECT_ROOT}/benchmarks/benchmark_kimi_k3.py" "${STD_ARGS[@]}" || echo "WARNING: Standard benchmark reported errors or timeouts."
  fi
fi

# 4. Execute Massive Stress Benchmark Suite
if [ "${MODE}" = "massive" ] || [ "${MODE}" = "all" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 3. Executing Massive Stress Benchmark Suite (concurrency=20, requests=100)..."
  echo "------------------------------------------------------------------------------"
  sleep 1
  
  MAS_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/massive_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  if [ -n "${CONCURRENCY}" ]; then MAS_ARGS+=("--concurrency=${CONCURRENCY}"); fi
  if [ -n "${REQUESTS}" ]; then MAS_ARGS+=("--requests=${REQUESTS}"); fi
  
  if [ -f "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" ]; then
    python3 "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "${MAS_ARGS[@]}" || echo "WARNING: Massive benchmark reported errors or timeouts."
  fi
fi

# 5. Execute Continuous Soak Benchmark Suite
if [ "${MODE}" = "soak" ] || [ "${MODE}" = "all" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 4. Executing Continuous Soak Suite (concurrency=18, duration=1800s)..."
  echo "------------------------------------------------------------------------------"
  sleep 1
  
  SOAK_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/soak_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  if [ -n "${CONCURRENCY}" ]; then SOAK_ARGS+=("--concurrency=${CONCURRENCY}"); fi
  
  if [ -f "${PROJECT_ROOT}/benchmarks/soak_benchmark_kimi_k3.py" ]; then
    python3 "${PROJECT_ROOT}/benchmarks/soak_benchmark_kimi_k3.py" "${SOAK_ARGS[@]}" || echo "WARNING: Soak benchmark reported errors or timeouts."
  fi
fi

# 6. Execute Prefill Benchmark Suite
if [ "${MODE}" = "prefill" ] || [ "${MODE}" = "all" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 5. Executing Prefill Benchmark Suite..."
  echo "------------------------------------------------------------------------------"
  sleep 1
  
  PREFILL_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/prefill_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  if [ -f "${PROJECT_ROOT}/benchmarks/run_prefill_benchmark_kimi_k3.py" ]; then
    python3 "${PROJECT_ROOT}/benchmarks/run_prefill_benchmark_kimi_k3.py" "${PREFILL_ARGS[@]}" || echo "WARNING: Prefill benchmark reported errors or timeouts."
  fi
fi

# 7. Execute Saturation Sweep Benchmark Suite
if [ "${MODE}" = "saturation" ] || [ "${MODE}" = "all" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 6. Executing Saturation Sweep Suite..."
  echo "------------------------------------------------------------------------------"
  sleep 1
  
  SAT_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/saturation_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  [ -n "${SWEEP_METRICS_ENDPOINT:-}" ] && SAT_ARGS+=("--metrics-endpoint=${SWEEP_METRICS_ENDPOINT}")
  [ -n "${SWEEP_METRICS_NAMES:-}" ]    && SAT_ARGS+=("--metrics-names=${SWEEP_METRICS_NAMES}")
  [ -n "${SWEEP_CONCURRENCY_LEVELS:-}" ] && SAT_ARGS+=("--concurrency-levels=${SWEEP_CONCURRENCY_LEVELS}")
  [ -n "${SWEEP_MAX_INFLIGHT:-}" ]       && SAT_ARGS+=("--max-inflight-prompt-tokens=${SWEEP_MAX_INFLIGHT}")
  if [ -f "${PROJECT_ROOT}/benchmarks/run_saturation_sweep_kimi_k3.py" ]; then
    python3 "${PROJECT_ROOT}/benchmarks/run_saturation_sweep_kimi_k3.py" "${SAT_ARGS[@]}" || echo "WARNING: Saturation sweep reported errors or timeouts."
  fi
fi

# Deliberately excluded from 'all'. This suite runs every cell twice, once per
# prompt arm, and its output is not one of the five files the provenance audit
# checks -- it exists to qualify the DSPARK numbers, not to add to them.
if [ "${MODE}" = "realistic" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 7. Executing Prompt Sensitivity Sweep (repeated passage vs non-repetitive corpus)..."
  echo "------------------------------------------------------------------------------"
  sleep 1

  REAL_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/realistic_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  [ -n "${SWEEP_METRICS_ENDPOINT:-}" ]    && REAL_ARGS+=("--metrics-endpoint=${SWEEP_METRICS_ENDPOINT}")
  [ -n "${SWEEP_METRICS_NAMES:-}" ]       && REAL_ARGS+=("--metrics-names=${SWEEP_METRICS_NAMES}")
  [ -n "${SWEEP_CONCURRENCY_LEVELS:-}" ]  && REAL_ARGS+=("--concurrency-levels=${SWEEP_CONCURRENCY_LEVELS}")
  [ -n "${SWEEP_MAX_INFLIGHT:-}" ]        && REAL_ARGS+=("--max-inflight-prompt-tokens=${SWEEP_MAX_INFLIGHT}")
  [ -n "${REALISTIC_ISL_OSL_GRID:-}" ]    && REAL_ARGS+=("--isl-osl-grid=${REALISTIC_ISL_OSL_GRID}")
  [ -n "${REALISTIC_ARMS:-}" ]            && REAL_ARGS+=("--arms=${REALISTIC_ARMS}")
  [ -n "${REALISTIC_ISSUANCE:-}" ]        && REAL_ARGS+=("--issuance=${REALISTIC_ISSUANCE}")
  [ -n "${REALISTIC_CORPUS_SEED:-}" ]     && REAL_ARGS+=("--corpus-seed=${REALISTIC_CORPUS_SEED}")
  [ -n "${REALISTIC_CORPUS_FILE:-}" ]     && REAL_ARGS+=("--corpus-file=${REALISTIC_CORPUS_FILE}")
  [ -n "${REALISTIC_SPEC_VERIFY_METRIC:-}" ] && REAL_ARGS+=("--spec-verify-metric=${REALISTIC_SPEC_VERIFY_METRIC}")

  if [ -z "${SWEEP_METRICS_ENDPOINT:-}" ]; then
    echo "    NOTE: SWEEP_METRICS_ENDPOINT is unset, so accepted tokens per verify"
    echo "          step cannot be measured and the headline result of this sweep"
    echo "          will be null. Set it in scripts/config.env before spending GPU time."
  fi

  if [ -f "${PROJECT_ROOT}/benchmarks/run_realistic_sweep_kimi_k3.py" ]; then
    echo "    Verifying the committed corpus is still non-repetitive before spending GPU time..."
    if ! python3 "${PROJECT_ROOT}/benchmarks/run_realistic_sweep_kimi_k3.py" --self-check \
         ${REALISTIC_ISL_OSL_GRID:+"--isl-osl-grid=${REALISTIC_ISL_OSL_GRID}"} \
         ${SWEEP_CONCURRENCY_LEVELS:+"--concurrency-levels=${SWEEP_CONCURRENCY_LEVELS}"} \
         ${REALISTIC_CORPUS_SEED:+"--corpus-seed=${REALISTIC_CORPUS_SEED}"} \
         ${REALISTIC_CORPUS_FILE:+"--corpus-file=${REALISTIC_CORPUS_FILE}"}; then
      echo "ERROR: Corpus self-check failed; refusing to run the sweep." >&2
      exit 1
    fi
    python3 "${PROJECT_ROOT}/benchmarks/run_realistic_sweep_kimi_k3.py" "${REAL_ARGS[@]}" || echo "WARNING: Prompt sensitivity sweep reported errors or timeouts."
  fi
fi

# Also excluded from 'all'. Unlike every other suite here this one wants prefix
# cache hits rather than avoiding them, so running it alongside the published
# suites would leave their prompts competing for the same KV pool.
if [ "${MODE}" = "prefix-reuse" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 8. Executing Prefix Reuse Bench (cold vs warm vs evicted prefix TTFT)..."
  echo "------------------------------------------------------------------------------"
  sleep 1

  PFX_ARGS=(
    "--endpoint=${TARGET_URL}"
    "--output=${RESULTS_DIR}/prefix_reuse_results.json"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--metadata=$(get_metadata_json)"
  )
  [ -n "${SWEEP_METRICS_ENDPOINT:-}" ]      && PFX_ARGS+=("--metrics-endpoint=${SWEEP_METRICS_ENDPOINT}")
  [ -n "${PREFIX_REUSE_ARMS:-}" ]           && PFX_ARGS+=("--arms=${PREFIX_REUSE_ARMS}")
  [ -n "${PREFIX_REUSE_PREFIX_TOKENS:-}" ]  && PFX_ARGS+=("--prefix-tokens=${PREFIX_REUSE_PREFIX_TOKENS}")
  [ -n "${PREFIX_REUSE_SUFFIX_TOKENS:-}" ]  && PFX_ARGS+=("--suffix-tokens=${PREFIX_REUSE_SUFFIX_TOKENS}")
  [ -n "${PREFIX_REUSE_REPEATS:-}" ]        && PFX_ARGS+=("--repeats=${PREFIX_REUSE_REPEATS}")
  [ -n "${PREFIX_REUSE_EVICT_REPEATS:-}" ]  && PFX_ARGS+=("--evict-repeats=${PREFIX_REUSE_EVICT_REPEATS}")
  [ -n "${PREFIX_REUSE_TARGET_HIT_RATE:-}" ] && PFX_ARGS+=("--target-hit-rate=${PREFIX_REUSE_TARGET_HIT_RATE}")
  [ -n "${PREFIX_REUSE_KV_POOL_TOKENS:-}" ] && PFX_ARGS+=("--kv-pool-tokens=${PREFIX_REUSE_KV_POOL_TOKENS}")
  [ -n "${PREFIX_REUSE_MAX_TOTAL_FLUSH:-}" ] && PFX_ARGS+=("--max-total-flush-tokens=${PREFIX_REUSE_MAX_TOTAL_FLUSH}")

  if [ -f "${PROJECT_ROOT}/benchmarks/run_prefix_reuse_bench.py" ]; then
    echo "    Verifying prefixes are shared exactly and suffixes are not..."
    if ! python3 "${PROJECT_ROOT}/benchmarks/run_prefix_reuse_bench.py" --self-check \
         ${PREFIX_REUSE_PREFIX_TOKENS:+"--prefix-tokens=${PREFIX_REUSE_PREFIX_TOKENS}"} \
         ${PREFIX_REUSE_SUFFIX_TOKENS:+"--suffix-tokens=${PREFIX_REUSE_SUFFIX_TOKENS}"} \
         ${PREFIX_REUSE_KV_POOL_TOKENS:+"--kv-pool-tokens=${PREFIX_REUSE_KV_POOL_TOKENS}"} \
         ${PREFIX_REUSE_EVICT_REPEATS:+"--evict-repeats=${PREFIX_REUSE_EVICT_REPEATS}"} \
         ${PREFIX_REUSE_MAX_TOTAL_FLUSH:+"--max-total-flush-tokens=${PREFIX_REUSE_MAX_TOTAL_FLUSH}"}; then
      echo "ERROR: Prefix self-check failed; refusing to run the bench." >&2
      exit 1
    fi
    python3 "${PROJECT_ROOT}/benchmarks/run_prefix_reuse_bench.py" "${PFX_ARGS[@]}" || echo "WARNING: Prefix reuse bench reported errors or timeouts."
  fi
fi

if [ "${MODE}" = "kv-accuracy" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 9. Capturing fp8 KV accuracy probe set..."
  echo "------------------------------------------------------------------------------"
  sleep 1

  # The label is what tells one capture from another, and comparing a capture
  # against itself would report a perfect score for any dtype. Refuse rather
  # than overwrite: each of these costs a full pass over 50 long-context probes
  # on a running cluster, and two of the three need a StatefulSet restart
  # between them.
  KV_LABEL="${KV_ACCURACY_LABEL:-}"
  if [ -z "${KV_LABEL}" ]; then
    echo "ERROR: KV_ACCURACY_LABEL must be set (e.g. bf16-a, bf16-b, fp8) so captures can be told apart." >&2
    exit 1
  fi
  KV_OUT="${RESULTS_DIR}/kv_accuracy_${KV_LABEL}.json"
  if [ -e "${KV_OUT}" ]; then
    echo "ERROR: ${KV_OUT} already exists; refusing to overwrite a capture. Use a different KV_ACCURACY_LABEL." >&2
    exit 1
  fi

  KVA_ARGS=(
    "capture"
    "--endpoint=${TARGET_URL}"
    "--output=${KV_OUT}"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--engine=${ENGINE}"
    "--label=${KV_LABEL}"
  )
  [ -n "${KV_ACCURACY_DTYPE:-}" ]           && KVA_ARGS+=("--kv-cache-dtype=${KV_ACCURACY_DTYPE}")
  [ -n "${KV_ACCURACY_CONTEXT_TOKENS:-}" ]  && KVA_ARGS+=("--context-tokens=${KV_ACCURACY_CONTEXT_TOKENS}")
  [ -n "${KV_ACCURACY_DEPTHS:-}" ]          && KVA_ARGS+=("--depths=${KV_ACCURACY_DEPTHS}")
  [ -n "${KV_ACCURACY_PROBES_PER_DEPTH:-}" ] && KVA_ARGS+=("--probes-per-depth=${KV_ACCURACY_PROBES_PER_DEPTH}")
  [ -n "${KV_ACCURACY_MAX_TOKENS:-}" ]      && KVA_ARGS+=("--max-tokens=${KV_ACCURACY_MAX_TOKENS}")
  [ -n "${KV_ACCURACY_CORPUS_SEED:-}" ]     && KVA_ARGS+=("--corpus-seed=${KV_ACCURACY_CORPUS_SEED}")

  if [ -z "${KV_ACCURACY_DTYPE:-}" ]; then
    echo "    NOTE: KV_ACCURACY_DTYPE is unset, so this capture will not record which KV dtype produced it."
  fi

  if [ -f "${PROJECT_ROOT}/benchmarks/kv_accuracy_gate.py" ]; then
    echo "    Verifying each needle is planted once at the depth it claims..."
    if ! python3 "${PROJECT_ROOT}/benchmarks/kv_accuracy_gate.py" capture --label="${KV_LABEL}" --self-check \
         ${KV_ACCURACY_CONTEXT_TOKENS:+"--context-tokens=${KV_ACCURACY_CONTEXT_TOKENS}"} \
         ${KV_ACCURACY_DEPTHS:+"--depths=${KV_ACCURACY_DEPTHS}"} \
         ${KV_ACCURACY_PROBES_PER_DEPTH:+"--probes-per-depth=${KV_ACCURACY_PROBES_PER_DEPTH}"} \
         ${KV_ACCURACY_CORPUS_SEED:+"--corpus-seed=${KV_ACCURACY_CORPUS_SEED}"}; then
      echo "ERROR: Probe self-check failed; refusing to run the capture." >&2
      exit 1
    fi
    python3 "${PROJECT_ROOT}/benchmarks/kv_accuracy_gate.py" "${KVA_ARGS[@]}" || echo "WARNING: KV accuracy capture reported errors or timeouts."
    echo ""
    echo "    Capture written. Once bf16-a, bf16-b and fp8 all exist, reach a verdict with:"
    echo "      python3 benchmarks/kv_accuracy_gate.py compare \\"
    echo "        --baseline ${RESULTS_DIR}/kv_accuracy_bf16-a.json \\"
    echo "        --repeat   ${RESULTS_DIR}/kv_accuracy_bf16-b.json \\"
    echo "        --candidate ${RESULTS_DIR}/kv_accuracy_fp8.json"
  fi
fi

if [ "${MODE}" = "failover" ]; then
  echo ""
  echo "------------------------------------------------------------------------------"
  sleep 1
  echo "--> 10. Measuring replica failover under steady load..."
  echo "------------------------------------------------------------------------------"
  sleep 1

  FAILOVER_LABEL="${FAILOVER_LABEL:-failover}"
  FO_OUT="${RESULTS_DIR}/failover_${FAILOVER_LABEL}.json"
  # Same reasoning as the KV captures: this one costs a replica restart to repeat, so an
  # accidental second run must not quietly overwrite the first.
  if [ -e "${FO_OUT}" ]; then
    echo "ERROR: ${FO_OUT} already exists; refusing to overwrite. Use a different FAILOVER_LABEL." >&2
    exit 1
  fi

  # Default to rank 0 of the primary replica. Rank 0 is the whole replica's serving surface
  # because kimi-k3-serving-svc selects on apps.kubernetes.io/pod-index: "0", so this is the
  # worst realistic single-pod loss rather than an arbitrary one.
  FAILOVER_TARGET_POD="${FAILOVER_TARGET_POD:-kimi-k3-serving-0}"
  FAILOVER_KILL_CMD="${FAILOVER_KILL_CMD:-kubectl delete pod ${FAILOVER_TARGET_POD} -n llm-serving --wait=false}"
  # --wait=false matters: the harness needs the kill to return immediately so T0 is the
  # instant the pod was told to go, not the instant Kubernetes finished reaping it.

  # Report the replica as back only once it is Ready, not merely Running. A pod that has
  # restarted but is still loading 2.8T of weights answers nothing.
  FAILOVER_REJOIN_CMD="${FAILOVER_REJOIN_CMD:-kubectl get pod ${FAILOVER_TARGET_POD} -n llm-serving -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'}"

  if [ "${TARGET}" != "gateway" ]; then
    # Pointing this at a single replica measures that replica dying. The question is whether
    # the *service* survives, and only the gateway can answer it.
    echo "    WARNING: --target ${TARGET} bypasses the gateway, so there is nothing to fail over to."
    echo "             The result will describe one replica restarting, not failover."
  fi

  FO_ARGS=(
    "measure"
    "--endpoint=${TARGET_URL}"
    "--output=${FO_OUT}"
    "--api-key=${DEV_KEY}"
    "--model=${SERVING_MODEL_NAME}"
    "--label=${FAILOVER_LABEL}"
    "--kill-command=${FAILOVER_KILL_CMD}"
    "--rejoin-command=${FAILOVER_REJOIN_CMD}"
    "--rejoin-expect=True"
    "--keep-samples"
  )
  [ -n "${FAILOVER_RPS:-}" ]               && FO_ARGS+=("--rps=${FAILOVER_RPS}")
  [ -n "${FAILOVER_BASELINE_SECONDS:-}" ]  && FO_ARGS+=("--baseline-seconds=${FAILOVER_BASELINE_SECONDS}")
  [ -n "${FAILOVER_RECOVERY_SECONDS:-}" ]  && FO_ARGS+=("--recovery-seconds=${FAILOVER_RECOVERY_SECONDS}")
  [ -n "${FAILOVER_MAX_INFLIGHT:-}" ]      && FO_ARGS+=("--max-inflight=${FAILOVER_MAX_INFLIGHT}")

  if [ -f "${PROJECT_ROOT}/benchmarks/run_failover_bench.py" ]; then
    echo "    Validating the run and bounding the kill command before anything is deleted..."
    if ! python3 "${PROJECT_ROOT}/benchmarks/run_failover_bench.py" measure --self-check \
         "--endpoint=${TARGET_URL}" \
         "--kill-command=${FAILOVER_KILL_CMD}" \
         ${FAILOVER_RPS:+"--rps=${FAILOVER_RPS}"} \
         ${FAILOVER_BASELINE_SECONDS:+"--baseline-seconds=${FAILOVER_BASELINE_SECONDS}"} \
         ${FAILOVER_RECOVERY_SECONDS:+"--recovery-seconds=${FAILOVER_RECOVERY_SECONDS}"}; then
      echo "ERROR: Failover self-check failed; refusing to kill anything." >&2
      exit 1
    fi
    echo "    Kill target: ${FAILOVER_TARGET_POD}"
    python3 "${PROJECT_ROOT}/benchmarks/run_failover_bench.py" "${FO_ARGS[@]}" || echo "WARNING: Failover measurement reported errors or timeouts."
    echo ""
    echo "    Apply the pre-registered rule with:"
    echo "      python3 benchmarks/run_failover_bench.py verdict --input ${FO_OUT}"
  else
    echo "ERROR: benchmarks/run_failover_bench.py not found." >&2
    exit 1
  fi
fi

# 8. Display Benchmark Summary (with 16x B200 GPU normalization factor)
echo ""
echo "=============================================================================="
echo "Benchmark Execution Summary (Engine: ${ENGINE} ${ENGINE_VERSION} on 16x B200 HGX Pool)"
echo "=============================================================================="
if [ -s "${RESULTS_DIR}/standard_results.json" ]; then
  echo "Standard Suite Results (${RESULTS_DIR}/standard_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/standard_results.json') as f:
    d = json.load(f)
succ = d.get('successful_requests') if d.get('successful_requests') is not None else d.get('execution_summary', {}).get('successful_requests', 0)
tot = d.get('total_requests') if d.get('total_requests') is not None else d.get('execution_summary', {}).get('total_requests', 0)
ttft = d.get('ttft_mean_ms') if d.get('ttft_mean_ms') is not None else d.get('metrics', {}).get('ttft_ms', {}).get('mean', 0.0)
tpot = d.get('tpot_mean_ms') if d.get('tpot_mean_ms') is not None else d.get('metrics', {}).get('tpot_ms', {}).get('mean', 0.0)
tps = d.get('throughput_tokens_sec') if d.get('throughput_tokens_sec') is not None else d.get('metrics', {}).get('cluster_throughput_tokens_per_sec', 0.0)
per_gpu = float(tps) / 16.0
print(f'  - Successful Requests: {succ} / {tot}')
print(f'  - Mean TTFT:           {float(ttft):.2f} ms')
print(f'  - Mean TPOT:           {float(tpot):.2f} ms')
print(f'  - Cluster Throughput:  {float(tps):.2f} tokens/sec')
print(f'  - Per-GPU Throughput:  {per_gpu:.2f} tokens/sec/GPU (Normalized across 16x B200)')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/massive_results.json" ]; then
  echo "Massive Suite Results (${RESULTS_DIR}/massive_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/massive_results.json') as f:
    d = json.load(f)
succ = d.get('successful_requests') if d.get('successful_requests') is not None else d.get('execution_summary', {}).get('successful_requests', 0)
tot = d.get('total_requests') if d.get('total_requests') is not None else d.get('execution_summary', {}).get('total_requests', 0)
ttft = d.get('ttft_mean_ms') if d.get('ttft_mean_ms') is not None else d.get('metrics', {}).get('ttft_ms', {}).get('mean', 0.0)
tpot = d.get('tpot_mean_ms') if d.get('tpot_mean_ms') is not None else d.get('metrics', {}).get('tpot_ms', {}).get('mean', 0.0)
tps = d.get('throughput_tokens_sec') if d.get('throughput_tokens_sec') is not None else d.get('metrics', {}).get('cluster_throughput_tokens_per_sec', 0.0)
per_gpu = float(tps) / 16.0
print(f'  - Successful Requests: {succ} / {tot}')
print(f'  - Mean TTFT:           {float(ttft):.2f} ms')
print(f'  - Mean TPOT:           {float(tpot):.2f} ms')
print(f'  - Cluster Throughput:  {float(tps):.2f} tokens/sec')
print(f'  - Per-GPU Throughput:  {per_gpu:.2f} tokens/sec/GPU (Normalized across 16x B200)')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/soak_results.json" ]; then
  echo "Soak Suite Results (${RESULTS_DIR}/soak_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/soak_results.json') as f:
    d = json.load(f)
completed = d.get('total_completed') if d.get('total_completed') is not None else d.get('successful_requests', d.get('execution_summary', {}).get('total_requests_completed', 0))
ttft = d.get('ttft_mean_ms') if d.get('ttft_mean_ms') is not None else d.get('metrics', {}).get('ttft_ms', {}).get('mean', 0.0)
tpot = d.get('tpot_mean_ms') if d.get('tpot_mean_ms') is not None else d.get('metrics', {}).get('tpot_ms', {}).get('mean', 0.0)
tps = d.get('throughput_tokens_sec') if d.get('throughput_tokens_sec') is not None else d.get('metrics', {}).get('sustained_cluster_tps', 0.0)
per_gpu = float(tps) / 16.0
print(f'  - Total Completed Cycles: {completed}')
print(f'  - Mean TTFT:              {float(ttft):.2f} ms')
print(f'  - Mean TPOT:              {float(tpot):.2f} ms')
print(f'  - Sustained Throughput:   {float(tps):.2f} tokens/sec')
print(f'  - Per-GPU Throughput:     {per_gpu:.2f} tokens/sec/GPU (Normalized across 16x B200)')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/prefill_results.json" ]; then
  echo "Prefill Suite Results (${RESULTS_DIR}/prefill_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/prefill_results.json') as f:
    d = json.load(f)
tok_s = d.get('prefill_tok_s_system', 0.0)
ttft = d.get('ttft_ms', 0.0)
per_gpu = float(tok_s) / 16.0
print(f'  - System Prefill Rate:    {float(tok_s):.2f} prompt tokens/sec')
print(f'  - Mean TTFT:              {float(ttft):.2f} ms')
print(f'  - Per-GPU Prefill Rate:   {per_gpu:.2f} prompt tokens/sec/GPU (Normalized across 16x B200)')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/saturation_results.json" ]; then
  echo "Saturation Sweep Results (${RESULTS_DIR}/saturation_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/saturation_results.json') as f:
    d = json.load(f)
results = d.get('sweep_results', [])
for r in results:
    c = r.get('concurrency')
    tps = r.get('aggregate_tok_s', 0.0)
    ttft = r.get('ttft_ms', {}).get('p99', 0.0)
    per_gpu = float(tps) / 16.0
    print(f'  - c={c}: {float(tps):.2f} tok/s ({per_gpu:.2f} tok/s/GPU), P99 TTFT: {float(ttft):.2f} ms')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/realistic_results.json" ]; then
  echo "Prompt Sensitivity Results (${RESULTS_DIR}/realistic_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/realistic_results.json') as f:
    d = json.load(f)
prof = d.get('corpus_profile') or {}
if prof:
    print('  - Corpus: {:.3f}% repeated 8-grams, longest repeated span {} words'.format(
        float(prof.get('duplicate_8gram_fraction', 0.0)) * 100.0,
        prof.get('longest_repeated_ngram_words', 0)))
print('  - Acceptance source: {}'.format(d.get('acceptance_source')))
for e in d.get('arm_comparison', []):
    rep = e.get('repeated_passage', {})
    nov = e.get('non_repetitive', {})
    acc_r = rep.get('accepted_tok_per_step')
    acc_n = nov.get('accepted_tok_per_step')
    acc = 'n/a' if acc_r is None or acc_n is None else '{:.2f} -> {:.2f}'.format(float(acc_r), float(acc_n))
    print('  - ISL={} c={}: accepted tok/step {} | tok/s {:.2f} -> {:.2f} (ratio {}) | prompt-token ratio {}'.format(
        e.get('isl_target'), e.get('concurrency'), acc,
        float(rep.get('aggregate_tok_s') or 0.0), float(nov.get('aggregate_tok_s') or 0.0),
        e.get('non_repetitive_over_repeated_tok_s'), e.get('prompt_token_ratio')))
print('  - Diagnostic only: not one of the five published suites.')
" 2>/dev/null || true
fi

if [ -s "${RESULTS_DIR}/prefix_reuse_results.json" ]; then
  echo "Prefix Reuse Results (${RESULTS_DIR}/prefix_reuse_results.json):"
  python3 -c "
import json
with open('${RESULTS_DIR}/prefix_reuse_results.json') as f:
    d = json.load(f)
v = d.get('verdict') or {}
split = d.get('layer_split') or {}
print('  - Layer split: {} full-attention of {} total (bound on reuse: {})'.format(
    split.get('full_attention_layers'), split.get('total_layers'),
    v.get('full_attention_layer_fraction')))
for arm, s in (v.get('slopes') or {}).items():
    print('  - {:<8} slope {} ms per prefix token over {} point(s)'.format(
        arm, s.get('ms_per_prefix_token'), s.get('points')))
for arm, eff in (v.get('reuse_efficiency') or {}).items():
    print('  - {:<8} reuse_efficiency {}'.format(arm, eff))
for m in d.get('mixed_arm', []):
    print('  - mixed hit_rate={}: token-level expected {} observed {}, P99 TTFT {} ms'.format(
        m.get('target_hit_rate'), m.get('token_level_hit_rate_expected'),
        m.get('token_level_hit_rate_observed'), (m.get('ttft_ms') or {}).get('p99')))
for line in v.get('interpretation', []):
    print('  * ' + line)
" 2>/dev/null || true
fi

# Every capture this run produced, not just the last one -- a run that only
# shows the newest file reads as though the earlier captures were lost.
for kv_file in "${RESULTS_DIR}"/kv_accuracy_*.json; do
  [ -s "${kv_file}" ] || continue
  echo "KV Accuracy Capture (${kv_file}):"
  python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
s = d.get('summary') or {}
ps = d.get('probe_set') or {}
print('  - Label / declared dtype: {} / {}'.format(
    d.get('capture_label'), d.get('kv_cache_dtype_declared')))
print('  - Probes succeeded:       {} at {} context tokens'.format(
    s.get('probes_succeeded'), ps.get('context_tokens')))
print('  - Retrieval accuracy:     {}'.format(s.get('retrieval_accuracy')))
print('  - Wrong-code rate:        {}'.format(s.get('wrong_code_rate')))
print('  - Degenerate rate:        {}'.format(s.get('degenerate_rate')))
for depth, acc in (s.get('retrieval_accuracy_by_depth') or {}).items():
    print('      depth {}: {}'.format(depth, acc))
print('  * A single capture cannot pass or fail fp8 on its own; run compare against a bf16 pair.')
" "${kv_file}" 2>/dev/null || true
done

echo "=============================================================================="
echo "To clean up all resources when finished, run:"
echo "  ./scripts/06_destroy_all.sh"
echo "=============================================================================="
