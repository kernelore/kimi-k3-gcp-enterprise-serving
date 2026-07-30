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
  --mode <standard|massive|soak|prefill|saturation|all>  Benchmark suite to run (default: all)
                                   - standard:   8 concurrent requests, 128 tokens
                                   - massive:    20 concurrent requests, 256 tokens (stress test)
                                   - soak:       30-minute continuous stability endurance test
                                   - prefill:    8192 prompt tokens ingestion stress test
                                   - saturation: ISL/OSL x concurrency sweep (1k-128k input, c=1..128) throughput ceiling
                                   - all:        Run all 5 benchmark suites sequentially
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

export SERVING_MODEL_NAME="${SERVING_MODEL_NAME:-kimi-k3-2.8t-mxfp4}"

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
if [ "${IN_CLUSTER}" != "true" ]; then
  SERVING_POD=$(kubectl get pod -n llm-serving -l app=kimi-k3-serving -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "${SERVING_POD}" ]; then
    if [ "${ENGINE}" = "sglang" ]; then
      ENGINE_VERSION=$(kubectl exec -n llm-serving "${SERVING_POD}" -c sglang-mpi-node -- python3 -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    else
      ENGINE_VERSION=$(kubectl exec -n llm-serving "${SERVING_POD}" -c trtllm-mpi-node -- python3 -c "import tensorrt_llm; print(getattr(tensorrt_llm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    fi
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

  BENCH_SCRIPT=""
  case "${MODE}" in
    soak)       BENCH_SCRIPT="soak_benchmark_kimi_k3.py" ;;
    standard)   BENCH_SCRIPT="benchmark_kimi_k3.py" ;;
    massive)    BENCH_SCRIPT="massive_benchmark_kimi_k3.py" ;;
    prefill)    BENCH_SCRIPT="run_prefill_benchmark_kimi_k3.py" ;;
    saturation) BENCH_SCRIPT="run_saturation_sweep_kimi_k3.py" ;;
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
  
  if [ -f "${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template" ]; then
    # shellcheck disable=SC2016
    safe_envsubst '${ENV_LABEL} ${OWNER_LABEL} ${SERVING_MODEL_NAME}' < "${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template" > "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
    if [ "${MODE}" != "soak" ]; then
      echo "    Configuring in-cluster job for mode '${MODE}' (${BENCH_SCRIPT})..."
      sed -i "s/soak_benchmark_kimi_k3\.py/${BENCH_SCRIPT}/g" "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
      sed -i "s/incluster_soak_results\.json/incluster_${MODE}_results.json/g" "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
    fi
    echo "    Applying in-cluster benchmark Job (${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml)..."
    kubectl delete job kimi-k3-incluster-benchmark -n llm-serving --ignore-not-found=true
    kubectl apply -f "${GENERATED_DIR}/08-in-cluster-benchmark-job.yaml"
  else
    echo "WARNING: ${TEMPLATE_DIR}/08-in-cluster-benchmark-job.yaml.template not found."
    exit 1
  fi

  TIMEOUT="1200s"
  if [ "${MODE}" = "soak" ]; then
    TIMEOUT="3600s"
  fi
  echo "    Waiting for in-cluster benchmark Job to complete (timeout: ${TIMEOUT})..."
  if ! kubectl wait --for=condition=complete job/kimi-k3-incluster-benchmark -n llm-serving --timeout="${TIMEOUT}"; then
    echo "ERROR: In-cluster benchmark Job failed or timed out!" >&2
    kubectl logs -n llm-serving -l app=kimi-k3-benchmark --tail=50 >&2 || true
    exit 1
  fi

  echo "    Retrieving in-cluster benchmark results..."
  BENCH_POD=$(kubectl get pod -n llm-serving -l app=kimi-k3-benchmark -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  mkdir -p "${RESULTS_DIR}"
  if [ -n "${BENCH_POD}" ]; then
    if ! kubectl cp -n llm-serving "${BENCH_POD}:/tmp/results/" "${RESULTS_DIR}/" 2>/dev/null; then
      echo "    [INFO] kubectl cp failed; extracting JSON result from pod logs..."
      kubectl logs -n llm-serving "${BENCH_POD}" | awk '/=== JSON_RESULT_START ===/{flag=1; next} /=== JSON_RESULT_END ===/{flag=0} flag' > "${RESULTS_DIR}/incluster_${MODE}_results.json" || true
      if [ ! -s "${RESULTS_DIR}/incluster_${MODE}_results.json" ]; then
        echo "WARNING: Extracted benchmark JSON from pod logs is empty." >&2
        rm -f "${RESULTS_DIR}/incluster_${MODE}_results.json"
      fi
    fi
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

echo "=============================================================================="
echo "To clean up all resources when finished, run:"
echo "  ./scripts/06_destroy_all.sh"
echo "=============================================================================="
