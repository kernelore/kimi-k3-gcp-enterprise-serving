#!/usr/bin/env bash
# ==============================================================================
# 03_deploy_workloads.sh - Render & Apply Kubernetes Workload Manifests (Kimi K3)
# ==============================================================================
# Renders templates from manifests/templates/ using active environment variables,
# applies local NVMe RAID daemonsets, RBAC/WIF, weights download/hydration jobs,
# compiles TensorRT-LLM engine (Phase 5c), flips volume mode to ReadOnlyMany, and
# deploys the multi-node MPI TensorRT-LLM serving engine and Enterprise AI Gateway.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${PROJECT_ROOT}/terraform"
TEMPLATE_DIR="${TF_DIR}/manifests/templates"
GENERATED_DIR="${TF_DIR}/manifests/generated"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.env}"

if [ ! -f "${CONFIG_FILE}" ]; then
  if [ "${1:-}" = "--render-only" ] || [ "${1:-}" = "--stage-only" ]; then
    echo "    [INFO] ${CONFIG_FILE} not found in render/stage mode. Falling back to config.env.example..."
    CONFIG_FILE="${SCRIPT_DIR}/config.env.example"
  else
    echo "ERROR: ${CONFIG_FILE} not found. Please run ./scripts/01_setup_and_check.sh first."
    exit 1
  fi
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

validate_hf_token() {
  local token="${HF_TOKEN:-}"
  if [[ -z "${token}" ]] || \
     [[ "${token}" != hf_* ]] || \
     [[ "${token}" =~ [Pp][Ll][Aa][Cc][Ee][Hh][Oo][Ll][Dd][Ee][Rr] ]] || \
     [[ "${token}" =~ [Yy][Oo][Uu][Rr]_ ]] || \
     [[ "${token}" =~ [Xx][Xx][Xx] ]] || \
     [[ "${token}" =~ [Tt][Oo][Dd][Oo] ]] || \
     [[ "${token}" == *"<"* ]] || \
     [[ "${token}" == *">"* ]]; then
    echo "ERROR: Invalid or placeholder HF_TOKEN detected in scripts/config.env. Please configure a valid Hugging Face access token (starting with 'hf_') with gated-repo licence access." >&2
    exit 1
  fi
}

export MODEL_REPO_ID="${MODEL_REPO_ID:-moonshotai/Kimi-K3}"
export SERVING_MODEL_NAME="${SERVING_MODEL_NAME:-moonshotai/Kimi-K3}"

# Kubernetes label VALUES may only contain [A-Za-z0-9._-] and must start and end with an
# alphanumeric, so the HuggingFace-style repo id in SERVING_MODEL_NAME ("moonshotai/Kimi-K3")
# is a legal label KEY prefix but an illegal label VALUE -- every apply of the gateway
# Deployment and the PodMonitoring was rejected outright by the API server. Derive a
# sanitised variant for the ai.gke.io/model label and keep SERVING_MODEL_NAME itself intact,
# because the gateway config and the engine both need the real repo id.
SERVING_MODEL_LABEL=$(printf '%s' "${SERVING_MODEL_NAME}" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -e 's/[^a-z0-9._-]/-/g' -e 's/^[^a-z0-9]*//' -e 's/[^a-z0-9]*$//' \
  | cut -c1-63 \
  | sed -e 's/[^a-z0-9]*$//')
export SERVING_MODEL_LABEL="${SERVING_MODEL_LABEL:-kimi-k3}"

export FABRIC_GATE_TIMEOUT_SECONDS="${FABRIC_GATE_TIMEOUT_SECONDS:-900}"
# Separate budget for getting the gate pods onto nodes at all. Keep it apart from
# FABRIC_GATE_TIMEOUT_SECONDS above: that one is meant to bound the fabric test, and if
# node provisioning is charged to the same clock then the gate reports a fabric fault
# whenever GKE is merely slow to hand over spot capacity. See wait_for_fabric_pods_running.
export FABRIC_POD_READY_TIMEOUT_SECONDS="${FABRIC_POD_READY_TIMEOUT_SECONDS:-1800}"
export TRTLLM_TP_SIZE="${TRTLLM_TP_SIZE:-8}"
export TRTLLM_PP_SIZE="${TRTLLM_PP_SIZE:-2}"
export TRTLLM_EP_SIZE="${TRTLLM_EP_SIZE:-8}"
export TRTLLM_MAX_SEQ_LEN="${TRTLLM_MAX_SEQ_LEN:-131072}"
export HYPERDISK_ML_SIZE_GB="${HYPERDISK_ML_SIZE_GB:-2000}"
export GPU_MAX_NODES="${GPU_MAX_NODES:-2}"
export SERVING_REPLICAS="${SERVING_REPLICAS:-1}"
export NODES_PER_REPLICA="${NODES_PER_REPLICA:-2}"

if [ -n "${GCS_WEIGHTS_BUCKET:-}" ] && [[ "${GCS_WEIGHTS_BUCKET}" != gs://* ]]; then
  export GCS_WEIGHTS_BUCKET="gs://${GCS_WEIGHTS_BUCKET}"
fi

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

echo "=============================================================================="
echo "Kimi K3 Sovereign Enterprise Inference - Phase 3: Workload Deployment"
echo "=============================================================================="
echo "Target Cluster: ${CLUSTER_NAME} (${ZONE})"
echo "Model Identity: ${MODEL_REPO_ID} (${HYPERDISK_ML_SIZE_GB} GB ROX Volume)"
echo "=============================================================================="

# Ensure generated directory exists and is clean
mkdir -p "${GENERATED_DIR}"
rm -f "${GENERATED_DIR}"/*.yaml

# Prepare base64 encoded token for Kubernetes secret template
HF_TOKEN_BASE64=$(echo -n "${HF_TOKEN:-placeholder_token}" | base64 -w 0 2>/dev/null || echo -n "${HF_TOKEN:-placeholder_token}" | base64)
export HF_TOKEN_BASE64
export GATEWAY_MASTER_KEY="${GATEWAY_MASTER_KEY:-sk-kimi-k3-master-secret-key-change-me}"
export DB_PASSWORD="${DB_PASSWORD:-kimi-k3-gateway-admin-secret}"
if [ "${1:-}" != "--render-only" ] && [ "${1:-}" != "--stage-only" ]; then
  if [ "${LIVE_VALIDATION:-no}" != "yes" ]; then
    echo "ERROR: refusing to create billable GCP resources without LIVE_VALIDATION=yes" >&2
    echo "Would have deployed workload manifests and inference engine StatefulSets to cluster '${CLUSTER_NAME:-unknown}' in project '${PROJECT_ID:-unknown}'." >&2
    exit 1
  fi
  if [ "${GATEWAY_MASTER_KEY}" = "sk-kimi-k3-master-secret-key-change-me" ]; then
    echo "ERROR: GATEWAY_MASTER_KEY must be set to a secure secret outside render-only/stage-only mode!" >&2
    echo "ERROR: Failed to obtain GATEWAY_MASTER_KEY (insecure default detected)" >&2
    exit 1
  fi
  if [ "${DB_PASSWORD}" = "kimi-k3-gateway-admin-secret" ]; then
    echo "ERROR: Failed to obtain DB_PASSWORD (insecure default detected)" >&2
    exit 1
  fi
fi

get_tf_output() {
  local val
  val=$( (cd "${TF_DIR}" && terraform output -raw "$1" 2>/dev/null) || true)
  if [ -n "${val}" ] && [[ "${val}" != *"╷"* ]] && [[ "${val}" != *"Warning:"* ]] && [[ "${val}" != *"Error:"* ]]; then
    echo "${val}"
  fi
}

get_gcloud_val() {
  if command -v gcloud >/dev/null 2>&1; then
    local val
    val=$(gcloud "$@" 2>/dev/null || true)
    if [ -n "${val}" ] && [[ "${val}" != *"ERROR:"* ]] && [[ "${val}" != *"WARNING:"* ]]; then
      echo "${val}" | head -n 1
    fi
  fi
}

REDIS_PASSWORD=$(get_tf_output redis_auth_string)
if [ -z "${REDIS_PASSWORD}" ]; then
  REDIS_PASSWORD=$(get_gcloud_val redis instances get-auth-string kimi-k3-gateway-cache --region="${REGION}" --format="value(authString)" --quiet)
fi
if [ -z "${REDIS_PASSWORD}" ]; then
  if [ "${1:-}" = "--render-only" ] || [ "${1:-}" = "--stage-only" ]; then
    REDIS_PASSWORD="redis-secret-password-change-me"
  else
    echo "ERROR: Failed to obtain REDIS_PASSWORD from Terraform output or gcloud!" >&2
    exit 1
  fi
fi
export REDIS_PASSWORD

REDIS_PASSWORD_ENCODED=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote_plus(sys.argv[1]))" "${REDIS_PASSWORD}")
export REDIS_PASSWORD_ENCODED

# Extract Redis and Cloud SQL details from Terraform (or gcloud fallback)
REDIS_HOST=$(get_tf_output redis_host)
if [ -z "${REDIS_HOST}" ]; then
  REDIS_HOST=$(get_gcloud_val redis instances describe kimi-k3-gateway-cache --region="${REGION}" --format="value(host)" --quiet)
fi
if [ -z "${REDIS_HOST}" ]; then
  if [ "${1:-}" = "--render-only" ] || [ "${1:-}" = "--stage-only" ]; then
    REDIS_HOST="redis-cache.local"
  else
    echo "ERROR: Failed to obtain REDIS_HOST from Terraform output or gcloud!" >&2
    exit 1
  fi
fi
export REDIS_HOST

DB_CONNECTION_NAME=$(get_tf_output db_instance_connection_name)
if [ -z "${DB_CONNECTION_NAME}" ]; then
  DB_CONNECTION_NAME=$(get_gcloud_val sql instances describe kimi-k3-gateway-db --format="value(connectionName)" --quiet)
fi
if [ -z "${DB_CONNECTION_NAME}" ]; then
  if [ "${1:-}" = "--render-only" ] || [ "${1:-}" = "--stage-only" ]; then
    DB_CONNECTION_NAME="${PROJECT_ID}:${REGION}:kimi-k3-gateway-db"
  else
    echo "ERROR: Failed to obtain DB_CONNECTION_NAME from Terraform output or gcloud!" >&2
    exit 1
  fi
fi
export DB_CONNECTION_NAME

export TRTLLM_VIP="kimi-k3-serving-svc.llm-serving.svc.cluster.local"
export SGLANG_PARALLEL_PROFILE="${SGLANG_PARALLEL_PROFILE:-tp16}"
if [ "${SGLANG_PARALLEL_PROFILE}" = "tp8pp2" ]; then
  export SGLANG_TP_SIZE="8"
  export SGLANG_PP_SIZE="2"
  export SGLANG_EP_SIZE="8"
else
  export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-16}"
  export SGLANG_PP_SIZE="${SGLANG_PP_SIZE:-1}"
  export SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-16}"
fi
export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-}"

# Parallel-geometry validation guard
# Note: The wider literature describes EP <= TP x DP with even divisibility; we enforce the narrower {1, TP} because those are the only values this architecture ever uses and a false reject at render time costs nothing, while a false accept costs a 16-GPU startup failure mid-incident.
EXPECTED_TOTAL_GPUS=$(( NODES_PER_REPLICA * 8 ))
ACTUAL_PARALLEL_GPUS=$(( SGLANG_TP_SIZE * SGLANG_PP_SIZE ))
if [ "${ACTUAL_PARALLEL_GPUS}" -ne "${EXPECTED_TOTAL_GPUS}" ]; then
  echo "ERROR: Invalid parallel geometry: SGLANG_TP_SIZE (${SGLANG_TP_SIZE}) * SGLANG_PP_SIZE (${SGLANG_PP_SIZE}) = ${ACTUAL_PARALLEL_GPUS}, expected ${EXPECTED_TOTAL_GPUS} (NODES_PER_REPLICA ${NODES_PER_REPLICA} * 8)." >&2
  exit 1
fi

if [ "${SGLANG_EP_SIZE}" -ne 1 ] && [ "${SGLANG_EP_SIZE}" -ne "${SGLANG_TP_SIZE}" ]; then
  echo "ERROR: Invalid SGLANG_EP_SIZE (${SGLANG_EP_SIZE}) for SGLANG_TP_SIZE (${SGLANG_TP_SIZE}). SGLang supports ep_size of 1 or ep_size == tp_size (EP is intra-stage: with PP=N each stage owns TP x DP GPUs and EP cannot exceed that)." >&2
  exit 1
fi
export SGLANG_PORT="${SGLANG_PORT:-8000}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
export SGLANG_SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-lpm}"
export SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
export SGLANG_ENABLE_TORCH_COMPILE="${SGLANG_ENABLE_TORCH_COMPILE:-false}"
# Attention backends. All empty by default: the SGLang cookbook pins no attention backend
# on any Blackwell cell. SGLANG_ATTENTION_BACKEND is accepted as a deprecated alias for the
# prefill value -- the old name emitted --attention-backend, which sets the base for BOTH
# phases and therefore silently pinned decode as well.
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ] && [ -z "${SGLANG_PREFILL_ATTENTION_BACKEND:-}" ]; then
  echo "WARNING: SGLANG_ATTENTION_BACKEND='${SGLANG_ATTENTION_BACKEND}' is deprecated and is being mapped to --prefill-attention-backend only." >&2
  echo "         The SGLang cookbook pins no attention backend on any Blackwell cell; clear it in scripts/config.env to match the B200 recipe," >&2
  echo "         or set SGLANG_PREFILL_ATTENTION_BACKEND explicitly to keep pinning prefill." >&2
fi
export SGLANG_PREFILL_ATTENTION_BACKEND="${SGLANG_PREFILL_ATTENTION_BACKEND:-${SGLANG_ATTENTION_BACKEND:-}}"
export SGLANG_DECODE_ATTENTION_BACKEND="${SGLANG_DECODE_ATTENTION_BACKEND:-}"
export SGLANG_LINEAR_ATTN_PREFILL_BACKEND="${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-}"
# MoE runner and KV-cache dtype. Also empty by default: opt-in overlays in the cookbook
# playground, absent from the B200 2-node cell this deployment follows.
export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-}"
export SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
export SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-false}"
export SGLANG_HICACHE_STORAGE_BACKEND="${SGLANG_HICACHE_STORAGE_BACKEND:-}"
export SGLANG_HICACHE_RATIO="${SGLANG_HICACHE_RATIO:-}"
export SGLANG_HICACHE_SIZE="${SGLANG_HICACHE_SIZE:-}"
export SGLANG_HICACHE_WRITE_POLICY="${SGLANG_HICACHE_WRITE_POLICY:-}"
export SGLANG_HICACHE_IO_BACKEND="${SGLANG_HICACHE_IO_BACKEND:-}"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="${SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE:-}"
# Must match the mount point created by the local-nvme-raid-formatter DaemonSet in
# terraform/manifests/templates/00-local-nvme-raid.yaml.template.
export SGLANG_HOST_SCRATCH_PATH="${SGLANG_HOST_SCRATCH_PATH:-/mnt/disks/local-scratch}"
export SGLANG_LOCAL_SCRATCH_MOUNT="${SGLANG_LOCAL_SCRATCH_MOUNT:-/mnt/scratch}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-}"
export SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-}"
export EXPECTED_MODEL_ARCHITECTURE="${EXPECTED_MODEL_ARCHITECTURE:-}"
export MIN_WEIGHTS_GIB="${MIN_WEIGHTS_GIB:-1000}"
export LEADER_ADDR="${LEADER_ADDR:-kimi-k3-serving-0.kimi-k3-workers-headless.llm-serving.svc.cluster.local}"

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
export INFERENCE_SERVER_LABEL="${INFERENCE_ENGINE}"
if [ "${INFERENCE_ENGINE}" = "sglang" ]; then
  IMAGE_NAME="sglang-blackwell"
  IMAGE_TAG="latest"
  DOCKERFILE="Dockerfile.sglang"
  SERVING_MANIFEST="09-kimi-k3-sglang-mpi.yaml"
elif [ "${INFERENCE_ENGINE}" = "trtllm" ]; then
  IMAGE_NAME="trtllm-blackwell"
  IMAGE_TAG="latest"
  DOCKERFILE="Dockerfile"
  SERVING_MANIFEST="09-kimi-k3-trtllm-mpi.yaml"
else
  echo "ERROR: Unsupported INFERENCE_ENGINE '${INFERENCE_ENGINE}'. Must be 'trtllm' or 'sglang'." >&2
  exit 1
fi
export SERVING_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod/${IMAGE_NAME}:${IMAGE_TAG}"
export INFERENCE_ENGINE INFERENCE_SERVER_LABEL SERVING_IMAGE

# 1. Render manifest templates (excluding HF_TOKEN from substitution to prevent plaintext baking)
echo "--> 1. Rendering manifest templates from ${TEMPLATE_DIR} to ${GENERATED_DIR}..."
# shellcheck disable=SC2016
BASE_ALLOWED_VARS='${PROJECT_ID} ${REGION} ${ZONE} ${CLUSTER_NAME} ${OWNER_LABEL} ${TTL_LABEL} ${ENV_LABEL} ${HF_TOKEN_BASE64} ${MODEL_REPO_ID} ${SERVING_MODEL_NAME} ${SERVING_MODEL_LABEL} ${TRTLLM_TP_SIZE} ${TRTLLM_PP_SIZE} ${TRTLLM_EP_SIZE} ${TRTLLM_MAX_SEQ_LEN} ${SGLANG_PARALLEL_PROFILE} ${SGLANG_TP_SIZE} ${SGLANG_PP_SIZE} ${SGLANG_PP_LAYER_PARTITION} ${SGLANG_EP_SIZE} ${SGLANG_PORT} ${SGLANG_MEM_FRACTION_STATIC} ${SGLANG_SCHEDULE_POLICY} ${SGLANG_CHUNKED_PREFILL_SIZE} ${SGLANG_MAX_RUNNING_REQUESTS} ${SGLANG_DISABLE_CUSTOM_ALL_REDUCE} ${SGLANG_SPECULATIVE_ALGORITHM} ${SGLANG_SPECULATIVE_DRAFT_MODEL_PATH} ${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE} ${SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC} ${DSPARK_DRAFT_DIR_NAME} ${SERVING_POD_SERVICE_ACCOUNT} ${SGLANG_QUANTIZATION} ${SGLANG_ENABLE_TORCH_COMPILE} ${SGLANG_PREFILL_ATTENTION_BACKEND} ${SGLANG_DECODE_ATTENTION_BACKEND} ${SGLANG_LINEAR_ATTN_PREFILL_BACKEND} ${SGLANG_MOE_RUNNER_BACKEND} ${SGLANG_KV_CACHE_DTYPE} ${SGLANG_ENABLE_HIERARCHICAL_CACHE} ${SGLANG_HICACHE_STORAGE_BACKEND} ${SGLANG_HICACHE_RATIO} ${SGLANG_HICACHE_SIZE} ${SGLANG_HICACHE_WRITE_POLICY} ${SGLANG_HICACHE_IO_BACKEND} ${SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE} ${SGLANG_HOST_SCRATCH_PATH} ${SGLANG_LOCAL_SCRATCH_MOUNT} ${SGLANG_CONTEXT_LENGTH} ${SGLANG_REASONING_PARSER} ${SGLANG_TOOL_CALL_PARSER} ${EXPECTED_MODEL_ARCHITECTURE} ${MIN_WEIGHTS_GIB} ${LEADER_ADDR} ${HYPERDISK_ML_SIZE_GB} ${GCS_WEIGHTS_BUCKET} ${DB_CONNECTION_NAME} ${DB_PASSWORD} ${REDIS_HOST} ${REDIS_PASSWORD} ${REDIS_PASSWORD_ENCODED} ${TRTLLM_VIP} ${GPU_MAX_NODES} ${SERVING_REPLICAS} ${NODES_PER_REPLICA} ${INFERENCE_ENGINE} ${INFERENCE_SERVER_LABEL} ${SERVING_IMAGE}'
for template_file in "${TEMPLATE_DIR}"/*.yaml.template; do
  if [ -f "${template_file}" ]; then
    basename=$(basename "${template_file}" .template)
    target_file="${GENERATED_DIR}/${basename}"
    echo "    Rendering ${basename}..."
    allowed_vars="${BASE_ALLOWED_VARS}"
    if [ "${basename}" = "04-enterprise-gateway-config.yaml" ] || [ "${basename}" = "05-enterprise-gateway-deployment.yaml" ]; then
      allowed_vars="${BASE_ALLOWED_VARS} \${GATEWAY_MASTER_KEY} \${GATEWAY_CONFIG_CHECKSUM}"
    fi
    # shellcheck disable=SC2016
    safe_envsubst "${allowed_vars}" < "${template_file}" > "${target_file}"

    # Fingerprint the rendered gateway config so the Deployment below can carry it as a pod
    # annotation. Without this, `kubectl apply` updates the ConfigMap but the running pods keep
    # serving with the config they booted from -- so rotating GATEWAY_MASTER_KEY appeared to
    # succeed while the old key stayed live and every client using the new one got 401s.
    # The glob is lexicographic, so 04- is always rendered before 05- consumes this.
    if [ "${basename}" = "04-enterprise-gateway-config.yaml" ]; then
      GATEWAY_CONFIG_CHECKSUM=$(sha256sum "${target_file}" | cut -c1-16)
      export GATEWAY_CONFIG_CHECKSUM
    fi
  fi
done
echo "    [OK] All manifest templates rendered cleanly."

if [ "${1:-}" = "--render-only" ] || [ "${1:-}" = "--stage-only" ]; then
  if [ "${1:-}" = "--render-only" ]; then
    echo "    Render-only mode complete."
    exit 0
  fi
fi

# 2. Check and self-heal container image in Artifact Registry (seeding via Cloud Build from docker/${DOCKERFILE} if missing)
echo "--> 2. Verifying Kimi K3 runtime container image (${SERVING_IMAGE}) in Artifact Registry..."
if command -v gcloud >/dev/null 2>&1; then
  if ! gcloud artifacts docker tags list "${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod/${IMAGE_NAME}" --format="value(tag)" --quiet 2>/dev/null | grep -E -q "^${IMAGE_TAG}$|^latest$"; then
    echo "    [INFO] Container image ${IMAGE_NAME}:${IMAGE_TAG} not found in ${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod."
    echo "    --> Triggering self-healing container build via Google Cloud Build from docker/${DOCKERFILE}..."
    gcloud services enable cloudbuild.googleapis.com --project="${PROJECT_ID}" --quiet 2>/dev/null || true
    BUILD_DIR=$(mktemp -d)
    cp -r "${PROJECT_ROOT}/docker/." "${BUILD_DIR}/"
    cp "${PROJECT_ROOT}/docker/${DOCKERFILE}" "${BUILD_DIR}/Dockerfile"
    echo "    --> Submitting build for docker/${DOCKERFILE} to Cloud Build..."
    if ! gcloud builds submit "${BUILD_DIR}" --tag "${SERVING_IMAGE}" --timeout=7200s --machine-type=e2-highcpu-32 --disk-size=200 --project="${PROJECT_ID}" --quiet; then
      echo "    [ERROR] Cloud Build failed! Inspect build logs at: https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}" >&2
      rm -rf "${BUILD_DIR}"
      exit 1
    fi
    rm -rf "${BUILD_DIR}"
    echo "    [OK] Self-healing container build step finished."
  else
    echo "    [OK] Container image ${IMAGE_NAME} verified in Artifact Registry."
  fi
fi

# 3. Apply base cluster resources (NVMe RAID formatter, NCCL gIB installer, RDMA networks & RBAC/WIF/Secret)
echo "--> 3. Applying Base Infrastructure DaemonSet, RDMA networks & Workload Identity RBAC..."
kubectl apply -f "${GENERATED_DIR}/00-local-nvme-raid.yaml"
kubectl apply -f "${GENERATED_DIR}/00a-nccl-gib-installer.yaml"
kubectl apply -f "${GENERATED_DIR}/00b-rdma-networks.yaml"
kubectl apply -f "${GENERATED_DIR}/01-rbac-wif.yaml"
kubectl apply -f "${GENERATED_DIR}/10-scheduled-turndown-cronjob.yaml"

echo "--> 4. Waiting for local-nvme-raid-formatter DaemonSet rollout..."
kubectl rollout status daemonset/local-nvme-raid-formatter -n kube-system --timeout=180s || echo "WARNING: DaemonSet rollout timeout (may be waiting for spot nodes to register)."

# The gIB installer stages the NCCL network plugin onto each GPU node: its init container
# pulls a multi-gigabyte image and then `cp -r`s into /home/kubernetes/bin/gib, so that
# directory exists and is incomplete for most of the operation. Everything below mounts it.
#
# This cannot be fully sequenced from here, and it is worth being explicit about why. The
# GPU pool has a minimum of zero nodes and the cluster autoscaler does not scale up for
# DaemonSet pods, so at this point there is usually no GPU node at all: the DaemonSet's
# desired count is 0 and `kubectl rollout status` returns success immediately. The first
# GPU node is created by the RoCE gate's own StatefulSet below, which means the gate pod
# and the installer land on a brand-new node at the same moment and race.
#
# So this wait is opportunistic -- it only means anything on a re-run against a warm
# cluster. The load-bearing guard is the bounded wait for set_nccl_env.sh inside the gate
# pod itself (00c-nccl-test-job.yaml.template), which is on the node where the race is.
GIB_DESIRED="$(kubectl get daemonset/nccl-gib-installer -n kube-system \
  -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo 0)"
if [ "${GIB_DESIRED:-0}" -gt 0 ]; then
  echo "--> 4b. Waiting for nccl-gib-installer DaemonSet rollout (${GIB_DESIRED} GPU node(s) present)..."
  if ! kubectl rollout status daemonset/nccl-gib-installer -n kube-system --timeout=600s; then
    echo "ERROR: nccl-gib-installer DaemonSet did not become ready within 600s." >&2
    echo "       Every NCCL collective below depends on the plugin it installs; continuing" >&2
    echo "       would measure a TCP fallback and call it a fabric result." >&2
    kubectl get pods -n kube-system -l app=nccl-gib-installer -o wide >&2 || true
    kubectl describe daemonset/nccl-gib-installer -n kube-system >&2 || true
    exit 1
  fi
else
  echo "--> 4b. No GPU node registered yet, so nccl-gib-installer has nothing scheduled."
  echo "        The gate pod waits for the plugin itself once its node exists."
fi

# 3b. Verify RoCEv2 Network Fabric and NCCL bus bandwidth floor (>= 100 GB/s)
cleanup_fabric_pods() {
  kubectl delete statefulset nccl-roce-test nccl-parity-check -n llm-serving --ignore-not-found=true >/dev/null 2>&1 || true
}

# Block until no pod matching the given label selector is left in the namespace. The fabric
# gates cannot overlap (see the sequencing note below), so the second gate must not be applied
# until the first one's pods have actually released their RDMA interfaces -- a deleted
# StatefulSet returns immediately while its pods are still Terminating.
wait_for_fabric_pods_gone() {
  local selector="$1"
  local deadline=$(( $(date +%s) + 180 ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ "$(kubectl get pods -n llm-serving -l "${selector}" --no-headers 2>/dev/null | wc -l)" -eq 0 ]; then
      return 0
    fi
    sleep 3
  done
  echo "    WARNING: pods matching '${selector}' still present after 180s; continuing anyway." >&2
  return 0
}

# Block until the gate's pods are actually Running, on their own clock, before anyone starts
# timing the fabric.
#
# The marker poll that follows each gate is meant to answer "is the fabric good". Applying the
# StatefulSet and immediately starting that poll makes it answer "is the fabric good AND did
# GKE hand us two spot B200 nodes quickly" -- and it reports the compound failure as a fabric
# fault. Measured, not assumed: window 2 of the measurement runs applied the gate at 13:07:28
# and its nodes only reached Ready at 13:17:15. That is 587s of a 900s budget spent on
# provisioning; the pods started at ~13:18:20 and the clock expired at 13:22:28, so an
# 8M-512M all_reduce sweep got ~4 minutes and had not finished NCCL init. Not one size row
# printed. The log said "marker absent from nccl-roce-test-0 at timeout (900s)", which reads
# as a dead fabric and was nothing of the kind.
#
# Pending is deliberately NOT treated as an error here: it is the normal state while a node
# pool scales from zero, which is exactly the case this gate runs in.
wait_for_fabric_pods_running() {
  local sts="$1" container="$2" replicas="${3:-2}"
  local deadline=$(( $(date +%s) + FABRIC_POD_READY_TIMEOUT_SECONDS ))
  local last_report running stuck now
  last_report="$(date +%s)"
  echo "    Waiting for ${replicas} ${sts} pod(s) to reach Running (timeout: ${FABRIC_POD_READY_TIMEOUT_SECONDS}s)."
  echo "    This covers spot GPU node provisioning and image pull, not the fabric test itself."
  while :; do
    running="$(kubectl get pods -n llm-serving -l "app=${sts}" \
      --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)"
    if [ "${running}" -ge "${replicas}" ]; then
      echo "    All ${replicas} ${sts} pod(s) Running; starting the ${FABRIC_GATE_TIMEOUT_SECONDS}s fabric clock now."
      return 0
    fi

    # Fail fast on waiting states that never resolve by themselves. Spending the whole
    # provisioning budget on a pod that cannot pull its image wastes GPU-hours that are
    # already billing, and buries the real cause under a timeout message.
    stuck="$(kubectl get pods -n llm-serving -l "app=${sts}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"="}{range .status.containerStatuses[*]}{.state.waiting.reason}{" "}{end}{"\n"}{end}' 2>/dev/null \
      | grep -E 'ImagePullBackOff|ErrImagePull|CrashLoopBackOff|CreateContainerConfigError|InvalidImageName' || true)"
    if [ -n "${stuck}" ]; then
      echo "ERROR: ${sts} pod(s) are in a terminal waiting state and will not start:" >&2
      printf '%s\n' "${stuck}" >&2
      dump_fabric_diagnostics "${sts}" "${container}"
      cleanup_fabric_pods
      exit 1
    fi

    now="$(date +%s)"
    if [ "${now}" -ge "${deadline}" ]; then
      echo "ERROR: only ${running}/${replicas} ${sts} pod(s) reached Running within ${FABRIC_POD_READY_TIMEOUT_SECONDS}s." >&2
      echo "       The fabric test never started, so this result says nothing about the fabric." >&2
      echo "       Most likely spot B200 capacity in this zone, not the RDMA configuration." >&2
      dump_fabric_diagnostics "${sts}" "${container}"
      cleanup_fabric_pods
      exit 1
    fi
    # A silent twenty-minute wait is indistinguishable from a hang to whoever is watching.
    if [ $(( now - last_report )) -ge 60 ]; then
      echo "    ... ${running}/${replicas} Running, $(( deadline - now ))s of provisioning budget left"
      last_report="${now}"
    fi
    sleep 10
  done
}

# Dump everything needed to diagnose a fabric gate failure without re-running it. These
# gates only fail on a live GPU cluster, so a truncated dump means either paying to
# reproduce or guessing. Both ranks, in full: a hang shows up as an absence in rank 0's
# log, and it is rank 1 that says what it was waiting for. Preceded by pod state and
# events, which is where scheduling and image-pull failures show up instead.
dump_fabric_diagnostics() {
  local sts="$1" container="$2"
  echo "--- fabric diagnostics for ${sts} ---" >&2
  kubectl get pods -n llm-serving -l "app=${sts}" -o wide >&2 2>&1 || true
  kubectl get events -n llm-serving --sort-by=.lastTimestamp 2>/dev/null | tail -30 >&2 || true
  local rank
  for rank in 0 1; do
    echo "--- full log: ${sts}-${rank} (container ${container}) ---" >&2
    kubectl logs "${sts}-${rank}" -n llm-serving -c "${container}" >&2 2>&1 || true
    echo "--- previous container log (if it restarted): ${sts}-${rank} ---" >&2
    kubectl logs "${sts}-${rank}" -n llm-serving -c "${container}" --previous >&2 2>&1 || true
  done
  echo "--- end fabric diagnostics ---" >&2
}

if [ "${SKIP_FABRIC_CHECK:-false}" = "true" ]; then
  echo "WARNING: SKIP_FABRIC_CHECK=true set. Bypassing RoCEv2 network fabric and NCCL bus bandwidth verification!" >&2
else
  echo "--> 3b. Verifying RoCEv2 RDMA network fabric and NCCL bus bandwidth (floor >= 100 GB/s)..."
  # The two fabric gates MUST run one after the other, never together. Each gate pod requests
  # all eight RDMA interfaces (networking.gke.io.networks/rdma-0..7), and a node advertises
  # exactly one of each -- so a node can host exactly one gate pod, whatever its GPU count.
  # Applying both 2-replica StatefulSets up front on the documented 2-node topology therefore
  # deadlocks permanently: two pods run, the other two stay Pending forever, and both gates
  # time out. That is why the RoCE bus-bandwidth number was never successfully captured.
  kubectl apply -f "${GENERATED_DIR}/00c-nccl-test-job.yaml"

  # This StatefulSet is what causes the GPU node pool to scale off zero, so the wait below
  # is the node-provisioning wait. It has to finish before the fabric clock starts.
  wait_for_fabric_pods_running nccl-roce-test nccl-test 2

  echo "    Polling NCCL RoCEv2 rank-0 pod (nccl-roce-test-0) for machine marker (timeout: ${FABRIC_GATE_TIMEOUT_SECONDS}s)..."
  MARKER_LINE=""
  FAIL_SEEN=""
  BUSBW_VAL=""
  MAX_ATTEMPTS=$(( FABRIC_GATE_TIMEOUT_SECONDS / 5 ))
  ATTEMPT=0
  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    LOG_OUTPUT=$(kubectl logs nccl-roce-test-0 -n llm-serving -c nccl-test 2>/dev/null || true)
    MARKER_LINE=$(echo "${LOG_OUTPUT}" | grep -E "^NCCL_GATE_RESULT " | tail -n 1 || true)
    FAIL_SEEN=$(echo "${LOG_OUTPUT}" | grep -E "^NCCL_GATE_RESULT fail" | head -n 1 || true)
    if [ -n "${MARKER_LINE}" ]; then
      break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 5
  done

  if [ -z "${MARKER_LINE}" ]; then
    echo "ERROR: NCCL RoCEv2 verification failed: marker absent from nccl-roce-test-0 at timeout (${FABRIC_GATE_TIMEOUT_SECONDS}s)!" >&2
    dump_fabric_diagnostics nccl-roce-test nccl-test
    cleanup_fabric_pods
    exit 1
  fi

  if [ -n "${FAIL_SEEN}" ]; then
    echo "ERROR: NCCL RoCEv2 verification reported failure marker (${FAIL_SEEN})!" >&2
    dump_fabric_diagnostics nccl-roce-test nccl-test
    cleanup_fabric_pods
    exit 1
  fi
  BUSBW_VAL=$(echo "${MARKER_LINE}" | awk -F'busbw_gbps=' '{print $2}' | awk '{print $1}' | xargs || true)
  if [ -z "${BUSBW_VAL}" ]; then
    echo "ERROR: NCCL RoCEv2 bus bandwidth value is empty (${MARKER_LINE})!" >&2
    cleanup_fabric_pods
    exit 1
  fi
  if ! echo "${BUSBW_VAL}" | grep -E -q '^[0-9]+(\.[0-9]+)?$'; then
    echo "ERROR: NCCL RoCEv2 bus bandwidth value '${BUSBW_VAL}' is non-numeric or invalid!" >&2
    cleanup_fabric_pods
    exit 1
  fi
  if ! awk -v val="${BUSBW_VAL}" 'BEGIN {exit !(val >= 100.0)}' 2>/dev/null; then
    echo "ERROR: NCCL RoCEv2 bus bandwidth (${BUSBW_VAL} GB/s) is below required 100 GB/s floor!" >&2
    cleanup_fabric_pods
    exit 1
  fi
  echo "    [OK] NCCL RoCEv2 bus bandwidth meets >= 100 GB/s requirement (${BUSBW_VAL} GB/s)."

  # First gate passed: tear it down and wait for its RDMA interfaces to be released before
  # the parity gate can be scheduled onto the same nodes.
  echo "    Releasing RoCEv2 gate pods before starting the serving-image parity gate..."
  kubectl delete statefulset nccl-roce-test -n llm-serving --ignore-not-found=true >/dev/null 2>&1 || true
  wait_for_fabric_pods_gone "app=nccl-roce-test"
  kubectl apply -f "${GENERATED_DIR}/00d-serving-nccl-parity-job.yaml"

  # Nodes already exist by this point and the serving image is warm in the local registry,
  # so this normally returns in seconds. It is here because the flaw is structural, not
  # specific to the first gate: any marker poll that also times scheduling can fail for
  # reasons that have nothing to do with what it claims to measure.
  wait_for_fabric_pods_running nccl-parity-check nccl-parity-check 2

  echo "    Polling NCCL parity check rank-0 pod (nccl-parity-check-0) for parity marker (timeout: ${FABRIC_GATE_TIMEOUT_SECONDS}s)..."
  PARITY_MARKER=""
  FAIL_SEEN=""
  MAX_ATTEMPTS=$(( FABRIC_GATE_TIMEOUT_SECONDS / 5 ))
  ATTEMPT=0
  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    LOG_OUTPUT=$(kubectl logs nccl-parity-check-0 -n llm-serving -c nccl-parity-check 2>/dev/null || true)
    PARITY_MARKER=$(echo "${LOG_OUTPUT}" | grep -E "^NCCL_PARITY_RESULT " | tail -n 1 || true)
    FAIL_SEEN=$(echo "${LOG_OUTPUT}" | grep -E "^NCCL_PARITY_RESULT fail" | head -n 1 || true)
    if [ -n "${PARITY_MARKER}" ]; then
      break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 5
  done

  if [ -z "${PARITY_MARKER}" ]; then
    echo "ERROR: NCCL parity check failed: marker absent from nccl-parity-check-0 at timeout (${FABRIC_GATE_TIMEOUT_SECONDS}s)!" >&2
    dump_fabric_diagnostics nccl-parity-check nccl-parity-check
    cleanup_fabric_pods
    exit 1
  fi
  if [ -n "${FAIL_SEEN}" ]; then
    echo "ERROR: NCCL parity check failed: rank 0 emitted fail marker (${FAIL_SEEN})!" >&2
    dump_fabric_diagnostics nccl-parity-check nccl-parity-check
    cleanup_fabric_pods
    exit 1
  fi
  if ! echo "${PARITY_MARKER}" | grep -E -q "^NCCL_PARITY_RESULT pass$"; then
    echo "ERROR: NCCL parity check reported failure or invalid marker (${PARITY_MARKER})!" >&2
    dump_fabric_diagnostics nccl-parity-check nccl-parity-check
    cleanup_fabric_pods
    exit 1
  fi
  echo "    [OK] NCCL serving-image parity verified (${PARITY_MARKER})."

  cleanup_fabric_pods
fi

# 4. Apply weights download/hydration job (if staging disk is empty or initial setup)
SERVING_ACTIVE=$(kubectl get statefulset kimi-k3-serving -n llm-serving -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [ -z "${SERVING_ACTIVE}" ]; then SERVING_ACTIVE="0"; fi

SKIP_STAGING_EXEC="false"
if [ "${SKIP_WEIGHT_JOB:-false}" != "true" ] && [ "${SKIP_WEIGHT_JOB:-false}" != "1" ] && [ "${SERVING_ACTIVE}" != "1" ] && [ "${SERVING_ACTIVE}" != "2" ]; then
  CURRENT_ACCESS_MODE=""
  DISK_PROBE_RESOLVED="false"
  DISK_PROBE_OUT=""
  if command -v gcloud >/dev/null 2>&1; then
    if DISK_PROBE_OUT=$(gcloud compute disks describe kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --format='value(accessMode)' --quiet 2>&1); then
      DISK_PROBE_RESOLVED="true"
      CURRENT_ACCESS_MODE=$(printf '%s\n' "${DISK_PROBE_OUT}" | head -n 1 | tr -d '[:space:]')
    elif printf '%s' "${DISK_PROBE_OUT}" | grep -E -qi 'was not found|notFound|does not exist'; then
      # Disk genuinely absent: this is first-time setup and staging is the correct path.
      DISK_PROBE_RESOLVED="true"
    fi
  fi

  # Fail closed. If the access mode could not be resolved -- no gcloud on PATH, an expired
  # credential, a wrong ZONE/PROJECT_ID in config.env, a transient API error -- the old code
  # left CURRENT_ACCESS_MODE empty and fell straight through to the staging branch below,
  # which deletes pvc-kimi-k3-weights-rox and pv-kimi-k3-weights-rox and then re-downloads
  # 1.45 TiB from scratch. An unreadable disk is not the same thing as an unstaged disk, and
  # guessing wrong destroys a hydrated checkpoint. Refuse to guess.
  if [ "${DISK_PROBE_RESOLVED}" != "true" ]; then
    echo "ERROR: could not determine the access mode of disk kimi-k3-weights-rox in zone '${ZONE}' (project '${PROJECT_ID}')." >&2
    echo "       Refusing to continue: the weight-staging path below deletes the ROX PVC/PV and re-stages 1.45 TiB." >&2
    echo "       Verify gcloud auth, ZONE and PROJECT_ID in scripts/config.env, then re-run." >&2
    echo "       If the disk really is absent and you intend a fresh staging pass, set SKIP_WEIGHT_JOB=false and confirm the disk name." >&2
    if [ -n "${DISK_PROBE_OUT}" ]; then
      echo "       gcloud reported: $(printf '%s' "${DISK_PROBE_OUT}" | head -n 3 | tr '\n' ' ')" >&2
    fi
    exit 1
  fi

  if [ "${CURRENT_ACCESS_MODE}" = "READ_ONLY_MANY" ]; then
    if [ "${FORCE_WEIGHT_JOB:-false}" = "true" ] || [ "${FORCE_WEIGHT_JOB:-false}" = "1" ]; then
      echo "--> Disk kimi-k3-weights-rox is in READ_ONLY_MANY mode and staging is forced."
      echo "--> Tearing down serving statefulset and volume claims before disk deletion..."
      kubectl delete statefulset kimi-k3-serving -n llm-serving --ignore-not-found=true
      kubectl delete pvc pvc-kimi-k3-weights-rox -n llm-serving --ignore-not-found=true
      kubectl delete pv pv-kimi-k3-weights-rox --ignore-not-found=true
      kubectl delete job kimi-k3-weight-staging-job -n llm-serving --ignore-not-found=true
      kubectl delete pvc pvc-kimi-k3-weights-staging -n llm-serving --ignore-not-found=true
      kubectl delete pv pv-kimi-k3-weights-staging --ignore-not-found=true

      echo "--> Deleting and recreating disk in READ_WRITE mode..."
      if ! gcloud compute disks delete kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --quiet; then
        echo "ERROR: Failed to delete disk kimi-k3-weights-rox. Ensure no workloads or PVs are holding an attachment."
        exit 1
      fi
      (cd "${TF_DIR}" && terraform apply -target=module.storage -auto-approve)
    else
      echo "--> Disk kimi-k3-weights-rox is already in READ_ONLY_MANY mode with staged weights. Skipping staging job."
      SKIP_STAGING_EXEC="true"
    fi
  fi

  if [ "${SKIP_STAGING_EXEC}" != "true" ]; then
    echo "--> 5. Preparing clean weight staging environment (removing any existing staging/serving claims)..."
    kubectl delete statefulset kimi-k3-serving -n llm-serving --ignore-not-found=true
    kubectl delete pvc pvc-kimi-k3-weights-rox -n llm-serving --ignore-not-found=true
    kubectl delete pv pv-kimi-k3-weights-rox --ignore-not-found=true
    kubectl delete job kimi-k3-weight-staging-job -n llm-serving --ignore-not-found=true
    kubectl delete pvc pvc-kimi-k3-weights-staging -n llm-serving --ignore-not-found=true
    kubectl delete pv pv-kimi-k3-weights-staging --ignore-not-found=true

    echo "--> Applying staging PV and PVC (ReadWriteOnce for 2,000 GB volume)..."
    kubectl apply -f "${GENERATED_DIR}/02-staging-pvc.yaml"

    if [ -n "${GCS_WEIGHTS_BUCKET:-}" ] && [ "${GCS_WEIGHTS_BUCKET}" != "" ] && [ "${POPULATE_WEIGHTS_CACHE:-false}" != "true" ] && [ -f "${GENERATED_DIR}/02-hydrate-weights-gcs.yaml" ]; then
      # Re-establish the serving SA's read grant on the weight backup bucket before the
      # staging job runs.
      #
      # The bucket is deliberately out-of-band: it survives 06_destroy_all.sh so a redeploy
      # does not re-download 1.45 TiB from Hugging Face. Terraform therefore owns none of
      # its IAM. But terraform DOES own kimi-k3-serving-sa, and destroying it and creating
      # it again yields the same email with a NEW unique id. GCS bindings resolve to the
      # uid, not the email, so the surviving grant decays into a dead
      # "deleted:serviceAccount:...?uid=<old>" tombstone and the new SA has no access.
      #
      # Left unhandled this is a guaranteed failure on every destroy/redeploy cycle, and it
      # surfaces ~50 minutes in as an opaque 403 inside a Kubernetes Job, after the cluster,
      # the image build and the GPU nodes have all been paid for. Establish it here, where
      # the SA is known to exist and the cost of being wrong is one API call.
      WEIGHTS_BUCKET_ROOT="$(printf '%s' "${GCS_WEIGHTS_BUCKET}" | sed -E 's#(gs://[^/]+).*#\1#')"
      SERVING_SA_EMAIL="kimi-k3-serving-sa@${PROJECT_ID}.iam.gserviceaccount.com"
      echo "--> 5a. Verifying serving SA read access to weight backup bucket ${WEIGHTS_BUCKET_ROOT}..."
      if gcloud storage buckets get-iam-policy "${WEIGHTS_BUCKET_ROOT}" \
           --project="${PROJECT_ID}" --format=json 2>/dev/null \
           | grep -q "\"serviceAccount:${SERVING_SA_EMAIL}\""; then
        echo "    [OK] ${SERVING_SA_EMAIL} already holds a binding on ${WEIGHTS_BUCKET_ROOT}."
      else
        echo "    [INFO] No live binding found (a destroy/recreate cycle invalidates it). Granting roles/storage.objectViewer..."
        if gcloud storage buckets add-iam-policy-binding "${WEIGHTS_BUCKET_ROOT}" \
             --project="${PROJECT_ID}" \
             --member="serviceAccount:${SERVING_SA_EMAIL}" \
             --role=roles/storage.objectViewer >/dev/null 2>&1; then
          echo "    [OK] Granted roles/storage.objectViewer on ${WEIGHTS_BUCKET_ROOT}."
        else
          echo "    [WARN] Could not grant roles/storage.objectViewer on ${WEIGHTS_BUCKET_ROOT}." >&2
          echo "           Hydration will fail with HTTP 403 unless an operator runs:" >&2
          echo "             gcloud storage buckets add-iam-policy-binding ${WEIGHTS_BUCKET_ROOT} \\" >&2
          echo "               --member=serviceAccount:${SERVING_SA_EMAIL} \\" >&2
          echo "               --role=roles/storage.objectViewer" >&2
        fi
      fi
      # Read-only by design: hydration only reads. Seeding the bucket is a separate,
      # explicitly opted-in path (POPULATE_WEIGHTS_CACHE=true) excluded by this branch.
      echo "--> 5b. Hydrating Kimi K3 weights directly from GCS (${GCS_WEIGHTS_BUCKET})..."
      # Arithmetic basis: 1,560,998,983,786 bytes = 1,453.7 GiB checkpoint across 96 shards.
      # At 1.0-2.5 GiB/s sustained parallel GCS read: 1453.7 / 2.5 = 581s (~10 min), 1453.7 / 1.0 = 1454s (~24 min).
      echo "    NOTE: High-throughput transfer from GCS runs at 1.0-2.5 GiB/s (~10-25 minutes total for 1,453.7 GiB)."
      kubectl apply -f "${GENERATED_DIR}/02-hydrate-weights-gcs.yaml"
      echo "    You can check job logs using: kubectl logs -n llm-serving -l app=kimi-k3-weight-staging -f"
    else
      if [ "${1:-}" != "--render-only" ]; then
        validate_hf_token
      fi
      echo "--> 5b. Applying Kimi K3 weight staging job from Hugging Face (${GENERATED_DIR}/02-download-weights.yaml)..."
      kubectl apply -f "${GENERATED_DIR}/02-download-weights.yaml"
      # Arithmetic basis: 1,560,998,983,786 bytes = 1,453.7 GiB checkpoint across 96 shards.
      # Over public internet HF CDN at 100-300 MiB/s (~0.1-0.3 GiB/s): 1453.7 / 0.3 = 1.4h (~1.5h), 1453.7 / 0.1 = 4.1h (~4h).
      echo "    NOTE: Hugging Face download takes ~1.5-4 hours for 1,453.7 GiB checkpoint over public internet (at 100-300 MiB/s)."
      echo "    You can check job logs using: kubectl logs -n llm-serving -l app=kimi-k3-weight-staging -f"
    fi
    echo "--> Waiting for weight staging job to complete (timeout: 7200s)..."
    STAGING_START=$(date +%s)
    STAGING_TIMEOUT=7200
    while true; do
      COMPLETE=$(kubectl get job kimi-k3-weight-staging-job -n llm-serving -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)
      FAILED=$(kubectl get job kimi-k3-weight-staging-job -n llm-serving -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
      NOW=$(date +%s)
      ELAPSED=$((NOW - STAGING_START))
      if [ "${COMPLETE}" = "True" ]; then
        echo "    [OK] Weight staging job completed successfully in ${ELAPSED}s."
        if [ "${POPULATE_WEIGHTS_CACHE:-false}" = "true" ] && [ -n "${GCS_WEIGHTS_BUCKET:-}" ]; then
          echo "--> POPULATE_WEIGHTS_CACHE=true: Seeding persistent GCS cache bucket (${GCS_WEIGHTS_BUCKET})..."
          BUCKET_ROOT="$(printf '%s' "${GCS_WEIGHTS_BUCKET}" | sed -E 's#(gs://[^/]+).*#\1#')"
          gcloud storage buckets create "${BUCKET_ROOT}" --project="${PROJECT_ID}" --location="${REGION}" --quiet 2>/dev/null || true
          cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: kimi-k3-cache-seeder
  namespace: llm-serving
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: kimi-k3-cache-seeder
    spec:
      serviceAccountName: kimi-k3-serving-sa
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-b200
        cloud.google.com/gke-spot: "true"
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      - key: "cloud.google.com/gke-spot"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: seeder
        image: google/cloud-sdk:500.0.0-slim
        command: ["gcloud", "storage", "rsync", "-r", "/weights", "${GCS_WEIGHTS_BUCKET}"]
        volumeMounts:
        - name: w
          mountPath: /weights
          readOnly: true
      volumes:
      - name: w
        persistentVolumeClaim:
          claimName: pvc-kimi-k3-weights-staging
EOF
          echo "    Waiting for cache seeder job to complete..."
          kubectl wait --for=condition=complete job/kimi-k3-cache-seeder -n llm-serving --timeout=3600s
          kubectl logs job/kimi-k3-cache-seeder -n llm-serving --tail=-1
          kubectl delete job kimi-k3-cache-seeder -n llm-serving --ignore-not-found=true
        fi
        break
      elif [ "${FAILED}" = "True" ]; then
        echo "ERROR: Weight staging job failed! Fetching recent job logs:"
        kubectl logs -n llm-serving -l app=kimi-k3-weight-staging --tail=50 || true
        exit 1
      elif [ "${ELAPSED}" -gt "${STAGING_TIMEOUT}" ]; then
        echo "ERROR: Weight staging job timed out after ${STAGING_TIMEOUT}s."
        kubectl logs -n llm-serving -l app=kimi-k3-weight-staging --tail=50 || true
        exit 1
      fi
      sleep 10
    done
  fi
else
  echo "--> 5. Skipping weight staging job as SKIP_WEIGHT_JOB=${SKIP_WEIGHT_JOB} or serving is already active."
fi


# Now clean up staging PVC and PV if we ran staging in this pass, releasing disk lock for ReadOnlyMany mode flip
if [ "${SKIP_STAGING_EXEC:-false}" != "true" ] && [ "${SKIP_WEIGHT_JOB:-false}" != "true" ] && [ "${SERVING_ACTIVE}" != "1" ] && [ "${SERVING_ACTIVE}" != "2" ]; then
  echo "--> Releasing READ_WRITE volume lock by removing completed staging job, PVC, and PV..."
  kubectl delete job kimi-k3-weight-staging-job -n llm-serving --ignore-not-found=true --timeout=60s || true
  kubectl delete pvc pvc-kimi-k3-weights-staging -n llm-serving --ignore-not-found=true --timeout=60s || true
  kubectl delete pv pv-kimi-k3-weights-staging --ignore-not-found=true --timeout=60s || true
fi

# 6. Multi-Node ReadOnlyMany Volume Mode Flipping
if command -v gcloud >/dev/null 2>&1; then
  CURRENT_ACCESS_MODE=$(gcloud compute disks describe kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --format='value(accessMode)' --quiet 2>/dev/null | head -n 1 || true)
  if [ "${CURRENT_ACCESS_MODE}" != "READ_ONLY_MANY" ]; then
    echo "--> Waiting for staging volume to fully detach from GKE nodes before flipping access mode..."
    DETACH_START=$(date +%s)
    DETACH_TIMEOUT=300
    while true; do
      USERS=$(gcloud compute disks describe kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --format='value(users)' --quiet 2>/dev/null || true)
      if [ -z "${USERS}" ]; then
        echo "    [OK] Volume kimi-k3-weights-rox detached cleanly."
        break
      fi
      NOW=$(date +%s)
      ELAPSED=$((NOW - DETACH_START))
      if [ "${ELAPSED}" -gt "${DETACH_TIMEOUT}" ]; then
        echo "ERROR: Timed out waiting for volume kimi-k3-weights-rox to detach after ${DETACH_TIMEOUT}s." >&2
        exit 1
      fi
      sleep 5
    done
    echo "--> Flipping Hyperdisk ML volume access mode to READ_ONLY_MANY..."
    if ! gcloud compute disks update kimi-k3-weights-rox --access-mode=READ_ONLY_MANY --zone="${ZONE}" --project="${PROJECT_ID}" --quiet; then
      echo "ERROR: Failed to update kimi-k3-weights-rox access-mode to READ_ONLY_MANY!" >&2
      exit 1
    fi
    CURRENT_ACCESS_MODE=$(gcloud compute disks describe kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --format='value(accessMode)' --quiet 2>/dev/null | head -n 1 || true)
    if [ "${CURRENT_ACCESS_MODE}" != "READ_ONLY_MANY" ]; then
      echo "ERROR: Hyperdisk ML kimi-k3-weights-rox access mode is '${CURRENT_ACCESS_MODE}', not READ_ONLY_MANY." >&2
      exit 1
    fi
  else
    echo "    [OK] Hyperdisk ML volume already operating in READ_ONLY_MANY mode."
  fi
fi

if [ "${1:-}" = "--stage-only" ]; then
  echo "--> Stage-only mode complete (weights staged and volume flipped to READ_ONLY_MANY). Exiting."
  exit 0
fi

# 7. Apply multi-node serving engine deployment
echo "--> 6. Applying ${INFERENCE_ENGINE} multi-node serving engine deployment (${GENERATED_DIR}/${SERVING_MANIFEST})..."
if [ -f "${GENERATED_DIR}/${SERVING_MANIFEST}" ]; then
  kubectl apply -f "${GENERATED_DIR}/${SERVING_MANIFEST}"
else
  echo "ERROR: Mandatory serving engine manifest (${SERVING_MANIFEST}) not found in ${GENERATED_DIR}."
  exit 1
fi

# 8. Deploy Enterprise AI Gateway & Proxy Layer
echo "--> 7. Applying Enterprise AI Gateway ConfigMap, Secret, and Deployment..."
if [ -f "${GENERATED_DIR}/04-enterprise-gateway-config.yaml" ]; then
  kubectl apply -f "${GENERATED_DIR}/04-enterprise-gateway-config.yaml"
fi
if [ -f "${GENERATED_DIR}/05-enterprise-gateway-deployment.yaml" ]; then
  kubectl apply -f "${GENERATED_DIR}/05-enterprise-gateway-deployment.yaml"
fi
if [ -f "${GENERATED_DIR}/06-model-observability-podmonitoring.yaml" ]; then
  echo "    Applying GKE AI/ML Model Observability PodMonitoring resource..."
  kubectl apply -f "${GENERATED_DIR}/06-model-observability-podmonitoring.yaml" 2>/dev/null || true
fi


echo "=============================================================================="
echo "SUCCESS: Kimi K3 workload manifests rendered and applied successfully to GKE!"
echo "To verify cluster status and serving health, run: ./scripts/04_verify_cluster.sh"
echo "=============================================================================="
