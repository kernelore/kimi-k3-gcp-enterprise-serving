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
CONFIG_FILE="${SCRIPT_DIR}/config.env"

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
if [ "${1:-}" != "--render-only" ]; then
  validate_hf_token
fi

export MODEL_REPO_ID="${MODEL_REPO_ID:-moonshotai/Kimi-K3}"
export SERVING_MODEL_NAME="${SERVING_MODEL_NAME:-kimi-k3-2.8t-mxfp4}"
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
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"

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
  if [ "${1:-}" = "--render-only" ]; then
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
  if [ "${1:-}" = "--render-only" ]; then
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
  if [ "${1:-}" = "--render-only" ]; then
    DB_CONNECTION_NAME="${PROJECT_ID}:${REGION}:kimi-k3-gateway-db"
  else
    echo "ERROR: Failed to obtain DB_CONNECTION_NAME from Terraform output or gcloud!" >&2
    exit 1
  fi
fi
export DB_CONNECTION_NAME

export TRTLLM_VIP="kimi-k3-serving-svc.llm-serving.svc.cluster.local"
export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-16}"
export SGLANG_PP_SIZE="${SGLANG_PP_SIZE:-1}"
export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-}"
export SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-16}"
export SGLANG_PORT="${SGLANG_PORT:-8000}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.90}"
export SGLANG_SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-lpm}"
export SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
export SGLANG_ENABLE_TORCH_COMPILE="${SGLANG_ENABLE_TORCH_COMPILE:-false}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
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
BASE_ALLOWED_VARS='${PROJECT_ID} ${REGION} ${ZONE} ${CLUSTER_NAME} ${OWNER_LABEL} ${TTL_LABEL} ${ENV_LABEL} ${HF_TOKEN_BASE64} ${MODEL_REPO_ID} ${SERVING_MODEL_NAME} ${TRTLLM_TP_SIZE} ${TRTLLM_PP_SIZE} ${TRTLLM_EP_SIZE} ${TRTLLM_MAX_SEQ_LEN} ${SGLANG_TP_SIZE} ${SGLANG_PP_SIZE} ${SGLANG_PP_LAYER_PARTITION} ${SGLANG_EP_SIZE} ${SGLANG_PORT} ${SGLANG_MEM_FRACTION_STATIC} ${SGLANG_SCHEDULE_POLICY} ${SGLANG_QUANTIZATION} ${SGLANG_ENABLE_TORCH_COMPILE} ${SGLANG_ATTENTION_BACKEND} ${SGLANG_CONTEXT_LENGTH} ${SGLANG_REASONING_PARSER} ${SGLANG_TOOL_CALL_PARSER} ${SGLANG_EXTRA_ARGS} ${EXPECTED_MODEL_ARCHITECTURE} ${MIN_WEIGHTS_GIB} ${LEADER_ADDR} ${HYPERDISK_ML_SIZE_GB} ${GCS_WEIGHTS_BUCKET} ${DB_CONNECTION_NAME} ${DB_PASSWORD} ${REDIS_HOST} ${REDIS_PASSWORD} ${REDIS_PASSWORD_ENCODED} ${TRTLLM_VIP} ${GPU_MAX_NODES} ${SERVING_REPLICAS} ${NODES_PER_REPLICA} ${INFERENCE_ENGINE} ${INFERENCE_SERVER_LABEL} ${SERVING_IMAGE}'
for template_file in "${TEMPLATE_DIR}"/*.yaml.template; do
  if [ -f "${template_file}" ]; then
    basename=$(basename "${template_file}" .template)
    target_file="${GENERATED_DIR}/${basename}"
    echo "    Rendering ${basename}..."
    allowed_vars="${BASE_ALLOWED_VARS}"
    if [ "${basename}" = "04-enterprise-gateway-config.yaml" ] || [ "${basename}" = "05-enterprise-gateway-deployment.yaml" ]; then
      allowed_vars="${BASE_ALLOWED_VARS} \${GATEWAY_MASTER_KEY}"
    fi
    # shellcheck disable=SC2016
    safe_envsubst "${allowed_vars}" < "${template_file}" > "${target_file}"
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

# 4. Apply weights download/hydration job (if staging disk is empty or initial setup)
SERVING_ACTIVE=$(kubectl get statefulset kimi-k3-serving -n llm-serving -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [ -z "${SERVING_ACTIVE}" ]; then SERVING_ACTIVE="0"; fi

SKIP_STAGING_EXEC="false"
if [ "${SKIP_WEIGHT_JOB:-false}" != "true" ] && [ "${SKIP_WEIGHT_JOB:-false}" != "1" ] && [ "${SERVING_ACTIVE}" != "1" ] && [ "${SERVING_ACTIVE}" != "2" ]; then
  CURRENT_ACCESS_MODE=""
  if command -v gcloud >/dev/null 2>&1; then
    CURRENT_ACCESS_MODE=$(gcloud compute disks describe kimi-k3-weights-rox --zone="${ZONE}" --project="${PROJECT_ID}" --format='value(accessMode)' --quiet 2>/dev/null | head -n 1 || true)
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
      echo "--> 5b. Hydrating Kimi K3 weights directly from GCS (${GCS_WEIGHTS_BUCKET})..."
      echo "    NOTE: High-throughput transfer from GCS runs at multi-GiB/s (~2 minutes total)."
      kubectl apply -f "${GENERATED_DIR}/02-hydrate-weights-gcs.yaml"
      echo "    You can check job logs using: kubectl logs -n llm-serving -l app=kimi-k3-weight-staging -f"
    else
      echo "--> 5b. Applying Kimi K3 weight staging job from Hugging Face (${GENERATED_DIR}/02-download-weights.yaml)..."
      kubectl apply -f "${GENERATED_DIR}/02-download-weights.yaml"
      echo "    NOTE: Hugging Face download takes ~15-20 min for 2.8T checkpoints."
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

if [ "${1:-}" = "--stage-only" ]; then
  echo "--> Stage-only mode complete. Exiting."
  exit 0
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
      if [ $((NOW - DETACH_START)) -gt ${DETACH_TIMEOUT} ]; then
        echo "ERROR: Timed out waiting for kimi-k3-weights-rox to detach! Users still attached: ${USERS}" >&2
        exit 1
      fi
      sleep 5
    done
    echo "--> Setting Hyperdisk ML volume access mode to READ_ONLY_MANY for multi-node MPI attach..."
    if ! gcloud compute disks update kimi-k3-weights-rox \
      --access-mode=READ_ONLY_MANY --zone="${ZONE}" --project="${PROJECT_ID}" --quiet; then
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
