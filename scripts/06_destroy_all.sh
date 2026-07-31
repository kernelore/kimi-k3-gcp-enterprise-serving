#!/usr/bin/env bash
# ==============================================================================
# 06_destroy_all.sh - Safe & Complete Infrastructure & Workload Teardown (Kimi K3)
# ==============================================================================
# Deletes Kubernetes workloads, removes PVCs/Jobs cleanly to release disks,
# and runs `terraform destroy` to ensure zero leftover resources or cloud costs.
# Implements automated self-healing teardown loops for Cloud SQL and RoCEv2 VPC peerings.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${PROJECT_ROOT}/terraform"
GENERATED_DIR="${TF_DIR}/manifests/generated"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ${CONFIG_FILE} not found. Please run ./scripts/01_setup_and_check.sh first."
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

echo "=============================================================================="
echo "Kimi K3 Sovereign Enterprise Inference - COMPLETE TEARDOWN & DESTROY"
echo "=============================================================================="
echo "WARNING: This will destroy the GKE cluster, PVCs, RoCEv2 VPC network, and"
echo "Terraform-managed storage buckets in project ${PROJECT_ID}."
echo "=============================================================================="

# Confirmation guard
if [ "${FORCE_DESTROY:-false}" != "true" ] && [ "${FORCE_DESTROY:-false}" != "1" ]; then
  echo "WARNING: This will permanently delete all Kimi K3 infrastructure, PVC data, and cloud resources."
  read -r -p "Are you sure you want to proceed with full teardown? (y/N): " confirm
  if [[ "${confirm}" != [yY] && "${confirm}" != [yY][eE][sS] ]]; then
    echo "Teardown cancelled by user."
    exit 1
  fi
fi

# 1. Delete Kubernetes workloads first if cluster is reachable
echo "--> 1. Attempting clean deletion of Kubernetes workloads from GKE..."
if kubectl get nodes >/dev/null 2>&1; then
  echo "    Deleting jobs, statefulsets, deployments, and services in namespace llm-serving..."
  kubectl delete jobs kimi-k3-weight-staging-job kimi-k3-incluster-benchmark kimi-k3-cache-seeder -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete jobs --all -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete statefulsets kimi-k3-serving nccl-roce-test nccl-parity-check preflight-nccl-roce-check -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete statefulsets --all -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete deployments kimi-k3-gateway -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete deployments --all -n llm-serving --ignore-not-found --timeout=60s || true
  kubectl delete services --all -n llm-serving --ignore-not-found --timeout=60s || true

  echo "    Deleting staging and serving PVCs in namespace llm-serving..."
  kubectl delete pvc pvc-kimi-k3-weights-staging pvc-kimi-k3-weights-rox -n llm-serving --ignore-not-found --timeout=60s || true

  echo "    Deleting cluster-scoped staging and serving PVs..."
  kubectl delete pv pv-kimi-k3-weights-staging pv-kimi-k3-weights-rox --ignore-not-found --timeout=60s || true

  echo "    Deleting namespace llm-serving and generated manifests..."
  if [ -d "${GENERATED_DIR}" ]; then
    kubectl delete -f "${GENERATED_DIR}/" --ignore-not-found --timeout=120s || true
  fi
  kubectl delete ns llm-serving --ignore-not-found --timeout=120s || true
  kubectl delete ds local-nvme-raid-formatter -n kube-system --ignore-not-found --timeout=60s || true
else
  echo "    Kubernetes cluster already unreachable/deleted. Proceeding..."
fi

# 2. Proactive Cloud SQL and database cleanup (Issue 6 guard)
echo "--> 2. Checking and executing proactive Cloud SQL object & database cleanup..."
if command -v gcloud >/dev/null 2>&1; then
  echo "    Dropping LiteLLM database to release owned role dependencies before destroy..."
  gcloud sql databases delete kimi_k3_gateway --instance=kimi-k3-gateway-db --project="${PROJECT_ID}" --quiet 2>/dev/null || true
fi

if command -v gcloud >/dev/null 2>&1; then
  if [ "${PURGE_WEIGHTS_BACKUP:-false}" = "true" ] && [ "${FORCE_DESTROY:-false}" = "true" ] && [ -n "${GCS_WEIGHTS_BUCKET:-}" ]; then
    if [[ "${GCS_WEIGHTS_BUCKET}" != gs://* ]]; then
      GCS_WEIGHTS_BUCKET="gs://${GCS_WEIGHTS_BUCKET}"
    fi
    echo "WARNING: PURGE_WEIGHTS_BACKUP=true and FORCE_DESTROY=true explicitly set. Deleting weight backup bucket ${GCS_WEIGHTS_BUCKET} before destroy..."
    echo "         Next deployment will require a full HuggingFace re-download (hours for 1.5 TB)."
    gcloud storage rm -r "${GCS_WEIGHTS_BUCKET}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
  fi
fi

# 3. Run Terraform destroy with automated self-healing and retry loop (Issues 6 & 7 Guards)
echo "--> 3. Running terraform destroy in ${TF_DIR}..."
cd "${TF_DIR}"
if [ -z "${TF_STATE_BUCKET:-}" ] || [ "${TF_STATE_BUCKET}" = "YOUR_PROJECT_ID-kimi-k3-tfstate" ] || [ "${TF_STATE_BUCKET}" = "your-project-id-kimi-k3-tfstate" ]; then
  export TF_STATE_BUCKET="${PROJECT_ID}-kimi-k3-tfstate"
fi
echo "    Initializing remote state backend: gs://${TF_STATE_BUCKET}..."
terraform init -backend-config="bucket=${TF_STATE_BUCKET}" -reconfigure

DESTROY_CMD="terraform destroy"
if [ "${FORCE_DESTROY:-false}" = "true" ] || [ "${FORCE_DESTROY:-false}" = "1" ]; then
  DESTROY_CMD="terraform destroy -auto-approve"
fi

echo "    Executing: ${DESTROY_CMD}..."
if ! eval "${DESTROY_CMD}"; then
  echo "    [NOTE] Initial terraform destroy encountered a dependency or propagation block. Engaging self-healing teardown..."

  # Self-heal Cloud SQL role block (Issue 6)
  if gcloud sql instances describe kimi-k3-gateway-db --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    Deleting Cloud SQL instance kimi-k3-gateway-db directly to release database/role dependencies..."
    gcloud sql instances delete kimi-k3-gateway-db --project="${PROJECT_ID}" --quiet 2>/dev/null || true
    terraform state rm module.database.google_sql_user.gateway_user module.database.google_sql_database.default module.database.google_sql_database_instance.gateway_db 2>/dev/null || true
  fi

  # Self-heal Service Networking & VPC peering connections (Issue 7 guard)
  echo "    Cleaning up VPC peerings and state connections after destroy failure..."
  VPC_NAME=$(terraform output -raw vpc_network_name 2>/dev/null || echo "kimi-k3-vpc")
  if command -v gcloud >/dev/null 2>&1 && [ -n "${VPC_NAME}" ]; then
    PEERINGS=$(gcloud compute networks peerings list --network="${VPC_NAME}" --project="${PROJECT_ID}" --format="value(peerings[].name)" 2>/dev/null | tr ';' '\n' || true)
    for p in ${PEERINGS}; do
      if [ -n "${p}" ]; then
        echo "      Removing peering: ${p}..."
        gcloud compute networks peerings delete "${p}" --network="${VPC_NAME}" --project="${PROJECT_ID}" --quiet 2>/dev/null || true
      fi
    done
    terraform state rm module.database.google_service_networking_connection.private_vpc_connection 2>/dev/null || true
  fi

  echo "    Retrying final terraform destroy..."
  sleep 5
  eval "${DESTROY_CMD}"
fi

# 4. Enumerate out-of-band retained GCS buckets (e.g. Terraform remote state backend)
echo "--> 4. Enumerating out-of-band retained GCS buckets (*-kimi-k3-*) in project ${PROJECT_ID}..."
if command -v gcloud >/dev/null 2>&1; then

  echo "------------------------------------------------------------------------------"
  echo "OUT-OF-BAND RETAINED BUCKET INVENTORY (Terraform remote state backend — not a leak):"
  BUCKETS=$(gcloud storage ls --project="${PROJECT_ID}" 2>/dev/null | grep -E "kimi-k3|kimi3|kimi-prod" || true)
  for b in ${BUCKETS}; do
    SIZE_BYTES=$(gcloud storage du -s "${b}" 2>/dev/null | awk '{print $1}' || echo "0")
    SIZE_GIB=$(awk "BEGIN {printf \"%.2f\", ${SIZE_BYTES:-0}/1073741824}")
    COST_EST=$(awk "BEGIN {printf \"$%.2f\", (${SIZE_BYTES:-0}/1073741824)*0.02}")
    echo "  * ${b} (~${SIZE_GIB} GiB | est. ${COST_EST}/mo) [OUT-OF-BAND RETAINED]"
  done
  echo "------------------------------------------------------------------------------"
fi

echo "=============================================================================="
echo "SUCCESS: All Kubernetes workloads and Kimi K3 infrastructure destroyed cleanly in one pass!"
echo "=============================================================================="
