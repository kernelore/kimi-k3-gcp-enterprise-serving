#!/usr/bin/env bash
# ==============================================================================
# 02_deploy_infra.sh - Provision GCP Infrastructure via Terraform (Kimi K3)
# ==============================================================================
# Initializes Terraform, validates configuration, applies all VPC, GKE, Storage,
# and IAM resources, and automatically fetches cluster credentials.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${PROJECT_ROOT}/terraform"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ${CONFIG_FILE} not found. Please run ./scripts/01_setup_and_check.sh first."
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

echo "=============================================================================="
echo "Kimi K3 Sovereign Enterprise Inference - Phase 1 & 2: Terraform Deployment"
echo "=============================================================================="
echo "Target Project: ${PROJECT_ID}"
echo "Target Region:  ${REGION} / Zone: ${ZONE}"
echo "Cluster Name:   ${CLUSTER_NAME}"
echo "=============================================================================="

cd "${TF_DIR}"

# 1. Initialize Terraform with GCS remote state backend
if [ -z "${TF_STATE_BUCKET:-}" ] || [ "${TF_STATE_BUCKET}" = "YOUR_PROJECT_ID-kimi-k3-tfstate" ] || [ "${TF_STATE_BUCKET}" = "your-project-id-kimi-k3-tfstate" ]; then
  export TF_STATE_BUCKET="${PROJECT_ID}-kimi-k3-tfstate"
fi
echo "--> 1. Initializing Terraform with GCS remote state bucket: gs://${TF_STATE_BUCKET}..."
terraform init -backend-config="bucket=${TF_STATE_BUCKET}" -reconfigure

# Check for pre-existing remote state in the shared bucket (Issue 10 guard)
echo "--> Checking existing remote state in gs://${TF_STATE_BUCKET}..."
EXISTING_STATE_RESOURCES=$(terraform state list 2>/dev/null || true)
if [ -n "${EXISTING_STATE_RESOURCES}" ]; then
  RESOURCE_COUNT=$(echo "${EXISTING_STATE_RESOURCES}" | grep -c -v "^$" || echo "0")
  echo "    [WARNING] Found existing non-empty Terraform state (${RESOURCE_COUNT} resources tracked):"
  echo "${EXISTING_STATE_RESOURCES}" | head -n 10 | sed 's/^/      - /'
  if [ "${RESOURCE_COUNT}" -gt 10 ]; then
    echo "      ... and $((RESOURCE_COUNT - 10)) more tracked resources."
  fi
  echo "    NOTE: If this state is from a previous or mismatched deployment, review with 'terraform state list'"
  echo "          or run './scripts/06_destroy_all.sh' / purge the state bucket before re-applying."
else
  echo "    [OK] Clean or empty Terraform remote state backend confirmed."
fi

# 2. Validate Terraform configuration
echo "--> 2. Validating Terraform configuration syntax..."
terraform validate

# 3. Apply Terraform configuration
echo "--> 3. Applying Terraform configuration (VPC RoCEv2, GKE Cluster, Hyperdisk, IAM)..."
if [ "${AUTO_APPROVE:-false}" = "true" ] || [ "${AUTO_APPROVE:-false}" = "1" ]; then
  echo "    Applying with -auto-approve..."
  terraform apply -auto-approve
else
  terraform apply
fi

# 4. Fetch outputs and configure kubectl credentials
echo "--> 4. Retrieving Terraform outputs & configuring kubectl cluster credentials..."
CLUSTER_ENDPOINT=$(terraform output -raw gke_cluster_endpoint 2>/dev/null || true)
VPC_NAME=$(terraform output -raw vpc_network_name 2>/dev/null || true)
TRAJECTORY_BUCKET=$(terraform output -raw trajectory_bucket_name 2>/dev/null || true)

echo "    Cluster Control Plane Endpoint: ${CLUSTER_ENDPOINT}"
echo "    RoCEv2 VPC Network Name:        ${VPC_NAME}"
echo "    Trajectory Storage Bucket:      ${TRAJECTORY_BUCKET}"

echo "--> 5. Fetching GKE cluster credentials..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT_ID}"

echo "--> 6. Verifying cluster connection..."
kubectl get nodes

echo "=============================================================================="
echo "SUCCESS: Infrastructure provisioned and GKE credentials configured cleanly!"
echo "Next step: Run ./scripts/03_deploy_workloads.sh to deploy Kimi K3 manifests."
echo "=============================================================================="
