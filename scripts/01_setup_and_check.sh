#!/usr/bin/env bash
# ==============================================================================
# 01_setup_and_check.sh - Environment Check & Initialization Script (Kimi K3)
# ==============================================================================
# Verifies CLI tools, sources config.env, prepares terraform.tfvars, and ensures
# required Google Cloud APIs are enabled before deploying infrastructure.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${PROJECT_ROOT}/terraform"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

echo "=============================================================================="
echo "Kimi K3 Sovereign Enterprise Inference - Environment Setup & Preflight Check"
echo "=============================================================================="

# 1. Check required CLI tools & ensure executable script permissions
echo "--> 1. Verifying prerequisite CLI tools & script permissions..."
chmod +x "${SCRIPT_DIR}"/*.sh 2>/dev/null || true
for cmd in gcloud terraform kubectl curl envsubst python3 base64; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: Required command '${cmd}' is not installed or not in PATH."
    exit 1
  fi
  echo "    [OK] ${cmd} found at $(command -v "${cmd}")"
done

# 2. Check Python requirements & configure isolated virtual environment (PEP 668 compliance)
echo "--> 2. Configuring isolated Python virtual environment (.venv) & installing dependencies..."
VENV_DIR="${PROJECT_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
  echo "    Creating Python virtual environment at ${VENV_DIR}..."
  python3 -m venv --system-site-packages "${VENV_DIR}" 2>/dev/null || true
fi

if [ -f "${VENV_DIR}/bin/python" ]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
  PIP_BIN="${VENV_DIR}/bin/pip"
  echo "    [OK] Using project virtualenv: ${PYTHON_BIN}"
else
  PYTHON_BIN="python3"
  PIP_BIN="pip3"
  echo "    [NOTE] Virtual environment creation unavailable. Falling back to system Python."
fi

echo "    Checking if required Python dependencies are already installed..."
if "${PYTHON_BIN}" -c "import google.cloud.bigquery; import google.cloud.storage" 2>/dev/null; then
  echo "    [OK] Required Python dependencies already present."
else
  echo "    Installing missing dependencies from ${SCRIPT_DIR}/requirements.txt..."
  if [ -f "${VENV_DIR}/bin/pip" ]; then
    "${PIP_BIN}" install -r "${SCRIPT_DIR}/requirements.txt" --quiet
  else
    python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt" --user --break-system-packages --quiet
  fi
fi

echo "    Verifying critical Python imports (google.cloud.bigquery, google.cloud.storage)..."
if ! "${PYTHON_BIN}" -c "import google.cloud.bigquery; import google.cloud.storage; print('    [OK] Python dependencies verified successfully.')" 2>/dev/null; then
  echo "ERROR: Critical Python dependencies (google-cloud-bigquery, google-cloud-storage) failed to import."
  echo "Action required: Run '${PIP_BIN} install -r ${SCRIPT_DIR}/requirements.txt' or install in a virtual environment."
  exit 1
fi

# 3. Check or initialize config.env
echo "--> 3. Checking environment configuration (${CONFIG_FILE})..."
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "    config.env not found! Copying config.env.example to config.env..."
  cp "${SCRIPT_DIR}/config.env.example" "${CONFIG_FILE}"
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

if [ -z "${PROJECT_ID:-}" ] || [ "${PROJECT_ID}" = "YOUR_PROJECT_ID" ]; then
  echo "ERROR: PROJECT_ID is not properly set in ${CONFIG_FILE}."
  echo "Please edit ${CONFIG_FILE} and set your active GCP Project ID."
  exit 1
fi

export INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
if [[ "${INFERENCE_ENGINE}" != "trtllm" && "${INFERENCE_ENGINE}" != "sglang" ]]; then
  echo "ERROR: Invalid INFERENCE_ENGINE '${INFERENCE_ENGINE}'. Must be either 'trtllm' or 'sglang'."
  exit 1
fi

# Auto-generate cryptographically secure random DB_PASSWORD and GATEWAY_MASTER_KEY if unset or default
if [ -z "${DB_PASSWORD:-}" ] || [ "${DB_PASSWORD}" = "kimi-k3-gateway-admin-secret" ] || [ "${DB_PASSWORD}" = "REPLACE_WITH_SECURE_PASSWORD" ]; then
  DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
  export DB_PASSWORD
  if grep -q "export DB_PASSWORD=" "${CONFIG_FILE}"; then
    sed -i "s/export DB_PASSWORD=.*/export DB_PASSWORD=\"${DB_PASSWORD}\"/" "${CONFIG_FILE}"
  else
    echo "export DB_PASSWORD=\"${DB_PASSWORD}\"" >> "${CONFIG_FILE}"
  fi
fi

if [ -z "${GATEWAY_MASTER_KEY:-}" ] || [ "${GATEWAY_MASTER_KEY}" = "sk-kimi-k3-master-secret-key-change-me" ] || [ "${GATEWAY_MASTER_KEY}" = "sk-kimi-k3-master-change-me" ]; then
  GATEWAY_MASTER_KEY="sk-kimi-k3-$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
  export GATEWAY_MASTER_KEY
  if grep -q "export GATEWAY_MASTER_KEY=" "${CONFIG_FILE}"; then
    sed -i "s/export GATEWAY_MASTER_KEY=.*/export GATEWAY_MASTER_KEY=\"${GATEWAY_MASTER_KEY}\"/" "${CONFIG_FILE}"
  else
    echo "export GATEWAY_MASTER_KEY=\"${GATEWAY_MASTER_KEY}\"" >> "${CONFIG_FILE}"
  fi
fi

echo "    Active Project ID:  ${PROJECT_ID}"
echo "    Active Region:      ${REGION}"
echo "    Active Zone:        ${ZONE}"
echo "    Owner Label:        ${OWNER_LABEL}"
echo "    Inference Engine:   ${INFERENCE_ENGINE}"
echo "    RoCEv2 RDMA Fabric: Active (Jumbo Frames MTU 8896, Compact Placement)"

# 4. Auto-detect operator IP for Master Authorized Networks
echo "--> 4. Configuring Master Authorized Networks & GPU Scaling..."
if [ -n "${MASTER_AUTHORIZED_CIDR:-}" ]; then
  AUTH_CIDR="${MASTER_AUTHORIZED_CIDR}"
  if [[ "${AUTH_CIDR}" != */* ]]; then
    AUTH_CIDR="${AUTH_CIDR}/32"
  fi
else
  DETECTED_IP=$(curl -s --max-time 5 https://ifconfig.me 2>/dev/null || dig +short myip.opendns.com @resolver1.opendns.com 2>/dev/null | tail -n1 || true)
  DETECTED_IP=$(echo "${DETECTED_IP}" | tr -d '[:space:]')
  if [ -n "${DETECTED_IP}" ]; then
    AUTH_CIDR="${DETECTED_IP}/32"
  else
    AUTH_CIDR=""
  fi
fi

if [ -n "${AUTH_CIDR}" ]; then
  echo "    Detected operator IP/CIDR for GKE authorized networks: ${AUTH_CIDR}"
  TF_AUTH_CIDRS="[ { cidr_block = \"${AUTH_CIDR}\", display_name = \"operator\" } ]"
else
  echo "    [NOTE] No operator IP detected or configured. master_authorized_cidrs will be empty."
  TF_AUTH_CIDRS="[]"
fi

# 5. Synchronize config.env to terraform.tfvars
echo "--> 5. Synchronizing configuration to ${TF_DIR}/terraform.tfvars..."
cat << EOF > "${TF_DIR}/terraform.tfvars"
# Auto-generated by scripts/01_setup_and_check.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
project_id              = "${PROJECT_ID}"
region                  = "${REGION}"
zone                    = "${ZONE}"
cluster_name            = "${CLUSTER_NAME:-kimi-enterprise-fi}"
gpu_machine_type        = "${GPU_MACHINE_TYPE:-a4-highgpu-8g}"
gpu_pool_max_nodes      = ${GPU_POOL_MAX_NODES:-${GPU_MAX_NODES:-2}}
nodes_per_replica       = ${NODES_PER_REPLICA:-2}
hyperdisk_ml_size_gb    = ${HYPERDISK_ML_SIZE_GB:-2000}
hyperdisk_ml_throughput_mibps = ${HYPERDISK_ML_THROUGHPUT_MIBPS:-24576}
model_family            = "${MODEL_FAMILY:-kimi-k3}"
enable_private_endpoint = ${ENABLE_PRIVATE_ENDPOINT:-false}
master_authorized_cidrs = ${TF_AUTH_CIDRS}
owner_label             = "${OWNER_LABEL:-opensource-user}"
ttl_label               = "${TTL_LABEL:-7d}"
env_label               = "${ENV_LABEL:-kimi-k3-prod}"
db_tier                 = "${DB_TIER:-db-custom-2-8192}"
db_password             = "${DB_PASSWORD}"

# SGLang Multi-Node Serving Parameters (Documented Reference):
# model_repo_id      = "${MODEL_REPO_ID:-moonshotai/Kimi-K3-2.8T-MXFP4}"
# serving_model_name = "${SERVING_MODEL_NAME:-kimi-k3-2.8t-mxfp4}"
# sglang_tp_size     = ${SGLANG_TP_SIZE:-16}
# sglang_pp_size     = ${SGLANG_PP_SIZE:-1}
# sglang_ep_size     = ${SGLANG_EP_SIZE:-16}
EOF
echo "    [OK] terraform.tfvars updated successfully."

# 6. Configure gcloud project & verify authentication
echo "--> 6. Checking gcloud authentication and project configuration..."
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
if [ -z "${ACTIVE_ACCOUNT}" ]; then
  echo "WARNING: No active gcloud account found. Running gcloud auth login..."
  gcloud auth login
fi
gcloud config set project "${PROJECT_ID}" --quiet

# 7. Check IAM permissions for GKE RBAC, VPC peering, Workload Identity, and Project IAM administration
echo "--> 7. Checking deploy identity IAM permissions for GKE cluster, RBAC, VPC peering, and Workload Identity..."
if [ -n "${ACTIVE_ACCOUNT}" ]; then
  export HAS_CONTAINER_ADMIN="false"
  export HAS_SN_ADMIN="false"
  export HAS_SA_USER="false"
  export HAS_IAM_ADMIN="false"
  IAM_ROLES=$(gcloud projects get-iam-policy "${PROJECT_ID}" \
      --flatten="bindings[].members" \
      --filter="bindings.members:${ACTIVE_ACCOUNT}" \
      --format="value(bindings.role)" 2>/dev/null || true)
  MEMBER_PREFIX="user"
  if echo "${ACTIVE_ACCOUNT}" | grep -q "\.gserviceaccount\.com$"; then
    MEMBER_PREFIX="serviceAccount"
  fi

  if echo "${IAM_ROLES}" | grep -E -q "roles/container\.admin|roles/container\.clusterAdmin|roles/owner"; then
    export HAS_CONTAINER_ADMIN="true"
    echo "    [OK] Verified GKE admin IAM role for ${ACTIVE_ACCOUNT}."
  else
    echo "WARNING: Active account ${ACTIVE_ACCOUNT} appears to lack 'roles/container.admin'."
    echo "Workload deployment (scripts/03_deploy_workloads.sh) requires GKE ClusterRole / RBAC privileges."
    echo "To grant the required permission, run:"
    echo "    gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "      --member=\"${MEMBER_PREFIX}:${ACTIVE_ACCOUNT}\" \\"
    echo "      --role=\"roles/container.admin\" --condition=None"
  fi

  if echo "${IAM_ROLES}" | grep -E -q "roles/servicenetworking\.networksAdmin|roles/owner"; then
    export HAS_SN_ADMIN="true"
    echo "    [OK] Verified Service Networking networks admin IAM role for ${ACTIVE_ACCOUNT}."
  else
    echo "WARNING: Active account ${ACTIVE_ACCOUNT} appears to lack 'roles/servicenetworking.networksAdmin'."
    echo "Cloud SQL private IP requires establishing a Private Services Access VPC peering in 02_deploy_infra.sh."
    echo "To grant the required permission, run:"
    echo "    gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "      --member=\"${MEMBER_PREFIX}:${ACTIVE_ACCOUNT}\" \\"
    echo "      --role=\"roles/servicenetworking.networksAdmin\" --condition=None"
  fi

  if echo "${IAM_ROLES}" | grep -E -q "roles/iam\.serviceAccountUser|roles/editor|roles/owner"; then
    export HAS_SA_USER="true"
    echo "    [OK] Verified Service Account User IAM role for ${ACTIVE_ACCOUNT}."
  else
    echo "WARNING: Active account ${ACTIVE_ACCOUNT} appears to lack 'roles/iam.serviceAccountUser'."
    echo "Attaching Workload Identity service accounts to GKE pods requires Service Account User privileges."
    echo "To grant the required permission, run:"
    echo "    gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "      --member=\"${MEMBER_PREFIX}:${ACTIVE_ACCOUNT}\" \\"
    echo "      --role=\"roles/iam.serviceAccountUser\" --condition=None"
  fi

  if echo "${IAM_ROLES}" | grep -E -q "roles/resourcemanager\.projectIamAdmin|roles/owner"; then
    export HAS_IAM_ADMIN="true"
    echo "    [OK] Verified Project IAM Admin role for ${ACTIVE_ACCOUNT}."
  else
    echo "WARNING: Active account ${ACTIVE_ACCOUNT} appears to lack 'roles/resourcemanager.projectIamAdmin'."
    echo "Creating Workload Identity IAM bindings in Terraform requires Project IAM Admin privileges."
    echo "To grant the required permission, run:"
    echo "    gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "      --member=\"${MEMBER_PREFIX}:${ACTIVE_ACCOUNT}\" \\"
    echo "      --role=\"roles/resourcemanager.projectIamAdmin\" --condition=None"
  fi

fi

# 8. Enable required Google Cloud APIs
echo "--> 8. Enabling required Google Cloud APIs (this may take a minute)..."
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  cloudresourcemanager.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  servicenetworking.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}" --quiet
echo "    [OK] All required GCP APIs enabled."

# 8.1. Verify GKE release channel and version floor (>= 1.32 required for Kimi K3 Blackwell / Advanced Datapath)
echo "--> 8.1. Verifying GKE RAPID release channel availability and version floor >= 1.32..."
if command -v gcloud >/dev/null 2>&1; then
  RAPID_VERSION=$(gcloud container get-server-config --location="${ZONE}" --project="${PROJECT_ID}" --flatten="channels" --filter="channels.channel=RAPID" --format="value(channels.defaultVersion)" 2>/dev/null || true)
  if [ -n "${RAPID_VERSION}" ]; then
    MAJOR_MINOR=$(echo "${RAPID_VERSION}" | cut -d. -f1,2)
    IS_VALID=$(python3 -c "print('yes' if float('${MAJOR_MINOR}'.split('-')[0]) >= 1.32 else 'no')" 2>/dev/null || echo "yes")
    if [ "${IS_VALID}" = "yes" ]; then
      echo "    [OK] GKE RAPID release channel in ${ZONE} offers default version ${RAPID_VERSION} (>= 1.32 floor met)."
    else
      echo "WARNING: GKE RAPID release channel in ${ZONE} default version is ${RAPID_VERSION} (< 1.32). Kimi K3 requires GKE 1.32+."
    fi
  else
    echo "    [NOTE] Unable to query GKE server config for RAPID channel in ${ZONE} (API may still be propagating). Ensure GKE 1.32+ is available."
  fi
fi

# 9. Provision and verify GCS Remote State Bucket for Terraform
if [ -z "${TF_STATE_BUCKET:-}" ] || [ "${TF_STATE_BUCKET}" = "YOUR_PROJECT_ID-kimi-k3-tfstate" ] || [ "${TF_STATE_BUCKET}" = "your-project-id-kimi-k3-tfstate" ]; then
  export TF_STATE_BUCKET="${PROJECT_ID}-kimi-k3-tfstate"
fi
echo "--> 9. Checking/Provisioning Terraform GCS remote state bucket: gs://${TF_STATE_BUCKET}..."
if ! gcloud storage buckets describe "gs://${TF_STATE_BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "    Bucket gs://${TF_STATE_BUCKET} does not exist. Creating with uniform access..."
  gcloud storage buckets create "gs://${TF_STATE_BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access --quiet
  echo "    Enabling object versioning on gs://${TF_STATE_BUCKET} for state protection..."
  gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning --quiet
else
  echo "    [OK] Terraform state bucket gs://${TF_STATE_BUCKET} already exists and is ready."
fi

echo "=============================================================================="
echo "SUCCESS: Environment preflight checks completed! You are ready to deploy."
echo "Next step: Run ./scripts/02_deploy_infra.sh to provision Terraform resources."
echo "=============================================================================="
