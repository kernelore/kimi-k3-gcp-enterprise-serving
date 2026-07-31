#!/bin/bash
# ==============================================================================
# test_cases_t5.sh - Tier 5: Adversarial White-Box Coverage Hardening Suite
# ==============================================================================
# Synthesizes and integrates all Tier 5 adversarial test cases from Challenger 1
# and Challenger 2, closing all 14 white-box coverage gaps across templates,
# scripts, containers, and Build, while verifying remediated bugs (ADV-T5-07..09).
# ==============================================================================
set -euo pipefail

# T5_ADV_01: Domain Specializations in Rendered Serving & Staging Manifests
t5_adv_01() {
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="trtllm" CLUSTER_NAME="test-cluster" REGION="europe-north1" ZONE="europe-north1-b" PROJECT_ID="test-proj" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local trtllm_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  assert_file_exists "${trtllm_yaml}"
  assert_match 'gke-spot: "true"' "${trtllm_yaml}" "Missing gke-spot nodeSelector in TRTLLM manifest"
  assert_match 'nvidia-b200' "${trtllm_yaml}" "Missing nvidia-b200 nodeSelector in TRTLLM manifest"
  assert_match 'tolerations:' "${trtllm_yaml}" "Missing spot tolerations in TRTLLM manifest"
  for i in $(seq 0 7); do
    assert_match "rdma-${i}" "${trtllm_yaml}" "Missing rdma-${i} annotation in TRTLLM manifest"
  done
  assert_match 'sizeLimit: 512Gi' "${trtllm_yaml}" "Missing 512Gi /dev/shm sizeLimit in TRTLLM manifest"
  assert_match 'storage: 2000Gi' "${trtllm_yaml}" "Missing 2000Gi storage claim in TRTLLM manifest"
  assert_match 'startupProbe:' "${trtllm_yaml}" "Missing startupProbe in TRTLLM manifest"
  assert_match 'apps.kubernetes.io/pod-index: "0"' "${trtllm_yaml}" "Missing leader-only pod-index selector in TRTLLM manifest"
  assert_match '/home/kubernetes/bin/nvidia' "${trtllm_yaml}" "Missing /home/kubernetes/bin/nvidia hostPath in TRTLLM manifest"
  assert_match '/home/kubernetes/bin/gib' "${trtllm_yaml}" "Missing /home/kubernetes/bin/gib hostPath in TRTLLM manifest"

  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="sglang" CLUSTER_NAME="test-cluster" REGION="europe-north1" ZONE="europe-north1-b" PROJECT_ID="test-proj" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  assert_file_exists "${sglang_yaml}"
  assert_match 'gke-spot: "true"' "${sglang_yaml}" "Missing gke-spot nodeSelector in SGLang manifest"
  assert_match 'nvidia-b200' "${sglang_yaml}" "Missing nvidia-b200 nodeSelector in SGLang manifest"
  for i in $(seq 0 7); do
    assert_match "rdma-${i}" "${sglang_yaml}" "Missing rdma-${i} annotation in SGLang manifest"
  done
  assert_match 'sizeLimit: 512Gi' "${sglang_yaml}" "Missing 512Gi /dev/shm sizeLimit in SGLang manifest"
  assert_match 'storage: 2000Gi' "${sglang_yaml}" "Missing 2000Gi storage claim in SGLang manifest"
  assert_match 'startupProbe:' "${sglang_yaml}" "Missing startupProbe in SGLang manifest"
  assert_match 'apps.kubernetes.io/pod-index: "0"' "${sglang_yaml}" "Missing leader-only pod-index selector in SGLang manifest"
  assert_match '/home/kubernetes/bin/nvidia' "${sglang_yaml}" "Missing /home/kubernetes/bin/nvidia hostPath in SGLang manifest"
  assert_match '/home/kubernetes/bin/gib' "${sglang_yaml}" "Missing /home/kubernetes/bin/gib hostPath in SGLang manifest"
  assert_match 'trust-remote-code' "${sglang_yaml}" "Missing trust-remote-code flag in SGLang manifest"
  assert_match 'enable-metrics' "${sglang_yaml}" "Missing enable-metrics flag in SGLang manifest"
  assert_match 'context-length ["]?131072' "${sglang_yaml}" "Missing context-length 131072 flag in SGLang manifest"

  local staging_pvc="${PROJECT_ROOT}/terraform/manifests/templates/02-staging-pvc.yaml.template"
  assert_match 'storage: 2000Gi' "${staging_pvc}" "Missing 2000Gi storage claim in staging PVC template"
  assert_no_match 'ENABLE_HPA' "${PROJECT_ROOT}/scripts/config.env.example" "Orphaned ENABLE_HPA still present in config.env.example"

  local turndown_yaml="${PROJECT_ROOT}/terraform/manifests/generated/10-scheduled-turndown-cronjob.yaml"
  assert_file_exists "${turndown_yaml}"
  assert_match 'scale statefulset kimi-k3-serving --replicas=0' "${turndown_yaml}" "Scheduled turndown manifest missing --replicas=0"
  assert_match 'scale statefulset kimi-k3-serving --replicas=2' "${turndown_yaml}" "Scheduled turnup manifest missing --replicas=2"
}



# T5_ADV_02: Hermeticity, Empty String Injection Resilience, and Lifecycle Templates
t5_adv_02() {
  local job_template="${PROJECT_ROOT}/terraform/manifests/templates/08-in-cluster-benchmark-job.yaml.template"
  assert_no_match 'pip install|apt-get install' "${job_template}" "Prohibited runtime package installation in benchmark job template"

  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="trtllm" CLUSTER_NAME="test-cluster" REGION="europe-north1" ZONE="europe-north1-b" PROJECT_ID="test-proj" TRTLLM_TP_SIZE="" HYPERDISK_ML_SIZE_GB="" EP_SIZE="" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local trtllm_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  assert_match 'tp_size.*8|TP_SIZE.*8' "${trtllm_yaml}" "Failed to fallback to TP=8 under empty string injection"
  assert_match 'storage: 2000Gi' "${trtllm_yaml}" "Failed to fallback to 2000Gi storage under empty string injection"

  assert_match 'MODEL_REPO_ID|Kimi-K3|moonshot' "${PROJECT_ROOT}/terraform/manifests/templates/02-download-weights.yaml.template" "Missing HuggingFace target model repo in download template"
  assert_match 'gsutil|gcloud|GCS_BUCKET' "${PROJECT_ROOT}/terraform/manifests/templates/02-hydrate-weights-gcs.yaml.template" "Missing GCS transfer tooling in hydration template"
  assert_match 'fla-core' "${PROJECT_ROOT}/docker/Dockerfile.sglang" "Missing fla-core requirement in docker/Dockerfile.sglang"
  assert_match '--served-model-name moonshotai/Kimi-K3' "${PROJECT_ROOT}/terraform/manifests/templates/09-kimi-k3-sglang-mpi.yaml.template" "Missing --served-model-name in SGLang template"
  assert_no_match '^\s*--dcp-size' "${PROJECT_ROOT}/terraform/manifests/templates/09-kimi-k3-sglang-mpi.yaml.template" "Prohibited active --dcp-size flag found in SGLang exec line"
}

# T5_ADV_03: Gateway, Observability, HPA, and MoE Compilation Specializations
t5_adv_03() {
  assert_match 'preserve_thought_blocks: true' "${PROJECT_ROOT}/terraform/manifests/templates/04-enterprise-gateway-config.yaml.template" "Missing LiteLLM CoT reasoning preservation config"
  assert_match 'drop_params: false' "${PROJECT_ROOT}/terraform/manifests/templates/04-enterprise-gateway-config.yaml.template" "Missing LiteLLM CoT parameter drop prevention"
  assert_match 'gke-nodepool: "np-system"' "${PROJECT_ROOT}/terraform/manifests/templates/05-enterprise-gateway-deployment.yaml.template" "Missing np-system node pool targeting in LiteLLM Gateway deployment"
  assert_no_match 'gke-spot: "true"' "${PROJECT_ROOT}/terraform/manifests/templates/05-enterprise-gateway-deployment.yaml.template" "Unexpected spot node pool targeting in LiteLLM Gateway deployment"
  assert_match 'raid0|mdadm|nvme' "${PROJECT_ROOT}/terraform/manifests/templates/00-local-nvme-raid.yaml.template" "Missing local NVMe RAID-0 DaemonSet configuration"
  assert_match 'PodMonitoring' "${PROJECT_ROOT}/terraform/manifests/templates/06-model-observability-podmonitoring.yaml.template" "Missing Model observability PodMonitoring custom resource"
}

# T5_ADV_04: Remediated Bug Fixes Verification (ADV-T5-07 & ADV-T5-08)
t5_adv_04() {
  # ADV-T5-07: Ensure 05_run_benchmarks.sh uses safe_envsubst and not raw envsubst on pod template
  assert_match 'safe_envsubst' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Missing safe_envsubst helper in 05_run_benchmarks.sh"
  assert_no_match 'envsubst < "\$\{TEMPLATE_DIR\}/08-in-cluster-benchmark-job\.yaml\.template"' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Unsafe raw envsubst used in 05_run_benchmarks.sh; must use safe_envsubst"

  # ADV-T5-08: Ensure zero external internet URL dependencies in deployment scripts and local template exists
  assert_no_match 'https://raw\.githubusercontent\.com' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Unhermetic external GitHub URL dependency found in 03_deploy_workloads.sh"

  # P4-7: Ensure 05_run_benchmarks.sh prevents directory doubling and validates empty output
  assert_no_match '\$\{RESULTS_DIR\}/\$\{INFERENCE_ENGINE\}' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Directory doubling found in 05_run_benchmarks.sh"
  assert_match '! -s "\$\{RESULT_FILE\}"|! -s "\$\{RESULTS_DIR\}/incluster_' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Missing empty output validation in 05_run_benchmarks.sh"
  assert_match 'ERROR: Extracted benchmark JSON from pod logs is empty' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Missing ERROR message on empty extracted benchmark JSON in 05_run_benchmarks.sh"
  assert_match 'python3 -m json\.tool' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Missing json.tool validation in 05_run_benchmarks.sh"

  # P4-5: Ensure 03_deploy_workloads.sh carries error guard and no raw fallback for REDIS_PASSWORD
  assert_match 'ERROR: Failed to obtain REDIS_PASSWORD' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Missing REDIS_PASSWORD error guard in 03_deploy_workloads.sh"
  assert_no_match 'REDIS_PASSWORD:-redis-secret-password-change-me' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Prohibited REDIS_PASSWORD default fallback found outside render-only conditional"

  # F6-6: Ensure cleanup_fabric_pods covers all exit 1 paths in section 3b
  assert_match 'cleanup_fabric_pods' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Missing cleanup_fabric_pods in 03_deploy_workloads.sh"
  local cleanup_cnt
  cleanup_cnt=$(grep -c 'cleanup_fabric_pods' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" || true)
  if [ "${cleanup_cnt}" -lt 9 ]; then
    echo "ERROR: cleanup_fabric_pods count (${cleanup_cnt}) is less than expected 9 (definition + 8 exit/success paths)" >&2
    return 1
  fi

  # F6-13: LIVE_VALIDATION=yes guard assertion
  assert_match 'LIVE_VALIDATION' "${PROJECT_ROOT}/scripts/02_deploy_infra.sh" "Missing LIVE_VALIDATION guard in 02_deploy_infra.sh"
  assert_match 'LIVE_VALIDATION' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Missing LIVE_VALIDATION guard in 03_deploy_workloads.sh"
}

# T5_ADV_05: Remediated Bug Fixes Verification (ADV-T5-09, Script Syntax, and Governance)
t5_adv_05() {
  # ADV-T5-09: Explicit exception handling in Python benchmark scripts
  for py_file in "${PROJECT_ROOT}"/benchmarks/*.py; do
    if [ -f "${py_file}" ]; then
      assert_match 'ZeroDivisionError|KeyError|Exception' "${py_file}" "Missing explicit exception handling in ${py_file}"
    fi
  done

  # Operational script syntax checks (ADV-10 in Challenger 1)
  for s in "${PROJECT_ROOT}"/scripts/*.sh; do
    if [ -f "${s}" ]; then
      assert_cmd_success "bash -n '${s}'" "Syntax error detected in ${s}"
    fi
  done
}

# T5_ADV_06: Weights Backup Consistency Verification (P7-1)
t5_adv_06() {
  assert_match 'PURGE_WEIGHTS_BACKUP' "${PROJECT_ROOT}/README.md" "Missing PURGE_WEIGHTS_BACKUP documentation in README.md"
  assert_no_match 'resource\s+"google_storage_bucket"\s+"[^"]*weights' "${PROJECT_ROOT}/terraform/modules/storage/main.tf" "terraform/modules/storage/main.tf declares a google_storage_bucket for weights"
  assert_match 'PURGE_WEIGHTS_BACKUP.*true.*FORCE_DESTROY.*true|FORCE_DESTROY.*true.*PURGE_WEIGHTS_BACKUP.*true' "${PROJECT_ROOT}/scripts/06_destroy_all.sh" "scripts/06_destroy_all.sh must require both PURGE_WEIGHTS_BACKUP=true and FORCE_DESTROY=true to delete weights backup bucket"
}

run_tier_5_tests() {
  log_info "=== Executing Tier 5: Adversarial White-Box Coverage Hardening Suite ==="
  run_test_case "T5_ADV_01_Domain_Manifest_Assertions" t5_adv_01
  run_test_case "T5_ADV_02_Hermeticity_Empty_String_Injection" t5_adv_02
  run_test_case "T5_ADV_03_Gateway_Observability_Config" t5_adv_03
  run_test_case "T5_ADV_04_Remediated_Script_Security_Network_Isolation" t5_adv_04
  run_test_case "T5_ADV_05_Benchmark_Exception_Resilience_Syntax" t5_adv_05
  run_test_case "T5_ADV_06_Weights_Cache_Consistency" t5_adv_06
}
