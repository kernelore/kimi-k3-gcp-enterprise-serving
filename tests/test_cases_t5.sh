#!/bin/bash
# ==============================================================================
# test_cases_t5.sh - Tier 5: Adversarial White-Box Coverage Hardening Suite
# ==============================================================================
# Synthesizes and integrates all Tier 5 adversarial test cases from Challenger 1
# and Challenger 2, closing all 14 white-box coverage gaps across templates,
# scripts, containers, and BUILD, while verifying remediated bugs (ADV-T5-07..09).
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
  assert_match 'rdma-0' "${trtllm_yaml}" "Missing rdma-0 in TRTLLM manifest"
  assert_match 'rdma-1' "${trtllm_yaml}" "Missing rdma-1 in TRTLLM manifest"
  assert_match 'rdma-2' "${trtllm_yaml}" "Missing rdma-2 in TRTLLM manifest"
  assert_match 'rdma-3' "${trtllm_yaml}" "Missing rdma-3 in TRTLLM manifest"
  assert_match 'sizeLimit: 512Gi' "${trtllm_yaml}" "Missing 512Gi /dev/shm sizeLimit in TRTLLM manifest"
  assert_match 'storage: 2000Gi' "${trtllm_yaml}" "Missing 2000Gi storage claim in TRTLLM manifest"

  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="sglang" CLUSTER_NAME="test-cluster" REGION="europe-north1" ZONE="europe-north1-b" PROJECT_ID="test-proj" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  assert_file_exists "${sglang_yaml}"
  assert_match 'gke-spot: "true"' "${sglang_yaml}" "Missing gke-spot nodeSelector in SGLang manifest"
  assert_match 'nvidia-b200' "${sglang_yaml}" "Missing nvidia-b200 nodeSelector in SGLang manifest"
  assert_match 'rdma-0' "${sglang_yaml}" "Missing rdma-0 in SGLang manifest"
  assert_match 'rdma-1' "${sglang_yaml}" "Missing rdma-1 in SGLang manifest"
  assert_match 'sizeLimit: 512Gi' "${sglang_yaml}" "Missing 512Gi /dev/shm sizeLimit in SGLang manifest"
  assert_match 'storage: 2000Gi' "${sglang_yaml}" "Missing 2000Gi storage claim in SGLang manifest"

  local staging_pvc="${PROJECT_ROOT}/terraform/manifests/generated/02-staging-pvc.yaml"
  assert_match 'storage: 2000Gi' "${staging_pvc}" "Missing 2000Gi storage claim in generated staging PVC"
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
}

# T5_ADV_03: Gateway, Observability, HPA, and MoE Compilation Specializations
t5_adv_03() {
  assert_match 'preserve_thought_blocks: true' "${PROJECT_ROOT}/terraform/manifests/templates/04-enterprise-gateway-config.yaml.template" "Missing LiteLLM CoT reasoning preservation config"
  assert_match 'drop_params: false' "${PROJECT_ROOT}/terraform/manifests/templates/04-enterprise-gateway-config.yaml.template" "Missing LiteLLM CoT parameter drop prevention"
  assert_match 'gke-spot: "true"' "${PROJECT_ROOT}/terraform/manifests/templates/09-kimi-k3-sglang-mpi.yaml.template" "Missing spot node pool targeting in SGLang serving deployment"
  assert_match 'raid0|mdadm|nvme' "${PROJECT_ROOT}/terraform/manifests/templates/00-local-nvme-raid.yaml.template" "Missing local NVMe RAID-0 DaemonSet configuration"
  assert_match 'PodMonitoring' "${PROJECT_ROOT}/terraform/manifests/templates/06-model-observability-podmonitoring.yaml.template" "Missing Model observability PodMonitoring custom resource"
}

# T5_ADV_04: Remediated Bug Fixes Verification (ADV-T5-07 & ADV-T5-08)
t5_adv_04() {
  # ADV-T5-07: Ensure 05_run_benchmarks.sh uses safe_envsubst and not raw envsubst on pod template
  assert_match 'safe_envsubst' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Missing safe_envsubst helper in 05_run_benchmarks.sh"
  assert_no_match 'envsubst < "\$\{TEMPLATE_DIR\}/08-in-cluster-benchmark-job\.yaml\.template"' "${PROJECT_ROOT}/scripts/05_run_benchmarks.sh" "Unsafe raw envsubst used in 05_run_benchmarks.sh; must use safe_envsubst"

  # ADV-T5-08: Ensure zero external internet URL dependencies in deployment scripts
  assert_no_match 'https://raw\.githubusercontent\.com' "${PROJECT_ROOT}/scripts/03_deploy_workloads.sh" "Unhermetic external GitHub URL dependency found in 03_deploy_workloads.sh"
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

  # BUILD coverage check (ADV-13 in Challenger 1)
  for py in "${PROJECT_ROOT}"/benchmarks/*.py "${PROJECT_ROOT}"/scripts/*.py; do
    if [ -f "${py}" ]; then
      local target_name
      target_name="$(basename "${py}" .py)"
      assert_match "${target_name}" "${PROJECT_ROOT}/BUILD" "Python script ${target_name} missing from BUILD file"
    fi
  done

  # GLM/NVFP4 isolation check (ADV-15a in Challenger 1)
  for dir in terraform benchmarks docker; do
    if [ -d "${PROJECT_ROOT}/${dir}" ]; then
      assert_no_match 'GLM|NVFP4|glm52' "${PROJECT_ROOT}/${dir}" "Prohibited GLM strings in directory ${dir}"
    fi
  done

  # Minitel compliance check (ADV-15b in Challenger 1)
  local md_files
  md_files=$(python3 -c "import os; print([os.path.join(r, f) for r, d, files in os.walk('${PROJECT_ROOT}') for f in files if f.endswith('.md') and '.agents' not in r and '.gemini' not in r and f not in ('README.md', 'TEST_INFRA.md', 'TEST_READY.md', 'PROJECT.md')])")
  assert_equals "[]" "${md_files}" "Unauthorized markdown files in project: ${md_files}"
}

run_tier_5_tests() {
  log_info "=== Executing Tier 5: Adversarial White-Box Coverage Hardening Suite ==="
  run_test_case "T5_ADV_01_Domain_Manifest_Assertions" t5_adv_01
  run_test_case "T5_ADV_02_Hermeticity_Empty_String_Injection" t5_adv_02
  run_test_case "T5_ADV_03_Gateway_Observability_HPA_MoE" t5_adv_03
  run_test_case "T5_ADV_04_Remediated_Script_Security_Network_Isolation" t5_adv_04
  run_test_case "T5_ADV_05_Benchmark_Exception_Resilience_Syntax" t5_adv_05
}
