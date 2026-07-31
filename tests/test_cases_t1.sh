#!/bin/bash
# ==============================================================================
# test_cases_t1.sh - Tier 1: Feature Coverage (Happy Path / Conformance)
# Note: This file defines test-case functions only. Running it directly is a
# no-op that exits 0. The test suite executes via tests/test_e2e_kimi_k3.sh,
# which sources this file and calls run_tier_1_tests.
# ==============================================================================
set -euo pipefail

# F1: Sovereign Architectural Audit & Parity
t1_f1_01() {
  return 77 # Purged internal build target assertions for open-source compliance
}

t1_f1_02() {
  return 77 # Purged internal tool assertions for open-source compliance
}

t1_f1_03() {
  return 77 # Purged internal Internal link assertions for open-source compliance
}

t1_f1_04() {
  for dir in benchmarks docker scripts terraform terraform/manifests/templates; do
    assert_true "[ -d '${PROJECT_ROOT}/${dir}' ]" "Directory ${dir} missing"
  done
  for submod in audit cache cluster database gateway_iam network node_pool_spot observability storage; do
    assert_true "[ -d '${PROJECT_ROOT}/terraform/modules/${submod}' ]" "Terraform submodule ${submod} missing"
  done
}

t1_f1_05() {
  # Zero GLM leakage in root config files
  for f in README.md .gitignore; do
    if [ -f "${PROJECT_ROOT}/${f}" ]; then
      assert_no_match 'glm52|glm-5|NVFP4|vllm' "${PROJECT_ROOT}/${f}" "Prohibited legacy model string in root file ${f}"
    fi
  done
  assert_no_match '1,130|42 concurrent' "${PROJECT_ROOT}/README.md" "Stale KV cache arithmetic 1,130 GB or 42 concurrent found in README.md"
}

# F2: Docker Container Runtimes & Python Benchmark Harnesses
t1_f2_01() {
  assert_cmd_success "python3 -m py_compile '${PROJECT_ROOT}'/benchmarks/*.py '${PROJECT_ROOT}'/scripts/*.py" "Python compilation failed"
}

t1_f2_02() {
  for f in benchmark_kimi_k3.py massive_benchmark_kimi_k3.py soak_benchmark_kimi_k3.py; do
    assert_match '/ 16\.0|/ 16\b' "${PROJECT_ROOT}/benchmarks/${f}" "Missing per-GPU normalization in ${f}"
  done
}

t1_f2_03() {
  assert_match 'europe-north1' "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "Missing europe-north1 region in massive_benchmark_kimi_k3.py"
  assert_match 'Hamina|Finland|TCO|CUD|pricing|cost' "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "Missing TCO pricing in massive_benchmark_kimi_k3.py"
  assert_match 'europe-north1' "${PROJECT_ROOT}/terraform/variables.tf" "Missing default europe-north1 in variables.tf"
}

t1_f2_04() {
  assert_file_exists "${PROJECT_ROOT}/docker/Dockerfile"
  assert_match 'tensorrt|TRTLLM' "${PROJECT_ROOT}/docker/Dockerfile" "Missing TensorRT-LLM references in Dockerfile"
}

t1_f2_05() {
  assert_file_exists "${PROJECT_ROOT}/docker/Dockerfile.sglang"
  assert_no_match 'pip install ray|import ray' "${PROJECT_ROOT}/docker/Dockerfile.sglang" "Ray leakage in Dockerfile.sglang"
  assert_match '--dist-init-addr' "${PROJECT_ROOT}/docker/Dockerfile.sglang" "Missing --dist-init-addr in Dockerfile.sglang"
}

# F3: Automation Scripts & Root Terraform Configuration / Submodules
t1_f3_01() {
  assert_cmd_success "(cd '${PROJECT_ROOT}' && terraform fmt -check -recursive ./terraform/)" "Terraform format check failed"
}

t1_f3_02() {
  for f in "${PROJECT_ROOT}"/scripts/*.sh; do
    assert_cmd_success "bash -n '${f}'" "Bash syntax error in ${f}"
  done
}

t1_f3_03() {
  assert_match '2\.8T|MXFP4|kimi-k3' "${PROJECT_ROOT}/terraform/variables.tf" "Missing domain parameters in variables.tf"
}

t1_f3_04() {
  assert_match 'hyperdisk-ml' "${PROJECT_ROOT}/terraform/modules/storage/main.tf" "Missing hyperdisk-ml in storage module"
  assert_match '2000' "${PROJECT_ROOT}/terraform/variables.tf" "Missing 2000 GB size in variables.tf"
}

t1_f3_05() {
  assert_match 'a4-highgpu-8g' "${PROJECT_ROOT}/terraform/variables.tf" "Missing default machine type a4-highgpu-8g in variables.tf"
  assert_match 'gpu_machine_type' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing gpu_machine_type reference in spot pool"
  assert_match 'COLLOCATED' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing COLLOCATED placement in spot pool"
}

# F4: Kubernetes Manifest Templates & Selectable Dual-Engine Serving
t1_f4_01() {
  assert_cmd_success "(cd '${PROJECT_ROOT}' && INFERENCE_ENGINE='trtllm' scripts/03_deploy_workloads.sh --render-only)" "Render TRTLLM manifests failed"
  assert_file_exists "${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
}

t1_f4_02() {
  assert_cmd_success "(cd '${PROJECT_ROOT}' && INFERENCE_ENGINE='sglang' scripts/03_deploy_workloads.sh --render-only)" "Render SGLang manifests failed"
  assert_file_exists "${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
}

t1_f4_03() {
  local trtllm_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  if [ ! -f "${trtllm_yaml}" ]; then
    (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="trtllm" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  fi
  assert_match 'mpirun -n 16' "${trtllm_yaml}" "Missing mpirun -n 16 in TRTLLM manifest"
  assert_match 'tp_size.*8|TP_SIZE.*8' "${trtllm_yaml}" "Missing TP=8 in TRTLLM manifest"
  assert_match 'pp_size.*2|PP_SIZE.*2' "${trtllm_yaml}" "Missing PP=2 in TRTLLM manifest"
  assert_match 'ep_size.*8|EP_SIZE.*8' "${trtllm_yaml}" "Missing EP=8 in TRTLLM manifest"
}

t1_f4_04() {
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  if [ ! -f "${sglang_yaml}" ]; then
    (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  fi
  assert_no_match 'ray' "${sglang_yaml}" "Ray references in SGLang manifest"
  assert_match '--dist-init-addr' "${sglang_yaml}" "Missing --dist-init-addr in SGLang manifest"
  assert_match '--nnodes 2|--nnodes=2' "${sglang_yaml}" "Missing nnodes 2 in SGLang manifest"
}

t1_f4_05() {
  for f in "${PROJECT_ROOT}"/terraform/manifests/generated/*.yaml; do
    if [ -f "${f}" ]; then
      assert_cmd_success "python3 -c \"import yaml; list(yaml.safe_load_all(open('${f}')))\"" "Malformed YAML in ${f}"
    fi
  done
}

# F5: Mandatory High-Performance Networking & Shared Memory
t1_f5_01() {
  assert_match 'count.*8|range\(8\)' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing 8 secondary networks in network module"
  assert_match 'range\(8\)' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing 8 NIC attachments in spot pool module"
  assert_match 'allow_internal_primary_vpc' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing allow_internal_primary_vpc in network module"
}

t1_f5_02() {
  assert_match 'mtu.*8896' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing mtu 8896 in network module"
}

t1_f5_03() {
  for f in "${PROJECT_ROOT}"/terraform/manifests/generated/09-*.yaml; do
    if [ -f "${f}" ]; then
      assert_match "set_nccl_env.sh" "${f}" "Missing set_nccl_env.sh in ${f}"
      assert_match "GLOO_SOCKET_IFNAME" "${f}" "Missing GLOO_SOCKET_IFNAME in ${f}"
      assert_match "expandable_segments" "${f}" "Missing expandable_segments in ${f}"
    fi
  done
  assert_match "GLOO_SOCKET_IFNAME" "${PROJECT_ROOT}/terraform/manifests/templates/00c-nccl-test-job.yaml.template" "Missing GLOO_SOCKET_IFNAME in 00c template"
  assert_match "GLOO_SOCKET_IFNAME" "${PROJECT_ROOT}/terraform/manifests/templates/00d-serving-nccl-parity-job.yaml.template" "Missing GLOO_SOCKET_IFNAME in 00d template"
}

t1_f5_04() {
  for f in "${PROJECT_ROOT}"/terraform/manifests/generated/09-*.yaml; do
    if [ -f "${f}" ]; then
      assert_match 'IPC_LOCK' "${f}" "Missing IPC_LOCK in ${f}"
    fi
  done
}

t1_f5_05() {
  for f in "${PROJECT_ROOT}"/terraform/manifests/generated/09-*.yaml; do
    if [ -f "${f}" ]; then
      assert_match '/dev/shm' "${f}" "Missing /dev/shm mount in ${f}"
      assert_match 'medium: Memory|medium: \"Memory\"' "${f}" "Missing emptyDir Memory in ${f}"
    fi
  done
}

# F6: Ironclad Sovereign Governance & VCS Discipline
t1_f6_01() {
  return 77 # Purged GLM reference repository isolation assertions for open-source compliance
}

t1_f6_02() {
  # Scans all directories under KIMI3_GCPA4/ to assert zero .md files exist except authorized documentation (ignoring .agents, .gemini, .venv, .git, .terraform)
  local md_files
  md_files=$(python3 -c "import os; print([os.path.join(r, f) for r, d, files in os.walk('${PROJECT_ROOT}') for f in files if f.endswith('.md') and not any(x in r for x in ('.agents', '.gemini', '.venv', '.git', '.terraform')) and f not in ('README.md', 'PHASE6_RUNBOOK.md')])")
  assert_equals "[]" "${md_files}" "Unauthorized markdown files found in project: ${md_files}"
}

t1_f6_03() {
  assert_file_exists "${PROJECT_ROOT}/.gitignore"
  assert_match '\*\.tfstate|\.terraform' "${PROJECT_ROOT}/.gitignore" "Missing terraform state in .gitignore"
}

t1_f6_04() {
  return 77 # Purged internal VCS clean status check for open-source compliance
}

t1_f6_05() {
  # Check across scripts/, terraform/, benchmarks/, docker/
  for dir in terraform benchmarks docker; do
    if [ -d "${PROJECT_ROOT}/${dir}" ]; then
      assert_no_match 'glm52|glm-5|NVFP4|vllm' "${PROJECT_ROOT}/${dir}" "Prohibited legacy model strings in directory ${dir}"
    fi
  done
  for f in "${PROJECT_ROOT}"/scripts/*; do
    if [ -f "${f}" ] && [[ ! "${f}" =~ test_.*\.sh$ ]]; then
      assert_no_match 'glm52|glm-5|NVFP4|vllm' "${f}" "Prohibited legacy model strings in script ${f}"
    fi
  done
}

run_tier_1_tests() {
  log_info "=== Executing Tier 1 Tests ==="
  run_test_case "T1_F1_01_Build_Targets_Exist" t1_f1_01
  run_test_case "T1_F1_02_Blaze_Build_Hermetic" t1_f1_02
  run_test_case "T1_F1_03_README_Open_Source_Compliance" t1_f1_03
  run_test_case "T1_F1_04_Structural_Parity_Dirs" t1_f1_04
  run_test_case "T1_F1_05_No_GLM_Leakage_Root" t1_f1_05

  run_test_case "T1_F2_01_Python_Compile_Clean" t1_f2_01
  run_test_case "T1_F2_02_Per_GPU_Normalization" t1_f2_02
  run_test_case "T1_F2_03_Europe_North1_Pricing" t1_f2_03
  run_test_case "T1_F2_04_Dockerfile_TRTLLM_Base" t1_f2_04
  run_test_case "T1_F2_05_Dockerfile_SGLang_Base" t1_f2_05

  run_test_case "T1_F3_01_Terraform_Fmt_Check" t1_f3_01
  run_test_case "T1_F3_02_Bash_Syntax_Validation" t1_f3_02
  run_test_case "T1_F3_03_TF_Domain_Params_Model" t1_f3_03
  run_test_case "T1_F3_04_TF_Hyperdisk_ML_ROX" t1_f3_04
  run_test_case "T1_F3_05_TF_Blackwell_Spot_Pool" t1_f3_05

  run_test_case "T1_F4_01_Render_TRTLLM_Manifests" t1_f4_01
  run_test_case "T1_F4_02_Render_SGLang_Manifests" t1_f4_02
  run_test_case "T1_F4_03_TRTLLM_MPI_Coordination" t1_f4_03
  run_test_case "T1_F4_04_SGLang_Native_Coordination" t1_f4_04
  run_test_case "T1_F4_05_Manifest_YAML_Syntax" t1_f4_05

  run_test_case "T1_F5_01_TF_Eight_Secondary_Subnets" t1_f5_01
  run_test_case "T1_F5_02_TF_RoCEv2_MTU_8896" t1_f5_02
  run_test_case "T1_F5_03_Manifest_NCCL_GDR_Level_5" t1_f5_03
  run_test_case "T1_F5_04_Manifest_IPC_LOCK_Cap" t1_f5_04
  run_test_case "T1_F5_05_Manifest_Dev_Shm_Mount" t1_f5_05

  run_test_case "T1_F6_01_GLM_Repo_Read_Only_Isolation" t1_f6_01
  run_test_case "T1_F6_02_MD_Exclusion_Audit" t1_f6_02
  run_test_case "T1_F6_03_Gitignore_Hgignore_Presence" t1_f6_03
  run_test_case "T1_F6_04_VCS_Clean_Status_Check" t1_f6_04
  run_test_case "T1_F6_05_No_GLM_Parameters_In_Code" t1_f6_05
}
