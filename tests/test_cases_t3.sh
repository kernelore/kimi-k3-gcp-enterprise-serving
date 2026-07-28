#!/bin/bash
# ==============================================================================
# test_cases_t3.sh - Tier 3: Cross-Feature Combinations (Pairwise Combinations)
# ==============================================================================
set -euo pipefail

t3_c01() {
  local trtllm_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  if [ ! -f "${trtllm_yaml}" ]; then
    (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="trtllm" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  fi
  assert_match 'mpirun -n 16' "${trtllm_yaml}" "Missing mpirun -n 16 in TRTLLM manifest"
  assert_match 'NCCL_NET_GDR_LEVEL' "${trtllm_yaml}" "Missing GDR level in TRTLLM manifest"
  assert_match 'IPC_LOCK' "${trtllm_yaml}" "Missing IPC_LOCK in TRTLLM manifest"
  assert_match 'mtu.*8896' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing mtu 8896 in network module"
  assert_match 'COLLOCATED' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing COLLOCATED in spot pool"
}

t3_c02() {
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  if [ ! -f "${sglang_yaml}" ]; then
    (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  fi
  assert_no_match 'ray' "${sglang_yaml}" "Ray reference in SGLang manifest"
  assert_match '--dist-init-addr' "${sglang_yaml}" "Missing dist-init-addr in SGLang manifest"
  assert_match 'ReadOnlyMany|ROX|rox' "${PROJECT_ROOT}/terraform/modules/storage/main.tf" "Missing ROX in storage module"
  assert_match '/dev/shm' "${sglang_yaml}" "Missing /dev/shm in SGLang manifest"
}

t3_c03() {
  assert_match 'a4-highgpu-8g' "${PROJECT_ROOT}/terraform/variables.tf" "Missing machine type in variables.tf"
  assert_match 'mtu.*8896' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing mtu 8896 in network module"
  assert_file_exists "${PROJECT_ROOT}/README.md"
  assert_match 'https://minitel\.corp\.google\.com/' "${PROJECT_ROOT}/README.md" "Missing Minitel links in README.md"
}

t3_c04() {
  assert_match '/ 16\.0|/ 16\b' "${PROJECT_ROOT}/benchmarks/benchmark_kimi_k3.py" "Missing divisor in benchmark"
  assert_match 'europe-north1' "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "Missing europe-north1 in massive benchmark"
  assert_file_exists "${PROJECT_ROOT}/docker/Dockerfile"
  assert_no_match 'GLM|NVFP4|glm52' "${PROJECT_ROOT}/benchmarks" "GLM string in benchmarks"
  assert_no_match 'GLM|NVFP4|glm52' "${PROJECT_ROOT}/docker" "GLM string in docker"
}

t3_c05() {
  assert_match '/ 16\.0|/ 16\b' "${PROJECT_ROOT}/benchmarks/soak_benchmark_kimi_k3.py" "Missing divisor in soak benchmark"
  assert_match 'europe-north1' "${PROJECT_ROOT}/terraform/variables.tf" "Missing default europe-north1 in variables.tf"
  assert_file_exists "${PROJECT_ROOT}/docker/Dockerfile.sglang"
  assert_no_match 'pip install ray|import ray' "${PROJECT_ROOT}/docker/Dockerfile.sglang" "Ray string in Dockerfile.sglang"
  for f in "${PROJECT_ROOT}"/scripts/*; do
    if [ -f "${f}" ] && [[ ! "${f}" =~ test_.*\.sh$ ]]; then
      assert_no_match 'GLM|NVFP4|glm52' "${f}" "Prohibited GLM strings in script ${f}"
    fi
  done
}

t3_c06() {
  assert_cmd_success "python3 -m py_compile '${PROJECT_ROOT}'/benchmarks/*.py '${PROJECT_ROOT}'/scripts/*.py" "Python compile failed"
  local blaze_bin="/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents"
  if [ ! -x "${blaze_bin}" ]; then blaze_bin="blaze"; fi
  assert_cmd_success "SKYBUILD=1 ${blaze_bin} build //experimental/users/donina/KIMI3_GCPA4:all //experimental/users/donina/KIMI3_GCPA4/..." "Blaze build failed"
  local untracked
  untracked=$(hg status -u --config ui.ignore="${PROJECT_ROOT}/.hgignore" "${PROJECT_ROOT}" 2>/dev/null || true)
  assert_equals "" "${untracked}" "Untracked files after build and compile: ${untracked}"
}

t3_c07() {
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="trtllm" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_file_exists "${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_file_exists "${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
}

t3_c08() {
  for submod in audit cache cluster database gateway_iam network node_pool_spot observability storage; do
    assert_true "[ -d '${PROJECT_ROOT}/terraform/modules/${submod}' ]" "Submodule ${submod} missing"
    assert_no_match 'GLM|NVFP4|glm52' "${PROJECT_ROOT}/terraform/modules/${submod}" "GLM leakage in submodule ${submod}"
  done
  local md_files
  md_files=$(python3 -c "import os; print([os.path.join(r, f) for r, d, files in os.walk('${PROJECT_ROOT}') for f in files if f.endswith('.md') and '.agents' not in r and '.gemini' not in r and f not in ('README.md', 'TEST_INFRA.md', 'TEST_READY.md', 'PROJECT.md')])")
  assert_equals "[]" "${md_files}" "Unauthorized markdown files in project: ${md_files}"
}

run_tier_3_tests() {
  log_info "=== Executing Tier 3 Tests ==="
  run_test_case "T3_C01_TRTLLM_RoCEv2_GDR" t3_c01
  run_test_case "T3_C02_SGLang_Native_ROX_Storage" t3_c02
  run_test_case "T3_C03_TF_Blackwell_Spot_RoCEv2" t3_c03
  run_test_case "T3_C04_Benchmark_TRTLLM_Europe_North1" t3_c04
  run_test_case "T3_C05_Benchmark_SGLang_Europe_North1" t3_c05
  run_test_case "T3_C06_Hermetic_Build_With_VCS_Governance" t3_c06
  run_test_case "T3_C07_Dual_Engine_Switching_Idempotency" t3_c07
  run_test_case "T3_C08_Sovereign_Audit_Across_All_Modules" t3_c08
}
