#!/bin/bash
# ==============================================================================
# test_cases_t4.sh - Tier 4: Real-World Application Scenarios (Workload Simulation)
# ==============================================================================
set -euo pipefail

t4_s01() {
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="trtllm" CLUSTER_NAME="kimi3-cluster" REGION="europe-north1" PROJECT_ID="test-proj" ZONE="europe-north1-b" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local yaml_file="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-trtllm-mpi.yaml"
  assert_file_exists "${yaml_file}"
  assert_match 'kind: StatefulSet|kind: Deployment' "${yaml_file}" "Not a StatefulSet or Deployment"
  assert_match 'mpirun -n 16' "${yaml_file}" "Missing mpirun -n 16"
  assert_match 'tp_size.*8|TP_SIZE.*8' "${yaml_file}" "Missing TP=8"
  assert_match 'pp_size.*2|PP_SIZE.*2' "${yaml_file}" "Missing PP=2"
  assert_match 'ep_size.*8|EP_SIZE.*8' "${yaml_file}" "Missing EP=8"
  assert_match 'IPC_LOCK' "${yaml_file}" "Missing IPC_LOCK"
  assert_match '/dev/shm' "${yaml_file}" "Missing /dev/shm"
  assert_match 'ReadOnlyMany|ROX|rox' "${yaml_file}" "Missing ReadOnlyMany"
  assert_match 'NCCL_NET_GDR_LEVEL' "${yaml_file}" "Missing NCCL_NET_GDR_LEVEL"
}

t4_s02() {
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="sglang" CLUSTER_NAME="kimi3-cluster" REGION="europe-north1" PROJECT_ID="test-proj" ZONE="europe-north1-b" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local yaml_file="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  assert_file_exists "${yaml_file}"
  assert_match 'sglang\.launch_server' "${yaml_file}" "Missing sglang.launch_server"
  assert_match '--dist-init-addr' "${yaml_file}" "Missing dist-init-addr"
  assert_match '--nnodes 2|--nnodes=2' "${yaml_file}" "Missing nnodes 2"
  assert_match '--node-rank' "${yaml_file}" "Missing node-rank"
  assert_no_match 'ray|6379|8265' "${yaml_file}" "Ray references found"
  assert_match 'IPC_LOCK' "${yaml_file}" "Missing IPC_LOCK"
  assert_match '/dev/shm' "${yaml_file}" "Missing /dev/shm"
  assert_match 'ReadOnlyMany|ROX|rox' "${yaml_file}" "Missing ReadOnlyMany"
  assert_match 'NCCL_NET_GDR_LEVEL' "${yaml_file}" "Missing NCCL_NET_GDR_LEVEL"
}

t4_s03() {
  assert_cmd_success "[ -z \"\$(hg status /google/src/cloud/donina/deploy_kimi_gcp_stack/google3/experimental/users/donina/GLM52NVFP4_GCPA4 2>/dev/null)\" ]" "GLM reference repository modified!"
  local md_files
  md_files=$(python3 -c "import os; print([os.path.join(r, f) for r, d, files in os.walk('${PROJECT_ROOT}') for f in files if f.endswith('.md') and '.agents' not in r and '.gemini' not in r and f not in ('README.md', 'TEST_INFRA.md', 'TEST_READY.md', 'PROJECT.md')])")
  assert_equals "[]" "${md_files}" "Unauthorized markdown files in project: ${md_files}"
  assert_file_exists "${PROJECT_ROOT}/README.md"
  assert_match 'https://minitel\.corp\.google\.com/' "${PROJECT_ROOT}/README.md" "Missing Minitel links in README"
  for dir in terraform benchmarks docker; do
    if [ -d "${PROJECT_ROOT}/${dir}" ]; then
      assert_no_match 'GLM|NVFP4|glm52' "${PROJECT_ROOT}/${dir}" "Prohibited GLM strings in directory ${dir}"
    fi
  done
  for f in "${PROJECT_ROOT}"/scripts/*; do
    if [ -f "${f}" ] && [[ ! "${f}" =~ test_.*\.sh$ ]]; then
      assert_no_match 'GLM|NVFP4|glm52' "${f}" "Prohibited GLM strings in script ${f}"
    fi
  done
}

t4_s04() {
  assert_match 'count.*[48]|range\([48]\)' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing 4 or 8 secondary networks"
  assert_match 'mtu.*8896' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing mtu 8896"
  assert_match 'COLLOCATED' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing COLLOCATED"
  assert_file_exists "${PROJECT_ROOT}/terraform/manifests/templates/00c-nccl-test-job.yaml.template"
  assert_match 'NCCL_NET_GDR_LEVEL' "${PROJECT_ROOT}/terraform/manifests/templates/00c-nccl-test-job.yaml.template" "Missing GDR level in NCCL test template"
  assert_match 'IPC_LOCK' "${PROJECT_ROOT}/terraform/manifests/templates/00c-nccl-test-job.yaml.template" "Missing IPC_LOCK in NCCL test template"
}

t4_s05() {
  local blaze_bin="/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents"
  if [ ! -x "${blaze_bin}" ]; then blaze_bin="blaze"; fi
  assert_cmd_success "SKYBUILD=1 ${blaze_bin} build //experimental/users/donina/KIMI3_GCPA4:all //experimental/users/donina/KIMI3_GCPA4/..." "Hermetic build failed in scenario 5"
  assert_cmd_success "(cd '${PROJECT_ROOT}' && terraform fmt -check -recursive ./terraform/)" "Terraform format check failed in scenario 5"
  for f in "${PROJECT_ROOT}"/scripts/*.sh; do
    assert_cmd_success "bash -n '${f}'" "Bash syntax error in ${f}"
  done
  assert_cmd_success "python3 -m py_compile '${PROJECT_ROOT}'/benchmarks/*.py '${PROJECT_ROOT}'/scripts/*.py" "Python compilation failed in scenario 5"
  assert_match '/ 16\.0|/ 16\b' "${PROJECT_ROOT}/benchmarks/benchmark_kimi_k3.py" "Missing divisor in benchmark"
  assert_match 'europe-north1' "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "Missing europe-north1 in massive benchmark"
  local untracked
  untracked=$(hg status -u --config ui.ignore="${PROJECT_ROOT}/.hgignore" "${PROJECT_ROOT}" 2>/dev/null || true)
  assert_equals "" "${untracked}" "Untracked files after full scenario 5 run: ${untracked}"
}

run_tier_4_tests() {
  log_info "=== Executing Tier 4 Scenarios ==="
  run_test_case "T4_S01_TRTLLM_E2E_MPI" t4_s01
  run_test_case "T4_S02_SGLang_E2E_NoRay" t4_s02
  run_test_case "T4_S03_Governance_Minitel_Audit" t4_s03
  run_test_case "T4_S04_RoCEv2_Network_Fabric" t4_s04
  run_test_case "T4_S05_Hermetic_Build_Syntax_TCO" t4_s05
}
