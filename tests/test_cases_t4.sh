#!/bin/bash
# ==============================================================================
# test_cases_t4.sh - Tier 4: Real-World Application Scenarios (Workload Simulation)
# Note: This file defines test-case functions only. Running it directly is a
# no-op that exits 0. The test suite executes via tests/test_e2e_kimi_k3.sh,
# which sources this file and calls run_tier_4_tests.
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
  local gw_secret="${PROJECT_ROOT}/terraform/manifests/generated/04-enterprise-gateway-config.yaml"
  assert_file_exists "${gw_secret}"
  assert_match 'sk-kimi-k3-master-secret-key-change-me' "${gw_secret}" "Missing GATEWAY_MASTER_KEY placeholder in gateway secret"
  assert_match 'kimi-k3-gateway-admin-secret' "${gw_secret}" "Missing DB_PASSWORD placeholder in gateway secret"
}

t4_s02() {
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="sglang" CLUSTER_NAME="kimi3-cluster" REGION="europe-north1" PROJECT_ID="test-proj" ZONE="europe-north1-b" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  local yaml_file="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  assert_file_exists "${yaml_file}"
  assert_match 'sglang\.launch_server' "${yaml_file}" "Missing sglang.launch_server"
  assert_match '--dist-init-addr' "${yaml_file}" "Missing dist-init-addr"
  assert_match '--nnodes 2|--nnodes=2' "${yaml_file}" "Missing nnodes 2"
  assert_match '--node-rank' "${yaml_file}" "Missing node-rank"
  # Word-anchored: a bare 'ray' substring also matches "array", which the NVMe
  # RAID-0 comments in the manifests use freely.
  assert_no_match '(^|[^[:alnum:]_])[Rr]ay([^[:alnum:]_]|$)|RAY_[A-Z_]+|6379|8265' "${yaml_file}" "Ray references found"
  assert_match 'IPC_LOCK' "${yaml_file}" "Missing IPC_LOCK"
  assert_match '/dev/shm' "${yaml_file}" "Missing /dev/shm"
  assert_match 'ReadOnlyMany|ROX|rox' "${yaml_file}" "Missing ReadOnlyMany"
}

t4_s03() {
  assert_file_exists "${PROJECT_ROOT}/README.md"
}

t4_s04() {
  assert_match 'count.*8|range\(8\)' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing 8 secondary networks"
  assert_match 'mtu.*8896' "${PROJECT_ROOT}/terraform/modules/network/main.tf" "Missing mtu 8896"
  assert_match 'COLLOCATED' "${PROJECT_ROOT}/terraform/modules/node_pool_spot/main.tf" "Missing COLLOCATED"
}

t4_s05() {
  assert_cmd_success "(cd '${PROJECT_ROOT}' && terraform fmt -check -recursive ./terraform/)" "Terraform format check failed in scenario 5"
  for f in "${PROJECT_ROOT}"/scripts/*.sh; do
    assert_cmd_success "bash -n '${f}'" "Bash syntax error in ${f}"
  done
  assert_cmd_success "python3 -m py_compile '${PROJECT_ROOT}'/benchmarks/*.py '${PROJECT_ROOT}'/scripts/*.py" "Python compilation failed in scenario 5"
  assert_match '/ 16\.0|/ 16\b' "${PROJECT_ROOT}/benchmarks/benchmark_kimi_k3.py" "Missing divisor in benchmark"
  assert_match 'europe-north1' "${PROJECT_ROOT}/benchmarks/massive_benchmark_kimi_k3.py" "Missing europe-north1 in massive benchmark"
  rm -rf "${PROJECT_ROOT}/terraform/manifests/generated"/*.yaml 2>/dev/null || true
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="trtllm" CLUSTER_NAME="kimi3-cluster" REGION="europe-north1" PROJECT_ID="test-proj" ZONE="europe-north1-b" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  (cd "${PROJECT_ROOT}" && env -i PATH="${PATH}" INFERENCE_ENGINE="sglang" CLUSTER_NAME="kimi3-cluster" REGION="europe-north1" PROJECT_ID="test-proj" ZONE="europe-north1-b" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
}

run_tier_4_tests() {
  log_info "=== Executing Tier 4 Scenarios ==="
  run_test_case "T4_S01_TRTLLM_E2E_MPI" t4_s01
  run_test_case "T4_S02_SGLang_E2E_NoRay" t4_s02
  run_test_case "T4_S03_Governance_Audit" t4_s03
  run_test_case "T4_S04_RoCEv2_Network_Fabric" t4_s04
  run_test_case "T4_S05_Hermetic_Build_Syntax_TCO" t4_s05
}
