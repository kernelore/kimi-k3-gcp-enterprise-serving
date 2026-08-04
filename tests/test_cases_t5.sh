#!/bin/bash
# ==============================================================================
# test_cases_t5.sh - Tier 5: Adversarial White-Box Coverage Hardening Suite
# Note: This file defines test-case functions only. Running it directly is a
# no-op that exits 0. The test suite executes via tests/test_e2e_kimi_k3.sh,
# which sources this file and calls run_tier_5_tests.
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

# T5_ADV_07: SGLang Parallel Profile Verification (P7-2)
t5_adv_07() {
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" SGLANG_PARALLEL_PROFILE="tp16" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_match '--tp-size "16"' "${sglang_yaml}" "Default SGLang render missing --tp-size 16"
  assert_match '--ep-size "16"' "${sglang_yaml}" "Default SGLang render missing --ep-size 16"
  assert_no_match '--pp-size "2"' "${sglang_yaml}" "Default SGLang render unexpectedly contains --pp-size 2"
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" SGLANG_PARALLEL_PROFILE="tp8pp2" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_match '--tp-size "8"' "${sglang_yaml}" "tp8pp2 SGLang render missing --tp-size 8"
  assert_match '--pp-size "2"' "${sglang_yaml}" "tp8pp2 SGLang render missing --pp-size 2"
  assert_match '--ep-size "8"' "${sglang_yaml}" "tp8pp2 SGLang render missing --ep-size 8"

  local tmp_dir
  tmp_dir=$(mktemp -d)
  local tmp_cfg_ep="${tmp_dir}/config_ep.env"
  local tmp_cfg_geom="${tmp_dir}/config_geom.env"

  cp "${PROJECT_ROOT}/scripts/config.env.example" "${tmp_cfg_ep}"
  sed -i 's/export SGLANG_EP_SIZE="[0-9]*"/export SGLANG_EP_SIZE="4"/' "${tmp_cfg_ep}"
  assert_cmd_fails "(cd '${PROJECT_ROOT}' && CONFIG_FILE='${tmp_cfg_ep}' INFERENCE_ENGINE=sglang scripts/03_deploy_workloads.sh --render-only)" "EP guard failed to reject invalid SGLANG_EP_SIZE=4 for SGLANG_TP_SIZE=16"

  cp "${PROJECT_ROOT}/scripts/config.env.example" "${tmp_cfg_geom}"
  sed -i 's/export SGLANG_TP_SIZE="[0-9]*"/export SGLANG_TP_SIZE="4"/' "${tmp_cfg_geom}"
  sed -i 's/export SGLANG_EP_SIZE="[0-9]*"/export SGLANG_EP_SIZE="4"/' "${tmp_cfg_geom}"
  sed -i 's/export SGLANG_PP_SIZE="[0-9]*"/export SGLANG_PP_SIZE="1"/' "${tmp_cfg_geom}"
  assert_cmd_fails "(cd '${PROJECT_ROOT}' && CONFIG_FILE='${tmp_cfg_geom}' INFERENCE_ENGINE=sglang scripts/03_deploy_workloads.sh --render-only)" "Geometry guard failed to reject invalid parallel geometry (SGLANG_TP_SIZE=4 * SGLANG_PP_SIZE=1)"

  rm -rf "${tmp_dir}"

  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
}

# T5_ADV_08: Data-parallel serving topology (SERVING_REPLICAS=1 vs 2)
#
# Every mistake this covers is one that renders cleanly and then costs an hour of
# four-node time to observe: a second replica whose pods join replica A's NCCL world,
# a leader marked Ready before it can serve, a gateway with one upstream, or -- the
# other direction -- a one-replica deploy that quietly grew a topology it did not ask
# for. The manifests are the only place any of it is visible before the meter starts.
t5_adv_08() {
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  local gateway_yaml="${PROJECT_ROOT}/terraform/manifests/generated/04-enterprise-gateway-config.yaml"
  local parity_yaml="${PROJECT_ROOT}/terraform/manifests/generated/00d-serving-nccl-parity-job.yaml"
  local turndown_yaml="${PROJECT_ROOT}/terraform/manifests/generated/10-scheduled-turndown-cronjob.yaml"

  # --- Default: one replica, two nodes, and no trace of the second one ------------
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_match '^  replicas: 2$' "${sglang_yaml}" "Default render is not a single 2-node replica"
  assert_match '\-\-nnodes "2"' "${sglang_yaml}" "Default render lost --nnodes 2"
  assert_match 'replicas: 2' "${parity_yaml}" "Default parity gate is not 2 nodes"
  assert_match 'kubectl scale statefulset kimi-k3-serving --replicas=2' "${turndown_yaml}" \
    "Turndown turnup does not restore the deployed pod count"
  # The B Service is rendered unconditionally so that enabling DP is a config change,
  # but at one replica it must select an ordinal that does not exist -- pointing it at
  # ordinal 0 would silently make it a second name for replica A.
  assert_match 'name: kimi-k3-serving-b-svc' "${sglang_yaml}" "Replica B Service is not rendered at all"
  assert_match 'apps.kubernetes.io/pod-index: "2"' "${sglang_yaml}" \
    "Replica B Service does not select the ordinal after replica A"
  # The gateway must not know about a replica that is not deployed: a second upstream
  # with no endpoints turns half the router's choices into connection errors.
  assert_no_match 'kimi-k3-serving-b-svc' "${gateway_yaml}" \
    "Single-replica gateway config references replica B"
  assert_no_match 'cooldown_time' "${gateway_yaml}" \
    "Single-replica gateway config carries failover settings with nowhere to fail over to"

  # --- SERVING_REPLICAS=2: four pods, two worlds, two upstreams -------------------
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" SERVING_REPLICAS="2" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
  assert_match '^  replicas: 4$' "${sglang_yaml}" "SERVING_REPLICAS=2 did not scale the StatefulSet to 4 pods"
  # --nnodes stays at one replica's width. If it followed the pod count, all four pods
  # would form a single 32-GPU world against a TP16 geometry and hang in dist-init.
  assert_match '\-\-nnodes "2"' "${sglang_yaml}" \
    "--nnodes followed the pod count instead of the replica width"
  assert_match 'NODE_RANK=\$\(\( POD_ORDINAL % 2 \)\)' "${sglang_yaml}" \
    "Rank is not computed per replica; ordinals 2 and 3 would claim ranks 2 and 3"
  assert_match 'LEADER_ORDINAL=\$\(\( REPLICA_INDEX \* 2 \)\)' "${sglang_yaml}" \
    "Leader is not computed per replica; replica B would dist-init against replica A"
  # Readiness must test "first ordinal of my replica", not "ordinal zero". Ordinal 2 is
  # a leader and answers HTTP; an ordinal-zero test would pass it on pgrep alone and
  # mark it Ready roughly seventeen minutes before it could serve a request.
  assert_match 'HOSTNAME##\*-\} % 2 \)\) -eq 0' "${sglang_yaml}" \
    "Leader probe does not use the per-replica ordinal test"
  assert_no_match '\\" = \\"0\\"' "${sglang_yaml}" \
    "Leader probe still tests for ordinal zero"
  # All four nodes get fabric-gated, not just replica A's two.
  assert_match 'replicas: 4' "${parity_yaml}" "Parity gate still covers only one replica's nodes"
  assert_match 'PARITY_WORLD_SIZE' "${parity_yaml}" "Parity world size is not parameterised"
  assert_no_match 'world_size=2,' "${parity_yaml}" "Parity test still hardcodes a 2-rank world"
  assert_match 'kubectl scale statefulset kimi-k3-serving --replicas=4' "${turndown_yaml}" \
    "Turndown turnup would bring back only half the deployment"
  assert_match 'kimi-k3-serving-b-svc' "${gateway_yaml}" "Gateway has no second upstream at SERVING_REPLICAS=2"
  assert_match 'cooldown_time: 60' "${gateway_yaml}" "Gateway has no cooldown, so a dead replica stays in rotation"

  # --- Guard: an unsupported replica count is refused at render time --------------
  # Replica C onwards has no leader Service and no gateway entry, so it would load 2.8T
  # of weights onto two more nodes and never receive a request.
  assert_cmd_fails "(cd '${PROJECT_ROOT}' && INFERENCE_ENGINE=sglang SERVING_REPLICAS=3 scripts/03_deploy_workloads.sh --render-only)" \
    "SERVING_REPLICAS=3 was accepted despite having no Service or gateway entry for replica C"

  # Leave the generated tree at the shipped default, as the other render cases do.
  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
}

# Cross-file agreement between the gateway's upstreams and the serving manifest.
#
# The two LiteLLM deployments do not come from one place: entry A is a literal in
# 04-enterprise-gateway-config.yaml.template and entry B is a heredoc in
# 03_deploy_workloads.sh. Nothing in the render couples them, so they can drift
# silently, and every way they drift is quiet rather than loud:
#
#   - unequal model_info => the same tokens are billed at two prices depending on
#     which replica the router happened to pick, and the BigQuery audit table
#     reconciles against neither;
#   - unequal rpm/tpm/drop_params => the replicas throttle differently, so "failover
#     is slower" turns out to mean "half the fleet was rate-limited all along";
#   - an api_base naming a Service that does not exist => LiteLLM resolves nothing,
#     cools that deployment down after allowed_fails, and quietly serves everything
#     from one replica while the dashboard still says two.
#
# Regex cannot see any of that, so this parses both files and compares them.
t5_adv_09() {
  local sglang_yaml="${PROJECT_ROOT}/terraform/manifests/generated/09-kimi-k3-sglang-mpi.yaml"
  local gateway_yaml="${PROJECT_ROOT}/terraform/manifests/generated/04-enterprise-gateway-config.yaml"

  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" SERVING_REPLICAS="2" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)

  if ! python3 - "${gateway_yaml}" "${sglang_yaml}" <<'PYEOF'
import sys
from urllib.parse import urlparse

import yaml

gateway_path, serving_path = sys.argv[1], sys.argv[2]
errors = []


def docs(path):
    with open(path, encoding="utf-8") as handle:
        return [d for d in yaml.safe_load_all(handle) if isinstance(d, dict)]


# --- The gateway's view: LiteLLM deployments of the served model ------------------
config = None
for doc in docs(gateway_path):
    if doc.get("kind") == "ConfigMap" and "litellm_config.yaml" in doc.get("data", {}):
        config = yaml.safe_load(doc["data"]["litellm_config.yaml"])
if config is None:
    print("FATAL: no litellm_config.yaml in the rendered gateway ConfigMap", file=sys.stderr)
    sys.exit(1)

entries = config.get("model_list", [])
if len(entries) != 2:
    print(f"FATAL: expected 2 LiteLLM deployments at SERVING_REPLICAS=2, found {len(entries)}", file=sys.stderr)
    sys.exit(1)
a, b = entries

# One model_name across both is the whole mechanism: it is what makes the router
# treat them as interchangeable rather than as two separate models.
if a.get("model_name") != b.get("model_name"):
    errors.append(f"model_name differs: {a.get('model_name')!r} vs {b.get('model_name')!r}; "
                  "the router will not fail over between two differently named models")

if a.get("model_info") != b.get("model_info"):
    errors.append(f"model_info differs: {a.get('model_info')} vs {b.get('model_info')}; "
                  "the same request would be billed differently per replica")

# Everything except the endpoint must match, or the replicas are not interchangeable.
params_a = dict(a.get("litellm_params", {}))
params_b = dict(b.get("litellm_params", {}))
base_a = params_a.pop("api_base", "")
base_b = params_b.pop("api_base", "")
if params_a != params_b:
    errors.append(f"litellm_params differ beyond api_base: {params_a} vs {params_b}")
if base_a == base_b:
    errors.append(f"both deployments point at the same endpoint {base_a!r}; there is no second replica")

# --- The serving manifest's view: which Services exist and what they select -------
services, statefulset = {}, None
for doc in docs(serving_path):
    if doc.get("kind") == "Service":
        services[doc["metadata"]["name"]] = doc
    elif doc.get("kind") == "StatefulSet" and doc["metadata"]["name"] == "kimi-k3-serving":
        statefulset = doc
if statefulset is None:
    print("FATAL: no kimi-k3-serving StatefulSet in the rendered serving manifest", file=sys.stderr)
    sys.exit(1)
pod_count = int(statefulset["spec"]["replicas"])

selected = []
for label, base in (("A", base_a), ("B", base_b)):
    url = urlparse(base)
    name = (url.hostname or "").split(".")[0]
    svc = services.get(name)
    if svc is None:
        errors.append(f"deployment {label} points at Service {name!r}, which the serving "
                      f"manifest does not define (it defines {sorted(services)})")
        continue
    ports = {p.get("port") for p in svc["spec"].get("ports", [])}
    if url.port not in ports:
        errors.append(f"deployment {label} calls {name} on port {url.port}, but that Service "
                      f"exposes {sorted(p for p in ports if p is not None)}")
    ordinal = svc["spec"].get("selector", {}).get("apps.kubernetes.io/pod-index")
    if ordinal is None:
        errors.append(f"Service {name} does not pin a leader ordinal, so it would round-robin "
                      "across every rank in the replica and half the requests would hit a worker")
        continue
    ordinal = int(ordinal)
    if ordinal >= pod_count:
        errors.append(f"Service {name} selects ordinal {ordinal}, but the StatefulSet has only "
                      f"{pod_count} pods; that upstream has no endpoints")
    selected.append((label, name, ordinal))

if len({o for _, _, o in selected}) != len(selected):
    errors.append(f"the two upstreams resolve to the same leader ordinal: {selected}")

for err in errors:
    print(f"DRIFT: {err}", file=sys.stderr)
sys.exit(1 if errors else 0)
PYEOF
  then
    log_fail "[${CURRENT_TEST_ID}] Gateway upstreams and the serving manifest disagree"
    (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
    return 1
  fi

  (cd "${PROJECT_ROOT}" && INFERENCE_ENGINE="sglang" scripts/03_deploy_workloads.sh --render-only >/dev/null 2>&1)
}

run_tier_5_tests() {
  log_info "=== Executing Tier 5: Adversarial White-Box Coverage Hardening Suite ==="
  run_test_case "T5_ADV_01_Domain_Manifest_Assertions" t5_adv_01
  run_test_case "T5_ADV_02_Hermeticity_Empty_String_Injection" t5_adv_02
  run_test_case "T5_ADV_03_Gateway_Observability_Config" t5_adv_03
  run_test_case "T5_ADV_04_Remediated_Script_Security_Network_Isolation" t5_adv_04
  run_test_case "T5_ADV_05_Benchmark_Exception_Resilience_Syntax" t5_adv_05
  run_test_case "T5_ADV_06_Weights_Cache_Consistency" t5_adv_06
  run_test_case "T5_ADV_07_SGLang_Parallel_Profile" t5_adv_07
  run_test_case "T5_ADV_08_Data_Parallel_Serving_Topology" t5_adv_08
  run_test_case "T5_ADV_09_Gateway_Serving_Upstream_Drift" t5_adv_09
}
