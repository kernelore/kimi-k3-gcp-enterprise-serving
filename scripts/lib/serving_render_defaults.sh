#!/usr/bin/env bash
# ==============================================================================
# serving_render_defaults.sh - values a re-render of the serving manifest needs
# ==============================================================================
# 03_deploy_workloads.sh is not the only script that renders the serving
# StatefulSet. The tuning sweep re-renders it once per variant, and the
# measurement window re-renders it for the fp8-KV and prefix-reuse legs. Both of
# those already shared 03's BASE_ALLOWED_VARS list, on the reasoning that a
# variable added in one place should not silently render empty in another.
#
# That was half of what it needed to be. The allow-list controls which names may
# be substituted; it says nothing about their values. Some are derived by 03 at
# deploy time rather than read from config.env, so every other renderer produced
# them empty -- and an empty image or hostPath is not a degraded manifest, it is
# one the API server refuses outright.
#
# The first live sweep failed all nine variants inside a minute for exactly this
# reason, against a cluster that was up and answering inference correctly:
#
#   spec.template.spec.containers[0].image: Required value
#   spec.template.spec.volumes[3].hostPath.path: Required value
#   spec.template.spec.containers[0].volumeMounts[2].mountPath: Required value
#
# So the values live here, sourced by every renderer except 03 itself. 03 keeps
# its own inline defaults: it derives them amongst a good deal of other deploy
# logic, and rewriting the one script whose deploy path is known to work was not
# worth the risk of fixing the two that did not.
#
# The first fix listed only the three variables that had actually broken, which
# left the same trap armed for every other value 03 defaults and config.env does
# not carry. SGLANG_PORT is one: config.env.example ships it commented out,
# because the port is fixed at 8000 everywhere it matters, so a real config.env
# is unlikely to set it and 03 supplies the 8000 itself. The measurement window's
# fp8-KV leg rendered `--port ""`, the API server accepted that manifest without
# complaint -- an empty flag value is perfectly valid YAML -- and both pods then
# crash-looped on `argument --port: invalid int value: ''`, eighteen minutes of
# rollout later. Structural validity is not the same property as a usable
# manifest, and it was the only one being checked.
#
# Hence both halves below: every default 03 derives is mirrored here, and
# assert_manifest_valid additionally refuses any manifest that renders a CLI flag
# with an empty value. The mirrored list stops the known cases; the flag check
# stops the class, including whatever gets added to the template next.
#
# Callers must define NAMESPACE and STATEFULSET, and have config.env sourced.
# ==============================================================================

# Populate and export the render values that config.env does not carry.
# Returns nonzero if any required value is still empty afterwards, which is
# always a bug in the caller rather than a condition worth continuing past.
ensure_serving_render_env() {
  # Prefer what is actually deployed. The sweep and the window both mutate a
  # StatefulSet that already exists, so anything they are not deliberately
  # changing should match the running object -- including the image actually
  # being served, which re-deriving a :latest tag does not guarantee.
  if [ -z "${SERVING_IMAGE:-}" ] && [ -n "${STATEFULSET:-}" ] && [ -n "${NAMESPACE:-}" ]; then
    SERVING_IMAGE="$(kubectl get "statefulset/${STATEFULSET}" -n "${NAMESPACE}" \
      -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || true)"
  fi
  if [ -z "${SERVING_IMAGE:-}" ]; then
    # Nothing deployed yet: fall back to the coordinates 03 builds to.
    case "${INFERENCE_ENGINE:-sglang}" in
      sglang) SERVING_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod/sglang-blackwell:latest" ;;
      trtllm) SERVING_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod/trtllm-blackwell:latest" ;;
      *)
        echo "ERROR: unsupported INFERENCE_ENGINE '${INFERENCE_ENGINE:-}'" >&2
        return 1
        ;;
    esac
  fi
  export SERVING_IMAGE

  # Must match the mount point created by the local-nvme-raid-formatter DaemonSet
  # in terraform/manifests/templates/00-local-nvme-raid.yaml.template, and the
  # defaults in 03. Empty renders an invalid hostPath volume, not an absent one.
  export SGLANG_HOST_SCRATCH_PATH="${SGLANG_HOST_SCRATCH_PATH:-/mnt/disks/local-scratch}"
  export SGLANG_LOCAL_SCRATCH_MOUNT="${SGLANG_LOCAL_SCRATCH_MOUNT:-/mnt/scratch}"

  # Every remaining value 03 defaults rather than reads from config.env. Each is
  # ${VAR:-default}, so a caller deliberately overriding one -- a sweep variant
  # setting mem-fraction, the window's fp8 leg setting the KV dtype -- still wins.
  # Values that 03 defaults to empty are not repeated here: empty is their correct
  # rendering, and it makes the template omit the flag rather than pass it blank.
  export SGLANG_PORT="${SGLANG_PORT:-8000}"
  export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
  export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
  export SGLANG_SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-lpm}"
  export SGLANG_ENABLE_TORCH_COMPILE="${SGLANG_ENABLE_TORCH_COMPILE:-false}"
  export SGLANG_ENABLE_HIERARCHICAL_CACHE="${SGLANG_ENABLE_HIERARCHICAL_CACHE:-false}"
  export MIN_WEIGHTS_GIB="${MIN_WEIGHTS_GIB:-1000}"
  export LEADER_ADDR="${LEADER_ADDR:-kimi-k3-serving-0.kimi-k3-workers-headless.llm-serving.svc.cluster.local}"
  export TRTLLM_VIP="${TRTLLM_VIP:-kimi-k3-serving-svc.llm-serving.svc.cluster.local}"
  export INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
  export INFERENCE_SERVER_LABEL="${INFERENCE_SERVER_LABEL:-${INFERENCE_ENGINE}}"

  # Serving topology. NODES_PER_REPLICA sizes one replica's NCCL world; SERVING_PODS
  # is the StatefulSet's replica count across every serving replica.
  export NODES_PER_REPLICA="${NODES_PER_REPLICA:-2}"
  export SERVING_REPLICAS="${SERVING_REPLICAS:-1}"

  # Prefer the deployed pod count, for the same reason SERVING_IMAGE does, but with
  # sharper teeth: a sweep re-render that recomputed this from a SERVING_REPLICAS the
  # operator passed to 03 on the command line -- and therefore not present in
  # config.env -- would render `replicas: 2` against a four-pod StatefulSet and delete
  # a whole serving replica between variants. Reading the live object cannot make that
  # mistake. Anything other than a positive integer (absent object, or a turndown
  # CronJob having scaled it to 0) falls through to the derived value.
  if [ -z "${SERVING_PODS:-}" ] && [ -n "${STATEFULSET:-}" ] && [ -n "${NAMESPACE:-}" ]; then
    local deployed_pods
    deployed_pods="$(kubectl get "statefulset/${STATEFULSET}" -n "${NAMESPACE}" \
      -o jsonpath='{.spec.replicas}' 2>/dev/null || true)"
    case "${deployed_pods}" in
      ''|*[!0-9]*|0) : ;;
      *) SERVING_PODS="${deployed_pods}" ;;
    esac
  fi
  export SERVING_PODS="${SERVING_PODS:-$(( SERVING_REPLICAS * NODES_PER_REPLICA ))}"

  # A partial trailing replica is not a smaller deployment, it is a replica whose
  # NCCL world never forms: its ranks wait on peers that do not exist until the
  # 3600s dist-init timeout expires, with every GPU in the pool billed throughout.
  if [ $(( SERVING_PODS % NODES_PER_REPLICA )) -ne 0 ]; then
    echo "ERROR: SERVING_PODS (${SERVING_PODS}) is not a multiple of NODES_PER_REPLICA (${NODES_PER_REPLICA});" >&2
    echo "       the trailing replica would hang in dist-init waiting for peers that do not exist." >&2
    return 1
  fi
  # Keep the derived count consistent with whatever SERVING_PODS ended up being, so a
  # value read from the cluster does not leave a stale SERVING_REPLICAS beside it.
  export SERVING_REPLICAS=$(( SERVING_PODS / NODES_PER_REPLICA ))

  # Replica B's leader is the first ordinal after replica A's block. Rendered even at
  # one replica: the Service that selects it then simply has no endpoints.
  export REPLICA_B_LEADER_POD_INDEX="${NODES_PER_REPLICA}"

  # The parallel geometry, including the tp8pp2 profile's fixed triple. 03
  # validates TP*PP against the node count and rejects an EP that is neither 1
  # nor TP; a re-render inherits an already-validated geometry, so this mirrors
  # the values without repeating the guards.
  export SGLANG_PARALLEL_PROFILE="${SGLANG_PARALLEL_PROFILE:-tp16}"
  if [ "${SGLANG_PARALLEL_PROFILE}" = "tp8pp2" ]; then
    export SGLANG_TP_SIZE="8" SGLANG_PP_SIZE="2" SGLANG_EP_SIZE="8"
  else
    export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-16}"
    export SGLANG_PP_SIZE="${SGLANG_PP_SIZE:-1}"
    export SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-16}"
  fi

  local v
  for v in SERVING_IMAGE SGLANG_HOST_SCRATCH_PATH SGLANG_LOCAL_SCRATCH_MOUNT \
           SGLANG_PORT SGLANG_CONTEXT_LENGTH SGLANG_MEM_FRACTION_STATIC \
           SGLANG_SCHEDULE_POLICY MIN_WEIGHTS_GIB LEADER_ADDR TRTLLM_VIP \
           INFERENCE_ENGINE INFERENCE_SERVER_LABEL SGLANG_PARALLEL_PROFILE \
           SGLANG_TP_SIZE SGLANG_PP_SIZE SGLANG_EP_SIZE \
           NODES_PER_REPLICA SERVING_REPLICAS SERVING_PODS REPLICA_B_LEADER_POD_INDEX; do
    if [ -z "${!v:-}" ]; then
      echo "ERROR: ${v} is empty; the rendered StatefulSet would be rejected" >&2
      return 1
    fi
  done
  return 0
}

# Reject a rendered manifest the API server would reject, before it costs a
# variant and a restart cycle to find out on a metered cluster.
#
# "Valid" here means two separate things, because checking only the first one is
# what let `--port ""` through. The API server validates structure: required
# fields present, types right. It has no opinion on the contents of a container's
# command, so a flag rendered with an empty value passes the dry run and fails
# eighteen minutes later in a crash loop, which is the most expensive place on
# this cluster to discover a substitution bug.
assert_manifest_valid() {
  local rendered="$1" empty_flags

  # No template contains a literal `--flag ""`; every occurrence is a
  # substitution that came out empty, so this needs no allow-list of exceptions.
  # It deliberately does not try to know which flags matter: the engine rejects
  # `--port ""` and quietly misreads others, and neither is worth a restart.
  empty_flags="$(grep -cE -- '--[a-z0-9-]+ +""' "${rendered}" 2>/dev/null || true)"
  if [ "${empty_flags:-0}" -gt 0 ]; then
    echo "ERROR: rendered manifest ${rendered} passes CLI flags with empty values:" >&2
    grep -nE -- '--[a-z0-9-]+ +""' "${rendered}" | sed 's/^/       /' >&2
    echo "       A variable in the render environment is unset. The API server would" >&2
    echo "       accept this manifest and the engine would crash-loop on it." >&2
    return 1
  fi

  if kubectl apply --dry-run=client -f "${rendered}" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: rendered manifest ${rendered} is invalid; kubectl reports:" >&2
  kubectl apply --dry-run=client -f "${rendered}" 2>&1 | sed 's/^/       /' >&2
  return 1
}
