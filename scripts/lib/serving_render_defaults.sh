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
# be substituted; it says nothing about their values. Three of them are derived
# by 03 at deploy time rather than read from config.env, so every other renderer
# produced them empty -- and an empty image or hostPath is not a degraded
# manifest, it is one the API server refuses outright.
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

  local v
  for v in SERVING_IMAGE SGLANG_HOST_SCRATCH_PATH SGLANG_LOCAL_SCRATCH_MOUNT; do
    if [ -z "${!v:-}" ]; then
      echo "ERROR: ${v} is empty; the rendered StatefulSet would be rejected" >&2
      return 1
    fi
  done
  return 0
}

# Reject a rendered manifest the API server would reject, before it costs a
# variant and a restart cycle to find out on a metered cluster.
assert_manifest_valid() {
  local rendered="$1"
  if kubectl apply --dry-run=client -f "${rendered}" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: rendered manifest ${rendered} is invalid; kubectl reports:" >&2
  kubectl apply --dry-run=client -f "${rendered}" 2>&1 | sed 's/^/       /' >&2
  return 1
}
