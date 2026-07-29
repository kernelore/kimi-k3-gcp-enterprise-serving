#!/usr/bin/env bash
# Shared helper to extract and normalize engine versions from Dockerfiles.
# Usage: get_engine_version <sglang|trtllm> [project_root]
set -euo pipefail

get_engine_version() {
  local engine="$1"
  local root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  local dockerfile=""
  if [ "${engine}" = "sglang" ]; then
    dockerfile="${root}/docker/Dockerfile.sglang"
  elif [ "${engine}" = "trtllm" ]; then
    dockerfile="${root}/docker/Dockerfile"
  else
    echo "ERROR: Unknown engine '${engine}'" >&2
    return 1
  fi
  if [ ! -f "${dockerfile}" ]; then
    echo "ERROR: Dockerfile not found at ${dockerfile}" >&2
    return 1
  fi
  local line image tag digest
  line=$(grep -E '^FROM[[:space:]]+' "${dockerfile}" | head -n 1)
  image=$(echo "${line}" | awk '{print $2}')
  if [ -z "${image}" ]; then
    echo "ERROR: Could not extract image from ${dockerfile}" >&2
    return 1
  fi
  if [[ "${image}" == *"@"* ]]; then
    digest="${image#*@}"
    tag="${image%%@*}"
    tag="${tag#*:}"
  else
    digest=""
    tag="${image#*:}"
  fi
  # Normalize: strip leading v/V, trailing -cu130, trailing -py3
  tag="${tag#v}"
  tag="${tag#V}"
  tag="${tag%-cu130}"
  tag="${tag%-py3}"
  echo "${tag}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <sglang|trtllm> [project_root]" >&2
    exit 1
  fi
  get_engine_version "$@"
fi
