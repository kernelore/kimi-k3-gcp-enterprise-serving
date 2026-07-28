#!/usr/bin/env bash
# Library for normalizing engine image versions and tags.

normalize_engine_version() {
  local image="${1:-}"
  local tag=""
  if [ -z "${image}" ]; then
    echo "unknown"
    return 0
  fi

  if [[ "${image}" == *"@"* ]]; then
    tag="${image%%@*}"
    tag="${tag#*:}"
  elif [[ "${image}" == *":"* ]]; then
    tag="${image#*:}"
  else
    tag="latest"
  fi

  echo "${tag}"
}
