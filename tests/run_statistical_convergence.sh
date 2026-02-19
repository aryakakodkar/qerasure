#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-release"

SHOTS="${1:-2000}"
DISTANCE="${2:-7}"
ROUNDS="${3:-1}"
SEED="${4:-12345}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target statistical_convergence_test -j
"${BUILD_DIR}/statistical_convergence_test" "${SHOTS}" "${DISTANCE}" "${ROUNDS}" "${SEED}"
