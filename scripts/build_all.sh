#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-release}"
JOBS="${JOBS:-4}"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${JOBS}"
cmake --build "${BUILD_DIR}" --target qerasure_python -j"${JOBS}"
"./${BUILD_DIR}/stim_translation_test"
