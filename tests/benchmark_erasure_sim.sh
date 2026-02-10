#!/usr/bin/env bash
set -euo pipefail

SHOTS=10000
DISTANCE=15
ROUNDS=15
SEED=12345
BUILD_DIR="build-release"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target bench_erasure_sim -j

elapsed_and_rate=$(python3 - <<'PY'
import subprocess
import time

shots = 10000
cmd = ["./build-release/bench_erasure_sim", "10000", "15", "15", "12345"]
start = time.perf_counter()
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
elapsed = time.perf_counter() - start
rate = shots / elapsed
print(f"{elapsed} {rate}")
PY
)

shots_per_sec=$(echo "${elapsed_and_rate}" | awk '{print $2}')
echo "Shots/sec: ${shots_per_sec}"
