# Refactor Workflow

## Plan followed
This implementation follows `codex_reports/ARCHITECTURE_AND_REFACTOR_PLAN.md`:
- **2) Architectural Changes**: reorganized sources into clear module boundaries:
  - `src/core/code`, `src/core/noise`, `src/core/sim`
  - `src/interop/python`
  - `benchmarks/`
  - canonical public headers in `include/qerasure/core/...`
- **3) Readability Changes**: replaced legacy header patterns with `#pragma once`, moved to typed sentinel (`qerasure::kNoPartner`), and kept stable compatibility headers under `include/qerasure/{code,noise,simulators}`.
- **4) Performance and Memory Review**: optimized simulator hot path via precomputed active qubits and a faster RNG sampling path with integer thresholds.
- **7) Final Checklist**: clean configure/build, smoke test run, and benchmark script validated.

## Commits
- `747d693` - Refactor core APIs and simulator internals
- `e89127e` - Add benchmark target, smoke test, and example workflow
- `164618a` - Document refactor workflow and post-refactor API

## Key code moves/renames and why
- `include/qerasure/core/code/rotated_surface_code.h`
- `include/qerasure/core/noise/noise_params.h`
- `include/qerasure/core/sim/erasure_simulator.h`
- `src/core/code/rotated_surface_code.cpp`
- `src/core/noise/noise_params.cpp`
- `src/core/sim/erasure_simulator.cpp`
- `src/core/sim/internal/fast_rng.h`
- `src/interop/python/pybindings.cpp`
- `benchmarks/bench_erasure_sim.cpp`

Why:
- Separate core compute modules from interop and benchmarks.
- Make internal implementation details explicit under `src/core/.../internal`.
- Keep compatibility include paths while establishing canonical include paths for new code.

## Ambiguous plan decisions and resolutions
- **Noise API migration**: plan requested typed channels. Canonical APIs now use `NoiseChannel`, while string overloads remain for compatibility.
- **Generated/vendor scope**: no deep edits under `build/`; CMake uses local pybind fallback for offline environments.
- **Public include stability**: added compatibility wrappers (`include/qerasure/code/code.h`, etc.) so old includes still compile.

## Build/test/benchmark commands used
- Configure (clean): `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release`
- Build benchmark + smoke test: `cmake --build build-release --target bench_erasure_sim smoke_sim_test -j`
- Smoke test: `ctest --test-dir build-release --output-on-failure`
- Benchmark direct: `./build-release/bench_erasure_sim 10000 15 15 12345`
- Benchmark script: `tests/benchmark_erasure_sim.sh`

## Running a simulation (post-refactor)
Canonical API:

```cpp
#include "qerasure/core/code/rotated_surface_code.h"
#include "qerasure/core/noise/noise_params.h"
#include "qerasure/core/sim/erasure_simulator.h"

qerasure::RotatedSurfaceCode code(5);
qerasure::NoiseParams noise;
noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.01);
noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.02);

qerasure::ErasureSimParams params(code, noise, 4, 100, 12345);
qerasure::ErasureSimulator simulator(params);
qerasure::ErasureSimResult result = simulator.simulate();
```

Runnable sample: `examples/run_erasure_sim.cpp`.

## Benchmark
- Benchmark executable: `bench_erasure_sim` (`benchmarks/bench_erasure_sim.cpp`)
- Defaults: `shots=10000`, `distance=15`, `rounds=15`, `seed=12345`
- Run script from repo root: `tests/benchmark_erasure_sim.sh`
- Current measured result (this branch): `Shots/sec: 22448.95668285694`

## Known limitations / follow-ups
- Parallel shot execution is still not implemented.
- Python wrappers in `python/qerasure/` remain mostly legacy-style convenience wrappers and can be aligned further to canonical typed APIs.
