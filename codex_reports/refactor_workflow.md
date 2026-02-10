# Refactor Workflow

## Plan followed
This implementation follows `codex_reports/ARCHITECTURE_AND_REFACTOR_PLAN.md`:
- **2) Architectural Changes**: tightened module responsibilities (`code`, `noise`, `simulators`, bindings), added a dedicated benchmark app, and kept Python conversion logic in bindings.
- **3) Readability Changes**: replaced macro guards with `#pragma once`, introduced typed sentinel (`qerasure::kNoPartner`), cleaned getter signatures, and added direct header includes.
- **4) Performance and Memory Review**: reduced hot-path branch waste by precomputing active qubits per schedule step; moved simulator bookkeeping into helper functions.
- **7) Final Checklist**: added smoke test and benchmark workflow, validated clean configure/build/test.

## Commits
- `747d693` - Refactor core APIs and simulator internals
  - Namespaced core types under `qerasure`.
  - Converted noise model from stringly-typed internals to `NoiseChannel` enum-backed storage with compatibility string overloads.
  - Refactored `RotatedSurfaceCode` to use dense coordinate indexing and vector-backed coordinate storage.
  - Split simulator logic into validation + helper methods and added deterministic seed support.
  - Updated pybind bindings to new API surface.
- `e89127e` - Add benchmark target, smoke test, and example workflow
  - Cleaned CMake target layout and retained optional `ENABLE_GPERFTOOLS` support.
  - Added `bench/bench_erasure_sim.cpp` (`bench_erasure_sim`).
  - Added `tests/benchmark_erasure_sim.sh` to build and report shots/sec.
  - Added minimal runnable example `examples/run_erasure_sim.cpp`.
  - Added `tests/smoke_sim_test.cpp` and wired it through CTest.

## Key code moves/renames and why
- **New benchmark entrypoint**: `bench/bench_erasure_sim.cpp` (separates performance runs from app/demo code).
- **New example entrypoint**: `examples/run_erasure_sim.cpp` (documents canonical API usage).
- **New smoke test**: `tests/smoke_sim_test.cpp` (basic functional guard before performance work).
- **No deep refactor in generated/vendored paths**: kept `build/` out of source refactoring scope.

## Ambiguous plan decisions and resolutions
- **Noise API migration risk**: plan requested typed channels. To avoid breaking Python and legacy call sites, I kept string overloads (`set/get(const std::string&)`) and made enum-based APIs canonical.
- **Coordinate map refactor depth**: replaced internal map-heavy lookup with dense array lookup while preserving externally consumed coordinate data.
- **gperftools cleanup**: kept support optional and OFF by default; now it fails fast only if explicitly enabled but not found.

## Build/test/benchmark commands used
- Clean configure: `cmake -S . -B build-clean -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build-clean -j`
- Run smoke test: `ctest --test-dir build-clean --output-on-failure`
- Quick benchmark executable run: `./build-clean/bench_erasure_sim 1000 5 3 123`
- Full benchmark script: `tests/benchmark_erasure_sim.sh`

## Running a simulation (post-refactor)
Minimal end-to-end C++ flow:

```cpp
#include "qerasure/code/code.h"
#include "qerasure/noise/noise.h"
#include "qerasure/simulators/erasure_simulator.h"

qerasure::RotatedSurfaceCode code(5);
qerasure::NoiseParams noise;
noise.set(qerasure::NoiseChannel::kTwoQubitErasure, 0.01);
noise.set(qerasure::NoiseChannel::kErasureCheckError, 0.02);

qerasure::ErasureSimParams params(code, noise, 4, 100, 12345);
qerasure::ErasureSimulator simulator(params);
qerasure::ErasureSimResult result = simulator.simulate();
```

Reference runnable sample: `examples/run_erasure_sim.cpp`.

## Benchmark
- Benchmark executable: `bench_erasure_sim`
- Default parameters: `shots=10000`, `distance=15`, `rounds=15`, `seed=12345`
- Manual run:
  - `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build-release --target bench_erasure_sim -j`
  - `./build-release/bench_erasure_sim 10000 15 15 12345`
- Scripted run from repo root:
  - `tests/benchmark_erasure_sim.sh`
  - Prints: `Shots/sec: <value>`

## Known limitations / follow-ups
- Parallel shot execution from the plan is not implemented yet; simulation remains single-threaded.
- Python wrappers in `python/qerasure/` still use mostly string-key patterns; bindings support both string and enum channels.
- `build/_deps/pybind11-src` fallback is used for offline environments; long-term dependency vendoring policy should be documented explicitly.
