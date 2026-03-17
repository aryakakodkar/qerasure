# Rail-Resolved Decoder Implementation Report

## Branch
- `feature/rail-resolved-decoder`

## Goal Implemented
- Added a new, surface-code-specific rail-resolved pathway that is separate from the main `qerasure` pipeline.
- The new pathway supports sparse ECR schedules (including non-every-round checks) and is explicitly constrained to `max_persistence=2`.

## New Architecture (Additive)
- New compiler wrapper:
  - `src/core/circuit/rail_surface_compile.h`
  - `src/core/circuit/rail_surface_compile.cc`
- New sampler:
  - `src/core/simulator/rail_stream_sampler.h`
  - `src/core/simulator/rail_stream_sampler.cc`
- New decoder/DEM builder:
  - `src/core/decode/rail_surface_dem_builder.h`
  - `src/core/decode/rail_surface_dem_builder.cc`

These are additive and do not replace existing `CompiledErasureProgram`, `StreamSampler`, or `SurfDemBuilder`.

## Behavior Implemented

### 1) Rail surface compile metadata
- `RailSurfaceCompiledProgram` wraps a standard `CompiledErasureProgram` and adds surface-specific metadata:
  - `op_index -> round` mapping.
  - `check_event -> (qubit, op_index)` lookup.
  - Data-qubit adjacent Z-ancilla slot mapping from `RotatedSurfaceCode`.
  - Interaction-op lookup for `(data_qubit, round, z_ancilla)`.
  - Round detector lookup for `(round, z_ancilla)`.
- Enforces `max_persistence == 2` for this pathway.

### 2) Rail-resolved sampling
- `RailStreamSampler` keeps existing onset and persistent spread semantics, then adds rail-specific spread:
  - When a data qubit becomes erased (transition to erased state), it samples a latent Z-ancilla rail:
    - Interior data qubits: choose one of two adjacent Z ancillas uniformly.
    - Boundary data qubits: 50% choose the lone adjacent Z ancilla, 50% choose none.
  - At each `CX` where that data qubit interacts with the selected Z ancilla while already erased, inject `X_ERROR(1)` on that ancilla.
  - Rail selection is cleared when reset succeeds.
- Measurement randomization behavior remains aligned with the corrected stream path (`state >= 1`).

### 3) Rail-conditioned decoding
- `RailSurfaceDemBuilder` composes:
  - Base `SurfDemBuilder` spread injections.
  - Additional rail injections calibrated using detector evidence.
- For each flagged **data-qubit** check event:
  - Rebuilds onset candidates over the same lookback window used by compiled metadata.
  - Applies check-likelihood terms (`fn/fp`, with `mp=2` forced-detection semantics).
  - Reweights candidate onsets using local adjacent Z-detector evidence from the last two rounds.
  - Marginalizes over latent rail hypotheses (selected slot or none).
  - Injects posterior-weighted `X` mass on corresponding Z-ancilla interaction emission points.
- Probability safety:
  - Rail merge respects remaining `PAULI_CHANNEL_1` budget so `p_x + p_y + p_z <= 1` is preserved.

## Python and Binding Integration

### New pybind files
- `src/core/circuit/pybind_rail_surface_compile.cc`
- `src/core/simulator/pybind_rail_stream_sampler.cc`
- `src/core/decode/pybind_rail_surface_dem_builder.cc`

### Existing binding entrypoint updated
- `src/core/circuit/pybind_module.cc`

### Python API additions
- `python/qerasure/circuit_model_utils.py`
  - `RailSurfaceCompiledProgram`
  - `RailStreamSampler`
  - `RailSurfaceDemBuilder`
  - `compile_rail_surface_sampler(...)`
- `python/qerasure/__init__.py`
  - exports for the new rail classes/helpers.

## Build System Updates
- `CMakeLists.txt`
  - Added new rail C++ sources to `qerasure`.
  - Added new rail pybind sources to `qerasure_python`.
  - Added new test target `tests/rail_surface_path_test.cpp`.

## Tests Added
- `tests/rail_surface_path_test.cpp`
  - Verifies rail sampler path runs and output dimensions are consistent.
  - Verifies detector-conditioned rail posterior biases X-injection mass toward evidence-consistent Z ancilla.

## Validation Performed
- Build:
  - `cmake -S . -B build-release -DQERASURE_BUILD_PYTHON=ON`
  - `cmake --build build-release -j4`
- Focused tests:
  - `ctest --test-dir build-release -R 'rail_surface_path_test|surf_rounds_per_check_test|surf_tail_correction_test' --output-on-failure`
- Full configured test suite:
  - `ctest --test-dir build-release --output-on-failure`
- Python smoke:
  - Imported new API via `PYTHONPATH=python`.
  - Built rail program, sampled syndromes, and built decoded circuit for one shot.

## Notes
- This implementation intentionally prioritizes a surface-code-specific path and does not attempt to keep the same flexibility as the main generic pathway.
- The rail-conditioned posterior currently augments the existing base `SurfDemBuilder` injections rather than replacing the entire generic posterior engine.
