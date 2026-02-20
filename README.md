# qerasure
A library for simulation of quantum erasure codes.

## Overview
`qerasure` currently supports an end-to-end workflow:
1. Perform erasure simulation on a rotated surface code.
2. Apply bespoke erasure lowering to convert erasure dynamics into Pauli-error events.
3. Translate lowered events into a Stim-compatible stabilizer circuit that is logically equivalent under the lowering assumptions.

## Pipeline
### 1) Erasure Simulation
- Construct `RotatedSurfaceCode`, `NoiseParams`, and `ErasureSimParams`.
- Run `ErasureSimulator::simulate()`.
- Output is sparse per-shot, per-timestep erasure/reset/check-error data.

### 2) Bespoke Erasure Lowering
- Configure `LoweringParams` (spread program + reset-error model).
- Run `Lowerer::lower(sim_result)`.
- Output is sparse lowered Pauli events with event origin tags (`SPREAD` or `RESET`).
- Current support is data-qubit-centric spread modeling (via data-qubit partner slots).

### 3) Stim Translation
- Build a circuit using `build_logical_stabilizer_circuit(code, lowering_result, shot_index)`.
- The translator injects lowered events into an unrolled stabilizer schedule:
  - spread-origin events are applied after their gate step;
  - reset-origin events are applied after the second Hadamard and before ancilla measurement (`MR`) in the corresponding round.
- Result is a Stim-format circuit string for analysis/decoder workflows.

## Why this design
- Separates physical erasure simulation from decoder-facing circuit generation.
- Keeps lowering assumptions explicit and configurable.
- Enables deterministic shot-level debugging, visualization, and benchmarking.
