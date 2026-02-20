#pragma once

#include <cstddef>
#include <string>

#include "qerasure/core/code/rotated_surface_code.h"

namespace stim {
struct Circuit;
}

namespace qerasure {
struct LoweringResult;

// Build a Stim-format circuit for rotated-surface-code syndrome extraction and readout.
//
// Semantics:
// - Repeats stabilizer extraction for (qec_rounds - 1) rounds.
// - Uses H on X ancillas, then 4 schedule-step CNOT layers, then H on X ancillas.
// - Measures all ancillas each extraction round and emits DETECTOR parity checks.
// - Performs final data measurement, Z-plaquette boundary detectors, and logical-Z observable.
stim::Circuit build_surf_stabilizer_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds);
std::string build_surf_stabilizer_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);

// Backward-compatible aliases.
stim::Circuit build_surface_code_stim_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds);
std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);

// Build the logically equivalent circuit from lowered erasure events for one shot.
// Lowering errors are injected immediately after the gate-step where they occur.
stim::Circuit build_logical_stabilizer_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);
std::string build_logical_stabilizer_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);

// Backward-compatible aliases.
stim::Circuit build_logically_equivalent_erasure_stim_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);
std::string build_logically_equivalent_erasure_stim_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index = 0);

}  // namespace qerasure
