#pragma once

#include <cstddef>
#include <string>

#include "qerasure/core/code/rotated_surface_code.h"

namespace qerasure {

// Build a Stim-format circuit for rotated-surface-code syndrome extraction and readout.
//
// Semantics:
// - Repeats stabilizer extraction for (qec_rounds - 1) rounds.
// - Uses H on X ancillas, then 4 schedule-step CNOT layers, then H on X ancillas.
// - Measures all ancillas each extraction round and emits DETECTOR parity checks.
// - Performs final data measurement, Z-plaquette boundary detectors, and logical-Z observable.
std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds);

}  // namespace qerasure
