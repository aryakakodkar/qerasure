#include "qerasure/core/translation/stim_translation.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "stim/circuit/circuit.h"

namespace qerasure {

namespace {

using translation_internal::append_extraction_round;
using translation_internal::append_final_readout_detectors_and_observable;
using translation_internal::build_context;
using translation_internal::CircuitBuildContext;

}  // namespace

// Builds a Stim circuit object for a standard surface code quantum memory.
// Note: this implementation does not use repeat blocks; it's optimized for clarity and error injection paths.
stim::Circuit build_surf_stabilizer_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds) {
  if (qec_rounds < 2) {
    throw std::invalid_argument("qec_rounds must be >= 2 for stabilizer-only circuit generation");
  }

  const CircuitBuildContext ctx = build_context(code);
  const std::size_t extraction_rounds = qec_rounds - 1;

  std::vector<uint32_t> detector_lookbacks;
  detector_lookbacks.reserve(8);
  std::vector<uint32_t> detector_targets;
  detector_targets.reserve(8);

  // Reuse prebuilt round bodies for fast pure-stabilizer generation.
  stim::Circuit first_round_body;
  append_extraction_round(&first_round_body, ctx, 0,
                          [](std::size_t, std::size_t) {},
                          [](std::size_t, std::size_t) {},
                          [](std::size_t) {},
                          &detector_lookbacks, &detector_targets);
  stim::Circuit temporal_round_body;
  append_extraction_round(&temporal_round_body, ctx, 1,
                          [](std::size_t, std::size_t) {},
                          [](std::size_t, std::size_t) {},
                          [](std::size_t) {},
                          &detector_lookbacks, &detector_targets);

  stim::Circuit circuit;
  if (extraction_rounds > 0) {
    circuit += first_round_body;
    for (std::size_t round = 1; round < extraction_rounds; ++round) {
      circuit += temporal_round_body;
    }
  }

  append_final_readout_detectors_and_observable(&circuit, ctx, &detector_lookbacks, &detector_targets);
  return circuit;
}

std::string build_surf_stabilizer_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds) {
  return build_surf_stabilizer_circuit_object(code, qec_rounds).str();
}

stim::Circuit build_surface_code_stim_circuit_object(const RotatedSurfaceCode& code, std::size_t qec_rounds) {
  return build_surf_stabilizer_circuit_object(code, qec_rounds);
}

std::string build_surface_code_stim_circuit(const RotatedSurfaceCode& code, std::size_t qec_rounds) {
  return build_surf_stabilizer_circuit(code, qec_rounds);
}

}  // namespace qerasure
