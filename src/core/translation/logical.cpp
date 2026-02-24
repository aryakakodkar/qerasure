#include "qerasure/core/translation/stim_translation.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "qerasure/core/lowering/lowering.h"
#include "common.h"
#include "stim/circuit/circuit.h"

namespace qerasure {

namespace {

using translation_internal::append_extraction_round;
using translation_internal::append_final_readout_detectors_and_observable;
using translation_internal::build_context;
using translation_internal::CircuitBuildContext;

// All single-qubit errors are guaranteed in the logically equivalent circuit
void append_single_qubit_error(stim::Circuit* circuit, const char* op,
                               const std::vector<uint32_t>& targets) {
  double p = 1.0;
  if (op[0] == 'D') { // Fast check, but not very robust
    p = 0.75;
  }
  if (!targets.empty()) {
    circuit->safe_append_ua(op, targets, p);
  }
}

std::size_t infer_qec_rounds_from_lowering_offsets(const std::vector<std::size_t>& offsets,
                                                   std::size_t explicit_rounds) {
  if (explicit_rounds != 0) {
    return explicit_rounds;
  }
  if (offsets.size() < 2 || ((offsets.size() - 2) % 4) != 0) {
    throw std::invalid_argument("LoweringResult has invalid timestep offsets for round inference");
  }
  return (offsets.size() - 2) / 4;
}

void append_lowering_timestep_errors(stim::Circuit* circuit, const std::vector<LoweredErrorEvent>& events,
                                     std::size_t start, std::size_t end,
                                     LoweredEventOrigin origin_filter,
                                     std::vector<uint32_t>* x_targets,
                                     std::vector<uint32_t>* y_targets,
                                     std::vector<uint32_t>* z_targets,
                                     std::vector<uint32_t>* d1_targets) {
  x_targets->clear();
  y_targets->clear();
  z_targets->clear();
  d1_targets->clear();

  for (std::size_t k = start; k < end; ++k) {
    const LoweredErrorEvent& event = events[k];
    if (event.origin != origin_filter) {
      continue;
    }
    const uint32_t q = static_cast<uint32_t>(event.qubit_idx);
    switch (event.error_type) {
      case PauliError::NO_ERROR:
        break;
      case PauliError::X_ERROR:
        x_targets->push_back(q);
        break;
      case PauliError::Y_ERROR:
        y_targets->push_back(q);
        break;
      case PauliError::Z_ERROR:
        z_targets->push_back(q);
        break;
      case PauliError::DEPOLARIZE:
        d1_targets->push_back(q);
        break;
    }
  }

  append_single_qubit_error(circuit, "X_ERROR", *x_targets);
  append_single_qubit_error(circuit, "Y_ERROR", *y_targets);
  append_single_qubit_error(circuit, "Z_ERROR", *z_targets);
  append_single_qubit_error(circuit, "DEPOLARIZE1", *d1_targets);
}

}  // namespace

// Builds a Stim circuit object for a surface code quantum memory with interleaved errors from 
// results of a lowered erasure simulation. Resulting circuit is logically equivalent to erasure
// circuit under specified lowering assumptions
stim::Circuit build_logical_stabilizer_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index) {
  if (shot_index >= lowering_result.sparse_cliffords.size() ||
      shot_index >= lowering_result.clifford_timestep_offsets.size()) {
    throw std::out_of_range("shot_index exceeds LoweringResult shot count");
  }

  const std::vector<LoweredErrorEvent>& shot_events = lowering_result.sparse_cliffords[shot_index];
  const std::vector<std::size_t>& offsets = lowering_result.clifford_timestep_offsets[shot_index];
  const std::size_t qec_rounds = infer_qec_rounds_from_lowering_offsets(offsets, lowering_result.qec_rounds);
  if (qec_rounds == 0) {
    throw std::invalid_argument("LoweringResult must contain at least one QEC round");
  }
  if (offsets.size() != qec_rounds * 4 + 2) {
    throw std::invalid_argument("LoweringResult timestep offsets do not match qec_rounds");
  }

  const CircuitBuildContext ctx = build_context(code);

  stim::Circuit circuit;
  std::vector<uint32_t> detector_lookbacks;
  detector_lookbacks.reserve(8);
  std::vector<uint32_t> detector_targets;
  detector_targets.reserve(8);

  std::vector<uint32_t> x_error_targets;
  std::vector<uint32_t> y_error_targets;
  std::vector<uint32_t> z_error_targets;
  std::vector<uint32_t> depolarize_targets;
  x_error_targets.reserve(16);
  y_error_targets.reserve(16);
  z_error_targets.reserve(16);
  depolarize_targets.reserve(16);

  auto pre_step_hook = [](std::size_t, std::size_t) {}; // Currently no errors are injected before gates

  auto post_step_hook = [&](std::size_t round, std::size_t step) {
    const std::size_t timestep = round * 4 + step;
    append_lowering_timestep_errors(&circuit, shot_events, offsets[timestep], offsets[timestep + 1],
                                    LoweredEventOrigin::SPREAD,
                                    &x_error_targets, &y_error_targets, &z_error_targets,
                                    &depolarize_targets);
  };
  auto pre_measure_hook = [&](std::size_t round) {
    const std::size_t reset_timestep = (round + 1) * 4;
    if (reset_timestep + 1 < offsets.size()) {
      append_lowering_timestep_errors(&circuit, shot_events, offsets[reset_timestep],
                                      offsets[reset_timestep + 1], LoweredEventOrigin::RESET,
                                      &x_error_targets, &y_error_targets, &z_error_targets,
                                      &depolarize_targets);
    }
  };

  for (std::size_t round = 0; round < qec_rounds; ++round) {
    append_extraction_round(&circuit, ctx, round, pre_step_hook, post_step_hook, pre_measure_hook,
                            &detector_lookbacks, &detector_targets);
  }

  const std::size_t terminal_timestep = qec_rounds * 4;
  append_lowering_timestep_errors(&circuit, shot_events, offsets[terminal_timestep],
                                  offsets[terminal_timestep + 1], LoweredEventOrigin::SPREAD,
                                  &x_error_targets,
                                  &y_error_targets, &z_error_targets, &depolarize_targets);

  append_final_readout_detectors_and_observable(&circuit, ctx, &detector_lookbacks, &detector_targets);
  return circuit;
}

std::string build_logical_stabilizer_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index) {
  return build_logical_stabilizer_circuit_object(code, lowering_result, shot_index).str();
}

stim::Circuit build_logically_equivalent_erasure_stim_circuit_object(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index) {
  return build_logical_stabilizer_circuit_object(code, lowering_result, shot_index);
}

std::string build_logically_equivalent_erasure_stim_circuit(
    const RotatedSurfaceCode& code, const LoweringResult& lowering_result, std::size_t shot_index) {
  return build_logical_stabilizer_circuit(code, lowering_result, shot_index);
}

}  // namespace qerasure
