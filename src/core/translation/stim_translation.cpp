#include "qerasure/core/translation/stim_translation.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "qerasure/core/lowering/lowering.h"
#include "stim/circuit/circuit.h"
#include "stim/circuit/gate_target.h"

namespace qerasure {

namespace {

// Holds precomputed data for efficient circuit construction later on.
struct CircuitBuildContext {
  std::size_t num_qubits = 0;
  std::size_t num_data = 0;
  std::size_t num_x_anc = 0;
  std::size_t num_z_anc = 0;
  std::size_t num_anc = 0;

  std::vector<uint32_t> data_qubits_u32;
  std::vector<uint32_t> x_ancillas_u32;
  std::vector<uint32_t> ancillas_u32;

  std::vector<std::vector<uint32_t>> cx_targets_by_step;
  std::vector<std::vector<std::size_t>> z_ancilla_supports;
  std::vector<std::size_t> logical_x_data_qubits;
};

std::vector<uint32_t> as_u32_targets(const std::vector<std::size_t>& indices) {
  std::vector<uint32_t> out;
  out.reserve(indices.size());
  for (const std::size_t q : indices) {
    out.push_back(static_cast<uint32_t>(q));
  }
  return out;
}

void append_index_op(stim::Circuit* circuit, const char* op, const std::vector<uint32_t>& indices) {
  if (!indices.empty()) {
    circuit->safe_append_u(op, indices);
  }
}

void append_detector_lookbacks(stim::Circuit* circuit, std::vector<uint32_t>* rec_targets,
                               const std::vector<uint32_t>& rec_lookbacks) {
  rec_targets->clear();
  rec_targets->reserve(rec_lookbacks.size());
  for (const uint32_t lookback : rec_lookbacks) {
    rec_targets->push_back(lookback | stim::TARGET_RECORD_BIT);
  }
  circuit->safe_append_u("DETECTOR", *rec_targets);
}

void append_single_qubit_error(stim::Circuit* circuit, const char* op,
                               const std::vector<uint32_t>& targets) {
  if (!targets.empty()) {
    circuit->safe_append_ua(op, targets, 1.0);
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

// Precomputes the standard features for building the surface code circuit
CircuitBuildContext build_context(const RotatedSurfaceCode& code) {
  CircuitBuildContext ctx;
  ctx.num_qubits = code.num_qubits();
  ctx.num_data = code.x_anc_offset();
  const std::size_t x_anc_offset = code.x_anc_offset();
  const std::size_t z_anc_offset = code.z_anc_offset();
  ctx.num_x_anc = z_anc_offset - x_anc_offset;
  ctx.num_z_anc = ctx.num_qubits - z_anc_offset;
  ctx.num_anc = ctx.num_x_anc + ctx.num_z_anc;

  std::vector<std::size_t> data_qubits;
  data_qubits.reserve(ctx.num_data);
  for (std::size_t q = 0; q < ctx.num_data; ++q) {
    data_qubits.push_back(q);
  }

  std::vector<std::size_t> x_ancillas;
  x_ancillas.reserve(ctx.num_x_anc);
  for (std::size_t q = x_anc_offset; q < z_anc_offset; ++q) {
    x_ancillas.push_back(q);
  }

  std::vector<std::size_t> z_ancillas;
  z_ancillas.reserve(ctx.num_z_anc);
  for (std::size_t q = z_anc_offset; q < ctx.num_qubits; ++q) {
    z_ancillas.push_back(q);
  }

  std::vector<std::size_t> ancillas = x_ancillas;
  ancillas.insert(ancillas.end(), z_ancillas.begin(), z_ancillas.end());

  ctx.data_qubits_u32 = as_u32_targets(data_qubits);
  ctx.x_ancillas_u32 = as_u32_targets(x_ancillas);
  ctx.ancillas_u32 = as_u32_targets(ancillas);

  const std::vector<Gate>& gates = code.gates();
  const std::size_t gates_per_step = code.gates_per_step();
  ctx.cx_targets_by_step.assign(4, {});
  // Flatten gates into array of control-target pairs for each step, for direct CNOT parametrization
  for (std::size_t step = 0; step < 4; ++step) {
    std::vector<uint32_t>& cx_targets = ctx.cx_targets_by_step[step];
    cx_targets.reserve(gates_per_step * 2);
    const std::size_t step_start = step * gates_per_step;
    for (std::size_t i = 0; i < gates_per_step; ++i) {
      const Gate& gate = gates[step_start + i];
      cx_targets.push_back(static_cast<uint32_t>(gate.first));
      cx_targets.push_back(static_cast<uint32_t>(gate.second));
    }
  }

  const std::vector<std::size_t>& partner_map = code.partner_map();
  ctx.z_ancilla_supports.assign(ctx.num_z_anc, {});
  // Compute Z-supports for end-of-circuit measurement
  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    const std::size_t z_anc = z_ancillas[zi];
    std::vector<std::size_t>& support = ctx.z_ancilla_supports[zi];
    support.reserve(4);
    for (std::size_t step = 0; step < 4; ++step) {
      const std::size_t partner = partner_map[step * ctx.num_qubits + z_anc];
      if (partner != kNoPartner && partner < ctx.num_data) {
        support.push_back(partner);
      }
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
  }

  // Identify data qubits in the support of a logical X operator
  const auto& coords = code.index_to_coord();
  for (const std::size_t q : data_qubits) {
    if (coords[q].first == 1) {
      ctx.logical_x_data_qubits.push_back(q);
    }
  }

  return ctx;
}

void append_round_detectors(stim::Circuit* circuit, const CircuitBuildContext& ctx, std::size_t round,
                            std::vector<uint32_t>* detector_lookbacks,
                            std::vector<uint32_t>* detector_targets) {
  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    const std::size_t ancilla_position = ctx.num_x_anc + zi;
    const uint32_t current_lookback = static_cast<uint32_t>(ctx.num_anc - ancilla_position);
    detector_lookbacks->clear();
    detector_lookbacks->push_back(current_lookback);
    if (round > 0) {
      const uint32_t previous_lookback = static_cast<uint32_t>(2 * ctx.num_anc - ancilla_position);
      detector_lookbacks->push_back(previous_lookback);
    }
    append_detector_lookbacks(circuit, detector_targets, *detector_lookbacks);
  }
}

void append_final_readout_detectors_and_observable(stim::Circuit* circuit, const CircuitBuildContext& ctx,
                                                   std::vector<uint32_t>* detector_lookbacks,
                                                   std::vector<uint32_t>* detector_targets) {
  append_index_op(circuit, "M", ctx.data_qubits_u32);

  for (std::size_t zi = 0; zi < ctx.num_z_anc; ++zi) {
    detector_lookbacks->clear();
    const std::size_t ancilla_position = ctx.num_x_anc + zi;
    const uint32_t ancilla_lookback_after_data =
        static_cast<uint32_t>(ctx.num_data + (ctx.num_anc - ancilla_position));
    detector_lookbacks->push_back(ancilla_lookback_after_data);
    for (const std::size_t data_q : ctx.z_ancilla_supports[zi]) {
      detector_lookbacks->push_back(static_cast<uint32_t>(ctx.num_data - data_q));
    }
    append_detector_lookbacks(circuit, detector_targets, *detector_lookbacks);
  }

  std::vector<uint32_t> logical_targets;
  logical_targets.reserve(ctx.logical_x_data_qubits.size());
  for (const std::size_t data_q : ctx.logical_x_data_qubits) {
    logical_targets.push_back((static_cast<uint32_t>(ctx.num_data - data_q)) | stim::TARGET_RECORD_BIT);
  }
  circuit->safe_append_ua("OBSERVABLE_INCLUDE", logical_targets, 0.0);
}

// Hook lambdas allow interleaving of pre-simulated lowered errors
template <typename PreStepHook, typename PostStepHook, typename PreMeasureHook>
void append_extraction_round(stim::Circuit* circuit, const CircuitBuildContext& ctx,
                             std::size_t round_index, PreStepHook&& pre_step_hook,
                             PostStepHook&& post_step_hook, PreMeasureHook&& pre_measure_hook,
                             std::vector<uint32_t>* detector_lookbacks,
                             std::vector<uint32_t>* detector_targets) {
  append_index_op(circuit, "H", ctx.x_ancillas_u32);
  for (std::size_t step = 0; step < 4; ++step) {
    pre_step_hook(round_index, step);
    circuit->safe_append_u("CX", ctx.cx_targets_by_step[step]);
    post_step_hook(round_index, step);
  }
  append_index_op(circuit, "H", ctx.x_ancillas_u32);
  pre_measure_hook(round_index);
  append_index_op(circuit, "MR", ctx.ancillas_u32);
  append_round_detectors(circuit, ctx, round_index, detector_lookbacks, detector_targets);
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

// Builds a Stim circuit object for a standard surface code quantum memory
// Note: this code does not employ repeat blocks, and is therefore inefficient for large numbers of rounds.
// It is intended for injection of pre-simulated errors 
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

}  // namespace qerasure
