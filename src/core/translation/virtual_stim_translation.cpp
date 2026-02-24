#include "qerasure/core/translation/stim_translation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "qerasure/core/lowering/lowering.h"
#include "stim/circuit/circuit.h"
#include "stim/circuit/gate_target.h"

namespace qerasure {

namespace {

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

double clamp_probability(double p) {
  if (p <= 0.0) {
    return 0.0;
  }
  if (p >= 1.0) {
    return 1.0;
  }
  return p;
}

std::array<double, 4> first_erasure_distribution_by_step(
    const std::array<bool, 4>& active_by_step, double two_qubit_erasure_probability,
    bool condition_on_erasure_in_round) {
  const double p = clamp_probability(two_qubit_erasure_probability);
  std::array<double, 4> out = {0.0, 0.0, 0.0, 0.0};
  double survival = 1.0;
  for (std::size_t step = 0; step < 4; ++step) {
    if (!active_by_step[step]) {
      continue;
    }
    out[step] = survival * p;
    survival *= (1.0 - p);
  }

  if (condition_on_erasure_in_round) {
    const double p_any = 1.0 - survival;
    if (p_any <= 0.0) {
      return {0.0, 0.0, 0.0, 0.0};
    }
    for (std::size_t step = 0; step < 4; ++step) {
      out[step] /= p_any;
    }
  }
  return out;
}

std::size_t resolve_data_slot_qubit(const RotatedSurfaceCode& code, std::size_t data_qubit_idx,
                                    PartnerSlot slot) {
  const std::size_t slot_idx = static_cast<std::size_t>(slot);
  if (slot_idx == 0) {
    return code.data_to_x_ancilla_slots()[data_qubit_idx].first;
  }
  if (slot_idx == 1) {
    return code.data_to_x_ancilla_slots()[data_qubit_idx].second;
  }
  if (slot_idx == 2) {
    return code.data_to_z_ancilla_slots()[data_qubit_idx].first;
  }
  if (slot_idx == 3) {
    return code.data_to_z_ancilla_slots()[data_qubit_idx].second;
  }
  throw std::invalid_argument("Spread target slot index out of range");
}

uint32_t pauli_target_u32(PauliError error_type, std::size_t qubit_idx) {
  const uint32_t q = static_cast<uint32_t>(qubit_idx);
  switch (error_type) {
    case PauliError::X_ERROR:
      return q | stim::TARGET_PAULI_X_BIT;
    case PauliError::Y_ERROR:
      return q | stim::TARGET_PAULI_X_BIT | stim::TARGET_PAULI_Z_BIT;
    case PauliError::Z_ERROR:
      return q | stim::TARGET_PAULI_Z_BIT;
    case PauliError::NO_ERROR:
    case PauliError::DEPOLARIZE:
      break;
  }
  throw std::invalid_argument("Pauli target conversion requires X/Y/Z error type");
}

bool is_cond_type(SpreadInstructionType type) {
  return type == SpreadInstructionType::COND_X_ERROR || type == SpreadInstructionType::COND_Y_ERROR ||
         type == SpreadInstructionType::COND_Z_ERROR;
}

bool is_else_type(SpreadInstructionType type) {
  return type == SpreadInstructionType::ELSE_X_ERROR || type == SpreadInstructionType::ELSE_Y_ERROR ||
         type == SpreadInstructionType::ELSE_Z_ERROR;
}

bool is_simple_type(SpreadInstructionType type) {
  return type == SpreadInstructionType::X_ERROR || type == SpreadInstructionType::Y_ERROR ||
         type == SpreadInstructionType::Z_ERROR || type == SpreadInstructionType::DEPOLARIZE1;
}

PauliError pauli_from_instruction_type(SpreadInstructionType type) {
  switch (type) {
    case SpreadInstructionType::X_ERROR:
    case SpreadInstructionType::COND_X_ERROR:
    case SpreadInstructionType::ELSE_X_ERROR:
      return PauliError::X_ERROR;
    case SpreadInstructionType::Y_ERROR:
    case SpreadInstructionType::COND_Y_ERROR:
    case SpreadInstructionType::ELSE_Y_ERROR:
      return PauliError::Y_ERROR;
    case SpreadInstructionType::Z_ERROR:
    case SpreadInstructionType::COND_Z_ERROR:
    case SpreadInstructionType::ELSE_Z_ERROR:
      return PauliError::Z_ERROR;
    case SpreadInstructionType::DEPOLARIZE1:
      return PauliError::DEPOLARIZE;
  }
  throw std::invalid_argument("Unsupported spread instruction type");
}

void append_virtual_timestep_spread_errors(stim::Circuit* circuit, const RotatedSurfaceCode& code,
                                           const CircuitBuildContext& ctx, std::size_t step,
                                           const SpreadProgram& spread_program,
                                           double two_qubit_erasure_probability,
                                           bool condition_on_erasure_in_round) {
  std::vector<uint32_t> correlated_targets;
  correlated_targets.reserve(4);
  std::vector<uint32_t> single_target(1);

  for (std::size_t data_q = 0; data_q < ctx.num_data; ++data_q) {
    std::array<bool, 4> active_by_step = {
        code.partner_map()[0 * ctx.num_qubits + data_q] != kNoPartner,
        code.partner_map()[1 * ctx.num_qubits + data_q] != kNoPartner,
        code.partner_map()[2 * ctx.num_qubits + data_q] != kNoPartner,
        code.partner_map()[3 * ctx.num_qubits + data_q] != kNoPartner,
    };
    const std::array<double, 4> first_probs =
        first_erasure_distribution_by_step(active_by_step, two_qubit_erasure_probability,
                                           condition_on_erasure_in_round);

    std::array<double, 4> erased_by_step = {0.0, 0.0, 0.0, 0.0};
    double cumulative = 0.0;
    for (std::size_t s = 0; s < 4; ++s) {
      cumulative += first_probs[s];
      erased_by_step[s] = cumulative;
    }

    const std::size_t current_partner = code.partner_map()[step * ctx.num_qubits + data_q];
    if (current_partner == kNoPartner) {
      continue;
    }

    bool emitted_correlated_chain = false;
    for (const SpreadInstruction& instruction : spread_program.instructions) {
      const double instruction_p = clamp_probability(instruction.probability);
      correlated_targets.clear();
      const std::size_t target_qubit = resolve_data_slot_qubit(code, data_q, instruction.target_slot);
      bool feasible_for_step = true;
      if (target_qubit == kNoPartner || target_qubit != current_partner) {
        feasible_for_step = false;
      } else {
        const PauliError error_type = pauli_from_instruction_type(instruction.type);
        if (error_type == PauliError::DEPOLARIZE) {
          single_target[0] = static_cast<uint32_t>(target_qubit);
        } else {
          correlated_targets.push_back(pauli_target_u32(error_type, target_qubit));
        }
      }
      const double p_apply_now =
          feasible_for_step ? clamp_probability(erased_by_step[step] * instruction_p) : 0.0;

      if (is_simple_type(instruction.type)) {
        if (p_apply_now <= 0.0) {
          continue;
        }
        const PauliError error_type = pauli_from_instruction_type(instruction.type);
        if (error_type == PauliError::X_ERROR) {
          circuit->safe_append_ua("X_ERROR", single_target, p_apply_now);
        } else if (error_type == PauliError::Y_ERROR) {
          circuit->safe_append_ua("Y_ERROR", single_target, p_apply_now);
        } else if (error_type == PauliError::Z_ERROR) {
          circuit->safe_append_ua("Z_ERROR", single_target, p_apply_now);
        } else if (error_type == PauliError::DEPOLARIZE) {
          circuit->safe_append_ua("DEPOLARIZE1", single_target, p_apply_now);
        }
        continue;
      }

      if (is_cond_type(instruction.type)) {
        circuit->safe_append_ua("CORRELATED_ERROR", correlated_targets, p_apply_now);
        emitted_correlated_chain = true;
        continue;
      }
      if (is_else_type(instruction.type)) {
        if (!emitted_correlated_chain) {
          circuit->safe_append_ua("CORRELATED_ERROR", {}, 0.0);
          emitted_correlated_chain = true;
        }
        const double else_conditional_p = feasible_for_step ? instruction_p : 0.0;
        circuit->safe_append_ua("ELSE_CORRELATED_ERROR", correlated_targets, else_conditional_p);
        continue;
      }
      throw std::invalid_argument("Unsupported spread instruction type in virtual translation");
    }
  }
}

}  // namespace

stim::Circuit build_virtual_decoder_stim_circuit_object(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const SpreadProgram& spread_program,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round) {
  if (qec_rounds == 0) {
    throw std::invalid_argument("qec_rounds must be > 0 for virtual decoder circuit generation");
  }

  const CircuitBuildContext ctx = build_context(code);
  stim::Circuit circuit;
  std::vector<uint32_t> detector_lookbacks;
  detector_lookbacks.reserve(8);
  std::vector<uint32_t> detector_targets;
  detector_targets.reserve(8);

  auto pre_step_hook = [](std::size_t, std::size_t) {};
  auto post_step_hook = [&](std::size_t, std::size_t step) {
    append_virtual_timestep_spread_errors(&circuit, code, ctx, step, spread_program,
                                          two_qubit_erasure_probability,
                                          condition_on_erasure_in_round);
  };
  auto pre_measure_hook = [](std::size_t) {};

  for (std::size_t round = 0; round < qec_rounds; ++round) {
    append_extraction_round(&circuit, ctx, round, pre_step_hook, post_step_hook, pre_measure_hook,
                            &detector_lookbacks, &detector_targets);
  }

  append_final_readout_detectors_and_observable(&circuit, ctx, &detector_lookbacks, &detector_targets);
  return circuit;
}

std::string build_virtual_decoder_stim_circuit(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const SpreadProgram& spread_program,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round) {
  return build_virtual_decoder_stim_circuit_object(code, qec_rounds, spread_program,
                                                   two_qubit_erasure_probability,
                                                   condition_on_erasure_in_round).str();
}

}  // namespace qerasure
