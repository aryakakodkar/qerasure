#include "qerasure/core/translation/stim_translation.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "qerasure/core/lowering/lowering.h"
#include "common.h"
#include "stim/circuit/circuit.h"
#include "stim/circuit/gate_target.h"

namespace qerasure {

namespace {

using translation_internal::append_extraction_round;
using translation_internal::append_final_readout_detectors_and_observable;
using translation_internal::build_context;
using translation_internal::CircuitBuildContext;

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
