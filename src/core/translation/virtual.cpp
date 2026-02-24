#include "qerasure/core/translation/stim_translation.h"

#include <array>
#include <cstdint>
#include <algorithm>
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
  double survival = 1.0; // Probability that the qubit has not been erased

  for (std::size_t step = 0; step < 4; ++step) {
    if (!active_by_step[step]) {
      continue;
    }
    out[step] = survival * p;
    survival *= (1.0 - p);
  }

  if (condition_on_erasure_in_round) {
    const double p_any = 1.0 - survival; // Probability of erasure occurring somewhere in SE round
    if (p_any <= 0.0) {
      return {0.0, 0.0, 0.0, 0.0};
    }
    for (std::size_t step = 0; step < 4; ++step) {
      out[step] /= p_any;
    }
  }
  return out;
}

std::size_t find_ancilla_index(const RotatedSurfaceCode& code, std::size_t data_qubit_idx,
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

// Stim-readable error target encoding
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

// TODO_ARYA: This mapping is very naive. Will need to find a better architecture.
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
                                           const LoweringParams& lowering_params,
                                           const std::vector<std::size_t>& erased_data_qubits,
                                           double two_qubit_erasure_probability,
                                           bool condition_on_erasure_in_round) {
  auto target_step_for = [&](std::size_t data_q, std::size_t target_qubit) -> std::size_t {
    if (target_qubit == kNoPartner) {
      return kNoPartner;
    }
    for (std::size_t s = 0; s < 4; ++s) {
      if (code.partner_map()[s * ctx.num_qubits + data_q] == target_qubit) {
        return s;
      }
    }
    return kNoPartner;
  };

  std::vector<uint32_t> correlated_targets;
  correlated_targets.reserve(1);
  std::vector<uint32_t> single_target(1);

  for (const std::size_t data_q : erased_data_qubits) {
    if (data_q >= ctx.num_data) {
      continue;
    }
    const std::array<bool, 4> active_by_step = {
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

    const SpreadProgram& program =
        (data_q < lowering_params.per_data_program_overrides.size() &&
         !lowering_params.per_data_program_overrides[data_q].instructions.empty())
            ? lowering_params.per_data_program_overrides[data_q]
            : lowering_params.default_data_program;

    for (std::size_t i = 0; i < program.instructions.size(); ++i) {
      const SpreadInstruction& instruction = program.instructions[i];
      const double instruction_p = clamp_probability(instruction.probability);
      const std::size_t target_qubit = find_ancilla_index(code, data_q, instruction.target_slot);
      const std::size_t target_step = target_step_for(data_q, target_qubit);
      const PauliError error_type = pauli_from_instruction_type(instruction.type);

      if (is_simple_type(instruction.type)) {
        if (target_step == kNoPartner || target_step != step) {
          continue;
        }
        const double p_apply = clamp_probability(erased_by_step[step] * instruction_p);
        if (p_apply <= 0.0) {
          continue;
        }
        single_target[0] = static_cast<uint32_t>(target_qubit);
        if (error_type == PauliError::X_ERROR) {
          circuit->safe_append_ua("X_ERROR", single_target, p_apply);
        } else if (error_type == PauliError::Y_ERROR) {
          circuit->safe_append_ua("Y_ERROR", single_target, p_apply);
        } else if (error_type == PauliError::Z_ERROR) {
          circuit->safe_append_ua("Z_ERROR", single_target, p_apply);
        } else if (error_type == PauliError::DEPOLARIZE) {
          circuit->safe_append_ua("DEPOLARIZE1", single_target, p_apply);
        }
        continue;
      }

      if (is_cond_type(instruction.type)) {
        // Find the full contiguous COND/ELSE chain.
        std::size_t chain_end = i + 1;
        while (chain_end < program.instructions.size() &&
               is_else_type(program.instructions[chain_end].type)) {
          ++chain_end;
        }

        // Emit the entire chain once, at the latest target gate-step among chain instructions.
        std::size_t emit_step = kNoPartner;
        for (std::size_t k = i; k < chain_end; ++k) {
          const SpreadInstruction& chain_instruction = program.instructions[k];
          const std::size_t chain_target = find_ancilla_index(code, data_q, chain_instruction.target_slot);
          const std::size_t chain_target_step = target_step_for(data_q, chain_target);
          if (chain_target_step != kNoPartner) {
            emit_step = emit_step == kNoPartner ? chain_target_step : std::max(emit_step, chain_target_step);
          }
        }
        if (emit_step == kNoPartner || emit_step != step || current_partner == kNoPartner) {
          i = chain_end - 1;
          continue;
        }

        bool emitted_correlated = false;
        double residual = 1.0;
        for (std::size_t k = i; k < chain_end; ++k) {
          const SpreadInstruction& chain_instruction = program.instructions[k];
          const double chain_instruction_p = clamp_probability(chain_instruction.probability);
          const std::size_t chain_target = find_ancilla_index(code, data_q, chain_instruction.target_slot);
          const std::size_t chain_target_step = target_step_for(data_q, chain_target);
          if (chain_target == kNoPartner || chain_target_step == kNoPartner || chain_target_step > emit_step) {
            continue;
          }
          const double chain_marginal_p = clamp_probability(erased_by_step[emit_step] * chain_instruction_p);
          if (chain_marginal_p <= 0.0) {
            continue;
          }
          const PauliError chain_error = pauli_from_instruction_type(chain_instruction.type);
          correlated_targets.clear();
          correlated_targets.push_back(pauli_target_u32(chain_error, chain_target));
          single_target[0] = static_cast<uint32_t>(chain_target);

          if (is_cond_type(chain_instruction.type)) {
            // Virtual decoder semantics use marginal, erasure-time-weighted probabilities
            // for each instruction in the chain.
            const double cond_p = clamp_probability(chain_marginal_p);
            if (cond_p > 0.0) {
              circuit->safe_append_ua("CORRELATED_ERROR", correlated_targets, cond_p);
              emitted_correlated = true;
              residual *= (1.0 - cond_p);
            }
            continue;
          }

          if (is_else_type(chain_instruction.type)) {
            if (emitted_correlated && residual > 0.0) {
              const double else_cond_p = clamp_probability(chain_marginal_p);
              if (else_cond_p > 0.0) {
                circuit->safe_append_ua("ELSE_CORRELATED_ERROR", correlated_targets, else_cond_p);
                residual *= (1.0 - else_cond_p);
              }
            } else {
              // If no correlated prefix could be emitted, degrade ELSE branch to independent channel.
              const double standalone_p = clamp_probability(chain_marginal_p * residual);
              if (standalone_p <= 0.0) {
                continue;
              }
              if (chain_error == PauliError::X_ERROR) {
                circuit->safe_append_ua("X_ERROR", single_target, standalone_p);
              } else if (chain_error == PauliError::Y_ERROR) {
                circuit->safe_append_ua("Y_ERROR", single_target, standalone_p);
              } else if (chain_error == PauliError::Z_ERROR) {
                circuit->safe_append_ua("Z_ERROR", single_target, standalone_p);
              }
            }
          }
        }

        i = chain_end - 1;
        continue;
      }

      if (is_else_type(instruction.type)) {
        // Orphan ELSE instructions are lowered as independent channels.
        if (target_step == kNoPartner || target_step != step) {
          continue;
        }
        const double p_apply = clamp_probability(erased_by_step[step] * instruction_p);
        if (p_apply <= 0.0) {
          continue;
        }
        single_target[0] = static_cast<uint32_t>(target_qubit);
        if (error_type == PauliError::X_ERROR) {
          circuit->safe_append_ua("X_ERROR", single_target, p_apply);
        } else if (error_type == PauliError::Y_ERROR) {
          circuit->safe_append_ua("Y_ERROR", single_target, p_apply);
        } else if (error_type == PauliError::Z_ERROR) {
          circuit->safe_append_ua("Z_ERROR", single_target, p_apply);
        }
        continue;
      }

      throw std::invalid_argument("Unsupported spread instruction type in virtual translation");
    }
  }
}

void append_virtual_round_reset_errors(stim::Circuit* circuit,
                                       const LoweringParams& lowering_params,
                                       const std::vector<std::size_t>& reset_qubits_for_round) {
  const PauliError reset_error = lowering_params.reset_params_.error_type;
  double p = clamp_probability(lowering_params.reset_params_.probability);
  if (reset_error == PauliError::NO_ERROR || p <= 0.0 || reset_qubits_for_round.empty()) {
    return;
  }

  std::vector<uint32_t> targets;
  targets.reserve(reset_qubits_for_round.size());
  for (const std::size_t q : reset_qubits_for_round) {
    targets.push_back(static_cast<uint32_t>(q));
  }

  if (reset_error == PauliError::X_ERROR) {
    circuit->safe_append_ua("X_ERROR", targets, p);
  } else if (reset_error == PauliError::Y_ERROR) {
    circuit->safe_append_ua("Y_ERROR", targets, p);
  } else if (reset_error == PauliError::Z_ERROR) {
    circuit->safe_append_ua("Z_ERROR", targets, p);
  } else if (reset_error == PauliError::DEPOLARIZE) {
    // Match logical translation convention: deterministic depolarizing events are represented
    // as DEPOLARIZE1(0.75), so scale configured probability accordingly.
    p = clamp_probability(0.75 * p);
    circuit->safe_append_ua("DEPOLARIZE1", targets, p);
  }
}

}  // namespace

stim::Circuit build_virtual_decoder_stim_circuit_object(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round) {
  if (qec_rounds == 0) {
    throw std::invalid_argument("qec_rounds must be > 0 for virtual decoder circuit generation");
  }
  const std::vector<std::uint8_t>* erasure_flags_ptr = nullptr;
  if (shot_index < lowering_result.erasure_round_flags.size() &&
      lowering_result.erasure_round_flags[shot_index].size() >= qec_rounds) {
    erasure_flags_ptr = &lowering_result.erasure_round_flags[shot_index];
  } else if (shot_index < lowering_result.check_error_round_flags.size() &&
             lowering_result.check_error_round_flags[shot_index].size() >= qec_rounds) {
    // Backward-compatible fallback for older results.
    erasure_flags_ptr = &lowering_result.check_error_round_flags[shot_index];
  } else {
    throw std::invalid_argument("LoweringResult erasure flags do not match requested shot/qec_rounds");
  }
  const std::vector<std::uint8_t>& erasure_flags = *erasure_flags_ptr;
  std::vector<std::vector<std::size_t>> erased_data_by_round(qec_rounds);
  std::vector<std::vector<std::size_t>> reset_qubits_by_round(qec_rounds);
  bool has_reset_evidence_for_shot = false;
  if (shot_index < lowering_result.reset_round_qubits.size()) {
    const auto& reset_pairs = lowering_result.reset_round_qubits[shot_index];
    has_reset_evidence_for_shot = !reset_pairs.empty();
    for (const auto& [round_idx, qubit_idx] : reset_pairs) {
      if (round_idx < qec_rounds) {
        reset_qubits_by_round[round_idx].push_back(qubit_idx);
        if (qubit_idx < code.x_anc_offset()) {
          erased_data_by_round[round_idx].push_back(qubit_idx);
        }
      }
    }
  }
  for (std::size_t round = 0; round < qec_rounds; ++round) {
    auto& qubits = erased_data_by_round[round];
    std::sort(qubits.begin(), qubits.end());
    qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
    auto& reset_qubits = reset_qubits_by_round[round];
    std::sort(reset_qubits.begin(), reset_qubits.end());
    reset_qubits.erase(std::unique(reset_qubits.begin(), reset_qubits.end()), reset_qubits.end());
  }

  const CircuitBuildContext ctx = build_context(code);
  std::vector<std::size_t> all_data_qubits;
  all_data_qubits.reserve(ctx.num_data);
  for (std::size_t q = 0; q < ctx.num_data; ++q) {
    all_data_qubits.push_back(q);
  }
  stim::Circuit circuit;
  std::vector<uint32_t> detector_lookbacks;
  detector_lookbacks.reserve(8);
  std::vector<uint32_t> detector_targets;
  detector_targets.reserve(8);

  auto pre_step_hook = [](std::size_t, std::size_t) {};
  auto post_step_hook = [&](std::size_t round, std::size_t step) {
    // Inject only in rounds proven to contain erasures by reset/check evidence.
    if (erasure_flags[round] == 0) {
      return;
    }
    // Virtual injection is qubit-specific: only data qubits with reset evidence in this
    // round contribute spread instructions.
    const std::vector<std::size_t>& round_erased_data =
        erased_data_by_round[round].empty() && !has_reset_evidence_for_shot
            ? all_data_qubits
            : erased_data_by_round[round];
    if (round_erased_data.empty()) {
      return;
    }
    append_virtual_timestep_spread_errors(&circuit, code, ctx, step, lowering_params, round_erased_data,
                                          two_qubit_erasure_probability, condition_on_erasure_in_round);
  };
  auto pre_measure_hook = [&](std::size_t round) {
    append_virtual_round_reset_errors(&circuit, lowering_params, reset_qubits_by_round[round]);
  };

  for (std::size_t round = 0; round < qec_rounds; ++round) {
    append_extraction_round(&circuit, ctx, round, pre_step_hook, post_step_hook, pre_measure_hook,
                            &detector_lookbacks, &detector_targets);
  }

  append_final_readout_detectors_and_observable(&circuit, ctx, &detector_lookbacks, &detector_targets);
  return circuit;
}

std::string build_virtual_decoder_stim_circuit(
    const RotatedSurfaceCode& code, std::size_t qec_rounds, const LoweringParams& lowering_params,
    const LoweringResult& lowering_result, std::size_t shot_index,
    double two_qubit_erasure_probability, bool condition_on_erasure_in_round) {
  return build_virtual_decoder_stim_circuit_object(code, qec_rounds, lowering_params,
                                                   lowering_result, shot_index,
                                                   two_qubit_erasure_probability,
                                                   condition_on_erasure_in_round).str();
}

}  // namespace qerasure
