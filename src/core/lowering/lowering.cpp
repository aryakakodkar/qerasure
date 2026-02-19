#include "qerasure/core/lowering/lowering.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../sim/internal/fast_rng.h"

namespace qerasure {

namespace {

// Build a default instruction program from legacy pair-based params.
// This preserves old construction paths while routing execution through one instruction model.
SpreadProgram make_legacy_program(const std::pair<LoweredErrorParams, LoweredErrorParams>& x_params,
                                  const std::pair<LoweredErrorParams, LoweredErrorParams>& z_params) {
  SpreadProgram program;
  program.add_error_channel(
      x_params.first.probability, {{x_params.first.error_type, PartnerSlot::X_1}});
  program.add_error_channel(
      x_params.second.probability, {{x_params.second.error_type, PartnerSlot::X_2}});
  program.add_error_channel(
      z_params.first.probability, {{z_params.first.error_type, PartnerSlot::Z_1}});
  program.add_error_channel(
      z_params.second.probability, {{z_params.second.error_type, PartnerSlot::Z_2}});
  return program;
}

}  // namespace

// Program-building helpers keep the public API concise and avoid manual instruction struct wiring.
void SpreadProgram::add_error_channel(double probability, std::vector<SpreadTargetOp> targets) {
  instructions.push_back({SpreadInstructionType::ERROR_CHANNEL, probability, std::move(targets)});
}

void SpreadProgram::add_correlated_error(double probability, std::vector<SpreadTargetOp> targets) {
  instructions.push_back({SpreadInstructionType::CORRELATED_ERROR, probability, std::move(targets)});
}

void SpreadProgram::add_else_correlated_error(double probability,
                                              std::vector<SpreadTargetOp> targets) {
  instructions.push_back(
      {SpreadInstructionType::ELSE_CORRELATED_ERROR, probability, std::move(targets)});
}

// Legacy constructors map old x/z pair params into instruction programs.
// This keeps backward compatibility while allowing the new Stim-like execution engine.
LoweringParams::LoweringParams(const SpreadProgram& default_program)
    : reset_params_({PauliError::NO_ERROR, 0.0}),
      x_ancilla_params_({{PauliError::NO_ERROR, 0.0}, {PauliError::NO_ERROR, 0.0}}),
      z_ancilla_params_({{PauliError::NO_ERROR, 0.0}, {PauliError::NO_ERROR, 0.0}}),
      default_data_program(default_program) {}

LoweringParams::LoweringParams(const SpreadProgram& default_program,
                               const LoweredErrorParams& reset)
    : reset_params_(reset),
      x_ancilla_params_({{PauliError::NO_ERROR, 0.0}, {PauliError::NO_ERROR, 0.0}}),
      z_ancilla_params_({{PauliError::NO_ERROR, 0.0}, {PauliError::NO_ERROR, 0.0}}),
      default_data_program(default_program) {}

LoweringParams::LoweringParams(const LoweredErrorParams& reset, const LoweredErrorParams& ancillas)
    : reset_params_(reset),
      x_ancilla_params_({ancillas, ancillas}),
      z_ancilla_params_({ancillas, ancillas}) {
  default_data_program = make_legacy_program(x_ancilla_params_, z_ancilla_params_);
}

LoweringParams::LoweringParams(const LoweredErrorParams& reset,
                               const LoweredErrorParams& x_ancillas,
                               const LoweredErrorParams& z_ancillas)
    : reset_params_(reset),
      x_ancilla_params_({x_ancillas, x_ancillas}),
      z_ancilla_params_({z_ancillas, z_ancillas}) {
  default_data_program = make_legacy_program(x_ancilla_params_, z_ancilla_params_);
}

LoweringParams::LoweringParams(
    const LoweredErrorParams& reset,
    const std::pair<LoweredErrorParams, LoweredErrorParams>& x_ancillas,
    const std::pair<LoweredErrorParams, LoweredErrorParams>& z_ancillas)
    : reset_params_(reset), x_ancilla_params_(x_ancillas), z_ancilla_params_(z_ancillas) {
  default_data_program = make_legacy_program(x_ancilla_params_, z_ancilla_params_);
}

void LoweringParams::set_default_data_program(const SpreadProgram& program) {
  default_data_program = program;
}

void LoweringParams::set_data_qubit_program(std::size_t data_qubit_idx, const SpreadProgram& program) {
  if (per_data_program_overrides.size() <= data_qubit_idx) {
    per_data_program_overrides.resize(data_qubit_idx + 1);
  }
  per_data_program_overrides[data_qubit_idx] = program;
}

// Construction compiles programs once so the hot loop only executes compact instructions.
Lowerer::Lowerer(const RotatedSurfaceCode& code, const LoweringParams& params)
    : code_(code), params_(params), rng_state_(0xA24BAED4963EE407ULL) {
  compile_programs();
}

std::uint8_t Lowerer::slot_to_index(PartnerSlot slot) {
  return static_cast<std::uint8_t>(slot);
}

// Compile high-level instructions into threshold-based, slot-resolved programs per data qubit.
// This front-loads work and minimizes branches/lookups during per-shot lowering.
void Lowerer::compile_programs() {
  const std::size_t num_data_qubits = code_.x_anc_offset();
  compiled_program_by_data_qubit_.assign(num_data_qubits, {});

  auto compile_program_for_data = [&](const SpreadProgram& src, std::size_t data_idx) {
    CompiledProgram out;
    out.instructions.reserve(src.instructions.size());
    for (const SpreadInstruction& instruction : src.instructions) {
      CompiledInstruction compiled;
      compiled.type = instruction.type;
      compiled.threshold = probability_to_threshold(instruction.probability);
      compiled.targets.reserve(instruction.targets.size());
      for (const SpreadTargetOp& target : instruction.targets) {
        compiled.targets.push_back(
            {target.error_type, resolve_data_slot_qubit(data_idx, slot_to_index(target.slot))});
      }
      out.instructions.push_back(std::move(compiled));
    }
    return out;
  };

  for (std::size_t data_idx = 0; data_idx < num_data_qubits; ++data_idx) {
    compiled_program_by_data_qubit_[data_idx] =
        compile_program_for_data(params_.default_data_program, data_idx);
  }
  for (std::size_t data_idx = 0;
       data_idx < params_.per_data_program_overrides.size() && data_idx < num_data_qubits;
       ++data_idx) {
    if (!params_.per_data_program_overrides[data_idx].instructions.empty()) {
      compiled_program_by_data_qubit_[data_idx] =
          compile_program_for_data(params_.per_data_program_overrides[data_idx], data_idx);
    }
  }
}

std::size_t Lowerer::resolve_data_slot_qubit(std::size_t data_qubit_idx, std::uint8_t slot_index) const {
  if (slot_index == 0) {
    return code_.data_to_x_ancilla_slots()[data_qubit_idx].first;
  }
  if (slot_index == 1) {
    return code_.data_to_x_ancilla_slots()[data_qubit_idx].second;
  }
  if (slot_index == 2) {
    return code_.data_to_z_ancilla_slots()[data_qubit_idx].first;
  }
  return code_.data_to_z_ancilla_slots()[data_qubit_idx].second;
}

// RNG and threshold sampling mirror simulator conventions for consistent fast Bernoulli draws.
std::uint64_t Lowerer::next_random_u64() {
  return internal::splitmix64_next(&rng_state_);
}

std::uint64_t Lowerer::probability_to_threshold(double p) {
  if (p <= 0.0) {
    return 0;
  }
  if (p >= 1.0) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(
      p * static_cast<long double>(std::numeric_limits<std::uint64_t>::max()));
}

bool Lowerer::sample_with_threshold(std::uint64_t threshold) {
  if (threshold == 0) {
    return false;
  }
  if (threshold == std::numeric_limits<std::uint64_t>::max()) {
    return true;
  }
  return next_random_u64() <= threshold;
}

// Lowering consumes sparse erasure events, maintains persistent erased state, and emits Pauli events.
// Programs are interpreted per erased data qubit at each gate timestep to realize Stim-like semantics.
LoweringResult Lowerer::lower(const ErasureSimResult& sim_result) {
  LoweringResult result;
  result.sparse_cliffords.resize(sim_result.sparse_erasures.size());
  result.clifford_timestep_offsets.resize(sim_result.erasure_timestep_offsets.size());

  const std::size_t num_qubits = code_.num_qubits();
  const std::size_t num_data_qubits = code_.x_anc_offset();
  const std::uint64_t reset_threshold = probability_to_threshold(params_.reset_params_.probability);

  // Reuse state buffers across shots to avoid repeated allocation/initialization.
  std::vector<std::uint8_t> erased_state(num_qubits, 0);
  std::vector<std::size_t> erased_pos(num_qubits, kNoPartner);
  std::vector<std::size_t> erased_qubits;
  erased_qubits.reserve(num_qubits / 4 + 1);

  for (std::size_t shot = 0; shot < sim_result.sparse_erasures.size(); ++shot) {
    std::size_t event_index = 0;
    std::size_t num_lowering_events = 0;

    const auto& events = sim_result.sparse_erasures[shot];
    const auto& offsets = sim_result.erasure_timestep_offsets[shot];
    auto& lowered_events = result.sparse_cliffords[shot];
    auto& lowered_offsets = result.clifford_timestep_offsets[shot];
    lowered_offsets.assign(offsets.size(), 0);
    lowered_events.clear();
    lowered_events.reserve(events.size() + events.size() / 2 + 8);

    erased_qubits.clear();

    for (std::size_t t = 0; t + 1 < offsets.size(); ++t) {
      const std::size_t end_index = offsets[t + 1];

      // First update erased/reset state from simulator events at this timestep.
      for (; event_index < end_index; ++event_index) {
        const EventType event_type = events[event_index].event_type;
        const std::size_t qubit_idx = events[event_index].qubit_idx;

        if (event_type == EventType::ERASURE) {
          if (erased_state[qubit_idx] == 0) {
            erased_state[qubit_idx] = 1;
            erased_pos[qubit_idx] = erased_qubits.size();
            erased_qubits.push_back(qubit_idx);
          }
        } else if (event_type == EventType::RESET) {
          if (erased_state[qubit_idx] != 0) {
            erased_state[qubit_idx] = 0;
            const std::size_t pos = erased_pos[qubit_idx];
            const std::size_t last = erased_qubits.back();
            erased_qubits[pos] = last;
            erased_pos[last] = pos;
            erased_qubits.pop_back();
            erased_pos[qubit_idx] = kNoPartner;
          }
          if (params_.reset_params_.error_type != PauliError::NO_ERROR &&
              sample_with_threshold(reset_threshold)) {
            lowered_events.push_back({qubit_idx, params_.reset_params_.error_type});
            ++num_lowering_events;
          }
        }
      }

      if (t < offsets.size() - 2) {
        // Stim-like lowering applies only for erased data qubits, since spread slots are data-relative.
        // We execute the compiled instruction chain per erased data qubit.
        const std::size_t step = t % 4;
        const std::size_t step_base = step * num_qubits;
        for (const std::size_t erased_qubit : erased_qubits) {
          if (erased_qubit >= num_data_qubits) {
            continue;
          }
          // Spread only to the ancilla actively interacting with this erased data qubit now.
          const std::size_t current_partner = code_.partner_map()[step_base + erased_qubit];
          if (current_partner == kNoPartner) {
            continue;
          }
          const CompiledProgram& program = compiled_program_by_data_qubit_[erased_qubit];
          if (program.instructions.empty()) {
            continue;
          }

          bool correlated_flag = false;
          for (const CompiledInstruction& instruction : program.instructions) {
            if (instruction.type == SpreadInstructionType::ELSE_CORRELATED_ERROR && correlated_flag) {
              continue;
            }

            // CORRELATED_ERROR/ELSE_CORRELATED_ERROR share one local flag per chain execution.
            const bool fires = sample_with_threshold(instruction.threshold);
            if (instruction.type == SpreadInstructionType::CORRELATED_ERROR ||
                instruction.type == SpreadInstructionType::ELSE_CORRELATED_ERROR) {
              correlated_flag = fires;
            }
            if (!fires) {
              continue;
            }

            for (const CompiledTargetOp& target : instruction.targets) {
              if (target.error_type == PauliError::NO_ERROR || target.qubit_idx == kNoPartner) {
                continue;
              }
              if (target.qubit_idx != current_partner) {
                continue;
              }
              lowered_events.push_back({target.qubit_idx, target.error_type});
              ++num_lowering_events;
            }
          }
        }
      }

      lowered_offsets[t + 1] = num_lowering_events;
    }

    for (const std::size_t q : erased_qubits) {
      erased_state[q] = 0;
      erased_pos[q] = kNoPartner;
    }
  }

  return result;
}

}  // namespace qerasure
