#include "qerasure/core/lowering/lowering.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../sim/internal/fast_rng.h"

namespace qerasure {

namespace {

double clamp_probability(double p) {
  if (p <= 0.0) {
    return 0.0;
  }
  if (p >= 1.0) {
    return 1.0;
  }
  return p;
}

bool is_cond_type(SpreadInstructionType type) {
  return type == SpreadInstructionType::COND_X_ERROR || type == SpreadInstructionType::COND_Y_ERROR ||
         type == SpreadInstructionType::COND_Z_ERROR;
}

bool is_else_type(SpreadInstructionType type) {
  return type == SpreadInstructionType::ELSE_X_ERROR || type == SpreadInstructionType::ELSE_Y_ERROR ||
         type == SpreadInstructionType::ELSE_Z_ERROR;
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

// Can't see where this might be useful
SpreadInstructionType simple_type_from_pauli(PauliError error_type) {
  switch (error_type) {
    case PauliError::X_ERROR:
      return SpreadInstructionType::X_ERROR;
    case PauliError::Y_ERROR:
      return SpreadInstructionType::Y_ERROR;
    case PauliError::Z_ERROR:
      return SpreadInstructionType::Z_ERROR;
    case PauliError::DEPOLARIZE:
      return SpreadInstructionType::DEPOLARIZE1;
    case PauliError::NO_ERROR:
      break;
  }
  throw std::invalid_argument("Cannot map NO_ERROR to a simple spread instruction");
}

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

// TODO_ARYA: Needs to be extended to support higher-weight checks in other qLDPC codes
PartnerSlot parse_partner_slot(const std::string& slot_text) {
  if (slot_text == "X_1") {
    return PartnerSlot::X_1;
  }
  if (slot_text == "X_2") {
    return PartnerSlot::X_2;
  }
  if (slot_text == "Z_1") {
    return PartnerSlot::Z_1;
  }
  if (slot_text == "Z_2") {
    return PartnerSlot::Z_2;
  }
  throw std::invalid_argument("Unknown partner slot token: " + slot_text);
}

SpreadInstructionType parse_instruction_name(const std::string& name) {
  if (name == "X_ERROR") {
    return SpreadInstructionType::X_ERROR;
  }
  if (name == "Y_ERROR") {
    return SpreadInstructionType::Y_ERROR;
  }
  if (name == "Z_ERROR") {
    return SpreadInstructionType::Z_ERROR;
  }
  if (name == "DEPOLARIZE1") { 
    return SpreadInstructionType::DEPOLARIZE1;
  }
  if (name == "COND_X_ERROR") {
    return SpreadInstructionType::COND_X_ERROR;
  }
  if (name == "COND_Y_ERROR") {
    return SpreadInstructionType::COND_Y_ERROR;
  }
  if (name == "COND_Z_ERROR") {
    return SpreadInstructionType::COND_Z_ERROR;
  }
  if (name == "ELSE_X_ERROR") {
    return SpreadInstructionType::ELSE_X_ERROR;
  }
  if (name == "ELSE_Y_ERROR") {
    return SpreadInstructionType::ELSE_Y_ERROR;
  }
  if (name == "ELSE_Z_ERROR") {
    return SpreadInstructionType::ELSE_Z_ERROR;
  }
  throw std::invalid_argument("Unknown spread instruction name: " + name);
}

// Takes a string-like spread instruction and turns it into a SpreadInstruction struct
SpreadInstruction parse_spread_instruction_str(const std::string& instruction_text) {
  const std::string text = trim(instruction_text);
  if (text.empty()) {
    throw std::invalid_argument("Empty spread instruction");
  }
  const std::size_t lparen = text.find('(');
  const std::size_t rparen = text.find(')');
  if (lparen == std::string::npos || rparen == std::string::npos || rparen <= lparen) {
    throw std::invalid_argument("Instruction must contain '(p)' argument: " + text);
  }
  const std::string op_name = trim(text.substr(0, lparen));
  const SpreadInstructionType type = parse_instruction_name(op_name);
  const double p = clamp_probability(std::stod(trim(text.substr(lparen + 1, rparen - lparen - 1))));

  SpreadInstruction instruction;
  instruction.type = type;
  instruction.probability = p;
  std::istringstream slot_stream(trim(text.substr(rparen + 1)));
  std::string slot_token;
  if (!(slot_stream >> slot_token)) {
    throw std::invalid_argument("Spread instruction must specify a target slot: " + text);
  }
  instruction.target_slot = parse_partner_slot(slot_token);
  if (slot_stream >> slot_token) {
    throw std::invalid_argument(
        "Each spread instruction must have at most one target. Use ';' to separate instructions.");
  }
  return instruction;
}

// Build a default instruction program from legacy pair-based params.
// This preserves old construction paths while routing execution through one instruction model.
SpreadProgram make_legacy_program(const std::pair<LoweredErrorParams, LoweredErrorParams>& x_params,
                                  const std::pair<LoweredErrorParams, LoweredErrorParams>& z_params) {
  SpreadProgram program;
  if (x_params.first.error_type != PauliError::NO_ERROR) {
    program.add_instruction(simple_type_from_pauli(x_params.first.error_type), x_params.first.probability,
                            PartnerSlot::X_1);
  }
  if (x_params.second.error_type != PauliError::NO_ERROR) {
    program.add_instruction(simple_type_from_pauli(x_params.second.error_type), x_params.second.probability,
                            PartnerSlot::X_2);
  }
  if (z_params.first.error_type != PauliError::NO_ERROR) {
    program.add_instruction(simple_type_from_pauli(z_params.first.error_type), z_params.first.probability,
                            PartnerSlot::Z_1);
  }
  if (z_params.second.error_type != PauliError::NO_ERROR) {
    program.add_instruction(simple_type_from_pauli(z_params.second.error_type), z_params.second.probability,
                            PartnerSlot::Z_2);
  }
  return program;
}

}  // namespace

// Allow for addition of instructions in a string-like manner, similar to Stim
void SpreadProgram::append(const std::string& instruction_strs) {
  std::size_t start = 0;
  while (start <= instruction_strs.size()) {
    std::size_t end = instruction_strs.find(';', start);
    if (end == std::string::npos) {
      end = instruction_strs.size();
    }
    const std::string piece = trim(instruction_strs.substr(start, end - start));
    if (!piece.empty()) {
      instructions.push_back(parse_spread_instruction_str(piece));
    }
    if (end == instruction_strs.size()) {
      break;
    }
    start = end + 1;
  }
}

void SpreadProgram::add_instruction(SpreadInstructionType type, double probability, PartnerSlot target) {
  SpreadInstruction instruction;
  instruction.type = type;
  instruction.probability = clamp_probability(probability);
  instruction.target_slot = target;
  instructions.push_back(instruction);
}

// Strongly-type helpers, not good for scalability
void SpreadProgram::add_x_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::X_ERROR, probability, target);
}
void SpreadProgram::add_y_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::Y_ERROR, probability, target);
}
void SpreadProgram::add_z_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::Z_ERROR, probability, target);
}
void SpreadProgram::add_depolarize1(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::DEPOLARIZE1, probability, target);
}
void SpreadProgram::add_cond_x_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::COND_X_ERROR, probability, target);
}
void SpreadProgram::add_cond_y_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::COND_Y_ERROR, probability, target);
}
void SpreadProgram::add_cond_z_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::COND_Z_ERROR, probability, target);
}
void SpreadProgram::add_else_x_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::ELSE_X_ERROR, probability, target);
}
void SpreadProgram::add_else_y_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::ELSE_Y_ERROR, probability, target);
}
void SpreadProgram::add_else_z_error(double probability, PartnerSlot target) {
  add_instruction(SpreadInstructionType::ELSE_Z_ERROR, probability, target);
}

// Backward-compatible wrappers. Should not be used in new code.
void SpreadProgram::add_error_channel(double probability, std::vector<SpreadTargetOp> targets) {
  for (const SpreadTargetOp& target : targets) {
    switch (target.error_type) {
      case PauliError::X_ERROR:
        add_instruction(SpreadInstructionType::X_ERROR, probability, target.slot);
        break;
      case PauliError::Y_ERROR:
        add_instruction(SpreadInstructionType::Y_ERROR, probability, target.slot);
        break;
      case PauliError::Z_ERROR:
        add_instruction(SpreadInstructionType::Z_ERROR, probability, target.slot);
        break;
      case PauliError::DEPOLARIZE:
        add_instruction(SpreadInstructionType::DEPOLARIZE1, probability, target.slot);
        break;
      case PauliError::NO_ERROR:
        break;
    }
  }
}

void SpreadProgram::add_correlated_error(double probability, std::vector<SpreadTargetOp> targets) {
  for (const SpreadTargetOp& target : targets) {
    switch (target.error_type) {
      case PauliError::X_ERROR:
        add_instruction(SpreadInstructionType::COND_X_ERROR, probability, target.slot);
        break;
      case PauliError::Y_ERROR:
        add_instruction(SpreadInstructionType::COND_Y_ERROR, probability, target.slot);
        break;
      case PauliError::Z_ERROR:
        add_instruction(SpreadInstructionType::COND_Z_ERROR, probability, target.slot);
        break;
      case PauliError::NO_ERROR:
      case PauliError::DEPOLARIZE:
        break;
    }
  }
}

void SpreadProgram::add_else_correlated_error(double probability,
                                              std::vector<SpreadTargetOp> targets) {
  for (const SpreadTargetOp& target : targets) {
    switch (target.error_type) {
      case PauliError::X_ERROR:
        add_instruction(SpreadInstructionType::ELSE_X_ERROR, probability, target.slot);
        break;
      case PauliError::Y_ERROR:
        add_instruction(SpreadInstructionType::ELSE_Y_ERROR, probability, target.slot);
        break;
      case PauliError::Z_ERROR:
        add_instruction(SpreadInstructionType::ELSE_Z_ERROR, probability, target.slot);
        break;
      case PauliError::NO_ERROR:
      case PauliError::DEPOLARIZE:
        break;
    }
  }
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
      const PauliError instruction_error = pauli_from_instruction_type(instruction.type);
      compiled.target =
          {instruction_error, resolve_data_slot_qubit(data_idx, slot_to_index(instruction.target_slot))};
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
  result.qec_rounds = sim_result.qec_rounds;
  result.sparse_cliffords.resize(sim_result.sparse_erasures.size());
  result.clifford_timestep_offsets.resize(sim_result.erasure_timestep_offsets.size());
  result.check_error_round_flags.resize(sim_result.sparse_erasures.size());
  result.erasure_round_flags.resize(sim_result.sparse_erasures.size());
  result.reset_round_qubits.resize(sim_result.sparse_erasures.size());

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
    auto& check_flags = result.check_error_round_flags[shot];
    auto& erasure_flags = result.erasure_round_flags[shot];
    auto& reset_round_qubits = result.reset_round_qubits[shot];
    check_flags.assign(result.qec_rounds, 0);
    erasure_flags.assign(result.qec_rounds, 0);
    reset_round_qubits.clear();
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
          if (t > 0 && (t % 4) == 0) {
            const std::size_t prior_round = t / 4 - 1;
            if (prior_round < erasure_flags.size()) {
              erasure_flags[prior_round] = 1;
              reset_round_qubits.push_back({prior_round, qubit_idx});
            }
          }
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
            lowered_events.push_back(
                {qubit_idx, params_.reset_params_.error_type, LoweredEventOrigin::RESET});
            ++num_lowering_events;
          }
        } else if (event_type == EventType::CHECK_ERROR) {
          // CHECK_ERROR at timestep (r+1)*4 implies erasure occurred in round r.
          if (t > 0 && (t % 4) == 0) {
            const std::size_t prior_round = t / 4 - 1;
            if (prior_round < check_flags.size()) {
              check_flags[prior_round] = 1;
              erasure_flags[prior_round] = 1;
            }
          }
        }
      }

      if (t < offsets.size() - 2) {
        // Lowering of erasures to Pauli errors applies only for erased data qubits (to be extended to ancillas in the future).
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

          bool conditional_flag = false;
          for (const CompiledInstruction& instruction : program.instructions) {
            if (is_else_type(instruction.type) && conditional_flag) {
              continue;
            }

            // Instruction is all-or-nothing: all targets must be on the active partner.
            bool feasible = !(instruction.target.qubit_idx == kNoPartner ||
                              instruction.target.qubit_idx != current_partner);

            if (!feasible) {
              if (is_cond_type(instruction.type)) {
                conditional_flag = false; // un-flag the conditional
              }
              continue;
            }

            const bool fires = sample_with_threshold(instruction.threshold);
            if (is_cond_type(instruction.type)) {
              conditional_flag = fires;
            } else if (is_else_type(instruction.type) && fires) {
              conditional_flag = true;
            }
            if (!fires) {
              continue;
            }

            const CompiledTargetOp& target = instruction.target;
            if (target.error_type == PauliError::NO_ERROR || target.qubit_idx == kNoPartner ||
                target.qubit_idx != current_partner) {
              continue;
            }
            lowered_events.push_back({target.qubit_idx, target.error_type, LoweredEventOrigin::SPREAD});
            ++num_lowering_events;
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
