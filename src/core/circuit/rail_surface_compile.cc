#include "core/circuit/rail_surface_compile.h"

#include <cstddef>
#include <stdexcept>

#include "core/circuit/instruction.h"

namespace qerasure::circuit {

namespace {

bool is_round_mr(
    const Instruction& instr,
    uint32_t x_anc_offset,
    uint32_t z_anc_offset,
    uint32_t num_qubits) {
  if (instr.op != OpCode::MR) {
    return false;
  }
  if (instr.targets.size() != num_qubits - x_anc_offset) {
    return false;
  }
  if (instr.targets.empty()) {
    return false;
  }
  if (instr.targets.front() != x_anc_offset || instr.targets.back() != num_qubits - 1) {
    return false;
  }
  if (z_anc_offset <= x_anc_offset) {
    return false;
  }
  return true;
}

}  // namespace

RailSurfaceCompiledProgram::RailSurfaceCompiledProgram(
    const ErasureCircuit& circuit,
    const ErasureModel& model,
    uint32_t distance,
    uint32_t rounds)
    : base_program_(circuit, model),
      code_(distance),
      distance_(distance),
      rounds_(rounds),
      num_data_qubits_(static_cast<uint32_t>(code_.x_anc_offset())),
      x_anc_offset_(static_cast<uint32_t>(code_.x_anc_offset())),
      z_anc_offset_(static_cast<uint32_t>(code_.z_anc_offset())),
      num_z_ancillas_(static_cast<uint32_t>(code_.num_qubits() - code_.z_anc_offset())),
      num_detectors_(0) {
  if (rounds_ == 0) {
    throw std::invalid_argument("RailSurfaceCompiledProgram requires rounds > 0");
  }
  if (model.max_persistence != 2) {
    throw std::invalid_argument(
        "RailSurfaceCompiledProgram currently supports only max_persistence=2");
  }
  if (base_program_.max_qubit_index() + 1 != code_.num_qubits()) {
    throw std::invalid_argument(
        "RailSurfaceCompiledProgram qubit-count mismatch between circuit and distance");
  }

  op_index_to_round_.assign(base_program_.operation_groups.size(), -1);
  check_event_to_qubit_.reserve(base_program_.num_checks());
  check_event_to_op_index_.reserve(base_program_.num_checks());
  round_z_detector_index_.assign(
      static_cast<size_t>(rounds_) * num_z_ancillas_, -1);
  data_round_z_interaction_op_.assign(
      static_cast<size_t>(num_data_qubits_) * rounds_ * num_z_ancillas_, -1);
  data_to_z_slots_.reserve(num_data_qubits_);
  data_schedule_type_.assign(num_data_qubits_, 0);

  for (uint32_t q = 0; q < num_data_qubits_; ++q) {
    const auto& slots = code_.data_to_z_ancilla_slots().at(q);
    const int32_t slot0 = slots.first == kNoPartner ? -1 : static_cast<int32_t>(slots.first);
    const int32_t slot1 = slots.second == kNoPartner ? -1 : static_cast<int32_t>(slots.second);
    data_to_z_slots_.push_back({slot0, slot1});
  }

  uint32_t current_round = 0;
  uint32_t detector_index = 0;
  int32_t detector_round = -1;
  uint32_t detector_round_cursor = 0;
  for (uint32_t op_index = 0; op_index < base_program_.operation_groups.size(); ++op_index) {
    op_index_to_round_[op_index] = static_cast<int32_t>(current_round);
    const OperationGroup& group = base_program_.operation_groups[op_index];
    if (!group.stim_instruction.has_value()) {
      for (const auto& check : group.checks) {
        check_event_to_qubit_.push_back(check.qubit_index);
        check_event_to_op_index_.push_back(op_index);
      }
      continue;
    }

    const Instruction& instr = *group.stim_instruction;
    if (instr.op == OpCode::CX) {
      for (size_t k = 0; k + 1 < instr.targets.size(); k += 2) {
        const uint32_t control = instr.targets[k];
        const uint32_t target = instr.targets[k + 1];
        if (current_round == 0 && target < num_data_qubits_ && control >= x_anc_offset_ &&
            control < z_anc_offset_ && data_schedule_type_[target] == 0) {
          data_schedule_type_[target] = 1;
        }
        if (current_round == 0 && control < num_data_qubits_ && target >= z_anc_offset_ &&
            data_schedule_type_[control] == 0) {
          data_schedule_type_[control] = 2;
        }
        if (control >= num_data_qubits_) {
          continue;
        }
        if (target < z_anc_offset_) {
          continue;
        }
        if (current_round >= rounds_) {
          continue;
        }
        const uint32_t z_local = target - z_anc_offset_;
        const size_t key = interaction_index_(control, current_round, z_local);
        if (data_round_z_interaction_op_[key] < 0) {
          data_round_z_interaction_op_[key] = static_cast<int32_t>(op_index);
        }
      }
    } else if (instr.op == OpCode::DETECTOR) {
      if (detector_round >= 0 && detector_round_cursor < num_z_ancillas_) {
        const size_t idx = static_cast<size_t>(detector_round) * num_z_ancillas_ + detector_round_cursor;
        round_z_detector_index_[idx] = static_cast<int32_t>(detector_index);
        detector_round_cursor++;
      }
      detector_index++;
    } else if (is_round_mr(
                   instr,
                   x_anc_offset_,
                   z_anc_offset_,
                   static_cast<uint32_t>(code_.num_qubits()))) {
      detector_round = static_cast<int32_t>(current_round);
      detector_round_cursor = 0;
      current_round++;
    } else if (detector_round >= 0) {
      detector_round = -1;
      detector_round_cursor = 0;
    }
  }

  num_detectors_ = detector_index;
  if (check_event_to_qubit_.size() != base_program_.num_checks()) {
    throw std::logic_error("RailSurfaceCompiledProgram check-event mapping mismatch");
  }
}

int32_t RailSurfaceCompiledProgram::data_qubit_schedule_type(uint32_t data_qubit) const {
  if (data_qubit >= data_schedule_type_.size()) {
    return 0;
  }
  return static_cast<int32_t>(data_schedule_type_[data_qubit]);
}

bool RailSurfaceCompiledProgram::data_qubit_is_boundary(uint32_t data_qubit) const {
  if (data_qubit >= data_to_z_slots_.size()) {
    return false;
  }
  const auto slots = data_to_z_slots_[data_qubit];
  const bool has0 = slots.first >= 0;
  const bool has1 = slots.second >= 0;
  return has0 != has1;
}

std::pair<int32_t, int32_t> RailSurfaceCompiledProgram::data_z_ancilla_slots(
    uint32_t data_qubit) const {
  if (data_qubit >= data_to_z_slots_.size()) {
    return {-1, -1};
  }
  return data_to_z_slots_[data_qubit];
}

int32_t RailSurfaceCompiledProgram::round_detector_index(
    uint32_t round_index,
    uint32_t z_ancilla) const {
  if (round_index >= rounds_) {
    return -1;
  }
  if (z_ancilla < z_anc_offset_) {
    return -1;
  }
  const uint32_t z_local = z_ancilla - z_anc_offset_;
  if (z_local >= num_z_ancillas_) {
    return -1;
  }
  return round_z_detector_index_[static_cast<size_t>(round_index) * num_z_ancillas_ + z_local];
}

int32_t RailSurfaceCompiledProgram::interaction_op_for_data_z_ancilla(
    uint32_t data_qubit,
    uint32_t z_ancilla,
    uint32_t round_index) const {
  if (data_qubit >= num_data_qubits_ || round_index >= rounds_ || z_ancilla < z_anc_offset_) {
    return -1;
  }
  const uint32_t z_local = z_ancilla - z_anc_offset_;
  if (z_local >= num_z_ancillas_) {
    return -1;
  }
  return data_round_z_interaction_op_[interaction_index_(data_qubit, round_index, z_local)];
}

size_t RailSurfaceCompiledProgram::interaction_index_(
    uint32_t data_qubit,
    uint32_t round_index,
    uint32_t z_local) const {
  return (static_cast<size_t>(data_qubit) * rounds_ + round_index) * num_z_ancillas_ + z_local;
}

}  // namespace qerasure::circuit
