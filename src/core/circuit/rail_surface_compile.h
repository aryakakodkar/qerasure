#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/circuit/circuit.h"
#include "core/circuit/compile.h"
#include "core/code/rotated_surface_code.h"

namespace qerasure::circuit {

class RailSurfaceCompiledProgram {
 public:
  RailSurfaceCompiledProgram(
      const ErasureCircuit& circuit,
      const ErasureModel& model,
      uint32_t distance,
      uint32_t rounds);

  const CompiledErasureProgram& base_program() const {
    return base_program_;
  }

  uint32_t distance() const {
    return distance_;
  }

  uint32_t rounds() const {
    return rounds_;
  }

  uint32_t num_data_qubits() const {
    return num_data_qubits_;
  }

  uint32_t x_anc_offset() const {
    return x_anc_offset_;
  }

  uint32_t z_anc_offset() const {
    return z_anc_offset_;
  }

  uint32_t num_z_ancillas() const {
    return num_z_ancillas_;
  }

  uint32_t num_detectors() const {
    return num_detectors_;
  }

  int32_t data_qubit_schedule_type(uint32_t data_qubit) const;

  bool data_qubit_is_boundary(uint32_t data_qubit) const;

  // True only when a data qubit has two X-ancilla and two Z-ancilla partners.
  bool data_qubit_is_full_interior(uint32_t data_qubit) const;

  bool is_data_qubit(uint32_t qubit) const {
    return qubit < num_data_qubits_;
  }

  int32_t op_round(uint32_t op_index) const {
    return op_index < op_index_to_round_.size() ? op_index_to_round_[op_index] : -1;
  }

  const std::vector<uint32_t>& check_event_to_qubit() const {
    return check_event_to_qubit_;
  }

  const std::vector<uint32_t>& check_event_to_op_index() const {
    return check_event_to_op_index_;
  }

  std::pair<int32_t, int32_t> data_z_ancilla_slots(uint32_t data_qubit) const;

  int32_t round_detector_index(uint32_t round_index, uint32_t z_ancilla) const;

  int32_t interaction_op_for_data_z_ancilla(
      uint32_t data_qubit,
      uint32_t z_ancilla,
      uint32_t round_index) const;

 private:
  size_t interaction_index_(
      uint32_t data_qubit,
      uint32_t round_index,
      uint32_t z_local) const;

  CompiledErasureProgram base_program_;
  RotatedSurfaceCode code_;

  uint32_t distance_;
  uint32_t rounds_;
  uint32_t num_data_qubits_;
  uint32_t x_anc_offset_;
  uint32_t z_anc_offset_;
  uint32_t num_z_ancillas_;
  uint32_t num_detectors_;

  std::vector<int32_t> op_index_to_round_;
  std::vector<uint32_t> check_event_to_qubit_;
  std::vector<uint32_t> check_event_to_op_index_;

  // Flattened [data_qubit][round][local_z_ancilla] -> op_index or -1.
  std::vector<int32_t> data_round_z_interaction_op_;
  // Flattened [round][local_z_ancilla] -> detector index or -1.
  std::vector<int32_t> round_z_detector_index_;
  // Flattened [data_qubit] -> (slot0, slot1) absolute ancilla indices or -1.
  std::vector<std::pair<int32_t, int32_t>> data_to_z_slots_;
  // Flattened [data_qubit] -> 0 unknown, 1 X-first (XZZX), 2 Z-first (ZXXZ).
  std::vector<uint8_t> data_schedule_type_;
};

}  // namespace qerasure::circuit
