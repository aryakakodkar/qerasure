#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace qerasure {

using QubitIndex = std::size_t;
using Gate = std::pair<QubitIndex, QubitIndex>;

// Sentinel used when a qubit has no interaction partner at a given schedule step.
inline constexpr std::size_t kNoPartner = std::numeric_limits<std::size_t>::max();

// Encodes the geometry and syndrome-extraction schedule for a rotated surface code.
//
// The object precomputes:
// - qubit coordinates (data + ancilla),
// - four-step CNOT schedule used each QEC round,
// - a flattened partner map for fast simulator lookup.
class RotatedSurfaceCode {
 public:
  explicit RotatedSurfaceCode(std::size_t distance);

  std::size_t distance() const noexcept { return distance_; }
  std::size_t num_qubits() const noexcept { return num_qubits_; }
  std::size_t gates_per_step() const noexcept { return gates_per_step_; }

  // Flat list of all schedule gates ordered by step.
  const std::vector<Gate>& gates() const noexcept { return gates_; }

  // Per-index (x, y) lattice coordinates.
  const std::vector<std::pair<QubitIndex, QubitIndex>>& index_to_coord() const noexcept {
    return index_to_coord_;
  }

  // For each step and qubit index, stores partner index or kNoPartner.
  const std::vector<std::size_t>& partner_map() const noexcept { return partner_map_; }

  // Ancilla ranges in packed qubit indexing: [x_anc_offset, z_anc_offset) and [z_anc_offset, num_qubits).
  std::size_t x_anc_offset() const noexcept { return x_anc_offset_; }
  std::size_t z_anc_offset() const noexcept { return z_anc_offset_; }

  // Precomputed ancilla partner slots for each data qubit.
  // The two slots are deterministic per data qubit and used by lowering.
  const std::vector<std::pair<std::size_t, std::size_t>>& data_to_x_ancilla_slots() const noexcept {
    return data_to_x_ancilla_slots_;
  }
  const std::vector<std::pair<std::size_t, std::size_t>>& data_to_z_ancilla_slots() const noexcept {
    return data_to_z_ancilla_slots_;
  }

 private:
  // Input distance d (odd, >=3).
  std::size_t distance_;

  // Total qubits in packed index space: data + ancilla.
  std::size_t num_qubits_;

  // Packed index boundaries for ancilla subsets.
  std::size_t x_anc_offset_;
  std::size_t z_anc_offset_;

  // Number of CNOTs per schedule step (same for each of 4 steps).
  std::size_t gates_per_step_;

  // Side length of dense coordinate table used for O(1) coordinate lookup.
  std::size_t dense_stride_;

  // index -> (x, y)
  std::vector<std::pair<QubitIndex, QubitIndex>> index_to_coord_;

  // Dense lookup table: (x, y) -> index or kNoPartner.
  std::vector<std::size_t> coord_to_index_dense_;

  // All schedule gates concatenated step-by-step.
  std::vector<Gate> gates_;

  // Flattened [step * num_qubits + qubit] partner lookup.
  std::vector<std::size_t> partner_map_;
  std::vector<std::pair<std::size_t, std::size_t>> data_to_x_ancilla_slots_;
  std::vector<std::pair<std::size_t, std::size_t>> data_to_z_ancilla_slots_;

  // High-level build sequence.
  void build();

  // Populate lattice coordinates and ancilla offsets.
  void build_lattice();

  // Populate 4-step CNOT schedule and partner map.
  void build_stabilizers();
  void build_data_partner_slots();

  // Helper converting 2D coordinates to dense table offsets.
  std::size_t dense_offset(QubitIndex x, QubitIndex y) const noexcept;

  // Register one qubit coordinate in both lookup structures.
  void set_coord(QubitIndex idx, QubitIndex x, QubitIndex y);

  // Safe coordinate lookup; returns kNoPartner when out of bounds or absent.
  std::size_t try_get_coord(std::ptrdiff_t x, std::ptrdiff_t y) const noexcept;
};

}  // namespace qerasure
