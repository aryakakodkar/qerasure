#include "qerasure/core/code/rotated_surface_code.h"

#include <stdexcept>

namespace qerasure {

RotatedSurfaceCode::RotatedSurfaceCode(std::size_t distance)
    : distance_(distance),
      num_qubits_(0),
      x_anc_offset_(0),
      z_anc_offset_(0),
      gates_per_step_(0),
      dense_stride_(2 * distance + 2) {
  if (distance < 3 || distance % 2 == 0) {
    throw std::invalid_argument("Distance must be an odd integer greater than or equal to 3");
  }
  build();
}

void RotatedSurfaceCode::build() {
  build_lattice();
  build_stabilizers();
  build_data_partner_slots();
}

// Flatten (x, y) into a single index so coordinate lookup is a simple array access.
std::size_t RotatedSurfaceCode::dense_offset(QubitIndex x, QubitIndex y) const noexcept {
  return x * dense_stride_ + y;
}

void RotatedSurfaceCode::set_coord(QubitIndex idx, QubitIndex x, QubitIndex y) {
  index_to_coord_[idx] = {x, y};
  coord_to_index_dense_[dense_offset(x, y)] = idx;
}

std::size_t RotatedSurfaceCode::try_get_coord(std::ptrdiff_t x, std::ptrdiff_t y) const noexcept {
  if (x < 0 || y < 0 || static_cast<std::size_t>(x) >= dense_stride_ ||
      static_cast<std::size_t>(y) >= dense_stride_) {
    return kNoPartner;
  }
  return coord_to_index_dense_[dense_offset(static_cast<QubitIndex>(x), static_cast<QubitIndex>(y))];
}

void RotatedSurfaceCode::build_lattice() {
  coord_to_index_dense_.assign(dense_stride_ * dense_stride_, kNoPartner);

  const std::size_t data_count = distance_ * distance_;
  const std::size_t ancilla_count = distance_ * distance_ - 1;
  const std::size_t total_qubits = data_count + ancilla_count;

  index_to_coord_.resize(total_qubits);

  // Data qubits live on odd lattice coordinates.
  QubitIndex index = 0;
  for (QubitIndex x = 1; x < 2 * distance_ + 1; x += 2) {
    for (QubitIndex y = 1; y < 2 * distance_ + 1; y += 2) {
      set_coord(index++, x, y);
    }
  }

  // X-ancilla qubits occupy the checkerboard sublattice used for X stabilizers.
  x_anc_offset_ = index;
  for (QubitIndex x = 2; x < 2 * distance_; x += 4) {
    for (QubitIndex y = 0; y < 2 * distance_ + 2; y += 2) {
      set_coord(index++, x + 2 - (y % 4), y);
    }
  }

  // Z-ancilla qubits occupy the complementary checkerboard sublattice.
  z_anc_offset_ = index;
  for (QubitIndex x = 0; x < 2 * distance_ + 2; x += 2) {
    for (QubitIndex y = 2; y < 2 * distance_; y += 4) {
      set_coord(index++, x, y + (x % 4));
    }
  }

  num_qubits_ = index;
}

void RotatedSurfaceCode::build_stabilizers() {
  // Per-step gate count: corners contribute 2, edges 3, interior sites 4.
  gates_per_step_ = 2 + 3 * (distance_ - 2) + (distance_ - 2) * (distance_ - 2);
  gates_.clear();
  gates_.reserve(gates_per_step_ * 4);

  // Four schedule steps x all qubits -> partner lookup table.
  partner_map_.assign(4 * num_qubits_, kNoPartner);

  // Direction vectors for X- and Z-ancilla interactions at each of the 4 steps.
  constexpr std::array<std::pair<int, int>, 4> kXDirections = {
      std::pair<int, int>{-1, 1},
      std::pair<int, int>{-1, -1},
      std::pair<int, int>{1, 1},
      std::pair<int, int>{1, -1},
  };
  constexpr std::array<std::pair<int, int>, 4> kZDirections = {
      std::pair<int, int>{-1, 1},
      std::pair<int, int>{1, 1},
      std::pair<int, int>{-1, -1},
      std::pair<int, int>{1, -1},
  };

  for (std::size_t step = 0; step < 4; ++step) {
    const std::size_t step_base = step * num_qubits_;

    // X checks: ancilla -> data CNOT orientation.
    for (QubitIndex idx = x_anc_offset_; idx < z_anc_offset_; ++idx) {
      const auto& coord = index_to_coord_[idx];
      const std::size_t partner = try_get_coord(
          static_cast<std::ptrdiff_t>(coord.first) + kXDirections[step].first,
          static_cast<std::ptrdiff_t>(coord.second) + kXDirections[step].second);
      if (partner != kNoPartner) {
        gates_.push_back({idx, partner});
        partner_map_[step_base + idx] = partner;
        partner_map_[step_base + partner] = idx;
      }
    }

    // Z checks: data -> ancilla CNOT orientation.
    for (QubitIndex idx = z_anc_offset_; idx < num_qubits_; ++idx) {
      const auto& coord = index_to_coord_[idx];
      const std::size_t partner = try_get_coord(
          static_cast<std::ptrdiff_t>(coord.first) + kZDirections[step].first,
          static_cast<std::ptrdiff_t>(coord.second) + kZDirections[step].second);
      if (partner != kNoPartner) {
        gates_.push_back({partner, idx});
        partner_map_[step_base + partner] = idx;
        partner_map_[step_base + idx] = partner;
      }
    }
  }
}

void RotatedSurfaceCode::build_data_partner_slots() {
  data_to_x_ancilla_slots_.assign(x_anc_offset_, {kNoPartner, kNoPartner});
  data_to_z_ancilla_slots_.assign(x_anc_offset_, {kNoPartner, kNoPartner});

  for (std::size_t data_idx = 0; data_idx < x_anc_offset_; ++data_idx) {
    for (std::size_t step = 0; step < 4; ++step) {
      const std::size_t partner = partner_map_[step * num_qubits_ + data_idx];
      if (partner == kNoPartner) {
        continue;
      }
      if (partner >= x_anc_offset_ && partner < z_anc_offset_) {
        auto& slots = data_to_x_ancilla_slots_[data_idx];
        if (slots.first == kNoPartner) {
          slots.first = partner;
        } else if (slots.second == kNoPartner && slots.first != partner) {
          slots.second = partner;
        }
      } else if (partner >= z_anc_offset_) {
        auto& slots = data_to_z_ancilla_slots_[data_idx];
        if (slots.first == kNoPartner) {
          slots.first = partner;
        } else if (slots.second == kNoPartner && slots.first != partner) {
          slots.second = partner;
        }
      }
    }
  }
}

}  // namespace qerasure
