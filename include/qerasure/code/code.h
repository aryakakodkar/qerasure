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

inline constexpr std::size_t kNoPartner = std::numeric_limits<std::size_t>::max();

class RotatedSurfaceCode {
 public:
  explicit RotatedSurfaceCode(std::size_t distance);

  std::size_t distance() const noexcept { return distance_; }
  std::size_t num_qubits() const noexcept { return num_qubits_; }
  std::size_t gates_per_step() const noexcept { return gates_per_step_; }

  const std::vector<Gate>& gates() const noexcept { return gates_; }
  const std::vector<std::pair<QubitIndex, QubitIndex>>& index_to_coord() const noexcept {
    return index_to_coord_;
  }
  const std::vector<std::size_t>& partner_map() const noexcept { return partner_map_; }

  std::size_t x_anc_offset() const noexcept { return x_anc_offset_; }
  std::size_t z_anc_offset() const noexcept { return z_anc_offset_; }

 private:
  std::size_t distance_;
  std::size_t num_qubits_;
  std::size_t x_anc_offset_;
  std::size_t z_anc_offset_;
  std::size_t gates_per_step_;
  std::size_t dense_stride_;

  std::vector<std::pair<QubitIndex, QubitIndex>> index_to_coord_;
  std::vector<std::size_t> coord_to_index_dense_;
  std::vector<Gate> gates_;
  std::vector<std::size_t> partner_map_;

  void build();
  void build_lattice();
  void build_stabilizers();

  std::size_t dense_offset(QubitIndex x, QubitIndex y) const noexcept;
  void set_coord(QubitIndex idx, QubitIndex x, QubitIndex y);
  std::size_t try_get_coord(std::ptrdiff_t x, std::ptrdiff_t y) const noexcept;
};

}  // namespace qerasure
